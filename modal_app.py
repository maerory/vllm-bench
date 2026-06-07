import modal

GPU = "A10G"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers==4.46.0",
        "torch==2.4.0",
        "accelerate",
    )
)

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("vllm==0.21.0")
)


app = modal.App("vllm-bench", image=image)

volume = modal.Volume.from_name("hf-cache", create_if_missing=True)
CACHE_DIR = "/cache"


@app.cls(
    gpu=GPU,
    volumes={CACHE_DIR: volume},
    timeout=600,
)
class TransformerRunner:
    @modal.enter()
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=CACHE_DIR,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
        )

        volume.commit()

    @modal.method()
    def reset_memory_stats(self):
        import torch
        torch.cuda.reset_peak_memory_stats()

    @modal.method()
    def get_gpu_memory_peak(self):
        import torch
        free, total = torch.cuda.mem_get_info()
        used = total - free
        return {
            "torch_peak_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "torch_current_gb": torch.cuda.memory_allocated() / (1024**3),
            "device_used_gb": used / (1024**3),
            "device_total_gb": total / (1024**3),
        }

    @modal.method()
    def generate(self, prompt: str) -> str:

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=False,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in \
                zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response
    
    @modal.method()
    def generate_stream(self, prompt: str):
        import time
        from transformers import TextIteratorStreamer
        from threading import Thread

        # 1. Apply chat template, tokenize
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        prompt_tokens = model_inputs.input_ids.shape[-1]

        # 2. Set up the streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # 3. Kick off generation in a background thread
        gen_kwargs = dict(
            **model_inputs,
            max_new_tokens=256,
            do_sample=False,
            streamer=streamer,
        )
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        # 4. Stream tokens out with timestamps
        # First yield: prompt_tokens (so the harness knows input length)
        yield {"type": "prompt_info", "prompt_tokens": prompt_tokens}

        for token_text in streamer:
            yield {
                "type": "token",
                "text": token_text,
                "server_ts": time.time(),
            }
        
        thread.join(timeout=1.0)
        yield {"type": "done"}


@app.cls(
    gpu=GPU,
    image=vllm_image,
    volumes={CACHE_DIR: volume},
    timeout=600,
)
@modal.concurrent(max_inputs=10)
class VLLMRunner:
    @modal.enter()
    def load_model(self):
        import os
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

        from vllm import LLM, SamplingParams, AsyncLLMEngine, AsyncEngineArgs
        from transformers import AutoTokenizer
        
        engine_args = AsyncEngineArgs(
            model=MODEL_ID,
            download_dir=CACHE_DIR,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            max_model_len=2048,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.sampling_params = SamplingParams(temperature=0, max_tokens=256)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    @modal.method()
    def reset_memory_stats(self):
        import torch
        torch.cuda.reset_peak_memory_stats()

    @modal.method()
    def get_gpu_memory_peak(self):
        import torch
        return {
            "peak_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        }

    @modal.method()
    async def generate_stream(self, prompt: str) -> str:
        """
        Synchronous generator method that internally drives an async engine.
        Yields the same envelope format as TransformerRunner.generate_stream:
            {"type": "prompt_info", "prompt_tokens": int}
            {"type": "token", "text": str}     (one or more, delta only)
            {"type": "done"}
        """
        import uuid
        from dataclasses import dataclass

        @dataclass
        class _Error:
            exc: BaseException
            traceback: str

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        prompt_tokens = len(prompt_token_ids)

        yield {"type": "prompt_info", "prompt_tokens": prompt_tokens}

        request_id = str(uuid.uuid4())
        prev = ""
        async for output in self.engine.generate(
            text,
            self.sampling_params,
            request_id=request_id,
        ):
            cumulative = output.outputs[0].text
            delta = cumulative[len(prev):]
            if delta:
                yield {"type": "token", "text": delta}
            prev = cumulative
        
        yield {"type": "done"}

    
        

@app.local_entrypoint()
def main(runner: str = "transformers"):
    prompt = open("prompts/sanity.txt").read().strip()
    
    if runner == "transformers":
        r = TransformerRunner()
    elif runner == "vllm":
        r = VLLMRunner()
    else:
        raise ValueError(f"Unknown Runner: {runner}")
    
    output = r.generate.remote(prompt)
    print("---PROMPT---")
    print(prompt)
    print("---OUTPUT---")
    print(output)