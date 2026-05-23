# Learning log

Running notes from building `vllm-bench`. Questions I asked, things I learned, surprises along the way. Written for future-me and as raw material for the retrospective.

---

## Session 1 — Getting both runners working on Modal

**Goal:** transformers + vLLM each load Qwen2.5-3B-Instruct on Modal and return coherent text for a single sanity prompt. No benchmarking yet.

### What broke and how I fixed it

#### Bug 1: State not stored on `self`

First take of `TransformersRunner.load_model` assigned `model` and `tokenizer` as local variables. They were garbage-collected when the method returned, then `generate` tried to reference them and crashed.

**Fix:** `self.model = ...` and `self.tokenizer = ...`. This is the whole point of the `@modal.cls` pattern — `@modal.enter` loads state once per container, methods reuse it. Only works if state lives on the instance.

**Lesson:** Modal's `@modal.cls` is doing a lot of work behind the scenes. The container lifecycle is: spin up → run `@modal.enter` once → run method calls many times → eventually shut down. Anything you want reused across calls must be on `self`.

#### Bug 2: `cache_dir` not passed to `from_pretrained`

Mounted the Volume at `/cache` but didn't tell HuggingFace to use it. Weights would have re-downloaded into the container's ephemeral FS on every cold start.

**Fix:** `AutoModelForCausalLM.from_pretrained(..., cache_dir=CACHE_DIR)` and same for the tokenizer.

**Lesson:** Mounting a Volume is one half; pointing the library at it is the other. Two things that have to agree.

#### Bug 3: FlashInfer JIT compilation failed (vLLM)

vLLM tried to JIT-compile a CUDA kernel for its sampling backend (FlashInfer). The compile needs `nvcc`, which isn't in `debian_slim`. Crashed during the "profile run" step at engine startup.

**Fix:** `os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"` at the top of `load_model`, before `from vllm import LLM`. Falls back to vLLM's native sampler, which doesn't need JIT compilation.

**Alternatives I didn't use:** switch to a CUDA-devel base image (heavier cold starts), or use Modal's pre-built vLLM image. Disabling FlashInfer was one line, didn't affect what I'm measuring (sampling is not the bottleneck for inference throughput).

#### Bug 4: `LLM.generate` returns a list

Treated `self.llm.generate(...)` return value as if it were a single object. It's `list[RequestOutput]` — even with one prompt in, you get a one-element list out.

**Fix:** `outputs[0].outputs[0].text` — first request, first sampled completion, the string.

---

## Concepts I asked about

### What is FlashInfer?

An open-source library of optimized CUDA kernels for LLM inference. Provides:
- **Attention kernels** (fused, KV-cache-aware — the big one)
- **Sampling kernels** (top-k, top-p, temperature, all on GPU)
- **Utilities** (RoPE, normalization, etc.)

vLLM uses FlashInfer as a backend for these operations. Alternatives: vLLM's own native kernels, FlashAttention, xFormers.

Why JIT compile? The optimal kernel depends on runtime info (hidden dim, head count, batch patterns). Rather than ship one mediocre binary, FlashInfer generates a kernel specialized to your config and compiles it on first use. That requires `nvcc`.

**Mental model for vLLM's speed:** two parts.
1. **Scheduling/batching innovations** — continuous batching, PagedAttention's KV cache management.
2. **Optimized kernels** — FlashInfer/FlashAttention for actual compute.

A complete "what makes vLLM fast?" answer covers both.

### What is the CUDA compiler / what's CUDA exactly?

Layers, from top to bottom:

| Layer | What it is |
|---|---|
| CUDA | NVIDIA's parallel-computing platform — only works on NVIDIA GPUs. AMD has ROCm, Apple has Metal. |
| CUDA C++ | A language: C++ with extensions for GPU code. Functions that run on GPU = "kernels". |
| `nvcc` | NVIDIA CUDA Compiler. CUDA C++ → PTX → SASS. |
| PTX | Intermediate assembly-like language, portable across GPU generations. |
| SASS | Actual machine code, specific to a GPU architecture (Ampere, Hopper, etc.). |

**Subtle but important framing:** CUDA isn't "faster than another language" for the same computation on the same hardware. CUDA is the *only* way to run code on an NVIDIA GPU at all. The speedup comes from running on the GPU (thousands of parallel cores) instead of the CPU (tens of cores).

The call stack when I run inference:
```
Python code (model.generate)
  → PyTorch
  → Optimized kernel libraries (FlashInfer, cuBLAS, cuDNN)
  → CUDA C++
  → nvcc compiles to PTX / SASS
  → NVIDIA GPU
```

### How does vLLM know which tokenizer to use?

It auto-loads the tokenizer from the same HuggingFace repo as the model. `LLM(model="Qwen/Qwen2.5-3B-Instruct")` internally:

1. Looks up the model on HF Hub or local cache.
2. Reads `config.json` for architecture.
3. Calls `AutoTokenizer.from_pretrained(model_id)` to load `tokenizer_config.json` + `tokenizer.json`.
4. Loads weights, sets up engine.

Tokenizers ship with models. Token ID 12345 means whatever the tokenizer says it means — a model's weights are meaningless without its matching tokenizer.

**Why `LLM.chat()` "just works":** `tokenizer_config.json` for instruct models includes a `chat_template` (Jinja2). `LLM.chat(messages)` is roughly `LLM.generate(tokenizer.apply_chat_template(messages))`. I did this manually with `AutoTokenizer.from_pretrained(MODEL_ID).apply_chat_template(...)` — same result, just explicit.

---

## Decisions I made and why

| Decision | Why |
|---|---|
| Modal over a persistent cloud VM | Pay-per-execution; no "did I forget to shut it down" risk |
| A10G GPU | 24GB is plenty for 3B model; cheaper than A100 |
| Qwen2.5-3B-Instruct | Public, no gating, well-supported by vLLM, small enough to iterate fast |
| Greedy decoding (temp=0) | Reproducibility — same input → same output. Will revisit when harness exists. |
| `max_new_tokens=256` | Bounded runs, comparable across runners |
| Separate Modal Images for the two runners | vLLM's deps don't merge cleanly with plain transformers |
| Disable FlashInfer sampler | One line vs Docker-image-fight; sampling perf doesn't matter for what I'm measuring |
| Persistent HF cache Volume | Pay the ~6GB download once |
| Manual chat templating in vLLM (not `LLM.chat()`) | Mirrors the transformers runner; same input string to both engines |

---

## Open questions / parking lot

Things I deferred and want to revisit:

- **vLLM web endpoint vs Python API.** Decided to defer; using Python API for session 1 because it's the minimum thing that proves vLLM works. Web endpoint is closer to "real serving" and arguably more interview-relevant. Decision point when harness design begins.
- **Sampling-related kernel paths.** Disabled FlashInfer; should sanity-check that vLLM's native sampler produces equivalent outputs at temp=0 (it should — greedy is deterministic regardless of backend — but worth verifying when I add more prompts).
- **`gpu_memory_utilization=0.9`.** Default-ish, fine for now. This is *the* knob to talk about in the retro re: KV cache memory management. Want to vary it once the harness exists and observe what happens.
- **`max_model_len=2048`.** Deliberately small to keep KV cache headroom predictable. May need to bump if benchmark prompts get longer.
- **Will the two runners produce comparable outputs across more prompts?** Session 1 only verified on one sanity prompt. If the prompt set is more diverse, do the two runners actually agree at temp=0? They should, modulo floating-point determinism quirks.

---

## Things I want to be able to say in interviews after this

(From the brief, surfacing them here as a checklist to verify against my own numbers later.)

- [ ] Continuous batching vs static batching — what the difference does at high concurrency, with my measured numbers
- [ ] KV cache — what it is, why it dominates long-context memory, what PagedAttention does, what `gpu_memory_utilization` actually controls
- [ ] TTFT vs inter-token latency vs throughput — which matters when, with measured numbers
- [ ] Prefill vs decode — why first token is expensive, why subsequent tokens are cheap, with measured TTFT-vs-ITL data
- [ ] Throughput curve shape — where the knee is on my setup, with the actual curve
- [ ] Cost framing — tokens/sec → $/M tokens given Modal A10G pricing