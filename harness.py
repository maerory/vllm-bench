"""
Benchmark harness for vllm-bench.

Runs prompts through a runner (TransformerRunner or VLLMRunner), records
per-prompt and per-token timings, and writes structured results to disk.

Session 2 scope: concurrency 1 only. Sequential loop over prompts.
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from pathlib import Path

from transformers import AutoTokenizer

# Import the Modal app and runner classes.
# Note: importing modal_app triggers Modal's local-side machinery but does
# NOT start a container. Containers spin up on the first .remote() call.
from modal_app import (
    app, TransformerRunner, VLLMRunner, 
    MODEL_ID
)


REPO_ROOT = Path(__file__).parent.resolve()

# =============================================================================
# Result data structures
# =============================================================================

@dataclass
class GenerationResult:
    """Per-prompt measurement record."""
    prompt_id: str
    tier: str
    prompt: str

    # Token counts
    prompt_tokens: int          # tokenized length AFTER chat templating
    output_tokens: int          # tokenized length of generated text

    # Output content
    output_text: str

    # Wall-clock timestamps (client-side, seconds since epoch)
    t_request_sent: float       # right before .remote() call
    t_first_token: float        # when first token yield arrived
    t_complete: float           # when the stream signaled done

    # Per-token arrival times, relative to t_request_sent (seconds)
    # Length should equal number of yield events that delivered tokens.
    # Note: these are *yield* timestamps, not 1:1 with tokens
    # (TextIteratorStreamer yields decoded fragments).
    token_arrival_times: list[float] = field(default_factory=list)

    # Batch context - for concurrency > 1, identified which batch this was in
    batch_id: int = 0
    batch_t_start: float = 0.0
    batch_t_end: float = 0.0


@dataclass
class RunMetadata:
    """Top-level metadata for a benchmark run."""
    runner: str                 # "transformers" or "vllm"
    model_id: str
    concurrency: int            # 1 for session 2
    started_at: str             # ISO 8601
    finished_at: str
    n_prompts: int
    n_batches: int = 0
    gpu_memory_peak: dict | None = None
    notes: str = ""


# =============================================================================
# Stream consumer
# =============================================================================

def consume_stream(stream, t_request_sent: float):
    """
    Consume a generate_stream() generator from a runner, recording timings.

    Expected envelope format from the runner:
        {"type": "prompt_info", "prompt_tokens": int}
        {"type": "token", "text": str}     (one or more)
        {"type": "done"}

    Returns: (prompt_tokens, output_text, t_first_token, t_complete, token_arrival_times)
    """
    prompt_tokens = None
    output_chunks = []
    token_arrival_times = []
    t_first_token = None
    t_complete = None

    for event in stream:
        if event["type"] == "prompt_info":
            prompt_tokens = event["prompt_tokens"]
        elif event["type"] == "token":
            if t_first_token is None:
                t_first_token = time.time()
            output_chunks.append(event["text"])
            token_arrival_times.append(time.time() - t_request_sent)
        elif event["type"] == "done":
            t_complete = time.time()
        elif event["type"] == "error":
            tb = event.get("traceback", "(no traceback captured)")
            raise RuntimeError(f"Runner error: {event['message']}\n\nRemote traceback:\n{tb}")
        else:
            raise ValueError(f"Unknown event: {event["type"]}")

    if prompt_tokens is None:
        raise ValueError(f"Prompt info event never arrived.")
    if t_first_token is None:
        raise ValueError(f"No token received")
    if t_complete is None:
        raise ValueError(f"No complete event: Stream was abruptly ended")
    

    output_text = "".join(output_chunks)
    return prompt_tokens, output_text, t_first_token, t_complete, token_arrival_times


# =============================================================================
# Async stream consumer
# =============================================================================

async def consume_stream_async(stream, t_request_sent: float):
    """
    Consume an async generator from a runner, recording timings.

    Same envelope format as sync version:
        {"type": "prompt_info", "prompt_tokens": int}
        {"type": "token", "text": str}
        {"type": "done"}
        {"type": "error", "message": str, "traceback": str}

    Returns: (prompt_tokens, output_text, t_first_token, t_complete, token_arrival_times)
    """
    prompt_tokens = None
    output_chunks = []
    token_arrival_times = []
    t_first_token = None
    t_complete = None

    async for event in stream:
        if event["type"] == "prompt_info":
            prompt_tokens = event["prompt_tokens"]
        elif event["type"] == "token":
            if t_first_token is None:
                t_first_token = time.time()
            output_chunks.append(event["text"])
            token_arrival_times.append(time.time() - t_request_sent)
        elif event["type"] == "done":
            t_complete = time.time()
        elif event["type"] == "error":
            tb = event.get("traceback", "(no traceback_captured)")
            raise RuntimeError(f"Runtime error: {event['message']}\n\nRemote traceback:\n{tb}")
        else:
            raise ValueError(f"Unknown event: {event["type"]}")
        
    if prompt_tokens is None:
        raise ValueError(f"Prompt info event never arrived.")
    if t_first_token is None:
        raise ValueError(f"No token received")
    if t_complete is None:
        raise ValueError(f"No complete event: Stream was abruptly ended")
    

    output_text = "".join(output_chunks)
    return prompt_tokens, output_text, t_first_token, t_complete, token_arrival_times

# =============================================================================
# Token counting (output side)
# =============================================================================

def count_output_tokens(text: str, tokenizer) -> int:
    """
    Count tokens in generated text using the same tokenizer the model uses.
    
    This is the canonical output-token count for throughput math.
    We re-tokenize the output rather than counting stream yields because
    TextIteratorStreamer yields text fragments, not 1:1 tokens.
    """
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])

# =============================================================================
# Per-prompt async worker
# =============================================================================

async def process_prompt(
    runner,
    prompt: dict,
    tokenizer,
    batch_id: int,
    batch_t_start: float,
) -> GenerationResult:
    """
    Fire one prompt at the runner, consume its stream, record the result.
    This coroutine is what gets fanned out via asyncio.gather.
    """
    t_request_sent = time.time()
    stream = runner.generate_stream.remote_gen.aio(prompt["prompt"])
    prompt_tokens, output_text, t_first_token, t_complete, token_arrival_times = await consume_stream_async(stream, t_request_sent)
    output_token_count = count_output_tokens(output_text, tokenizer)

    return GenerationResult(
        prompt_id=prompt["id"],
        tier=prompt["tier"],
        prompt=prompt["prompt"],
        prompt_tokens=prompt_tokens,
        output_tokens=output_token_count,
        output_text=output_text,
        t_request_sent=t_request_sent,
        t_first_token=t_first_token,
        t_complete=t_complete,
        token_arrival_times=token_arrival_times,
        batch_id=batch_id,
        batch_t_start=batch_t_start,
        batch_t_end=0.0,
    )

# =============================================================================
# Concurrent batch runner
# =============================================================================

def chunks(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


async def run_batch(
    runner,
    batch: list[dict],
    tokenizer,
    batch_id: int,
) -> list[GenerationResult]:
    """
    Fire all prompts in a batch concurrently, wait for all to complete,
    return the per-prompt results with batch-level timestamps filled in.
    """
    batch_t_start = time.time()

    coros = [
        process_prompt(
            runner, 
            prompt_record, 
            tokenizer, 
            batch_id, 
            batch_t_start
        ) 
        for prompt_record in batch
    ]

    results: list[GenerationResult] = await asyncio.gather(*coros, return_exceptions=False)

    batch_t_end = time.time()

    # Stamp the batch_t_end on every result
    for r in results:
        r.batch_t_end = batch_t_end

    return results


# =============================================================================
# Main benchmark loop
# =============================================================================

def run_benchmark(
    runner_name: str,
    prompts: list[dict],
    output_path: Path,
    warmup: bool = True,
) -> None:
    """
    Sequentially run all prompts through the chosen runner, recording timings.
    
    Args:
        runner_name: "transformers" or "vllm"
        prompts: list of {id, tier, prompt, target_tokens} dicts
        output_path: where to write results JSON
        warmup: if True, do one throwaway request first to exclude cold-start
    """
    runner = None

    if runner_name == "transformers":
        runner = TransformerRunner()
    elif runner_name == "vllm":
        runner = VLLMRunner()
    else:
        raise ValueError(f"Unknown runner - {runner}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Warmup: one throwaway request to exclude cold-start container spin-up
    # from the actual measurements. The first .remote() call provisions the
    # container, loads the model, and runs generate — typically 30-60s.
    # We don't want any of that in our timing data.
    if warmup:
        print("Warming up runner (this triggers container start + model load)...")
        list(runner.generate_stream.remote_gen("Ping."))
        runner.reset_memory_stats.remote()

    # Main benchmark loop
    results: list[GenerationResult] = []
    started_at = datetime.now(UTC).isoformat()

    for i, p in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {p['id']} ({p['tier']})...", end=" ", flush=True)

        t_request_sent = time.time()
        generator = runner.generate_stream.remote_gen(p["prompt"])

        prompt_tokens, output_text, t_first_token, t_complete, token_arrival_times = consume_stream(generator, t_request_sent)
        output_tokens = count_output_tokens(output_text, tokenizer)

        generation_result = GenerationResult(
            prompt_id=p["id"],
            tier=p["tier"],
            prompt=p["prompt"],
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            output_text=output_text,
            t_request_sent=t_request_sent,
            t_first_token=t_first_token,
            t_complete=t_complete,
            token_arrival_times=token_arrival_times,
        )

        results.append(generation_result)

        total_time = t_complete - t_request_sent
        ttft = t_first_token - t_request_sent
        itl_mean = (t_complete - t_first_token) / max(output_tokens - 1, 1)

        print(
            f"Prompt {p['id']}-{p['tier']}: ttft={ttft*1000:.0f}ms itl={itl_mean*1000:.1f}ms  total={total_time:.2f}s out_tokens={output_tokens}"
        )

    finished_at = datetime.now(UTC).isoformat()
    
    # Capture GPU memory peak from the runner
    try:
        gpu_memory_peak = runner.get_gpu_memory_peak.remote()
    except Exception as exc:
        print(f"Warning: failed to capture GPU memory: {exc}")
        gpu_memory_peak = None

    # Build full result blob and write to disk
    metadata = RunMetadata(
        runner=runner_name,
        model_id=MODEL_ID,
        concurrency=1,
        started_at=started_at,
        finished_at=finished_at,
        n_prompts=len(prompts),
        gpu_memory_peak=gpu_memory_peak,
    )

    blob = {
        "metadata": asdict(metadata),
        "results": [asdict(r) for r in results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(blob, f, indent=2)

    print(f"\nWrote {len(results)} results to {output_path}")


async def run_benchmark_async(
    runner_name: str,
    prompts: list[dict],
    concurrency: int,
    output_path: Path,
    warmup: bool = True,
) -> None:
    
    if runner_name == "transformers":
        if concurrency != 1:
            raise ValueError(
                "transformers runner only supports concurrency=1 in this benchmark. "
                "See LEARNING.md for why."
            )
        runner = TransformerRunner()
    elif runner_name == "vllm":
        runner = VLLMRunner()
    else:
        raise ValueError(f"Unknown runner: {runner_name}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if warmup:
        print("Warming up runner...")
        async for _ in runner.generate_stream.remote_gen.aio("Ping."):
            pass
        # Reset memory tracking so the peak only reflects the timed runs
        await runner.reset_memory_stats.remote.aio()

    # Iterate over batches
    results: list[GenerationResult] = []
    started_at = datetime.now(UTC).isoformat()

    batches = list(chunks(prompts, concurrency))
    print(f"Will run {len(prompts)} prompts in {len(batches)} batches of concurrency {concurrency}")

    for batch_id, batch in enumerate(batches):
        print(f"\n[Batch {batch_id+1}/{len(batches)}] firing {len(batch)} prompts concurrently...")
        batch_results = await run_batch(runner, batch, tokenizer, batch_id)

        # Per-prompt summary lines
        batch_wall = batch_results[0].batch_t_end - batch_results[0].batch_t_start
        total_out_tokens = sum(r.output_tokens for r in batch_results)
        batch_throughput = total_out_tokens / batch_wall if batch_wall > 0 else 0.0

        for r in batch_results:
            ttft = r.t_first_token - r.t_request_sent
            total_time = r.t_complete - r.t_request_sent
            print(
                f"  {r.prompt_id} ({r.tier}): "
                f"ttft={ttft*1000:.0f}ms total={total_time:.2f}s out_tokens={r.output_tokens}"
            )
        print(
            f"  → batch wall time: {batch_wall:.2f}s, "
            f"system throughput: {batch_throughput:.1f} tok/s"
        )

        results.extend(batch_results)

    finished_at = datetime.now(UTC).isoformat()
    # Capture GPU memory peak from the runner
    try:
        gpu_memory_peak = await runner.get_gpu_memory_peak.remote.aio()
    except Exception as exc:
        print(f"Warning: failed to capture GPU memory: {exc}")
        gpu_memory_peak = None

    metadata = RunMetadata(
        runner=runner_name,
        model_id=MODEL_ID,
        concurrency=concurrency,
        started_at=started_at,
        finished_at=finished_at,
        n_prompts=len(prompts),
        n_batches=len(batches),
        gpu_memory_peak=gpu_memory_peak,
    )

    blob = {
        "metadata": asdict(metadata),
        "results": [asdict(r) for r in results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(blob, f, indent=2)

    print(f"\nWrote {len(results)} results to {output_path}")




# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="vllm-bench harness")
    parser.add_argument("--runner", choices=["transformers", "vllm"], required=True)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent in-flight requests (vLLM only for C>1)",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=REPO_ROOT / "prompts" / "benchmark_set.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.prompts) as f:
        prompts = json.load(f)
    if args.limit:
        prompts = prompts[: args.limit]

    if args.output is None:
        date_str = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        args.output = REPO_ROOT / "results" / f"{date_str}-{args.runner}-c{args.concurrency}.json"

    with app.run():
        asyncio.run(
            run_benchmark_async(
                runner_name=args.runner,
                prompts=prompts,
                concurrency=args.concurrency,
                output_path=args.output,
                warmup=not args.no_warmup,
            )
        )

if __name__ == "__main__":
    main()