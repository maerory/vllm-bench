"""
Benchmark harness for vllm-bench.

Runs prompts through a runner (TransformerRunner or VLLMRunner), records
per-prompt and per-token timings, and writes structured results to disk.

Session 2 scope: concurrency 1 only. Sequential loop over prompts.
"""

import argparse
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
    """Per-prompt measurement record. Raw data; derived metrics computed later."""
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


@dataclass
class RunMetadata:
    """Top-level metadata for a benchmark run."""
    runner: str                 # "transformers" or "vllm"
    model_id: str
    concurrency: int            # 1 for session 2
    started_at: str             # ISO 8601
    finished_at: str
    n_prompts: int
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

    # Build full result blob and write to disk
    metadata = RunMetadata(
        runner=runner_name,
        model_id=MODEL_ID,
        concurrency=1,
        started_at=started_at,
        finished_at=finished_at,
        n_prompts=len(prompts),
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
    parser.add_argument(
        "--runner",
        choices=["transformers", "vllm"],
        required=True,
        help="Which runner to benchmark",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=REPO_ROOT / "prompts" / "benchmark_set.json",
        help="Path to prompt set JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: results/<date>-<runner>-c1.json)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup request (not recommended; cold-start will pollute first result)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N prompts (for quick iteration during development)",
    )
    args = parser.parse_args()

    # Load prompts
    with open(args.prompts) as f:
        prompts = json.load(f)
    if args.limit:
        prompts = prompts[: args.limit]

    # Default output path
    if args.output is None:
        date_str = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        args.output = REPO_ROOT / "results" / f"{date_str}-{args.runner}-c1.json"

    # Modal-aware: the harness needs to run inside `with app.run():` so that
    # Modal connects to its backend and provisions containers on demand.
    # Without this, .remote() calls will fail.
    with app.run():
        run_benchmark(
            runner_name=args.runner,
            prompts=prompts,
            output_path=args.output,
            warmup=not args.no_warmup,
        )


if __name__ == "__main__":
    main()