"""
Aggregate a benchmark results JSON into per-tier, per-batch, and run-level summaries.

Usage:
    python analyze.py results/20260524-071149-transformers-c1.json
"""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path


# =============================================================================
# Per-prompt derived metrics
# =============================================================================

def derive_metrics(result: dict) -> dict:
    """Compute per-prompt metrics from a raw result record."""
    ttft = result["t_first_token"] - result["t_request_sent"]
    total_time = result["t_complete"] - result["t_request_sent"]
    decode_time = result["t_complete"] - result["t_first_token"]
    output_tokens = result["output_tokens"]

    throughput_e2e = output_tokens / total_time if total_time > 0 else 0.0
    throughput_decode = output_tokens / decode_time if decode_time > 0 else 0.0

    n_yields = len(result["token_arrival_times"])
    if n_yields >= 2:
        first_yield_t = result["token_arrival_times"][0]
        last_yield_t = result["token_arrival_times"][-1]
        itl_mean = (last_yield_t - first_yield_t) / (n_yields - 1)
    else:
        itl_mean = 0.0

    return {
        "tier": result["tier"],
        "prompt_id": result["prompt_id"],
        "prompt_tokens": result["prompt_tokens"],
        "output_tokens": output_tokens,
        "ttft_ms": ttft * 1000,
        "total_s": total_time,
        "decode_s": decode_time,
        "throughput_e2e": throughput_e2e,
        "throughput_decode": throughput_decode,
        "itl_ms": itl_mean * 1000,
        "n_yields": n_yields,
        # Batch context (defaulted to 0/0.0 for older c=1 results)
        "batch_id": result.get("batch_id", 0),
        "batch_t_start": result.get("batch_t_start", 0.0),
        "batch_t_end": result.get("batch_t_end", 0.0),
        # Raw request times for run-level aggregation
        "t_request_sent": result["t_request_sent"],
        "t_complete": result["t_complete"],
    }


# =============================================================================
# Per-tier aggregation (per-request view — "what each user experienced")
# =============================================================================

def aggregate_by_tier(metrics: list[dict]) -> dict[str, dict]:
    by_tier: dict[str, list[dict]] = {}
    for m in metrics:
        by_tier.setdefault(m["tier"], []).append(m)

    summary = {}
    for tier, items in by_tier.items():
        summary[tier] = {
            "n": len(items),
            "prompt_tokens_mean": statistics.mean(i["prompt_tokens"] for i in items),
            "output_tokens_mean": statistics.mean(i["output_tokens"] for i in items),
            "ttft_ms_mean": statistics.mean(i["ttft_ms"] for i in items),
            "ttft_ms_p50": statistics.median(i["ttft_ms"] for i in items),
            "ttft_ms_p95": percentile([i["ttft_ms"] for i in items], 95),
            "total_s_mean": statistics.mean(i["total_s"] for i in items),
            "throughput_e2e_mean": statistics.mean(i["throughput_e2e"] for i in items),
            "throughput_decode_mean": statistics.mean(i["throughput_decode"] for i in items),
            "itl_ms_mean": statistics.mean(i["itl_ms"] for i in items),
        }
    return summary


def percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


# =============================================================================
# Per-batch aggregation (system view — "what the engine produced")
# =============================================================================

def aggregate_by_batch(metrics: list[dict]) -> list[dict]:
    """
    Group metrics by batch_id, compute per-batch throughput.
    For c=1 runs (no batch context), each prompt is its own batch.
    """
    by_batch: dict[int, list[dict]] = {}
    for m in metrics:
        by_batch.setdefault(m["batch_id"], []).append(m)

    summaries = []
    for batch_id in sorted(by_batch.keys()):
        items = by_batch[batch_id]

        # Use recorded batch_t_start/end if present, else derive from request times
        if items[0]["batch_t_start"] > 0:
            batch_start = items[0]["batch_t_start"]
            batch_end = items[0]["batch_t_end"]
        else:
            # c=1 fallback — single request defines the "batch"
            batch_start = min(i["t_request_sent"] for i in items)
            batch_end = max(i["t_complete"] for i in items)

        batch_wall = batch_end - batch_start
        total_out_tokens = sum(i["output_tokens"] for i in items)
        system_throughput = total_out_tokens / batch_wall if batch_wall > 0 else 0.0

        summaries.append({
            "batch_id": batch_id,
            "n_requests": len(items),
            "batch_wall_s": batch_wall,
            "total_output_tokens": total_out_tokens,
            "system_throughput": system_throughput,
        })
    return summaries


# =============================================================================
# Run-level aggregation (the headline cross-concurrency number)
# =============================================================================

def aggregate_run(metrics: list[dict], metadata: dict) -> dict:
    """
    Compute headline numbers for the whole run.
    
    The system-level throughput across the entire run is what matters
    for cross-concurrency comparison.
    """
    total_output_tokens = sum(m["output_tokens"] for m in metrics)

    # Run wall time: from started_at to finished_at in metadata
    started = datetime.fromisoformat(metadata["started_at"])
    finished = datetime.fromisoformat(metadata["finished_at"])
    run_wall_s = (finished - started).total_seconds()

    # Alternative: from earliest request_sent to latest t_complete
    earliest_request = min(m["t_request_sent"] for m in metrics)
    latest_complete = max(m["t_complete"] for m in metrics)
    request_window_s = latest_complete - earliest_request

    system_throughput_run = total_output_tokens / run_wall_s if run_wall_s > 0 else 0.0
    system_throughput_window = total_output_tokens / request_window_s if request_window_s > 0 else 0.0

    return {
        "total_output_tokens": total_output_tokens,
        "run_wall_s": run_wall_s,
        "request_window_s": request_window_s,
        "system_throughput_run": system_throughput_run,
        "system_throughput_window": system_throughput_window,
        # Per-request latency aggregates (across all prompts, all tiers)
        "ttft_ms_p50_overall": percentile([m["ttft_ms"] for m in metrics], 50),
        "ttft_ms_p95_overall": percentile([m["ttft_ms"] for m in metrics], 95),
        "ttft_ms_p99_overall": percentile([m["ttft_ms"] for m in metrics], 99),
        "total_s_p50_overall": percentile([m["total_s"] for m in metrics], 50),
        "total_s_p95_overall": percentile([m["total_s"] for m in metrics], 95),
    }


# =============================================================================
# Rendering
# =============================================================================

def render_run_summary(run: dict, metadata: dict) -> str:
    """Render the run-level headline numbers."""
    lines = []
    lines.append(f"# Run summary: {metadata['runner']} (concurrency={metadata['concurrency']})")
    lines.append("")
    lines.append(f"- Model: `{metadata['model_id']}`")
    lines.append(f"- Prompts: {metadata['n_prompts']}")
    lines.append(f"- Batches: {metadata.get('n_batches', 'n/a')}")
    lines.append(f"- Started: {metadata['started_at']}")
    lines.append("")
    lines.append("## System-level (cross-concurrency headline numbers)")
    lines.append("")
    lines.append(f"- **Total output tokens**: {run['total_output_tokens']:,}")
    lines.append(f"- **Run wall time** (started_at → finished_at): {run['run_wall_s']:.2f}s")
    lines.append(f"- **Request window** (earliest sent → latest complete): {run['request_window_s']:.2f}s")
    lines.append(f"- **System throughput (run-level)**: {run['system_throughput_run']:.1f} tok/s")
    lines.append(f"- **System throughput (request window)**: {run['system_throughput_window']:.1f} tok/s")
    lines.append("")
    lines.append("## Per-request latency (across all prompts)")
    lines.append("")
    lines.append(f"- TTFT p50: {run['ttft_ms_p50_overall']:.0f}ms")
    lines.append(f"- TTFT p95: {run['ttft_ms_p95_overall']:.0f}ms")
    lines.append(f"- TTFT p99: {run['ttft_ms_p99_overall']:.0f}ms")
    lines.append(f"- Total time p50: {run['total_s_p50_overall']:.2f}s")
    lines.append(f"- Total time p95: {run['total_s_p95_overall']:.2f}s")
    return "\n".join(lines)


def render_batch_table(batches: list[dict]) -> str:
    """Render per-batch system throughput. Only meaningful at c>1."""
    if len(batches) <= 1:
        return ""  # for c=1 with many prompts, this would print 30 single-row batches; skip
    lines = []
    lines.append("\n## Per-batch throughput")
    lines.append("")
    lines.append("| Batch | Requests | Wall (s) | Out tokens | System tput (tok/s) |")
    lines.append("|---|---|---|---|---|")
    for b in batches:
        row = [
            str(b["batch_id"]),
            str(b["n_requests"]),
            f"{b['batch_wall_s']:.2f}",
            str(b["total_output_tokens"]),
            f"{b['system_throughput']:.1f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    # Aggregate across batches
    mean_tput = statistics.mean(b["system_throughput"] for b in batches)
    lines.append("")
    lines.append(f"**Mean batch system throughput: {mean_tput:.1f} tok/s**")
    return "\n".join(lines)


def render_tier_table(summary: dict[str, dict]) -> str:
    """Render the per-tier per-request table (existing, unchanged)."""
    tier_order = ["short", "medium", "long"]
    tiers_present = [t for t in tier_order if t in summary]

    lines = []
    lines.append("\n## Per-tier (per-request view — what each user experienced)")
    lines.append("")

    headers = [
        "Tier", "n",
        "Prompt tok", "Out tok",
        "TTFT mean (ms)", "TTFT p50 (ms)", "TTFT p95 (ms)",
        "Total (s)", "Tput per-req (tok/s)", "Tput decode (tok/s)", "ITL mean (ms)",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for tier in tiers_present:
        s = summary[tier]
        row = [
            tier,
            str(s["n"]),
            f"{s['prompt_tokens_mean']:.0f}",
            f"{s['output_tokens_mean']:.0f}",
            f"{s['ttft_ms_mean']:.0f}",
            f"{s['ttft_ms_p50']:.0f}",
            f"{s['ttft_ms_p95']:.0f}",
            f"{s['total_s_mean']:.2f}",
            f"{s['throughput_e2e_mean']:.1f}",
            f"{s['throughput_decode_mean']:.1f}",
            f"{s['itl_ms_mean']:.1f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def render_per_prompt_table(metrics: list[dict]) -> str:
    headers = ["ID", "Tier", "Batch", "P.tok", "O.tok", "TTFT(ms)", "Total(s)", "Tput(tok/s)", "ITL(ms)"]
    lines = []
    lines.append("\n## Per-prompt detail")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for m in metrics:
        row = [
            m["prompt_id"],
            m["tier"],
            str(m["batch_id"]),
            str(m["prompt_tokens"]),
            str(m["output_tokens"]),
            f"{m['ttft_ms']:.0f}",
            f"{m['total_s']:.2f}",
            f"{m['throughput_e2e']:.1f}",
            f"{m['itl_ms']:.1f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze vllm-bench results")
    parser.add_argument("results_file", type=Path, help="Path to results JSON")
    parser.add_argument("--detail", action="store_true", help="Also print per-prompt detail table")
    args = parser.parse_args()

    with open(args.results_file) as f:
        blob = json.load(f)

    metadata = blob["metadata"]
    raw_results = blob["results"]

    metrics = [derive_metrics(r) for r in raw_results]
    tier_summary = aggregate_by_tier(metrics)
    batch_summary = aggregate_by_batch(metrics)
    run_summary = aggregate_run(metrics, metadata)

    print(render_run_summary(run_summary, metadata))
    print(render_batch_table(batch_summary))
    print(render_tier_table(tier_summary))
    if args.detail:
        print(render_per_prompt_table(metrics))


if __name__ == "__main__":
    main()

