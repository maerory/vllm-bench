"""
Aggregate a benchmark results JSON into a per-tier summary table.

Usage:
    python analyze.py results/20260524-071149-transformers-c1.json
"""

import argparse
import json
import statistics
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

    # Throughputs
    throughput_e2e = output_tokens / total_time if total_time > 0 else 0.0
    throughput_decode = output_tokens / decode_time if decode_time > 0 else 0.0

    # Mean ITL — note this is yield-event ITL, not actual per-token ITL
    # (see LEARNING.md re: TextIteratorStreamer buffering)
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
    }


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_by_tier(metrics: list[dict]) -> dict[str, dict]:
    """Group metrics by tier and compute summary stats."""
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
    """Compute the p-th percentile via linear interpolation. Simple version."""
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
# Rendering
# =============================================================================

def render_markdown_table(summary: dict[str, dict], metadata: dict) -> str:
    """Render a per-tier summary as a Markdown table."""
    tier_order = ["short", "medium", "long"]
    tiers_present = [t for t in tier_order if t in summary]

    lines = []
    lines.append(f"# Benchmark summary: {metadata['runner']} (concurrency={metadata['concurrency']})")
    lines.append("")
    lines.append(f"- Model: `{metadata['model_id']}`")
    lines.append(f"- Prompts: {metadata['n_prompts']}")
    lines.append(f"- Started: {metadata['started_at']}")
    lines.append("")

    # Main table
    headers = [
        "Tier", "n",
        "Prompt tok",
        "Out tok",
        "TTFT mean (ms)",
        "TTFT p50 (ms)",
        "TTFT p95 (ms)",
        "Total (s)",
        "Tput e2e (tok/s)",
        "Tput decode (tok/s)",
        "ITL mean (ms)",
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
    """Render the per-prompt detail rows. Useful when investigating outliers."""
    headers = ["ID", "Tier", "P.tok", "O.tok", "TTFT(ms)", "Total(s)", "Tput(tok/s)", "ITL(ms)"]
    lines = []
    lines.append("\n## Per-prompt detail")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for m in metrics:
        row = [
            m["prompt_id"],
            m["tier"],
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
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Also print per-prompt detail table",
    )
    args = parser.parse_args()

    with open(args.results_file) as f:
        blob = json.load(f)

    metadata = blob["metadata"]
    raw_results = blob["results"]

    metrics = [derive_metrics(r) for r in raw_results]
    summary = aggregate_by_tier(metrics)

    print(render_markdown_table(summary, metadata))
    if args.detail:
        print(render_per_prompt_table(metrics))


if __name__ == "__main__":
    main()