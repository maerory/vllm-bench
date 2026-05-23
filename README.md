# vllm-bench

A benchmark harness comparing **vLLM** against a naive `transformers` baseline for LLM inference. Built to develop hands-on intuition for the numbers that matter in production serving — throughput, TTFT, tail latency — and how they move when you change knobs.

This is an interview-prep project, not a product. The artifact at the end is a benchmark suite, a set of numbers, and a written retrospective.

## Status

🚧 **In progress.** Session 1 complete: both runners load Qwen2.5-3B-Instruct on Modal and return coherent text for a single prompt. Benchmark harness, concurrency tests, and retrospective still to come.

## What this measures

- **Throughput** — output tokens/sec across concurrency levels
- **TTFT** — time to first token
- **Inter-token latency** — P50/P95/P99
- **Peak GPU memory**

…against a fixed prompt set, on a single GPU, for two runners:

1. **Sequential baseline** — `transformers` library, one prompt at a time via `model.generate`.
2. **vLLM** — continuous batching, PagedAttention, OpenAI-compatible API (or Python API; TBD).

The point is the *delta* between the two, across concurrency levels, with reproducible numbers.

## Setup

### Prerequisites

- Python 3.11+
- [Modal](https://modal.com) account and CLI installed locally (`pip install modal`, `modal token new`)
- HuggingFace access to `Qwen/Qwen2.5-3B-Instruct` (public model, no token required)

### Install

```bash
git clone <repo>
cd vllm-bench
pip install modal
```

All GPU dependencies (`transformers`, `torch`, `vllm`) are installed inside Modal containers, not locally — your local environment just needs `modal`.

## Usage

### Run the transformers baseline on a single prompt

```bash
modal run modal_app.py --runner transformers
```

### Run vLLM on the same prompt

```bash
modal run modal_app.py --runner vllm
```

The sanity prompt lives in `prompts/sanity.txt`. Both runners use:
- Model: `Qwen/Qwen2.5-3B-Instruct` (bfloat16)
- Greedy decoding (`temperature=0`, `do_sample=False`)
- `max_new_tokens` / `max_tokens` = 256
- Same system prompt and chat template

### First-run notes

The first invocation will download ~6GB of model weights to a persistent Modal Volume (`hf-cache`). Subsequent runs reuse the cache. Expect:
- First run: ~3–5 minutes (download + cold start + model load)
- Subsequent cold starts: ~30–60s
- Warm containers: a few seconds

## Project layout

```
vllm-bench/
├── README.md
├── LEARNING.md           # running log of questions & learnings
├── modal_app.py          # both runners (TransformerRunner, VLLMRunner)
└── prompts/
    └── sanity.txt        # single prompt for session 1
```

The brief calls for a richer structure (`runners/`, `harness.py`, `results/`, `analyze.py`, `retro.md`) — those will be added as the project grows. Flat for now.

## Design decisions

- **Modal for GPU**, not a persistent VM. Pay-per-execution sidesteps the "did I remember to shut the instance down" problem.
- **A10G** as the default GPU. 24GB is plenty for a 3B model; cheaper than A100.
- **Qwen2.5-3B-Instruct** as the target model. Small enough to iterate fast, large enough to produce realistic numbers, well-supported by vLLM.
- **Greedy decoding** (temperature 0) for reproducibility — same input should produce the same output across runs.
- **Persistent Modal Volume** for model weights, so we pay the ~6GB download once.
- **Separate Modal Images** for the two runners. vLLM has opinionated dependencies (specific torch, custom kernels) that don't merge cleanly with a plain transformers env.
- **`VLLM_USE_FLASHINFER_SAMPLER=0`** in the vLLM image to avoid JIT-compiling FlashInfer's sampling kernel, which requires the full CUDA toolkit (not present in `debian_slim`). Sampling perf is not the bottleneck for what we're measuring.

## Roadmap

- [x] **Session 1:** Both runners working end-to-end on a single prompt
- [ ] **Session 2:** Benchmark harness — fixed 50–100 prompt set, async concurrency runner, timing instrumentation
- [ ] **Session 3:** First numbers at concurrency 1, 4, 16; results table
- [ ] **Session 4:** Iterate, sanity-check, fix bugs until numbers are stable across runs
- [ ] **Session 5:** Written retrospective (`retro.md`)

Explicit non-goals: multi-GPU, quantization, comparison against other serving frameworks (SGLang, TGI, TensorRT-LLM), production hardening, web UI.

## License

MIT (or your preference — update before publishing)