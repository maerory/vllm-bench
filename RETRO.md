# vLLM Benchmarking — Project Retrospective

## Why I built this

I embarked on this project to understand the serving side of LLM deployment more deeply. Specifically, I wanted hands-on experience with vLLM — the canonical open-source serving library — to learn what makes LLM inference efficient in production conditions, and to build the kind of grounded, measured intuition that distinguishes "I've read about continuous batching" from "I've measured continuous batching on my own setup."

The deliverable is a benchmark harness comparing vLLM against a naive `transformers` baseline on a single GPU, a set of numbers across concurrency levels, and this retrospective. It is interview-prep work, not a product.

## Setup decisions

**GPU compute via Modal.** Without local GPU hardware, I used Modal as a cloud GPU provider. I had prior experience with Modal from fine-tuning a SmolLM model, so the RPC model was already familiar, and the pay-per-execution billing sidesteps the "did I remember to shut down the instance" problem of persistent VMs.

**Model: Qwen 2.5 3B Instruct in bfloat16.** A standard open-source instruct model, small enough to iterate quickly within my budget but large enough to produce representative numbers. Qwen 2.5 is well-supported by vLLM and is publicly accessible with no gating. bfloat16 is the native inference precision for modern GPUs and matches what production deployments actually use.

**GPU: A10G (24 GB).** Cost-efficient on Modal and more than sufficient for a 3B model. Larger GPUs (A100, H100) would have been overkill and obscured the kind of resource-constraint trade-offs that matter most for cost-conscious self-hosting.

**Output capped at 256 tokens, max model length 2048.** This bounds run times and makes batches comparable. With longer outputs, prompts that naturally hit EOS early would create artificially noisy throughput numbers.

## Architecture

Three files, kept deliberately simple:

- `modal_app.py` — Modal class definitions for `TransformerRunner` and `VLLMRunner`. Each exposes an async `generate_stream` method that yields per-token events. I used streaming for both engines because measuring TTFT and inter-token latency requires per-token timestamps, not a single end-of-generation return.

- `harness.py` — local script that loads the benchmark prompt set, fires requests at a runner (sequentially or concurrently via `asyncio.gather`), consumes the streams, records timings, and writes structured results to JSON. The harness itself doesn't generate text — it measures generation.

- `analyze.py` — aggregates a results JSON into per-tier, per-batch, and run-level summary tables. Pure stdlib, no pandas.

## Experiment design

I assembled 30 prompts spanning three length tiers — 10 short (~45 tokens), 10 medium (~120 tokens), 10 long (~270 tokens) — diverse across technical, creative, factual, and reasoning content within each tier. The mix lets me see how prompt length affects prefill cost and how decode behavior changes with growing KV cache.

I ran the benchmark at three concurrency levels: C=1 for the naive `transformers` baseline, and C=1, C=4, C=8 for vLLM. The original plan was to reach C=16 per the project brief's recommendation, but my Modal account tier capped concurrency at 10, so C=8 became the practical ceiling. This turned out not to materially affect the narrative — the interesting transitions for a 3B model on consumer-tier hardware happen at the low end of the concurrency curve anyway.

## Data

**System-level throughput across the concurrency sweep:**

| Runner | C | System tput (tok/s) | Speedup vs. transformers C=1 |
|---|---|---|---|
| transformers | 1 | 31.2 | 1.00x |
| vLLM | 1 | 56.9 | 1.82x |
| vLLM | 4 | 201.1 | 6.4x |
| vLLM | 8 | ~389 (mean of 3 runs, individual 366–404) | 12.5x |

**Per-request decode throughput stays constant across concurrency for vLLM:**

| Runner | C | Decode tput (tok/s, per request) |
|---|---|---|
| transformers | 1 | ~27 |
| vLLM | 1, 4, 8 | ~65 |

**TTFT (P50 / P95 in ms):**

| Runner | C | TTFT P50 | TTFT P95 |
|---|---|---|---|
| transformers | 1 | 429 | 432 |
| vLLM | 1 | 440 | 463 |
| vLLM | 4 | 449 | 456 |
| vLLM | 8 | 456 | 628 |

**GPU memory:**

| Runner | C | Device memory |
|---|---|---|
| transformers | 1 | 6.34 GB |
| vLLM | 8 | 20.01 GB |

## Findings

**TTFT was dominated by RPC overhead, not prefill cost.** Both engines showed ~430-460 ms TTFT at C=1 across all three prompt tiers, despite prompts varying in length by 6x (45 to 271 tokens). At the prompt sizes I tested, engine-side prefill is likely tens of milliseconds — buried under the network and serialization round-trip between my local machine and the Modal container. Client-observed TTFT and engine-side prefill latency are different metrics, and at small prompt sizes the former tells you mostly about the transport, not the model.

At C=8, TTFT degraded modestly (~35% on P95) because the vLLM scheduler had to interleave prefill of newly arriving requests with continued decode of the seven other in-flight ones. They share GPU time at each forward pass; the new request's prefill is slightly slower than it would be in isolation. But "slightly slower" was the whole story — no catastrophic tail behavior, no queueing collapse.

**vLLM was 2.1x faster than transformers per-request even without batching.** Single-stream decode throughput was 65 tok/s for vLLM versus 27 for transformers at C=1. This entire gap comes from kernel-level optimizations independent of batching: FlashAttention reduces memory traffic in the attention pass, fused operations cut kernel-launch overhead, the native sampler avoids JIT-compile paths, and the inner Python loop is tighter. None of these involve concurrent requests. They compound to roughly 2x decode speed even at concurrency 1. The continuous batching win at higher concurrency stacks on top of this single-stream advantage — it doesn't replace it.

**Continuous batching's win is in system throughput, not per-request speed.** This is the insight I want to be able to articulate clearly: at C=1, C=4, and C=8, each individual vLLM request decoded at ~65 tok/s. The per-request rate didn't change. What changed was that the engine ran 8 of those streams concurrently at C=8, while transformers could only run one. System throughput went from ~57 to ~389 tok/s — about 6.8x scaling at C=8, or 87% of perfectly linear. Cross-engine, vLLM at C=8 hit about 12.5x the system throughput of transformers at C=1.

The principle: continuous batching does not make individual requests faster. It shares the GPU's per-step compute across many in-flight requests, so the cost of reading the model weights from HBM is amortized across all of them. Per-user latency is essentially preserved; system efficiency scales nearly linearly with concurrency until you hit a memory-bandwidth or scheduler ceiling.

**vLLM commits ~3x more GPU memory than transformers, before serving a single request.** vLLM allocated about 20 GB on a 24 GB A10G at startup, regardless of concurrency level — its `gpu_memory_utilization=0.9` parameter tells it to claim 90% of available memory for the KV cache pool. Transformers used about 6 GB total. The ~13 GB gap is KV cache headroom that vLLM is sitting on, enough for ~185 concurrent 2K-token sequences according to its startup log. Most of it goes unused at C=8.

The trade-off is concrete: vLLM trades upfront memory commitment for the ability to scale concurrency without runtime allocation overhead. Transformers grows its KV cache lazily per request — a tiny footprint when idle, but no headroom for concurrent sequences. On consumer-tier 16 GB GPUs, vLLM's pre-allocation strategy could be operationally restrictive; on a shared-tenancy setup, it precludes co-locating other models.

## Implications for production serving

The numbers translate directly into self-hosting economics. At C=8 on A10G, vLLM produces about 400 tok/s of system throughput. Modal's A10G pricing is roughly $1.10/hour. Doing the arithmetic: $1.10 ÷ 3600 s ÷ 400 tok/s ≈ $0.00076 per 1000 tokens, or **about $0.76 per million tokens**.

Compare this to OpenAI's gpt-4o-mini at roughly $0.50 per million tokens blended. Self-hosting a small open-source model on Modal is in the same order of magnitude as API pricing for comparable-tier service. The crossover depends heavily on utilization: if your A10G is idle most of the day, the hourly cost makes API pricing far cheaper. If you can sustain near-C=8 utilization 24/7, self-hosting becomes competitive or favorable.

The latency-versus-throughput trade-off also looks gentler than the marketing case often suggests, at least at this concurrency range. Going from C=1 to C=8, system throughput rose 7x while P95 TTFT degraded only 35%. For interactive use cases with a latency SLO, C=4 to C=8 is viable on this hardware. Beyond C=10 was untested and might break this pattern.

The 20 GB upfront memory commitment is a real operational consideration. On dedicated A10G or larger, it's fine. On a shared GPU, it precludes co-tenancy. On 16 GB consumer hardware, it would force lower `gpu_memory_utilization`, which reduces the concurrency ceiling and partially undoes vLLM's batching advantage. The pre-allocation tax is a real cost of PagedAttention, not just a footnote.

## What I'd investigate with another week

**Engine-side TTFT instrumentation.** Client-side TTFT was dominated by RPC overhead at the prompt sizes I tested, hiding the actual prefill scaling curve. Adding timestamps inside the engine — when did prefill start, when did the first token come out — would let me see the prefill-vs-prompt-length relationship cleanly and report what the engine is actually doing, separately from what the network adds.

**Longer prompts to surface KV cache cost.** At 45–271 token prompts, decode throughput was flat across prompt length, meaning KV cache scan cost wasn't measurable. Pushing prompts to 2K–4K tokens would surface the prompt-length impact on decode speed, which matters for RAG, code assistants, and long-context applications. This is where the "prefill is compute-bound, decode is memory-bound" framing gets quantitative pressure.

**Variance characterization across repeated runs.** Three C=8 runs showed ~5% variance, with one run consistently slower than the other two — likely from shared-tenancy GPU heterogeneity or CUDA cache warmth differences. A rigorous benchmark would run each configuration 10+ times and report mean ± standard deviation. Single-run numbers are within tolerance for this project but not statistically defensible for serious comparison.

## Closing thought

The simple chat-completion workflow this project measured is only the beginning of how LLMs actually get used. Production deployments increasingly involve agentic patterns — RAG retrieval, tool use, multi-step reasoning, MCP-style tool calls — where a single user-facing request triggers many internal model calls. Continuous batching of those inter-step calls becomes more complex because many share prefix overlap: the same system prompt, the same conversation history, the same retrieved context. **Prefix caching**, which vLLM supports but I didn't measure, would likely matter more than continuous batching alone in those workflows. That's the natural next direction from here.

The headline takeaways I'd carry into an interview: vLLM at C=8 is roughly 12.5x faster system throughput than naive transformers, the win decomposes into kernel efficiency (~2x at C=1) and continuous batching (~7x on top of that at C=8), the trade-off is paid in upfront memory commitment and modest TTFT degradation, and the per-token cost at this scale is competitive with hosted API pricing if you can sustain high utilization. Those are the numbers I measured myself, and they're the ones I can defend.

---

## Appendix: additional learning points

**KV cache.** Stores the key and value tensors computed at every attention layer for every previously processed token. During decode, when generating token N+1, the model only needs to compute the query for token N+1 and attend to the cached keys/values from earlier tokens — it doesn't need to re-run the prompt through the model. Without KV cache, every new token would require a full forward pass over the entire context. The cache grows linearly with context length and dominates memory for long contexts, which is why KV cache management is the core problem PagedAttention is designed to solve.

**`gpu_memory_utilization`.** Controls how much GPU memory vLLM grabs at startup for the KV cache pool. The pool isn't faster per se — it lets vLLM allocate KV pages from a pre-sized memory region rather than calling cudaMalloc per request, and it sets the system's concurrency ceiling. Transformers, by contrast, lazily allocates KV cache per request: smaller idle footprint, no headroom for concurrent sequences.

**Prefill vs decode.** The first token is the most expensive because the entire prompt must be processed in a single forward pass to populate the KV cache. Subsequent tokens are cheaper because the model attends to cached keys/values and only computes one new query/key/value per step. Prefill is compute-bound (lots of parallel matmul, GPU saturates); decode is memory-bandwidth-bound (tiny per-step compute, but the model weights must be read from HBM every step). Optimizing the two is essentially a different problem, and serving frameworks make trade-offs based on which dominates the workload.

**Continuous batching.** vLLM's scheduler updates the active batch at every decode step. As soon as a request finishes, a new one can join the batch on the next forward pass — no waiting for the entire batch to complete. This maximizes GPU utilization across variable-length requests and keeps tail latency from collapsing under load. Contrast with static batching, where N requests are processed together and all must complete before the next batch starts; throughput is capped by the slowest request, and GPU utilization drops as faster requests sit idle waiting.

**TTFT, ITL, and throughput map to different stakeholders.** TTFT (time to first token) and ITL (inter-token latency) are user-facing metrics — they map to perceived responsiveness in interactive use cases like chatbots, voice assistants, and code completion. Throughput is operator-facing — it maps to cost per million tokens and matters for batch workloads, document parsing, agentic pipelines, and any application where total output throughput matters more than per-request responsiveness. The metrics often trade against each other; optimizing one can hurt another. Knowing which one matters for your use case shapes the entire deployment strategy.

**Client-side concurrency ≠ server-side concurrency.** My initial concurrency sweep produced suspiciously weak scaling because Modal's default `max_inputs=1` per container queued my "concurrent" requests at the container boundary — vLLM's batcher never saw more than one at a time. When measuring concurrent throughput, the diagnostic question is: where does parallelism actually start? Client-side concurrency is necessary but not sufficient; any serialization point below your target concurrency caps the rest.

**Cross-process introspection caveats.** vLLM 0.21's V1 engine runs the model in a subprocess. `torch.cuda.max_memory_allocated()` queried from the main Modal process returned 0 because that process never allocated GPU tensors — the subprocess did. `nvidia-smi` queries the driver directly and sees all processes regardless of Python-process boundaries; it was the right tool for cross-process truth. Generalizable: be careful with per-process introspection APIs when measuring multi-process systems.

**Async glue for async-native engines.** When the engine is async (vLLM's `AsyncLLMEngine`) and the transport is async (Modal RPC), the glue between them should also be async. Trying to bridge sync-to-async with a thread + queue + `asyncio.run()` per call breaks the engine's event-loop assumptions, because `AsyncLLMEngine` lazily attaches background tasks to whatever loop is running at first use — and a per-call loop tears down before those tasks can run. Making the Modal method itself `async def` is cleaner and avoids the entire bridging problem.