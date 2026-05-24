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

---

## Session 2 — Designing the benchmark harness

**Goal:** Build the measurement code (the "harness"), prove it produces sensible numbers at concurrency 1 only. Concurrency >1 is session 3.

### Concepts I asked about

#### What is a "harness"?

Plain software-engineering term: a **benchmark harness** (or test harness) is code that exercises another piece of code in a controlled, instrumented way. Named after the literal harness that straps onto a horse — it "straps onto" the thing under test and collects measurements.

In our project: `harness.py` is a script that loads prompts, calls the runner, records timestamps, and saves results. It does *not* implement generation — it *measures* generation.

A bug in the model code makes the model produce bad text (visible). A bug in the harness makes the numbers wrong in ways that are invisible. Worth being paranoid about.

(Not to be confused with "agentic harness" / "LLM harness" — same word, different domain. There, "harness" wraps an LLM with prompt-injection logic and tool-use loops. Same conceptual shape: wrap component + instrument. Different layer of the stack.)

#### What is RPC, and why does it matter for TTFT?

RPC = **Remote Procedure Call**. Call a function as if local; it executes on a different machine.

When I write `runner.generate.remote(prompt)`:
1. Local script serializes args to bytes
2. Bytes sent over network to Modal
3. Container deserializes, runs `generate(prompt)`, serializes return
4. Bytes sent back
5. Local script deserializes

Each step takes time. The full round-trip = "RPC overhead", maybe 10–50ms outside the model's actual work.

**Implication for TTFT semantics:**

| Definition | What it includes | What it measures |
|---|---|---|
| Client-side TTFT | RPC + network + queueing + prefill | User experience |
| Engine-side TTFT | prefill only | Pure model perf |

I chose **client-side TTFT** because (a) it reflects what real serving costs, (b) same RPC overhead applies to both runners so it cancels in comparisons, (c) engine-side requires hooking into vLLM/transformers internals — annoying and engine-specific.

Will document in the retro: "TTFT as measured includes Modal RPC overhead."

#### Tokenizer vs embedding layer

These are **separate** components with a clean boundary. Common confusion: the tokenizer does *not* do embeddings.

Pipeline:
```
"Explain attention"
       ↓
   TOKENIZER  (vocab lookup + BPE merges)
       ↓
   [9908, 6529]                    ← integer IDs
       ↓
   EMBEDDING LAYER  (lookup in trained matrix)
       ↓
   [[-0.31, 0.82, ..., 0.05], ...]  ← shape: [seq_len, hidden_dim]
       ↓
   TRANSFORMER LAYERS
       ↓
   ...
```

- **Tokenizer:** deterministic string processor. Vocab (~152K entries for Qwen2.5) + BPE merge rules. Output is integers. No learned LLM parameters; vocab is learned separately and once during tokenizer training. Microseconds to run.
- **Embedding layer:** learned matrix `[vocab_size, hidden_dim]`. Part of the model. For Qwen2.5-3B, ~150K × 2048 ≈ 300M params — meaningful chunk of total.

Subtle bonus: at the *end* of the model, hidden states are projected back into vocab space to produce logits. In modern LLMs the projection ("unembedding") is usually **tied** to the embedding matrix — same params, transposed. So "embedding" plays a role at both ends. But that's still the model's job, not the tokenizer's.

#### `add_generation_prompt=True` vs `False`

Default for inference is `True` — appends `<|im_start|>assistant\n` so the model knows to start its turn.

`False` is for:
1. **Training / SFT data prep** — example includes the assistant's turn as text; you're not generating it
2. **Perplexity / logprob scoring on complete conversations**
3. **Manually appending tokens** — e.g., forcing assistant to start with a specific prefix
4. **Multi-turn eval scripts where you control turn structure**

For inference: always `True`. Good interview probe — knowing this distinction signals more than copy-paste familiarity with templating.

#### Threading and GPU concurrency — the important one

Why we use a thread in the streaming code: **producer-consumer control flow, not performance.**

`model.generate()` blocks. The `TextIteratorStreamer` puts tokens into an internal queue *as they're produced*, but if `generate` is on the main thread, no one drains the queue until `generate` returns — defeating streaming. Putting `generate` on a background thread lets the main thread consume tokens as they're produced.

**The much bigger lesson — why naive multi-threading doesn't give you concurrency on a single GPU:**

LLM generation is **GPU-compute-bound**, not IO-bound. Workload types:

| Type | Bottleneck | Threading helps? |
|---|---|---|
| IO-bound | network, disk waits | Yes (GIL released during waits) |
| CPU-bound | Python interpreter math | No (GIL serializes) |
| GPU-bound | GPU math | GIL is released, but... |

For GPU-bound: GIL isn't the constraint, but **two threads each calling `model.generate` against the same model on one GPU don't run in parallel** — they contend for the GPU's compute resources, and CUDA serializes the work into a single execution stream. You'd get the same throughput as sequential, possibly worse (context-switching overhead).

**This is exactly the problem vLLM exists to solve.** vLLM doesn't multi-thread; it runs *one* engine that **batches multiple in-flight requests at the model layer**. At each decode step, the GPU does the math for all active requests in a single batched forward pass. Parallelism happens *inside* the model forward, not via OS threading.

The interview-ready phrasing:

> "Naively threading multiple `generate` calls against a single GPU doesn't give you concurrency — they serialize at the GPU. vLLM achieves concurrency by batching at the model layer: at each forward pass, the GPU processes all in-flight requests as a single batched operation. That's what makes continuous batching architecturally different from just multi-threading a transformers loop."

**Implication for session 3 baseline at concurrency >1:** there's no meaningful "transformers at concurrency 4" — option 1 (sequential, treat concurrency as N/A for the baseline) is the honest comparison. The asymmetry tells the right story: vLLM scales with concurrency, naive baseline doesn't.

### Design decisions this session

| Decision | Why |
|---|---|
| Client-side TTFT (includes RPC overhead) | What users experience; same overhead on both runners cancels in comparisons; engine-side requires engine-specific hooks |
| Harness lives on local machine, not in Modal | Simpler; timestamps reflect real client experience; matches Modal RPC pattern users would write |
| Both runners get a `generate_stream` method, generator-style | Streaming is the only way to measure TTFT and per-token latency |
| Generator yields `{"type": ..., ...}` envelopes | Carries token events + metadata events (prompt_info, done) in one stream |
| Record raw per-token timestamps, not derived metrics | Can compute new metrics later without re-running |
| Token count via re-tokenizing final output, not counting yields | `TextIteratorStreamer` yields *text fragments*, not tokens 1:1 |
| Use `Thread` for transformers streaming | Producer-consumer pattern; not performance optimization |

### What I want to keep in mind

- The harness is where benchmark validity lives. A wrong measurement is worse than no measurement.
- For session 2: concurrency 1 only. Resist the urge to add concurrency-4 mid-session.
- Concurrency-1 numbers are a sanity check: vLLM and transformers should be in the same ballpark at concurrency 1. vLLM's win is *batching*, not single-stream speed. If concurrency-1 vLLM is 5x faster, something is measured wrong.

---

### First numbers: transformers runner at concurrency 1

Ran all 30 prompts. Headline summary:

| Tier | n | Prompt tok | Out tok | TTFT mean (ms) | Total (s) | Tput e2e (tok/s) | Tput decode (tok/s) | ITL mean (ms) |
|---|---|---|---|---|---|---|---|---|
| short | 10 | 45 | 203 | 442 | 7.84 | 25.4 | 27.8 | 35.9 |
| medium | 10 | 119 | 256 | 430 | 9.89 | 25.9 | 27.1 | 37.0 |
| long | 10 | 271 | 256 | 433 | 9.89 | 25.9 | 27.1 | 36.9 |

**Headline finding: TTFT is effectively constant across tiers.**

Prompt tokens scale 6x (45 → 271) but TTFT barely moves (442 → 433 ms). That's not what theory predicts — prefill cost should scale with prompt length. Three candidate explanations:

1. **Modal RPC overhead dominates TTFT at these prompt sizes.** Network + serialization round-trip is probably 300-400ms by itself, hiding the prefill signal. Prefill for 271 tokens on A10G is plausibly only 60-150ms, well under the noise floor.
2. **Prefill is genuinely fast at these sizes.** 3B model + bfloat16 + A10G is a quick combination; until you're in the thousands-of-tokens regime, prefill is cheap.
3. **Some Modal cold-effect** — but warmup should rule this out.

Interview-ready phrasing:

> "On my setup, client-observed TTFT for prompts under ~300 tokens was dominated by network and RPC overhead, not prefill compute. To see prefill scale meaningfully you'd need much longer prompts or engine-side timing rather than client-side."

**Second finding: decode throughput is remarkably stable across tiers.** ~26-27 tok/s regardless of prompt length. The KV cache cost of attending over a longer context isn't visible at this scale. Decode at this size is dominated by per-step model forward, not by attention scanning. Would invert at 4K+ context lengths.

**Third finding: tier-level throughput averages are pulled down by prompts that stop early.** Several short-tier prompts (`short_04` Hamlet summary, `short_09` 4-line poem, `short_02` rainy afternoon) naturally hit EOS before the 256-token cap because their content invites short answers. This drags the short-tier mean total time down (~7.8s vs ~9.9s for medium/long) but the *generation rate* stays the same. The slightly lower short-tier e2e throughput (25.4 vs 25.9) is because TTFT becomes a larger fraction of total time when outputs are short.

**Outlier: `short_09` (4-line poem).** 29 output tokens, ITL 30.7ms (lower than every other row). With only ~10 yield events the buffering-vs-real-decode-step ratio swings — this is the "yield events ≠ tokens" effect in extreme form. Don't overread.

### Concepts learned from the data

#### Yield events ≠ tokens

`TextIteratorStreamer` doesn't flush one yield per token. It buffers BPE tokens until they form a clean UTF-8 string, then flushes. So `token_arrival_times` measures **yield arrival times, not per-token times**.

The buffering pattern depends on content. For markdown-heavy output (`**Hash**`, `###`, numbered lists), tokens cluster into 3-4-yield batches at ~120ms intervals. For clean prose, you get roughly one yield per decode step at ~35-40ms intervals. Same underlying physics, different yield distributions.

Implications:
- Re-tokenize the output text to get the canonical `output_tokens` count. Counting yields is wrong.
- Mean ITL from `np.diff(token_arrival_times)` is "mean per-yield gap," not "true per-token decode latency." For an honest decode-latency number, use `decode_time / output_tokens`.
- Histograms of `np.diff(token_arrival_times)` are bimodal for some prompts. Percentile stats on them mean less than they look.

#### KV cache cost grows with context length, but isn't visible at our scale

In theory, every decode step attends over all prior keys/values (prompt + generated so far). At our prompt sizes (45-300 tokens), the per-step cost is dominated by the model forward pass, not by attention scanning. The data shows this clearly: decode throughput is flat across tiers.

To make KV cache cost dominant you'd need prompts in the thousands of tokens, or generation runs long enough that the generated cache itself becomes the dominant context.

#### `.remote()` vs `.remote_gen()` (Modal)

Two different RPC protocols. Regular methods use `.remote()` (single request/response). Generator methods (anything with `yield`) require `.remote_gen()` (streaming response). Modal rejects the wrong variant with a clear error. One-line fix when it happens.

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