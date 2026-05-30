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

### vLLM streaming: the bridging saga

Trying to wire up vLLM's `AsyncLLMEngine` streaming into a Modal `generate_stream` method took several false starts. Worth writing up because the technical lesson and the meta-lesson are both interview-relevant.

#### The architectural choice

vLLM offers `LLM` (sync, batch-style) and `AsyncLLMEngine` (async, streaming-native). Chose `AsyncLLMEngine` because it's what production serving uses — closer to interview-relevant territory.

#### The wrong approach: bridge sync to async with queue + thread

First attempt: keep the Modal method synchronous (`def generate_stream`), drive vLLM's async engine in a background thread via `asyncio.run`, use a thread-safe `queue.Queue` as the producer-consumer channel.

```python
@modal.method()
def generate_stream(self, prompt: str):
    q = queue.Queue()
    
    async def async_producer():
        async for output in self.engine.generate(...):
            q.put(...)
        q.put(_DONE)
    
    def thread_main():
        asyncio.run(async_producer())  # new event loop per request
    
    thread = threading.Thread(target=thread_main, daemon=True)
    thread.start()
    
    while True:
        item = q.get()
        if item is _DONE: break
        yield item
```

This *seemed* right. It compiled, it ran, and produced a generic `EngineDeadError` from `add_request`. After several hypotheses (FlashInfer JIT, prompt keyword, Triton compilation, CUDA toolkit absence), the real root cause turned out to be:

**`AsyncLLMEngine` requires a persistent event loop.** It lazily starts background tasks on whatever loop is running when `generate()` is first called. `asyncio.run()` creates a fresh loop per invocation and closes it on return — so the engine's tasks get attached to a loop that's immediately torn down. The engine becomes unusable.

#### The right approach: make the Modal method `async def`

Modal supports async generator methods natively. The client side (`.remote_gen()`) bridges async-to-sync automatically. The whole queue/thread machinery is unnecessary:

```python
@modal.method()
async def generate_stream(self, prompt: str):
    yield {"type": "prompt_info", "prompt_tokens": prompt_tokens}
    
    prev = ""
    async for output in self.engine.generate(
        text, self.sampling_params, request_id=str(uuid.uuid4())
    ):
        cumulative = output.outputs[0].text
        delta = cumulative[len(prev):]
        if delta:
            yield {"type": "token", "text": delta}
        prev = cumulative
    
    yield {"type": "done"}
```

Modal handles the event loop. vLLM gets the persistent loop it needs. The harness is unchanged — `.remote_gen()` returns a sync iterator either way.

#### The technical lesson

When you have an async producer (vLLM), an async transport (Modal), and your code is the glue — **make the glue async too**. The queue/thread/asyncio.run pattern is for when one of the boundaries genuinely forces sync. Neither did here.

Decision tree:
1. Is the engine async-native? → Yes (vLLM)
2. Does the transport support async? → Yes (Modal)
3. Then write the method as `async def`. Don't bridge.

Reach for queue+thread bridging only when one boundary is sync-only (a library that must be sync, a framework without async support).

#### The meta-lesson: diagnose before iterating on implementation

After the first failure, I (Claude) proposed fixes inside the queue/thread framing: improve error reporting, swap base images, try positional args. None worked. The bug never was inside that framing — it was the framing itself.

The right move at the second or third failure would have been to step back and ask "is the architecture right?" rather than "what's the next thing to try inside the architecture?" Iterating on a wrong frame produces tired engineers and no progress. Generalizable: when iterating isn't working, change the frame.

This is the same principle Claude has been coaching me on for interviews — diagnose before prescribing, exhaust cheap interventions before architectural changes — applied in reverse. Worth flagging.

#### Side concept: how exceptions cross boundaries (threads, processes, RPC)

Hunting the original error surfaced something fundamental about exception propagation.

**Exceptions don't cross boundaries for free.** Whenever code spans a thread, process, or network boundary, you must:
1. Catch the exception inside the boundary
2. Serialize its useful state (type, message, **full traceback as a string**)
3. Propagate the serialized form across
4. Reconstruct/raise on the other side

Anything not explicitly serialized is lost.

Specifically:
- Thread exceptions die silently by default (no auto-propagation to main thread)
- `traceback` objects are not picklable — they reference C-level frame objects bound to the originating interpreter
- `traceback.format_exc()` produces a plain string that survives any boundary
- This applies to Python threads, multiprocessing, Celery, gRPC, REST, Modal RPC — same principle, different surface

Caught the exception in the producer (good), but only stringified the message (`str(exc)`), not the traceback. That's why the first error was "EngineCore encountered an issue" with no detail — the actual traceback was attached to the exception object, but never extracted before the object was discarded.

The fix:
```python
except BaseException as exc:
    import traceback
    q.put(_Error(exc, traceback.format_exc()))
```

Once the traceback was preserved across the boundary, the real error (`EngineDeadError` from `add_request`) was immediately visible.

### Session 2 final results: vLLM vs transformers at concurrency 1

Both runners working end-to-end. Full 30-prompt benchmark on each.

| Metric | Tier | transformers | vLLM | Delta |
|---|---|---|---|---|
| TTFT mean (ms) | short | 442 | 457 | +3% |
| | medium | 430 | 435 | +1% |
| | long | 433 | 442 | +2% |
| ITL mean (ms) | short | 35.9 | 15.2 | **–58%** |
| | medium | 37.0 | 15.3 | **–59%** |
| | long | 36.9 | 15.4 | **–58%** |
| Decode tput (tok/s) | short | 27.8 | 66.6 | **+139%** |
| | medium | 27.1 | 65.6 | **+142%** |
| | long | 27.1 | 65.4 | **+141%** |
| E2E tput (tok/s) | short | 25.4 | 55.7 | **+119%** |
| | medium | 25.9 | 59.0 | **+128%** |
| | long | 25.9 | 58.7 | **+127%** |
| Total time (s) | short | 7.84 | 3.62 | **–54%** |
| | medium | 9.89 | 4.34 | **–56%** |
| | long | 9.89 | 4.36 | **–56%** |

#### Headline findings

**1. vLLM at concurrency 1 is ~2.4x faster on decode than transformers.** Bigger than I expected. This is *without* batching — there's only one request in flight, so vLLM's continuous batching isn't doing anything. The win is purely from per-stream efficiency: FlashAttention kernels, native sampler, cleaner Python/GPU memory boundary, less per-step overhead.

Interview-ready phrasing:

> "On my own setup, vLLM at concurrency 1 was about 2.4x faster on per-token decode than naive transformers, on a 3B model on an A10G. That's entirely from kernel efficiency — continuous batching is inactive at concurrency 1. The batching wins come on top of this baseline once concurrency rises."

**2. TTFT is essentially identical across both runners** (within 1-3%). Confirms the prior hypothesis: RPC overhead dominates TTFT at these prompt sizes, drowning out any prefill differences between engines.

Interview-ready phrasing:

> "Client-side TTFT for prompts under ~300 tokens was dominated by network and RPC overhead, hiding any prefill differences between engines. To see prefill scale, you'd need much longer prompts or engine-side timing."

**3. ITL is remarkably tier-flat for both runners.** Decode step cost doesn't depend on prompt length at these sizes. KV cache effects aren't visible until much longer contexts.

**4. Absolute time savings grow with output length.** Short tier saves ~4.2s; medium and long save ~5.5s each. The longer the generation, the more decode efficiency compounds into wall-time savings.

#### Sanity check on the absolute numbers

For a 3B model on A10G in bfloat16, A10G memory bandwidth is ~600 GB/s. Decode is memory-bandwidth-bound for small batches. Theoretical ceiling ~100-120 tok/s. Observed:

- transformers @ 27 tok/s = ~25% of bandwidth — sensible for naive PyTorch
- vLLM @ 65 tok/s = ~55% of bandwidth — sensible for optimized kernels

Both numbers pass the sniff test. If transformers had been 5 tok/s or vLLM 200 tok/s, something would be broken.

#### What's still missing for "interview-ready"

The 2.4x is the per-stream win. Concurrency >1 is where vLLM's *architectural* win (continuous batching) shows up. Session 3 will produce the batching numbers — that's where the throughput curve really takes off and the cost-per-token economics become defensible.

### Core concepts: prefill vs decode, and what "kernel" means

Two foundational concepts that come up constantly in LLM serving discussions. Worth crisp definitions.

#### Two phases of inference: prefill and decode

LLM generation has two phases with **very different compute characteristics**.

**Prefill.** Process the entire input prompt in one big forward pass. All prompt tokens go through every transformer layer simultaneously, computing the keys and values for each token at each layer (this populates the "KV cache"). One forward pass, but it processes hundreds of tokens at once.

**Decode.** Generate output, one token at a time. For each new token:
1. Run the model on *just the one most recently generated token*, attending over all the cached KVs from the prompt + previous output tokens.
2. Get logits for the next token.
3. Sample / argmax.
4. Detokenize the chosen ID into text.
5. Repeat.

So "decode phase" = "the autoregressive generation loop." Each iteration is one forward pass with sequence length 1.

**The bottlenecks are different:**

| Phase | Seq len per forward | Bottleneck | Why |
|---|---|---|---|
| Prefill | hundreds of tokens | Compute-bound | Lots of parallel matmul work; GPU compute units saturate |
| Decode | 1 token | Memory-bandwidth-bound | Tiny compute per step, but must read all 5.79GB of weights every step |

This explains why TTFT (mostly prefill) and ITL (per-decode-step) move so differently:
- TTFT roughly scales with prompt length × model size, bounded by compute
- ITL is roughly constant per step regardless of prompt length, bounded by memory bandwidth

**The two meanings of "decode" — a vocabulary trap:**

There's a separate, narrower meaning of "decode": the tokenizer step `tokenizer.decode(token_id) → text` that turns an integer back into a string. This is the inverse of tokenizer-encode and happens *inside* each decode-phase step (step 4 in the list above). It's microseconds — pure string lookup.

When people say "decode is the bottleneck" in serving, they mean the *whole autoregressive loop* (the model forward pass dominates), not the tokenizer.decode call. Distinction worth keeping straight.

**Interview-ready phrasing:**

> "Prefill is compute-bound — long sequence in one forward pass, lots of parallel matmul work, GPU saturates. Decode is memory-bandwidth-bound — each step does very little compute but has to read all the model's weights from HBM to produce one token. Optimizing the two is basically a different problem."

#### What is a "kernel"?

A **kernel** in GPU programming is just a function that runs on the GPU. Terminology comes from CUDA.

It's overloaded with unrelated meanings elsewhere (OS kernel, kernel functions in math/ML, kernel methods in SVMs). All unrelated. In serving discussions: kernel = "GPU function written in CUDA C++, compiled by nvcc, runs on thousands of GPU threads in parallel."

When you call `torch.matmul(A, B)` in Python, PyTorch eventually calls a CUDA kernel under the hood. The kernel does the actual math; PyTorch is orchestration.

**Why kernel efficiency varies:**

The same operation can be implemented as many different kernels with very different performance. Concrete: matrix multiplication has dozens of CUDA implementations, each making different choices about:
- How to split work across thousands of GPU threads
- Which memory tier (registers vs shared memory vs HBM) at each step
- How to overlap computation with memory loading
- Boundary handling for non-divisible dimensions
- Whether to use specialized tensor-core instructions

Naive matmul kernel: ~10% of theoretical throughput. Well-tuned (cuBLAS): 90%+. Same math, 9x performance difference. *That's* kernel efficiency.

#### Why vLLM's kernels beat transformers' kernels

`transformers` is general-purpose: arbitrary models, debuggability, broad compatibility. Default ops call generic CUDA kernels (cuBLAS, cuDNN) — excellent for general math, blind to LLM-specific structure.

vLLM is purpose-built for LLM inference. Specific wins:

1. **FlashAttention.** Standard attention: compute `Q @ K^T` → huge intermediate matrix → softmax → multiply by `V`. The intermediate gets written to HBM and read back. FlashAttention restructures so the intermediate stays in on-chip memory. Same math, much less memory traffic. ~2-4x on attention specifically. `transformers` can use it (`attn_implementation="flash_attention_2"`) but doesn't by default; vLLM does by default.

2. **Fused operations.** Each kernel launch has overhead (CPU → GPU dispatch). At decode (seq_len=1), individual ops are tiny, so launch overhead dominates. Naive PyTorch: layer norm, linear, activation, residual = 4 separate kernel calls. vLLM fuses these into single kernels. Modest individually (~1.2x) but adds up.

3. **PagedAttention.** vLLM's signature contribution. Standard attention assumes contiguous KV cache memory; wastes memory with variable-length requests or batches. PagedAttention manages cache in non-contiguous pages (like OS virtual memory), enabling much better utilization and supporting continuous batching efficiently. More memory-layout than raw kernel speed, but enables behaviors that pure kernel optimization can't.

4. **Optimized sampler kernels.** Top-k, top-p, temperature scaling. PyTorch's defaults are general but not optimized for "small batch, low latency, high frequency." vLLM uses FlashInfer (or its own native sampler — what we forced with `VLLM_USE_FLASHINFER_SAMPLER=0`).

5. **Reduced Python overhead per step.** Each PyTorch forward involves non-trivial Python work — kernel dispatch, type checks, etc. For decode (tiny per step), this matters. vLLM has a tighter inner loop.

**The wins compound, they don't add.** Each is modest individually (1.2-2x). Multiplied together: ~2-3x, which matches our observed 2.4x decode speedup.

**Interview-ready phrasing:**

> "vLLM's single-stream win over transformers comes from a handful of LLM-specific kernel optimizations: FlashAttention to reduce attention's memory traffic, fused operations to reduce kernel-launch overhead, optimized sampler kernels, and a tighter Python inner loop. Each is modest individually — maybe 20-50% — and they compound. On my A10G with a 3B model I observed about 2.4x, which is consistent with compounded gains. The architectural win — continuous batching via PagedAttention — is separate and only activates at concurrency >1."

This is the answer that signals you know the difference between "vLLM is fast because batching" (the marketing answer) and the actual mechanism.

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