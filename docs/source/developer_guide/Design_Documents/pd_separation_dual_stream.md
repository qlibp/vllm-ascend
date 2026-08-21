# Single-card Prefill/Decode Separation (Dual npu-stream)

## Why single-card P/D separation?

In a mixed batch, the default vLLM Ascend scheduler packs prefill and decode
tokens into one `SchedulerOutput` and the model runs on the **default stream**
with a single FULL/PIECEWISE aclgraph. A long prefill request therefore blocks
all decode requests in the same step, hurting **TPOT** (time per output token)
under concurrent load.

This feature keeps the workload on **one card** but separates it into two
independent npu-streams:

* a **prefill-stream** replaying a **prefill-graph**, and
* a **decode-stream** replaying a **decode-graph**.

Both graphs run *concurrently* on the same card, share the model weights and the
kv-cache (they live in the same worker process), and are kept from trampling each
other by a set of invariants described below. Scheduling does **not** use
chunk-prefill: prefill requests are batched with prefill, decode with decode.

> The stream/partition mechanics follow the pattern validated by
> `pytorch/npu_graph_parallel_stream_demo.py`.

---

## Usage

Set the master switch and (optionally) the stream core-partition knobs, then
start vLLM normally.

```bash
VLLM_ASCEND_ENABLE_PD_SEPARATION=1 \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-0.5B-Instruct
```

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `VLLM_ASCEND_ENABLE_PD_SEPARATION` | `0` | Master switch. `1` enables the `SchedulerPDSeparation` scheduler and the `PDDualStreamModelRunner`. |
| `VLLM_ASCEND_PD_SEPARATION_PREFILL_CUBE_NUM` | `12` | AI **cube** cores partitioned to the prefill-stream. `-1` keeps the runtime default (skip `set_stream_limit`). |
| `VLLM_ASCEND_PD_SEPARATION_PREFILL_VECTOR_NUM` | `24` | AI **vector** cores partitioned to the prefill-stream. `-1` keeps the runtime default. |
| `VLLM_ASCEND_PD_SEPARATION_DECODE_CUBE_NUM` | `12` | AI **cube** cores partitioned to the decode-stream. `-1` keeps the runtime default. |
| `VLLM_ASCEND_PD_SEPARATION_DECODE_VECTOR_NUM` | `24` | AI **vector** cores partitioned to the decode-stream. `-1` keeps the runtime default. |
| `VLLM_ASCEND_PD_SEPARATION_MAX_PREFILL_TOKENS` | `-1` | Static prefill-graph batch size in tokens. `-1` derives it from `max_num_batched_tokens`. |
| `VLLM_ASCEND_PD_SEPARATION_MAX_DECODE_TOKENS` | `-1` | Static decode-graph batch size in tokens. `-1` derives it from `max_num_seqs` (one token per request). |

When the master switch is on, `platform.py` also forces
`enable_chunked_prefill = False` because the scheduler is designed around
non-chunked prefill.

---

## How It Works

### 1. Design approach

A single scheduler step is split into two disjoint request groups:

1. The scheduler (`SchedulerPDSeparation`) reorders `self.running` so decode
   requests come first, each advanced by exactly `decode_query_len`
   (`1 + num_spec_tokens`) tokens, then schedules prefill requests with their
   **full remaining prompt** (subject to the token budget — if the budget is
   insufficient the request waits, it is *never* chunked).
2. The runner (`PDDualStreamModelRunner`) splits the combined
   `num_scheduled_tokens` back into a prefill group and a decode group via
   `split_prefill_decode` (a request is *decode* iff it is advanced by exactly
   `decode_query_len` tokens).
3. Each group's inputs are staged into its own static buffers, and the two
   groups are replayed concurrently on two streams by
   `PDDualStreamGraphManager`.

```
                       ┌──────────────────────────────────┐
                       │        Scheduler (one step)       │
                       │  decode-first, no chunk-prefill   │
                       └───────────────┬──────────────────┘
                                       │ one SchedulerOutput
                                       ▼
                 split_prefill_decode(num_scheduled_tokens)
                          │                    │
             prefill group │                    │ decode group
                          ▼                    ▼
                ┌──────────────────┐  ┌──────────────────┐
                │  prefill-stream  │  │   decode-stream  │
                │  prefill-graph   │  │   decode-graph   │
                │  (own pool +     │  │  (own pool +     │
                │   own buffers)   │  │   own buffers)   │
                └────────┬─────────┘  └────────┬─────────┘
                         │     concurrent replay     │
                         └────────────┬─────────────┘
                                      ▼
                           concat → compute logits → sample
```

### 2. Implementation design

The feature is fully contained in `vllm-ascend` and gated behind
`VLLM_ASCEND_ENABLE_PD_SEPARATION` (off by default):

* **`vllm_ascend/core/scheduler_pd_separation.py`** — `SchedulerPDSeparation`,
  a `Scheduler` subclass that overrides `schedule()` to produce the decode-first,
  no-chunk ordering. It reuses the parent's slot allocation and
  `SchedulerOutput` construction helpers.
* **`vllm_ascend/worker/pd_separation.py`** — the mechanics layer:
  * `split_prefill_decode` — pure helper that classifies each request.
  * `PDSeparationConfig` — reads the env vars above.
  * `PDStreamContext` — owns one stream, one capture stream, one `NPUGraph`,
    and that graph's static input/output buffers.
  * `PDDualStreamGraphManager` — captures the two graphs on side capture-streams
    (deliberately *without* a shared graph pool) and concurrently replays them.
* **`vllm_ascend/worker/model_runner_pd.py`** — `PDDualStreamModelRunner`, an
  `NPUModelRunner` subclass overriding `execute_model` and `sample_tokens` while
  reusing the parent's input preparation, attention-metadata construction,
  model forward, logits and sampling.
* **`vllm_ascend/platform.py`** — registers `SchedulerPDSeparation` and forces
  `enable_chunked_prefill = False` when the switch is on.
* **`vllm_ascend/worker/worker.py`** — selects `PDDualStreamModelRunner` in
  `init_device` when the switch is on.

### 3. Core invariants

These must not be broken; they are the contract that makes concurrent replay
correct.

1. **The two graphs never share a graph pool.** `torch.npu.graph` is called
   *without* `pool=` for each graph, so each one gets a private pool. Sharing a
   pool across two concurrently-replayed graphs would let their scratch buffers
   trample each other.
2. **The two graphs never share static input/output buffers.** Every buffer is
   owned by exactly one `PDStreamContext`. This is the real trampling risk — not
   the kv-cache.
3. **Capture runs with `aclgraph_runtime_mode=NONE`.** The model is wrapped in
   an `ACLGraphWrapper`; in `FULL` mode that wrapper would capture its *own*
   sub-graph and conflict with the outer raw `torch.npu.graph` capture. `NONE`
   makes it pass through eagerly so the outer capture records the forward.
4. **kv-cache is shared, on purpose, and safe.** Both graphs run in the same
   worker process against the same `kv_caches`, but the scheduler hands the
   prefill group and the decode group **disjoint blocks**, so no kv-cache slot is
   ever written by both streams in the same step.
5. **Event-level sync, no device barrier on the hot path.** Before replay each
   stream `wait_stream(default_stream)` so the host→device copies of the static
   buffers are visible; each stream records a `done_event` immediately after
   replay; the default stream `wait_event`s both done events before reading
   logits. `torch.npu.synchronize()` is called *only* once, after capture.
6. **The single cross-stream dependency is the cross-step P→D handoff.** Step
   N+1's decode stream `wait_event`s step N's prefill `done_event` (a request
   that just prefilled now enters decode and its kv-cache must be visible).
   Within a step the two streams never wait on each other.
7. **Decode-first token order.** The scheduler reorders `self.running`
   decode-first, and the runner reorders its persistent `input_batch` the same
   way before slicing the per-token tensors, so decode tokens are contiguous at
   the front (`input_ids[:num_decode_tokens]`) and prefill tokens at the tail.
8. **`set_stream_limit` is applied only when enabled.** A `-1` cube/vector count
   disables `set_stream_limit` for that stream and keeps the runtime default.

### 4. Execution-level pipeline boundary

The two streams are **persistent** and keep computing without a device-level
barrier, but the host engine loop is still step-by-step. vLLM v1 drives the
runner synchronously (`schedule → execute_model → sample_tokens →
update_from_output`), and on a single card the `UniProcExecutor` runs
`execute_model` in the same process — there is no background thread, so
`non_block=True` still completes the step before returning. Consequently the
"continuous" behavior is at the **NPU stream level**: the prefill-kernels and
decode-kernels interleave on the device without any `torch.npu.synchronize()`
in the hot path, while the host advances one step at a time.

Within that boundary the only cross-stream dependency is the cross-step P→D
handoff: step N+1's decode stream waits on step N's prefill `done_event`
(because a request that prefilled at step N enters decode at step N+1 and its
kv-cache writes must be visible). Within a step the prefill and decode streams
are fully independent — they are replayed back-to-back and overlap on the
device. This is a genuine *pipeline* at the execution layer; full scheduling
decoupling (overlapping the host schedule of step N+1 with the execution of
step N) is out of scope and would require changes to vLLM core.

---

## Tuning the cube/vector core counts

The two streams partition the device's AI cores via
`torch_npu.npu.set_stream_limit(stream, cube_num=..., vector_num=...)`. The
defaults (12 cube / 24 vector per stream) are a starting point, not a tuned
value. To tune:

* Give the **prefill-stream** a larger `cube_num` when prefill is compute-bound
  (long prompts, large `max_num_batched_tokens`).
* Give the **decode-stream** a larger share when decode is memory/latency-bound
  and TPOT matters most.
* The sum of the two streams' `cube_num` should not exceed the device's cube core
  count; likewise for `vector_num`. Exceeding it is rejected by the runtime.
* Set a value to `-1` to leave that stream on the runtime default (no
  `set_stream_limit` call).

Profile both streams (e.g. with the CANN profiler) to confirm the two graphs are
actually overlapping and to measure the TTFT/TPOT trade-off before changing the
partition.

---

## Limitations

* Single card only (qwen2-0.5b target). No distributed concerns.
* Not compatible with: speculative decoding, pipeline parallelism, context
  parallelism (decode or prefill), multimodal / encoder-decoder models, LoRA,
  and `kv_sharing_fast_prefill`. These are rejected at runner construction.
* Degenerate one-token prompts are indistinguishable from decode steps by the
  `num_tokens == decode_query_len` rule and are treated as decode. Such a
  request is also placed in the prefill region by the decode-first reorder, so
  it may be mis-split when mixed with a real prefill in the same step.
* The static prefill/decode graphs capture a fixed batch shape
  (`MAX_PREFILL_TOKENS` / `MAX_DECODE_TOKENS`). A prefill request larger than the
  static prefill batch is *not* chunked — it waits for a step where it fits.
* **Per-group attention-metadata refresh is the remaining integration point
  (not yet resolved).** The graphs are captured with *dummy* attention metadata
  (block tables / slot mappings / sequence lengths). Before each replay those
  per-group attention params must be refreshed into the graph workspaces for
  the actual requests; until then the graphs compute against stale kv-cache
  addresses. This is flagged `TODO(remote-validate)` in
  `model_runner_pd.py::_run_dual_stream` and must be validated on the NPU.
