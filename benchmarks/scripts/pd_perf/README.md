# P/D dual-stream performance probes

Standalone `torch_npu` probes (no vllm import, no model) that isolate the
execution-layer behaviours the P/D dual-stream feature depends on. They are
meant to be run on the remote NPU and their output fed back here.

## Why these probes exist

The P/D design assumes three things:

1. two aclgraphs replayed on two streams actually overlap on the device;
2. `torch_npu.npu.set_stream_limit` partitions the AI cores between the two
   streams;
3. the per-step host work (two attention-metadata refreshes + two replays) does
   not dominate the step time.

Each assumption maps to a probe below. If an assumption is false on the target
chip/runtime version, the feature degrades no matter how the vllm code is
arranged.

## Root-cause hypothesis (from source reading)

### set_stream_limit is consumed at operator tiling time, not at graph replay time

`torch_npu.npu.set_stream_limit` ends up in `SetStreamResLimit`, which only sets
a global flag and calls `aclrtSetStreamResLimit`:

- `pytorch/torch_npu/csrc/core/npu/NPUFunctions.cpp:559-566`

The actual core count is read later by the operator tiling path:

- `op-plugin/op_plugin/utils/op_api_common.h:428-439` reads
  `GetResInCurrentThread(...CUBE_CORE/VECTOR_CORE)` and passes `aic_num` /
  `aiv_num` into `GetWorkspaceSize` / tiling.

Aclgraph replay does **not** go through that path. `NPUGraph::replay()` simply
submits the pre-captured `model_ri` to the current stream:

- `pytorch/torch_npu/csrc/core/npu/NPUGraph.cpp:392-408`
  (`AclmdlRIExecuteAsync`, no re-tiling).

Therefore a limit set on the **replay** stream after capture is a no-op for the
captured graph. This is exactly the bug test02 confirmed in the original P/D
code: it called `apply_stream_limit()` after capture and targeted
`self.stream` (replay), not `self.capture_stream`.

To bake a core partition into a graph, `set_stream_limit` must be applied to
the **capture** stream **before** `torch.npu.graph(...)`. `torch_npu.npu.graph`
records on the capture stream (`NPUGraph::capture_begin` uses
`capture_stream_`), and the op tiling is computed during capture.

The P/D code has been fixed accordingly: `apply_stream_limit` now targets the
capture stream and the update stream, and is called *before* capture, with
`allow_internal_format = False` set first.

### set_stream_limit also requires `allow_internal_format = False`

The API docs note that core control must be paired with
`torch_npu.npu.config.allow_internal_format = False`; the P/D path never sets
it, so even eager attention-update ops may ignore the limit.

### The two attention-metadata refresh passes are duplicated host work

`PDDualStreamGraphManager.run()` refreshes each graph's attention task groups
before each replay (`pd_separation.py:393-402`). Each refresh re-issues one
eager FIA op per layer via `graph_task_update_begin/end` (see
`vllm_ascend/attention/attention_v1.py` `update_graph_params`). The native path
pays one refresh per step; the P/D path pays two, plus two replays and the
event bookkeeping.

### `stream.synchronize()` before each refresh is likely free in the sync loop

`run()` calls `prefill_ctx.stream.synchronize()` / `decode_ctx.stream.synchronize()`
(`pd_separation.py:393,401`) to protect the single-buffered task-group params.
In the current synchronous engine loop (`sample_tokens` already syncs the
default stream, which waits on both done events), those synchronize calls return
immediately. They would only become a bottleneck if host scheduling is later
made truly async.

## Probes

Run each in a separate process for clean device state.

| Probe | Question it answers |
|-------|---------------------|
| `test01_dual_stream_replay_overlap.py` | Do two compute-bound graphs replayed on two streams actually overlap? |
| `test02_set_stream_limit_effect.py` | Does `set_stream_limit` affect the graph? On the replay stream or only on the capture stream? |
| `test03_host_enqueue_overhead.py` | How much CPU time does the per-step dual update+replay enqueue cost? |
| `test04_memory_bandwidth_contention.py` | Is HBM bandwidth the limiter (decode is memory-bound)? |

### test01

```bash
python benchmarks/scripts/pd_perf/test01_dual_stream_replay_overlap.py
```

Watch `overlap_ratio` (1.0 = full overlap). If it is near 0, the device is
serializing the two streams and there is no execution-layer win to recover.

### test02

```bash
python benchmarks/scripts/pd_perf/test02_set_stream_limit_effect.py \
    --cube 8 --vector 16
```

Expected if the hypothesis holds:

- `eager_limit/eager_nolimit > 1` (the control works for eager ops);
- `graph_replay_limit/graph_nolimit ~ 1` (replay-stream limit is a no-op);
- `graph_capture_limit/graph_nolimit > 1` (capture-time limit is baked in).

If confirmed, the fix is to move `set_stream_limit` onto `capture_stream`
before capture and set `torch_npu.npu.config.allow_internal_format = False`.

### test03

```bash
python benchmarks/scripts/pd_perf/test03_host_enqueue_overhead.py \
    --num-updates 32 --dim 512 --layers 2
```

Compare `dual_update` against the step time measured by the model-level
`pd_dual_stream_microbench.py`. If `dual_update` is a large fraction of the
step, the attention-refresh host cost (and/or the lack of async scheduling) is
the bottleneck.

### test04

```bash
python benchmarks/scripts/pd_perf/test04_memory_bandwidth_contention.py \
    --numel 16777216 --copies 8
```

Compare its speedup with test01. Low speedup here but high overlap in test01
means HBM bandwidth, not AI cores, is the limit.

## Follow-up verification after the vllm fix

The `pd_separation.py` fix is now applied:

1. `allow_internal_format = False` is set before capture.
2. `apply_stream_limit` now targets `capture_stream` and `update_stream` and is
   called *before* each `torch.npu.graph(...)`.

Remaining checks on the model-level path:

1. Profile both streams with the CANN profiler (`kernel_details.csv` /
   `op_summary.csv`) to confirm the two graphs overlap and that the `BlockDim`
   now reflects the configured cube/vector counts.
2. Re-run `pd_dual_stream_microbench.py --mode pd` and compare against
   `--mode baseline-nc-limit` (the limited sequential baseline) rather than the
   unlimited baseline, because a single stream also pays the partition cost
   when the limit is effective.
3. Confirm `graph_task_update` (the per-step attention refresh) still produces
   correct output with the now-limited `update_stream` tiling.
