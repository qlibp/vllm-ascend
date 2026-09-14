"""Test 02: does ``set_stream_limit`` actually partition the aclgraph?

This is the most important probe for the P/D dual-stream design.

The design (`pd_separation.py`` / design doc invariant 7) calls
``torch_npu.npu.set_stream_limit`` on the **replay** stream *after* the graph is
captured, assuming this partitions the AI cores between the two replays.

From the source, ``set_stream_limit`` is consumed at **operator tiling time**
through ``aclrtUseStreamResInCurrentThread`` / ``aclrtGetResInCurrentThread``
(op-plugin ``EXEC_NPU_CMD_V2``).  Aclgraph replay
(``NPUGraph::replay`` -> ``AclmdlRIExecuteAsync``) does NOT re-tile; it only
submits the SQE sequence that was baked at capture time.  Therefore a limit set
on the replay stream after capture should have no effect on the replay.

This probe verifies exactly that by comparing four cases:

* ``eager_nolimit`` / ``eager_limit``  -- control: confirm the limit mechanism
  actually works for eager (non-graph) operator execution.
* ``graph_nolimit``                    -- graph captured and replayed without any
  limit.
* ``graph_replay_limit``               -- graph captured without limit, then the
  limit is set on the replay stream (the current P/D behaviour).
* ``graph_capture_limit``              -- limit set on the *capture* stream
  *before* capture, then replayed without any replay-stream limit.

Expected outcome if the hypothesis is right:

* ``eager_limit`` > ``eager_nolimit`` (limiting cores slows a compute-bound op).
* ``graph_replay_limit`` ~= ``graph_nolimit`` (replay-stream limit is a no-op).
* ``graph_capture_limit`` > ``graph_nolimit`` (capture-time limit is baked in).
"""

from __future__ import annotations

import argparse

import torch

import common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--cube", type=int, default=8)
    parser.add_argument("--vector", type=int, default=16)
    args = parser.parse_args()

    device = common.get_device(args.device)
    common.disable_internal_format()

    print(
        f"[setup] device={device} dim={args.dim} layers={args.layers} "
        f"limit_cube={args.cube} limit_vector={args.vector}"
    )

    weights, biases = common.make_mlp_weights(args.dim, args.layers, device)
    x = torch.randn(args.dim, args.dim, device=device)
    compute_fn = common.make_compute_fn(weights, biases)

    # ------------------------------------------------------------------ #
    # 1. Eager control: does set_stream_limit work at all for eager ops?
    # ------------------------------------------------------------------ #
    eager_stream = torch.npu.Stream(device=device)

    def eager_run():
        with torch.npu.stream(eager_stream):
            compute_fn(x)

    eager_nolimit = common.time_call_ms(eager_run, args.iters, args.warmup, stream=eager_stream)

    common.set_stream_limit(eager_stream, args.cube, args.vector)
    lim_info = common.get_stream_limit(eager_stream)
    print(f"[eager] after set_stream_limit -> {lim_info}")
    eager_limit = common.time_call_ms(eager_run, args.iters, args.warmup, stream=eager_stream)
    common.reset_stream_limit(eager_stream)

    # ------------------------------------------------------------------ #
    # 2. Graph control: no limit anywhere.
    # ------------------------------------------------------------------ #
    capture_no = torch.npu.Stream(device=device)
    replay_no = torch.npu.Stream(device=device)
    graph_no, out_no = common.capture_graph(
        capture_no, lambda: compute_fn(x)
    )
    graph_no_replay = common.time_call_ms(
        common.replay_on(graph_no, replay_no), args.iters, args.warmup, stream=replay_no
    )

    # ------------------------------------------------------------------ #
    # 3. Current P/D behaviour: limit on the replay stream, after capture.
    # ------------------------------------------------------------------ #
    capture_rep = torch.npu.Stream(device=device)
    replay_rep = torch.npu.Stream(device=device)
    graph_rep, out_rep = common.capture_graph(
        capture_rep, lambda: compute_fn(x)
    )
    common.set_stream_limit(replay_rep, args.cube, args.vector)
    rep_lim_info = common.get_stream_limit(replay_rep)
    print(f"[graph_replay_limit] replay stream limit -> {rep_lim_info}")
    graph_replay_limit = common.time_call_ms(
        common.replay_on(graph_rep, replay_rep), args.iters, args.warmup, stream=replay_rep
    )

    # ------------------------------------------------------------------ #
    # 4. Limit on the capture stream, BEFORE capture.
    # ------------------------------------------------------------------ #
    capture_pre = torch.npu.Stream(device=device)
    replay_pre = torch.npu.Stream(device=device)
    common.set_stream_limit(capture_pre, args.cube, args.vector)
    cap_lim_info = common.get_stream_limit(capture_pre)
    print(f"[graph_capture_limit] capture stream limit -> {cap_lim_info}")
    graph_pre, out_pre = common.capture_graph(
        capture_pre, lambda: compute_fn(x)
    )
    graph_capture_limit = common.time_call_ms(
        common.replay_on(graph_pre, replay_pre), args.iters, args.warmup, stream=replay_pre
    )
    common.reset_stream_limit(capture_pre)

    # Keep outputs alive.
    _ = (out_no, out_rep, out_pre)

    print("\n== Result (median_ms) ==")
    print(f"eager_nolimit        : {eager_nolimit['median_ms']:.3f}")
    print(f"eager_limit          : {eager_limit['median_ms']:.3f}")
    print(f"graph_nolimit        : {graph_no_replay['median_ms']:.3f}")
    print(f"graph_replay_limit   : {graph_replay_limit['median_ms']:.3f}")
    print(f"graph_capture_limit  : {graph_capture_limit['median_ms']:.3f}")

    print("\n== Interpretation ==")
    eager_ratio = eager_limit["median_ms"] / max(eager_nolimit["median_ms"], 1e-6)
    replay_ratio = graph_replay_limit["median_ms"] / max(graph_no_replay["median_ms"], 1e-6)
    capture_ratio = graph_capture_limit["median_ms"] / max(graph_no_replay["median_ms"], 1e-6)
    print(f"eager_limit/eager_nolimit      = {eager_ratio:.3f} (>1 means core limit works for eager)")
    print(f"graph_replay_limit/graph_nolimit = {replay_ratio:.3f} (~1 means replay-stream limit is a no-op)")
    print(f"graph_capture_limit/graph_nolimit = {capture_ratio:.3f} (>1 means capture-time limit is baked into the graph)")

    if replay_ratio < 1.05 and capture_ratio > 1.05:
        print(
            "\n=> CONFIRMED: set_stream_limit must be applied to the capture stream "
            "before capture. The current P/D code applies it to the replay stream "
            "after capture, so the two graphs are NOT actually core-partitioned."
        )
    elif replay_ratio > 1.05:
        print(
            "\n=> Unexpected: replay-stream limit DID change replay time. "
            "Re-check the runtime version / whether replay re-tiles."
        )
    else:
        print("\n=> Inconclusive (timing noise or limits not enforced). Check get_stream_limit output above.")


if __name__ == "__main__":
    main()
