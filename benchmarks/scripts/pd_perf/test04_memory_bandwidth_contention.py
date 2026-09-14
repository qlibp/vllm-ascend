"""Test 04: HBM bandwidth contention between two concurrently-replayed graphs.

Even with perfect AI-core partitioning, the two streams share the same HBM
controller(s).  If the workload is memory-bound (decode is dominated by KV-cache
reads and small reductions), running two memory-bound graphs concurrently may
not beat running them sequentially because both are limited by the same memory
bandwidth.

This probe captures two memory-bound graphs (large copies + adds) and compares
concurrent vs sequential replay, exactly as test01 does but for a memory-bound
workload.  Compare its ``overlap_ratio``/``speedup`` with test01 (compute-bound):

* compute-bound overlap high but memory-bound overlap low -> the P/D degradation
  is HBM bandwidth, not AI-core contention.
* both low -> the streams are serialized at the scheduler level.
"""

from __future__ import annotations

import argparse

import torch

import common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    parser.add_argument("--numel", type=int, default=16 * 1024 * 1024, help="elements per tensor")
    parser.add_argument("--copies", type=int, default=8)
    args = parser.parse_args()

    device = common.get_device(args.device)

    cap_a = torch.npu.Stream(device=device)
    cap_b = torch.npu.Stream(device=device)
    stream_a = torch.npu.Stream(device=device)
    stream_b = torch.npu.Stream(device=device)

    # Each graph is a fresh memory-bound workload (independent buffers).
    graph_a, out_a = common.capture_graph(
        cap_a, common.make_memory_fn(args.numel, args.copies, device)
    )
    graph_b, out_b = common.capture_graph(
        cap_b, common.make_memory_fn(args.numel, args.copies, device)
    )
    common.sync()

    print(
        f"[setup] numel={args.numel} copies={args.copies} "
        f"bytes_per_graph={args.numel * 4 * args.copies / (1024 ** 2):.1f} MiB moved"
    )

    result = common.bench_dual_stream_overlap(
        graph_a, stream_a, graph_b, stream_b, args.iters, args.warmup
    )

    print("\n== Result ==")
    for k, v in result.items():
        print(f"{k}: {v:.3f}")

    print("\n== Interpretation ==")
    print(
        f"memory-bound concurrent/sequential speedup: "
        f"{result['speedup_concurrent_vs_sequential']:.3f}x"
    )
    if result["speedup_concurrent_vs_sequential"] < 1.2:
        print("=> HBM bandwidth is the limiting factor; dual-stream does not help here.")
    else:
        print("=> memory bandwidth is not saturated; contention is elsewhere.")


if __name__ == "__main__":
    main()
