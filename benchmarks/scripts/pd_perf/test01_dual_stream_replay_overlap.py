"""Test 01: does concurrent dual-stream aclgraph replay actually overlap?

This isolates the *raw* replay mechanism from every vllm detail (attention
metadata update, staging copies, sampling, scheduler).  Two identical
compute-bound graphs are captured on two private capture streams and replayed:

* sequentially -- both on stream A (the device serializes them), and
* concurrently -- one on stream A and one on stream B.

If the NPU truly co-schedules the two streams, ``concurrent_mean_ms`` should be
close to ``max(device_a, device_b)`` (i.e. close to one graph's duration) and
``overlap_ratio`` close to 1.0.  If it stays near ``device_a + device_b``, the
device is serializing the two streams and P/D dual-stream cannot win at the
execution layer regardless of the vllm code.

Why this matters: ``pd_separation.py`` replays the prefill and decode graphs on
two streams, but the two graphs were each captured with full-core tiling (see
test02).  This test tells us whether the runtime/hardware gives any overlap at
all in that situation.
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
    parser.add_argument("--disable-internal-format", action="store_true")
    args = parser.parse_args()

    device = common.get_device(args.device)
    if args.disable_internal_format:
        common.disable_internal_format()

    print(f"[setup] device={device} dim={args.dim} layers={args.layers}")

    # Two independent graphs with identical compute.
    stream_a = torch.npu.Stream(device=device)
    stream_b = torch.npu.Stream(device=device)
    capture_a = torch.npu.Stream(device=device)
    capture_b = torch.npu.Stream(device=device)

    wa, ba = common.make_mlp_weights(args.dim, args.layers, device)
    wb, bb = common.make_mlp_weights(args.dim, args.layers, device)
    x = torch.randn(args.dim, args.dim, device=device)

    def make_capture(weights, biases):
        def fn():
            return common.make_compute_fn(weights, biases)(x)

        return fn

    graph_a, out_a = common.capture_graph(capture_a, make_capture(wa, ba))
    graph_b, out_b = common.capture_graph(capture_b, make_capture(wb, bb))
    common.sync()

    print("[setup] captured two private graphs")

    result = common.bench_dual_stream_overlap(
        graph_a, stream_a, graph_b, stream_b, args.iters, args.warmup
    )

    print("\n== Result ==")
    for k, v in result.items():
        print(f"{k}: {v:.3f}")

    print("\n== Interpretation ==")
    print(
        f"sequential          : {result['sequential_mean_ms']:.3f} ms "
        f"(~= devA {result['device_s1_mean_ms']:.3f} + devB {result['device_s2_mean_ms']:.3f})"
    )
    print(f"concurrent          : {result['concurrent_mean_ms']:.3f} ms")
    print(f"overlap_ratio       : {result['overlap_ratio']:.3f} (1.0 = full overlap)")
    if result["overlap_ratio"] > 0.7:
        print("=> streams overlap at the execution layer. Bottleneck is elsewhere.")
    elif result["overlap_ratio"] > 0.3:
        print("=> partial overlap. Core/bandwidth contention is limiting the win.")
    else:
        print("=> streams are effectively serialized. Dual-stream has no execution win here.")


if __name__ == "__main__":
    main()
