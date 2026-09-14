"""Test 03: host-side per-step enqueue overhead of the P/D run() structure.

The P/D path does more *host* work per step than the native full-graph path:

* two ``update_full_graph_params`` passes (one per graph, each re-issuing one
  eager attention op per layer on a dedicated update stream), and
* two ``graph.replay()`` calls plus two ``done_event.record()`` /
  ``wait_event`` bookkeeping calls.

For a decode-heavy step, the device decode time is tiny, so this host enqueue
time can dominate the step time.  This probe measures the amortized CPU enqueue
cost of:

* ``single``          -- one graph replay (native-like),
* ``dual``            -- two graph replays on two streams (PD without update),
* ``dual_update``     -- two (update + replay) pairs, where each update is
  ``--num-updates`` eager ops on a dedicated update stream (PD-like).

The ``dual_update`` case approximates the per-step host cost of the attention
metadata refresh path (``update_graph_params`` re-issues one FIA op per layer).

If ``dual_update`` enqueue time is a large fraction of the measured step time
from the model microbenchmark, the attention-metadata refresh host cost is a
primary bottleneck.
"""

from __future__ import annotations

import argparse
import time

import torch

import common


def host_enqueue_per_call(fn, n: int) -> float:
    """Average CPU time per call to enqueue work, excluding device completion."""
    common.sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t1 = time.perf_counter()
    common.sync()  # let device drain after measurement window
    return (t1 - t0) / n * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--num-updates", type=int, default=32, help="simulated attention layers per graph")
    parser.add_argument("--n-host", type=int, default=200, help="enqueue calls for the host-cost average")
    args = parser.parse_args()

    device = common.get_device(args.device)

    weights, biases = common.make_mlp_weights(args.dim, args.layers, device)
    x = torch.randn(args.dim, args.dim, device=device)
    compute_fn = common.make_compute_fn(weights, biases)

    # Two private graphs.
    cap_a = torch.npu.Stream(device=device)
    cap_b = torch.npu.Stream(device=device)
    replay_a = torch.npu.Stream(device=device)
    replay_b = torch.npu.Stream(device=device)
    update_a = torch.npu.Stream(device=device)
    update_b = torch.npu.Stream(device=device)
    default = torch.npu.default_stream(device)

    graph_a, out_a = common.capture_graph(cap_a, lambda: compute_fn(x))
    graph_b, out_b = common.capture_graph(cap_b, lambda: compute_fn(x))
    common.sync()

    # Small eager op to stand in for one attention-layer metadata update.
    small = torch.randn(64, 64, device=device)
    upd_buf = torch.empty_like(small)

    def single():
        with torch.npu.stream(replay_a):
            graph_a.replay()

    def dual():
        with torch.npu.stream(replay_a):
            graph_a.replay()
        with torch.npu.stream(replay_b):
            graph_b.replay()

    def update_then_replay(update_stream, replay_stream, graph):
        with torch.npu.stream(update_stream):
            for _ in range(args.num_updates):
                torch.add(small, 1.0, out=upd_buf)
        with torch.npu.stream(replay_stream):
            replay_stream.wait_stream(update_stream)
            graph.replay()

    def dual_update():
        update_then_replay(update_a, replay_a, graph_a)
        update_then_replay(update_b, replay_b, graph_b)

    single_ms = host_enqueue_per_call(single, args.n_host)
    dual_ms = host_enqueue_per_call(dual, args.n_host)
    dual_update_ms = host_enqueue_per_call(dual_update, args.n_host)

    print("\n== Host enqueue cost per step ==")
    print(f"single (1 replay)                    : {single_ms * 1e3:.1f} us")
    print(f"dual   (2 replays)                   : {dual_ms * 1e3:.1f} us")
    print(
        f"dual_update (2 x {args.num_updates} update ops + 2 replays): "
        f"{dual_update_ms * 1e3:.1f} us"
    )

    print("\n== Interpretation ==")
    print(
        "Compare dual_update against the per-step time reported by the model "
        "microbenchmark (pd step). If it is >10-20% of the step time, the "
        "attention-metadata refresh (graph_task_update) host cost is material."
    )


if __name__ == "__main__":
    main()
