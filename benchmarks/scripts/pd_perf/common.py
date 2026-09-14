"""Shared helpers for the P/D dual-stream performance microbenchmarks.

These scripts are standalone torch_npu probes (they do not import vllm), so
they can be run directly on the remote NPU without the model / engine.

Run each ``testNN_*.py`` separately in its own process for the cleanest device
state:

    python benchmarks/scripts/pd_perf/test01_dual_stream_replay_overlap.py
    python benchmarks/scripts/pd_perf/test02_set_stream_limit_effect.py
    python benchmarks/scripts/pd_perf/test03_host_enqueue_overhead.py
    python benchmarks/scripts/pd_perf/test04_memory_bandwidth_contention.py
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import torch

try:
    import torch_npu  # noqa: F401  (registers the privateuse1 backend)
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("torch_npu is required to run these probes") from exc


def get_device(device_index: int = 0) -> torch.device:
    device = torch.device(f"npu:{device_index}")
    torch.npu.set_device(device)
    return device


def sync() -> None:
    torch.npu.synchronize()


def disable_internal_format() -> bool:
    """Disable internal format so ``set_stream_limit`` can take effect.

    ``torch_npu.npu.set_stream_limit`` documents that it must be paired with
    ``torch_npu.npu.config.allow_internal_format = False`` (otherwise
    core-control may be silently ignored / produce wrong results).
    """
    try:
        torch_npu.npu.config.allow_internal_format = False
        return True
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not set allow_internal_format=False: {exc}")
        return False


def set_stream_limit(stream: torch.npu.Stream, cube: int, vector: int) -> None:
    torch_npu.npu.set_stream_limit(stream, cube_num=cube, vector_num=vector)


def get_stream_limit(stream: torch.npu.Stream) -> dict:
    return torch_npu.npu.get_stream_limit(stream)


def reset_stream_limit(stream: torch.npu.Stream) -> None:
    torch_npu.npu.reset_stream_limit(stream)


def mstx_mark(message: str, stream: torch.npu.Stream | None = None) -> None:
    """Drop a device-side MSTX mark on a stream (profiler-trace annotation only).

    MSTX markers/range are *profiling* annotations consumed by the msprof trace;
    they do not return timestamps to Python.  For programmatic elapsed time use
    ``time_call_ms`` (which uses ``torch.npu.Event`` with timing enabled).
    """
    torch_npu.npu.mstx.mark(message, stream=stream)


def _fmt(stats: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.3f}" for k, v in stats.items())


def time_call_ms(
    fn: Callable[[], None],
    iters: int = 50,
    warmup: int = 10,
    stream: torch.npu.Stream | None = None,
) -> dict[str, float]:
    """Measure the device-side duration of ``fn()`` using timed events.

    Rather than a hard ``torch.npu.synchronize()`` around every call (which adds
    host round-trips and perturbs the device pipeline), this records a start/end
    event pair on ``stream`` around each call, then syncs **once** at the end and
    reads all elapsed times.  The returned value is the pure device duration of
    ``fn()``'s work, excluding host enqueue time.

    ``fn`` must enqueue its work on ``stream`` (the caller passes the matching
    stream).  If ``stream`` is None the current stream is used.
    """
    for _ in range(warmup):
        fn()
    sync()

    if stream is None:
        stream = torch.npu.current_stream()

    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record(stream)
        fn()
        ends[i].record(stream)
    sync()

    times: list[float] = [starts[i].elapsed_time(ends[i]) for i in range(iters)]

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "p90_ms": _percentile(times, 90),
    }


def _percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def replay_on(graph, stream: torch.npu.Stream) -> Callable[[], None]:
    """Return a zero-arg callable that replays ``graph`` on ``stream``."""

    def _replay() -> None:
        with torch.npu.stream(stream):
            graph.replay()

    return _replay


def make_mlp_weights(dim: int, layers: int, device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    weights = [torch.randn(dim, dim, device=device) for _ in range(layers)]
    biases = [torch.randn(dim, device=device) for _ in range(layers)]
    return weights, biases


def make_compute_fn(weights: list[torch.Tensor], biases: list[torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    """A compute-bound feed-forward chain (matmul + silu), pure graph-safe ops."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        for w, b in zip(weights, biases):
            x = torch.nn.functional.linear(x, w, b)
            x = torch.nn.functional.silu(x)
        return x

    return fn


def make_memory_fn(numel: int, copies: int, device: torch.device) -> Callable[[], torch.Tensor]:
    """A memory-bandwidth-bound workload: repeated large copies/adds.

    The tensors are created eagerly (before capture) and the returned callable
    only enqueues device work, so it is graph-capturable.
    """
    a = torch.randn(numel, device=device)
    b = torch.randn(numel, device=device)
    out = torch.empty_like(a)

    def fn() -> torch.Tensor:
        out.copy_(a)
        for _ in range(copies - 1):
            out.add_(b)
        return out

    return fn


def capture_graph(
    capture_stream: torch.npu.Stream,
    fn: Callable[[], torch.Tensor],
) -> tuple[torch.npu.NPUGraph, torch.Tensor]:
    """Capture ``fn`` into a fresh NPUGraph on ``capture_stream``.

    Returns ``(graph, output)``.  ``output`` must be kept alive so replay keeps
    writing to the same address.
    """
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=capture_stream):
        output = fn()
    return graph, output


def bench_dual_stream_overlap(
    g1: torch.npu.NPUGraph,
    s1: torch.npu.Stream,
    g2: torch.npu.NPUGraph,
    s2: torch.npu.Stream,
    iters: int = 50,
    warmup: int = 10,
) -> dict[str, float]:
    """Compare sequential (both on s1) vs concurrent (s1 + s2) replay.

    Also returns per-stream device durations measured with timed events and the
    observed overlap ratio.  The key signal is
    ``concurrent_mean_ms`` vs ``sequential_mean_ms``: if concurrent is close to
    ``max(device1, device2)`` the streams truly overlap; if it is close to
    ``device1 + device2`` the device serializes them.
    """
    for _ in range(warmup):
        with torch.npu.stream(s1):
            g1.replay()
        with torch.npu.stream(s2):
            g2.replay()
    sync()

    seq_times: list[float] = []
    con_times: list[float] = []
    dev1: list[float] = []
    dev2: list[float] = []

    e1_start = torch.npu.Event(enable_timing=True)
    e1_end = torch.npu.Event(enable_timing=True)
    e2_start = torch.npu.Event(enable_timing=True)
    e2_end = torch.npu.Event(enable_timing=True)

    for _ in range(iters):
        # Sequential: both replays queued on s1.
        sync()
        t0 = time.perf_counter()
        with torch.npu.stream(s1):
            g1.replay()
            g2.replay()
        sync()
        seq_times.append((time.perf_counter() - t0) * 1e3)

        # Concurrent: one replay per stream.
        sync()
        t0 = time.perf_counter()
        with torch.npu.stream(s1):
            e1_start.record()
            g1.replay()
            e1_end.record()
        with torch.npu.stream(s2):
            e2_start.record()
            g2.replay()
            e2_end.record()
        sync()
        con_times.append((time.perf_counter() - t0) * 1e3)
        dev1.append(e1_start.elapsed_time(e1_end))
        dev2.append(e2_start.elapsed_time(e2_end))

    seq_mean = statistics.mean(seq_times)
    con_mean = statistics.mean(con_times)
    d1 = statistics.mean(dev1)
    d2 = statistics.mean(dev2)
    # Ideal-overlap lower bound is max(d1, d2); serialized upper bound is d1 + d2.
    overlap = 0.0
    if d1 + d2 > 0:
        overlap = max(0.0, (d1 + d2 - con_mean) / min(d1, d2)) if min(d1, d2) > 0 else 0.0

    return {
        "sequential_mean_ms": seq_mean,
        "concurrent_mean_ms": con_mean,
        "device_s1_mean_ms": d1,
        "device_s2_mean_ms": d2,
        "overlap_ratio": overlap,
        "speedup_concurrent_vs_sequential": seq_mean / con_mean if con_mean > 0 else 0.0,
    }


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
