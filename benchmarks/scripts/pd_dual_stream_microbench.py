#!/usr/bin/env python3
"""Microbenchmark: native chunked-prefill vs P/D dual-stream compute time.

This benchmark bypasses the scheduler entirely.  It constructs fake
``SchedulerOutput`` objects and drives a worker's model runner directly through
``execute_model()`` / ``sample_tokens()``, so the measured time is pure compute
(prefill/decode forward + sampling + bookkeeping) without any scheduler or
engine-loop overhead.

Three scenarios are compared:

* baseline      -- native vLLM chunked-prefill.  One 200-token prompt is split
  into 128 + 72 token chunks and run *sequentially*, then 128 beams decode for
  2 steps.
* baseline-nc   -- native vLLM single-shot prefill.  The same 200-token prompt
  is prefilled in one step (no chunking), then 128 beams decode for 2 steps.
* pd            -- P/D dual-stream.  A pioneer prefill runs alone, then the
  next prefill and the pioneer's 128-beam decode run *concurrently* on two
  streams.

KV-cache data is intentionally NOT meaningful here.  The block ids are just
monotonic integers; only the compute amount (num prefill tokens, num decode
tokens, num requests) matches the target scenario.

Usage:
    # Recommended: run each scenario in its own process (clean device state).
    python benchmarks/scripts/pd_dual_stream_microbench.py --model <model> --mode baseline
    python benchmarks/scripts/pd_dual_stream_microbench.py --model <model> --mode baseline-nc
    python benchmarks/scripts/pd_dual_stream_microbench.py --model <model> --mode pd

    # Or run all back-to-back in one process (needs ~3x device memory).
    python benchmarks/scripts/pd_dual_stream_microbench.py --model <model> --mode both \
        --profile --profile-dir /tmp/pd_traces
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass
from math import ceil

import torch

from vllm.config import ProfilerConfig, set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)

from vllm_ascend.worker.worker import NPUWorker


def _cdiv(a: int, b: int) -> int:
    return ceil(a / b)


def get_runner_block_size(runner) -> int:
    """Return the KV-cache manager block size actually used by ``runner``.

    ``cache_config.block_size`` (what the CLI ``--block-size`` controls) is only
    the requested page size.  On Ascend the attention backend may select a
    different common block size (``select_common_block_size``), and
    ``kv_cache_spec.block_size`` ends up being the value the block table is
    actually sized with.  We must allocate fake block ids in units of that
    value, otherwise ``BlockTable.append_row`` overruns a row and raises
    ``ValueError: could not broadcast ... into shape (0,)``.
    """
    return runner.input_batch.block_table.block_tables[0].physical_block_size


def _check_max_model_len(runner, P: int, D: int) -> None:
    """Fail fast instead of hitting an obscure block-table broadcast error.

    A request may use at most ``max_num_blocks_per_req`` blocks, which is derived
    from ``max_model_len``.  The decode beams hold ``P`` prompt tokens plus up to
    ``D`` output tokens, so ``P + D`` must fit within the model's context length.
    """
    max_len = runner.max_model_len
    if P + D > max_len:
        raise ValueError(
            f"prefill_len + output_tokens ({P + D}) exceeds model max_model_len "
            f"({max_len}); reduce --prefill-len / --output-tokens or increase "
            f"--max-model-len"
        )


@dataclass
class BlockAlloc:
    """Monotonic block-id allocator.  Correctness of KV data is irrelevant;
    only the ids must be valid non-negative integers so slot_mapping stays in
    range."""

    base: int = 0
    next_id: int = 0

    def __post_init__(self) -> None:
        self.next_id = self.base

    def alloc(self, n: int) -> list[int]:
        ids = list(range(self.next_id, self.next_id + n))
        self.next_id += n
        return ids


def make_new_prefill(
    req_id: str,
    prompt_len: int,
    num_scheduled: int,
    num_computed: int,
    blocks: list[int],
    sampling_params: SamplingParams,
) -> NewRequestData:
    """Build a prefill NewRequestData (num_scheduled > 1)."""
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=[1] * prompt_len,
        mm_features=[],
        sampling_params=sampling_params,
        pooling_params=None,
        block_ids=(blocks,),
        num_computed_tokens=num_computed,
        lora_request=None,
    )


def make_new_decode(
    req_id: str,
    prompt_len: int,
    num_computed: int,
    blocks: list[int],
    sampling_params: SamplingParams,
) -> NewRequestData:
    """Build a decode NewRequestData.

    ``prompt_len`` is the prompt length P.  We append one dummy output token so
    that ``num_computed == P`` reads a valid token at index P (the first
    generated token).  This request is scheduled with ``num_scheduled_tokens=1``
    and is therefore classified as decode by ``split_prefill_decode``.
    """
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=[1] * (prompt_len + 1),
        mm_features=[],
        sampling_params=sampling_params,
        pooling_params=None,
        block_ids=(blocks,),
        num_computed_tokens=num_computed,
        lora_request=None,
    )


def make_cached(
    req_ids: list[str],
    num_computed_list: list[int],
    new_block_ids_list: list[tuple[list[int], ...] | None],
    num_output_tokens_list: list[int],
) -> CachedRequestData:
    return CachedRequestData(
        req_ids=list(req_ids),
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=list(new_block_ids_list),
        num_computed_tokens=list(num_computed_list),
        num_output_tokens=list(num_output_tokens_list),
    )


def make_scheduler_output(
    runner,
    new_reqs: list[NewRequestData],
    cached_reqs: CachedRequestData,
    num_scheduled_tokens: dict[str, int],
    finished: set[str] | None = None,
) -> SchedulerOutput:
    num_groups = len(runner.attn_groups)
    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached_reqs,
        num_scheduled_tokens=dict(num_scheduled_tokens),
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0] * num_groups,
        finished_req_ids=set(finished or ()),
        free_encoder_mm_hashes=[],
    )


def build_worker(
    model: str,
    *,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    enable_pd: bool,
    enable_chunked_prefill: bool,
    block_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    profiler_config: ProfilerConfig | None,
) -> NPUWorker:
    """Build a fully initialized worker following the vllm_ascend lifecycle."""
    # Single-process benchmark: ``NPUWorker`` initializes torch.distributed with
    # ``init_method="env://"``, which still requires MASTER_ADDR / MASTER_PORT /
    # RANK / WORLD_SIZE even when world_size == 1.  Provide sane defaults without
    # overriding values the caller may have already exported.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    # Must be set before config creation: PDSeparationConfig is read from env
    # inside PDDualStreamModelRunner.__init__ and SchedulerPDSeparation.
    os.environ["VLLM_ASCEND_ENABLE_PD_SEPARATION"] = "1" if enable_pd else "0"

    engine_args = EngineArgs(
        model=model,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        block_size=block_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        enable_prefix_caching=False,
        enable_chunked_prefill=enable_chunked_prefill,
    )
    if profiler_config is not None:
        engine_args.profiler_config = profiler_config

    vllm_config = engine_args.create_engine_config()

    # Normal vLLM goes through WorkerWrapperBase, which wraps init_device /
    # load_model / initialize_from_config in ``set_current_vllm_config(...)``.
    # We instantiate NPUWorker directly, so replicate that here; otherwise HCCL
    # pg-option setup calls get_current_vllm_config() and raises AssertionError.
    with set_current_vllm_config(vllm_config):
        worker = NPUWorker(
            vllm_config=vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        worker.init_device()
        worker.load_model()

        available_memory = worker.determine_available_memory()
        kv_cache_spec = worker.model_runner.get_kv_cache_spec()
        kv_cache_config = get_kv_cache_configs(
            vllm_config, [kv_cache_spec], [available_memory]
        )[0]
        worker.initialize_from_config(kv_cache_config)
        worker.compile_or_warm_up_model()

    return worker


def run_step(runner, scheduler_output: SchedulerOutput):
    """Run one step, handling the async execute_model/sample_tokens split.

    The baseline runner (NPUModelRunner) runs forward + sampling inside
    ``execute_model`` and returns a ModelRunnerOutput, leaving
    ``execute_model_state`` as None.  The PD runner returns None and stores the
    state, so ``sample_tokens`` must be called afterwards.
    """
    out = runner.execute_model(scheduler_output)
    if getattr(runner, "execute_model_state", None) is not None:
        out = runner.sample_tokens(None)
    return out


def timed_step(runner, scheduler_output: SchedulerOutput) -> float:
    torch.npu.synchronize()
    t0 = time.perf_counter()
    run_step(runner, scheduler_output)
    torch.npu.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e3  # ms


def reset_runner(runner, req_ids: set[str]) -> None:
    if not req_ids:
        return
    so = SchedulerOutput.make_empty()
    so.finished_req_ids = set(req_ids)
    runner.execute_model(so)


# --------------------------------------------------------------------------- #
# Scenario step builders
# --------------------------------------------------------------------------- #
def _append_decode_steps(
    runner,
    steps: list[SchedulerOutput],
    blk: BlockAlloc,
    P: int,
    B: int,
    D: int,
    sampling_params: SamplingParams,
    prefix_blocks: tuple[int, ...],
    finished_req_id: str,
) -> set[str]:
    """Append the decode steps shared by both baseline variants.

    The first decode step fans out the finished prefill request into ``B`` new
    beams; subsequent steps continue decoding those beams.  Returns the set of
    beam request ids that need to be reset between iterations.
    """
    block_size = get_runner_block_size(runner)
    nblocks = lambda t: _cdiv(t, block_size)

    # First decode step -> B new beams, the prefill request is finished.
    # The beams share ``prefix_blocks`` (nblocks(P) prompt blocks) and only add
    # the extra blocks needed for the first output token.
    first_output_blocks = nblocks(P + 1) - nblocks(P)
    new_reqs = []
    for i in range(B):
        blocks = list(prefix_blocks) + blk.alloc(first_output_blocks)
        new_reqs.append(make_new_decode(f"d{i}", P, P, blocks, sampling_params))
    num_sched = {f"d{i}": 1 for i in range(B)}
    steps.append(
        make_scheduler_output(
            runner,
            new_reqs,
            CachedRequestData.make_empty(),
            num_sched,
            finished={finished_req_id},
        )
    )

    # Remaining decode steps (D - 1 = 1 by default).  Allocate only the blocks
    # newly required when advancing from ``P + step`` to ``P + step + 1`` tokens.
    beam_ids = [f"d{i}" for i in range(B)]
    for step in range(1, D):
        step_blocks = nblocks(P + step + 1) - nblocks(P + step)
        steps.append(
            make_scheduler_output(
                runner,
                [],
                make_cached(
                    beam_ids,
                    [P + step] * B,
                    [(blk.alloc(step_blocks),) for _ in range(B)],
                    [step] * B,
                ),
                num_sched,
            )
        )

    return set(beam_ids)


def build_baseline_steps(
    runner,
    P: int,
    B: int,
    D: int,
    chunk: int,
    sampling_params: SamplingParams,
) -> tuple[list[SchedulerOutput], set[str]]:
    """Native chunked-prefill, prefill chunks then decode, all sequential."""
    _check_max_model_len(runner, P, D)
    block_size = get_runner_block_size(runner)
    blk = BlockAlloc()
    nblocks = lambda t: _cdiv(t, block_size)
    steps: list[SchedulerOutput] = []

    # Split the prompt into ceil(P / chunk) chunks.  Allocate only the blocks
    # newly needed for each chunk (delta of total-token block counts) so the
    # final request uses exactly nblocks(P) blocks instead of over-counting at
    # chunk boundaries.
    r0_blocks: list[int] = []
    computed = 0
    first = True
    while computed < P:
        chunk_len = min(chunk, P - computed)
        new_blocks = blk.alloc(nblocks(computed + chunk_len) - nblocks(computed))
        if first:
            steps.append(
                make_scheduler_output(
                    runner,
                    [make_new_prefill("r0", P, chunk_len, 0, new_blocks, sampling_params)],
                    CachedRequestData.make_empty(),
                    {"r0": chunk_len},
                )
            )
            first = False
        else:
            steps.append(
                make_scheduler_output(
                    runner,
                    [],
                    make_cached(
                        ["r0"],
                        [computed],
                        [(new_blocks,)],
                        [0],
                    ),
                    {"r0": chunk_len},
                )
            )
        r0_blocks += new_blocks
        computed += chunk_len

    beam_ids = _append_decode_steps(
        runner, steps, blk, P, B, D, sampling_params, tuple(r0_blocks), "r0"
    )
    return steps, beam_ids


def build_baseline_no_chunk_steps(
    runner,
    P: int,
    B: int,
    D: int,
    sampling_params: SamplingParams,
) -> tuple[list[SchedulerOutput], set[str]]:
    """Native single-shot prefill (no chunking), then decode, all sequential."""
    _check_max_model_len(runner, P, D)
    block_size = get_runner_block_size(runner)
    blk = BlockAlloc()
    nblocks = lambda t: _cdiv(t, block_size)
    steps: list[SchedulerOutput] = []

    # Single full prefill, scheduled as one step.
    r0_blocks = blk.alloc(nblocks(P))
    steps.append(
        make_scheduler_output(
            runner,
            [make_new_prefill("r0", P, P, 0, r0_blocks, sampling_params)],
            CachedRequestData.make_empty(),
            {"r0": P},
        )
    )

    beam_ids = _append_decode_steps(
        runner, steps, blk, P, B, D, sampling_params, tuple(r0_blocks), "r0"
    )
    return steps, beam_ids


def build_pd_steps(
    runner,
    P: int,
    B: int,
    D: int,
    sampling_params: SamplingParams,
) -> tuple[list[SchedulerOutput], set[str]]:
    """P/D dual-stream timeline.

    B0: pioneer prefill A alone.
    B1: A's 128-beam decode + prefill B in dual-stream.
    B2 (optional): A's beams decode + prefill C in dual-stream.
    """
    _check_max_model_len(runner, P, D)
    block_size = get_runner_block_size(runner)
    blk = BlockAlloc()
    nblocks = lambda t: _cdiv(t, block_size)
    steps: list[SchedulerOutput] = []

    # B0: pioneer prefill.
    A_blocks = blk.alloc(nblocks(P))
    steps.append(
        make_scheduler_output(
            runner,
            [make_new_prefill("A", P, P, 0, A_blocks, sampling_params)],
            CachedRequestData.make_empty(),
            {"A": P},
        )
    )

    beam_ids = [f"d{i}" for i in range(B)]
    prefix_blocks = tuple(A_blocks)

    # B1: dual-stream step -- 128 new decode beams + a new prefill B.
    # Beams share ``prefix_blocks`` (nblocks(P)) and add only the blocks needed
    # for their first output token.
    first_output_blocks = nblocks(P + 1) - nblocks(P)
    new_reqs = []
    for i in range(B):
        blocks = list(prefix_blocks) + blk.alloc(first_output_blocks)
        new_reqs.append(make_new_decode(f"d{i}", P, P, blocks, sampling_params))
    B_blocks = blk.alloc(nblocks(P))
    new_reqs.append(make_new_prefill("B", P, P, 0, B_blocks, sampling_params))

    num_sched = {f"d{i}": 1 for i in range(B)}
    num_sched["B"] = P
    steps.append(
        make_scheduler_output(
            runner,
            new_reqs,
            CachedRequestData.make_empty(),
            num_sched,
            finished={"A"},
        )
    )

    # Remaining decode steps, each overlapped with one more prefill.
    pending_prefill = "B"
    for step in range(1, D):
        next_prefill = f"P{step}"
        p_blocks = blk.alloc(nblocks(P))
        new_reqs = [make_new_prefill(next_prefill, P, P, 0, p_blocks, sampling_params)]
        num_sched = {f"d{i}": 1 for i in range(B)}
        num_sched[next_prefill] = P
        step_blocks = nblocks(P + step + 1) - nblocks(P + step)
        steps.append(
            make_scheduler_output(
                runner,
                new_reqs,
                make_cached(
                    beam_ids,
                    [P + step] * B,
                    [(blk.alloc(step_blocks),) for _ in range(B)],
                    [step] * B,
                ),
                num_sched,
                finished={pending_prefill},
            )
        )
        pending_prefill = next_prefill

    # At the end the beams and the last injected prefill are still resident.
    return steps, set(beam_ids) | {pending_prefill}


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def measure_scenario(
    runner,
    steps: list[SchedulerOutput],
    req_ids_to_reset: set[str],
    warmup: int,
    iters: int,
    profile: bool,
    worker: NPUWorker,
    profile_prefix: str,
) -> dict[str, float]:
    # Warmup.
    for _ in range(warmup):
        for so in steps:
            run_step(runner, so)
        reset_runner(runner, req_ids_to_reset)

    if profile:
        worker.profile(is_start=True, profile_prefix=profile_prefix)

    step_times: list[list[float]] = [[] for _ in steps]
    totals: list[float] = []
    for _ in range(iters):
        it_total = 0.0
        for idx, so in enumerate(steps):
            dt = timed_step(runner, so)
            step_times[idx].append(dt)
            it_total += dt
        totals.append(it_total)
        reset_runner(runner, req_ids_to_reset)

    if profile:
        worker.profile(is_start=False)

    result: dict[str, float] = {
        "total_mean_ms": statistics.mean(totals),
        "total_median_ms": statistics.median(totals),
    }
    for idx, times in enumerate(step_times):
        result[f"step{idx}_mean_ms"] = statistics.mean(times)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HuggingFace model id or path")
    parser.add_argument(
        "--mode",
        choices=["baseline", "baseline-nc", "pd", "both"],
        default="both",
        help=(
            "Which scenario to run. 'baseline', 'baseline-nc' and 'pd' run a "
            "single scenario in this process (recommended for clean memory/device "
            "state); 'both' runs all three scenarios back-to-back in one process "
            "for a direct comparison."
        ),
    )
    parser.add_argument("--prefill-len", type=int, default=200)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=128, help="baseline chunk size")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-dir", type=str, default="/tmp/pd_dual_stream_traces")
    args = parser.parse_args()

    P = args.prefill_len
    B = args.beam_width
    D = args.output_tokens
    assert D >= 1, "output-tokens must be >= 1"
    if args.mode in ("baseline", "both"):
        assert P > args.chunk_size, "prefill-len must be > chunk-size for chunked baseline"

    sampling_params = SamplingParams(n=1, temperature=0.0)

    profiler_config = None
    if args.profile:
        os.makedirs(args.profile_dir, exist_ok=True)
        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=os.path.abspath(args.profile_dir),
        )

    # ``max_num_batched_tokens`` is only the runner's buffer capacity.  The
    # baseline chunk size is controlled independently by --chunk-size in the
    # fake SchedulerOutput below.
    default_mnt = max(P, B, args.max_num_seqs)
    max_num_batched_tokens = (
        default_mnt if args.max_num_batched_tokens is None else args.max_num_batched_tokens
    )

    mode = args.mode

    if mode in ("baseline", "both"):
        print("== Building baseline worker (native chunked prefill) ==")
        baseline_worker = build_worker(
            args.model,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            enable_pd=False,
            enable_chunked_prefill=True,
            block_size=args.block_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            profiler_config=profiler_config,
        )
        baseline_runner = baseline_worker.model_runner

        baseline_steps, baseline_reset = build_baseline_steps(
            baseline_runner, P, B, D, args.chunk_size, sampling_params
        )

        print("\n== Baseline (chunked prefill) ==")
        baseline_result = measure_scenario(
            baseline_runner,
            baseline_steps,
            baseline_reset,
            args.warmup,
            args.iters,
            args.profile,
            baseline_worker,
            "baseline",
        )
        print(baseline_result)

    if mode in ("baseline-nc", "both"):
        print("== Building baseline-nc worker (single-shot prefill) ==")
        baseline_nc_worker = build_worker(
            args.model,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            enable_pd=False,
            enable_chunked_prefill=False,
            block_size=args.block_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            profiler_config=profiler_config,
        )
        baseline_nc_runner = baseline_nc_worker.model_runner

        baseline_nc_steps, baseline_nc_reset = build_baseline_no_chunk_steps(
            baseline_nc_runner, P, B, D, sampling_params
        )

        print("\n== Baseline no-chunk (single-shot prefill) ==")
        baseline_nc_result = measure_scenario(
            baseline_nc_runner,
            baseline_nc_steps,
            baseline_nc_reset,
            args.warmup,
            args.iters,
            args.profile,
            baseline_nc_worker,
            "baseline-nc",
        )
        print(baseline_nc_result)

    if mode in ("pd", "both"):
        print("== Building PD worker (P/D dual-stream) ==")
        pd_worker = build_worker(
            args.model,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            enable_pd=True,
            enable_chunked_prefill=False,
            block_size=args.block_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            profiler_config=profiler_config,
        )
        pd_runner = pd_worker.model_runner

        pd_steps, pd_reset = build_pd_steps(
            pd_runner, P, B, D, sampling_params
        )

        print("\n== PD dual-stream ==")
        pd_result = measure_scenario(
            pd_runner,
            pd_steps,
            pd_reset,
            args.warmup,
            args.iters,
            args.profile,
            pd_worker,
            "pd",
        )
        print(pd_result)

    if mode == "both":
        print("\n== Summary ==")

        # Baseline step layout: [prefill chunk0..k-1, decode1, decode2, ...]
        n_prefill_steps = _cdiv(P, args.chunk_size)
        baseline_prefill = sum(
            baseline_result[f"step{i}_mean_ms"] for i in range(n_prefill_steps)
        )
        baseline_decode1 = baseline_result[f"step{n_prefill_steps}_mean_ms"]
        baseline_prefill_plus_decode = baseline_prefill + baseline_decode1

        # Baseline-nc step layout: [prefill, decode1, decode2, ...]
        baseline_nc_prefill = baseline_nc_result["step0_mean_ms"]
        baseline_nc_decode1 = baseline_nc_result["step1_mean_ms"]
        baseline_nc_prefill_plus_decode = baseline_nc_prefill + baseline_nc_decode1

        # PD step layout: [pioneer prefill, dual-stream1, dual-stream2, ...]
        pd_prefill_only = pd_result["step0_mean_ms"]
        pd_dual_stream = pd_result["step1_mean_ms"]

        print(f"baseline    prefill(200, chunked)     : {baseline_prefill:.3f} ms")
        print(f"baseline    decode step (128 beams)   : {baseline_decode1:.3f} ms")
        print(f"baseline    prefill+decode (sequential): {baseline_prefill_plus_decode:.3f} ms")
        print(f"baseline-nc prefill(200, single-shot) : {baseline_nc_prefill:.3f} ms")
        print(f"baseline-nc decode step (128 beams)   : {baseline_nc_decode1:.3f} ms")
        print(
            f"baseline-nc prefill+decode (sequential): "
            f"{baseline_nc_prefill_plus_decode:.3f} ms"
        )
        print(f"pd          prefill-only (200)        : {pd_prefill_only:.3f} ms")
        print(f"pd          dual-stream step (P+D overlap): {pd_dual_stream:.3f} ms")
        print(
            f"speedup vs sequential (chunked)       : "
            f"{baseline_prefill_plus_decode / pd_dual_stream:.3f}x"
        )
        print(
            f"speedup vs sequential (no-chunk)      : "
            f"{baseline_nc_prefill_plus_decode / pd_dual_stream:.3f}x"
        )


if __name__ == "__main__":
    main()
