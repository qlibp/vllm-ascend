#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Single-card prefill/decode (P/D) separation via two npu-streams.

This module implements the *mechanics* of running one prefill aclgraph and one
decode aclgraph concurrently on two independent npu-streams, following the
pattern validated by ``pytorch/npu_graph_parallel_stream_demo.py``.

Core invariants (do not break these):

1. **The two graphs never share a graph pool.**  ``torch.npu.graph`` is called
   *without* ``pool=`` for each graph so each one gets a private pool.  Sharing
   a pool across two concurrently-replayed graphs would let their scratch
   buffers trample each other.
2. **The two graphs never share static input/output buffers.**  Every buffer is
   owned by exactly one :class:`PDStreamContext`.  The scheduler guarantees the
   request sets are disjoint, so the same kv-cache slot is never written by
   both streams in the same step (kv-cache itself *is* shared, on purpose).
3. **Capture runs with ``aclgraph_runtime_mode=NONE``.**  The model is wrapped
   in an ``ACLGraphWrapper`` (``model_runner_v1.load_model``); in ``FULL`` mode
   that wrapper would capture its *own* sub-graph and conflict with our outer
   raw ``torch.npu.graph`` capture.  ``NONE`` makes it pass through eagerly so
   the outer capture records the eager forward.
4. **Event-level sync, no device barrier on the hot path.**  Before replay each
   stream ``wait_stream(default)`` so the host->device copies of the static
   buffers are visible; each stream records a fresh ``done_event`` immediately
   after replay; the default stream ``wait_event``s both done events before
   reading logits.  ``torch.npu.synchronize()`` is called *only* once, after
   capture.  A per-stream host ``stream.synchronize()`` is issued before each
   context's attention-param update (see invariant 6): it is a stream-level
   host wait, not a device-wide barrier, and is required for async scheduling.
5. **The single cross-stream dependency is the cross-step P->D handoff, and it
   is transitively covered by ``wait_stream(default)``.**  Step N's default
   stream waits on step N's prefill ``done_event`` before any logits/sampling
   work, so step N+1's decode stream ``wait_stream(default)`` ensures a request
   that just prefilled and now enters decode sees its kv-cache writes.  We do
   not wait directly on the previous step's event: re-recording an event while
   another stream may still be waiting on it is unsafe on NPU.  Within a step
   the two streams never wait on each other.
6. **Same-context update and replay are serialized across steps.**  The
   attention task-group params (``graph_params.events`` / handles) are
   single-buffered and re-recorded on each step's update.  With async
   scheduling the host may enqueue step N+1's update while step N's replay is
   still consuming those params; re-recording them while the graph waits on
   them is unsafe on NPU.  Before each ``_update_attention_metadata`` we
   therefore ``stream.synchronize()`` on that context's replay stream, so the
   previous replay (and the update work it waited on) has completed.
7. **``set_stream_limit`` is applied only when enabled** (non-negative cube/vec
   counts), once per stream after capture.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch_npu

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import init_logger

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.compilation.acl_graph import (
    reset_graph_params,
    set_graph_params,
    update_full_graph_params,
)

logger = init_logger(__name__)


def split_prefill_decode(
    num_scheduled_tokens: dict[str, int],
    decode_query_len: int,
) -> tuple[list[str], list[str]]:
    """Split a scheduler output into (prefill_req_ids, decode_req_ids).

    A request is considered a *decode* request iff it is advanced by exactly
    ``decode_query_len`` (``1 + num_spec_tokens``) tokens in this step; anything
    larger is a (non-chunked) prefill step.  This mirrors the invariant that the
    ``SchedulerPDSeparation`` scheduler enforces: prefill requests are always
    scheduled with their full remaining prompt, never a single token.

    NOTE: a request whose entire prompt is exactly ``decode_query_len`` tokens
    (a one-token prompt with no speculation) is indistinguishable from a decode
    step by this rule.  Such degenerate prompts are out of scope for P/D
    separation and are treated as decode.
    """
    prefill_req_ids: list[str] = []
    decode_req_ids: list[str] = []
    for req_id, num_tokens in num_scheduled_tokens.items():
        if num_tokens == decode_query_len:
            decode_req_ids.append(req_id)
        else:
            prefill_req_ids.append(req_id)
    return prefill_req_ids, decode_req_ids


@dataclass
class PDSeparationConfig:
    """Configuration for single-card P/D separation.

    All values are read from ``vllm_ascend.envs`` (``VLLM_ASCEND_*``).  A
    ``cube_num`` / ``vector_num`` of ``-1`` disables ``set_stream_limit`` for
    that stream (the runtime default is kept).
    """

    enabled: bool = False
    prefill_cube_num: int = 12
    prefill_vector_num: int = 24
    decode_cube_num: int = 12
    decode_vector_num: int = 24
    max_prefill_tokens: int = -1
    max_decode_tokens: int = -1

    @classmethod
    def from_env(cls) -> "PDSeparationConfig":
        from vllm_ascend import envs

        return cls(
            enabled=envs.VLLM_ASCEND_ENABLE_PD_SEPARATION,
            prefill_cube_num=envs.VLLM_ASCEND_PD_SEPARATION_PREFILL_CUBE_NUM,
            prefill_vector_num=envs.VLLM_ASCEND_PD_SEPARATION_PREFILL_VECTOR_NUM,
            decode_cube_num=envs.VLLM_ASCEND_PD_SEPARATION_DECODE_CUBE_NUM,
            decode_vector_num=envs.VLLM_ASCEND_PD_SEPARATION_DECODE_VECTOR_NUM,
            max_prefill_tokens=envs.VLLM_ASCEND_PD_SEPARATION_MAX_PREFILL_TOKENS,
            max_decode_tokens=envs.VLLM_ASCEND_PD_SEPARATION_MAX_DECODE_TOKENS,
        )

    def should_partition_stream(self, cube_num: int, vector_num: int) -> bool:
        """Return whether we should call ``set_stream_limit`` for a stream."""
        return cube_num >= 0 or vector_num >= 0


class PDStreamContext:
    """Owns one stream, one capture stream, one ``NPUGraph`` and its buffers.

    A ``PDStreamContext`` is strictly single-owner: the static buffers it holds
    are written only by the runner before replay, and only on this context's
    stream.  There is exactly one prefill context and one decode context; they
    never share buffers.
    """

    def __init__(
        self,
        name: str,
        device: torch.device,
        cube_num: int,
        vector_num: int,
        max_tokens: int,
    ) -> None:
        self.name = name
        self.device = device
        # Static batch capacity (in tokens) this graph is captured for.
        self.max_tokens = max_tokens
        # The stream that replays (and, during capture, runs) this graph.
        self.stream = torch.npu.Stream(device=device)
        # NPU graph capture must happen on a non-default stream; keep a
        # dedicated capture stream separate from the replay stream so that
        # capture never interferes with a stream the scheduler is using.
        self.capture_stream = torch.npu.Stream(device=device)
        # Side stream on which the attention task-group params are refreshed
        # (``graph_task_update_begin/end``).  The replay stream waits on it
        # before every replay so the updated params are visible.
        self.update_stream = torch.npu.Stream(device=device)
        self.cube_num = cube_num
        self.vector_num = vector_num

        # Done events: two event objects swapped after each step.  The current
        # step records ``_spare_done_event`` and the default stream waits on it
        # for readback; the other slot (``done_event``) holds the previous
        # step's recorded event until it is safe to reuse.  This avoids
        # re-recording an event while the default stream may still be waiting
        # on the previous record, which is unsafe on NPU.
        self.done_event = torch.npu.Event()
        self._spare_done_event = torch.npu.Event()

        # Static input tensors whose *addresses* are baked into the graph.  The
        # runner copies the current step's values into these (front-packed,
        # tail zero-padded) before replay.  input_ids is int32, positions int64.
        self.static_input_ids = torch.zeros(max_tokens, dtype=torch.int32, device=device)
        self.static_positions = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        # KV-write slot_mapping.  reshape_and_cache is a *plain* captured op
        # that reads the baked address on replay (it is not part of the FIA
        # task-group update path), so each stream needs its own private buffer
        # that the runner re-stages before every replay.  Keeping one buffer
        # per stream is what stops the prefill and decode graphs from reading
        # each other's slots.
        #
        # Must be int32 to match the native slot_mapping buffer
        # (``vllm_ascend.worker.block_table.BlockTable`` allocates it as
        # torch.int32), which is what ``_npu_reshape_and_cache`` /
        # ``ReshapeAndCacheOperation`` expects for its ``slots`` input.
        self.static_slot_mapping = torch.zeros(max_tokens, dtype=torch.int32, device=device)

        # Populated by ``PDDualStreamGraphManager.capture``.
        self.graph: torch.npu.NPUGraph | None = None
        self.output: torch.Tensor | None = None

    @property
    def is_captured(self) -> bool:
        return self.graph is not None

    def apply_stream_limit(self) -> None:
        """Partition AI cores for this stream (``aclrtSetStreamResLimit``)."""
        if self.cube_num < 0 and self.vector_num < 0:
            return
        logger.info(
            "Setting stream limit for %s stream: cube_num=%s vector_num=%s",
            self.name,
            self.cube_num,
            self.vector_num,
        )
        torch_npu.npu.set_stream_limit(
            self.stream,
            cube_num=self.cube_num,
            vector_num=self.vector_num,
        )


class PDDualStreamGraphManager:
    """Captures and concurrently replays the prefill-graph and decode-graph.

    ``capture`` runs each capture callback on that context's (non-default)
    capture stream inside ``torch.npu.graph(graph)`` -- deliberately *without*
    a shared ``pool`` so the two graphs get independent memory pools.

    ``run`` replays both graphs concurrently on their two streams, using
    event-level cross-stream synchronization (see module docstring invariants 4
    and 5): no device-level ``synchronize()`` is ever issued on the hot path.
    """

    def __init__(
        self,
        config: PDSeparationConfig,
        device: torch.device,
        prefill_tokens: int,
        decode_tokens: int,
        attn_backend: Any,
        vllm_config: VllmConfig,
    ) -> None:
        self.config = config
        self.device = device
        self.attn_backend = attn_backend
        self.vllm_config = vllm_config
        self.default_stream = torch.npu.default_stream(device)
        self.prefill_ctx = PDStreamContext(
            "prefill",
            device,
            config.prefill_cube_num,
            config.prefill_vector_num,
            max_tokens=prefill_tokens,
        )
        self.decode_ctx = PDStreamContext(
            "decode",
            device,
            config.decode_cube_num,
            config.decode_vector_num,
            max_tokens=decode_tokens,
        )

    @property
    def is_captured(self) -> bool:
        return self.prefill_ctx.is_captured and self.decode_ctx.is_captured

    def capture(
        self,
        prefill_capture_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        decode_capture_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        """Capture the two graphs on their dedicated side capture-streams.

        Each callback is handed its context's *static* input tensors
        (``static_input_ids``, ``static_positions``), whose addresses get baked
        into the graph, and must return the static output tensor (which the
        graph later overwrites on replay).
        """
        if self.is_captured:
            raise RuntimeError("PD dual-stream graphs have already been captured")

        # The attention ops record updatable task groups into the *global*
        # graph params (see AscendAttentionBackendImpl.full_graph_* and
        # update_full_graph_params).  Both static graphs must therefore be
        # registered before either capture callback runs, keyed by their static
        # token counts.
        reset_graph_params()
        set_graph_params(sorted({self.prefill_ctx.max_tokens, self.decode_ctx.max_tokens}))

        self.prefill_ctx.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.prefill_ctx.graph, stream=self.prefill_ctx.capture_stream):
            self.prefill_ctx.output = prefill_capture_fn(
                self.prefill_ctx.static_input_ids,
                self.prefill_ctx.static_positions,
            )
        logger.info("Captured prefill graph on %s stream.", self.prefill_ctx.name)

        self.decode_ctx.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.decode_ctx.graph, stream=self.decode_ctx.capture_stream):
            self.decode_ctx.output = decode_capture_fn(
                self.decode_ctx.static_input_ids,
                self.decode_ctx.static_positions,
            )
        logger.info("Captured decode graph on %s stream.", self.decode_ctx.name)

        self.prefill_ctx.apply_stream_limit()
        self.decode_ctx.apply_stream_limit()
        # One-time barrier after capture (not on the replay hot path).
        torch.npu.synchronize()

    def run(
        self,
        prefill_attn_metadata: Any,
        decode_attn_metadata: Any,
    ) -> None:
        """Refresh per-group attention params and concurrently replay both graphs.

        ``prefill_attn_metadata`` / ``decode_attn_metadata`` are the static-
        shaped per-layer attention metadata for the *actual* requests of this
        step (front-packed and zero-padded to the graphs' capture shapes).  They
        are refreshed into the two graphs' task groups on each context's
        ``update_stream`` immediately before replay.

        The host engine loop is synchronous (one ``run`` per step), so the two
        replay streams are the only concurrency: they interleave their kernels
        on the device while the host proceeds step-by-step.  Cross-step P->D
        handoff is the only place a replay stream waits on the *other* stream's
        work (and always on the *previous* step's prefill).
        """
        if not self.is_captured:
            raise RuntimeError("PD dual-stream graphs have not been captured")

        prefill_ctx = self.prefill_ctx
        decode_ctx = self.decode_ctx
        prefill_graph = prefill_ctx.graph
        decode_graph = decode_ctx.graph
        assert prefill_graph is not None and decode_graph is not None

        # The event objects recorded in this step.  The spare event is what this
        # step records and the default stream waits on; the other slot
        # (``done_event``) is the previous step's recorded event, which stays
        # untouched until it is safe to reuse.  This prevents us from re-recording
        # an event while another stream might still be waiting on it.
        prefill_done_event = prefill_ctx._spare_done_event
        decode_done_event = decode_ctx._spare_done_event

        logger.info(
            "[lqf] PDDualStreamGraphManager.run enter "
            "prefill_max_tokens=%s decode_max_tokens=%s "
            "prefill_attn_metadata=%s decode_attn_metadata=%s",
            prefill_ctx.max_tokens,
            decode_ctx.max_tokens,
            bool(prefill_attn_metadata),
            bool(decode_attn_metadata),
        )

        # Serialize this context's update with its previous replay.  The
        # attention task-group params (graph_params.events / handles) are
        # single-buffered and re-recorded below; under async scheduling the host
        # may reach this point while the previous step's replay on ``ctx.stream``
        # is still consuming the previous update.  ``stream.synchronize()`` is a
        # stream-level host wait (not a device-wide barrier) and ensures the
        # previous replay, and the update work it waited on, have both finished
        # before we re-record those params.  See module docstring invariant 6.
        prefill_ctx.stream.synchronize()
        # Refresh each graph's attention task-group params on its update stream.
        # This re-binds seq_lens / block_tables / query lengths for the actual
        # requests; without it the graphs would compute against the dummy
        # metadata captured at startup.
        self._update_attention_metadata(prefill_ctx, prefill_attn_metadata)
        logger.info("[lqf] PDDualStreamGraphManager.run prefill metadata updated")

        decode_ctx.stream.synchronize()
        self._update_attention_metadata(decode_ctx, decode_attn_metadata)
        logger.info("[lqf] PDDualStreamGraphManager.run decode metadata updated")

        # Concurrent replay.  Each stream first waits on the default stream so
        # the host->device copies of its static input buffers are visible, and
        # on its update stream so the refreshed attention params are visible,
        # then replays and immediately records its done event.  No device-level
        # synchronize() anywhere on this hot path.
        #
        # The cross-step P->D handoff is covered by
        # ``decode_ctx.stream.wait_stream(self.default_stream)`` below: step N's
        # default stream already waits on step N's prefill ``done_event`` before
        # any logits/sampling work, so a later decode stream that waits on the
        # default stream transitively waits for step N's prefill kv-cache writes.
        # We deliberately do NOT wait directly on the previous step's event; that
        # would require re-using an event object while another stream may still
        # be waiting on it, which is unsafe on NPU.
        with torch.npu.stream(prefill_ctx.stream):
            prefill_ctx.stream.wait_stream(self.default_stream)
            prefill_ctx.stream.wait_stream(prefill_ctx.update_stream)
            prefill_graph.replay()
            prefill_done_event.record()
        logger.info("[lqf] PDDualStreamGraphManager.run prefill replay + done_event enqueued")
        with torch.npu.stream(decode_ctx.stream):
            decode_ctx.stream.wait_stream(self.default_stream)
            decode_ctx.stream.wait_stream(decode_ctx.update_stream)
            decode_graph.replay()
            decode_done_event.record()
        logger.info("[lqf] PDDualStreamGraphManager.run decode replay + done_event enqueued")

        # Readback: the default stream reads both outputs for logits/sampling,
        # so it waits on each graph's done event (event-level, not a barrier).
        self.default_stream.wait_event(prefill_done_event)
        self.default_stream.wait_event(decode_done_event)
        logger.info("[lqf] PDDualStreamGraphManager.run default-stream wait events enqueued")

        # Swap the event objects so the event just recorded becomes the previous
        # event for the next step, and the older event becomes the spare to be
        # recorded next step.  With max_concurrent_batches == 2 this guarantees
        # the spare event has no remaining waiters when it is recorded again.
        prefill_ctx.done_event, prefill_ctx._spare_done_event = (
            prefill_ctx._spare_done_event,
            prefill_ctx.done_event,
        )
        decode_ctx.done_event, decode_ctx._spare_done_event = (
            decode_ctx._spare_done_event,
            decode_ctx.done_event,
        )

        logger.info("[lqf] PDDualStreamGraphManager.run exit")

    def _update_attention_metadata(self, ctx: "PDStreamContext", attn_metadata: Any) -> None:
        """Refresh one graph's attention task groups for the current step.

        ``update_full_graph_params`` re-issues the per-layer attention op with
        ``graph_task_update_begin/end`` on ``ctx.update_stream`` using the
        actual per-layer metadata, re-binding the seq_lens / block_tables /
        query-lengths that were baked into the graph at capture time.
        """
        if not attn_metadata:
            # A graph whose group is empty this step still needs no refresh:
            # its padded metadata keeps zero seq_lens / zero block tables and
            # the replay output is sliced away by the caller.
            return
        with set_ascend_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=ctx.max_tokens,
            num_tokens_across_dp=None,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=ctx.max_tokens),
            num_actual_tokens=ctx.max_tokens,
            model_instance=None,
        ):
            forward_context = get_forward_context()
            update_full_graph_params(
                self.attn_backend,
                ctx.update_stream,
                forward_context,
                ctx.max_tokens,
                self.vllm_config,
            )
