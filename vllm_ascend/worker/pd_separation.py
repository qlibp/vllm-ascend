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
3. **Stream ordering.**  Before replay, each stream ``wait_stream(default)`` so
   the host->device copies of the static buffers are visible; after replay the
   default stream ``wait_stream`` both streams so logits are read only after
   both graphs complete.
4. **``set_stream_limit`` is applied only when enabled** (non-negative cube/vec
   counts), once per stream after capture.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch_npu

from vllm.logger import init_logger

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
    ) -> None:
        self.name = name
        self.device = device
        # The stream that replays (and, during capture, runs) this graph.
        self.stream = torch.npu.Stream(device=device)
        # NPU graph capture must happen on a non-default stream; keep a
        # dedicated capture stream separate from the replay stream so that
        # capture never interferes with a stream the scheduler is using.
        self.capture_stream = torch.npu.Stream(device=device)
        self.cube_num = cube_num
        self.vector_num = vector_num

        # Populated by ``PDDualStreamGraphManager.capture``.
        self.graph: torch.npu.NPUGraph | None = None
        self.output: torch.Tensor | None = None
        # Static input tensors whose addresses are baked into the graph.  The
        # runner copies the current step's values into these before replay.
        self.static_inputs: list[torch.Tensor] = []

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

    ``run`` replays both graphs concurrently on their two streams and inserts
    the required cross-stream dependencies (see module docstring invariant 3).
    """

    def __init__(self, config: PDSeparationConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.default_stream = torch.npu.default_stream(device)
        self.prefill_ctx = PDStreamContext(
            "prefill", device, config.prefill_cube_num, config.prefill_vector_num
        )
        self.decode_ctx = PDStreamContext(
            "decode", device, config.decode_cube_num, config.decode_vector_num
        )

    @property
    def is_captured(self) -> bool:
        return self.prefill_ctx.is_captured and self.decode_ctx.is_captured

    def capture(
        self,
        prefill_capture_fn: Callable[[], torch.Tensor],
        decode_capture_fn: Callable[[], torch.Tensor],
    ) -> None:
        """Capture the two graphs on their dedicated side capture-streams.

        Each callback must run the model forward with the *static* inputs of
        its group already staged and the forward context already set, and must
        return the static output tensor (which the graph will later overwrite
        on replay).
        """
        if self.is_captured:
            raise RuntimeError("PD dual-stream graphs have already been captured")

        self.prefill_ctx.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.prefill_ctx.graph, stream=self.prefill_ctx.capture_stream):
            self.prefill_ctx.output = prefill_capture_fn()
        logger.info("Captured prefill graph on %s stream.", self.prefill_ctx.name)

        self.decode_ctx.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.decode_ctx.graph, stream=self.decode_ctx.capture_stream):
            self.decode_ctx.output = decode_capture_fn()
        logger.info("Captured decode graph on %s stream.", self.decode_ctx.name)

        self.prefill_ctx.apply_stream_limit()
        self.decode_ctx.apply_stream_limit()
        torch.npu.synchronize()

    def run(self) -> None:
        """Concurrently replay both graphs on their two streams."""
        if not self.is_captured:
            raise RuntimeError("PD dual-stream graphs have not been captured")

        prefill_graph = self.prefill_ctx.graph
        decode_graph = self.decode_ctx.graph
        assert prefill_graph is not None and decode_graph is not None

        # The static buffers were written on the default stream; make each
        # replay stream wait on those copies before launching.
        with torch.npu.stream(self.prefill_ctx.stream):
            self.prefill_ctx.stream.wait_stream(self.default_stream)
            prefill_graph.replay()
        with torch.npu.stream(self.decode_ctx.stream):
            self.decode_ctx.stream.wait_stream(self.default_stream)
            decode_graph.replay()

        # The default stream will read both outputs during logits/sampling, so
        # it must wait for both graphs to finish.
        self.default_stream.wait_stream(self.prefill_ctx.stream)
        self.default_stream.wait_stream(self.decode_ctx.stream)
