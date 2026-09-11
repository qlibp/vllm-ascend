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
"""A model runner that executes prefill and decode on two npu-streams.

``PDDualStreamModelRunner`` subclasses ``NPUModelRunner`` and overrides only the
two entry points that matter for P/D separation -- :meth:`execute_model` and
:meth:`sample_tokens` -- while reusing the parent's input preparation, attention
metadata construction, model forward, logits computation and sampling.

The one structural change is that a single scheduler step is split into a
prefill group and a decode group (see
:func:`vllm_ascend.worker.pd_separation.split_prefill_decode`), each group's
inputs are staged into its own static buffers, and the two groups are then
replayed concurrently on two streams via
:class:`vllm_ascend.worker.pd_separation.PDDualStreamGraphManager`.

Scope: single card, qwen2-0.5b, no speculative decoding, no context parallelism,
no pipeline parallelism, no multimodal inputs, no LoRA.  Any of those are
rejected loudly at construction (see :meth:`_assert_pd_separation_supported`).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch

from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import reorder_batch_to_split_decodes_and_prefills
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendMetadata
from vllm_ascend.worker.model_runner_v1 import ExecuteModelState, NPUModelRunner
from vllm_ascend.worker.pd_separation import (
    PDDualStreamGraphManager,
    PDSeparationConfig,
    split_prefill_decode,
)

logger = init_logger(__name__)


class PDDualStreamModelRunner(NPUModelRunner):
    """NPUModelRunner variant that runs prefill/decode concurrently on two streams.

    The prefill-graph and decode-graph share the model weights and the kv-cache
    (same worker process, same ``self.kv_caches``).  They never trample each
    other's kv-cache writes because the scheduler hands the two groups disjoint
    blocks, and they never trample each other's scratch/static buffers because
    each graph has its own private pool and its own ``PDStreamContext`` buffers.
    """

    def __init__(self, vllm_config, device: torch.device):
        super().__init__(vllm_config, device)
        self.pd_config = PDSeparationConfig.from_env()
        if not self.pd_config.enabled:
            raise RuntimeError(
                "PDDualStreamModelRunner was constructed but "
                "VLLM_ASCEND_ENABLE_PD_SEPARATION is not enabled."
            )
        self._assert_pd_separation_supported()

        # decode_query_len == 1 + num_spec_tokens; with no spec decode this is 1.
        self.decode_query_len = self.decode_threshold

        # Set only after both graphs are captured (see capture_model).
        self._pd_graphs_captured = False
        # Static per-group buffer capacity.  Derive it here (once) from the
        # runner's batch limits when the env var is left at its -1 default, so
        # the graph manager can pre-allocate its static input buffers up front.
        self._pd_prefill_tokens = self.pd_config.max_prefill_tokens
        self._pd_decode_tokens = self.pd_config.max_decode_tokens
        if self._pd_prefill_tokens <= 0:
            self._pd_prefill_tokens = self.max_num_tokens
        if self._pd_decode_tokens <= 0:
            self._pd_decode_tokens = self.max_num_reqs

        self._pd_manager = PDDualStreamGraphManager(
            self.pd_config,
            self.device,
            prefill_tokens=self._pd_prefill_tokens,
            decode_tokens=self._pd_decode_tokens,
            attn_backend=self.attn_backend,
            vllm_config=self.vllm_config,
        )

    # ------------------------------------------------------------------ #
    # Guard rails
    # ------------------------------------------------------------------ #
    def _assert_pd_separation_supported(self) -> None:
        def fail(feature: str) -> None:
            raise RuntimeError(
                f"P/D separation does not support {feature}; disable it or set "
                "VLLM_ASCEND_ENABLE_PD_SEPARATION=0."
            )

        if self.vllm_config.speculative_config is not None:
            fail("speculative decoding")
        if self.parallel_config.pipeline_parallel_size > 1:
            fail("pipeline parallelism")
        if self.parallel_config.decode_context_parallel_size > 1:
            fail("decode context parallelism")
        if self.parallel_config.prefill_context_parallel_size > 1:
            fail("prefill context parallelism")
        if self.model_config.is_multimodal_model or self.model_config.is_encoder_decoder:
            fail("multimodal / encoder-decoder models")
        if self.lora_config is not None:
            fail("LoRA")
        if self.cache_config.kv_sharing_fast_prefill:
            fail("kv_sharing_fast_prefill")

    @property
    def _pd_enabled(self) -> bool:
        return self.pd_config.enabled and self._pd_graphs_captured

    # ------------------------------------------------------------------ #
    # Graph capture
    # ------------------------------------------------------------------ #
    def capture_model(self) -> int:
        """Capture the two static graphs (prefill + decode) on side streams.

        The parent's FULL graph capture is *not* used; instead we capture a
        single static prefill graph and a single static decode graph, each with
        its own pool (see PDDualStreamGraphManager.capture).  The shapes are
        fixed: decode graph = ``max_decode_tokens`` (one token per request),
        prefill graph = ``max_prefill_tokens``.
        """
        # Run one eager dummy forward to warm up lazy init / op caches before
        # capturing, mirroring _dummy_run but avoiding the parent's per-shape
        # FULL capture machinery.
        self._warm_up_for_pd_capture()

        def capture_prefill(input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            return self._forward_for_capture(
                num_tokens=self._pd_prefill_tokens,
                num_reqs=max(1, self.max_num_reqs),
                is_decode=False,
                input_ids=input_ids,
                positions=positions,
                static_slot_mapping=self._pd_manager.prefill_ctx.static_slot_mapping,
            )

        def capture_decode(input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            return self._forward_for_capture(
                num_tokens=self._pd_decode_tokens,
                num_reqs=max(1, self.max_num_reqs),
                is_decode=True,
                input_ids=input_ids,
                positions=positions,
                static_slot_mapping=self._pd_manager.decode_ctx.static_slot_mapping,
            )

        # ``profile_cudagraph_memory`` disables cudagraph capturing globally
        # before ``capture_model`` is called.  The parent runner re-enables it
        # around its own capture; do the same for the raw ``torch.npu.graph``
        # capture below, then restore the disabled state so any unexpected
        # capture after startup is still detected.
        set_cudagraph_capturing_enabled(True)
        try:
            self._pd_manager.capture(capture_prefill, capture_decode)
        finally:
            set_cudagraph_capturing_enabled(False)
        self._pd_graphs_captured = True
        # Pool memory is managed internally by the two private pools; report 0
        # extra bytes to the caller so it does not double-count.
        return 0

    def _warm_up_for_pd_capture(self) -> None:
        # Run an *eager* dummy forward to warm up lazy init / op caches before
        # the raw graph capture.  Passing NONE keeps the parent ``_dummy_run``
        # from entering its FULL/PIECEWISE capture machinery (which would call
        # ``validate_cudagraph_capturing_enabled`` and, after
        # ``profile_cudagraph_memory``, trip over the globally-disabled flag).
        self._dummy_run(self.max_num_reqs, cudagraph_runtime_mode=CUDAGraphMode.NONE)

        # ``AttentionMaskBuilder`` is a singleton and builds its splitfuse mask
        # lazily on first use via a CPU -> device copy
        # (``get_splitfuse_attn_mask``).  The dummy run above passes NONE, which
        # intentionally does *not* build attention metadata, so the mask is
        # still cold here.  If it were left cold, the first call would happen
        # inside ``torch.npu.graph`` on a non-default capture stream, and the
        # host->device copy + ``rtStreamSynchronize`` is illegal during graph
        # capture.  Pre-build it on the default stream so the capture path
        # reuses the cached device tensor.
        AttentionMaskBuilder(self.device).get_splitfuse_attn_mask()

    @staticmethod
    def _num_scheduled_tokens_for_capture(
        num_tokens: int, num_reqs: int
    ) -> np.ndarray:
        """Distribute ``num_tokens`` across ``num_reqs`` for capture metadata.

        Mirrors the non-uniform branch of ``_dummy_run``: each request gets
        ``num_tokens // num_reqs`` tokens and the last request absorbs the
        remainder, so the cumulative sum ends exactly at ``num_tokens``.
        """
        min_tokens_per_req = num_tokens // num_reqs
        remainder = num_tokens % num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += remainder
        return np.array(num_scheduled_tokens_list, dtype=np.int32)

    @torch.inference_mode()
    def _forward_for_capture(
        self,
        num_tokens: int,
        num_reqs: int,
        is_decode: bool,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        static_slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        """Run one forward with the *static* inputs for graph capture.

        ``input_ids`` / ``positions`` / ``static_slot_mapping`` are the graph
        manager's pre-allocated static tensors; their addresses get baked into
        the captured graph so that staging the current step's values into the
        same tensors (before replay) is what the graph actually reads.
        """
        max_query_len = 1 if is_decode else num_tokens

        # Populate the persistent query-length buffers *before* building the
        # attention metadata.  ``_build_attention_metadata`` derives
        # ``actual_seq_lengths_q`` from ``self.query_start_loc.cpu``; the FIA
        # TND kernel requires its last element to equal ``num_tokens`` (the
        # first dim of the query/hidden states).  The warm-up dummy run leaves
        # these buffers sized for its own (small) batch, so without this the
        # prefill capture would reuse e.g. ``actual_seq_lengths_q[-1] ==
        # max_num_reqs`` while ``num_tokens == max_prefill_tokens``, tripping
        # the FIA ``queryT == actualSequenceLengthQ[-1]`` check.
        #
        # These are pure CPU-side assignments (no device copies), so they are
        # not baked into the captured graph; they only fix the metadata the
        # capture-time forward sees.
        num_scheduled_tokens = self._num_scheduled_tokens_for_capture(
            num_tokens, num_reqs
        )
        cum_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np
        )
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
        # Mirrors ``_dummy_run``: during graph capture every request is given
        # the same (dummy) seq_len; the per-request values are re-bound before
        # each replay via the attention task-group update path.
        self.optimistic_seq_lens_cpu[:num_reqs] = max_query_len
        self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)

        attn_metadata, _ = self._build_attention_metadata(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_tokens_padded=num_tokens,
            num_reqs_padded=num_reqs,
            max_query_len=max_query_len,
            for_cudagraph_capture=True,
        )
        # Rebind the KV-write slot_mapping to this graph's *private* static
        # buffer.  reshape_and_cache is a plain captured op (not part of the
        # FIA task-group update path), so it reads whatever address was baked
        # here; pointing each graph at its own buffer and re-staging that
        # buffer before each replay keeps prefill/decode slots disjoint.
        for layer_name, meta in attn_metadata.items():
            if meta is not None:
                attn_metadata[layer_name] = dataclasses.replace(
                    meta, slot_mapping=static_slot_mapping
                )
        with set_ascend_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=None,
            # NONE, not FULL: ``self.model`` is wrapped in an ACLGraphWrapper,
            # which in FULL mode would capture its own sub-graph and conflict
            # with the outer raw ``torch.npu.graph`` capture.  NONE makes it
            # pass through eagerly so the outer capture records the forward.
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
            num_actual_tokens=num_tokens,
            model_instance=self.model,
            input_ids=input_ids,
        ):
            # ``set_ascend_forward_context`` resets ``capturing`` to False.
            # Flip it back to True while inside ``torch.npu.graph`` so the
            # attention ops record updatable task groups (graph_task_group_*
            # + attn_params/handles/events) instead of the eager path.
            _EXTRA_CTX.capturing = torch.npu.is_current_stream_capturing()
            hidden_states = self._model_forward(
                num_tokens, input_ids=input_ids, positions=positions
            )
        return hidden_states

    # ------------------------------------------------------------------ #
    # Forward / execute
    # ------------------------------------------------------------------ #
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors=None,
    ) -> ModelRunnerOutput | None:
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called after execute_model() returns None."
            )

        self._update_states(scheduler_output)

        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if not total_num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT

        prefill_req_ids, decode_req_ids = split_prefill_decode(
            scheduler_output.num_scheduled_tokens, self.decode_query_len
        )

        # The scheduler reorders ``self.running`` decode-first, but the runner's
        # persistent ``input_batch`` keeps its *own* ordering, so decode tokens
        # are not necessarily contiguous at the front of the per-token tensors.
        # Align the batch here so the ``input_ids[:num_decode_tokens]`` split in
        # ``_run_dual_stream`` separates the two groups correctly (decode tokens
        # front, prefill tokens tail).
        self._reorder_decode_first(scheduler_output)

        # Split the scheduled tokens into two contiguous groups.
        req_ids = self.input_batch.req_ids
        num_scheduled_tokens_np = np.array(
            [scheduler_output.num_scheduled_tokens[i] for i in req_ids],
            dtype=np.int32,
        )
        num_decode_tokens = int(
            num_scheduled_tokens_np[
                np.isin(req_ids, decode_req_ids)
            ].sum()
        )
        num_prefill_tokens = total_num_scheduled_tokens - num_decode_tokens

        logits_indices, spec_decode_metadata, _ = self._prepare_inputs(
            scheduler_output, num_scheduled_tokens_np
        )

        # Build the combined attention metadata once, then run the two groups.
        num_reqs = self.input_batch.num_reqs
        max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
        attn_metadata, spec_decode_common_attn_metadata = self._build_attention_metadata(
            num_tokens=total_num_scheduled_tokens,
            num_tokens_padded=total_num_scheduled_tokens,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs,
            max_query_len=max_num_scheduled_tokens,
            logits_indices=logits_indices,
            use_spec_decode=False,
            num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
        )

        input_ids, inputs_embeds, positions, intermediate_tensors, model_kwargs, _ = (
            self._preprocess(scheduler_output, total_num_scheduled_tokens, intermediate_tensors)
        )

        if self._pd_enabled:
            hidden_states = self._run_dual_stream(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
                attn_metadata=attn_metadata,
                num_decode_reqs=len(decode_req_ids),
                num_prefill_reqs=len(prefill_req_ids),
                num_decode_tokens=num_decode_tokens,
                num_prefill_tokens=num_prefill_tokens,
                total_num_scheduled_tokens=total_num_scheduled_tokens,
                model_kwargs=model_kwargs,
                intermediate_tensors=intermediate_tensors,
            )
        else:
            # Fallback (graphs not captured yet): run eagerly on the default
            # stream so the very first steps still produce correct output.
            with set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=total_num_scheduled_tokens,
                num_tokens_across_dp=None,
                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=BatchDescriptor(num_tokens=total_num_scheduled_tokens),
                num_actual_tokens=total_num_scheduled_tokens,
                model_instance=self.model,
                input_ids=input_ids,
            ):
                hidden_states = self._model_forward(
                    total_num_scheduled_tokens,
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

        sample_hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            None,  # aux_hidden_states
            attn_metadata,
            positions,
            None,  # ec_connector_output
            None,  # cudagraph_stats
            BatchDescriptor(num_tokens=total_num_scheduled_tokens),
        )
        return None

    def _reorder_decode_first(self, scheduler_output: "SchedulerOutput") -> None:
        """Reorder the persistent batch so decode requests precede prefill.

        ``SchedulerPDSeparation`` reorders ``self.running`` decode-first, but the
        runner's ``input_batch`` keeps its own persistent ordering, so decode
        tokens are *not* necessarily contiguous at the front of the per-token
        tensors.  This makes the ``input_ids[:num_decode_tokens]`` split in
        ``_run_dual_stream`` hold: with no chunked prefill, every request with
        ``num_scheduled_tokens <= decode_query_len`` is a decode request, and
        ``reorder_batch_to_split_decodes_and_prefills`` places exactly those at
        the front.

        Degenerate one-token prompts (``num_computed == 0``) are classified as
        prefill by the reorder and as decode by ``split_prefill_decode``; this
        is the documented out-of-scope case and may mis-split such a request.
        """
        reorder_batch_to_split_decodes_and_prefills(
            self.input_batch,
            scheduler_output,
            decode_threshold=self.decode_query_len,
        )

    def _run_dual_stream(
        self,
        input_ids,
        positions,
        inputs_embeds,
        attn_metadata,
        num_decode_reqs: int,
        num_prefill_reqs: int,
        num_decode_tokens: int,
        num_prefill_tokens: int,
        total_num_scheduled_tokens: int,
        model_kwargs: dict[str, Any],
        intermediate_tensors,
    ) -> torch.Tensor:
        """Stage each group's inputs and replay the two graphs concurrently.

        Returns the concatenated hidden states in the original (decode-first)
        token order so the caller can index into it with ``logits_indices``.
        """
        prefill_ctx = self._pd_manager.prefill_ctx
        decode_ctx = self._pd_manager.decode_ctx

        decode_input_ids = input_ids[:num_decode_tokens]
        decode_positions = positions[:num_decode_tokens]
        prefill_input_ids = input_ids[num_decode_tokens:]
        prefill_positions = positions[num_decode_tokens:]

        self._stage_static_inputs(decode_ctx, decode_input_ids, decode_positions)
        self._stage_static_inputs(prefill_ctx, prefill_input_ids, prefill_positions)

        # Stage each group's KV-write slot_mapping into its private static
        # buffer.  The combined slot_mapping is decode-first: decode slots live
        # in [0, num_decode_tokens) and prefill slots in
        # [num_decode_tokens, total).  The tail is filled with -1 (PAD_SLOT_ID)
        # so the padded tokens of each static graph are no-op KV writes.
        combined_slot_mapping = self._get_combined_slot_mapping(attn_metadata)
        self._stage_static_slot_mapping(
            decode_ctx, combined_slot_mapping[:num_decode_tokens]
        )
        self._stage_static_slot_mapping(
            prefill_ctx,
            combined_slot_mapping[num_decode_tokens : num_decode_tokens + num_prefill_tokens],
        )

        # Build static-shaped per-group attention metadata for the actual
        # requests (front-packed and zero-padded to each graph's capture shape)
        # and refresh each graph's attention task groups before replay.
        decode_attn_metadata = self._build_group_attention_metadata(
            attn_metadata,
            start_req=0,
            num_group_reqs=num_decode_reqs,
            num_group_tokens=num_decode_tokens,
            max_reqs=self.max_num_reqs,
            max_tokens=decode_ctx.max_tokens,
        )
        prefill_attn_metadata = self._build_group_attention_metadata(
            attn_metadata,
            start_req=num_decode_reqs,
            num_group_reqs=num_prefill_reqs,
            num_group_tokens=num_prefill_tokens,
            max_reqs=self.max_num_reqs,
            max_tokens=prefill_ctx.max_tokens,
        )

        logger.info(
            "[lqf] _run_dual_stream before manager.run "
            "num_decode_reqs=%s num_prefill_reqs=%s "
            "num_decode_tokens=%s num_prefill_tokens=%s",
            num_decode_reqs,
            num_prefill_reqs,
            num_decode_tokens,
            num_prefill_tokens,
        )
        self._pd_manager.run(prefill_attn_metadata, decode_attn_metadata)
        logger.info("[lqf] _run_dual_stream after manager.run")

        # Read back the two outputs and concatenate them in decode-first order.
        # The graphs replay to their full static shape, so slice each output
        # back to its actual token count before concatenating.
        decode_out = decode_ctx.output
        prefill_out = prefill_ctx.output
        assert decode_out is not None and prefill_out is not None
        decode_out = decode_out[:num_decode_tokens]
        prefill_out = prefill_out[:num_prefill_tokens]
        return torch.cat([decode_out, prefill_out], dim=0)

    def _build_group_attention_metadata(
        self,
        attn_metadata: dict[str, AscendMetadata],
        start_req: int,
        num_group_reqs: int,
        num_group_tokens: int,
        max_reqs: int,
        max_tokens: int,
    ) -> dict[str, AscendMetadata]:
        """Rebase/pad a slice of the combined metadata to a group's static shape.

        The combined metadata is already decode-first after
        :meth:`_reorder_decode_first`, so the decode group is ``[0, num_decode_reqs)``
        and the prefill group is ``[num_decode_reqs, num_decode_reqs + num_prefill_reqs)``.
        ``seq_lens`` / ``seq_lens_list`` / ``block_tables`` are per-request and can be
        sliced directly; ``actual_seq_lengths_q`` is a cumulative prefix-sum and is
        rebased so it starts at 0 for the group.
        """
        group_metadata: dict[str, AscendMetadata] = {}
        for layer_name, meta in attn_metadata.items():
            if meta is None or meta.seq_lens is None or meta.block_tables is None:
                continue

            seq_lens = meta.seq_lens[start_req : start_req + num_group_reqs]
            seq_lens_list = meta.seq_lens_list[start_req : start_req + num_group_reqs]
            block_tables = meta.block_tables[start_req : start_req + num_group_reqs]

            base = int(meta.actual_seq_lengths_q[start_req - 1]) if start_req > 0 else 0
            real_cumulative = [
                int(v) - base
                for v in meta.actual_seq_lengths_q[start_req : start_req + num_group_reqs]
            ]

            group_metadata[layer_name] = dataclasses.replace(
                meta,
                seq_lens=self._pad_1d_tensor(seq_lens, max_reqs),
                seq_lens_list=list(seq_lens_list) + [0] * (max_reqs - num_group_reqs),
                actual_seq_lengths_q=self._pad_cumulative_query_lens(
                    real_cumulative, num_group_tokens, max_reqs, max_tokens
                ),
                block_tables=self._pad_block_tables(block_tables, max_reqs),
            )
        return group_metadata

    @staticmethod
    def _pad_1d_tensor(tensor: torch.Tensor, max_reqs: int) -> torch.Tensor:
        padded = torch.zeros(max_reqs, dtype=tensor.dtype, device=tensor.device)
        padded[: tensor.shape[0]] = tensor
        return padded

    @staticmethod
    def _pad_block_tables(block_tables: torch.Tensor, max_reqs: int) -> torch.Tensor:
        num_group_reqs = block_tables.shape[0]
        max_blocks = block_tables.shape[1]
        padded = torch.zeros(
            (max_reqs, max_blocks), dtype=block_tables.dtype, device=block_tables.device
        )
        padded[:num_group_reqs] = block_tables
        return padded

    @staticmethod
    def _pad_cumulative_query_lens(
        real_cumulative: list[int],
        num_group_tokens: int,
        max_reqs: int,
        max_tokens: int,
    ) -> list[int]:
        """Pad a group's cumulative query lengths to the graph's static shape.

        The group's ``actual_seq_lengths_q`` is a prefix-sum ending at
        ``num_group_tokens``.  The captured graph replays with a fixed
        ``max_tokens``-row query, so the list must end at ``max_tokens``.  When a
        free request slot exists we absorb the residual ``max_tokens - num_group_tokens``
        into one zero-context padding request; when the group already fills every
        slot the residual is left unabsorbed (documented on-device validation case).
        """
        padded = list(real_cumulative)
        if len(padded) < max_reqs:
            padded.append(max_tokens)
        while len(padded) < max_reqs:
            padded.append(max_tokens)
        return padded

    @staticmethod
    def _stage_static_inputs(ctx, input_ids: torch.Tensor, positions: torch.Tensor) -> None:
        """Copy the current step's per-group inputs into the static buffers.

        The actual tokens are front-packed and the tail is zero-padded to the
        static shape; the padded tail is masked off by the (static) attention
        metadata and sliced away on readback.
        """
        num_tokens = input_ids.shape[0]
        # Zero the tail so replay never reads stale values from a previous
        # step's larger batch (positions in particular must not be negative).
        ctx.static_input_ids.zero_()
        ctx.static_positions.zero_()
        ctx.static_input_ids[:num_tokens].copy_(input_ids, non_blocking=True)
        ctx.static_positions[:num_tokens].copy_(positions, non_blocking=True)

    @staticmethod
    def _get_combined_slot_mapping(attn_metadata: dict[str, AscendMetadata]) -> torch.Tensor:
        """Return the combined (decode-first) slot_mapping tensor.

        All layers in the single kv-cache group share the same
        ``AscendMetadata`` instance (and therefore the same slot_mapping), so
        the first non-None entry is authoritative.
        """
        for meta in attn_metadata.values():
            if meta is not None and meta.slot_mapping is not None:
                return meta.slot_mapping
        raise RuntimeError(
            "PD dual-stream requires a slot_mapping tensor in attention metadata"
        )

    @staticmethod
    def _stage_static_slot_mapping(ctx, slot_mapping: torch.Tensor) -> None:
        """Copy a group's slot_mapping into its private static buffer.

        The actual slots are front-packed and the tail is filled with -1
        (PAD_SLOT_ID), so the padded tokens of the static graph write no KV.
        This runs on the default stream before replay; each replay stream waits
        on the default stream (see ``PDDualStreamGraphManager.run``), so the
        copy is visible to the replayed ``reshape_and_cache`` op.
        """
        num_tokens = slot_mapping.shape[0]
        ctx.static_slot_mapping.fill_(-1)
        ctx.static_slot_mapping[:num_tokens].copy_(slot_mapping, non_blocking=True)

    # ------------------------------------------------------------------ #
    # Sample
    # ------------------------------------------------------------------ #
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncGPUModelRunnerOutput:
        if self.execute_model_state is None:
            return EMPTY_MODEL_RUNNER_OUTPUT

        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            attn_metadata,
            positions,
            ec_connector_output,
            cudagraph_stats,
            batch_desc,
        ) = self.execute_model_state
        self.execute_model_state = None

        logger.info(
            "[lqf] sample_tokens before _sample total_tokens=%s",
            scheduler_output.total_num_scheduled_tokens,
        )
        sampler_output = self._sample(logits, spec_decode_metadata)
        logger.info("[lqf] sample_tokens after _sample")

        (
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            scheduler_output.total_num_scheduled_tokens,
            spec_decode_metadata,
        )

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            cudagraph_stats=cudagraph_stats,
        )

        logger.info("[lqf] sample_tokens after _bookkeeping_sync")

        if not self.use_async_scheduling:
            return model_runner_output

        # Mirror NPUModelRunner.sample_tokens: under async scheduling the
        # scheduler advances requests via num_output_placeholders and the actual
        # sampled token ids are copied to the host asynchronously.  Returning the
        # raw ModelRunnerOutput here (with empty sampled_token_ids produced by
        # _bookkeeping_sync's async branch) would make the scheduler emit no
        # EngineCoreOutputs and never finish the request.
        async_output = AsyncGPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            logprobs_tensors=sampler_output.logprobs_tensors,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
            vocab_size=self.input_batch.vocab_size,
        )
        logger.info("[lqf] sample_tokens created AsyncGPUModelRunnerOutput")
        self.input_batch.set_async_sampled_token_ids(
            async_output.sampled_token_ids_cpu,
            async_output.async_copy_ready_event,
        )
        logger.info("[lqf] sample_tokens returning async output")
        return async_output
