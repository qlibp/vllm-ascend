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

from typing import Any

import numpy as np
import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import reorder_batch_to_split_decodes_and_prefills
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
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
            )

        def capture_decode(input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            return self._forward_for_capture(
                num_tokens=self._pd_decode_tokens,
                num_reqs=max(1, self.max_num_reqs),
                is_decode=True,
                input_ids=input_ids,
                positions=positions,
            )

        self._pd_manager.capture(capture_prefill, capture_decode)
        self._pd_graphs_captured = True
        # Pool memory is managed internally by the two private pools; report 0
        # extra bytes to the caller so it does not double-count.
        return 0

    def _warm_up_for_pd_capture(self) -> None:
        self._dummy_run(self.max_num_reqs)

    def _forward_for_capture(
        self,
        num_tokens: int,
        num_reqs: int,
        is_decode: bool,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Run one forward with the *static* inputs for graph capture.

        ``input_ids`` / ``positions`` are the graph manager's pre-allocated
        static tensors; their addresses get baked into the captured graph so
        that staging the current step's values into the same tensors (before
        replay) is what the graph actually reads.
        """
        attn_metadata, _ = self._build_attention_metadata(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_tokens_padded=num_tokens,
            num_reqs_padded=num_reqs,
            max_query_len=1 if is_decode else num_tokens,
            for_cudagraph_capture=True,
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

        logits_indices, spec_decode_metadata, _, num_scheduled_tokens_compressed = (
            self._prepare_inputs(scheduler_output, num_scheduled_tokens_np)
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
            num_scheduled_tokens_compressed_list=num_scheduled_tokens_compressed,
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

        # TODO(remote-validate): the graphs were captured with *dummy* attention
        # metadata (block tables / slot mappings / sequence lengths).  Before
        # each replay those per-group attention params must be refreshed into
        # the graph workspaces for the *actual* requests, otherwise the graphs
        # compute against stale kv-cache addresses.  This is the one remaining
        # integration point that must be iterated on the NPU -- the static-input
        # staging and output slicing above are final.
        self._pd_manager.run()

        # Read back the two outputs and concatenate them in decode-first order.
        # The graphs replay to their full static shape, so slice each output
        # back to its actual token count before concatenating.
        decode_out = decode_ctx.output
        prefill_out = prefill_ctx.output
        assert decode_out is not None and prefill_out is not None
        decode_out = decode_out[:num_decode_tokens]
        prefill_out = prefill_out[:num_prefill_tokens]
        return torch.cat([decode_out, prefill_out], dim=0)

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

    # ------------------------------------------------------------------ #
    # Sample
    # ------------------------------------------------------------------ #
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput:
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

        sampler_output = self._sample(logits, spec_decode_metadata)

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

        return ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            cudagraph_stats=cudagraph_stats,
        )
