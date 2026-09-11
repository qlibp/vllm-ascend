#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""A single-card P/D (prefill/decode) separation scheduler.

This scheduler groups ``prefill`` requests and ``decode`` requests into two
disjoint batches within a single :class:`SchedulerOutput` so that the
``PDDualStreamModelRunner`` can execute the prefill-graph and the decode-graph
concurrently on two independent npu-streams.

Key scheduling semantics (deliberately different from the default vLLM v1
scheduler):

1. **Decode-first, grouped.**  ``self.running`` is re-ordered so that decode
   requests (``num_computed_tokens >= num_prompt_tokens``) come before prefill
   requests.  Each decode request is advanced by exactly ``decode_query_len``
   (``1 + num_spec_tokens``) tokens per step.

2. **No chunk-prefill.**  A prefill request is scheduled with its *entire*
   remaining prompt in a single step.  If the whole prompt does not fit in the
   remaining token budget, the request is not partially scheduled -- it simply
   waits for a later step where it fits.  This guarantees that a request never
   appears as a *partial* prefill inside the prefill batch, which is what keeps
   the two streams' kv-cache writes disjoint.

The two groups never overlap: a request is either in the decode group (its
prompt is fully computed) or in the prefill group (none of the current prompt
chunk is computed yet).  Since both groups are built from the same
``self.running`` list, the scheduler hands each request disjoint kv-cache
blocks via ``allocate_slots``.
"""

import time

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVEventBatch
from vllm.logger import logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager


class SchedulerPDSeparation(Scheduler):
    """Scheduler that separates prefill and decode requests into two groups.

    See the module docstring for the exact scheduling semantics.  This class is
    intentionally close to ``SchedulerDynamicBatch`` (decode-first FCFS), but
    removes chunk-prefill entirely and does not refine the token budget
    dynamically.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int | None = None,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            hash_block_size=hash_block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )
        self.running: list[Request] = []
        # Chunk-prefill is fundamentally incompatible with P/D separation:
        # a chunked prefill would leave a request half-prefilled and re-enter
        # the prefill group on a later step, which is fine for correctness but
        # defeats the "one prefill graph per prompt" grouping this feature
        # relies on.  Fail loudly if the flag was not disabled by platform.py.
        if self.scheduler_config.enable_chunked_prefill:
            logger.warning(
                "SchedulerPDSeparation requires enable_chunked_prefill=False; "
                "chunked prefill will be ignored by this scheduler."
            )

    def schedule(self) -> SchedulerOutput:
        # NOTE: This scheduling algorithm is a decode-first, no-chunk-prefill
        # variant of ``super().schedule()``:
        # 1. Running requests are split into a decode group (scheduled first,
        #    one token each) and a prefill group (scheduled second, whole
        #    remaining prompt, never chunked).
        # 2. Waiting requests are admitted with their entire prompt at once;
        #    if the prompt does not fit in the remaining budget they wait.
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens

        # Re-order running so the decode group precedes the prefill group.
        # The relative order inside each group is preserved (FCFS).
        #
        # This decode-first order is a *contract* with the runner
        # (``PDDualStreamModelRunner``): the runner re-derives the same
        # decode/prefill partition via ``split_prefill_decode`` and then reorders
        # its persistent ``input_batch`` decode-first before slicing the
        # per-token tensors into ``input_ids[:num_decode_tokens]``.  Note that
        # the runner's ``input_batch`` does *not* inherit this order
        # automatically (it keeps its own persistent ordering), so the runner
        # must do the reorder itself (``_reorder_decode_first``).
        decode_reqs = [
            req
            for req in self.running
            if req.num_computed_tokens >= req.num_prompt_tokens
        ]
        prefill_reqs = [
            req
            for req in self.running
            if req.num_computed_tokens < req.num_prompt_tokens
        ]
        self.running = decode_reqs + prefill_reqs

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        # First, schedule the RUNNING requests (decode group first, then any
        # still-incomplete prefill group).
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )

            if request.num_computed_tokens < request.num_prompt_tokens:
                # Prefill phase: schedule the whole remaining prompt or wait.
                # No chunking -- if it does not fit in the remaining budget,
                # stop here (subsequent prefill requests also wait).
                if num_new_tokens > token_budget:
                    break
            else:
                # Decode phase: one step of decode (1 + num_spec_tokens tokens).
                num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                )

            if num_new_tokens == 0:
                # Nothing to schedule for this request (budget/encoder
                # exhausted, or the request has reached its limit).
                req_index += 1
                break

            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                )
                if new_blocks is None:
                    # The request cannot be scheduled; preempt the lowest
                    # priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)
                    else:
                        preempted_req = self.running.pop()

                    self.kv_cache_manager.free(preempted_req)
                    self.encoder_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp
                        )

                    self.waiting.prepend_request(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        can_schedule = False
                        break
                else:
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            logger.info(
                "[lqf] schedule RUNNING request_id=%s phase=%s "
                "num_new_tokens=%s num_computed_tokens=%s num_tokens=%s "
                "num_prompt_tokens=%s num_output_tokens=%s "
                "num_output_placeholders=%s is_prefill_chunk=%s",
                request.request_id,
                "decode" if request.num_computed_tokens >= request.num_prompt_tokens else "prefill",
                num_new_tokens,
                request.num_computed_tokens,
                request.num_tokens,
                request.num_prompt_tokens,
                request.num_output_tokens,
                request.num_output_placeholders,
                request.is_prefill_chunk,
            )
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens + request.num_computed_tokens - request.num_tokens
                )
                if num_scheduled_spec_tokens > 0:
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids
                    )

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = encoder_inputs_to_schedule
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget

        # Record the LoRAs in scheduled_running_reqs.
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Next, schedule the WAITING requests (full prompt, never chunked).
        if not preempted_reqs:
            step_skipped_waiting = create_request_queue(self.policy)

            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request_queue = self._select_waiting_queue_for_scheduling()
                if request_queue is None:
                    break

                request = request_queue.peek_request()

                if self._is_blocked_waiting_status(
                    request.status
                ) and not self._try_promote_blocked_waiting_request(request):
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id,
                        )
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False

                if request.num_computed_tokens == 0:
                    (
                        new_computed_blocks,
                        num_new_local_computed_tokens,
                    ) = self.kv_cache_manager.get_computed_blocks(request)

                    if self.connector is not None:
                        (
                            num_external_computed_tokens,
                            load_kv_async,
                        ) = self.connector.get_num_new_matched_tokens(
                            request, num_new_local_computed_tokens
                        )

                        if num_external_computed_tokens is None:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue

                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                else:
                    new_computed_blocks = self.kv_cache_manager.create_empty_block_list()
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    # No chunk-prefill: schedule the entire remaining prompt,
                    # or leave the request waiting if it does not fit in the
                    # remaining budget this step.
                    if num_new_tokens > token_budget:
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue
                    assert num_new_tokens > 0

                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            _,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                        )
                        if num_new_tokens == 0:
                            break

                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                if self.is_encoder_decoder and request.has_encoder_inputs:
                    num_encoder_tokens = (
                        self.scheduler_config.max_num_encoder_input_tokens
                    )
                else:
                    num_encoder_tokens = 0

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_computed_tokens,
                    num_new_local_computed_tokens,
                    new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    break

                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        new_computed_blocks + new_blocks,
                        num_external_computed_tokens,
                    )

                request = request_queue.pop_request()
                if load_kv_async:
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
                    request.num_computed_tokens = num_computed_tokens
                    continue

                req_index += 1
                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id)
                )
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                logger.info(
                    "[lqf] schedule WAITING/PREEMPTED request_id=%s status=%s "
                    "num_new_tokens=%s num_computed_tokens=%s "
                    "num_prompt_tokens=%s num_tokens=%s "
                    "num_output_placeholders=%s",
                    request.request_id,
                    request.status,
                    num_new_tokens,
                    num_computed_tokens,
                    request.num_prompt_tokens,
                    request.num_tokens,
                    request.num_output_placeholders,
                )
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule
                    )
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget

            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

        # Check that the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        assert (
            len(scheduled_new_reqs)
            + len(scheduled_resumed_reqs)
            + len(scheduled_running_reqs)
            <= len(self.running)
        )

        # Get the longest common prefix among all requests in the running queue.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(
                any_request.request_id
            )

        new_reqs_data = [
            NewRequestData.from_request(req, req_to_new_blocks[req.request_id].get_block_ids())
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_blocks,
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
        )

        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        events = self.kv_cache_manager.take_events()
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        self._update_after_schedule(scheduler_output)
        return scheduler_output


class AsyncSchedulerPDSeparation(AsyncScheduler, SchedulerPDSeparation):
    """Async-scheduling variant of :class:`SchedulerPDSeparation`.

    ``SchedulerPDSeparation`` only reimplements ``schedule()``; the async
    scheduler relies on ``num_output_placeholders`` bookkeeping in
    ``AsyncScheduler._update_after_schedule`` / ``_update_request_with_output``
    to advance decode requests (the worker caches sampled tokens on-device and
    does not return them to the scheduler).  Without that bookkeeping the
    scheduler sees ``num_tokens_with_spec == num_computed_tokens`` after the
    first prefill and spins forever scheduling zero tokens.
    """
