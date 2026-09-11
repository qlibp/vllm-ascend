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
from unittest.mock import MagicMock, patch

import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.base import TestBase
from vllm_ascend.core.scheduler_pd_separation import SchedulerPDSeparation

MODEL = "Qwen/Qwen2-0.5B"
BLOCK_SIZE = 16


def create_requests(num_requests, num_tokens=10, max_tokens=16):
    init_none_hash(sha256)
    sampling_params = SamplingParams(ignore_eos=False, max_tokens=max_tokens)
    requests = []
    for i in range(num_requests):
        request = Request(
            request_id=f"{i}",
            prompt_token_ids=[i] * num_tokens,
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
        )
        requests.append(request)
    return requests


def make_output(scheduler):
    req_ids = [req.request_id for req in scheduler.running]
    req_id_to_index = {req.request_id: i for i, req in enumerate(scheduler.running)}
    sampled_token_ids = [[1000]] * len(scheduler.running)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


class TestSchedulerPDSeparation(TestBase):
    @patch("vllm.config.ModelConfig.__post_init__", MagicMock())
    @patch("vllm.config.VllmConfig.__post_init__", MagicMock())
    def create_scheduler(self, max_num_batched_tokens=8192, max_num_seqs=16, max_model_len=8192):
        mock_hf_config = MagicMock()
        mock_hf_config.model_type = "qwen2"
        mock_hf_config.is_encoder_decoder = False
        mock_hf_config.architectures = ["Qwen2ForCausalLM"]
        model_config = ModelConfig(
            model=MODEL,
            tokenizer=MODEL,
            trust_remote_code=True,
            dtype="float16",
            seed=42,
            max_model_len=max_model_len,
        )
        model_config.hf_config = mock_hf_config
        model_config.hf_text_config = MagicMock()
        model_config.hf_text_config.is_encoder_decoder = False
        model_config.runner_type = "generate"

        scheduler_config = SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            long_prefill_token_threshold=0,
            disable_chunked_mm_input=False,
            enable_chunked_prefill=False,
            max_num_batched_tokens=max_num_batched_tokens,
            is_encoder_decoder=False,
        )
        scheduler_config.max_num_encoder_input_tokens = 10000
        scheduler_config.encoder_cache_size = 10000

        cache_config = CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
        )

        vllm_config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
        )
        from unittest.mock import PropertyMock

        type(model_config).is_encoder_decoder = PropertyMock(return_value=False)
        vllm_config.model_config.hf_config.is_encoder_decoder = False

        kv_cache_config = KVCacheConfig(
            num_blocks=10000,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
                    ),
                )
            ],
        )
        kv_cache_config.hash_block_size = BLOCK_SIZE
        cache_config.num_gpu_blocks = 10000

        scheduler = SchedulerPDSeparation(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            block_size=BLOCK_SIZE,
            log_stats=True,
            structured_output_manager=MagicMock(spec=StructuredOutputManager),
        )
        should_advance = MagicMock()
        should_advance.return_value = False
        scheduler.structured_output_manager.should_advance = should_advance
        return scheduler

    def test_init_rejects_chunked_prefill_warning(self):
        scheduler = self.create_scheduler()
        self.assertFalse(scheduler.scheduler_config.enable_chunked_prefill)

    def test_schedule_new_requests_as_prefill_no_chunk(self):
        """All waiting requests are scheduled with their full prompt at once."""
        scheduler = self.create_scheduler()
        requests = create_requests(num_requests=5, num_tokens=100)
        for req in requests:
            scheduler.add_request(req)

        output = scheduler.schedule()
        self.assertEqual(len(output.scheduled_new_reqs), 5)
        self.assertEqual(len(output.scheduled_cached_reqs.req_ids), 0)
        # Every request got its entire 100-token prompt in one step (no chunk).
        for req_id, num_tokens in output.num_scheduled_tokens.items():
            self.assertEqual(num_tokens, 100)
        self.assertEqual(output.total_num_scheduled_tokens, 500)
        self.assertEqual(len(scheduler.running), 5)

    def test_decode_group_after_prefill(self):
        """After prefill, running requests advance one token per step (decode)."""
        scheduler = self.create_scheduler()
        requests = create_requests(num_requests=3, num_tokens=10, max_tokens=8)
        for req in requests:
            scheduler.add_request(req)

        output1 = scheduler.schedule()
        self.assertEqual(output1.total_num_scheduled_tokens, 30)

        model_output = make_output(scheduler)
        scheduler.update_from_output(output1, model_output)

        output2 = scheduler.schedule()
        # Decode: one token per request, no prefill remaining.
        self.assertEqual(len(output2.scheduled_new_reqs), 0)
        self.assertEqual(len(output2.scheduled_cached_reqs.req_ids), 3)
        for num_tokens in output2.num_scheduled_tokens.values():
            self.assertEqual(num_tokens, 1)
        self.assertEqual(output2.total_num_scheduled_tokens, 3)

    def test_no_chunk_when_budget_insufficient(self):
        """A prefill that does not fit the remaining budget is never chunked."""
        scheduler = self.create_scheduler(max_num_batched_tokens=250)
        requests = create_requests(num_requests=3, num_tokens=100)
        for req in requests:
            scheduler.add_request(req)

        output = scheduler.schedule()
        # 100 + 100 fit in 250, the third (100) does not -> it waits.
        self.assertEqual(len(output.scheduled_new_reqs), 2)
        self.assertEqual(output.total_num_scheduled_tokens, 200)
        self.assertEqual(len(scheduler.waiting), 1)
        # The waiting request is still intact (no partial prefill).
        self.assertEqual(len(scheduler.running), 2)

    def test_prefill_and_decode_are_disjoint(self):
        """A single step schedules a request in exactly one group."""
        scheduler = self.create_scheduler(max_num_batched_tokens=1024)
        requests = create_requests(num_requests=4, num_tokens=10, max_tokens=8)
        for req in requests:
            scheduler.add_request(req)

        output1 = scheduler.schedule()
        # Prefill step: only prefill, no decode.
        self.assertEqual(len(output1.scheduled_new_reqs), 4)
        self.assertEqual(len(output1.scheduled_cached_reqs.req_ids), 0)

        scheduler.update_from_output(output1, make_output(scheduler))
        output2 = scheduler.schedule()
        # Decode step: only decode, no prefill.
        self.assertEqual(len(output2.scheduled_new_reqs), 0)
        self.assertEqual(len(output2.scheduled_cached_reqs.req_ids), 4)
