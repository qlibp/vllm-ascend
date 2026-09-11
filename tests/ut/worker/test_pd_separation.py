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
from unittest.mock import patch

from tests.ut.base import TestBase
from vllm_ascend.worker.pd_separation import (
    PDSeparationConfig,
    split_prefill_decode,
)


class TestSplitPrefillDecode(TestBase):
    def test_mixed_batch(self):
        num_scheduled_tokens = {"prefill-0": 100, "decode-0": 1, "decode-1": 1, "prefill-1": 7}
        prefill_reqs, decode_reqs = split_prefill_decode(num_scheduled_tokens, decode_query_len=1)
        self.assertCountEqual(prefill_reqs, ["prefill-0", "prefill-1"])
        self.assertCountEqual(decode_reqs, ["decode-0", "decode-1"])

    def test_spec_decode_query_len(self):
        # decode_query_len = 1 + num_spec_tokens = 2 with 1 speculative token.
        num_scheduled_tokens = {"prefill-0": 50, "decode-0": 2}
        prefill_reqs, decode_reqs = split_prefill_decode(num_scheduled_tokens, decode_query_len=2)
        self.assertEqual(prefill_reqs, ["prefill-0"])
        self.assertEqual(decode_reqs, ["decode-0"])

    def test_empty(self):
        prefill_reqs, decode_reqs = split_prefill_decode({}, decode_query_len=1)
        self.assertEqual(prefill_reqs, [])
        self.assertEqual(decode_reqs, [])

    def test_order_preserved(self):
        num_scheduled_tokens = {"d0": 1, "p0": 5, "d1": 1, "p1": 9}
        prefill_reqs, decode_reqs = split_prefill_decode(num_scheduled_tokens, decode_query_len=1)
        self.assertEqual(prefill_reqs, ["p0", "p1"])
        self.assertEqual(decode_reqs, ["d0", "d1"])


class TestPDSeparationConfig(TestBase):
    def _patch_envs(self, **overrides):
        values = {
            "VLLM_ASCEND_ENABLE_PD_SEPARATION": False,
            "VLLM_ASCEND_PD_SEPARATION_PREFILL_CUBE_NUM": 12,
            "VLLM_ASCEND_PD_SEPARATION_PREFILL_VECTOR_NUM": 24,
            "VLLM_ASCEND_PD_SEPARATION_DECODE_CUBE_NUM": 12,
            "VLLM_ASCEND_PD_SEPARATION_DECODE_VECTOR_NUM": 24,
            "VLLM_ASCEND_PD_SEPARATION_MAX_PREFILL_TOKENS": -1,
            "VLLM_ASCEND_PD_SEPARATION_MAX_DECODE_TOKENS": -1,
        }
        values.update(overrides)
        return patch.multiple("vllm_ascend.envs", **values)

    def test_defaults_disabled(self):
        with self._patch_envs():
            cfg = PDSeparationConfig.from_env()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.prefill_cube_num, 12)
        self.assertEqual(cfg.decode_vector_num, 24)

    def test_enabled_overrides(self):
        with self._patch_envs(
            VLLM_ASCEND_ENABLE_PD_SEPARATION=True,
            VLLM_ASCEND_PD_SEPARATION_PREFILL_CUBE_NUM=6,
            VLLM_ASCEND_PD_SEPARATION_DECODE_VECTOR_NUM=12,
            VLLM_ASCEND_PD_SEPARATION_MAX_PREFILL_TOKENS=4096,
        ):
            cfg = PDSeparationConfig.from_env()
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.prefill_cube_num, 6)
        self.assertEqual(cfg.decode_vector_num, 12)
        self.assertEqual(cfg.max_prefill_tokens, 4096)

    def test_should_partition_stream(self):
        cfg = PDSeparationConfig()
        self.assertTrue(cfg.should_partition_stream(12, 24))
        self.assertTrue(cfg.should_partition_stream(-1, 24))
        self.assertFalse(cfg.should_partition_stream(-1, -1))
