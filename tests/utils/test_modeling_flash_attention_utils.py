# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import unittest

import torch

from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids
from transformers.testing_utils import mockenv_context


class FlashAttentionUtilsTest(unittest.TestCase):
    def test_prepare_fa_kwargs_from_position_ids_reuses_cached_tensors(self):
        with mockenv_context("TRANSFORMERS_DISABLE_FA_POSITION_IDS_CACHE"):
            position_ids = torch.tensor([[0, 1, 2, 0, 1]])

            first = prepare_fa_kwargs_from_position_ids(position_ids)
            second = prepare_fa_kwargs_from_position_ids(position_ids)

        self.assertIs(first[0][0], second[0][0])
        self.assertIs(first[0][1], second[0][1])
        self.assertIs(first[1][0], second[1][0])
        torch.testing.assert_close(first[0][0], torch.tensor([0, 3, 5], dtype=torch.int32))
        self.assertEqual(first[1][0].item(), 3)

    def test_prepare_fa_kwargs_from_position_ids_invalidates_after_mutation(self):
        with mockenv_context("TRANSFORMERS_DISABLE_FA_POSITION_IDS_CACHE"):
            position_ids = torch.tensor([[0, 1, 2, 0, 1]])

            first = prepare_fa_kwargs_from_position_ids(position_ids)
            position_ids[0, 3] = 3
            second = prepare_fa_kwargs_from_position_ids(position_ids)

        self.assertIsNot(first[0][0], second[0][0])
        torch.testing.assert_close(second[0][0], torch.tensor([0, 5], dtype=torch.int32))
        self.assertEqual(second[1][0].item(), 5)

    def test_prepare_fa_kwargs_from_position_ids_cache_can_be_disabled(self):
        with mockenv_context(TRANSFORMERS_DISABLE_FA_POSITION_IDS_CACHE="1"):
            position_ids = torch.tensor([[0, 1, 2, 0, 1]])

            first = prepare_fa_kwargs_from_position_ids(position_ids)
            second = prepare_fa_kwargs_from_position_ids(position_ids)

        self.assertIsNot(first[0][0], second[0][0])
