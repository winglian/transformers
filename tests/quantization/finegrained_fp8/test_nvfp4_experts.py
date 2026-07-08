# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Unit tests for the NVFP4 mixed-precision experts support in `FineGrainedFP8HfQuantizer`.

These tests intentionally avoid the GPU/network-gated `FP8QuantizerTest` class in
`test_fp8.py`: they only exercise loading/dequantize plumbing on synthetic tensors and a
tiny in-memory model, so they run on CPU without any Hub access.
"""

import unittest
from types import SimpleNamespace

import torch

from transformers import FineGrainedFP8Config, PreTrainedConfig, PreTrainedModel
from transformers.core_model_loading import Concatenate, MergeModulelist, WeightConverter, WeightRenaming
from transformers.integrations.finegrained_fp8 import Fp8Dequantize, FP8Experts
from transformers.modeling_utils import LoadStateDictConfig, convert_and_load_state_dict_in_model
from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer


def _pack_fp4(nibble_indices: torch.Tensor) -> torch.Tensor:
    """Pack a `(..., 2n)` tensor of nibble indices (each in `[0, 16)`) into an `(..., n)`
    `int8` tensor, two e2m1 nibbles per byte — the inverse of `Fp8Dequantize._unpack_fp4`."""
    low = nibble_indices[..., 0::2].to(torch.uint8)
    high = nibble_indices[..., 1::2].to(torch.uint8)
    return ((high << 4) | low).view(torch.int8)


class FineGrainedFP8ConfigNVFP4Test(unittest.TestCase):
    def test_moe_fields_default_to_none(self):
        config = FineGrainedFP8Config()
        self.assertIsNone(config.moe_quant_algo)
        self.assertIsNone(config.moe_group_size)

    def test_moe_quant_algo_nvfp4_defaults_group_size(self):
        config = FineGrainedFP8Config(moe_quant_algo="NVFP4")
        self.assertEqual(config.moe_quant_algo, "nvfp4")
        self.assertEqual(config.moe_group_size, 16)

    def test_moe_quant_algo_explicit_group_size(self):
        config = FineGrainedFP8Config(moe_quant_algo="nvfp4", moe_group_size=32)
        self.assertEqual(config.moe_group_size, 32)

    def test_group_size_kwarg_alias(self):
        # Raw checkpoints ship the field as `group_size`, not `moe_group_size`.
        config = FineGrainedFP8Config.from_dict({"quant_method": "fp8", "moe_quant_algo": "nvfp4", "group_size": 16})
        self.assertEqual(config.moe_quant_algo, "nvfp4")
        self.assertEqual(config.moe_group_size, 16)

    def test_moe_quant_algo_rejects_unsupported_value(self):
        with self.assertRaises(ValueError):
            FineGrainedFP8Config(moe_quant_algo="mxfp4")

    def test_moe_group_size_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            FineGrainedFP8Config(moe_quant_algo="nvfp4", moe_group_size=0)


class FP8ExpertsNVFP4AllocationTest(unittest.TestCase):
    E, HIDDEN, INTERMEDIATE, GROUP = 4, 32, 16, 16

    def _experts_config(self):
        return SimpleNamespace(
            hidden_size=self.HIDDEN,
            num_local_experts=self.E,
            moe_intermediate_size=self.INTERMEDIATE,
            hidden_act="silu",
        )

    def _build(self, has_gate=True, moe_quant_algo="nvfp4", moe_group_size=16):
        return FP8Experts(
            self._experts_config(),
            has_gate=has_gate,
            moe_quant_algo=moe_quant_algo,
            moe_group_size=moe_group_size,
        )

    def test_down_proj_allocation(self):
        experts = self._build()
        self.assertEqual(experts.down_proj.shape, (self.E, self.HIDDEN, self.INTERMEDIATE // 2))
        self.assertEqual(experts.down_proj.dtype, torch.int8)
        self.assertEqual(experts.down_proj_scale.shape, (self.E, self.HIDDEN, self.INTERMEDIATE // self.GROUP))
        self.assertEqual(experts.down_proj_scale.dtype, torch.float8_e4m3fn)
        self.assertEqual(experts.down_proj_scale_2.shape, (self.E,))
        self.assertEqual(experts.down_proj_scale_2.dtype, torch.float32)
        self.assertFalse(hasattr(experts, "down_proj_scale_inv"))

    def test_gate_up_proj_allocation(self):
        experts = self._build(has_gate=True)
        self.assertEqual(experts.gate_up_proj.shape, (self.E, 2 * self.INTERMEDIATE, self.HIDDEN // 2))
        self.assertEqual(experts.gate_up_proj.dtype, torch.int8)
        self.assertEqual(experts.gate_up_proj_scale.shape, (self.E, 2 * self.INTERMEDIATE, self.HIDDEN // self.GROUP))
        self.assertEqual(experts.gate_up_proj_scale.dtype, torch.float8_e4m3fn)
        # w1 (gate) and w3 (up) each carry an independent per-expert scalar.
        self.assertEqual(experts.gate_up_proj_scale_2.shape, (2 * self.E,))
        self.assertEqual(experts.gate_up_proj_scale_2.dtype, torch.float32)
        self.assertFalse(hasattr(experts, "gate_up_proj_scale_inv"))

    def test_up_proj_allocation_when_not_gated(self):
        experts = self._build(has_gate=False)
        self.assertEqual(experts.up_proj.shape, (self.E, self.INTERMEDIATE, self.HIDDEN // 2))
        self.assertEqual(experts.up_proj_scale.shape, (self.E, self.INTERMEDIATE, self.HIDDEN // self.GROUP))
        self.assertEqual(experts.up_proj_scale_2.shape, (self.E,))
        self.assertFalse(hasattr(experts, "gate_up_proj"))

    def test_legacy_fp4_path_is_unaffected(self):
        # `expert_dtype == "fp4"` (DeepSeek-V4 single-level format) must keep allocating
        # the old `*_scale_inv` blockwise buffer, never the new NVFP4 params.
        config = self._experts_config()
        config.expert_dtype = "fp4"
        experts = FP8Experts(config, has_gate=True)  # moe_quant_algo=None
        self.assertTrue(hasattr(experts, "down_proj_scale_inv"))
        self.assertFalse(hasattr(experts, "down_proj_scale"))
        self.assertFalse(hasattr(experts, "down_proj_scale_2"))

    def test_default_fp8_path_is_unaffected(self):
        experts = FP8Experts(self._experts_config(), has_gate=True, block_size=(16, 16))
        self.assertTrue(hasattr(experts, "down_proj_scale_inv"))
        self.assertFalse(hasattr(experts, "down_proj_scale_2"))
        self.assertEqual(experts.down_proj.dtype, torch.float8_e4m3fn)


class Fp8DequantizeNVFP4Test(unittest.TestCase):
    """Pure-tensor round-trips for the NVFP4 two-level dequant math."""

    def setUp(self):
        self.dequant = Fp8Dequantize(hf_quantizer=None)
        self.lut = torch.tensor(Fp8Dequantize._FP4_E2M1_LUT, dtype=torch.float32)

    def test_single_projection_two_level_round_trip(self):
        out_dim, in_dim, group = 4, 32, 16
        num_groups = in_dim // group
        torch.manual_seed(0)
        nibble_idx = torch.randint(0, 16, (out_dim, in_dim))
        packed = _pack_fp4(nibble_idx)
        scale = torch.rand(out_dim, num_groups).to(torch.float8_e4m3fn)
        scale_2 = torch.tensor(0.37)

        out = self.dequant._dequantize_one(packed, scale, output_dtype=torch.float32, scale_2=scale_2)

        expected = self.lut[nibble_idx] * scale.to(torch.float32).repeat_interleave(group, dim=1) * scale_2
        torch.testing.assert_close(out, expected)

    def test_scale_2_of_one_is_pure_group16_dequant(self):
        # Sanity: scale_2=1 must reduce exactly to the single-level (group-only) dequant.
        out_dim, in_dim, group = 2, 32, 16
        num_groups = in_dim // group
        torch.manual_seed(1)
        nibble_idx = torch.randint(0, 16, (out_dim, in_dim))
        packed = _pack_fp4(nibble_idx)
        scale = torch.rand(out_dim, num_groups).to(torch.float8_e4m3fn)

        without_scale_2 = self.dequant._dequantize_one(packed, scale, output_dtype=torch.float32)
        with_scale_2_one = self.dequant._dequantize_one(
            packed, scale, output_dtype=torch.float32, scale_2=torch.tensor(1.0)
        )
        torch.testing.assert_close(without_scale_2, with_scale_2_one)

    def test_gate_up_dequant_uses_independent_scale_2_per_half(self):
        """`convert()`'s generic chain path must dequantize `w1`/`w3` separately, each with
        its own `weight_scale_2` — never assuming the two halves share one scalar."""
        out_dim, in_dim, group, num_experts = 4, 32, 16, 2
        num_groups = in_dim // group
        torch.manual_seed(2)

        def make_half():
            idx = torch.randint(0, 16, (num_experts, out_dim, in_dim))
            packed = [_pack_fp4(idx[i]) for i in range(num_experts)]
            scales = [torch.rand(out_dim, num_groups).to(torch.float8_e4m3fn) for _ in range(num_experts)]
            return idx, packed, scales

        w1_idx, w1_packed, w1_scale = make_half()
        w3_idx, w3_packed, w3_scale = make_half()
        # Deliberately distinct scale_2 values between the two halves (and across experts).
        w1_scale_2 = [torch.tensor(0.5), torch.tensor(1.5)]
        w3_scale_2 = [torch.tensor(2.0), torch.tensor(3.0)]

        input_dict = {
            "experts.*.w1.weight$": w1_packed,
            "experts.*.w1.weight_scale$": w1_scale,
            "experts.*.w1.weight_scale_2$": w1_scale_2,
            "experts.*.w3.weight$": w3_packed,
            "experts.*.w3.weight_scale$": w3_scale,
            "experts.*.w3.weight_scale_2$": w3_scale_2,
        }
        out = self.dequant.convert(input_dict)

        # Scale (and scale_2) entries are consumed and dropped; only the weight keys remain.
        self.assertEqual(set(out.keys()), {"experts.*.w1.weight$", "experts.*.w3.weight$"})

        for i in range(num_experts):
            expected_w1 = (
                self.lut[w1_idx[i]] * w1_scale[i].to(torch.float32).repeat_interleave(group, dim=1) * w1_scale_2[i]
            )
            expected_w3 = (
                self.lut[w3_idx[i]] * w3_scale[i].to(torch.float32).repeat_interleave(group, dim=1) * w3_scale_2[i]
            )
            torch.testing.assert_close(out["experts.*.w1.weight$"][i].float(), expected_w1, rtol=5e-2, atol=5e-2)
            torch.testing.assert_close(out["experts.*.w3.weight$"][i].float(), expected_w3, rtol=5e-2, atol=5e-2)
            # If the code accidentally reused w1's scale_2 for w3 (or vice-versa) these would match.
            self.assertFalse(torch.allclose(out["experts.*.w1.weight$"][i].float(), expected_w3, atol=5e-2))

    def test_legacy_scale_inv_path_unaffected(self):
        """The classic single-level `weight_scale_inv` dequant path (DeepSeek-V4 FP4 / plain
        blockwise FP8) must keep working exactly as before — no `weight_scale`/`weight_scale_2`
        siblings involved at all."""
        out_dim, in_dim, block = 4, 8, 4
        torch.manual_seed(3)
        weight = torch.randn(out_dim, in_dim).to(torch.float8_e4m3fn)
        scale = torch.rand(out_dim // block, in_dim // block)

        input_dict = {"layer.weight$": [weight], "layer.weight_scale_inv$": [scale]}
        out = self.dequant.convert(input_dict)
        self.assertEqual(set(out.keys()), {"layer.weight$"})
        expected = self.dequant._dequantize_one(weight, scale, output_dtype=torch.float32)
        torch.testing.assert_close(out["layer.weight$"][0].to(torch.float32), expected)


class _ExpertsOnlyModel(PreTrainedModel):
    base_model_prefix = ""

    def __init__(self, config, experts_config, has_gate, moe_quant_algo, moe_group_size):
        super().__init__(config)
        self.experts = FP8Experts(
            experts_config,
            activation_scheme="dynamic",
            has_gate=has_gate,
            moe_quant_algo=moe_quant_algo,
            moe_group_size=moe_group_size,
        )
        self.post_init()


class NVFP4ExpertWeightConversionTest(unittest.TestCase):
    """End-to-end (but GPU/network-free) check that the NVFP4 expert weight converters
    anchor `.weight$`, route `weight_scale` / `weight_scale_2` to the right targets, and
    load a synthetic state dict with no MISSING/UNEXPECTED keys — in both dequantize modes.
    """

    E, HIDDEN, INTERMEDIATE, GROUP = 2, 32, 16, 16

    def _experts_config(self):
        return SimpleNamespace(
            hidden_size=self.HIDDEN,
            num_local_experts=self.E,
            moe_intermediate_size=self.INTERMEDIATE,
            hidden_act="silu",
        )

    def _original_conversions(self):
        return [
            WeightConverter(
                source_patterns=["experts.*.w1.weight", "experts.*.w3.weight"],
                target_patterns="experts.gate_up_proj",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            ),
            WeightConverter(
                source_patterns="experts.*.w2.weight",
                target_patterns="experts.down_proj",
                operations=[MergeModulelist(dim=0)],
            ),
        ]

    def _synthetic_state_dict(self):
        torch.manual_seed(42)
        state_dict = {}
        for e in range(self.E):
            for name, out_dim, in_dim in [
                ("w1", self.INTERMEDIATE, self.HIDDEN),
                ("w3", self.INTERMEDIATE, self.HIDDEN),
                ("w2", self.HIDDEN, self.INTERMEDIATE),
            ]:
                prefix = f"experts.{e}.{name}"
                state_dict[f"{prefix}.weight"] = _pack_fp4(torch.randint(0, 16, (out_dim, in_dim)))
                state_dict[f"{prefix}.weight_scale"] = torch.rand(out_dim, in_dim // self.GROUP).to(
                    torch.float8_e4m3fn
                )
                state_dict[f"{prefix}.weight_scale_2"] = torch.rand(())
                # Dynamic activation scheme: ships a per-tensor input_scale we never consume.
                state_dict[f"{prefix}.input_scale"] = torch.rand(())
        return state_dict

    def test_keep_quantized_no_missing_or_unexpected(self):
        quant_config = FineGrainedFP8Config(moe_quant_algo="nvfp4", moe_group_size=self.GROUP)
        quantizer = FineGrainedFP8HfQuantizer(quant_config)
        self.assertTrue(quantizer.pre_quantized)
        self.assertFalse(quant_config.dequantize)

        model = _ExpertsOnlyModel(
            PreTrainedConfig(),
            self._experts_config(),
            has_gate=True,
            moe_quant_algo="nvfp4",
            moe_group_size=self.GROUP,
        )
        weight_mapping = quantizer.update_weight_conversions(self._original_conversions())

        # The anchored `.weight$` converters must no longer swallow the scale siblings: check
        # that a `weight_scale`/`weight_scale_2` twin converter with the right target exists.
        targets = {c._original_target_patterns[0] for c in weight_mapping if isinstance(c, WeightConverter)}
        self.assertIn("experts.gate_up_proj", targets)
        self.assertIn("experts.gate_up_proj_scale", targets)
        self.assertIn("experts.gate_up_proj_scale_2", targets)
        self.assertIn("experts.down_proj_scale", targets)
        self.assertIn("experts.down_proj_scale_2", targets)

        state_dict = self._synthetic_state_dict()
        load_config = LoadStateDictConfig(weight_mapping=weight_mapping, hf_quantizer=quantizer)
        loading_info, _ = convert_and_load_state_dict_in_model(model, state_dict, load_config, tp_plan=None)

        self.assertEqual(loading_info.missing_keys, set())
        self.assertEqual(loading_info.mismatched_keys, set())
        self.assertEqual(loading_info.conversion_errors, {})
        # `input_scale` is intentionally not mapped anywhere (dynamic activation scheme) —
        # it's the only expected "unexpected" key.
        self.assertEqual(
            loading_info.unexpected_keys,
            {f"experts.{e}.{name}.input_scale" for e in range(self.E) for name in ("w1", "w2", "w3")},
        )

        model_state = model.state_dict()
        self.assertEqual(model_state["experts.gate_up_proj"].dtype, torch.int8)
        self.assertEqual(model_state["experts.gate_up_proj_scale"].dtype, torch.float8_e4m3fn)
        self.assertEqual(model_state["experts.gate_up_proj_scale_2"].dtype, torch.float32)
        self.assertEqual(model_state["experts.gate_up_proj_scale_2"].shape, (2 * self.E,))
        self.assertEqual(model_state["experts.down_proj_scale_2"].shape, (self.E,))

        # gate_up_proj_scale_2 must retain the two independently-scaled halves, in
        # [w1 experts..., w3 experts...] order.
        expected_scale_2 = torch.cat(
            [
                torch.stack([state_dict[f"experts.{e}.w1.weight_scale_2"] for e in range(self.E)]),
                torch.stack([state_dict[f"experts.{e}.w3.weight_scale_2"] for e in range(self.E)]),
            ]
        )
        torch.testing.assert_close(model_state["experts.gate_up_proj_scale_2"], expected_scale_2)

    def test_dequantize_true_no_missing_or_unexpected(self):
        quant_config = FineGrainedFP8Config(moe_quant_algo="nvfp4", moe_group_size=self.GROUP, dequantize=True)
        quantizer = FineGrainedFP8HfQuantizer(quant_config)
        self.assertTrue(quantizer.pre_quantized)
        self.assertTrue(quant_config.dequantize)

        class _BF16ExpertsModel(PreTrainedModel):
            base_model_prefix = ""

            def __init__(self, config, num_experts, hidden, intermediate):
                super().__init__(config)
                self.experts = torch.nn.Module()
                self.experts.gate_up_proj = torch.nn.Parameter(
                    torch.zeros(num_experts, 2 * intermediate, hidden, dtype=torch.bfloat16)
                )
                self.experts.down_proj = torch.nn.Parameter(
                    torch.zeros(num_experts, hidden, intermediate, dtype=torch.bfloat16)
                )
                self.post_init()

        model = _BF16ExpertsModel(PreTrainedConfig(), self.E, self.HIDDEN, self.INTERMEDIATE)
        weight_mapping = quantizer.update_weight_conversions(self._original_conversions())
        state_dict = self._synthetic_state_dict()

        load_config = LoadStateDictConfig(weight_mapping=weight_mapping, hf_quantizer=quantizer)
        loading_info, _ = convert_and_load_state_dict_in_model(model, state_dict, load_config, tp_plan=None)

        self.assertEqual(loading_info.missing_keys, set())
        self.assertEqual(loading_info.mismatched_keys, set())
        self.assertEqual(loading_info.conversion_errors, {})
        self.assertEqual(
            loading_info.unexpected_keys,
            {f"experts.{e}.{name}.input_scale" for e in range(self.E) for name in ("w1", "w2", "w3")},
        )

        model_state = model.state_dict()
        self.assertEqual(model_state["experts.gate_up_proj"].dtype, torch.bfloat16)
        self.assertEqual(model_state["experts.gate_up_proj"].shape, (self.E, 2 * self.INTERMEDIATE, self.HIDDEN))
        self.assertEqual(model_state["experts.down_proj"].shape, (self.E, self.HIDDEN, self.INTERMEDIATE))

        # Spot-check expert 0's down_proj half against the manual two-level dequant formula.
        dequant = Fp8Dequantize(quantizer)
        expected_down_0 = dequant._dequantize_one(
            state_dict["experts.0.w2.weight"],
            state_dict["experts.0.w2.weight_scale"],
            output_dtype=torch.bfloat16,
            scale_2=state_dict["experts.0.w2.weight_scale_2"],
        )
        torch.testing.assert_close(model_state["experts.down_proj"][0], expected_down_0)


class UpdateWeightConversionsNVFP4Test(unittest.TestCase):
    """Focused checks on `FineGrainedFP8HfQuantizer.update_weight_conversions` itself."""

    def _quantizer(self, dequantize=False):
        config = FineGrainedFP8Config(moe_quant_algo="nvfp4", moe_group_size=16, dequantize=dequantize)
        return FineGrainedFP8HfQuantizer(config)

    def _original_conversions(self):
        return [
            WeightConverter(
                source_patterns=["experts.*.w1.weight", "experts.*.w3.weight"],
                target_patterns="experts.gate_up_proj",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            ),
            WeightConverter(
                source_patterns="experts.*.w2.weight",
                target_patterns="experts.down_proj",
                operations=[MergeModulelist(dim=0)],
            ),
        ]

    def test_moe_quant_algo_none_is_byte_for_byte_unchanged(self):
        config = FineGrainedFP8Config()  # moe_quant_algo=None
        quantizer = FineGrainedFP8HfQuantizer(config)
        original = self._original_conversions()
        updated = quantizer.update_weight_conversions(list(original))

        # Same shape as before this change: scale_rename + the 2 original converters (no
        # extra scale twins are ever added when moe_quant_algo is unset).
        converters = [c for c in updated if isinstance(c, WeightConverter)]
        self.assertEqual(len(converters), 2)
        for conv, orig in zip(converters, original):
            self.assertEqual(conv.source_patterns, orig.source_patterns)
            self.assertEqual(conv._original_target_patterns, orig._original_target_patterns)

    def test_keep_quantized_anchors_and_adds_twins(self):
        quantizer = self._quantizer(dequantize=False)
        updated = quantizer.update_weight_conversions(self._original_conversions())
        converters = [c for c in updated if isinstance(c, WeightConverter)]

        by_target = {c._original_target_patterns[0]: c for c in converters}
        self.assertEqual(
            set(by_target),
            {
                "experts.gate_up_proj",
                "experts.gate_up_proj_scale",
                "experts.gate_up_proj_scale_2",
                "experts.down_proj",
                "experts.down_proj_scale",
                "experts.down_proj_scale_2",
            },
        )

        gate_up = by_target["experts.gate_up_proj"]
        self.assertEqual(gate_up.source_patterns, ["experts.*.w1.weight$", "experts.*.w3.weight$"])

        gate_up_scale = by_target["experts.gate_up_proj_scale"]
        self.assertEqual(gate_up_scale.source_patterns, ["experts.*.w1.weight_scale$", "experts.*.w3.weight_scale$"])

        gate_up_scale_2 = by_target["experts.gate_up_proj_scale_2"]
        self.assertEqual(
            gate_up_scale_2.source_patterns, ["experts.*.w1.weight_scale_2$", "experts.*.w3.weight_scale_2$"]
        )
        # 0-D-per-expert scalars must fuse via stack + a dim-0 concat, never the weight's
        # dim-1 concat (which only makes sense on the 3-D (E, out, in) weight/weight_scale).
        self.assertIsInstance(gate_up_scale_2.operations[0], MergeModulelist)
        self.assertEqual(gate_up_scale_2.operations[0].dim, 0)
        self.assertIsInstance(gate_up_scale_2.operations[1], Concatenate)
        self.assertEqual(gate_up_scale_2.operations[1].dim, 0)

        down_scale_2 = by_target["experts.down_proj_scale_2"]
        self.assertEqual(down_scale_2.source_patterns, ["experts.*.w2.weight_scale_2$"])
        # Single source → no Concatenate needed at all.
        self.assertEqual(len(down_scale_2.operations), 1)
        self.assertIsInstance(down_scale_2.operations[0], MergeModulelist)

    def test_dequantize_true_feeds_scale_and_scale_2_into_same_converter(self):
        quantizer = self._quantizer(dequantize=True)
        quantizer.pre_quantized = True
        updated = quantizer.update_weight_conversions(self._original_conversions())
        converters = [c for c in updated if isinstance(c, WeightConverter)]

        gate_up = next(c for c in converters if c._original_target_patterns == ["experts.gate_up_proj"])
        self.assertEqual(
            gate_up.source_patterns,
            [
                "experts.*.w1.weight$",
                "experts.*.w3.weight$",
                "experts.*.w1.weight_scale$",
                "experts.*.w3.weight_scale$",
                "experts.*.w1.weight_scale_2$",
                "experts.*.w3.weight_scale_2$",
            ],
        )
        self.assertIsInstance(gate_up.operations[0], Fp8Dequantize)

    def test_non_expert_converter_is_untouched_by_nvfp4_handling(self):
        # A converter whose target has nothing to do with `.experts.` (e.g. a QKV fuse) must
        # not get scale_inv/scale twins injected at all.
        quantizer = self._quantizer(dequantize=False)
        qkv_converter = WeightConverter(
            source_patterns="self_attn.qkv_proj.weight",
            target_patterns=["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"],
            operations=[MergeModulelist(dim=0)],
        )
        updated = quantizer.update_weight_conversions([qkv_converter])
        converters = [c for c in updated if isinstance(c, WeightConverter)]
        self.assertEqual(len(converters), 1)
        self.assertEqual(converters[0].source_patterns, ["self_attn.qkv_proj.weight"])

    def test_scale_rename_is_prepended_and_harmless_for_nvfp4(self):
        quantizer = self._quantizer(dequantize=False)
        updated = quantizer.update_weight_conversions(self._original_conversions())
        renamings = [c for c in updated if isinstance(c, WeightRenaming)]
        self.assertEqual(len(renamings), 1)
        renamed, _ = renamings[0].rename_source_key("experts.0.w1.weight_scale")
        # `^(.+)\.scale$` must not fire on `...weight_scale` (no literal `.scale` suffix).
        self.assertEqual(renamed, "experts.0.w1.weight_scale")


if __name__ == "__main__":
    unittest.main()
