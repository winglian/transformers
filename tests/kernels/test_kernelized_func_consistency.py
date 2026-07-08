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
"""Static consistency checks for hub kernel usage in modeling files. Kept separate from `test_kernels.py`
so it always runs, without requiring `kernels` or GPU."""

import ast
import unittest
from pathlib import Path

import transformers


def _decorator_call_name(decorator: ast.expr) -> str | None:
    if not isinstance(decorator, ast.Call):
        return None
    if isinstance(decorator.func, ast.Name):
        return decorator.func.id
    if isinstance(decorator.func, ast.Attribute):
        return decorator.func.attr
    return None


class KernelizedFuncConsistencyTest(unittest.TestCase):
    def test_kernelize_does_not_register_bare_function(self):
        """Functions registered through a class-level `@use_kernelized_func(...)` must be decorated with
        `@use_kernel_func_from_hub(...)` in the same modeling file, otherwise `kernelize()` raises
        `ValueError` on the bare function at runtime. Checked statically because the runtime decoration
        depends on the installed `kernels` version. Regression test for `MiMoV2FlashAttention`, which
        inherited Qwen2's decorator through modular conversion while calling a bare `apply_rotary_pos_emb`.
        """
        src_dir = Path(transformers.__file__).parent
        violations = []
        checked_classes = 0

        for path in sorted((src_dir / "models").rglob("modeling_*.py")):
            text = path.read_text(encoding="utf-8")
            if "use_kernelized_func" not in text:
                continue

            tree = ast.parse(text, filename=str(path))
            rel_path = path.relative_to(src_dir.parent)
            kernelizable_funcs = {
                node.name
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and any(_decorator_call_name(d) == "use_kernel_func_from_hub" for d in node.decorator_list)
            }

            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                for decorator in node.decorator_list:
                    if _decorator_call_name(decorator) != "use_kernelized_func":
                        continue
                    args = decorator.args
                    elts = list(args[0].elts) if args and isinstance(args[0], (ast.List, ast.Tuple)) else args
                    if decorator.keywords or not all(isinstance(elt, ast.Name) for elt in elts):
                        violations.append(
                            f"`{node.name}` in {rel_path}: cannot statically resolve the arguments of "
                            "`@use_kernelized_func`; pass bare function names, or update this test."
                        )
                        continue
                    if not elts:
                        continue
                    checked_classes += 1
                    for fn_name in (elt.id for elt in elts):
                        if fn_name not in kernelizable_funcs:
                            violations.append(
                                f"`{node.name}` in {rel_path} registers `{fn_name}` via `@use_kernelized_func`, "
                                f"but `{fn_name}` is not decorated with `@use_kernel_func_from_hub` in that file, "
                                "so `kernelize()` will raise ValueError on the bare function. If the decorator "
                                "was inherited through modular conversion, add `@no_inherit_decorator` to the "
                                "class in the model's `modular_*.py` (see `modular_mimo_v2_flash.py`)."
                            )

        self.assertEqual(violations, [], "\n\n".join(violations))
        # Guard against this test silently rotting into a no-op if the decorator patterns change.
        self.assertGreaterEqual(
            checked_classes, 20, f"Only matched {checked_classes} `@use_kernelized_func` classes; expected >= 20."
        )
