# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Stride signatures of the paged KV views seen by the vLLM varlen entry."""

import importlib.util
import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "paged_layout", ROOT / "flash_attn" / "paged_layout.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

sys.path.insert(0, str(ROOT))
try:
    from flash_attn.flash_attn_interface import _paged_layout
except ImportError:  # the compiled extension may not be built
    _paged_layout = None


def _pool(layout, page, heads, dim=16, pages=4):
    if layout == "bshd":
        return torch.empty(pages, page, heads, dim)
    return torch.empty(pages, heads, page, dim).permute(0, 2, 1, 3)


class PagedKvLayoutTest(unittest.TestCase):
    def test_layouts_by_stride(self):
        cases = (
            ("bshd", 64, 8, "bshd"),
            ("bshd", 64, 64, "bshd"),
            ("bhsd", 64, 8, "bhsd"),
            ("bhsd", 64, 64, "bhsd"),
        )
        for layout, page, heads, expected in cases:
            with self.subTest(layout=layout, page=page, heads=heads):
                k = _pool(layout, page, heads)
                self.assertEqual(module.paged_kv_layout(k, k), expected)

    def test_matching_value_layouts(self):
        for layout in ("bshd", "bhsd"):
            with self.subTest(layout=layout):
                k = _pool(layout, 64, 8)
                v = _pool(layout, 64, 8)
                self.assertEqual(module.paged_kv_layout(k, v), layout)

    def test_value_layout_must_agree_with_key(self):
        for k_layout, v_layout in (("bshd", "bhsd"), ("bhsd", "bshd")):
            with self.subTest(k=k_layout, v=v_layout):
                k = _pool(k_layout, 64, 8)
                v = _pool(v_layout, 64, 8)
                self.assertEqual(module.paged_kv_layout(k, v), "custom")

    def test_non_paged_keeps_the_default(self):
        flat = torch.empty(8, 16)
        self.assertEqual(module.paged_kv_layout(flat, flat), "custom")
        self.assertEqual(module.paged_kv_layout(flat, None), "custom")


@unittest.skipIf(_paged_layout is None, "flash_attn_2_cuda is not built")
class PagedLayoutGateTest(unittest.TestCase):
    """The varlen entry only uses the layout for paged decode/verify shapes."""

    def _args(self, layout="bshd", **overrides):
        args = dict(
            k=_pool(layout, 64, 8),
            v=_pool(layout, 64, 8),
            q=torch.empty(4, 8, 16),
            max_seqlen_q=2,
            cu_seqlens_q=torch.tensor([0, 2, 4], dtype=torch.int32),
            block_table=torch.zeros(4, 1, dtype=torch.int32),
            window_size=(-1, -1),
            softcap=0.0,
        )
        args.update(overrides)
        return args

    def test_decode_shapes_pass_the_gate(self):
        for layout in ("bshd", "bhsd"):
            with self.subTest(layout=layout):
                self.assertEqual(_paged_layout(**self._args(layout)), layout)

    def test_gate_falls_back_to_custom(self):
        cases = {
            "softcap": {"softcap": 1.0},
            "window": {"window_size": (4, 4)},
            "no_block_table": {"block_table": None},
            "no_max_seqlen_q": {"max_seqlen_q": None},
            "q_head_dim": {"q": torch.empty(4, 8, 8)},
            "q_token_count": {"q": torch.empty(6, 8, 16)},
            "q_heads_not_multiple": {"q": torch.empty(4, 3, 16)},
            "v_layout": {"v": _pool("bhsd", 64, 8)},
            "v_not_a_tensor": {"v": None},
        }
        for name, overrides in cases.items():
            with self.subTest(case=name):
                self.assertEqual(_paged_layout(**self._args(**overrides)), "custom")


if __name__ == "__main__":
    unittest.main()
