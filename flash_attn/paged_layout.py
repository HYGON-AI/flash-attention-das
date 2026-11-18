# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Stride-based paged KV layout, kept free of the compiled extension.

The vLLM varlen entry receives a ``[pages, page_size, heads, dim]`` view of a
paged pool.  A page-major pool is already contiguous; a head-major pool is a
permute of ``[pages, heads, page_size, dim]``.  Tell them apart by the exact
stride signature, never by a dimension that happens to equal the page size.
When both signatures fit, the two layouts address the same memory, so the
caller keeps the vLLM default instead of guessing.
"""

import torch


def _signatures(t):
    """Return the ``(bshd, bhsd)`` stride signatures of a ``[P, S, H, D]`` view."""
    bshd = (
        t.stride(2) == t.shape[-1]
        and t.stride(1) == t.shape[-2] * t.shape[-1]
    )
    bhsd = (
        t.stride(1) == t.shape[-1]
        and t.stride(2) == t.shape[-3] * t.shape[-1]
    )
    return bshd, bhsd


def paged_kv_layout(k, v):
    """``bshd``, ``bhsd`` or ``custom`` for a ``[P, S, H, D]`` KV view.

    ``k`` and ``v`` share a pool, so both must carry the same unambiguous
    stride signature; a value tensor that disagrees keeps the vLLM default.
    """
    if not (torch.is_tensor(k) and torch.is_tensor(v)):
        return "custom"
    if k.dim() != 4 or v.dim() != 4:
        return "custom"
    k_bshd, k_bhsd = _signatures(k)
    v_bshd, v_bhsd = _signatures(v)
    if k_bshd and k_bhsd:
        return "custom"
    if k_bshd:
        return "bshd" if v_bshd else "custom"
    if k_bhsd:
        return "bhsd" if v_bhsd else "custom"
    return "custom"
