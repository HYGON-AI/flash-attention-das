# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import random
import re

import pytest
import torch
from einops import rearrange, repeat
from flash_attn import vllm_flash_attn_with_kvcache as flash_attn_with_kvcache


def _get_gfx_arch():
    """Return the current ROCm architecture name, for example ``gfx942``."""
    if not torch.cuda.is_available() or torch.version.hip is None:
        return None

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = getattr(properties, "gcnArchName", "")
    return arch.split(":", 1)[0].lower() or None


def _is_gfx936_or_earlier(arch):
    """Whether *arch* is an AMD gfx architecture no newer than gfx936."""
    if arch is None:
        return False

    match = re.match(r"^gfx(\d+)", arch)
    return match is not None and int(match.group(1)) <= 936


def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / (query.size(-1) ** 0.5)
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype, device=query.device)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool, device=query.device).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value


def run_single_test(b, s_q, mean_sk, h_q, h_kv, d, causal, is_fp8, torch_dtype):
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    random.seed(0)

    block_size = 64
    max_seqlen_pad = (mean_sk//block_size+1)*block_size
    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32, device=device)
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32, device=device).view(
        b, max_seqlen_pad // block_size
    )
    k_scale = v_scale = None
    q = torch.randn(b, s_q, h_q, d, device=device, dtype=torch_dtype)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d, device=device, dtype=torch_dtype)
    blocked_v = torch.randn(block_table.numel(), block_size, h_kv, d, device=device, dtype=torch_dtype)
    if is_fp8:
        fp8type = torch.float8_e5m2 if _is_gfx936_or_earlier(_get_gfx_arch()) else torch.float8_e4m3fn
        q = q.to(fp8type).to(torch_dtype)
        blocked_k = blocked_k.to(fp8type)
        blocked_v = blocked_k.to(fp8type)

    blocked_k_ref = blocked_k.to(torch_dtype)
    blocked_v_ref = blocked_v.to(torch_dtype)

    blocked_k_ = blocked_k.permute(0, 2, 1, 3).contiguous()
    blocked_v_ = blocked_v.permute(0, 2, 3, 1).contiguous()

    def flash_kvcache():
        return flash_attn_with_kvcache(
            q,
            blocked_k_,
            blocked_v_,
            k_scale=k_scale,
            v_scale=v_scale,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            causal=causal,
            num_splits=0,
            max_seqlen_k=mean_sk,
            return_softmax_lse=False,
        )

    def ref_kvcache():
        out = torch.empty(b, s_q, h_q, d, dtype=torch.float32, device=device)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            output = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k_ref.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v_ref.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                h_q=h_q,
                h_kv=h_kv,
                is_causal=causal,
            )
            out[i] = output.transpose(0, 1)
        return out.to(q.dtype)

    out_flash = flash_kvcache()
    out_torch = ref_kvcache()
    if is_fp8:
        atol=7e-2
    else :
        atol=1e-2
    assert torch.allclose(out_flash, out_torch, atol=atol, rtol=1e-2), (
        f"b={b}, s_q={s_q}, sk={mean_sk}, h_q={h_q}, h_kv={h_kv}, d={d}, "
        f"causal={causal}, fp8={is_fp8}, dtype={torch_dtype}"
    )


test_configs = [
    (1, 1, 1025, 16, 16, 128, True),
    (20, 1, 16384, 28, 4, 128, True),
    (8, 1, 17408, 16, 2, 128, True),
    (240, 1, 2048, 6, 1, 128, True),
    (32, 1, 3584, 32, 8, 128, True),
    (64, 1, 3584, 16, 4, 128, True),
    (70, 1, 3584, 4, 1, 128, True),
    (80, 1, 3584, 32, 4, 128, True),
    (100, 1, 3584, 16, 2, 128, True),
    (80, 1, 3584, 8, 1, 128, True),
    (24, 1, 2048, 8, 1, 128, True),
    (32, 1, 2048, 8, 1, 128, True),
    (48, 1, 2048, 8, 1, 128, True),
    (48, 1, 1514, 10, 2, 128, True),
    (2, 1, 32402, 10, 2, 128, True),
    (1, 1, 51200, 10, 2, 128, True),
    (1, 1, 455, 24, 1, 128, True),
    (10, 1, 2048, 16, 2, 128, True),
]


@pytest.mark.parametrize("b,s_q,mean_sk,h_q,h_kv,d,causal", test_configs)
@pytest.mark.parametrize("torch_dtype,is_fp8",[
        (torch.bfloat16, False),
        (torch.float16, False),
        (torch.bfloat16, True),
        (torch.float16, True),
    ])
def test_flash_kvcache(b, s_q, mean_sk, h_q, h_kv, d, causal, torch_dtype, is_fp8):

    run_single_test(b, s_q, mean_sk, h_q, h_kv, d, causal, is_fp8, torch_dtype)
