"""WeLM ops checks: paged_attention_partial, and hdim256 bf16 varlen_fwd vs WeLM's fa256.

Run with the freshly built extension first on sys.path, e.g.
  PYTHONPATH=<build/lib...> python tests/welm/test_welm_ops.py
Optional reference .so files (WELM_PA_PART_SO / WELM_FA256_SO) give bit-exact comparisons.
"""
import importlib.util, math, os, sys
import torch

torch.manual_seed(0)
dev = 'cuda'
H, KVH, D, PAGE = 6, 1, 256, 64
OUT_OFF, MAX_OFF = 10_000_000, 5_000_000


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def merge(tmp, nh, maxp, nparts, nv, mtp):
    """Combine partials -> [nv, mtp*H, D] (log2 units)."""
    f = tmp.view(torch.float32)
    n = nv * nh * maxp
    es = f[:n].view(nv, nh, maxp)[..., :nparts]
    mx = f[MAX_OFF // 2: MAX_OFF // 2 + n].view(nv, nh, maxp)[..., :nparts]
    o = tmp[OUT_OFF: OUT_OFF + n * D].view(nv, nh, maxp, D)[:, :, :nparts].float()
    m = mx.max(-1, keepdim=True).values
    w = es * torch.exp2(mx - m)
    return (o * w[..., None]).sum(2) / w.sum(-1)[..., None]


def bits_equal(a, b):
    # bitwise (== would report identical NaN bit patterns as different)
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def ref_attn(q, k, v, scale, causal):
    # q [n, H, D], k/v [L, D] (one kv head); causal = bottom-right aligned
    s = torch.einsum('nhd,ld->nhl', q.float(), k.float()) * scale
    if causal:
        L = k.shape[0]; n = q.shape[0]
        pos = torch.arange(L, device=dev)
        qpos = L - n + torch.arange(n, device=dev)
        s = s.masked_fill(pos[None, None, :] > qpos[:, None, None], float('-inf'))
    return torch.einsum('nhl,ld->nhd', s.softmax(-1), v.float()), torch.logsumexp(s, -1).transpose(0, 1)


def test_pa_partial(fa):
    ok = True
    ref_mod = load(os.environ['WELM_PA_PART_SO'], 'welm_pa_part') if os.environ.get('WELM_PA_PART_SO') else None
    for nv, mtp, L in [(3, 5, 512), (4, 1, 300), (2, 4, 1000), (8, 5, 37)]:
        nb = nv * 16 + 4
        kc = torch.randn(nb, KVH, PAGE, D, device=dev, dtype=torch.bfloat16)
        vc = torch.randn(nb, KVH, D, PAGE, device=dev, dtype=torch.bfloat16)
        nblk = math.ceil(L / PAGE)
        bt = torch.stack([torch.randperm(nb, device=dev)[:nblk] for _ in range(nv)]).int()
        seq = torch.full((nv,), L, dtype=torch.int32, device=dev)
        nh = mtp * H
        ps = 256 if nh > 15 else 512
        maxp = math.ceil(nblk * PAGE / ps)
        q = torch.randn(nv, mtp, H, D, device=dev, dtype=torch.bfloat16)
        tmp = torch.zeros(OUT_OFF + nv * nh * maxp * D + 64, dtype=torch.bfloat16, device=dev)
        fa.paged_attention_partial(tmp, q, kc, vc, bt, seq, mtp, ps, maxp)
        torch.cuda.synchronize()
        got = merge(tmp, nh, maxp, math.ceil(L / ps), nv, mtp).view(nv, mtp, H, D)
        err = 0.0
        for i in range(nv):
            k = kc[bt[i].long(), 0].reshape(-1, D)[:L]
            v = vc[bt[i].long(), 0].transpose(1, 2).reshape(-1, D)[:L]
            # partial mode is non-causal: every query slot sees the whole sequence
            err = max(err, (got[i] - ref_attn(q[i], k, v, D ** -0.5, False)[0]).abs().max().item())
        exact = ''
        if ref_mod is not None:
            tmp2 = torch.zeros_like(tmp)
            ref_mod.pa_partial(tmp2, q, kc, vc, bt, seq, mtp, ps, maxp, None)
            torch.cuda.synchronize()
            exact = f' bitexact_vs_so={bits_equal(tmp, tmp2)}'
            ok &= bits_equal(tmp, tmp2)
        # q_rows gather: rows from a flat [R, H, D] query, -1 = padding
        R = nv * mtp + 3
        qf = torch.randn(R, H, D, device=dev, dtype=torch.bfloat16)
        rows = torch.randperm(R, device=dev)[: nv * mtp].int().view(nv, mtp)
        rows[0, -1] = -1
        tmp3 = torch.zeros_like(tmp)
        fa.paged_attention_partial(tmp3, qf, kc, vc, bt, seq, mtp, ps, maxp, rows)
        qg = qf[rows.clamp(min=0).long()]
        tmp4 = torch.zeros_like(tmp)
        fa.paged_attention_partial(tmp4, qg.contiguous(), kc, vc, bt, seq, mtp, ps, maxp)
        torch.cuda.synchronize()
        g3 = merge(tmp3, nh, maxp, math.ceil(L / ps), nv, mtp).view(nv, mtp, H, D)
        g4 = merge(tmp4, nh, maxp, math.ceil(L / ps), nv, mtp).view(nv, mtp, H, D)
        valid = (rows >= 0)
        qr = (g3[valid] - g4[valid]).abs().max().item()
        if ref_mod is not None:
            tmp5 = torch.zeros_like(tmp)
            ref_mod.pa_partial(tmp5, qf, kc, vc, bt, seq, mtp, ps, maxp, rows)
            torch.cuda.synchronize()
            ok &= bits_equal(tmp3, tmp5)
            exact += f' q_rows_bitexact_vs_so={bits_equal(tmp3, tmp5)}'
        good = err < 2e-2 and qr == 0.0
        ok &= good
        print(f'pa_partial nv={nv} mtp={mtp} L={L}: max_err_vs_fp32={err:.4f} q_rows_vs_gather={qr}{exact} {"OK" if good else "FAIL"}')
    return ok


def test_fa256(fa):
    """hdim256 bf16 varlen_fwd (output + LSE) vs fp32; optionally no worse than WeLM's fa256."""
    ok = True
    ref_mod = load(os.environ['WELM_FA256_SO'], 'welm_fa256') if os.environ.get('WELM_FA256_SO') else None
    for nq, nk, causal in [(3000, 3000, True), (2048, 9000, True), (777, 777, True), (2048, 8192, False)]:
        q = torch.randn(nq, 6, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn(nk, 1, D, device=dev, dtype=torch.bfloat16); v = torch.randn_like(k)
        cq = torch.tensor([0, nq], dtype=torch.int32, device=dev); ck = torch.tensor([0, nk], dtype=torch.int32, device=dev)
        r = fa.varlen_fwd(q, k, v, None, cq, ck, None, None, None, None, nq, nk, 0.0, D ** -0.5, False, causal,
                          -1, -1, 0.0, False, None, None, None, None, None)
        out, lse = r[0], r[5]
        torch.cuda.synchronize()
        ro, rl = ref_attn(q, k[:, 0], v[:, 0], D ** -0.5, causal)
        err = (out.float() - ro).abs().max().item()
        lerr = (lse.float() - rl).abs().max().item()
        line = f'fa hdim256 bf16 nq={nq} nk={nk} causal={causal}: max_err_vs_fp32={err:.4f} lse_err={lerr:.1e}'
        good = err < 2e-2 and lerr < 1e-3
        if ref_mod is not None:
            o2 = torch.empty_like(q)
            ref_mod.varlen_fwd256(q, k, v, o2, cq, ck, nq, nk, D ** -0.5, causal, -1, -1, None, 20)
            torch.cuda.synchronize()
            e2 = (o2.float() - ro).abs().max().item()
            line += f' fa256_err={e2:.4f}'
            good &= err <= e2 * 1.05 + 1e-4
        ok &= good
        print(line, 'OK' if good else 'FAIL')
    return ok


if __name__ == '__main__':
    import flash_attn_2_cuda as fa
    print('flash_attn_2_cuda:', fa.__file__)
    ok = test_pa_partial(fa) & test_fa256(fa)
    print('ALL OK' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)
