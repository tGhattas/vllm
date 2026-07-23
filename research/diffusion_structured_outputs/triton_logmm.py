# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel for the batched log-semiring matmul used by the log-depth sampler.

Computes, for a batch of ``[N, N]`` log-space blocks,

    C[b, i, k] = logsumexp_j ( A[b, i, j] + B[b, j, k] )

which is the tropical/log-semiring "matmul" at the heart of the constrained-sampler
product tree (`gpu_constrained_sampler.build_levels`). One Triton program handles
one batch block and does a FlashAttention-style numerically-stable single-pass
accumulation over the contraction index j (running max + rescaled running sum), so
there is no O(N³) intermediate.

Import-safe without Triton/CUDA (``HAS_TRITON = False``); install into the sampler
with ``gpu_constrained_sampler.set_logmm_impl(logmm_semiring)`` on a CUDA device.
"""

from __future__ import annotations

import torch

try:  # Triton ships with CUDA torch; absent on CPU-only installs.
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - exercised only on CPU-only hosts
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _logmm_kernel(a_ptr, b_ptr, c_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
        # Two-pass log-semiring matmul in log2 space: pass 1 finds the per-(i,k)
        # max (no exp/rescale, breaking the online-softmax loop-carried chain),
        # pass 2 sums a single hardware exp2 per term. LOG2E is folded into the
        # per-j loads so the N*N inner path has no multiply. exp2/log2 hit the same
        # SFU primitives as exp/log (2^(LOG2E*x)=e^x), so this is exact to fp32 ULP.
        # N is constexpr => loop runs exactly N times, addresses stay in-block.
        LOG2E: tl.constexpr = 1.4426950408889634
        LN2: tl.constexpr = 0.6931471805599453
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        blk = N * N
        a_base = a_ptr + pid * blk
        b_base = b_ptr + pid * blk
        neg_inf = float("-inf")

        m = tl.full((BLOCK, BLOCK), neg_inf, tl.float32)
        for j in range(0, N):
            aj = tl.load(a_base + offs * N + j, mask=mask, other=neg_inf) * LOG2E
            bj = tl.load(b_base + j * N + offs, mask=mask, other=neg_inf) * LOG2E
            m = tl.maximum(m, aj[:, None] + bj[None, :])

        ms = tl.where(m == neg_inf, 0.0, m)  # avoid -inf - -inf = NaN in pass 2
        s = tl.zeros((BLOCK, BLOCK), tl.float32)
        for j in range(0, N):
            aj = tl.load(a_base + offs * N + j, mask=mask, other=neg_inf) * LOG2E
            bj = tl.load(b_base + j * N + offs, mask=mask, other=neg_inf) * LOG2E
            s += tl.math.exp2(aj[:, None] + bj[None, :] - ms)

        out = tl.where(m == neg_inf, neg_inf, (m + tl.math.log2(s)) * LN2)
        c_base = c_ptr + pid * blk
        store_mask = mask[:, None] & mask[None, :]
        tl.store(c_base + offs[:, None] * N + offs[None, :], out, mask=store_mask)


def logmm_semiring(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Log-semiring matmul over the trailing ``[N, N]`` dims via the Triton kernel.

    ``a`` and ``b`` are ``[..., N, N]`` CUDA tensors; returns ``[..., N, N]``.
    Falls back to nothing — call only when ``HAS_TRITON`` and tensors are on CUDA.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    assert a.shape == b.shape and a.shape[-1] == a.shape[-2]
    lead = a.shape[:-2]
    n = a.shape[-1]
    a2 = a.reshape(-1, n, n).contiguous().to(torch.float32)
    b2 = b.reshape(-1, n, n).contiguous().to(torch.float32)
    bk = a2.shape[0]
    c2 = torch.empty_like(a2)
    block = triton.next_power_of_2(n)
    num_warps = 8 if block >= 64 else 4
    _logmm_kernel[(bk,)](a2, b2, c2, N=n, BLOCK=block, num_warps=num_warps)
    return c2.reshape(*lead, n, n).to(a.dtype)
