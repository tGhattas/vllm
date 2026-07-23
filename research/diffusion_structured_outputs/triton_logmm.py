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
        # N is constexpr so the contraction loop runs exactly N (not BLOCK) times,
        # and addresses offs*N + j stay in-block for offs < N, j < N (no OOB).
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        blk = N * N
        a_base = a_ptr + pid * blk
        b_base = b_ptr + pid * blk

        neg_inf = float("-inf")
        m = tl.full((BLOCK, BLOCK), neg_inf, tl.float32)
        s = tl.zeros((BLOCK, BLOCK), tl.float32)

        for j in range(0, N):
            aj = tl.load(a_base + offs * N + j, mask=mask, other=neg_inf)  # A[:, j]
            bj = tl.load(b_base + j * N + offs, mask=mask, other=neg_inf)  # B[j, :]
            t = aj[:, None] + bj[None, :]
            new_m = tl.maximum(m, t)
            finite = new_m != neg_inf
            corr = tl.where(finite, tl.exp(m - new_m), 0.0)
            tt = tl.where(finite, tl.exp(t - new_m), 0.0)
            s = s * corr + tt
            m = new_m

        out = tl.where(m != neg_inf, m + tl.log(s), neg_inf)
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
