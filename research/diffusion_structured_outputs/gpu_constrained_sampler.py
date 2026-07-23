# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched GPU (torch) log-depth constrained sampler.

Research POC for vLLM issue #45572. A vectorized torch implementation of the
log-depth algorithm in `parallel_sampler.py` (Dang & Ermon, arXiv:2607.07026),
batched over a request batch and runnable on GPU:

  - transition matrices via batched scatter-logsumexp (`transition_matrices`);
  - a **level-batched** log-semiring product tree — O(log L) batched matmuls
    (`build_levels`);
  - a **level-batched** top-down midpoint-state sampler (O(log L) launches);
  - fully-parallel per-position token emission.

This is a *vectorized GPU implementation*, not a hand-written CUDA/Triton kernel;
it proves the algorithm ports to batched device ops and matches the numpy
reference exactly, and it is the artifact a production Triton kernel would mirror.
All B requests in a call share one DFA (batch same-schema requests); heterogeneous
schemas would loop or pad — see `design.md` §6.

Shapes: log_probs [B, L, V]; transition table next_state [N, V] (−1 = dead);
accepting_mask [N] bool; matrices/levels carry an [N, N] log-space block.
"""

from __future__ import annotations

import numpy as np
import torch
from finite_automaton import DFA

NEG_INF = float("-inf")


def build_transition_table(dfa: DFA) -> tuple[torch.Tensor, torch.Tensor, int]:
    """DFA → (next_state [N, V] long, accepting_mask [N] bool, start_state)."""
    N, V = dfa.num_states, dfa.vocab_size
    nxt = torch.full((N, V), -1, dtype=torch.long)
    for (s, v), s2 in dfa.transitions.items():
        nxt[s, v] = s2
    acc = torch.zeros(N, dtype=torch.bool)
    for s in dfa.accepting:
        acc[s] = True
    return nxt, acc, dfa.start_state


def _scatter_logsumexp(vals: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """logsumexp of ``vals`` [B, L, K] grouped by ``index`` [K] into [B, L, n]."""
    B, L, K = vals.shape
    idx = index.view(1, 1, K).expand(B, L, K)
    mx = torch.full((B, L, n), NEG_INF, device=vals.device, dtype=vals.dtype)
    mx.scatter_reduce_(2, idx, vals, reduce="amax", include_self=True)
    mxe = mx.gather(2, idx)
    ex = torch.exp(vals - mxe)
    sm = torch.zeros((B, L, n), device=vals.device, dtype=vals.dtype)
    sm.scatter_add_(2, idx, ex)
    return torch.where(sm > 0, mx + torch.log(sm), torch.full_like(mx, NEG_INF))


def transition_matrices(
    log_probs: torch.Tensor, next_state: torch.Tensor
) -> torch.Tensor:
    """Per-position log transition matrices M [B, L, N, N].

    One edge-based scatter over the DFA's defined transitions (typically far
    fewer than N·V), instead of a Python loop over states: for each edge
    (s, v) → s', route log_probs[..., v] into flat bucket s·N + s'.
    """
    B, L, V = log_probs.shape
    N = next_state.shape[0]
    next_state = next_state.to(log_probs.device)
    s_idx, v_idx = (next_state >= 0).nonzero(as_tuple=True)  # [E] edges
    dst_flat = s_idx * N + next_state[s_idx, v_idx]  # [E] into [N*N]
    vals = log_probs[:, :, v_idx]  # [B, L, E]
    flat = _scatter_logsumexp(vals, dst_flat, N * N)  # [B, L, N*N]
    return flat.reshape(B, L, N, N)


# Optional drop-in log-semiring matmul (e.g. the Triton kernel in triton_logmm.py).
# Set via set_logmm_impl(); used for CUDA tensors, else the torch fallback runs.
_LOGMM_IMPL = None


def set_logmm_impl(fn) -> None:
    """Install a custom ``(A, B) -> C`` log-semiring matmul (Triton kernel, etc.)."""
    global _LOGMM_IMPL
    _LOGMM_IMPL = fn


def _logmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched log-semiring matmul: C[...,i,k] = logsumexp_j a[...,i,j] + b[...,j,k].

    Uses the installed kernel for CUDA tensors if present; otherwise accumulates
    over the contraction index j with ``logaddexp`` so peak memory is O(...·N²)
    instead of the O(...·N³) of a materialized outer sum.
    """
    if _LOGMM_IMPL is not None and a.is_cuda:
        return _LOGMM_IMPL(a, b)
    n = a.shape[-1]
    out = a[..., :, 0:1] + b[..., 0:1, :]  # j = 0
    for j in range(1, n):
        out = torch.logaddexp(out, a[..., :, j : j + 1] + b[..., j : j + 1, :])
    return out


def _identity(n: int, device, dtype) -> torch.Tensor:
    m = torch.full((n, n), NEG_INF, device=device, dtype=dtype)
    m.fill_diagonal_(0.0)
    return m


def build_levels(matrices: torch.Tensor) -> list[torch.Tensor]:
    """Bottom-up log-semiring product tree; pads L to a power of two with identity.

    Returns levels[0] = padded matrices [B, Lpad, N, N], levels[d] = pairwise
    products (node width 2^d). O(log Lpad) batched matmuls.
    """
    B, L, N, _ = matrices.shape
    Lpad = 1 if L <= 1 else 1 << (L - 1).bit_length()
    if Lpad > L:
        ident = _identity(N, matrices.device, matrices.dtype)
        pad = ident.expand(B, Lpad - L, N, N)
        matrices = torch.cat([matrices, pad], dim=1)
    levels = [matrices]
    while levels[-1].shape[1] > 1:
        cur = levels[-1]
        levels.append(_logmm(cur[:, 0::2], cur[:, 1::2]))
    return levels


def log_partition(
    levels: list[torch.Tensor], start: int, accepting_mask: torch.Tensor
) -> torch.Tensor:
    """Log Z per batch element, from the root product. Shape [B]."""
    root = levels[-1][:, 0]  # [B, N, N]
    w = root[:, start, :]  # [B, N]
    w = torch.where(accepting_mask.to(w.device), w, torch.full_like(w, NEG_INF))
    return torch.logsumexp(w, dim=-1)


def _gumbel_argmax(logits: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    u = torch.rand(
        logits.shape, generator=gen, device=logits.device, dtype=logits.dtype
    )
    g = -torch.log(-torch.log(u.clamp_min(1e-30)))
    return torch.argmax(logits + g, dim=-1)


def sample(
    log_probs: torch.Tensor,
    next_state: torch.Tensor,
    accepting_mask: torch.Tensor,
    start: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one exact constrained canvas per batch element. Returns tokens [B, L].

    Every row is DFA-accepted by construction (same distribution as the numpy
    sequential/log-depth samplers). Raises if any batch element is unsatisfiable.
    """
    device = log_probs.device
    next_state = next_state.to(device)
    M = transition_matrices(log_probs, next_state)
    levels = build_levels(M)
    return sample_from_levels(
        log_probs, levels, next_state, accepting_mask, start, generator
    )


def sample_from_levels(
    log_probs: torch.Tensor,
    levels: list[torch.Tensor],
    next_state: torch.Tensor,
    accepting_mask: torch.Tensor,
    start: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample canvases given a prebuilt product tree (state fill + token emission)."""
    device = log_probs.device
    if generator is None:
        generator = torch.Generator(device=device)
    B, L, V = log_probs.shape
    N = next_state.shape[0]
    next_state = next_state.to(device)
    accepting_mask = accepting_mask.to(device)
    Lpad = levels[0].shape[1]

    z = torch.zeros((B, Lpad + 1), dtype=torch.long, device=device)
    z[:, 0] = start
    # end state ~ root[start, ·] restricted to accepting
    root = levels[-1][:, 0]
    w_end = root[:, start, :]
    w_end = torch.where(accepting_mask, w_end, torch.full_like(w_end, NEG_INF))
    if bool((torch.logsumexp(w_end, dim=-1) == NEG_INF).any()):
        raise ValueError("some batch element has no accepted canvas (logZ = -inf)")
    z[:, Lpad] = _gumbel_argmax(w_end, generator)

    depth = len(levels) - 1
    for d in range(depth, 0, -1):
        width = 1 << d
        num = levels[d].shape[1]
        below = levels[d - 1]
        left = below[:, 0::2]  # [B, num, N, N] product over [l, m)
        right = below[:, 1::2]  # [B, num, N, N] product over [m, r)
        l_pos = torch.arange(num, device=device) * width
        r_pos = l_pos + width
        m_pos = l_pos + width // 2
        zl = z[:, l_pos]  # [B, num]
        zr = z[:, r_pos]  # [B, num]
        lft = left.gather(2, zl[:, :, None, None].expand(B, num, 1, N)).squeeze(2)
        rgt = right.gather(3, zr[:, :, None, None].expand(B, num, N, 1)).squeeze(3)
        z[:, m_pos] = _gumbel_argmax(lft + rgt, generator)

    # token emission (fully parallel): x_i ~ p_i restricted to δ(z_i, ·) = z_{i+1}
    zin = z[:, :L]  # [B, L]
    ztgt = z[:, 1 : L + 1]  # [B, L]
    rows = next_state[zin]  # [B, L, V]
    valid = rows == ztgt[:, :, None]
    masked = torch.where(valid, log_probs, torch.full_like(log_probs, NEG_INF))
    return _gumbel_argmax(masked, generator)


def to_torch_logprobs(log_probs: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Helper: numpy [L, V] or [B, L, V] log-probs → torch [B, L, V]."""
    arr = np.asarray(log_probs, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None]
    return torch.tensor(arr, dtype=torch.float64, device=device)
