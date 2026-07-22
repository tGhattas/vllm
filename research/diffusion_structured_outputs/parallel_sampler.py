# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Log-depth parallel constrained sampler (novelty of Dang & Ermon, 2607.07026).

Samples the same constrained mean-field posterior `P_D` as
`constrained_sampler.sample`, but with **O(log L) parallel depth** instead of the
O(L) sequential ancestral pass. CPU reference implementation of the paper's
Algorithm 1: a chain graphical model over automaton states, with

  - per-position transition matrices `M[i][s, s'] = Σ_{v: δ(s,v)=s'} pᵢ(v)`
    (built in log space);
  - a bottom-up **segment tree of range products** under the log-semiring matmul
    (`⊗`), computable in O(log L) dependency depth;
  - a top-down **midpoint-state recursion** that samples each segment's middle
    automaton state from the segment products and recurses into the two halves in
    parallel (O(log L) depth);
  - independent per-position token emission given consecutive states (O(1) depth).

The factoring is exact, so the output distribution equals the sequential sampler's
(verified against brute force in tests). This is a correctness/depth reference, not
a GPU kernel; `sampling_depth()` reports the sequential dependency levels to show
the O(log L) span.

Complexity: FLOPs O(L·(N³ + |edges|·V)), parallel depth O(log L), memory O(L·N²).
"""

from __future__ import annotations

from collections.abc import Mapping

import constrained_sampler as cs
import numpy as np
from finite_automaton import DFA

NEG_INF = cs.NEG_INF


def _logmm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Log-semiring matmul: out[i,k] = logsumexp_j (a[i,j] + b[j,k])."""
    # a: [N, N], b: [N, N] -> broadcast to [N, N, N] then reduce middle axis.
    s = a[:, :, None] + b[None, :, :]
    m = np.max(s, axis=1)
    with np.errstate(invalid="ignore"):
        out = m + np.log(np.sum(np.exp(s - m[:, None, :]), axis=1))
    out[~np.isfinite(m)] = NEG_INF  # all -inf column stays -inf, not nan
    return out


def transition_matrices(
    log_probs: np.ndarray,
    dfa: DFA,
    fixed_positions: Mapping[int, int] | None = None,
) -> np.ndarray:
    """Per-position log transition matrices, shape ``[L, N, N]``.

    ``M[i][s, s'] = logsumexp over tokens v with δ(s, v)=s' of log pᵢ(v)``. A fixed
    position restricts the token set at that position to its committed token.
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    fixed = dict(fixed_positions or {})
    L, V = log_probs.shape
    N = dfa.num_states
    mats = np.full((L, N, N), NEG_INF, dtype=np.float64)
    for i in range(L):
        tokens = [fixed[i]] if i in fixed else range(V)
        for s in range(N):
            for v in tokens:
                s2 = dfa.step(s, v)
                if s2 is None:
                    continue
                mats[i, s, s2] = np.logaddexp(mats[i, s, s2], log_probs[i, v])
    return mats


class _SegmentTree:
    """Balanced binary tree of log-semiring range products over positions [0, L)."""

    def __init__(self, mats: np.ndarray) -> None:
        self.L = mats.shape[0]
        self.N = mats.shape[1]
        # product[(l, r)] = M[l] ⊗ ... ⊗ M[r-1]; built bottom-up.
        self._prod: dict[tuple[int, int], np.ndarray] = {}
        self._depth = 0
        self._build(0, self.L, mats)

    def _build(self, lo: int, hi: int, mats: np.ndarray) -> np.ndarray:
        if hi - lo == 1:
            p = mats[lo]
        else:
            m = (lo + hi) // 2
            left = self._build(lo, m, mats)
            right = self._build(m, hi, mats)
            p = _logmm(left, right)
        self._prod[(lo, hi)] = p
        return p

    def product(self, lo: int, hi: int) -> np.ndarray:
        return self._prod[(lo, hi)]

    @property
    def height(self) -> int:
        return max(1, (self.L - 1).bit_length())


def log_partition(
    log_probs: np.ndarray,
    dfa: DFA,
    fixed_positions: Mapping[int, int] | None = None,
) -> float:
    """Log Z via the root range product (matches the sequential module's value)."""
    mats = transition_matrices(log_probs, dfa, fixed_positions)
    root = _SegmentTree(mats).product(0, mats.shape[0])
    acc = [root[dfa.start_state, s] for s in dfa.accepting]
    return cs._logsumexp(np.array(acc, dtype=np.float64))


def _sample_from_logweights(logw: np.ndarray, rng: np.random.Generator) -> int:
    lse = cs._logsumexp(logw)
    p = np.exp(logw - lse)
    p = np.where(np.isfinite(p), p, 0.0)
    p /= p.sum()
    return int(rng.choice(logw.shape[0], p=p))


def sample(
    log_probs: np.ndarray,
    dfa: DFA,
    rng: np.random.Generator,
    fixed_positions: Mapping[int, int] | None = None,
) -> tuple[list[int], float]:
    """Draw one exact sample from `P_D` using the O(log L)-depth algorithm.

    Returns the canvas and its `log P_D(x)`. Every canvas is DFA-accepted by
    construction.

    Raises:
        constrained_sampler.ImpossibleConstraintError: if no accepted canvas exists.
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    fixed = dict(fixed_positions or {})
    L, V = log_probs.shape
    mats = transition_matrices(log_probs, dfa, fixed)
    tree = _SegmentTree(mats)

    root = tree.product(0, L)
    # End state: accepting states weighted by the full range product from start.
    acc = list(dfa.accepting)
    end_w = np.array([root[dfa.start_state, s] for s in acc], dtype=np.float64)
    logZ = cs._logsumexp(end_w)
    if logZ == NEG_INF:
        raise cs.ImpossibleConstraintError("no canvas is accepted by the DFA")
    z_end = acc[_sample_from_logweights(end_w, rng)]

    # Boundary automaton states z[0..L]; z[0]=start, z[L]=sampled end.
    z = [dfa.start_state] + [None] * (L - 1) + [z_end]

    def fill(lo: int, hi: int) -> None:
        if hi - lo <= 1:
            return
        m = (lo + hi) // 2
        left = tree.product(lo, m)  # z_lo -> z_m
        right = tree.product(m, hi)  # z_m -> z_hi
        w = left[z[lo], :] + right[:, z[hi]]
        z[m] = _sample_from_logweights(w, rng)
        fill(lo, m)
        fill(m, hi)

    fill(0, L)

    # Emit tokens given consecutive states (independent per position, O(1) depth).
    seq: list[int] = []
    for i in range(L):
        tokens = [fixed[i]] if i in fixed else range(V)
        cand, logw = [], []
        for v in tokens:
            if dfa.step(z[i], v) == z[i + 1]:
                cand.append(v)
                logw.append(log_probs[i, v])
        idx = _sample_from_logweights(np.array(logw, dtype=np.float64), rng)
        seq.append(cand[idx])

    # log P_D(x) = Σ_i log pᵢ(xᵢ) − log Z.
    logpd = float(sum(log_probs[i, seq[i]] for i in range(L)) - logZ)
    return seq, logpd


def sampling_depth(length: int) -> int:
    """Sequential dependency depth of the parallel sampler ≈ tree height = O(log L)."""
    if length <= 1:
        return 1
    return (length - 1).bit_length()
