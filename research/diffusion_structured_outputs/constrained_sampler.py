# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact constrained sampling over a fixed-length canvas subject to a DFA.

Research POC for vLLM issue #45572 (canvas-aware structured outputs for diffusion
language models). Standalone, CPU-only, NumPy-only.

Problem (the "constrained mean-field posterior" of Dang & Ermon, arXiv:2607.07026):
given independent per-position token distributions ``p_i(t)`` over a fixed canvas
of length ``L`` and a DFA ``D``, define the base joint distribution

    P(x) = prod_i p_i(x_i)

and sample from / reason about the distribution conditioned on ``x`` being
accepted by ``D``:

    P_D(x) = P(x) * 1[D accepts x] / Z ,   Z = sum_{x : D accepts x} P(x).

We implement the correctness-first *sequential ancestral* sampler via a numerically
stable log-space forward/backward dynamic program (not the paper's log-depth
parallel variant). ``fixed_positions`` conditions specific canvas slots on a known
token (e.g. positions already committed by an earlier denoising step); this only
restricts a position's support to that token and leaves the constrained
distribution over the remaining free positions exact.

Complexity (L = canvas length, V = vocab size, N = DFA states):
    backward / forward / marginals : O(L * N * V) time, O(L * N) memory.
    one ancestral sample           : O(L * V) time given cached backward messages.
The paper reduces sampling depth to O(log L) via arithmetic-circuit techniques;
correctness, not parallelism, is the goal here.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from finite_automaton import DFA

NEG_INF = -np.inf


class ImpossibleConstraintError(ValueError):
    """Raised when no accepted canvas exists (partition function is zero)."""


def _logsumexp(values: np.ndarray) -> float:
    """Numerically stable log-sum-exp that treats an all ``-inf`` input as ``-inf``."""
    if values.size == 0:
        return NEG_INF
    m = float(np.max(values))
    if m == NEG_INF:
        return NEG_INF
    return m + float(np.log(np.sum(np.exp(values - m))))


def logits_to_logprobs(logits: np.ndarray) -> np.ndarray:
    """Convert ``[L, V]`` logits to normalized per-position log-probabilities."""
    logits = np.asarray(logits, dtype=np.float64)
    m = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _allowed_tokens(i: int, vocab_size: int, fixed: Mapping[int, int]) -> list[int]:
    if i in fixed:
        return [fixed[i]]
    return list(range(vocab_size))


def backward_messages(
    log_probs: np.ndarray,
    dfa: DFA,
    fixed_positions: Mapping[int, int] | None = None,
) -> np.ndarray:
    """Log completion weights ``beta[i, q]`` = log P(reach acceptance | q, i).

    ``beta`` has shape ``[L + 1, N]``. ``beta[L, q]`` is 0 for accepting states and
    ``-inf`` otherwise. The log partition function is ``beta[0, start_state]``.
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    fixed = dict(fixed_positions or {})
    L, V = log_probs.shape
    N = dfa.num_states

    beta = np.full((L + 1, N), NEG_INF, dtype=np.float64)
    for q in range(N):
        beta[L, q] = 0.0 if q in dfa.accepting else NEG_INF

    for i in range(L - 1, -1, -1):
        tokens = _allowed_tokens(i, V, fixed)
        for q in range(N):
            terms = []
            for t in tokens:
                nq = dfa.step(q, t)
                if nq is None:
                    continue
                b = beta[i + 1, nq]
                if b == NEG_INF:
                    continue
                terms.append(log_probs[i, t] + b)
            beta[i, q] = _logsumexp(np.array(terms, dtype=np.float64))
    return beta


def forward_messages(
    log_probs: np.ndarray,
    dfa: DFA,
    fixed_positions: Mapping[int, int] | None = None,
) -> np.ndarray:
    """Log prefix weights ``alpha[i, q]`` = log(sum over length-i prefixes reaching q).

    ``alpha`` has shape ``[L + 1, N]``; ``alpha[0, start_state] = 0``.
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    fixed = dict(fixed_positions or {})
    L, V = log_probs.shape
    N = dfa.num_states

    alpha = np.full((L + 1, N), NEG_INF, dtype=np.float64)
    alpha[0, dfa.start_state] = 0.0

    for i in range(L):
        tokens = _allowed_tokens(i, V, fixed)
        # Accumulate contributions into position i+1.
        contrib: list[list[float]] = [[] for _ in range(N)]
        for q in range(N):
            a = alpha[i, q]
            if a == NEG_INF:
                continue
            for t in tokens:
                nq = dfa.step(q, t)
                if nq is None:
                    continue
                contrib[nq].append(a + log_probs[i, t])
        for nq in range(N):
            if contrib[nq]:
                alpha[i + 1, nq] = _logsumexp(np.array(contrib[nq], dtype=np.float64))
    return alpha


def log_partition(
    log_probs: np.ndarray,
    dfa: DFA,
    fixed_positions: Mapping[int, int] | None = None,
) -> float:
    """Log of the constrained partition function ``Z`` (``-inf`` if impossible)."""
    beta = backward_messages(log_probs, dfa, fixed_positions)
    return float(beta[0, dfa.start_state])


def sample(
    log_probs: np.ndarray,
    dfa: DFA,
    rng: np.random.Generator,
    fixed_positions: Mapping[int, int] | None = None,
    beta: np.ndarray | None = None,
) -> tuple[list[int], float]:
    """Draw one exact sample from the constrained distribution ``P_D``.

    Returns the sampled canvas and its log-probability ``log P_D(x)``. Every
    returned canvas is guaranteed accepted by ``dfa`` (satisfaction by
    construction). Pass a cached ``beta`` to amortize the backward pass.

    Raises:
        ImpossibleConstraintError: if no accepted canvas exists.
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    fixed = dict(fixed_positions or {})
    L, V = log_probs.shape
    if beta is None:
        beta = backward_messages(log_probs, dfa, fixed)

    if beta[0, dfa.start_state] == NEG_INF:
        raise ImpossibleConstraintError("no canvas is accepted by the DFA")

    q = dfa.start_state
    seq: list[int] = []
    logp = 0.0
    for i in range(L):
        tokens = _allowed_tokens(i, V, fixed)
        cand: list[tuple[int, int]] = []
        weights: list[float] = []
        for t in tokens:
            nq = dfa.step(q, t)
            if nq is None:
                continue
            b = beta[i + 1, nq]
            if b == NEG_INF:
                continue
            weights.append(log_probs[i, t] + b)
            cand.append((t, nq))
        w = np.array(weights, dtype=np.float64)
        lse = _logsumexp(w)  # equals beta[i, q]
        probs = np.exp(w - lse)
        probs /= probs.sum()  # guard against fp drift before rng.choice
        idx = int(rng.choice(len(cand), p=probs))
        t, nq = cand[idx]
        seq.append(t)
        q = nq
        logp += float(w[idx] - lse)
    return seq, logp


def marginals(
    log_probs: np.ndarray,
    dfa: DFA,
    fixed_positions: Mapping[int, int] | None = None,
) -> np.ndarray:
    """Exact per-position marginals ``P_D(x_i = t)`` as a ``[L, V]`` array.

    Rows sum to 1 (up to fp error); a fixed position puts all mass on its token.

    Raises:
        ImpossibleConstraintError: if no accepted canvas exists.
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    fixed = dict(fixed_positions or {})
    L, V = log_probs.shape

    alpha = forward_messages(log_probs, dfa, fixed)
    beta = backward_messages(log_probs, dfa, fixed)
    logZ = beta[0, dfa.start_state]
    if logZ == NEG_INF:
        raise ImpossibleConstraintError("no canvas is accepted by the DFA")

    out = np.zeros((L, V), dtype=np.float64)
    for i in range(L):
        tokens = _allowed_tokens(i, V, fixed)
        for t in tokens:
            terms = []
            for q in range(dfa.num_states):
                a = alpha[i, q]
                if a == NEG_INF:
                    continue
                nq = dfa.step(q, t)
                if nq is None:
                    continue
                b = beta[i + 1, nq]
                if b == NEG_INF:
                    continue
                terms.append(a + log_probs[i, t] + b)
            lm = _logsumexp(np.array(terms, dtype=np.float64))
            out[i, t] = np.exp(lm - logZ) if lm != NEG_INF else 0.0
    return out


def greedy(
    log_probs: np.ndarray,
    dfa: DFA,
    fixed_positions: Mapping[int, int] | None = None,
) -> tuple[list[int], float]:
    """Return the maximum-probability accepted canvas (constrained Viterbi / MAP).

    This is the joint argmax over accepted sequences, which in general differs
    from taking the per-position argmax of the marginals.

    Raises:
        ImpossibleConstraintError: if no accepted canvas exists.
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    fixed = dict(fixed_positions or {})
    L, V = log_probs.shape
    N = dfa.num_states

    gamma = np.full((L + 1, N), NEG_INF, dtype=np.float64)
    back: dict[tuple[int, int], tuple[int, int]] = {}
    for q in range(N):
        gamma[L, q] = 0.0 if q in dfa.accepting else NEG_INF

    for i in range(L - 1, -1, -1):
        tokens = _allowed_tokens(i, V, fixed)
        for q in range(N):
            best = NEG_INF
            best_t = best_nq = None
            for t in tokens:
                nq = dfa.step(q, t)
                if nq is None:
                    continue
                val = log_probs[i, t] + gamma[i + 1, nq]
                if val > best:
                    best, best_t, best_nq = val, t, nq
            gamma[i, q] = best
            if best_t is not None:
                back[(i, q)] = (best_t, best_nq)

    if gamma[0, dfa.start_state] == NEG_INF:
        raise ImpossibleConstraintError("no canvas is accepted by the DFA")

    q = dfa.start_state
    seq: list[int] = []
    for i in range(L):
        t, nq = back[(i, q)]
        seq.append(t)
        q = nq
    return seq, float(gamma[0, dfa.start_state])
