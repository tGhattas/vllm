# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact brute-force oracle for the constrained-canvas distribution.

Enumerates the entire ``V**L`` canvas space, filters to DFA-accepted sequences and
renormalizes, giving ground truth to validate ``constrained_sampler`` against. Only
tractable for tiny ``L`` and ``V`` — that is the point: an independent reference
implemented by exhaustive enumeration rather than the DP.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping

import numpy as np
from finite_automaton import DFA


def _consistent_with_fixed(seq: tuple[int, ...], fixed: Mapping[int, int]) -> bool:
    return all(seq[i] == tok for i, tok in fixed.items())


def enumerate_constrained(
    log_probs: np.ndarray,
    dfa: DFA,
    fixed_positions: Mapping[int, int] | None = None,
) -> dict[tuple[int, ...], float]:
    """Return the exact constrained distribution as ``{sequence: probability}``.

    Weights each accepted, fixed-consistent sequence by ``prod_i p_i(x_i)`` and
    renormalizes. Empty dict means the constraint is impossible.
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    fixed = dict(fixed_positions or {})
    L, V = log_probs.shape

    weights: dict[tuple[int, ...], float] = {}
    total = 0.0
    for seq in itertools.product(range(V), repeat=L):
        if not _consistent_with_fixed(seq, fixed):
            continue
        if not dfa.accepts(seq):
            continue
        logw = float(sum(log_probs[i, seq[i]] for i in range(L)))
        w = np.exp(logw)
        weights[seq] = w
        total += w

    if total == 0.0:
        return {}
    return {seq: w / total for seq, w in weights.items()}


def brute_force_log_partition(
    log_probs: np.ndarray,
    dfa: DFA,
    fixed_positions: Mapping[int, int] | None = None,
) -> float:
    """Exact ``log Z`` by enumeration (``-inf`` if impossible)."""
    log_probs = np.asarray(log_probs, dtype=np.float64)
    fixed = dict(fixed_positions or {})
    L, V = log_probs.shape

    logws = []
    for seq in itertools.product(range(V), repeat=L):
        if not _consistent_with_fixed(seq, fixed):
            continue
        if not dfa.accepts(seq):
            continue
        logws.append(float(sum(log_probs[i, seq[i]] for i in range(L))))
    if not logws:
        return float("-inf")
    arr = np.array(logws, dtype=np.float64)
    m = float(np.max(arr))
    return m + float(np.log(np.sum(np.exp(arr - m))))


def brute_force_marginals(
    log_probs: np.ndarray,
    dfa: DFA,
    fixed_positions: Mapping[int, int] | None = None,
) -> np.ndarray:
    """Exact per-position marginals ``[L, V]`` by enumeration."""
    log_probs = np.asarray(log_probs, dtype=np.float64)
    L, V = log_probs.shape
    dist = enumerate_constrained(log_probs, dfa, fixed_positions)
    out = np.zeros((L, V), dtype=np.float64)
    for seq, p in dist.items():
        for i in range(L):
            out[i, seq[i]] += p
    return out


def brute_force_map(
    log_probs: np.ndarray,
    dfa: DFA,
    fixed_positions: Mapping[int, int] | None = None,
) -> tuple[tuple[int, ...] | None, float]:
    """Exact maximum-probability accepted canvas and its ``log P(x)`` (unnormalized)."""
    log_probs = np.asarray(log_probs, dtype=np.float64)
    fixed = dict(fixed_positions or {})
    L, V = log_probs.shape

    best_seq: tuple[int, ...] | None = None
    best_logw = float("-inf")
    for seq in itertools.product(range(V), repeat=L):
        if not _consistent_with_fixed(seq, fixed):
            continue
        if not dfa.accepts(seq):
            continue
        logw = float(sum(log_probs[i, seq[i]] for i in range(L)))
        if logw > best_logw:
            best_logw, best_seq = logw, seq
    return best_seq, best_logw
