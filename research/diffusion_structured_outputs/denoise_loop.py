# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-step constrained denoising loop over a fixed canvas.

Research POC for vLLM issue #45572. Simulates a confidence-based diffusion
denoising schedule where, each step, some canvas positions are *committed*
(fixed) and the DFA constraint is re-solved conditioned on the fixed positions
(exactly the design in `design.md` §3-§4). Uses the exact constrained sampler in
`constrained_sampler.py` for the per-step marginals.

Key correctness fact (verified in tests): committing ONE position per step by
sampling it from its constrained marginal `P_D(x_i | fixed)` yields an *exact*
sample from `P_D`, regardless of the order positions are chosen (chain rule) — so
a confidence-ordered schedule is exact. Committing k>1 positions per step in
parallel (as real diffusion does) is a mean-field approximation that can strand
the constraint; a satisfiability guard falls back so the invariant
`log Z(fixed) > -inf` is preserved and the final canvas is always accepted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import constrained_sampler as cs
import numpy as np
from finite_automaton import DFA

# A step's logits provider: given the currently-fixed positions, return the
# per-position log-probs [L, V] the model would emit this denoising step. A
# static function (ignoring `fixed`) models constant per-position distributions;
# a real model re-runs a forward pass with the fixed tokens placed.
LogitsFn = Callable[[Mapping[int, int]], np.ndarray]


def static_logits(log_probs: np.ndarray) -> LogitsFn:
    """A LogitsFn returning the same per-position log-probs every step."""
    lp = np.asarray(log_probs, dtype=np.float64)
    return lambda _fixed: lp


def _entropy(row: np.ndarray) -> float:
    p = row[row > 0.0]
    return float(-np.sum(p * np.log(p)))


def denoise_step_order(marg: np.ndarray, free: Sequence[int]) -> list[int]:
    """Rank free positions by lowest constrained-marginal entropy (most confident)."""
    return sorted(free, key=lambda i: _entropy(marg[i]))


def _commit_token(row: np.ndarray, mode: str, rng: np.random.Generator) -> int:
    """Pick a token for one position from its constrained marginal `row`."""
    if mode == "greedy":
        return int(np.argmax(row))
    return int(rng.choice(row.shape[0], p=row / row.sum()))


def constrained_denoise(
    logits_fn: LogitsFn,
    dfa: DFA,
    length: int,
    rng: np.random.Generator | None = None,
    mode: str = "sample",
    fix_per_step: int = 1,
) -> dict:
    """Run a confidence-based constrained denoising loop to a full canvas.

    Each step: compute constrained marginals given the fixed positions, pick the
    `fix_per_step` most-confident free positions, and commit them (argmax if
    `mode="greedy"`, else a draw from the constrained marginal). A satisfiability
    guard preserves `log Z(fixed) > -inf` when committing in parallel.

    Returns a dict with the final canvas, per-step history, step count, and the
    per-step satisfiability flags.

    Raises:
        constrained_sampler.ImpossibleConstraintError: if the constraint is
            unsatisfiable from the start.
    """
    if mode not in ("sample", "greedy"):
        raise ValueError("mode must be 'sample' or 'greedy'")
    if rng is None:
        rng = np.random.default_rng(0)

    fixed: dict[int, int] = {}
    history: list[dict] = []
    steps = 0
    while len(fixed) < length:
        steps += 1
        log_probs = np.asarray(logits_fn(fixed), dtype=np.float64)
        # Raises ImpossibleConstraintError if the fixed set stranded the DFA.
        marg = cs.marginals(log_probs, dfa, fixed)
        free = [i for i in range(length) if i not in fixed]
        order = denoise_step_order(marg, free)
        chosen = order[:fix_per_step]

        newly = {i: _commit_token(marg[i], mode, rng) for i in chosen}
        # Parallel commit can jointly violate the DFA even though each position's
        # marginal is individually valid; fall back to the single most-confident
        # position (always satisfiable) if so.
        if cs.log_partition(log_probs, dfa, {**fixed, **newly}) == cs.NEG_INF:
            i = chosen[0]
            newly = {i: _commit_token(marg[i], mode, rng)}
        fixed = {**fixed, **newly}
        history.append(
            {"step": steps, "committed": dict(newly), "num_fixed": len(fixed)}
        )

    canvas = [fixed[i] for i in range(length)]
    return {
        "canvas": canvas,
        "history": history,
        "steps": steps,
        "accepted": dfa.accepts(canvas),
    }


def loop_probability(
    log_probs: np.ndarray,
    dfa: DFA,
    target: Sequence[int],
) -> float:
    """Exact probability the single-fix confidence-ordered loop yields `target`.

    Walks the deterministic confidence order the loop would take when forced to
    commit `target`'s tokens, multiplying the constrained marginal used at each
    step. By the chain rule this equals `P_D(target)` — the analytic check that
    the multi-step loop is an exact sampler (no Monte Carlo needed).
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    length = log_probs.shape[0]
    fixed: dict[int, int] = {}
    prob = 1.0
    while len(fixed) < length:
        marg = cs.marginals(log_probs, dfa, fixed)
        free = [i for i in range(length) if i not in fixed]
        i = denoise_step_order(marg, free)[0]  # most-confident position
        t = int(target[i])
        if marg[i, t] == 0.0:
            return 0.0  # loop can never emit this (DFA-rejected) target
        prob *= float(marg[i, t])
        fixed[i] = t
    return prob
