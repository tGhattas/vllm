# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bridge real per-position model logits into the verified CPU constrained sampler.

Model-agnostic and CPU-testable: accepts a ``[L, V]`` array of per-position logits
(from any framework — call ``.detach().cpu().float().numpy()`` on a torch tensor
first) and runs the exact constrained sampler against a DFA. This is the seam that a
GPU validation script (see ``scripts/``) uses to feed one denoising step's logits
into the oracle-verified algorithm.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import constrained_sampler as cs  # noqa: E402
from finite_automaton import DFA  # noqa: E402


@dataclass
class ConstrainResult:
    sampled: list[int]
    greedy: list[int]
    log_partition: float
    accepted: bool  # DFA-acceptance of the sampled canvas (must be True)
    unconstrained_argmax: list[int]  # per-position argmax of the raw logits
    unconstrained_accepted: bool  # whether raw argmax happens to satisfy the DFA


def constrain_canvas_from_logits(
    logits: np.ndarray,
    dfa: DFA,
    fixed_positions: Mapping[int, int] | None = None,
    seed: int = 0,
) -> ConstrainResult:
    """Run the exact constrained sampler on real per-position logits.

    Args:
        logits: ``[L, V]`` per-position logits for one denoising step.
        dfa: constraint automaton over the token vocabulary.
        fixed_positions: canvas slots already committed by earlier denoising.
        seed: deterministic RNG seed for the stochastic sample.

    Returns:
        A ConstrainResult comparing constrained vs. unconstrained decoding.

    Raises:
        constrained_sampler.ImpossibleConstraintError: if no accepted canvas exists.
    """
    logits = np.asarray(logits, dtype=np.float64)
    log_probs = cs.logits_to_logprobs(logits)
    fixed = dict(fixed_positions or {})

    beta = cs.backward_messages(log_probs, dfa, fixed)
    logZ = float(beta[0, dfa.start_state])
    if logZ == cs.NEG_INF:
        raise cs.ImpossibleConstraintError("no canvas is accepted by the DFA")

    rng = np.random.default_rng(seed)
    sampled, _ = cs.sample(log_probs, dfa, rng, fixed_positions=fixed, beta=beta)
    greedy, _ = cs.greedy(log_probs, dfa, fixed)

    raw_argmax = [int(x) for x in np.argmax(logits, axis=-1)]
    return ConstrainResult(
        sampled=sampled,
        greedy=greedy,
        log_partition=logZ,
        accepted=dfa.accepts(sampled),
        unconstrained_argmax=raw_argmax,
        unconstrained_accepted=dfa.accepts(raw_argmax),
    )
