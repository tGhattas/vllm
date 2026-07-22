# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU test for the real-logits bridge, using synthetic logits as a stand-in."""

import constrained_sampler as cs
import numpy as np
import pytest
from finite_automaton import DFA
from model_adapters.logits_capture import constrain_canvas_from_logits


def test_bridge_forces_acceptance_when_raw_argmax_would_violate():
    L, V = 3, 4
    # Raw argmax favors (0,0,0); constrain to the single string (1,2,3).
    logits = np.full((L, V), -5.0)
    logits[:, 0] = 5.0  # unconstrained argmax = [0,0,0]
    dfa = DFA.from_exact_sequence([1, 2, 3], V)

    res = constrain_canvas_from_logits(logits, dfa, seed=0)
    assert res.sampled == [1, 2, 3]
    assert res.greedy == [1, 2, 3]
    assert res.accepted is True
    assert res.unconstrained_argmax == [0, 0, 0]
    assert res.unconstrained_accepted is False


def test_bridge_impossible_raises():
    L, V = 3, 3
    logits = np.zeros((L, V))
    dfa = DFA.from_exact_sequence([0, 1, 2, 0], V)  # length-4 acceptor, canvas len 3
    with pytest.raises(cs.ImpossibleConstraintError):
        constrain_canvas_from_logits(logits, dfa)
