# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify vllm's DiffusionConstraint (torch) against the numpy oracle.

The vLLM integration replaces canvas logits with these constrained marginals, so
they must equal `constrained_sampler.marginals` (the brute-force-validated
reference) to fp tolerance.
"""

import os
import sys

import constrained_sampler as cs
import numpy as np
import pytest
import torch
from finite_automaton import DFA
from test_constrained_sampler import parity_dfa, random_logprobs

# Import the vLLM module directly (it depends only on torch).
_VLLM_MODELS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "vllm",
    "model_executor",
    "models",
)
sys.path.insert(0, _VLLM_MODELS)
from diffusion_constraint import DiffusionConstraint  # noqa: E402


def _next_state(dfa: DFA) -> torch.Tensor:
    ns = np.full((dfa.num_states, dfa.vocab_size), -1, dtype=np.int64)
    for (s, v), d in dfa.transitions.items():
        ns[s, v] = d
    return torch.tensor(ns)


def _accepting(dfa: DFA) -> torch.Tensor:
    a = torch.zeros(dfa.num_states, dtype=torch.bool)
    for s in dfa.accepting:
        a[s] = True
    return a


def _torch_marginals(dfa, logits):
    con = DiffusionConstraint.from_next_state(
        _next_state(dfa), _accepting(dfa), dfa.start_state
    )
    out = con.constrained_log_marginals(torch.tensor(logits[None], dtype=torch.float64))
    return torch.softmax(out[0], dim=-1).numpy()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_oracle_parity(seed):
    L, V = 6, 4
    dfa = parity_dfa(L, V)
    lp = random_logprobs(L, V, seed)
    logits = lp  # already log-probs; constraint renormalizes internally
    got = _torch_marginals(dfa, logits)
    want = cs.marginals(lp, dfa)
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_matches_oracle_partial_dfa():
    target = [2, 0, 3, 1]
    dfa = DFA.from_exact_sequence(target, 4)
    lp = random_logprobs(4, 4, seed=5)
    got = _torch_marginals(dfa, lp)
    want = cs.marginals(lp, dfa)
    np.testing.assert_allclose(got, want, atol=1e-9)
    # forbidden tokens carry ~0 mass; argmax recovers the only accepted string
    assert got.argmax(-1).tolist() == target


def test_file_roundtrip(tmp_path):
    dfa = parity_dfa(5, 3)
    con = DiffusionConstraint.from_next_state(
        _next_state(dfa), _accepting(dfa), dfa.start_state
    )
    p = tmp_path / "c.pt"
    torch.save(
        {
            "num_states": con.N,
            "vocab_size": con.V,
            "start": con.start,
            "accepting": con.accepting,
            "edge_s": con.edge_s,
            "edge_v": con.edge_v,
            "edge_d": con.edge_d,
        },
        p,
    )
    con2 = DiffusionConstraint.from_file(str(p))
    lp = random_logprobs(5, 3, seed=1)
    x = torch.tensor(lp[None], dtype=torch.float64)
    assert torch.allclose(
        con.constrained_log_marginals(x), con2.constrained_log_marginals(x)
    )
