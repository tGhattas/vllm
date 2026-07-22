# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Oracle tests for the multi-step constrained denoising loop.

Headline check: committing one position per step from its constrained marginal
reproduces P_D exactly (analytic, no Monte Carlo) — proving the design's
"re-solve as positions get fixed" is exact regardless of schedule order. Also
checks the satisfiability invariant and that parallel commits stay accepted.
"""

import constrained_sampler as cs
import denoise_loop as dl
import numpy as np
import pytest
from brute_force_oracle import enumerate_constrained
from finite_automaton import DFA
from test_constrained_sampler import parity_dfa, random_logprobs


def test_loop_probability_matches_bruteforce_exactly():
    """Single-fix confidence-ordered loop is an exact P_D sampler (analytic)."""
    L, V = 4, 3
    lp = random_logprobs(L, V, seed=17)
    dfa = parity_dfa(L, V)
    exact = enumerate_constrained(lp, dfa)
    total = 0.0
    import itertools

    for seq in itertools.product(range(V), repeat=L):
        got = dl.loop_probability(lp, dfa, seq)
        want = exact.get(seq, 0.0)
        assert got == pytest.approx(want, abs=1e-9), f"{seq}: {got} vs {want}"
        total += got
    assert total == pytest.approx(1.0, abs=1e-9)  # proper distribution


def test_loop_single_fix_takes_L_steps_and_accepts():
    L, V = 5, 3
    lp = random_logprobs(L, V, seed=3)
    dfa = parity_dfa(L, V)
    fn = dl.static_logits(lp)
    for seed in range(20):
        out = dl.constrained_denoise(
            fn, dfa, L, np.random.default_rng(seed), mode="sample", fix_per_step=1
        )
        assert out["accepted"]
        assert out["steps"] == L  # one commit per step
        nfixed = [h["num_fixed"] for h in out["history"]]
        assert nfixed == sorted(nfixed) and nfixed[-1] == L  # monotonic to full


def test_loop_greedy_accepted():
    L, V = 5, 3
    lp = random_logprobs(L, V, seed=9)
    dfa = parity_dfa(L, V)
    out = dl.constrained_denoise(dl.static_logits(lp), dfa, L, mode="greedy")
    assert out["accepted"]


def test_parallel_commit_stays_accepted():
    """Committing many positions per step must never violate the DFA (guard)."""
    L, V = 6, 3
    lp = random_logprobs(L, V, seed=21)
    dfa = parity_dfa(L, V)
    fn = dl.static_logits(lp)
    for seed in range(30):
        out = dl.constrained_denoise(
            fn, dfa, L, np.random.default_rng(seed), mode="sample", fix_per_step=L
        )
        assert out["accepted"]  # satisfiability guard preserved the invariant
        assert out["steps"] <= L


def test_loop_mc_matches_bruteforce():
    """Exercise constrained_denoise's sampling path against enumeration (MC)."""
    L, V = 3, 3
    lp = random_logprobs(L, V, seed=8)
    dfa = parity_dfa(L, V)
    exact = enumerate_constrained(lp, dfa)
    fn = dl.static_logits(lp)
    rng = np.random.default_rng(123)
    n = 20_000
    counts: dict[tuple, int] = {}
    for _ in range(n):
        out = dl.constrained_denoise(fn, dfa, L, rng, mode="sample", fix_per_step=1)
        assert out["accepted"]
        key = tuple(out["canvas"])
        counts[key] = counts.get(key, 0) + 1
    for key, p in exact.items():
        assert counts.get(key, 0) / n == pytest.approx(p, abs=0.02)


def test_loop_impossible_raises():
    L, V = 3, 3
    dfa = DFA.from_exact_sequence([0, 1, 2, 0], V)  # length-4 acceptor, canvas 3
    lp = random_logprobs(L, V, seed=2)
    with pytest.raises(cs.ImpossibleConstraintError):
        dl.constrained_denoise(dl.static_logits(lp), dfa, L)


def test_fixed_input_positions_are_respected():
    """A prompt/prefixed position provided via the DFA is honored end to end."""
    L, V = 4, 3
    seqs = [(1, 0, 1, 0), (1, 1, 0, 0), (1, 0, 0, 1)]  # all start with token 1
    dfa = DFA.from_sequence_set(seqs, V)
    lp = random_logprobs(L, V, seed=4)
    out = dl.constrained_denoise(dl.static_logits(lp), dfa, L, mode="greedy")
    assert tuple(out["canvas"]) in set(seqs)
    assert out["canvas"][0] == 1
