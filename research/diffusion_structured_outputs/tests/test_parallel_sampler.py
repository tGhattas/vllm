# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Oracle tests for the log-depth parallel constrained sampler (paper novelty).

Verifies the O(log L)-depth sampler draws from the SAME distribution as the
sequential ancestral sampler / brute force, and that its dependency depth grows
like log L (not L).
"""

import math

import constrained_sampler as cs
import numpy as np
import parallel_sampler as ps
import pytest
from brute_force_oracle import enumerate_constrained
from finite_automaton import DFA
from test_constrained_sampler import parity_dfa, random_logprobs


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_parallel_logpartition_matches(seed):
    L, V = 5, 3
    lp = random_logprobs(L, V, seed)
    dfa = parity_dfa(L, V)
    assert ps.log_partition(lp, dfa) == pytest.approx(
        cs.log_partition(lp, dfa), abs=1e-9
    )


def test_parallel_matches_bruteforce_distribution():
    L, V = 5, 3  # odd length -> unbalanced segment tree splits, exercises recursion
    lp = random_logprobs(L, V, seed=31)
    dfa = parity_dfa(L, V)
    exact = enumerate_constrained(lp, dfa)
    rng = np.random.default_rng(7)
    n = 40_000
    counts: dict[tuple, int] = {}
    for _ in range(n):
        seq, logpd = ps.sample(lp, dfa, rng)
        assert dfa.accepts(seq)  # accepted by construction
        key = tuple(seq)
        counts[key] = counts.get(key, 0) + 1
        # returned log-prob equals the exact constrained probability
        assert math.exp(logpd) == pytest.approx(exact[key], abs=1e-9)
    for key, p in exact.items():
        assert counts.get(key, 0) / n == pytest.approx(p, abs=0.02)


def test_parallel_single_string():
    L, V = 6, 4
    target = [3, 1, 0, 2, 2, 1]
    dfa = DFA.from_exact_sequence(target, V)
    lp = random_logprobs(L, V, seed=5)
    rng = np.random.default_rng(0)
    for _ in range(15):
        seq, _ = ps.sample(lp, dfa, rng)
        assert seq == target


def test_parallel_fixed_positions_match_sequential():
    L, V = 5, 3
    lp = random_logprobs(L, V, seed=12)
    dfa = parity_dfa(L, V)
    fixed = {0: 1, 2: 0}
    # partition with fixed positions must match the sequential module exactly
    assert ps.log_partition(lp, dfa, fixed) == pytest.approx(
        cs.log_partition(lp, dfa, fixed), abs=1e-9
    )
    rng = np.random.default_rng(3)
    for _ in range(30):
        seq, _ = ps.sample(lp, dfa, rng, fixed_positions=fixed)
        assert seq[0] == 1 and seq[2] == 0
        assert dfa.accepts(seq)


def test_parallel_impossible_raises():
    V = 3
    dfa = DFA.from_exact_sequence([0, 1, 2, 0], V)  # length-4 acceptor
    lp = random_logprobs(3, V, seed=2)
    assert ps.log_partition(lp, dfa) == float("-inf")
    with pytest.raises(cs.ImpossibleConstraintError):
        ps.sample(lp, dfa, np.random.default_rng(0))


def test_parallel_and_sequential_agree_on_marginals():
    """Empirical marginals of the parallel sampler match exact constrained marginals."""
    L, V = 4, 3
    lp = random_logprobs(L, V, seed=44)
    dfa = parity_dfa(L, V)
    exact_marg = cs.marginals(lp, dfa)
    rng = np.random.default_rng(9)
    n = 40_000
    emp = np.zeros((L, V))
    for _ in range(n):
        seq, _ = ps.sample(lp, dfa, rng)
        for i in range(L):
            emp[i, seq[i]] += 1
    emp /= n
    np.testing.assert_allclose(emp, exact_marg, atol=0.02)


@pytest.mark.parametrize(
    "length,expected",
    [(1, 1), (2, 1), (3, 2), (4, 2), (7, 3), (8, 3), (16, 4), (31, 5), (32, 5)],
)
def test_sampling_depth_is_logarithmic(length, expected):
    assert ps.sampling_depth(length) == expected
    if length > 1:
        assert ps.sampling_depth(length) == math.ceil(math.log2(length))


def test_depth_far_below_length_at_scale():
    L = 1024
    assert ps.sampling_depth(L) == 10  # vs. L=1024 sequential steps
    # the segment tree really has O(log L) height
    lp = random_logprobs(L, 2, seed=1)
    dfa = parity_dfa(L, 2)
    tree = ps._SegmentTree(ps.transition_matrices(lp, dfa))
    assert tree.height == 10
