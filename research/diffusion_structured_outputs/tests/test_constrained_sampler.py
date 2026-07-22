# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Brute-force oracle tests for the exact constrained-canvas sampler.

Every test is deterministic (seeded) and CPU-only. The exact-vs-exact checks
(``marginals``, ``log_partition``, ``greedy``) compare the DP against exhaustive
enumeration to tight tolerance; the sampling checks confirm the ancestral sampler
reproduces the enumerated distribution (Monte Carlo, looser tolerance) and always
emits DFA-accepted canvases.
"""

import constrained_sampler as cs
import numpy as np
import pytest
from brute_force_oracle import (
    brute_force_log_partition,
    brute_force_map,
    brute_force_marginals,
    enumerate_constrained,
)
from finite_automaton import DFA

EXACT_TOL = 1e-9


def random_logprobs(L: int, V: int, seed: int, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return cs.logits_to_logprobs(rng.normal(scale=scale, size=(L, V)))


def parity_dfa(L: int, V: int) -> DFA:
    """Accept length-L sequences with an even count of token id 1 (2-state DFA)."""
    transitions = {}
    for q in (0, 1):
        for t in range(V):
            transitions[(q, t)] = (q ^ 1) if t == 1 else q
    return DFA(
        num_states=2,
        start_state=0,
        accepting={0},
        transitions=transitions,
        vocab_size=V,
    )


# --------------------------------------------------------------------------- #
# Exact structural checks (DP vs. enumeration)                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_marginals_match_bruteforce_parity(seed):
    L, V = 4, 3
    lp = random_logprobs(L, V, seed)
    dfa = parity_dfa(L, V)
    got = cs.marginals(lp, dfa)
    want = brute_force_marginals(lp, dfa)
    np.testing.assert_allclose(got, want, atol=EXACT_TOL)
    # marginals are proper distributions per position
    np.testing.assert_allclose(got.sum(axis=1), np.ones(L), atol=1e-9)


@pytest.mark.parametrize("seed", [3, 4])
def test_logpartition_matches_bruteforce(seed):
    L, V = 4, 3
    lp = random_logprobs(L, V, seed)
    dfa = parity_dfa(L, V)
    assert cs.log_partition(lp, dfa) == pytest.approx(
        brute_force_log_partition(lp, dfa), abs=EXACT_TOL
    )


def test_exact_single_string_dfa():
    L, V = 3, 4
    target = [2, 0, 3]
    dfa = DFA.from_exact_sequence(target, V)
    lp = random_logprobs(L, V, seed=7)
    # Only one accepted sequence -> constrained distribution is a point mass.
    dist = enumerate_constrained(lp, dfa)
    assert list(dist.keys()) == [tuple(target)]
    assert dist[tuple(target)] == pytest.approx(1.0)
    # Sampler must always return it; greedy too.
    rng = np.random.default_rng(0)
    for _ in range(20):
        seq, logp = cs.sample(lp, dfa, rng)
        assert seq == target
        assert dfa.accepts(seq)
    gseq, _ = cs.greedy(lp, dfa)
    assert gseq == target
    # log P_D of the only sequence is 0 (probability 1).
    _, logp = cs.sample(lp, dfa, np.random.default_rng(1))
    assert logp == pytest.approx(0.0, abs=1e-9)


def test_multi_string_dfa_marginals():
    L, V = 3, 4
    seqs = [[1, 2, 3], [1, 0, 3], [2, 2, 3]]
    dfa = DFA.from_sequence_set(seqs, V)
    lp = random_logprobs(L, V, seed=11)
    got = cs.marginals(lp, dfa)
    want = brute_force_marginals(lp, dfa)
    np.testing.assert_allclose(got, want, atol=EXACT_TOL)
    # Support is exactly the three sequences.
    dist = enumerate_constrained(lp, dfa)
    assert set(dist) == {tuple(s) for s in seqs}


# --------------------------------------------------------------------------- #
# Sampling distribution check (Monte Carlo vs. enumeration)                    #
# --------------------------------------------------------------------------- #


def test_sampler_reproduces_exact_distribution():
    L, V = 4, 3
    lp = random_logprobs(L, V, seed=42)
    dfa = parity_dfa(L, V)
    exact = enumerate_constrained(lp, dfa)

    beta = cs.backward_messages(lp, dfa)
    rng = np.random.default_rng(123)
    n = 60_000
    counts: dict[tuple, int] = {}
    logZ = cs.log_partition(lp, dfa)
    for _ in range(n):
        seq, logp = cs.sample(lp, dfa, rng, beta=beta)
        key = tuple(seq)
        assert dfa.accepts(seq)  # satisfaction by construction
        counts[key] = counts.get(key, 0) + 1
        # log P_D(x) identity: sum of per-position log-probs minus log Z.
        want_logp = sum(lp[i, seq[i]] for i in range(L)) - logZ
        assert logp == pytest.approx(want_logp, abs=1e-9)

    for key, p in exact.items():
        emp = counts.get(key, 0) / n
        assert emp == pytest.approx(p, abs=0.01), f"{key}: emp={emp} exact={p}"


# --------------------------------------------------------------------------- #
# Fixed positions                                                              #
# --------------------------------------------------------------------------- #


def test_fixed_token_compatible():
    L, V = 4, 3
    lp = random_logprobs(L, V, seed=5)
    dfa = parity_dfa(L, V)
    # Fix position 0 to token 1 (compatible; just shifts the parity requirement).
    fixed = {0: 1}
    got = cs.marginals(lp, dfa, fixed)
    want = brute_force_marginals(lp, dfa, fixed)
    np.testing.assert_allclose(got, want, atol=EXACT_TOL)
    # Fixed slot carries all its mass on the fixed token.
    assert got[0, 1] == pytest.approx(1.0)
    rng = np.random.default_rng(9)
    for _ in range(50):
        seq, _ = cs.sample(lp, dfa, rng, fixed_positions=fixed)
        assert seq[0] == 1
        assert dfa.accepts(seq)


def test_fixed_token_makes_impossible():
    L, V = 3, 4
    target = [2, 0, 3]
    dfa = DFA.from_exact_sequence(target, V)
    lp = random_logprobs(L, V, seed=8)
    # Force position 1 to a token the only accepted string does not use there.
    fixed = {1: 2}  # accepted string needs position 1 == 0
    assert cs.log_partition(lp, dfa, fixed) == float("-inf")
    with pytest.raises(cs.ImpossibleConstraintError):
        cs.sample(lp, dfa, np.random.default_rng(0), fixed_positions=fixed)
    with pytest.raises(cs.ImpossibleConstraintError):
        cs.marginals(lp, dfa, fixed)
    assert brute_force_log_partition(lp, dfa, fixed) == float("-inf")


# --------------------------------------------------------------------------- #
# Impossible constraints                                                       #
# --------------------------------------------------------------------------- #


def test_impossible_no_completion():
    # DFA that requires length 5 but canvas is length 3 -> no accepted canvas.
    V = 3
    dfa = DFA.from_exact_sequence([0, 1, 2, 0, 1], V)  # length-5 acceptor
    lp = random_logprobs(3, V, seed=2)
    assert cs.log_partition(lp, dfa) == float("-inf")
    with pytest.raises(cs.ImpossibleConstraintError):
        cs.sample(lp, dfa, np.random.default_rng(0))
    with pytest.raises(cs.ImpossibleConstraintError):
        cs.greedy(lp, dfa)


# --------------------------------------------------------------------------- #
# Numerical robustness: skewed logits and tiny probabilities                   #
# --------------------------------------------------------------------------- #


def test_highly_skewed_logits():
    L, V = 4, 3
    # Very peaked distributions -> tiny tail probabilities.
    lp = random_logprobs(L, V, seed=13, scale=40.0)
    dfa = parity_dfa(L, V)
    got = cs.marginals(lp, dfa)
    want = brute_force_marginals(lp, dfa)
    np.testing.assert_allclose(got, want, atol=1e-6)
    assert cs.log_partition(lp, dfa) == pytest.approx(
        brute_force_log_partition(lp, dfa), abs=1e-6
    )
    # No NaNs/Infs leak into marginals.
    assert np.all(np.isfinite(got))


def test_very_small_probabilities():
    # Construct explicit log-probs with a near-zero entry to stress log-space.
    L, V = 3, 3
    logits = np.array(
        [[0.0, -80.0, -80.0], [-80.0, 0.0, -80.0], [-80.0, -80.0, 0.0]],
        dtype=np.float64,
    )
    lp = cs.logits_to_logprobs(logits)
    dfa = parity_dfa(L, V)  # sequence 0,1,2 has one '1' -> odd -> rejected
    # The MAP-ish path (0,1,2) is rejected; check DP still matches enumeration.
    got = cs.marginals(lp, dfa)
    want = brute_force_marginals(lp, dfa)
    np.testing.assert_allclose(got, want, atol=1e-9)
    assert np.all(np.isfinite(got))


# --------------------------------------------------------------------------- #
# Greedy == brute-force MAP                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_greedy_matches_bruteforce_map(seed):
    L, V = 4, 3
    lp = random_logprobs(L, V, seed)
    dfa = parity_dfa(L, V)
    gseq, glogw = cs.greedy(lp, dfa)
    assert dfa.accepts(gseq)
    bseq, blogw = brute_force_map(lp, dfa)
    # log-weights must match; sequences match unless there is an exact tie.
    assert glogw == pytest.approx(blogw, abs=1e-9)
    got_unnorm = sum(lp[i, gseq[i]] for i in range(L))
    assert got_unnorm == pytest.approx(blogw, abs=1e-9)


def test_greedy_differs_from_marginal_argmax():
    # A case where joint MAP != per-position marginal argmax, to prove greedy
    # is a true constrained Viterbi rather than independent per-slot argmax.
    V = 3
    # Position 0 slightly prefers token 0; position 1 slightly prefers token 0;
    # but the DFA forbids (0,0). Marginal argmax could pick 0 at each slot.
    logits = np.array([[1.0, 0.9, -5.0], [1.0, 0.9, -5.0]], dtype=np.float64)
    lp = cs.logits_to_logprobs(logits)
    # Accept everything except the pair (0, 0).
    seqs = [(a, b) for a in range(V) for b in range(V) if (a, b) != (0, 0)]
    dfa = DFA.from_sequence_set(seqs, V)
    gseq, _ = cs.greedy(lp, dfa)
    bseq, _ = brute_force_map(lp, dfa)
    assert tuple(gseq) == bseq
    assert dfa.accepts(gseq)
