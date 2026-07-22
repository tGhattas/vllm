# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the batched torch log-depth sampler against the numpy reference.

Runs on CPU-torch here; the same code runs on GPU (see scripts/bench_gpu_sampler.py).
Checks per-batch logZ equals the numpy sampler, every batched sample is
DFA-accepted, empirical marginals match, and the schema path yields valid JSON.
"""

import json

import constrained_sampler as cs
import gpu_constrained_sampler as gs
import numpy as np
import pytest
import torch
from schema_compiler import build_schema_dfa, decode_canvas
from test_constrained_sampler import parity_dfa, random_logprobs
from test_schema_compiler import PAD, VOCAB

torch.manual_seed(0)


def test_gpu_logpartition_matches_numpy_per_batch():
    L, V, B = 5, 3, 6
    dfa = parity_dfa(L, V)
    nxt, acc, start = gs.build_transition_table(dfa)
    lps = [random_logprobs(L, V, seed=s) for s in range(B)]
    lp_t = torch.tensor(np.stack(lps), dtype=torch.float64)
    levels = gs.build_levels(gs.transition_matrices(lp_t, nxt))
    z_gpu = gs.log_partition(levels, start, acc)
    for b in range(B):
        assert float(z_gpu[b]) == pytest.approx(cs.log_partition(lps[b], dfa), abs=1e-9)


def test_gpu_samples_accepted_and_marginals_match():
    L, V, B = 4, 3, 6000
    dfa = parity_dfa(L, V)
    nxt, acc, start = gs.build_transition_table(dfa)
    lp = random_logprobs(L, V, seed=17)
    lp_t = torch.tensor(np.broadcast_to(lp, (B, L, V)).copy(), dtype=torch.float64)
    gen = torch.Generator().manual_seed(1)
    tokens = gs.sample(lp_t, nxt, acc, start, gen).numpy()  # [B, L]

    assert all(dfa.accepts(list(row)) for row in tokens[:200])  # spot-check acceptance
    emp = np.zeros((L, V))
    for row in tokens:
        for i in range(L):
            emp[i, row[i]] += 1
    emp /= B
    np.testing.assert_allclose(emp, cs.marginals(lp, dfa), atol=0.02)


def test_gpu_matches_single_string_dfa():
    from finite_automaton import DFA

    target = [2, 0, 3, 1]
    dfa = DFA.from_exact_sequence(target, 4)
    nxt, acc, start = gs.build_transition_table(dfa)
    lp = random_logprobs(4, 4, seed=3)
    lp_t = torch.tensor(np.broadcast_to(lp, (16, 4, 4)).copy(), dtype=torch.float64)
    tokens = gs.sample(lp_t, nxt, acc, start, torch.Generator().manual_seed(0))
    for row in tokens.tolist():
        assert row == target


def test_gpu_schema_end_to_end_valid_json():
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "order": ["ok"],
    }
    L, B = 14, 64
    dfa = build_schema_dfa(schema, VOCAB, pad_token_id=PAD)
    nxt, acc, start = gs.build_transition_table(dfa)
    lp = cs.logits_to_logprobs(np.random.default_rng(0).normal(size=(L, len(VOCAB))))
    lp_t = torch.tensor(
        np.broadcast_to(lp, (B, L, len(VOCAB))).copy(), dtype=torch.float64
    )
    tokens = gs.sample(lp_t, nxt, acc, start, torch.Generator().manual_seed(0))
    for row in tokens.tolist():
        assert dfa.accepts(row)
        obj = json.loads(decode_canvas(row, VOCAB, PAD))
        assert isinstance(obj["ok"], bool)


def test_gpu_impossible_raises():
    from finite_automaton import DFA

    dfa = DFA.from_exact_sequence([0, 1, 2, 0], 3)  # length-4 acceptor
    nxt, acc, start = gs.build_transition_table(dfa)
    lp = random_logprobs(3, 3, seed=2)
    lp_t = torch.tensor(lp[None], dtype=torch.float64)
    with pytest.raises(ValueError):
        gs.sample(lp_t, nxt, acc, start)
