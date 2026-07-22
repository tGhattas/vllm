# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the schema -> regex -> token-DFA compiler and end-to-end constraint.

Verifies the regex engine against Python `re`, token lifting with multi-char
tokens, and the full path: JSON schema -> DFA -> constrained sampling over a
fixed-length canvas -> decoded string is valid JSON matching the schema.
"""

import itertools
import json

import constrained_sampler as cs
import numpy as np
import pytest
import regex as re
from finite_automaton import DFA
from regex_dfa import char_vocab, regex_to_token_dfa
from schema_compiler import (
    build_schema_dfa,
    decode_canvas,
    json_schema_to_regex,
)

# A single-char JSON vocabulary + a trailing PAD token (empty string).
_CHARS = list('{}[]":,.-') + list("0123456789") + list("abcdefghijklmnopqrstuvwxyz")
VOCAB = _CHARS + [""]
PAD = len(_CHARS)
CID = {c: i for i, c in enumerate(_CHARS)}


@pytest.mark.parametrize(
    "pattern",
    ["[01]+", "a(b|c)*d", "-?[0-9]+", "[a-c]+", "(ab|cd)+", "a.c", "[^0]+", "\\d+"],
)
def test_regex_engine_matches_python_re(pattern):
    alpha = "abcd012"
    vocab = char_vocab(list(alpha))
    cid = {c: i for i, c in enumerate(alpha)}
    dfa = regex_to_token_dfa(pattern, vocab)
    for length in range(0, 5):
        for combo in itertools.product(alpha, repeat=length):
            s = "".join(combo)
            got = dfa.accepts([cid[c] for c in s])
            want = re.fullmatch(pattern, s) is not None
            assert got == want, f"{pattern!r} {s!r}: {got} vs {want}"


def test_multichar_token_lifting():
    # Tokens of length > 1 must be walked char-by-char through the char-DFA.
    vocab = ["ab", "c", "d", "abc"]  # ids 0..3
    dfa = regex_to_token_dfa("(abc)+", vocab)
    # "abc" via one token, or via "ab"+"c"
    assert dfa.accepts([3])  # "abc"
    assert dfa.accepts([0, 1])  # "ab"+"c"
    assert dfa.accepts([3, 3])  # "abcabc"
    assert not dfa.accepts([0])  # "ab" incomplete
    assert not dfa.accepts([1])  # "c" alone


def test_schema_to_regex_roundtrips_with_json():
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}, "n": {"type": "integer"}},
        "order": ["ok", "n"],
    }
    pat = json_schema_to_regex(schema)
    for obj in [{"ok": True, "n": 0}, {"ok": False, "n": -42}, {"ok": True, "n": 7}]:
        s = json.dumps(obj, separators=(",", ":"))
        assert re.fullmatch(pat, s), f"{s} should match {pat}"
    for bad in ['{"ok":true,"n":01}', '{"ok":yes,"n":1}', '{"n":1,"ok":true}']:
        assert not re.fullmatch(pat, bad)


def _sample_valid(schema, L, seeds=25):
    dfa = build_schema_dfa(schema, VOCAB, pad_token_id=PAD)
    V = len(VOCAB)
    rng = np.random.default_rng(0)
    lp = cs.logits_to_logprobs(rng.normal(size=(L, V)))
    results = []
    for s in range(seeds):
        seq, _ = cs.sample(lp, dfa, np.random.default_rng(s))
        assert dfa.accepts(seq)  # accepted by construction
        results.append(decode_canvas(seq, VOCAB, PAD))
    gseq, _ = cs.greedy(lp, dfa)
    results.append(decode_canvas(gseq, VOCAB, PAD))
    return results


def test_end_to_end_boolean_schema():
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "order": ["ok"],
    }
    for text in _sample_valid(schema, L=14):
        obj = json.loads(text)  # must be valid JSON
        assert set(obj) == {"ok"} and isinstance(obj["ok"], bool)


def test_end_to_end_integer_schema():
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "order": ["n"],
    }
    for text in _sample_valid(schema, L=11):
        obj = json.loads(text)
        assert set(obj) == {"n"} and isinstance(obj["n"], int)


def test_end_to_end_enum_schema():
    schema = {
        "type": "object",
        "properties": {"color": {"enum": ["red", "green"]}},
        "order": ["color"],
    }
    for text in _sample_valid(schema, L=16):
        obj = json.loads(text)
        assert obj["color"] in ("red", "green")


def test_pad_fills_canvas_but_content_is_prefix():
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "order": ["ok"],
    }
    dfa = build_schema_dfa(schema, VOCAB, pad_token_id=PAD)
    lp = cs.logits_to_logprobs(np.random.default_rng(1).normal(size=(14, len(VOCAB))))
    seq, _ = cs.greedy(lp, dfa)
    # once PAD starts it never returns to content
    first_pad = next((i for i, t in enumerate(seq) if t == PAD), len(seq))
    assert all(t == PAD for t in seq[first_pad:])
    assert json.loads(decode_canvas(seq, VOCAB, PAD))["ok"] in (True, False)


def test_impossible_when_canvas_too_short():
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "order": ["ok"],
    }
    # shortest match is `{"ok":true}` = 11 tokens; a length-5 canvas can't match
    dfa = build_schema_dfa(schema, VOCAB, pad_token_id=PAD)
    lp = cs.logits_to_logprobs(np.random.default_rng(2).normal(size=(5, len(VOCAB))))
    assert cs.log_partition(lp, dfa) == float("-inf")
    with pytest.raises(cs.ImpossibleConstraintError):
        cs.sample(lp, dfa, np.random.default_rng(0))


def test_dfa_uses_finite_automaton_type():
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    dfa = build_schema_dfa(schema, VOCAB, pad_token_id=PAD)
    assert isinstance(dfa, DFA)
    assert dfa.vocab_size == len(VOCAB)
