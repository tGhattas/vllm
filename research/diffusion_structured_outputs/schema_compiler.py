# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restricted-JSON-schema -> regex -> token DFA, for a fixed-length canvas.

Research POC for vLLM issue #45572. Translates a regular subset of JSON Schema
into a regex (`json_schema_to_regex`) and then into a `finite_automaton.DFA` over
token ids (`build_schema_dfa`) that `constrained_sampler` can enforce.

Supported schema subset (regular ⇒ expressible as a DFA):
  - top-level object with a FIXED, ordered set of properties (regular languages
    cannot express arbitrary key order without blow-up);
  - value types: ``string`` (optional ``enum`` or ``pattern``), ``integer``,
    ``number``, ``boolean``, and ``enum`` of string/number/bool literals.
No nesting/recursion (that is context-free — out of scope; see LOGDEPTH_PLAN /
LAVE, arXiv:2602.00612).

Fixed-length canvas: a real diffusion canvas has a fixed length, so an optional
PAD token is allowed *after* the match completes (a terminal pad-sink), letting a
short JSON object fill an L-slot canvas — exactly how a diffusion decoder pads.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from finite_automaton import DFA
from regex_dfa import regex_to_token_dfa

_META = set(".\\()[]|*+?^{}")


def _esc(s: str) -> str:
    """Backslash-escape regex metacharacters in a literal string."""
    return "".join("\\" + c if c in _META else c for c in s)


def _value_regex(spec: dict[str, Any]) -> str:
    t = spec.get("type")
    if "enum" in spec:
        return "(" + "|".join(_esc(json.dumps(v)) for v in spec["enum"]) + ")"
    if t == "string":
        if "pattern" in spec:
            return '"' + spec["pattern"] + '"'
        return '"[^"]*"'
    if t == "integer":
        return "-?(0|[1-9][0-9]*)"
    if t == "number":
        return "-?(0|[1-9][0-9]*)(\\.[0-9]+)?"
    if t == "boolean":
        return "(true|false)"
    raise ValueError(f"unsupported value spec: {spec}")


def json_schema_to_regex(schema: dict[str, Any]) -> str:
    """Translate a restricted object schema into a regex matching its JSON.

    Property order is taken from ``schema['order']`` if present, else the
    ``properties`` dict order. No insignificant whitespace is allowed (compact
    JSON), which keeps the language regular and small.
    """
    if schema.get("type") != "object":
        raise ValueError("top-level schema must be type 'object'")
    props: dict[str, Any] = schema["properties"]
    order = schema.get("order", list(props.keys()))
    parts = []
    for key in order:
        parts.append(_esc(json.dumps(key)) + ":" + _value_regex(props[key]))
    return "\\{" + ",".join(parts) + "\\}"


def build_schema_dfa(
    schema: dict[str, Any],
    vocab_strings: Sequence[str],
    pad_token_id: int | None = None,
) -> DFA:
    """Compile a schema straight to a token DFA (with optional trailing PAD)."""
    return build_regex_dfa(json_schema_to_regex(schema), vocab_strings, pad_token_id)


def build_regex_dfa(
    pattern: str,
    vocab_strings: Sequence[str],
    pad_token_id: int | None = None,
) -> DFA:
    """Compile a regex to a token DFA, optionally allowing a trailing PAD token.

    With ``pad_token_id`` set, a dedicated accepting pad-sink is appended: after
    the regex matches, only the PAD token is accepted (looping), so a short match
    can fill a fixed-length canvas without corrupting the content.
    """
    dfa = regex_to_token_dfa(pattern, vocab_strings)
    if pad_token_id is None:
        return dfa

    pad_sink = dfa.num_states
    transitions = dict(dfa.transitions)
    for a in dfa.accepting:
        transitions[(a, pad_token_id)] = pad_sink
    transitions[(pad_sink, pad_token_id)] = pad_sink
    return DFA(
        num_states=dfa.num_states + 1,
        start_state=dfa.start_state,
        accepting=set(dfa.accepting) | {pad_sink},
        transitions=transitions,
        vocab_size=dfa.vocab_size,
    )


def decode_canvas(
    token_ids: Sequence[int],
    vocab_strings: Sequence[str],
    pad_token_id: int | None = None,
) -> str:
    """Concatenate token strings, dropping PAD, to recover the emitted text."""
    return "".join(vocab_strings[t] for t in token_ids if t != pad_token_id)
