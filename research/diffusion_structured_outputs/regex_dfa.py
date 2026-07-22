# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile a regular expression to a token-level DFA (xgrammar-style, minimal).

Research POC for vLLM issue #45572. Turns a schema's regular constraint into the
`finite_automaton.DFA` over *token ids* that `constrained_sampler` consumes:

    regex --(Thompson)--> NFA --(lazy subset construction)--> char-DFA
          --(token lifting over a vocabulary)--> token-level DFA

Supported regex subset (enough for a regular JSON value subset): literals, `.`,
escapes (`\\d \\w \\s \\D \\W \\S` and escaped metacharacters), character classes
`[...]` / `[^...]` with ranges, groups `(...)`, alternation `|`, and quantifiers
`*`, `+`, `?`. No `{m,n}`, backreferences, or anchors.

Token lifting: for each char-DFA state and each vocabulary token, walk the token's
characters through the char-DFA; if it never dies, add that token transition. A
token-DFA state is accepting iff its char-DFA state is regex-accepting (i.e. the
committed token string so far fully matches the regex).

This is the model-agnostic "schema compiler" seam; a production path would reuse
xgrammar's compiled representation instead (see `design.md` §10).
"""

from __future__ import annotations

import string
from collections.abc import Callable, Sequence

from finite_automaton import DFA

_WORD = set(string.ascii_letters + string.digits + "_")
_SPACE = set(" \t\n\r\f\v")
_DIGIT = set(string.digits)


# --------------------------------------------------------------------------- #
# NFA with epsilon edges and character-predicate edges (Thompson construction) #
# --------------------------------------------------------------------------- #


class _NFA:
    def __init__(self) -> None:
        self.eps: list[set[int]] = []
        self.sym: list[list[tuple[Callable[[str], bool], int]]] = []
        self.start = 0
        self.accept = 0

    def new_state(self) -> int:
        self.eps.append(set())
        self.sym.append([])
        return len(self.eps) - 1

    def add_eps(self, s: int, t: int) -> None:
        self.eps[s].add(t)

    def add_sym(self, s: int, pred: Callable[[str], bool], t: int) -> None:
        self.sym[s].append((pred, t))

    def eps_closure(self, states: frozenset[int]) -> frozenset[int]:
        stack = list(states)
        seen = set(states)
        while stack:
            s = stack.pop()
            for t in self.eps[s]:
                if t not in seen:
                    seen.add(t)
                    stack.append(t)
        return frozenset(seen)

    def move(self, states: frozenset[int], ch: str) -> frozenset[int]:
        out: set[int] = set()
        for s in states:
            for pred, t in self.sym[s]:
                if pred(ch):
                    out.add(t)
        return self.eps_closure(frozenset(out))


class _Parser:
    """Recursive-descent regex parser producing Thompson NFA fragments."""

    def __init__(self, pattern: str) -> None:
        self.p = pattern
        self.i = 0
        self.nfa = _NFA()

    def _peek(self) -> str | None:
        return self.p[self.i] if self.i < len(self.p) else None

    def _next(self) -> str:
        ch = self.p[self.i]
        self.i += 1
        return ch

    def parse(self) -> _NFA:
        start, accept = self._alt()
        if self.i != len(self.p):
            raise ValueError(f"unexpected char at {self.i}: {self.p[self.i]!r}")
        self.nfa.start = start
        self.nfa.accept = accept
        return self.nfa

    def _alt(self) -> tuple[int, int]:
        frags = [self._concat()]
        while self._peek() == "|":
            self._next()
            frags.append(self._concat())
        if len(frags) == 1:
            return frags[0]
        s = self.nfa.new_state()
        a = self.nfa.new_state()
        for fs, fa in frags:
            self.nfa.add_eps(s, fs)
            self.nfa.add_eps(fa, a)
        return s, a

    def _concat(self) -> tuple[int, int]:
        frags = []
        while self._peek() not in (None, "|", ")"):
            frags.append(self._repeat())
        if not frags:
            s = self.nfa.new_state()  # empty match
            return s, s
        for i in range(len(frags) - 1):
            self.nfa.add_eps(frags[i][1], frags[i + 1][0])
        return frags[0][0], frags[-1][1]

    def _repeat(self) -> tuple[int, int]:
        s, a = self._atom()
        while self._peek() in ("*", "+", "?"):
            q = self._next()
            ns = self.nfa.new_state()
            na = self.nfa.new_state()
            self.nfa.add_eps(ns, s)
            self.nfa.add_eps(a, na)
            if q in ("*", "?"):
                self.nfa.add_eps(ns, na)  # skip
            if q in ("*", "+"):
                self.nfa.add_eps(a, s)  # repeat
            s, a = ns, na
        return s, a

    def _atom(self) -> tuple[int, int]:
        ch = self._peek()
        if ch == "(":
            self._next()
            frag = self._alt()
            if self._peek() != ")":
                raise ValueError("unbalanced '('")
            self._next()
            return frag
        if ch == "[":
            return self._char_class()
        if ch == ".":
            self._next()
            return self._sym_frag(lambda c: c != "\n")
        if ch == "\\":
            self._next()
            return self._sym_frag(self._escape(self._next()))
        if ch in (")", "|", "*", "+", "?"):
            raise ValueError(f"unexpected {ch!r} at {self.i}")
        self._next()
        return self._sym_frag(lambda c, lit=ch: c == lit)

    def _sym_frag(self, pred: Callable[[str], bool]) -> tuple[int, int]:
        s = self.nfa.new_state()
        a = self.nfa.new_state()
        self.nfa.add_sym(s, pred, a)
        return s, a

    def _escape(self, e: str) -> Callable[[str], bool]:
        table = {
            "d": lambda c: c in _DIGIT,
            "D": lambda c: c not in _DIGIT,
            "w": lambda c: c in _WORD,
            "W": lambda c: c not in _WORD,
            "s": lambda c: c in _SPACE,
            "S": lambda c: c not in _SPACE,
            "n": lambda c: c == "\n",
            "t": lambda c: c == "\t",
        }
        if e in table:
            return table[e]
        return lambda c, lit=e: c == lit  # escaped literal

    def _char_class(self) -> tuple[int, int]:
        self._next()  # consume '['
        negate = False
        if self._peek() == "^":
            negate = True
            self._next()
        singles: set[str] = set()
        ranges: list[tuple[str, str]] = []
        preds: list[Callable[[str], bool]] = []
        while self._peek() not in (None, "]"):
            c = self._next()
            if c == "\\":
                preds.append(self._escape(self._next()))
                continue
            if (
                self._peek() == "-"
                and self.i + 1 < len(self.p)
                and self.p[self.i + 1] != "]"
            ):
                self._next()  # consume '-'
                hi = self._next()
                ranges.append((c, hi))
            else:
                singles.add(c)
        if self._peek() != "]":
            raise ValueError("unbalanced '['")
        self._next()

        def base(ch: str) -> bool:
            if ch in singles:
                return True
            if any(lo <= ch <= hi for lo, hi in ranges):
                return True
            return any(p(ch) for p in preds)

        pred = (lambda ch: not base(ch)) if negate else base
        return self._sym_frag(pred)


# --------------------------------------------------------------------------- #
# regex -> token-level DFA                                                      #
# --------------------------------------------------------------------------- #


def regex_to_token_dfa(
    pattern: str,
    vocab_strings: Sequence[str],
    max_states: int = 100_000,
) -> DFA:
    """Compile ``pattern`` into a `DFA` over token ids (index into ``vocab_strings``).

    A token sequence is accepted iff its concatenation fully matches ``pattern``.
    """
    nfa = _Parser(pattern).parse()
    V = len(vocab_strings)

    start_set = nfa.eps_closure(frozenset({nfa.start}))
    state_id: dict[frozenset[int], int] = {start_set: 0}
    order: list[frozenset[int]] = [start_set]
    transitions: dict[tuple[int, int], int] = {}

    def walk_token(cur: frozenset[int], tok: str) -> frozenset[int] | None:
        for ch in tok:
            cur = nfa.move(cur, ch)
            if not cur:
                return None
        return cur

    i = 0
    while i < len(order):
        cur = order[i]
        cur_id = i
        for tid in range(V):
            tok = vocab_strings[tid]
            if tok == "":
                continue
            dest = walk_token(cur, tok)
            if dest is None:
                continue
            if dest not in state_id:
                if len(order) >= max_states:
                    raise ValueError("token-DFA exceeded max_states")
                state_id[dest] = len(order)
                order.append(dest)
            transitions[(cur_id, tid)] = state_id[dest]
        i += 1

    accepting = {sid for st, sid in state_id.items() if nfa.accept in st}
    return DFA(
        num_states=len(order),
        start_state=0,
        accepting=accepting,
        transitions=transitions,
        vocab_size=V,
    )


def char_vocab(chars: Sequence[str]) -> list[str]:
    """A single-character-per-token vocabulary (token id == index)."""
    return list(chars)
