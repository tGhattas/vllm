# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canvas-aware constrained decoding for diffusion LMs (experimental, opt-in).

Given a diffusion model's per-position logits over a fixed canvas and a token-level
DFA, this computes the **constrained mean-field marginals** ``P_D(x_i = v)`` — the
per-position distribution conditioned on the whole canvas being accepted by the DFA
(Dang & Ermon, arXiv:2607.07026) — via a numerically stable log-space
forward/backward pass. Replacing the sampler's logits with these marginals steers
every denoising step toward a DFA-valid canvas.

This is loaded only when ``VLLM_DIFFUSION_CONSTRAINT`` points at a precompiled
constraint file (built offline from a schema/regex + the model tokenizer), so the
default diffusion path and the request-time structured-output guard are untouched.

The DFA is a partial function: ``next_state[s, v] == -1`` means token ``v`` is not
allowed from state ``s``. Only defined edges carry probability mass; every other
token gets ``-inf`` (never sampled).
"""

from __future__ import annotations

import string
from collections.abc import Callable, Sequence

import torch

NEG_INF = float("-inf")

# --------------------------------------------------------------------------- #
# Minimal regex -> token-DFA compiler (self-contained; regular subset only).   #
# Thompson NFA -> lazy subset construction -> token lifting over a vocabulary.  #
# --------------------------------------------------------------------------- #

_WORD = set(string.ascii_letters + string.digits + "_")
_SPACE = set(" \t\n\r\f\v")
_DIGIT = set(string.digits)
_META = set(".\\()[]|*+?^{}")


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

    def eps_closure(self, states: frozenset[int]) -> frozenset[int]:
        stack, seen = list(states), set(states)
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
    """Recursive-descent regex parser (literals, ., escapes, classes, |, */+/?)."""

    def __init__(self, pattern: str) -> None:
        self.p, self.i, self.nfa = pattern, 0, _NFA()

    def _peek(self):
        return self.p[self.i] if self.i < len(self.p) else None

    def _next(self):
        c = self.p[self.i]
        self.i += 1
        return c

    def parse(self) -> _NFA:
        s, a = self._alt()
        if self.i != len(self.p):
            raise ValueError(f"bad regex at {self.i}")
        self.nfa.start, self.nfa.accept = s, a
        return self.nfa

    def _alt(self):
        frags = [self._concat()]
        while self._peek() == "|":
            self._next()
            frags.append(self._concat())
        if len(frags) == 1:
            return frags[0]
        s, a = self.nfa.new_state(), self.nfa.new_state()
        for fs, fa in frags:
            self.nfa.eps[s].add(fs)
            self.nfa.eps[fa].add(a)
        return s, a

    def _concat(self):
        frags = []
        while self._peek() not in (None, "|", ")"):
            frags.append(self._repeat())
        if not frags:
            s = self.nfa.new_state()
            return s, s
        for i in range(len(frags) - 1):
            self.nfa.eps[frags[i][1]].add(frags[i + 1][0])
        return frags[0][0], frags[-1][1]

    def _repeat(self):
        s, a = self._atom()
        while self._peek() in ("*", "+", "?"):
            q = self._next()
            ns, na = self.nfa.new_state(), self.nfa.new_state()
            self.nfa.eps[ns].add(s)
            self.nfa.eps[a].add(na)
            if q in ("*", "?"):
                self.nfa.eps[ns].add(na)
            if q in ("*", "+"):
                self.nfa.eps[a].add(s)
            s, a = ns, na
        return s, a

    def _atom(self):
        c = self._peek()
        if c == "(":
            self._next()
            frag = self._alt()
            if self._peek() != ")":
                raise ValueError("unbalanced (")
            self._next()
            return frag
        if c == "[":
            return self._char_class()
        if c == ".":
            self._next()
            return self._sym(lambda ch: ch != "\n")
        if c == "\\":
            self._next()
            return self._sym(self._escape(self._next()))
        if c in (")", "|", "*", "+", "?"):
            raise ValueError(f"unexpected {c}")
        self._next()
        return self._sym(lambda ch, lit=c: ch == lit)

    def _sym(self, pred):
        s, a = self.nfa.new_state(), self.nfa.new_state()
        self.nfa.sym[s].append((pred, a))
        return s, a

    def _escape(self, e):
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
        return table.get(e, lambda c, lit=e: c == lit)

    def _char_class(self):
        self._next()
        negate = self._peek() == "^"
        if negate:
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
                self._next()
                ranges.append((c, self._next()))
            else:
                singles.add(c)
        if self._peek() != "]":
            raise ValueError("unbalanced [")
        self._next()

        def base(ch: str) -> bool:
            return (
                ch in singles
                or any(lo <= ch <= hi for lo, hi in ranges)
                or any(p(ch) for p in preds)
            )

        return self._sym((lambda ch: not base(ch)) if negate else base)


def _regex_charset(pat: str) -> set[str] | None:
    """Characters a regex can match, or None if unbounded (`.`/`[^..]`/`\\w`...)."""
    chars: set[str] = set()
    i = 0
    while i < len(pat):
        c = pat[i]
        if c == "\\":
            n = pat[i + 1]
            if n == "d":
                chars |= _DIGIT
            elif n in "wsWSD":
                return None
            else:
                chars.add(n)
            i += 2
            continue
        if c == ".":
            return None
        if c == "[":
            j = pat.index("]", i)
            cls = pat[i + 1 : j]
            if cls.startswith("^"):
                return None
            k = 0
            while k < len(cls):
                if k + 2 < len(cls) and cls[k + 1] == "-":
                    chars |= {chr(o) for o in range(ord(cls[k]), ord(cls[k + 2]) + 1)}
                    k += 3
                else:
                    chars.add(cls[k])
                    k += 1
            i = j + 1
            continue
        if c not in "(){}|*+?":
            chars.add(c)
        i += 1
    return chars


def _relevant_tokens(tokenizer, charset: set[str]) -> tuple[list[int], list[str]]:
    """Vocab tokens whose surface is non-empty and within `charset` (exact prune).

    Two-pass so the (per-token) ``convert_tokens_to_string`` is called only on the
    few candidates, not the whole vocab: pass 1 filters raw vocab pieces cheaply
    (mapping the byte/SentencePiece space markers to a space); pass 2 decodes the
    candidates for an exact surface + re-check. This keeps compile time bounded on
    a 250k+ vocab.
    """
    allowed = charset | {" "}  # a marker-mapped leading space is fine to pre-pass
    cand = []
    for piece, tid in tokenizer.get_vocab().items():
        approx = piece.replace("▁", " ").replace("Ġ", " ").replace("Ċ", "\n")
        if approx and all(ch in allowed for ch in approx):
            cand.append((int(tid), piece))
    rel = []
    for tid, piece in cand:
        s = tokenizer.convert_tokens_to_string([piece])
        if s and all(ch in charset for ch in s):
            rel.append((tid, s))
    rel.sort()
    return [t for t, _ in rel], [s for _, s in rel]


def choice_to_regex(choices: Sequence[str]) -> str:
    """A `choice` list becomes an alternation of its escaped literals."""
    return "(" + "|".join(_escape_literal(c) for c in choices) + ")"


def _escape_literal(s: str) -> str:
    return "".join("\\" + c if c in _META else c for c in s)


def _compile_regex_to_edges(
    regex: str, vocab_strings: Sequence[str], pad_reduced: int | None
):
    """regex -> token DFA over `vocab_strings`; returns (num_states, {(s,t):d}, acc)."""
    nfa = _Parser(regex).parse()
    start_set = nfa.eps_closure(frozenset({nfa.start}))
    state_id = {start_set: 0}
    order = [start_set]
    transitions: dict[tuple[int, int], int] = {}
    i = 0
    while i < len(order):
        cur = order[i]
        for tid, tok in enumerate(vocab_strings):
            if not tok:
                continue
            dest = cur
            for ch in tok:
                dest = nfa.move(dest, ch)
                if not dest:
                    dest = None
                    break
            if dest is None:
                continue
            if dest not in state_id:
                state_id[dest] = len(order)
                order.append(dest)
            transitions[(i, tid)] = state_id[dest]
        i += 1
    accepting = {sid for st, sid in state_id.items() if nfa.accept in st}
    n = len(order)
    if pad_reduced is not None:  # trailing PAD sink to fill a fixed canvas
        sink = n
        for a in accepting:
            transitions[(a, pad_reduced)] = sink
        transitions[(sink, pad_reduced)] = sink
        accepting = accepting | {sink}
        n += 1
    return n, transitions, accepting


class DiffusionConstraint:
    """A token-DFA constraint that reweights canvas logits to its accepted language."""

    def __init__(
        self,
        num_states: int,
        vocab_size: int,
        start: int,
        accepting: torch.Tensor,  # [N] bool
        edge_s: torch.Tensor,  # [E] source states
        edge_v: torch.Tensor,  # [E] tokens
        edge_d: torch.Tensor,  # [E] destination states
    ) -> None:
        self.N = int(num_states)
        self.V = int(vocab_size)
        self.start = int(start)
        self.accepting = accepting.to(torch.bool)
        self.edge_s = edge_s.long().contiguous()
        self.edge_v = edge_v.long().contiguous()
        self.edge_d = edge_d.long().contiguous()
        # Relevant token columns (edges touch far fewer than V tokens); marginals
        # are computed over these and scattered back, so the hot path never
        # materializes O(B·L·V) intermediates.
        self.rel_v, self.edge_v_local = torch.unique(self.edge_v, return_inverse=True)

    @classmethod
    def from_next_state(cls, next_state, accepting, start) -> DiffusionConstraint:
        n, v = next_state.shape
        s_idx, v_idx = (next_state >= 0).nonzero(as_tuple=True)
        return cls(n, v, start, accepting, s_idx, v_idx, next_state[s_idx, v_idx])

    @classmethod
    def from_file(cls, path: str, device="cpu") -> DiffusionConstraint:
        d = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            d["num_states"],
            d["vocab_size"],
            d["start"],
            d["accepting"],
            d["edge_s"],
            d["edge_v"],
            d["edge_d"],
        ).to(device)

    @classmethod
    def from_regex(
        cls, regex: str, tokenizer, vocab_size: int, pad_token_id: int | None = None
    ) -> DiffusionConstraint:
        """Compile a regex + tokenizer into a per-token-id constraint.

        PAD (the model's EOS by default) fills the fixed canvas after the match.
        Only the regular subset is supported (no unbounded `.`/`[^..]`).
        """
        cset = _regex_charset(regex)
        if cset is None:
            raise ValueError(f"regex is not a bounded regular language: {regex!r}")
        allowed, strings = _relevant_tokens(tokenizer, cset)
        pad = pad_token_id if pad_token_id is not None else tokenizer.eos_token_id
        if pad is None:
            raise ValueError("no pad/eos token id for canvas padding")
        allowed = allowed + [int(pad)]
        strings = strings + [""]
        pad_reduced = len(allowed) - 1
        n, transitions, accepting = _compile_regex_to_edges(regex, strings, pad_reduced)
        edge_s, edge_v, edge_d = [], [], []
        for (s, rv), d in transitions.items():
            edge_s.append(s)
            edge_v.append(int(allowed[rv]))
            edge_d.append(d)
        acc = torch.zeros(n, dtype=torch.bool)
        for s in accepting:
            acc[s] = True
        return cls(
            n,
            vocab_size,
            0,
            acc,
            torch.tensor(edge_s, dtype=torch.long),
            torch.tensor(edge_v, dtype=torch.long),
            torch.tensor(edge_d, dtype=torch.long),
        )

    @classmethod
    def from_choice(
        cls, choices, tokenizer, vocab_size: int, pad_token_id: int | None = None
    ) -> DiffusionConstraint:
        return cls.from_regex(
            choice_to_regex(choices), tokenizer, vocab_size, pad_token_id
        )

    @classmethod
    def from_structured_outputs(
        cls, structured_outputs, tokenizer, vocab_size: int
    ) -> DiffusionConstraint | None:
        """Build from a SamplingParams.structured_outputs (regex/choice only)."""
        if structured_outputs is None:
            return None
        if structured_outputs.regex is not None:
            return cls.from_regex(structured_outputs.regex, tokenizer, vocab_size)
        if structured_outputs.choice is not None:
            return cls.from_choice(structured_outputs.choice, tokenizer, vocab_size)
        return None

    def to(self, device) -> DiffusionConstraint:
        self.accepting = self.accepting.to(device)
        self.edge_s = self.edge_s.to(device)
        self.edge_v = self.edge_v.to(device)
        self.edge_d = self.edge_d.to(device)
        self.rel_v = self.rel_v.to(device)
        self.edge_v_local = self.edge_v_local.to(device)
        return self

    def _transition_matrices(self, logp_rel: torch.Tensor) -> torch.Tensor:
        """M[b,i,s,s'] = logsumexp over tokens v (s->s') of logp[b,i,v]; [B,L,N,N].

        ``logp_rel`` is [B, L, U] over the relevant token columns; edges index it
        via ``edge_v_local``.
        """
        B, L, _ = logp_rel.shape
        N = self.N
        vals = logp_rel[:, :, self.edge_v_local]  # [B, L, E]
        flat = torch.full(
            (B, L, N * N), NEG_INF, device=logp_rel.device, dtype=logp_rel.dtype
        )
        idx = (self.edge_s * N + self.edge_d).view(1, 1, -1).expand(B, L, -1)
        mx = torch.full_like(flat, NEG_INF)
        mx.scatter_reduce_(2, idx, vals, reduce="amax", include_self=True)
        ex = torch.exp(vals - mx.gather(2, idx))
        sm = torch.zeros_like(flat)
        sm.scatter_add_(2, idx, ex)
        flat = torch.where(sm > 0, mx + torch.log(sm), torch.full_like(flat, NEG_INF))
        return flat.view(B, L, N, N)

    @staticmethod
    def _logmv(v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Log-semiring vec@mat: out[b,k]=logsumexp_s v[b,s]+m[b,s,k]. [B,N]."""
        return torch.logsumexp(v.unsqueeze(-1) + m, dim=-2)

    @staticmethod
    def _logmv_t(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Log-semiring mat@vec: out[b,s]=logsumexp_k m[b,s,k]+v[b,k]. [B,N]."""
        return torch.logsumexp(m + v.unsqueeze(-2), dim=-1)

    def constrained_log_marginals(self, logits: torch.Tensor) -> torch.Tensor:
        """Reweight ``logits`` [B, L, V] to constrained log-marginals [B, L, V].

        All heavy work is over the ``U`` relevant token columns (edges touch far
        fewer than V tokens); the full-V result is a single scatter of those U
        columns into a ``-inf`` tensor. Tokens with no DFA transition stay
        ``-inf``; a request whose constraint is unsatisfiable within L (logZ ==
        -inf) is left unchanged.
        """
        B, L, V = logits.shape
        N = self.N
        rel = self.rel_v  # [U] relevant token ids
        U = rel.shape[0]
        dt = logits.dtype
        lse = torch.logsumexp(logits, dim=-1, keepdim=True)  # [B, L, 1]
        logp_rel = logits[:, :, rel] - lse  # [B, L, U]
        M = self._transition_matrices(logp_rel)  # [B, L, N, N]

        alpha = torch.full((B, L + 1, N), NEG_INF, device=logits.device, dtype=dt)
        alpha[:, 0, self.start] = 0.0
        for i in range(L):
            alpha[:, i + 1] = self._logmv(alpha[:, i], M[:, i])
        beta = torch.full((B, L + 1, N), NEG_INF, device=logits.device, dtype=dt)
        beta[:, L] = torch.where(
            self.accepting,
            torch.zeros(N, device=logits.device, dtype=dt),
            torch.full((N,), NEG_INF, device=logits.device, dtype=dt),
        )
        for i in range(L - 1, -1, -1):
            beta[:, i] = self._logmv_t(M[:, i], beta[:, i + 1])

        ok = torch.isfinite(beta[:, 0, self.start])  # [B] logZ finite

        # per-edge weight alpha[b,i,edge_s] + beta[b,i+1,edge_d], accumulated per
        # relevant token (edge_v_local) via scatter-logsumexp -> [B, L, U].
        E = self.edge_v.shape[0]
        w = (
            alpha[:, :L][:, :, self.edge_s] + beta[:, 1:][:, :, self.edge_d]
        )  # [B, L, E]
        idx = self.edge_v_local.view(1, 1, E).expand(B, L, E)
        mx = torch.full((B, L, U), NEG_INF, device=logits.device, dtype=dt)
        mx.scatter_reduce_(2, idx, w, reduce="amax", include_self=True)
        ex = torch.exp(w - mx.gather(2, idx))
        sm = torch.zeros((B, L, U), device=logits.device, dtype=dt)
        sm.scatter_add_(2, idx, ex)
        acc = torch.where(sm > 0, mx + torch.log(sm), torch.full_like(mx, NEG_INF))
        cu = acc + logp_rel  # [B, L, U] constrained logits at relevant tokens

        out = torch.full((B, L, V), NEG_INF, device=logits.device, dtype=dt)
        out[:, :, rel] = cu
        if not bool(ok.all()):
            out = torch.where(ok.view(B, 1, 1), out, logits)
        return out
