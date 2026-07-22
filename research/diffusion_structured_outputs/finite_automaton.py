# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deterministic finite automaton (DFA) over an integer token alphabet.

Research POC for canvas-aware constrained decoding of diffusion language models
(vLLM issue #45572). This is standalone, CPU-only, dependency-light code that is
*not* part of production vLLM.

A DFA here recognizes fixed- or variable-length sequences of integer token ids.
Transitions are a partial function: an undefined ``(state, token)`` pair means the
sequence is rejected (as if it fell into an implicit non-accepting dead sink).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


class DFA:
    """A deterministic finite automaton over integer tokens.

    Args:
        num_states: Number of states, labelled ``0 .. num_states - 1``.
        start_state: The initial state.
        accepting: Iterable of accepting (final) state ids.
        transitions: Mapping from ``(state, token)`` to the next state. Any pair
            not present is treated as a transition into an implicit dead sink,
            i.e. the sequence is rejected.
        vocab_size: Size of the token alphabet ``0 .. vocab_size - 1``.
    """

    def __init__(
        self,
        num_states: int,
        start_state: int,
        accepting: Iterable[int],
        transitions: Mapping[tuple[int, int], int],
        vocab_size: int,
    ) -> None:
        self.num_states = num_states
        self.start_state = start_state
        self.accepting = frozenset(accepting)
        self.transitions = dict(transitions)
        self.vocab_size = vocab_size

        if not 0 <= start_state < num_states:
            raise ValueError(f"start_state {start_state} out of range")
        for q in self.accepting:
            if not 0 <= q < num_states:
                raise ValueError(f"accepting state {q} out of range")
        for (q, t), nq in self.transitions.items():
            if not 0 <= q < num_states or not 0 <= nq < num_states:
                raise ValueError(f"transition {(q, t)}->{nq} has out-of-range state")
            if not 0 <= t < vocab_size:
                raise ValueError(f"transition token {t} out of vocab range")

    def step(self, state: int, token: int) -> int | None:
        """Return the next state, or ``None`` if the transition is undefined."""
        return self.transitions.get((state, token))

    def accepts(self, sequence: Sequence[int]) -> bool:
        """Return whether ``sequence`` drives the DFA into an accepting state."""
        q: int | None = self.start_state
        for tok in sequence:
            q = self.step(q, tok)
            if q is None:
                return False
        return q in self.accepting

    # -- convenience constructors used by tests / adapters ------------------

    @classmethod
    def from_exact_sequence(cls, sequence: Sequence[int], vocab_size: int) -> DFA:
        """DFA accepting exactly the single sequence ``sequence``."""
        transitions: dict[tuple[int, int], int] = {}
        for i, tok in enumerate(sequence):
            transitions[(i, tok)] = i + 1
        return cls(
            num_states=len(sequence) + 1,
            start_state=0,
            accepting={len(sequence)},
            transitions=transitions,
            vocab_size=vocab_size,
        )

    @classmethod
    def from_sequence_set(
        cls, sequences: Iterable[Sequence[int]], vocab_size: int
    ) -> DFA:
        """DFA accepting exactly the given finite set of sequences.

        Built as a prefix trie, so shared prefixes share states. Works for
        sequences of differing lengths.
        """
        transitions: dict[tuple[int, int], int] = {}
        accepting: set[int] = set()
        next_state = 1  # state 0 is the (shared) start / empty prefix

        # Map prefix tuple -> state id.
        prefix_state: dict[tuple[int, ...], int] = {(): 0}
        for seq in sequences:
            q = 0
            prefix: tuple[int, ...] = ()
            for tok in seq:
                prefix = prefix + (tok,)
                if prefix not in prefix_state:
                    prefix_state[prefix] = next_state
                    next_state += 1
                nq = prefix_state[prefix]
                transitions[(q, tok)] = nq
                q = nq
            accepting.add(q)
        return cls(
            num_states=next_state,
            start_state=0,
            accepting=accepting,
            transitions=transitions,
            vocab_size=vocab_size,
        )
