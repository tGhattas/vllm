# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Constrain SAVED Dream logits with the verified CPU sampler (no model reload).

Loads only the tokenizer + the [L, V] logits captured by capture_dream.py, then
runs the exact DFA-constrained sampler. Fast digit-token scan via get_vocab()
(None-safe, unlike per-id decode over the full 152k range).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constrained_sampler as cs  # noqa: E402
from finite_automaton import DFA  # noqa: E402

MODEL = "Dream-org/Dream-v0-Base-7B"
LOGITS = "/workspace/research/dream_canvas_logits.npy"


def digit_token_ids(tokenizer) -> list[int]:
    """Ids whose token is a single digit (with/without a byte-level space marker)."""
    out = []
    for tokstr, tid in tokenizer.get_vocab().items():
        core = tokstr.replace("Ġ", "").replace("Ċ", "")  # Ġ, Ċ markers
        if len(core) == 1 and core.isdigit():
            out.append(int(tid))
    return sorted(out)


def digits_only_dfa(allowed, V):
    return DFA(1, 0, {0}, {(0, t): 0 for t in allowed}, V)


def even_count_dfa(allowed, target, V):
    tr = {}
    for q in (0, 1):
        for t in allowed:
            tr[(q, t)] = (q ^ 1) if t == target else q
    return DFA(2, 0, {0}, tr, V)


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    logits = np.load(LOGITS).astype(np.float64)
    L, V = logits.shape
    print(f"[data] logits {logits.shape} from {MODEL}")

    log_probs = cs.logits_to_logprobs(logits)
    raw_argmax = [int(x) for x in np.argmax(logits, axis=-1)]
    print("\n=== UNCONSTRAINED (raw per-position argmax) ===")
    print("ids :", raw_argmax)
    print("text:", repr(tok.decode(raw_argmax)))

    allowed = digit_token_ids(tok)
    print(f"\n[dfa] {len(allowed)} single-digit tokens: {allowed}")
    target = allowed[0]

    for name, dfa in [
        ("digits-only (N=1)", digits_only_dfa(allowed, V)),
        (
            f"digits + even-count-of-token-{target} (N=2)",
            even_count_dfa(allowed, target, V),
        ),
    ]:
        print(f"\n=== CONSTRAINED: {name} ===")
        t = time.time()
        beta = cs.backward_messages(log_probs, dfa)
        logZ = float(beta[0, dfa.start_state])
        if logZ == cs.NEG_INF:
            print("  IMPOSSIBLE (logZ = -inf)")
            continue
        samp, samp_lp = cs.sample(log_probs, dfa, np.random.default_rng(0), beta=beta)
        gred, _ = cs.greedy(log_probs, dfa)
        dt = time.time() - t
        print(f"  logZ={logZ:.3f}  DP+sample+greedy={dt:.2f}s")
        print(f"  greedy: {gred} -> {tok.decode(gred)!r}")
        print(f"  sample: {samp} -> {tok.decode(samp)!r}")
        acc = f"{dfa.accepts(samp)}/{dfa.accepts(gred)}"
        print(f"  accepted sample/greedy = {acc}")
        print(f"  raw_argmax_satisfies_dfa={dfa.accepts(raw_argmax)}")
        if dfa.num_states == 2:  # even-count DFA should yield an even count
            print(f"  count(token {target}) in greedy = {gred.count(target)}")

    print("\n[done] verified CPU sampler ran on REAL Dream-7B logits")


if __name__ == "__main__":
    main()
