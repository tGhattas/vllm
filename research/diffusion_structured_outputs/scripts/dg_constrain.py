# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Constrain saved DiffusionGemma logits with the verified sampler (cheap, offline).

Loads one denoise step's logits (from dg_capture.py, or --synthetic random logits
for a zero-GPU dry run), builds a token DFA from a schema, runs the oracle-verified
constrained sampler, and checks the result is DFA-accepted + valid, contrasting it
with the unconstrained argmax. Runs many times without re-touching the 52 GB model.

Modes:
  --mode digits            constrain the whole canvas to digit tokens (fast, robust;
                           the DiffusionGemma analog of the Dream validation)
  --mode schema --schema f.json   general JSON-schema -> token DFA over the real
                           tokenizer vocab (slower first build; cached)

Examples:
  # zero-GPU pipeline dry run before capture (validates plumbing instantly):
  python dg_constrain.py --synthetic --model <repo> --mode digits --canvas-length 16
  # on real captured logits:
  python dg_constrain.py --logits dg_logits.npy --model <repo> --mode digits
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constrained_sampler as cs  # noqa: E402
from finite_automaton import DFA  # noqa: E402
from schema_compiler import (  # noqa: E402
    build_regex_dfa,
    decode_canvas,
    json_schema_to_regex,
)


def load_logits(args) -> np.ndarray:
    if args.synthetic:
        rng = np.random.default_rng(args.seed)
        # V from tokenizer if available, else a placeholder; L from --canvas-length.
        v = args.vocab_size
        return cs.logits_to_logprobs(rng.normal(size=(args.canvas_length, v)))
    arr = np.load(args.logits).astype(np.float64)
    if arr.ndim == 3:  # [steps, L, V] -> pick a step
        arr = arr[args.step]
    return cs.logits_to_logprobs(arr)


def digit_token_ids(tok) -> list[int]:
    # ASCII 0-9 only (str.isdigit() would also match Unicode digits from many
    # scripts, giving a noisy demo); single-character surface tokens.
    out = []
    for tokstr, tid in tok.get_vocab().items():
        core = tokstr.replace("Ġ", "").replace("Ċ", "").replace("▁", "")
        if len(core) == 1 and core in "0123456789":
            out.append(int(tid))
    return sorted(out)


def regex_charset(pat: str) -> set[str] | None:
    """The set of characters a regex can match, or None if unbounded (`.`/`[^..]`)."""
    chars: set[str] = set()
    i = 0
    while i < len(pat):
        c = pat[i]
        if c == "\\":
            n = pat[i + 1]
            if n == "d":
                chars |= set("0123456789")
            elif n in "wsWSD":
                return None  # broad classes -> don't prune
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
                    for o in range(ord(cls[k]), ord(cls[k + 2]) + 1):
                        chars.add(chr(o))
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


def relevant_tokens(tok, charset: set[str]) -> tuple[list[int], list[str]]:
    """Vocab tokens whose surface string is non-empty and within `charset`.

    Other tokens can never appear in an accepted canvas, so pruning to these is
    exact and keeps the token-DFA build small over a huge vocab.
    """
    rel = []
    for tokstr, tid in tok.get_vocab().items():
        s = tok.convert_tokens_to_string([tokstr])
        if s and all(ch in charset for ch in s):
            rel.append((int(tid), s))
    rel.sort()
    return [t for t, _ in rel], [s for _, s in rel]


def run_cpu(log_probs, dfa, seed):
    beta = cs.backward_messages(log_probs, dfa)
    if beta[0, dfa.start_state] == cs.NEG_INF:
        raise SystemExit(
            "IMPOSSIBLE: no accepted canvas (logZ=-inf). "
            "Increase --canvas-length or relax the schema."
        )
    rng = np.random.default_rng(seed)
    samp, _ = cs.sample(log_probs, dfa, rng, beta=beta)
    gred, _ = cs.greedy(log_probs, dfa)
    return samp, gred, float(beta[0, dfa.start_state])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", default=None, help="dg_logits.npy from dg_capture")
    ap.add_argument("--synthetic", action="store_true", help="random logits, no model")
    ap.add_argument("--model", default=None, help="repo id for the tokenizer")
    ap.add_argument("--mode", choices=["digits", "schema"], default="digits")
    ap.add_argument("--schema", default=None, help="JSON schema file for --mode schema")
    ap.add_argument("--canvas-length", type=int, default=16, help="L for --synthetic")
    ap.add_argument("--vocab-size", type=int, default=256000, help="V for --synthetic")
    ap.add_argument("--pad-token-id", type=int, default=None)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument(
        "--positions",
        type=int,
        default=None,
        help="use only the first N canvas positions (for a short match)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = None
    if args.model:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        args.vocab_size = int(getattr(tok, "vocab_size", args.vocab_size))

    log_probs = load_logits(args)
    if args.positions is not None:
        log_probs = log_probs[: args.positions]
    L, V = log_probs.shape
    print(
        f"[data] logits [L,V]=[{L},{V}]  mode={args.mode}  "
        f"{'(synthetic)' if args.synthetic else args.logits}"
    )

    raw_argmax = [int(x) for x in np.argmax(log_probs, axis=-1)]
    if tok is not None:
        print(f"[unconstrained] argmax text: {tok.decode(raw_argmax)!r}")

    if tok is None:
        raise SystemExit("--model is required (for the tokenizer)")

    # Reduce to the relevant token set (exact: other tokens carry no accepted
    # mass), then run the sampler over just those columns. `strings` are the
    # surface strings of the reduced tokens; `pad` is the reduced pad index.
    if args.mode == "digits":
        allowed = digit_token_ids(tok)
        if not allowed:
            raise SystemExit("no single-digit tokens found in this tokenizer")
        strings = [
            tok.convert_tokens_to_string([tok.convert_ids_to_tokens(t)])
            for t in allowed
        ]
        dfa = DFA(1, 0, {0}, {(0, t): 0 for t in range(len(allowed))}, len(allowed))
        pad = None
        print(f"[dfa] digits-only over {len(allowed)} digit tokens")
    else:  # schema
        if args.schema is None:
            raise SystemExit("--mode schema needs --schema file")
        with open(args.schema) as f:
            schema = json.load(f)
        regex = json_schema_to_regex(schema)
        cset = regex_charset(regex)
        if cset is None:
            raise SystemExit(
                'schema has unbounded strings ([^"]*); use a bounded '
                "schema (integer/boolean/enum) for the pruned demo"
            )
        t0 = time.time()
        allowed, strings = relevant_tokens(tok, cset)
        allowed.append(-1)  # sentinel real id for PAD
        strings.append("")  # PAD surface
        pad = len(allowed) - 1
        dfa = build_regex_dfa(regex, strings, pad_token_id=pad)
        print(f"[dfa] regex={regex}")
        print(
            f"[dfa] {len(allowed)} relevant tokens (of {V}); {dfa.num_states} "
            f"states in {time.time() - t0:.1f}s"
        )

    lp = log_probs[:, [t if t >= 0 else 0 for t in allowed]]
    if pad is not None:
        lp[:, pad] = np.log(0.5)  # give PAD a fixed modest mass so it can fill
    samp_r, gred_r, logZ = run_cpu(lp, dfa, args.seed)
    txt_s = decode_canvas(samp_r, strings, pad)
    txt_g = decode_canvas(gred_r, strings, pad)

    print(f"[constrained] logZ={logZ:.3f}")
    print(f"[constrained] sample text: {txt_s!r}")
    print(f"[constrained] greedy text: {txt_g!r}")
    if args.mode == "schema":
        try:
            print(f"[constrained] greedy VALID JSON: {json.loads(txt_g)}")
        except Exception as e:  # noqa: BLE001
            print(f"[constrained] greedy JSON parse failed: {e}")
    print("[done] accepted-by-construction=True")


if __name__ == "__main__":
    main()
