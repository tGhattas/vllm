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
from schema_compiler import build_schema_dfa, decode_canvas  # noqa: E402


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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = None
    if args.model:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        args.vocab_size = int(getattr(tok, "vocab_size", args.vocab_size))

    log_probs = load_logits(args)
    L, V = log_probs.shape
    print(
        f"[data] logits [L,V]=[{L},{V}]  mode={args.mode}  "
        f"{'(synthetic)' if args.synthetic else args.logits}"
    )

    raw_argmax = [int(x) for x in np.argmax(log_probs, axis=-1)]
    if tok is not None:
        print(f"[unconstrained] argmax text: {tok.decode(raw_argmax)!r}")

    if args.mode == "digits":
        if tok is None:
            raise SystemExit("--mode digits needs --model for the tokenizer")
        allowed = digit_token_ids(tok)
        if not allowed:
            raise SystemExit("no single-digit tokens found in this tokenizer")
        print(f"[dfa] digits-only over {len(allowed)} digit tokens")
        # reduce to the relevant columns (exact; other tokens carry no accepted mass)
        lp = log_probs[:, allowed]
        rdfa = DFA(1, 0, {0}, {(0, t): 0 for t in range(len(allowed))}, len(allowed))
        samp_r, gred_r, logZ = run_cpu(lp, rdfa, args.seed)
        samp = [allowed[t] for t in samp_r]
        gred = [allowed[t] for t in gred_r]
    else:  # schema
        if tok is None or args.schema is None:
            raise SystemExit("--mode schema needs --model and --schema file")
        with open(args.schema) as f:
            schema = json.load(f)
        pad = args.pad_token_id
        if pad is None:
            pad = getattr(tok, "pad_token_id", None) or getattr(
                tok, "mask_token_id", None
            )
        if pad is None:
            raise SystemExit("no pad/mask token id; pass --pad-token-id")
        print(
            f"[dfa] building token DFA from schema over V={V} (first build is slow)..."
        )
        vocab_strings = [
            tok.convert_tokens_to_string([tok.convert_ids_to_tokens(i)])
            for i in range(V)
        ]
        t0 = time.time()
        dfa = build_schema_dfa(schema, vocab_strings, pad_token_id=pad)
        print(f"[dfa] {dfa.num_states} states in {time.time() - t0:.1f}s")
        samp, gred, logZ = run_cpu(log_probs, dfa, args.seed)

    accepted = True  # run_cpu already guarantees acceptance by construction
    print(f"[constrained] logZ={logZ:.3f}")
    if tok is not None:
        if args.mode == "schema":
            txt_s = decode_canvas(samp, vocab_strings, pad)
            txt_g = decode_canvas(gred, vocab_strings, pad)
        else:
            txt_s, txt_g = tok.decode(samp), tok.decode(gred)
        print(f"[constrained] sample ids={samp}")
        print(f"[constrained] sample text: {txt_s!r}")
        print(f"[constrained] greedy text: {txt_g!r}")
        if args.mode == "schema":
            try:
                obj = json.loads(txt_g)
                print(f"[constrained] greedy is VALID JSON: {obj}")
            except Exception as e:  # noqa: BLE001
                print(f"[constrained] greedy JSON parse failed: {e}")
    print(f"[done] accepted-by-construction={accepted}")


if __name__ == "__main__":
    main()
