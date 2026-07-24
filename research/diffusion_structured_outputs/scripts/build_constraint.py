# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile a schema/regex + model tokenizer into a DiffusionConstraint file.

The output (edge list over the model's real token ids + accepting/start) is loaded
by vllm's diffusion_constraint at runtime when VLLM_DIFFUSION_CONSTRAINT points at
it, enabling opt-in canvas-aware constrained decoding for DiffusionGemma.

Trailing PAD is the model's EOS token, so a short match can fill the fixed canvas
(the model emits EOS after the JSON).

Usage:
  python build_constraint.py --model <repo> --schema person.json --out person.pt
  python build_constraint.py --model <repo> --regex '[0-9]+' --out digits.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # schema_compiler, finite_automaton
sys.path.insert(0, _HERE)  # dg_constrain helpers

import dg_constrain as dgc  # noqa: E402
from schema_compiler import build_regex_dfa, json_schema_to_regex  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--schema", default=None, help="JSON schema file")
    ap.add_argument("--regex", default=None, help="raw regex (alternative to --schema)")
    ap.add_argument("--out", default="constraint.pt")
    ap.add_argument("--vocab-size", type=int, default=None)
    args = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.schema:
        with open(args.schema) as f:
            regex = json_schema_to_regex(json.load(f))
    elif args.regex:
        regex = args.regex
    else:
        raise SystemExit("pass --schema or --regex")
    print(f"[regex] {regex}")

    cset = dgc.regex_charset(regex)
    if cset is None:
        raise SystemExit("unbounded regex ([^..]/.); use a bounded schema/regex")

    allowed, strings = dgc.relevant_tokens(tok, cset)
    eos = tok.eos_token_id
    if eos is None:
        raise SystemExit("tokenizer has no eos_token_id for PAD")
    allowed.append(int(eos))  # PAD = EOS (fills canvas after the match)
    strings.append("")
    pad_reduced = len(allowed) - 1
    dfa = build_regex_dfa(regex, strings, pad_token_id=pad_reduced)

    if args.vocab_size is not None:
        vocab = args.vocab_size
    else:
        cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        tc = getattr(cfg, "text_config", cfg)
        vocab = int(
            getattr(tc, "vocab_size", getattr(cfg, "vocab_size", tok.vocab_size))
        )

    edge_s, edge_v, edge_d = [], [], []
    for (s, rv), d in dfa.transitions.items():
        edge_s.append(s)
        edge_v.append(int(allowed[rv]))  # reduced id -> real model token id
        edge_d.append(d)
    accepting = torch.zeros(dfa.num_states, dtype=torch.bool)
    for s in dfa.accepting:
        accepting[s] = True

    torch.save(
        {
            "num_states": dfa.num_states,
            "vocab_size": vocab,
            "start": dfa.start_state,
            "accepting": accepting,
            "edge_s": torch.tensor(edge_s, dtype=torch.long),
            "edge_v": torch.tensor(edge_v, dtype=torch.long),
            "edge_d": torch.tensor(edge_d, dtype=torch.long),
        },
        args.out,
    )
    print(
        f"[saved] {args.out}: {dfa.num_states} states, {len(edge_s)} edges, "
        f"{len(allowed)} relevant tokens, vocab={vocab}"
    )


if __name__ == "__main__":
    main()
