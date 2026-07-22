# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real multi-step constrained denoising loop on Dream-7B (pod GPU).

Runs the confidence-based constrained denoise loop (denoise_loop.py) with a
LogitsFn that re-runs a real Dream forward pass each step, placing committed
tokens back into the canvas and re-masking the rest — i.e. a genuine remasking
schedule with real per-step logits. Confirms the loop keeps the DFA satisfiable
across real steps and the final canvas is accepted.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constrained_sampler as cs  # noqa: E402
import denoise_loop as dl  # noqa: E402
from finite_automaton import DFA  # noqa: E402

MODEL = "Dream-org/Dream-v0-Base-7B"
PROMPT = "Here is a list of numbers: "
L = 12
FIX_PER_STEP = 2


def digit_ids(tok):
    out = []
    for tokstr, tid in tok.get_vocab().items():
        core = tokstr.replace("Ġ", "").replace("Ċ", "")
        if len(core) == 1 and core.isdigit():
            out.append(int(tid))
    return sorted(out)


def main():
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        .to("cuda")
        .eval()
    )
    mask_id = int(model.config.mask_token_id)
    V = int(model.config.vocab_size)
    prompt_ids = tok(PROMPT, return_tensors="pt").input_ids[0].tolist()

    allowed = digit_ids(tok)
    dfa = DFA(1, 0, {0}, {(0, t): 0 for t in allowed}, V)  # digits-only

    n_forward = [0]

    def logits_fn(fixed):
        n_forward[0] += 1
        canvas = [fixed.get(i, mask_id) for i in range(L)]
        ids = torch.tensor([prompt_ids + canvas], device="cuda", dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=ids)
        logits = out.logits if hasattr(out, "logits") else out[0]
        lg = logits[0, -L:, :].float().cpu().numpy()
        return cs.logits_to_logprobs(lg)

    out = dl.constrained_denoise(
        logits_fn,
        dfa,
        L,
        np.random.default_rng(0),
        mode="greedy",
        fix_per_step=FIX_PER_STEP,
    )
    print(f"[dream] model={MODEL} L={L} fix_per_step={FIX_PER_STEP}")
    print(f"[dream] steps={out['steps']}  forward_passes={n_forward[0]}")
    for h in out["history"]:
        commits = {i: tok.decode([t]) for i, t in h["committed"].items()}
        print(f"  step {h['step']}: committed {commits}  (num_fixed={h['num_fixed']})")
    print(f"[dream] final canvas ids : {out['canvas']}")
    print(f"[dream] final canvas text: {tok.decode(out['canvas'])!r}")
    print(f"[dream] accepted by DFA  : {out['accepted']}")
    assert out["accepted"], "final canvas must satisfy the digits-only DFA"
    print("[done] real multi-step constrained denoise loop validated on Dream-7B")


if __name__ == "__main__":
    main()
