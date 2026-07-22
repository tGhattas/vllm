#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""RunPod GPU-validation TEMPLATE: real diffusion logits → verified CPU sampler.

STATUS: template / not yet executed. Requires a GPU + a small diffusion LM. The
exact hook for extracting one denoising step's per-position logits ([L, V]) depends
on the chosen model and is marked TODO below. Do NOT claim results from this until
it has actually been run on RunPod.

Goal (first GPU milestone, per the task plan):
  1. Load the smallest practical diffusion LM (batch size 1, short canvas).
  2. Run one denoising step; extract per-position logits over the canvas.
  3. Feed those logits into the oracle-verified CPU constrained sampler.
  4. Assert every constrained canvas is accepted by the DFA.
  5. Compare constrained vs. unconstrained (raw-argmax) output.
  6. Record: model revision, GPU memory, canvas/prompt settings.

This does NOT integrate into the production DiffusionGemma sampler.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from finite_automaton import DFA  # noqa: E402
from model_adapters.logits_capture import constrain_canvas_from_logits  # noqa: E402


def extract_denoise_logits(model_id: str, prompt: str, canvas_length: int):
    """Return one denoising step's per-position logits as an [L, V] numpy array.

    TODO(runpod): implement for the chosen small diffusion LM. Two viable routes:
      (a) transformers: load the dLLM, run a single forward on the initialized
          canvas, take model output logits over the canvas positions.
      (b) vLLM internals: hook `compute_logits` in the DiffusionGemma runner
          (vllm/model_executor/models/diffusion_gemma.py:334) for one step.
    Keep batch size 1 and a short canvas. Return float32 [canvas_length, vocab].
    """
    raise NotImplementedError(
        "Implement real-logits extraction on RunPod for the selected model."
    )


def demo_dfa(vocab_size: int, canvas_length: int) -> DFA:
    """Placeholder DFA: accept any canvas whose count of token id 1 is even."""
    transitions = {}
    for q in (0, 1):
        for t in range(vocab_size):
            transitions[(q, t)] = (q ^ 1) if t == 1 else q
    return DFA(2, 0, {0}, transitions, vocab_size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id + pinned revision")
    ap.add_argument("--prompt", default="Extract the JSON:")
    ap.add_argument("--canvas-length", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logits = extract_denoise_logits(args.model, args.prompt, args.canvas_length)
    logits = np.asarray(logits, dtype=np.float64)
    L, V = logits.shape
    dfa = demo_dfa(V, L)

    res = constrain_canvas_from_logits(logits, dfa, seed=args.seed)
    print(f"model            : {args.model}")
    print(f"canvas [L, V]    : [{L}, {V}]")
    print(f"log Z            : {res.log_partition:.4f}")
    print(f"constrained samp : {res.sampled}")
    print(f"constrained gred : {res.greedy}")
    print(f"accepted         : {res.accepted}")
    print(f"unconstrained arg: {res.unconstrained_argmax}")
    print(f"raw argmax valid : {res.unconstrained_accepted}")
    assert res.accepted, "constrained sample must satisfy the DFA"


if __name__ == "__main__":
    main()
