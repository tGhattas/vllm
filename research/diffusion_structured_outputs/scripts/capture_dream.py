# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RunPod GPU validation: real Dream-7B denoising logits -> verified CPU sampler.

Loads the Dream-v0 masked diffusion LM, runs ONE denoising step over a masked
canvas, extracts per-position logits [L, V], and feeds them into the
oracle-verified CPU constrained sampler. Confirms constrained output satisfies the
DFA and contrasts it with unconstrained argmax. Self-introspects the mask token.

Run on the pod:
  HF_HOME=/workspace/huggingface-cache \
  /workspace/venv-dllm/bin/python capture_dream.py --model Dream-org/Dream-v0-Base-7B
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constrained_sampler as cs  # noqa: E402
from finite_automaton import DFA  # noqa: E402


def find_mask_id(model, tokenizer) -> int:
    for src, val in [
        ("config.mask_token_id", getattr(model.config, "mask_token_id", None)),
        ("tokenizer.mask_token_id", getattr(tokenizer, "mask_token_id", None)),
    ]:
        if val is not None:
            print(f"[mask] {src} = {val}")
            return int(val)
    raise SystemExit(
        "Could not find a mask token id. config keys: "
        f"{[k for k in vars(model.config) if 'mask' in k.lower()]}"
    )


def digit_token_ids(tokenizer, vocab_size: int) -> list[int]:
    """Token ids that decode to a single digit (optionally space-prefixed)."""
    out = []
    for tid in range(vocab_size):
        s = tokenizer.decode([tid]).strip()
        if len(s) == 1 and s.isdigit():
            out.append(tid)
    return out


def digits_only_dfa(allowed: list[int], vocab_size: int) -> DFA:
    """1-state DFA: every position must be an allowed digit token; any length ok."""
    tr = {(0, t): 0 for t in allowed}
    return DFA(1, 0, {0}, tr, vocab_size)


def even_count_dfa(allowed: list[int], target: int, vocab_size: int) -> DFA:
    """2-state DFA: digits only AND an even number of `target` token."""
    tr = {}
    for q in (0, 1):
        for t in allowed:
            tr[(q, t)] = (q ^ 1) if t == target else q
    return DFA(2, 0, {0}, tr, vocab_size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Dream-org/Dream-v0-Base-7B")
    ap.add_argument("--prompt", default="Here is a list of numbers: ")
    ap.add_argument("--canvas-length", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="/workspace/research")
    args = ap.parse_args()

    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
    except Exception as e:  # noqa: BLE001
        print(f"[load] AutoModelForCausalLM failed ({e}); trying AutoModel")
        model = AutoModel.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
    model = model.to("cuda").eval()
    print(f"[load] {args.model} in {time.time() - t0:.1f}s")
    print(f"[load] revision/commit: {getattr(model.config, '_commit_hash', 'n/a')}")

    mask_id = find_mask_id(model, tok)
    vocab_size = int(getattr(model.config, "vocab_size", tok.vocab_size))
    L = args.canvas_length

    prompt_ids = tok(args.prompt, return_tensors="pt").input_ids[0].tolist()
    input_ids = torch.tensor(
        [prompt_ids + [mask_id] * L], device="cuda", dtype=torch.long
    )

    # Dream is a bidirectional (non-causal) diffusion decoder: run full attention
    # by not passing an explicit mask (its SDPA path rejects a long-dtype mask).
    # (GPU peak-memory was measured out-of-band via nvidia-smi: ~15.3 GB.)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits_t = out.logits if hasattr(out, "logits") else out[0]
    canvas_logits = logits_t[0, -L:, :].float().cpu().numpy()  # [L, V]
    print(f"[fwd] logits {canvas_logits.shape}")

    np.save(os.path.join(args.outdir, "dream_canvas_logits.npy"), canvas_logits)

    allowed = digit_token_ids(tok, vocab_size)
    print(f"[dfa] {len(allowed)} digit-like tokens in vocab {vocab_size}")
    if not allowed:
        raise SystemExit("no digit tokens found; adjust the constraint")
    target = allowed[0]  # the token whose parity DFA2 controls

    log_probs = cs.logits_to_logprobs(canvas_logits)
    raw_argmax = [int(x) for x in np.argmax(canvas_logits, axis=-1)]
    print("\n=== UNCONSTRAINED (raw per-position argmax) ===")
    print("ids   :", raw_argmax)
    print("text  :", repr(tok.decode(raw_argmax)))

    for name, dfa in [
        ("digits-only (N=1)", digits_only_dfa(allowed, vocab_size)),
        (
            "digits + even-count-of-first-digit (N=2)",
            even_count_dfa(allowed, target, vocab_size),
        ),
    ]:
        print(f"\n=== CONSTRAINED: {name} ===")
        t = time.time()
        beta = cs.backward_messages(log_probs, dfa)
        logZ = float(beta[0, dfa.start_state])
        if logZ == cs.NEG_INF:
            print("  IMPOSSIBLE (logZ = -inf)")
            continue
        rng = np.random.default_rng(args.seed)
        samp, _ = cs.sample(log_probs, dfa, rng, beta=beta)
        gred, _ = cs.greedy(log_probs, dfa)
        dt = time.time() - t
        print(f"  logZ={logZ:.3f}  DP+sample={dt:.2f}s")
        print(f"  greedy: {gred} -> {tok.decode(gred)!r}")
        print(f"  sample: {samp} -> {tok.decode(samp)!r}")
        acc = f"{dfa.accepts(samp)}/{dfa.accepts(gred)}"
        print(f"  accepted sample/greedy = {acc}")
        print(f"  raw_argmax_satisfies_dfa={dfa.accepts(raw_argmax)}")

    print("\n[done] real-logits constrained sampling validated on Dream-7B")


if __name__ == "__main__":
    main()
