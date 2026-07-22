# GPU Validation: exact CPU constrained sampler on real diffusion logits

First GPU milestone (per the research plan): *the oracle-verified CPU constrained
sampler operates correctly on real per-position logits from a smaller
diffusion-language model.* **Achieved 2026-07-22.**

## Setup

| Item | Value |
| --- | --- |
| Host | RunPod, NVIDIA **A40 45 GB** (Ampere sm86), Ubuntu 22.04, driver CUDA 12.8 |
| Env | Python 3.11 venv; **torch 2.6.0+cu124** (the default cu130 wheel failed: "driver too old"), **transformers 4.46.3** (Dream's remote code breaks on transformers 5.x RoPE API), numpy |
| Model | **`Dream-org/Dream-v0-Base-7B`** (Qwen2-based masked diffusion LM), revision **`6572adb5535263e4d1a337b56942ba48b6dee2a9`**, bf16 |
| `mask_token_id` | 151666 (from `config.mask_token_id`) |
| Vocab | 152064 |
| Prompt | `"Here is a list of numbers: "` |
| Canvas | L = 12 mask tokens appended; single bidirectional forward pass (one denoising step) |
| Peak GPU mem (fwd) | **15.30 GB** |
| Logits captured | `[12, 152064]` float32 → `dream_canvas_logits.npy` (not committed; regenerable) |

## How logits were captured

`scripts/capture_dream.py`: load Dream via `AutoModel` (it is not registered for
`AutoModelForCausalLM`), build `input_ids = prompt + [mask_id]*12`, run one full
(non-causal) forward pass, take `logits[0, -12:, :]` → per-position distributions.
`scripts/constrain_saved.py` then feeds the saved logits into the verified CPU
sampler (`constrained_sampler.py`) with no model reload.

## Result (verbatim)

```text
=== UNCONSTRAINED (raw per-position argmax) ===
ids : [11, 220, 11, 11, 220, 18, 11, 220, 16, 11, 220, 220]
text: ', ,, 3, 1,  '

[dfa] 13 single-digit tokens: [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 110, 111, 117]

=== CONSTRAINED: digits-only (N=1) ===
  logZ=-11.885  DP+sample+greedy=0.79s
  greedy ids : [15, 17, 17, 18, 16, 18, 16, 16, 16, 16, 16, 16]  -> '022313111111'
  sample ids : [18, 17, 16, 15, 21, 23, 19, 20, 19, 23, 21, 15]  -> '321068454860'
  accepted(sample)=True  accepted(greedy)=True
  raw_argmax_satisfies_dfa=False

=== CONSTRAINED: digits + even-count-of-token-15 (N=2) ===
  logZ=-12.532  DP+sample+greedy=0.98s
  greedy ids : [15, 17, 17, 18, 15, 18, 16, 16, 16, 16, 16, 16]  -> '022303111111'
  sample ids : [19, 17, 16, 15, 21, 23, 18, 20, 19, 23, 17, 15]  -> '421068354820'
  accepted(sample)=True  accepted(greedy)=True
  raw_argmax_satisfies_dfa=False
  count(token 15) in greedy = 2 (must be even)
```

(token id → digit: 15→`0`, 16→`1`, 17→`2`, 18→`3`, 19→`4`, 20→`5`, 21→`6`,
22→`7`, 23→`8`, 24→`9`.)

## What this demonstrates

- **Constraint satisfaction by construction on real logits:** every constrained
  sample and greedy output is DFA-accepted; the unconstrained argmax is not.
- **Constrained ≠ unconstrained:** raw output `', ,, 3, 1,  '` vs. constrained
  all-digit strings.
- **Genuine cross-position work (Viterbi, not per-slot argmax):** the even-count
  DFA changed the greedy from `022313111111` (one `0` → odd) to `022303111111`
  (flipping position 4 `1`→`0`, giving two `0`s → even) — the minimum-log-prob-cost
  edit to satisfy a constraint no single position can enforce alone.
- **Scales to real vocab:** the O(L·N·V) DP over V = 152064 ran in ~0.8–1.0 s on CPU
  (unoptimized Python) — confirming the correctness-first reference is usable for
  validation, and pinpointing V as the term a production GPU kernel must attack.

## Log-depth sampler on the same real logits

The O(log L) parallel sampler (`parallel_sampler.py`, the paper novelty) was run on
these same real Dream logits (digits-only DFA; the constraint makes only the 13
digit-token columns relevant, so the full-softmax log-probs are sliced to them — an
exact reduction, `logZ` unchanged):

- `logZ` full-vocab = reduced = sequential = **log-depth = -11.885183** (all match).
- Every log-depth sample is DFA-accepted; empirical marginals match the sequential
  sampler (max diff 0.008, MC noise) and the exact constrained marginals (0.006).
- **Depth reduced on real data:** sequential 12 steps vs. log-depth **4 levels**
  (⌈log₂12⌉). So the novelty is confirmed on real diffusion logits, not just
  synthetic oracle cases.

## Caveats / not yet done

- All GPU work used **Dream-v0-Base-7B only** — not a second small diffusion model,
  and **not DiffusionGemma** itself (its NVFP4 target needs Blackwell / an A100
  80 GB; the A40 cannot run it).
- This validates the **algorithm on one real denoising step's logits** (plus the
  multi-step loop, confirmed separately) — not a production denoise/renoise kernel.
- The mean-field independence assumption is used as-is; a real integration re-solves
  per step as positions get fixed (see `design.md`).
- No production vLLM code was touched; the `sampling_params.py:915` guard stands.

## Reproduce

```text
# on the pod
export HF_HOME=/workspace/huggingface-cache HF_HUB_ENABLE_HF_TRANSFER=0
/workspace/venv-dllm/bin/python scripts/capture_dream.py \
    --model Dream-org/Dream-v0-Base-7B --canvas-length 12 --seed 0
/workspace/venv-dllm/bin/python scripts/constrain_saved.py
```
