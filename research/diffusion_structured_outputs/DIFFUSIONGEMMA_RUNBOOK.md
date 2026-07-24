# DiffusionGemma runbook (A100) — capture real logits, constrain, prototype

Goal: spend A100 time only on the one thing that needs the GPU — capturing real
DiffusionGemma per-denoise-step logits — and do everything else (DFA build,
constrained sampling, validation) cheaply and offline on the saved logits.
This mirrors the Dream-7B validation that already worked, on the real target model.

Scripts (in `scripts/`): `dg_preflight.py` → `dg_capture.py` → `dg_constrain.py`.

---

## 0. Setup (once)

```bash
# match torch build to the driver (check: nvidia-smi). Do NOT take the default wheel.
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124   # or cu126/cu128
# vLLM with DiffusionGemma (PR #45163) — python-only build:
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
export HF_HOME=/workspace/huggingface-cache            # persistent volume
export HF_TOKEN=...                                    # gated nvidia checkpoint
```

## 1. PREFLIGHT — fail fast before burning GPU time

```bash
python research/diffusion_structured_outputs/scripts/dg_preflight.py --model <REPO>
```

Decision tree:

- **NVFP4 checkpoint + non-Blackwell GPU (A100 is sm 8.0) → NO-GO.** NVFP4 kernels
  need Blackwell (sm 10.0/12.0). Get a bf16 DiffusionGemma checkpoint, or a Blackwell
  card. Do not proceed.
- **`canvas_length` missing from config → wrong checkpoint** (vLLM won't treat it as
  diffusion). Fix the model id.
- **weights + ~12 GB headroom > GPU mem → TOO BIG.** 26B bf16 ≈ 52 GB → fits 80 GB at
  batch 1; if it doesn't fit, lower `--max-model-len` / `--gpu-mem-util`.
- Otherwise **GO**.

## 2. VALIDATE THE OFFLINE PIPELINE FIRST (zero GPU, zero model)

Prove the constrain path works before the expensive capture:

```bash
python .../dg_constrain.py --synthetic --model <REPO> --mode digits --canvas-length 16
```

Loads only the tokenizer (small), runs the verified constrained sampler on random
logits, and prints an all-digit, DFA-accepted canvas. If this works, the only
unknown left is the capture hook.

## 3. CAPTURE — the one expensive GPU step (run once)

```bash
python .../dg_capture.py --model <REPO> \
    --prompt "Here is a JSON object with a name and age: " \
    --num-capture 1 --gpu-mem-util 0.9 --max-model-len 2048
```

Saves `dg_logits.npy` (`[steps, L, V]`) + `dg_meta.json` and prints the
unconstrained generation. **VERIFY:** the captured shape's middle dim should be the
canvas length and the last dim the vocab size (see `####` markers in the script). If
`No logits captured`, the `compute_logits` hook name/path needs adjusting for this
vLLM version (fallback: hook `DiffusionSampler.__call__` instead).

## 4. CONSTRAIN — iterate freely offline on the saved logits

```bash
# digits demo (robust; the DiffusionGemma analog of the Dream result):
python .../dg_constrain.py --logits dg_logits.npy --model <REPO> --mode digits

# general JSON schema (first build is slower; token-DFA over the full vocab):
python .../dg_constrain.py --logits dg_logits.npy --model <REPO> \
    --mode schema --schema my_schema.json
```

Expected: constrained sample/greedy is **DFA-accepted by construction** and (schema
mode) decodes to **valid JSON matching the schema**, while the unconstrained argmax
generally does not. This is the milestone: *the verified constrained sampler operates
on real DiffusionGemma logits.*

`my_schema.json` (regular subset) example:

```json
{"type":"object",
 "properties":{"name":{"type":"string"},"age":{"type":"integer"},"active":{"type":"boolean"}},
 "order":["name","age","active"]}
```

## 5. In-denoising-loop guided decoding — INTEGRATED (flag-gated)

Option (c) is now integrated into vLLM behind an opt-in flag (default off; the
`sampling_params.py:915` guard is untouched):

- `vllm/model_executor/models/diffusion_constraint.py` — `DiffusionConstraint`
  reweights canvas logits to the constrained marginals `P_D(x_i=v)` (oracle-verified).
- `diffusion_gemma.py` `DiffusionSampler.__call__` — when
  `VLLM_DIFFUSION_CONSTRAINT` points at a precompiled constraint file, replaces the
  canvas logits with those marginals before the compiled sample step, steering every
  denoising step toward a DFA-accepted canvas.

Build the constraint + run it:

```bash
python scripts/build_constraint.py --model <repo> --schema person.json \
    --vocab-size 262144 --out /workspace/person_constraint.pt
VLLM_DIFFUSION_CONSTRAINT=/workspace/person_constraint.pt \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
LD_LIBRARY_PATH=.../nvidia/cu13/lib:$LD_LIBRARY_PATH \
  python gen.py   # LLM(...).generate(...)
```

**Verified on the RTX PRO 6000 Blackwell** with `nvidia/diffusiongemma-26B-A4B-it-
NVFP4` and schema `{"active":boolean,"age":integer}` (33-state DFA, 179 edges):
baseline output `''`; **constrained output `{"active":true,"age":6}`** — valid JSON,
schema-conformant, from the real NVFP4 model.

Notes: fp32 DP over `[num_decode, CL, V]` adds a transient — computed over the
~thousands of relevant token columns and scattered to full-V once
(`constrained_log_marginals`), ~20 ms/call at CL=256.

### Per-request via the normal API (regex / choice) — status

Wired end to end (commits d100ce710 → 056be7154): the guard allows `regex`/`choice`
for diffusion, `core.py` detaches `structured_output_request` so the AR grammar
machinery is bypassed, `DiffusionSampler.add_request` compiles+caches a per-slot
`DiffusionConstraint` from `sampling_params.structured_outputs`, and `__call__`
applies it per decode row. The compiler and marginals are oracle-exact (unit
tests). Usage:

```python
from vllm.sampling_params import StructuredOutputsParams as SO
SamplingParams(temperature=1.0, structured_outputs=SO(regex=r"..."))   # or choice=[...]
```

**Known issue (open):** the *runtime* constraint compile — `get_vocab()` +
`convert_tokens_to_string` over the 262k vocab through the **serving tokenizer
wrapper on the worker** — stalls the engine, even though the same scan is ~0.3 s
in a standalone process and the DFA build is ~0.2 s. So the per-request path is
not yet usable end-to-end. The **precompiled-file path works** (3 s, valid JSON —
see above), which proves the sampler/marginals mechanism; the fix is to compile
the DFA **off the worker hot-path** — in the engine-core input thread with the
fast tokenizer (as the AR `StructuredOutputManager` does), threading the compiled
edge list to the worker via `NewRequestData`, or precompiling per distinct spec.
Also open (design.md §5): feed the diffusion convergence/confidence check the
constrained marginals so a peaked constraint doesn't slow commit.

---

## Verified on RTX PRO 6000 Blackwell (sm_120, native NVFP4)

Real run of `nvidia/diffusiongemma-26B-A4B-it-NVFP4` (public, `modelopt_fp4`,
canvas_length 256, vocab 262144, 128 experts). The verified constrained sampler ran
on **real captured logits**: unconstrained argmax was `' Fits Fits ...'`; the
digits-only DFA gave `99999999`. On a real JSON schema
(`{"active":boolean,"age":integer}`), charset-pruning cut the 262144 vocab to 4462
relevant tokens (33-state DFA, 0.4 s) and greedy produced **valid JSON**
`{"active":true,"age":1}` while the unconstrained argmax was newlines/`<eos>`. Non-obvious fixes learned:

- **torch must match the precompiled `_C`'s CUDA.** `uv pip install -e .
  --torch-backend=auto` pulled torch **cu128**, but the precompiled vLLM `_C` needs
  **CUDA 13** → `ImportError: libcudart.so.13`. Fix: reinstall torch **cu130**
  (`--index-url https://download.pytorch.org/whl/cu130`). The Blackwell driver (580)
  supports CUDA 13.
- **nvrtc path:** `libnvrtc.so.13` ships under `.../site-packages/nvidia/cu13/lib/`
  (non-standard). Add that dir to `LD_LIBRARY_PATH` or you get an `ImportError` (seen
  during cumem/shutdown).
- **Capture needs in-process execution.** vLLM V1 runs the model in a separate
  `EngineCore` subprocess, so a `compute_logits` monkeypatch in the launcher never
  fires. Run capture with **`VLLM_ENABLE_V1_MULTIPROCESSING=0`**. dg_capture.py hooks
  both `compute_logits` and `DiffusionSampler.__call__` (vocab-dim filter).

## What must stay true (correctness)

- Every constrained canvas is DFA-accepted (guaranteed by construction; assert it).
- fp32/fp64 only in the sampler value path (bf16 flips Gumbel-argmax at logZ≈100s).
- Compare against the numpy oracle where feasible (logZ/marginals, rtol 1e-6).

## Tokenizer caveats (schema mode)

The token→surface-string mapping uses `convert_tokens_to_string`; byte-BPE space
markers (Ġ/▁) are handled, but partial/again-tokenizable pieces are the classic
FSM-over-tokens subtlety xgrammar solves carefully. The `digits` mode sidesteps this
(single-char digit tokens). For schema mode, sanity-check the decoded JSON parses.
