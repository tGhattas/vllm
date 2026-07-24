# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture real DiffusionGemma per-denoise-step logits via vLLM (the ONE GPU step).

Monkeypatches ``DiffusionGemmaForConditionalGeneration.compute_logits`` to stash
the per-position logits of the first denoising step(s), runs one short generation,
and saves ``dg_logits.npy`` ([steps, L, V]) + ``dg_meta.json``. Everything after
this (DFA build, constrained sampling, decoding) is cheap and runs offline on the
saved logits via dg_constrain.py — so this expensive step runs once.

VERIFY-ON-MACHINE markers (####) flag vLLM-version-specific bits to sanity-check.

Usage (after dg_preflight.py says GO):
  python dg_capture.py --model <bf16-diffusiongemma> --prompt "Extract JSON: ..." \
      --num-capture 1 --gpu-mem-util 0.9 --max-model-len 2048
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Here is a JSON object with a name and age: ")
    ap.add_argument("--num-capture", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="dg_logits.npy")
    ap.add_argument(
        "--canvas-length",
        type=int,
        default=None,
        help="only if vLLM needs it explicitly via diffusion_config",
    )
    args = ap.parse_args()

    import torch  # noqa: F401  (ensures CUDA context / clear error early)

    # #### Adjust the import path/class name if the model module differs.
    import vllm.model_executor.models.diffusion_gemma as dgmod
    from vllm import LLM, SamplingParams

    cls = dgmod.DiffusionGemmaForConditionalGeneration
    # Keep the widest [L, V] tensor seen (the full-canvas denoise step, not the
    # short prompt prefill), and log every fired shape.
    best: dict = {"arr": None, "where": None}
    shapes: list[tuple] = []

    # NOTE: vLLM V1 runs the model in a separate EngineCore subprocess, so a
    # monkeypatch here only fires if the engine runs IN-PROCESS. Launch with
    # VLLM_ENABLE_V1_MULTIPROCESSING=0 (see run script) for the hook to work.
    def _capture(x, where):
        import torch as _t

        if not isinstance(x, _t.Tensor) or x.ndim < 2 or x.shape[-1] < 1000:
            return
        arr = x.detach().float().cpu().numpy()
        if arr.ndim == 3:  # [B, L, V] -> [L, V]
            arr = arr[0]
        shapes.append((where, tuple(arr.shape)))
        if best["arr"] is None or arr.shape[0] > best["arr"].shape[0]:
            best["arr"], best["where"] = arr, where

    # Hook the model's compute_logits (per-position logits over the canvas).
    _orig_cl = cls.compute_logits

    def _cl_hook(self, *a, **k):
        out = _orig_cl(self, *a, **k)
        _capture(out, "compute_logits")
        return out

    cls.compute_logits = _cl_hook

    # Fallback: hook DiffusionSampler.__call__ and grab its logits argument.
    ds = getattr(dgmod, "DiffusionSampler", None)
    if ds is not None:
        _orig_ds = ds.__call__

        def _ds_hook(self, *a, **k):
            for x in list(a) + list(k.values()):
                _capture(x, "DiffusionSampler")
            return _orig_ds(self, *a, **k)

        ds.__call__ = _ds_hook

    # #### diffusion_config may be auto-derived from the HF config's canvas_length;
    # pass explicitly only if construction complains.
    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enforce_eager=True,  # simpler + lets the hook see uncompiled logits
    )
    if args.canvas_length is not None:
        llm_kwargs["diffusion_config"] = {"canvas_length": args.canvas_length}

    try:
        llm = LLM(**llm_kwargs)
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if "fp4" in msg or "sm_" in msg or "capability" in msg:
            raise SystemExit(
                "Model load failed — looks NVFP4/arch related. Re-run "
                "dg_preflight.py; you likely need a bf16 checkpoint or a "
                "Blackwell GPU.\n" + str(e)
            ) from e
        raise

    # diffusion rejects non-default sampling params; temperature must be 1.0.
    sp = SamplingParams(temperature=1.0, max_tokens=64)
    out = llm.generate([args.prompt], sp)
    text = out[0].outputs[0].text if out and out[0].outputs else ""

    if best["arr"] is None:
        raise SystemExit(
            "No logits captured — the hook never fired. Check the class/method "
            "name (####) against this vLLM version, or run with "
            "VLLM_ENABLE_V1_MULTIPROCESSING=0 so the model runs in-process."
        )

    stacked = best["arr"][None]  # [1, L, V]
    np.save(args.out, stacked)
    meta = {
        "model": args.model,
        "prompt": args.prompt,
        "kept_from": best["where"],
        "kept_shape": list(best["arr"].shape),
        "all_captured_shapes": shapes,
        "generated_text": text,
    }
    with open("dg_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[capture] saved {args.out} shape={stacked.shape}; meta=dg_meta.json")
    print(f"[capture] generated (unconstrained): {text!r}")
    print(
        f"[capture] next: python dg_constrain.py --logits {args.out} "
        f"--model {args.model}"
    )


if __name__ == "__main__":
    main()
