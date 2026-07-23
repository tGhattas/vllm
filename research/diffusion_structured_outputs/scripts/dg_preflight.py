# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-fast compatibility check before spending A100 time on DiffusionGemma.

Run this FIRST. It checks GPU compute capability + memory, loads the checkpoint
config (no weights), detects NVFP4 (Blackwell-only) quantization, confirms vLLM
would see the model as a diffusion LM (``canvas_length``), and estimates whether
the weights fit. Prints a clear GO / NO-GO with reasons so we don't discover an
incompatibility mid-run.

Usage:
  python dg_preflight.py --model nvidia/diffusiongemma-26B-A4B-it-NVFP4
  python dg_preflight.py --model <bf16-diffusiongemma-repo>
"""

from __future__ import annotations

import argparse
import json
import subprocess


def gpu_info() -> list[dict]:
    """GPU name / memory / compute capability via nvidia-smi (avoids torch.cuda)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception as e:  # noqa: BLE001
        return [{"error": str(e)}]
    gpus = []
    for line in out.splitlines():
        name, mem, cap = (x.strip() for x in line.split(","))
        gpus.append({"name": name, "mem_gb": float(mem) / 1024, "compute_cap": cap})
    return gpus


def detect_nvfp4(cfg) -> bool:
    q = getattr(cfg, "quantization_config", None)
    blob = json.dumps(q, default=str).lower() if q else ""
    name = ""  # some configs carry a method/name field
    if isinstance(q, dict):
        name = str(q.get("quant_method", "")) + str(q.get("method", ""))
    return "nvfp4" in blob or "nvfp4" in name.lower() or "fp4" in blob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--gpu-mem-headroom-gb",
        type=float,
        default=12.0,
        help="reserve for activations/KV/canvas at batch 1",
    )
    args = ap.parse_args()

    print("=" * 64)
    print("DiffusionGemma preflight")
    print("=" * 64)

    gpus = gpu_info()
    print("\n[GPU]")
    for g in gpus:
        print(f"  {g}")
    caps = [g.get("compute_cap") for g in gpus if "compute_cap" in g]
    mem = max((g.get("mem_gb", 0) for g in gpus if "mem_gb" in g), default=0.0)
    sm_major = int(caps[0].split(".")[0]) if caps and caps[0] else 0

    from transformers import AutoConfig

    print("\n[checkpoint config]")
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tc = getattr(cfg, "text_config", cfg)
    canvas_length = getattr(cfg, "canvas_length", getattr(tc, "canvas_length", None))
    vocab = getattr(tc, "vocab_size", getattr(cfg, "vocab_size", None))
    is_nvfp4 = detect_nvfp4(cfg)
    for k in (
        "model_type",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "num_experts_per_tok",
        "num_local_experts",
    ):
        v = getattr(tc, k, getattr(cfg, k, None))
        if v is not None:
            print(f"  {k}: {v}")
    print(f"  vocab_size: {vocab}")
    print(
        f"  canvas_length: {canvas_length}  "
        f"(=> is_diffusion: {canvas_length is not None})"
    )
    print(f"  NVFP4/FP4 quantized: {is_nvfp4}")

    # weight-size estimate
    total_params = getattr(cfg, "num_parameters", None)
    print("\n[verdict]")
    ok = True
    if is_nvfp4 and sm_major < 10:
        ok = False
        print("  NO-GO: checkpoint is NVFP4 (Blackwell sm100/sm120 format) but this")
        print(
            f"         GPU is compute cap {caps[0] if caps else '?'} (Ampere/Hopper)."
        )
        print("         NVFP4 kernels cannot run here. Use a bf16 checkpoint, or a")
        print("         Blackwell GPU (RTX PRO 6000 / B200).")
    else:
        if is_nvfp4:
            wbytes, dtype = 0.5, "nvfp4(~4bit)"
        else:
            wbytes, dtype = 2.0, "bf16/fp16"
        if total_params:
            wgb = total_params * wbytes / 1e9
            print(
                f"  weights ~{wgb:.1f} GB ({dtype}, {total_params / 1e9:.1f}B params)"
            )
            need = wgb + args.gpu_mem_headroom_gb
            fits = need <= mem
            ok = ok and fits
            hr = args.gpu_mem_headroom_gb
            verdict = "FITS" if fits else "TOO BIG"
            print(
                f"  need ~{need:.1f} GB (weights + {hr} GB headroom) "
                f"vs {mem:.1f} GB avail -> {verdict}"
            )
        else:
            print(f"  weights: dtype {dtype}; num_parameters not in config -- estimate")
            print("    from the checkpoint safetensors total size on disk.")
        if canvas_length is None:
            print("  WARN: no canvas_length in config -> vLLM will NOT treat this as a")
            print("        diffusion model; check you have the right checkpoint.")
        print(
            f"  {'GO' if ok else 'REVIEW'}: proceed to dg_capture.py"
            if ok
            else "  REVIEW the above before capture."
        )

    print(f"\nNext: python dg_capture.py --model {args.model}")


if __name__ == "__main__":
    main()
