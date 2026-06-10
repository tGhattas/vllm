#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Quantify the numerical divergence between the two Mamba2 decode paths
(vllm-project/vllm issue #43301).

A single decode token can be computed by two kernels that are algebraically
equivalent but numerically different:

- Flow A: ``mamba_chunk_scan_combined_varlen`` called with a 1-token sequence
  and ``initial_states`` (the chunked-prefill / pre-#42430 recompute path).
  Internally uses bf16/fp16 ``tl.dot`` with low-precision intermediate casts.
- Flow B: ``selective_state_update`` (the single-token decode / post-#42430
  path). Performs the SSM step entirely in fp32.

Both are compared against:

- Flow C: a pure-PyTorch reference recurrence executed in fp64 (the arbiter)
  and in fp32 (to bound pure-dtype effects).

All flows start from the *same* prefill state, produced by running
``mamba_chunk_scan_combined_varlen`` over P prompt tokens, so that the
reported numbers isolate the per-decode-step divergence.

Usage:
    # Full sweep on a GPU machine (writes markdown to stdout + JSON file):
    python benchmarks/kernels/check_mamba_kernel_divergence.py

    # Small, fast configuration:
    python benchmarks/kernels/check_mamba_kernel_divergence.py --quick

    # CPU-only validation of the reference implementation:
    python benchmarks/kernels/check_mamba_kernel_divergence.py --self-test
"""

import argparse
import dataclasses
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import torch
import torch.nn.functional as F

# NOTE: vLLM (and therefore Triton/CUDA) is imported lazily inside the GPU
# code paths only, so that ``--self-test`` runs on machines without a GPU
# and without a vLLM installation.

# Default shapes follow the Mamba2 layers of ibm-granite/granite-4.0-h-tiny
# (https://huggingface.co/ibm-granite/granite-4.0-h-tiny/blob/main/config.json):
#   mamba_n_heads=48, mamba_d_head=64, mamba_d_state=128, mamba_n_groups=1,
#   mamba_chunk_size=256, torch_dtype=bfloat16.
GRANITE_TINY_CONFIG = {
    "nheads": 48,
    "headdim": 64,
    "dstate": 128,
    "ngroups": 1,
    "chunk_size": 256,
}

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@dataclasses.dataclass
class SSMInputs:
    """Synthetic inputs shaped/scaled like MambaMixer2 produces them.

    All tensors are generated in fp64 ("golden" values) and downcast per
    flow, so every flow consumes bit-identical starting data.
    """

    x: torch.Tensor  # (seqlen, nheads, headdim)
    dt: torch.Tensor  # (seqlen, nheads), raw (pre-bias, pre-softplus)
    A: torch.Tensor  # (nheads,), negative
    B: torch.Tensor  # (seqlen, ngroups, dstate)
    C: torch.Tensor  # (seqlen, ngroups, dstate)
    D: torch.Tensor  # (nheads,)
    dt_bias: torch.Tensor  # (nheads,)


def generate_inputs(
    seqlen: int,
    nheads: int,
    headdim: int,
    dstate: int,
    ngroups: int,
    seed: int,
    device: torch.device,
) -> SSMInputs:
    """Generate inputs matching the magnitude/distribution seen by the kernels.

    In MambaMixer2, x/B/C are slices of the causal-conv1d output after a SiLU
    activation, so they follow a silu(gaussian) distribution. dt is a raw
    linear-projection output; the kernel adds dt_bias and applies softplus
    internally (dt_softplus=True). A is loaded as -exp(A_log) with the Mamba2
    init A ~ Uniform(1, 16); dt_bias is the inverse softplus of a
    LogUniform(1e-3, 1e-1) target step size (the standard Mamba2 dt init).
    D is initialized to ones (small jitter added to avoid exact symmetry).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    def rand(*shape):
        return torch.rand(*shape, generator=gen, dtype=torch.float64)

    x = F.silu(randn(seqlen, nheads, headdim))
    B = F.silu(randn(seqlen, ngroups, dstate))
    C = F.silu(randn(seqlen, ngroups, dstate))
    dt = randn(seqlen, nheads) * 0.5
    A = -(1.0 + 15.0 * rand(nheads))
    # dt_target in [1e-3, 1e-1], log-uniform; dt_bias = softplus^-1(dt_target)
    dt_target = torch.exp(
        rand(nheads) * (math.log(1e-1) - math.log(1e-3)) + math.log(1e-3)
    )
    dt_bias = dt_target + torch.log(-torch.expm1(-dt_target))
    D = 1.0 + randn(nheads) * 0.1

    return SSMInputs(
        x=x.to(device),
        dt=dt.to(device),
        A=A.to(device),
        B=B.to(device),
        C=C.to(device),
        D=D.to(device),
        dt_bias=dt_bias.to(device),
    )


def quantize_inputs(inputs: SSMInputs, dtype: torch.dtype) -> SSMInputs:
    """Quantize inputs once, as the model would hold them at inference time.

    Activations (x, dt, B, C) and the dt_bias/D parameters are stored in the
    model dtype; A is materialized in fp32 (``-exp(A_log.float())`` in
    MambaMixer2). Every flow — including the fp64 reference — then consumes
    these *identical* quantized values, so the measured divergence reflects
    kernel arithmetic only, not input quantization.
    """
    return SSMInputs(
        x=inputs.x.to(dtype),
        dt=inputs.dt.to(dtype),
        A=inputs.A.to(torch.float32),
        B=inputs.B.to(dtype),
        C=inputs.C.to(dtype),
        D=inputs.D.to(dtype),
        dt_bias=inputs.dt_bias.to(dtype),
    )


# ---------------------------------------------------------------------------
# Flow C: pure-PyTorch reference recurrence
# ---------------------------------------------------------------------------


def reference_ssm_step(
    state: torch.Tensor,  # (nheads, headdim, dstate), compute dtype
    x_t: torch.Tensor,  # (nheads, headdim)
    dt_t: torch.Tensor,  # (nheads,), raw
    A: torch.Tensor,  # (nheads,)
    B_t: torch.Tensor,  # (ngroups, dstate)
    C_t: torch.Tensor,  # (ngroups, dstate)
    D: torch.Tensor,  # (nheads,)
    dt_bias: torch.Tensor,  # (nheads,)
) -> tuple[torch.Tensor, torch.Tensor]:
    """One SSM decode step: returns (y_t, new_state).

    Computes, per head h (g = group of h):
        dt'   = softplus(dt[h] + dt_bias[h])
        dA    = exp(dt' * A[h])
        state = state * dA + dt' * outer(x[h], B[g])
        y[h]  = state @ C[g] + D[h] * x[h]

    All math in ``state.dtype`` (fp64 or fp32).
    """
    nheads, headdim, dstate = state.shape
    ngroups = B_t.shape[0]
    dtype = state.dtype

    x_t = x_t.to(dtype)
    dt_eff = F.softplus(dt_t.to(dtype) + dt_bias.to(dtype))  # (nheads,)
    dA = torch.exp(dt_eff * A.to(dtype))  # (nheads,)

    B_h = B_t.to(dtype).repeat_interleave(nheads // ngroups, dim=0)
    C_h = C_t.to(dtype).repeat_interleave(nheads // ngroups, dim=0)

    dBx = (dt_eff[:, None, None] * x_t[:, :, None]) * B_h[:, None, :]
    new_state = state * dA[:, None, None] + dBx
    y = torch.einsum("hpn,hn->hp", new_state, C_h) + D.to(dtype)[:, None] * x_t
    return y, new_state


def reference_ssm_step_bruteforce(
    state: torch.Tensor,
    x_t: torch.Tensor,
    dt_t: torch.Tensor,
    A: torch.Tensor,
    B_t: torch.Tensor,
    C_t: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scalar-loop version of :func:`reference_ssm_step` for --self-test."""
    nheads, headdim, dstate = state.shape
    ngroups = B_t.shape[0]
    dtype = state.dtype
    new_state = torch.empty_like(state)
    y = torch.empty(nheads, headdim, dtype=dtype)
    for h in range(nheads):
        g = h // (nheads // ngroups)
        dt_eff = float(F.softplus(dt_t[h].to(dtype) + dt_bias[h].to(dtype)))
        dA = math.exp(dt_eff * float(A[h]))
        for p in range(headdim):
            acc = 0.0
            for n in range(dstate):
                s = float(state[h, p, n]) * dA + dt_eff * float(x_t[h, p]) * float(
                    B_t[g, n]
                )
                new_state[h, p, n] = s
                acc += s * float(C_t[g, n])
            y[h, p] = acc + float(D[h]) * float(x_t[h, p])
    return y, new_state


def run_reference_decode(
    initial_state: torch.Tensor,  # (nheads, headdim, dstate), any dtype
    inputs: SSMInputs,
    start: int,
    nsteps: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``nsteps`` reference decode steps; returns (ys, states).

    ys: (nsteps, nheads, headdim); states: (nsteps, nheads, headdim, dstate).
    """
    state = initial_state.to(dtype)
    ys, states = [], []
    for i in range(nsteps):
        t = start + i
        y, state = reference_ssm_step(
            state,
            inputs.x[t],
            inputs.dt[t],
            inputs.A,
            inputs.B[t],
            inputs.C[t],
            inputs.D,
            inputs.dt_bias,
        )
        ys.append(y)
        states.append(state)
    return torch.stack(ys), torch.stack(states)


# ---------------------------------------------------------------------------
# GPU flows (lazy vLLM imports)
# ---------------------------------------------------------------------------


def run_kernel_prefill(
    inputs: SSMInputs,
    prefill_len: int,
    chunk_size: int,
    state_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefill P tokens with mamba_chunk_scan_combined_varlen.

    ``inputs`` must already be quantized (see :func:`quantize_inputs`).
    Returns (out, final_state) with final_state (nheads, headdim, dstate)
    in ``state_dtype``.
    """
    from vllm.model_executor.layers.mamba.ops.ssd_combined import (
        mamba_chunk_scan_combined_varlen,
    )
    from vllm.v1.attention.backends.mamba2_attn import compute_varlen_chunk_metadata

    device = inputs.x.device
    x = inputs.x[:prefill_len]

    cu_seqlens = torch.tensor([0, prefill_len], dtype=torch.int32, device=device)
    cu_chunk_seqlens, last_chunk_indices, seq_idx = compute_varlen_chunk_metadata(
        cu_seqlens, chunk_size
    )
    out = torch.empty_like(x)
    final_state = mamba_chunk_scan_combined_varlen(
        x,
        inputs.dt[:prefill_len],
        inputs.A,
        inputs.B[:prefill_len],
        inputs.C[:prefill_len],
        chunk_size,
        cu_seqlens=cu_seqlens,
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk_indices=last_chunk_indices,
        seq_idx=seq_idx,
        out=out,
        D=inputs.D,
        dt_bias=inputs.dt_bias,
        dt_softplus=True,
        state_dtype=state_dtype,
    )
    return out, final_state.squeeze(0)


def run_flow_a_chunk_scan(
    initial_state: torch.Tensor,  # (nheads, headdim, dstate), state_dtype
    inputs: SSMInputs,
    start: int,
    nsteps: int,
    chunk_size: int,
    state_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flow A: decode via 1-token mamba_chunk_scan_combined_varlen calls.

    The state round-trips through ``state_dtype`` between steps, matching
    how the SSM cache behaves in vLLM. Returns (ys, states).
    """
    from vllm.model_executor.layers.mamba.ops.ssd_combined import (
        mamba_chunk_scan_combined_varlen,
    )
    from vllm.v1.attention.backends.mamba2_attn import compute_varlen_chunk_metadata

    device = inputs.x.device
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32, device=device)
    cu_chunk_seqlens, last_chunk_indices, seq_idx = compute_varlen_chunk_metadata(
        cu_seqlens, chunk_size
    )

    state = initial_state.unsqueeze(0).to(state_dtype)
    ys, states = [], []
    for i in range(nsteps):
        t = start + i
        x_t = inputs.x[t : t + 1]
        out = torch.empty_like(x_t)
        state = mamba_chunk_scan_combined_varlen(
            x_t,
            inputs.dt[t : t + 1],
            inputs.A,
            inputs.B[t : t + 1],
            inputs.C[t : t + 1],
            chunk_size,
            cu_seqlens=cu_seqlens,
            cu_chunk_seqlens=cu_chunk_seqlens,
            last_chunk_indices=last_chunk_indices,
            seq_idx=seq_idx,
            out=out,
            D=inputs.D,
            dt_bias=inputs.dt_bias,
            initial_states=state,
            dt_softplus=True,
            state_dtype=state_dtype,
        )
        ys.append(out.squeeze(0))
        states.append(state.squeeze(0).clone())
    return torch.stack(ys), torch.stack(states)


def run_flow_b_selective_state_update(
    initial_state: torch.Tensor,  # (nheads, headdim, dstate), state_dtype
    inputs: SSMInputs,
    start: int,
    nsteps: int,
    state_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flow B: decode via selective_state_update (Triton backend).

    Mirrors the MambaMixer2 decode call: A is expanded to
    (nheads, headdim, dstate) in fp32 and dt/dt_bias/D are expanded across
    headdim with stride-0 views, so the kernel takes its TIE_HDIM fast path
    exactly as in real inference. Returns (ys, states).
    """
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update

    nheads, headdim, dstate = initial_state.shape
    state = initial_state.unsqueeze(0).to(state_dtype).clone()

    # Stride-0 expansions, as in MambaMixer2.forward_cuda's decode branch.
    A_d = inputs.A[:, None, None].expand(nheads, headdim, dstate)
    dt_bias_d = inputs.dt_bias[:, None].expand(nheads, headdim)
    D_d = inputs.D[:, None].expand(nheads, headdim)

    ys, states = [], []
    for i in range(nsteps):
        t = start + i
        x_t = inputs.x[t : t + 1]  # (1, nheads, headdim)
        dt_t = inputs.dt[t : t + 1][:, :, None].expand(1, nheads, headdim)
        out = torch.empty_like(x_t)
        selective_state_update(
            state,
            x_t,
            dt_t,
            A_d,
            inputs.B[t : t + 1],
            inputs.C[t : t + 1],
            D_d,
            dt_bias_d,
            dt_softplus=True,
            out=out,
        )
        ys.append(out.squeeze(0))
        states.append(state.squeeze(0).clone())
    return torch.stack(ys), torch.stack(states)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def error_stats(
    a: torch.Tensor, b: torch.Tensor, ref: torch.Tensor
) -> dict[str, float]:
    """Error stats of (a - b), relative to ``ref`` magnitudes."""
    a64 = a.to(torch.float64)
    b64 = b.to(torch.float64)
    ref64 = ref.to(torch.float64)
    diff = (a64 - b64).abs()
    rel = diff / ref64.abs().clamp_min(1e-8)
    rel_flat = rel.flatten()
    return {
        "max_abs": diff.max().item(),
        "rel_mean": rel_flat.mean().item(),
        "rel_p50": rel_flat.quantile(0.5).item(),
        "rel_p99": rel_flat.quantile(0.99).item(),
    }


def compare_flows(
    ys: dict[str, torch.Tensor],  # flow -> (nsteps, nheads, headdim)
    states: dict[str, torch.Tensor],
    nsteps: int,
) -> dict[str, Any]:
    """Per-step and cumulative metrics for A-C, B-C, A-B (+ state drift)."""
    pairs = [("A", "C"), ("B", "C"), ("A", "B")]
    per_step: list[dict[str, Any]] = []
    for i in range(nsteps):
        row: dict[str, Any] = {"step": i}
        for lhs, rhs in pairs:
            row[f"{lhs}-{rhs}"] = error_stats(ys[lhs][i], ys[rhs][i], ys["C"][i])
            row[f"state_{lhs}-{rhs}"] = error_stats(
                states[lhs][i], states[rhs][i], states["C"][i]
            )
        per_step.append(row)
    cumulative = {
        f"{lhs}-{rhs}": error_stats(ys[lhs], ys[rhs], ys["C"]) for lhs, rhs in pairs
    }
    cumulative.update(
        {
            f"state_{lhs}-{rhs}": error_stats(
                states[lhs][-1], states[rhs][-1], states["C"][-1]
            )
            for lhs, rhs in pairs
        }
    )
    return {"per_step": per_step, "cumulative": cumulative}


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


def run_case(
    args: argparse.Namespace,
    dtype: torch.dtype,
    prefill_len: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    state_dtype = dtype if args.state_dtype == "same" else DTYPE_MAP[args.state_dtype]
    nsteps = args.decode_steps
    seqlen = prefill_len + nsteps
    inputs = quantize_inputs(
        generate_inputs(
            seqlen, args.nheads, args.headdim, args.dstate, args.ngroups, seed, device
        ),
        dtype,
    )

    _, prefill_state = run_kernel_prefill(
        inputs, prefill_len, args.chunk_size, state_dtype
    )

    ys_a, st_a = run_flow_a_chunk_scan(
        prefill_state, inputs, prefill_len, nsteps, args.chunk_size, state_dtype
    )
    ys_b, st_b = run_flow_b_selective_state_update(
        prefill_state, inputs, prefill_len, nsteps, state_dtype
    )
    ys_c64, st_c64 = run_reference_decode(
        prefill_state, inputs, prefill_len, nsteps, torch.float64
    )
    ys_c32, st_c32 = run_reference_decode(
        prefill_state, inputs, prefill_len, nsteps, torch.float32
    )

    metrics = compare_flows(
        {"A": ys_a, "B": ys_b, "C": ys_c64},
        {"A": st_a, "B": st_b, "C": st_c64},
        nsteps,
    )
    # fp32 reference vs fp64 reference: bounds the error attributable to
    # running the recurrence itself in fp32 (vs kernel-specific effects).
    metrics["cumulative"]["C32-C64"] = error_stats(ys_c32, ys_c64, ys_c64)
    metrics["cumulative"]["state_C32-C64"] = error_stats(
        st_c32[-1], st_c64[-1], st_c64[-1]
    )

    return {
        "dtype": dtype_name(dtype),
        "state_dtype": dtype_name(state_dtype),
        "prefill_len": prefill_len,
        "decode_steps": nsteps,
        "seed": seed,
        **metrics,
    }


def dtype_name(dtype: torch.dtype) -> str:
    return {v: k for k, v in DTYPE_MAP.items()}.get(dtype, str(dtype))


def collect_env_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        info["commit"] = "unknown"
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["gpu_name"] = props.name
        info["compute_capability"] = f"{props.major}.{props.minor}"
        info["cuda"] = torch.version.cuda
        try:
            import triton

            info["triton"] = triton.__version__
        except ImportError:
            info["triton"] = "unavailable"
        try:
            import vllm

            info["vllm"] = vllm.__version__
        except ImportError:
            info["vllm"] = "unavailable"
    return info


def format_markdown(results: list[dict[str, Any]], env: dict[str, Any]) -> str:
    lines = [
        "## Mamba2 decode-path divergence (issue #43301)",
        "",
        f"Env: {env.get('gpu_name', 'cpu')} (CC {env.get('compute_capability', '-')}),"
        f" torch {env['torch']}, triton {env.get('triton', '-')},"
        f" vllm {env.get('vllm', '-')}, commit {env['commit'][:9]}",
        "",
        "Flows: A = 1-token `mamba_chunk_scan_combined_varlen` w/ initial_states,",
        "B = `selective_state_update`, C = fp64 PyTorch reference.",
        "Relative errors are w.r.t. the fp64 reference magnitudes.",
        "",
        "| dtype | state | P | seed | pair | max_abs | rel_mean | rel_p50 |"
        " rel_p99 | state rel_p99 (last) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        for pair in ("A-C", "B-C", "A-B", "C32-C64"):
            cum = r["cumulative"][pair]
            st = r["cumulative"][f"state_{pair}"]
            lines.append(
                f"| {r['dtype']} | {r['state_dtype']} | {r['prefill_len']} |"
                f" {r['seed']} | {pair} | {cum['max_abs']:.3e} |"
                f" {cum['rel_mean']:.3e} | {cum['rel_p50']:.3e} |"
                f" {cum['rel_p99']:.3e} | {st['rel_p99']:.3e} |"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test (CPU-only)
# ---------------------------------------------------------------------------


def self_test() -> int:
    """Validate the vectorized fp64 reference against a brute-force loop."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    nheads, headdim, dstate, ngroups = 4, 8, 16, 2
    nsteps = 3
    inputs = generate_inputs(nsteps, nheads, headdim, dstate, ngroups, 0, device)
    state0 = torch.randn(nheads, headdim, dstate, dtype=torch.float64)

    for dtype in (torch.float64, torch.float32):
        ys_vec, st_vec = run_reference_decode(state0, inputs, 0, nsteps, dtype)
        state_bf = state0.to(dtype)
        for i in range(nsteps):
            y_bf, state_bf = reference_ssm_step_bruteforce(
                state_bf,
                inputs.x[i].to(dtype),
                inputs.dt[i].to(dtype),
                inputs.A.to(dtype),
                inputs.B[i].to(dtype),
                inputs.C[i].to(dtype),
                inputs.D.to(dtype),
                inputs.dt_bias.to(dtype),
            )
            tol = 1e-12 if dtype == torch.float64 else 1e-5
            torch.testing.assert_close(ys_vec[i], y_bf, atol=tol, rtol=tol)
            torch.testing.assert_close(st_vec[i], state_bf, atol=tol, rtol=tol)

    # Sanity: state contracts (A < 0 => |dA| < 1) and outputs are finite.
    assert torch.isfinite(ys_vec).all() and torch.isfinite(st_vec).all()
    print("self-test passed: fp64/fp32 reference matches brute-force recurrence")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--nheads", type=int, default=GRANITE_TINY_CONFIG["nheads"])
    p.add_argument("--headdim", type=int, default=GRANITE_TINY_CONFIG["headdim"])
    p.add_argument("--dstate", type=int, default=GRANITE_TINY_CONFIG["dstate"])
    p.add_argument("--ngroups", type=int, default=GRANITE_TINY_CONFIG["ngroups"])
    p.add_argument("--chunk-size", type=int, default=GRANITE_TINY_CONFIG["chunk_size"])
    p.add_argument(
        "--prefill-lens",
        type=int,
        nargs="+",
        default=[64, 512],
        help="Prompt lengths processed by the chunked prefill kernel.",
    )
    p.add_argument(
        "--decode-steps",
        type=int,
        default=32,
        help="Number of single-token decode steps per flow.",
    )
    p.add_argument(
        "--dtypes",
        nargs="+",
        choices=list(DTYPE_MAP),
        default=["bf16", "fp16", "fp32"],
        help="Input/output dtypes to sweep.",
    )
    p.add_argument(
        "--state-dtype",
        choices=["same", *DTYPE_MAP],
        default="same",
        help="SSM cache dtype; 'same' mirrors vLLM's default "
        "(mamba_ssm_cache_dtype='auto' -> model dtype).",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--quick", action="store_true", help="One small configuration only.")
    p.add_argument(
        "--self-test",
        action="store_true",
        help="CPU-only validation of the reference recurrence; no GPU needed.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()

    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA device required (use --self-test for CPU-only checks).",
            file=sys.stderr,
        )
        return 1
    device = torch.device("cuda")

    if args.quick:
        args.prefill_lens = [64]
        args.decode_steps = 8
        args.seeds = args.seeds[:1]

    dtypes = []
    for name in args.dtypes:
        if name == "bf16" and not torch.cuda.is_bf16_supported():
            print("WARNING: skipping bf16 (no native support on this GPU)")
            continue
        dtypes.append(DTYPE_MAP[name])

    results = []
    for dtype in dtypes:
        for prefill_len in args.prefill_lens:
            for seed in args.seeds:
                print(
                    f"running dtype={dtype_name(dtype)} P={prefill_len} seed={seed}...",
                    file=sys.stderr,
                )
                results.append(run_case(args, dtype, prefill_len, seed, device))

    env = collect_env_info(device)
    print(format_markdown(results, env))

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    gpu_tag = env.get("gpu_name", "cpu").replace(" ", "_")
    out_path = os.path.join(args.results_dir, f"divergence_{gpu_tag}_{stamp}.json")
    with open(out_path, "w") as f:
        json.dump({"env": env, "args": vars(args), "results": results}, f, indent=2)
    print(f"\nJSON written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
