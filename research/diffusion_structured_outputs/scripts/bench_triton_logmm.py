# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate + benchmark the Triton log-semiring matmul on GPU (pod).

1. Correctness: Triton logmm vs the torch logaddexp reference (incl. -inf entries).
2. End-to-end: install the Triton kernel in the sampler; logZ matches numpy, all
   canvases DFA-accepted.
3. Speedup: time the product-tree build (build_levels) and full sample() with the
   torch logmm vs the Triton kernel.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constrained_sampler as cs  # noqa: E402
import gpu_constrained_sampler as gs  # noqa: E402
import triton_logmm as tk  # noqa: E402
from schema_compiler import build_schema_dfa  # noqa: E402

_CHARS = list('{}[]":,.-') + list("0123456789") + list("abcdefghijklmnopqrstuvwxyz")
VOCAB = _CHARS + [""]
PAD = len(_CHARS)
SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "active": {"type": "boolean"},
    },
    "order": ["name", "age", "active"],
}


def _torch_logmm(a, b):
    n = a.shape[-1]
    out = a[..., :, 0:1] + b[..., 0:1, :]
    for j in range(1, n):
        out = torch.logaddexp(out, a[..., :, j : j + 1] + b[..., j : j + 1, :])
    return out


def correctness(device):
    torch.manual_seed(0)
    for BK, N in [(1000, 8), (500, 33), (300, 64)]:
        a = torch.randn(BK, N, N, device=device)
        b = torch.randn(BK, N, N, device=device)
        # sprinkle -inf to mimic dead transitions
        a[torch.rand_like(a) < 0.3] = float("-inf")
        b[torch.rand_like(b) < 0.3] = float("-inf")
        ref = _torch_logmm(a, b)
        got = tk.logmm_semiring(a, b)
        diff = (got - ref).abs()
        diff = diff[torch.isfinite(ref)]
        maxd = float(diff.max()) if diff.numel() else 0.0
        both_inf = ((got == float("-inf")) == (ref == float("-inf"))).all().item()
        print(f"  BK={BK:5d} N={N:3d}  max|d|={maxd:.2e}  inf_match={both_inf}")


def timed(fn, sync, reps=10):
    fn()
    sync()
    t = time.time()
    for _ in range(reps):
        fn()
    sync()
    return (time.time() - t) / reps


def main():
    if not torch.accelerator.is_available():
        print("no accelerator; Triton kernel needs CUDA")
        return
    device = str(torch.accelerator.current_accelerator())
    print(f"torch {torch.__version__}  device={device}  HAS_TRITON={tk.HAS_TRITON}")

    print("== correctness: triton logmm vs torch reference ==")
    correctness(device)

    dfa = build_schema_dfa(SCHEMA, VOCAB, pad_token_id=PAD)
    nxt, acc, start = gs.build_transition_table(dfa)
    nxt, acc = nxt.to(device), acc.to(device)
    V = len(VOCAB)

    # end-to-end with the Triton kernel installed
    gs.set_logmm_impl(tk.logmm_semiring)
    lp1 = np.random.default_rng(1).normal(size=(64, V)).astype(np.float64)
    lp1_t = torch.tensor(lp1[None], device=device, dtype=torch.float64)
    z_tr = float(
        gs.log_partition(
            gs.build_levels(gs.transition_matrices(lp1_t, nxt)), start, acc
        )[0]
    )
    z_np = cs.log_partition(lp1, dfa)
    print(
        f"== end-to-end: logZ triton={z_tr:.5f} numpy={z_np:.5f} "
        f"match={abs(z_tr - z_np) < 1e-3} =="
    )

    def sync():
        torch.accelerator.synchronize()

    print("== speedup: build_levels + sample (torch logmm vs triton) ==")
    for L in [64, 128, 256]:
        B = 512
        lp = torch.tensor(
            np.random.default_rng(0).normal(size=(B, L, V)).astype(np.float32),
            device=device,
            dtype=torch.float32,
        )
        M = gs.transition_matrices(lp, nxt)
        gen = torch.Generator(device=device).manual_seed(0)

        def build(M=M):
            return gs.build_levels(M)

        def draw(lp=lp, gen=gen):
            return gs.sample(lp, nxt, acc, start, gen)

        gs.set_logmm_impl(None)
        t_tree_torch = timed(build, sync)
        t_samp_torch = timed(draw, sync)

        gs.set_logmm_impl(tk.logmm_semiring)
        t_tree_tri = timed(build, sync)
        t_samp_tri = timed(draw, sync)

        tokens = gs.sample(lp, nxt, acc, start, gen)
        ok = all(dfa.accepts(r) for r in tokens[:16].tolist())
        tt = f"{t_tree_torch * 1e3:.1f}->{t_tree_tri * 1e3:.1f}ms"
        ss = f"{t_samp_torch * 1e3:.1f}->{t_samp_tri * 1e3:.1f}ms"
        tx = t_tree_torch / t_tree_tri
        sx = t_samp_torch / t_samp_tri
        print(
            f"  L={L:4d} B={B}  tree {tt} ({tx:.1f}x)  sample {ss} ({sx:.1f}x)  ok={ok}"
        )
    gs.set_logmm_impl(None)


if __name__ == "__main__":
    main()
