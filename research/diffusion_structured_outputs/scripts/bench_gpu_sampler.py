# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the batched GPU log-depth constrained sampler on real hardware.

Runs gpu_constrained_sampler on CUDA over a batch of requests for increasing
canvas length L, reports throughput and the O(log L) tree depth, and cross-checks
logZ against the numpy CPU reference. Structured-output constraint = a JSON schema
compiled to a token DFA.
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


def bench(device: str) -> None:
    dfa = build_schema_dfa(SCHEMA, VOCAB, pad_token_id=PAD)
    nxt, acc, start = gs.build_transition_table(dfa)
    V, N = len(VOCAB), dfa.num_states
    print(f"device={device}  DFA states N={N}  vocab V={V}")
    nxt, acc = nxt.to(device), acc.to(device)
    dtype = torch.float32

    # min valid JSON for this schema is ~33 tokens; keep L above that.
    for L in [48, 64, 128, 256]:
        for B in [128, 512]:
            rng = np.random.default_rng(0)
            lp = torch.tensor(
                rng.normal(size=(B, L, V)).astype(np.float32),
                device=device,
                dtype=dtype,
            )
            gen = torch.Generator(device=device).manual_seed(0)
            # warmup + time
            gs.sample(lp, nxt, acc, start, gen)
            if device != "cpu":
                torch.accelerator.synchronize()
            t = time.time()
            reps = 5
            for _ in range(reps):
                tokens = gs.sample(lp, nxt, acc, start, gen)
            if device != "cpu":
                torch.accelerator.synchronize()
            dt = (time.time() - t) / reps
            depth = max(1, (L - 1).bit_length())
            canvases_per_s = B / dt
            # acceptance spot check
            ok = all(dfa.accepts(row) for row in tokens[:32].tolist())
            print(
                f"  L={L:4d} B={B:4d}  {dt * 1e3:7.1f} ms/batch  "
                f"{canvases_per_s:9.0f} canvases/s  tree_depth={depth}  accepted={ok}"
            )

    # logZ cross-check vs numpy reference (one request)
    L = 64
    lp = np.random.default_rng(1).normal(size=(L, V)).astype(np.float64)
    lp_t = torch.tensor(lp[None], device=device, dtype=torch.float64)
    nxt64 = nxt.to(device)
    levels = gs.build_levels(gs.transition_matrices(lp_t, nxt64))
    z_gpu = float(gs.log_partition(levels, start, acc.to(device))[0])
    z_np = cs.log_partition(lp, dfa)
    print(
        f"logZ cross-check L={L}: gpu={z_gpu:.5f}  numpy={z_np:.5f}  "
        f"match={abs(z_gpu - z_np) < 1e-4}"
    )


if __name__ == "__main__":
    has_accel = torch.accelerator.is_available()
    dev = str(torch.accelerator.current_accelerator()) if has_accel else "cpu"
    if not has_accel:
        print("WARNING: no accelerator available; running on CPU")
    print(f"torch {torch.__version__}  device={dev}")
    bench(dev)
