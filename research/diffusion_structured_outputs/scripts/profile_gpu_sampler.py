# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-stage profiler for the batched GPU constrained sampler (pod).

Breaks a full sample() into its stages and times each on CUDA to find the
bottleneck: transition-matrix build, product-tree build (Triton logmm), and
state-fill + token emission. Reports ms and share of total.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def timed(fn, reps=20):
    fn()
    torch.accelerator.synchronize()
    t = time.time()
    for _ in range(reps):
        r = fn()
    torch.accelerator.synchronize()
    return (time.time() - t) / reps, r


def main():
    if not torch.accelerator.is_available():
        print("no accelerator")
        return
    device = str(torch.accelerator.current_accelerator())
    gs.set_logmm_impl(tk.logmm_semiring)  # use the Triton tree
    dfa = build_schema_dfa(SCHEMA, VOCAB, pad_token_id=PAD)
    nxt, acc, start = gs.build_transition_table(dfa)
    nxt, acc = nxt.to(device), acc.to(device)
    V, N = len(VOCAB), dfa.num_states
    n_edges = int((nxt >= 0).sum())
    print(f"device={device} N={N} V={V} edges={n_edges}")

    for L in [128, 256]:
        B = 512
        lp = torch.tensor(
            np.random.default_rng(0).normal(size=(B, L, V)).astype(np.float32),
            device=device,
            dtype=torch.float32,
        )
        gen = torch.Generator(device=device).manual_seed(0)

        def tm(lp=lp):
            return gs.transition_matrices(lp, nxt)

        t_tm, M = timed(tm)

        def bl(M=M):
            return gs.build_levels(M)

        t_bl, levels = timed(bl)

        def sm(levels=levels, gen=gen, lp=lp):
            return gs.sample_from_levels(lp, levels, nxt, acc, start, gen)

        t_sm, _ = timed(sm)

        def full(lp=lp, gen=gen):
            return gs.sample(lp, nxt, acc, start, gen)

        t_full, _ = timed(full)
        tot = t_tm + t_bl + t_sm
        p_tm, p_bl, p_sm = 100 * t_tm / tot, 100 * t_bl / tot, 100 * t_sm / tot
        print(f"\nL={L} B={B}  full sample={t_full * 1e3:.1f}ms")
        print(f"  transition_matrices : {t_tm * 1e3:7.2f} ms  ({p_tm:4.1f}%)")
        print(f"  build_levels (tree) : {t_bl * 1e3:7.2f} ms  ({p_bl:4.1f}%)")
        print(f"  state-fill + emit   : {t_sm * 1e3:7.2f} ms  ({p_sm:4.1f}%)")


if __name__ == "__main__":
    main()
