# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fine-grained profiler for the GPU constrained sampler (pod).

Breaks the two hot stages further: per-level timing of the product tree, and the
sub-steps of transition_matrices (edge index / gather / scatter-logsumexp). Also
reports a dtype experiment. Used to target the next optimization.
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


def sync():
    torch.accelerator.synchronize()


def timed(fn, reps=20):
    fn()
    sync()
    t = time.time()
    for _ in range(reps):
        r = fn()
    sync()
    return (time.time() - t) / reps, r


def profile_tree(M):
    """Per-level timing of the log-semiring product tree."""
    B, L, N, _ = M.shape
    Lpad = 1 if L <= 1 else 1 << (L - 1).bit_length()
    if Lpad > L:
        ident = gs._identity(N, M.device, M.dtype).expand(B, Lpad - L, N, N)
        cur = torch.cat([M, ident], dim=1)
    else:
        cur = M
    lvl = 0
    while cur.shape[1] > 1:
        a, b = cur.contiguous()[:, 0::2], cur.contiguous()[:, 1::2]

        def step(a=a, b=b):
            return gs._logmm(a, b)

        dt, nxt = timed(step, reps=30)
        print(f"    level {lvl}: nodes={cur.shape[1] // 2:4d}  {dt * 1e3:6.2f} ms")
        cur = nxt
        lvl += 1


def profile_tm(lp, nxt):
    """Sub-step timing of transition_matrices."""
    N = nxt.shape[0]

    def edges():
        s_idx, v_idx = (nxt >= 0).nonzero(as_tuple=True)
        return s_idx, v_idx, s_idx * N + nxt[s_idx, v_idx]

    dt_e, (s_idx, v_idx, dst) = timed(edges, reps=50)

    def gather(v_idx=v_idx):
        return lp[:, :, v_idx]

    dt_g, vals = timed(gather)

    def scat(vals=vals, dst=dst):
        return gs._scatter_logsumexp(vals, dst, N * N)

    dt_s, _ = timed(scat)
    print(
        f"    edge-index={dt_e * 1e3:6.2f}ms  gather={dt_g * 1e3:6.2f}ms  "
        f"scatter-lse={dt_s * 1e3:6.2f}ms"
    )


def main():
    if not torch.accelerator.is_available():
        print("no accelerator")
        return
    device = str(torch.accelerator.current_accelerator())
    gs.set_logmm_impl(tk.logmm_semiring)
    dfa = build_schema_dfa(SCHEMA, VOCAB, pad_token_id=PAD)
    nxt, acc, start = gs.build_transition_table(dfa)
    nxt, acc = nxt.to(device), acc.to(device)
    V, N = len(VOCAB), dfa.num_states
    print(f"device={device} N={N} V={V} edges={int((nxt >= 0).sum())}")

    for L in [128, 256]:
        B = 512
        lp = torch.tensor(
            np.random.default_rng(0).normal(size=(B, L, V)).astype(np.float32),
            device=device,
            dtype=torch.float32,
        )
        print(f"\n=== L={L} B={B} ===")

        def tm(lp=lp):
            return gs.transition_matrices(lp, nxt)

        t_tm, M = timed(tm)
        print(f"  transition_matrices total: {t_tm * 1e3:.2f} ms")
        profile_tm(lp, nxt)

        def bl(M=M):
            return gs.build_levels(M)

        t_bl, _ = timed(bl)
        print(f"  build_levels total: {t_bl * 1e3:.2f} ms")
        profile_tree(M)


if __name__ == "__main__":
    main()
