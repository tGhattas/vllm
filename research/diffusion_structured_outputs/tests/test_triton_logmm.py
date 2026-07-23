# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-side import/safety checks for the Triton log-semiring matmul.

The kernel itself only runs on CUDA (validated + benchmarked on the A40 via
scripts/bench_triton_logmm.py); here we only assert the module imports without
Triton/CUDA and that the sampler's swappable-logmm hook works.
"""

import gpu_constrained_sampler as gs
import numpy as np
import torch
import triton_logmm as tk
from test_constrained_sampler import parity_dfa, random_logprobs


def test_module_imports_and_flag_is_bool():
    assert isinstance(tk.HAS_TRITON, bool)


def test_logmm_semiring_requires_triton_on_cpu():
    if tk.HAS_TRITON:
        return  # on a Triton host this path isn't exercised
    import pytest

    with pytest.raises(RuntimeError):
        tk.logmm_semiring(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2))


def test_hook_is_ignored_for_cpu_tensors():
    # Installing an impl must not affect CPU tensors (dispatch is CUDA-only).
    called = {"n": 0}

    def spy(a, b):
        called["n"] += 1
        return a

    gs.set_logmm_impl(spy)
    try:
        L, V = 5, 3
        dfa = parity_dfa(L, V)
        nxt, acc, start = gs.build_transition_table(dfa)
        lp = torch.tensor(random_logprobs(L, V, seed=1)[None], dtype=torch.float64)
        z = gs.log_partition(
            gs.build_levels(gs.transition_matrices(lp, nxt)), start, acc
        )
        assert np.isfinite(float(z[0]))
        assert called["n"] == 0  # CPU path never calls the custom impl
    finally:
        gs.set_logmm_impl(None)
