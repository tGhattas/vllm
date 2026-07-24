# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canvas-aware constrained decoding for diffusion LMs (experimental, opt-in).

Given a diffusion model's per-position logits over a fixed canvas and a token-level
DFA, this computes the **constrained mean-field marginals** ``P_D(x_i = v)`` — the
per-position distribution conditioned on the whole canvas being accepted by the DFA
(Dang & Ermon, arXiv:2607.07026) — via a numerically stable log-space
forward/backward pass. Replacing the sampler's logits with these marginals steers
every denoising step toward a DFA-valid canvas.

This is loaded only when ``VLLM_DIFFUSION_CONSTRAINT`` points at a precompiled
constraint file (built offline from a schema/regex + the model tokenizer), so the
default diffusion path and the request-time structured-output guard are untouched.

The DFA is a partial function: ``next_state[s, v] == -1`` means token ``v`` is not
allowed from state ``s``. Only defined edges carry probability mass; every other
token gets ``-inf`` (never sampled).
"""

from __future__ import annotations

import torch

NEG_INF = float("-inf")


class DiffusionConstraint:
    """A token-DFA constraint that reweights canvas logits to its accepted language."""

    def __init__(
        self,
        num_states: int,
        vocab_size: int,
        start: int,
        accepting: torch.Tensor,  # [N] bool
        edge_s: torch.Tensor,  # [E] source states
        edge_v: torch.Tensor,  # [E] tokens
        edge_d: torch.Tensor,  # [E] destination states
    ) -> None:
        self.N = int(num_states)
        self.V = int(vocab_size)
        self.start = int(start)
        self.accepting = accepting.to(torch.bool)
        self.edge_s = edge_s.long().contiguous()
        self.edge_v = edge_v.long().contiguous()
        self.edge_d = edge_d.long().contiguous()

    @classmethod
    def from_next_state(cls, next_state, accepting, start) -> DiffusionConstraint:
        n, v = next_state.shape
        s_idx, v_idx = (next_state >= 0).nonzero(as_tuple=True)
        return cls(n, v, start, accepting, s_idx, v_idx, next_state[s_idx, v_idx])

    @classmethod
    def from_file(cls, path: str, device="cpu") -> DiffusionConstraint:
        d = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            d["num_states"],
            d["vocab_size"],
            d["start"],
            d["accepting"],
            d["edge_s"],
            d["edge_v"],
            d["edge_d"],
        ).to(device)

    def to(self, device) -> DiffusionConstraint:
        self.accepting = self.accepting.to(device)
        self.edge_s = self.edge_s.to(device)
        self.edge_v = self.edge_v.to(device)
        self.edge_d = self.edge_d.to(device)
        return self

    def _transition_matrices(self, logp: torch.Tensor) -> torch.Tensor:
        """M[b,i,s,s'] = logsumexp over tokens v (s->s') of logp[b,i,v]; [B,L,N,N]."""
        B, L, _ = logp.shape
        N = self.N
        vals = logp[:, :, self.edge_v]  # [B, L, E]
        flat = torch.full((B, L, N * N), NEG_INF, device=logp.device, dtype=logp.dtype)
        idx = (self.edge_s * N + self.edge_d).view(1, 1, -1).expand(B, L, -1)
        # scatter-logsumexp: max then log-sum-exp of shifted terms
        mx = torch.full_like(flat, NEG_INF)
        mx.scatter_reduce_(2, idx, vals, reduce="amax", include_self=True)
        ex = torch.exp(vals - mx.gather(2, idx))
        sm = torch.zeros_like(flat)
        sm.scatter_add_(2, idx, ex)
        flat = torch.where(sm > 0, mx + torch.log(sm), torch.full_like(flat, NEG_INF))
        return flat.view(B, L, N, N)

    @staticmethod
    def _logmv(v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Log-semiring vec@mat: out[b,k]=logsumexp_s v[b,s]+m[b,s,k]. [B,N]."""
        return torch.logsumexp(v.unsqueeze(-1) + m, dim=-2)

    @staticmethod
    def _logmv_t(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Log-semiring mat@vec: out[b,s]=logsumexp_k m[b,s,k]+v[b,k]. [B,N]."""
        return torch.logsumexp(m + v.unsqueeze(-2), dim=-1)

    def constrained_log_marginals(self, logits: torch.Tensor) -> torch.Tensor:
        """Reweight ``logits`` [B, L, V] to constrained log-marginals [B, L, V].

        Tokens with no DFA transition get ``-inf``. Positions of a request whose
        constraint is unsatisfiable within L are left unchanged (logZ == -inf).
        """
        B, L, V = logits.shape
        N = self.N
        logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        M = self._transition_matrices(logp)  # [B, L, N, N]

        # forward alpha[i] and backward beta[i]
        alpha = torch.full(
            (B, L + 1, N), NEG_INF, device=logits.device, dtype=logp.dtype
        )
        alpha[:, 0, self.start] = 0.0
        for i in range(L):
            alpha[:, i + 1] = self._logmv(alpha[:, i], M[:, i])
        beta = torch.full(
            (B, L + 1, N), NEG_INF, device=logits.device, dtype=logp.dtype
        )
        beta[:, L] = torch.where(
            self.accepting,
            torch.zeros(N, device=logits.device),
            torch.full((N,), NEG_INF, device=logits.device),
        )
        for i in range(L - 1, -1, -1):
            beta[:, i] = self._logmv_t(M[:, i], beta[:, i + 1])

        logZ = beta[:, 0, self.start]  # [B]
        ok = torch.isfinite(logZ)

        # out[b,i,v] = logp[b,i,v] + logsumexp_s(alpha[b,i,s] + beta[b,i+1, d(s,v)]);
        # accumulate per (b,i,v) over edges via scatter-logsumexp.
        E = self.edge_v.shape[0]
        # weight per edge per (b,i): alpha[b,i,edge_s] + beta[b,i+1,edge_d]
        a_e = alpha[:, :L][:, :, self.edge_s]  # [B, L, E]
        b_e = beta[:, 1:][:, :, self.edge_d]  # [B, L, E]
        w = a_e + b_e  # [B, L, E]
        idxv = self.edge_v.view(1, 1, E).expand(B, L, E)
        out = torch.full((B, L, V), NEG_INF, device=logits.device, dtype=logp.dtype)
        mx = torch.full((B, L, V), NEG_INF, device=logits.device, dtype=logp.dtype)
        mx.scatter_reduce_(2, idxv, w, reduce="amax", include_self=True)
        ex = torch.exp(w - mx.gather(2, idxv))
        sm = torch.zeros((B, L, V), device=logits.device, dtype=logp.dtype)
        sm.scatter_add_(2, idxv, ex)
        acc = torch.where(sm > 0, mx + torch.log(sm), torch.full_like(mx, NEG_INF))
        out = acc + logp  # add emission; -inf where no edge
        # leave unsatisfiable requests unchanged
        out = torch.where(ok.view(B, 1, 1), out, logits)
        return out
