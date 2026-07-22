# Research: canvas-aware constrained decoding for diffusion LMs (issue #45572)

Private, **non-production** research scaffolding for understanding and prototyping
structured outputs / guided decoding for diffusion language models (DiffusionGemma).
Nothing here is proposed for upstream merge as-is; it is a learning + correctness
harness. See the top-level task guardrails.

- Base: `vllm-project/vllm@1a659a0c3` (upstream/main), branch
  `research/diffusion-structured-outputs`.
- Status: research-only. No issue comment, no upstream PR planned.

## Contents

| File | What |
| --- | --- |
| `architecture.md` | Verified code map: AR structured-output path, the real diffusion guard (`sampling_params.py:915`), DiffusionGemma denoise/commit, the exact AR-vs-diffusion mismatch, a sequence diagram, and the insertion seam. |
| `design.md` | Integration design exploration answering the 10 design questions; a derived (non-final) `DiffusionConstraint` interface. |
| `finite_automaton.py` | `DFA` over integer tokens (partial transition = reject) + trie/exact-string constructors. |
| `constrained_sampler.py` | Exact constrained-canvas sampler: log-space forward/backward DP → `marginals`, `sample`, `greedy` (Viterbi/MAP), `log_partition`, `fixed_positions`, impossibility detection. **O(L) sequential.** |
| `denoise_loop.py` | Multi-step constrained denoising loop (confidence-based schedule, re-solving as positions are committed). Verified exact via chain rule. |
| `parallel_sampler.py` | **Log-depth (O(log L)) parallel sampler** — the paper's novelty (Dang & Ermon Alg. 1): log-semiring segment tree of transition-matrix products + top-down midpoint-state recursion. Same distribution as the sequential sampler. |
| `brute_force_oracle.py` | Exhaustive-enumeration ground truth for tiny cases. |
| `LOGDEPTH_PLAN.md` | Phase-2 plan mapping the paper's Algorithm 1 to `parallel_sampler.py`. |
| `GPU_VALIDATION.md` | Real Dream-7B logit results (single-step; the multi-step loop confirmed separately). |
| `tests/` | 46 seeded, CPU-only oracle tests (sampler 21, denoise loop 7, parallel sampler 18). |
| `model_adapters/` | Real-diffusion-logits bridge into the verified sampler (CPU-tested). |

## The problem the sampler solves

Given independent per-position token distributions `p_i(t)` over a fixed canvas of
length `L` and a DFA `D`, define `P(x) = ∏_i p_i(x_i)` and sample from / reason about
the distribution conditioned on `x` being accepted by `D`:

```text
P_D(x) = P(x)·1[D accepts x] / Z,   Z = Σ_{x accepted} P(x)
```

This is the **constrained mean-field posterior** of Dang & Ermon (arXiv:2607.07026).
The POC implements the correctness-first **sequential ancestral** sampler, not the
paper's log-depth parallel variant.

### Method (log-space forward/backward DP)

- Backward completion weights `β[i,q] = log P(reach acceptance | state q, position i)`,
  with `β[L,q] = 0` iff `q` accepting. `log Z = β[0, start]`.
- Ancestral sampling: at position `i`, state `q`, draw token `t ∝ p_i(t)·β[i+1, δ(q,t)]`,
  then advance `q ← δ(q,t)`. Exact sample from `P_D`; **accepted by construction**.
- Forward prefix weights `α` + `β` give exact marginals `P_D(x_i=t)`.
- Viterbi (max instead of log-sum-exp) gives the max-probability accepted canvas.
- `fixed_positions` restricts a position's support to a known token (positions already
  committed by an earlier denoising step).
- `log Z = -inf` ⇒ impossible constraint (`ImpossibleConstraintError`).

### Complexity

`L` = canvas length, `V` = vocab size, `N` = DFA states.

- DP (backward/forward/marginals): **O(L·N·V)** time, **O(L·N)** memory.
- one ancestral sample: **O(L·V)** given cached `β`.

The real-vocab (`V≈256k`) cost is the bottleneck; production would exploit transition
sparsity (character classes) and a batched GPU kernel. Correctness-first: this CPU
reference is the oracle for any future GPU version.

## Running the tests

```bash
# from repo root, using the project venv (per AGENTS.md)
cd research/diffusion_structured_outputs
../../.venv/bin/python -m pytest tests/ -q
```

Current result: **46 passed**. Coverage: exact-single-string DFA, multi-string DFA,
regular pattern (parity), impossible constraint, compatible fixed token, impossible
fixed token, highly-skewed logits, very-small probabilities, greedy == brute-force
MAP, sampler-reproduces-exact-distribution (Monte Carlo), and the
`log P_D(x) = Σ log p_i(x_i) − log Z` identity. Exact-vs-exact checks use `atol=1e-9`;
the Monte-Carlo distribution check uses 60k samples at `atol=0.01`.

## Related work (verified from primary sources)

- **Dang & Ermon, "Constrained Decoding for Diffusion Language Models via Efficient
  Inference over Finite Automata" (arXiv:2607.07026).** Exact, tractable sampling from
  the constrained mean-field posterior for any finite-automaton constraint;
  satisfaction by construction; fixed positions absorbed into the factorized
  distribution; ancestral sampler reduced to **O(log L)** depth via arithmetic-circuit
  depth reduction; compatible with parallel/block-wise decoding under arbitrary
  remasking schedules. → the algorithm this POC implements (sequential variant).
- **Zhang et al., "Lookahead-then-Verify (LAVE): Reliable Constrained Decoding for
  Diffusion LLMs under Context-Free Grammars" (arXiv:2602.00612).** Targets **CFG**
  (more expressive than regular): uses the parallel all-position distributions to
  look ahead and verify that a proposed token can still be extended to a valid
  sentence, avoiding dead-ends; negligible overhead. → relevant later, when moving
  beyond the regular/DFA subset toward recursive JSON.

**Assumption that may not match DiffusionGemma:** the mean-field posterior treats
canvas positions as independent given the per-step logits. DiffusionGemma's
bidirectional attention couples positions and the per-position distributions change
every denoising step, so the independence holds only *within* a single step's
predicted marginals. This is why the design (`design.md`) applies the DP per denoise
step and re-solves as positions get fixed, rather than once.

## Scope (explicitly out for the first milestone)

Full JSON Schema, general CFGs, variable-length generation, multiple committed
canvases, tool-call parsing, GPU kernels, and xgrammar internals are **out of scope**.
The guard in `sampling_params.py:915` is **not** touched.
