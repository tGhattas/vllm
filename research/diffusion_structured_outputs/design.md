# Integration Design Note (private, non-production)

Research note for issue #45572. Derives a candidate vLLM integration for
canvas-aware constrained decoding from the verified code map (`architecture.md`) and
the exact CPU POC in this directory. **This is a design exploration, not a proposal
to merge.** Per the current decision, no upstream comment/PR is planned; this exists
to make the eventual seam legible and to record open questions.

---

## 1. What should the constraint layer return?

The POC exposes three primitives; the choice depends on the seam:

- **Constrained marginals** `P_D(x_i = t)` — a `[L, V]` distribution
  (`constrained_sampler.marginals`). Cheap to consume: the diffusion sampler can keep
  its existing per-position Gumbel/argmax/confidence machinery but read from these
  masked-and-renormalized marginals instead of raw softmax. Does **not** by itself
  guarantee the *committed* block is accepted, because independent per-position
  argmax of the marginals can leave the joint outside the language (the POC's
  `test_greedy_differs_from_marginal_argmax` demonstrates the gap).
- **A jointly-sampled accepted canvas** `x ~ P_D` (`constrained_sampler.sample`) or
  the **joint MAP** (`constrained_sampler.greedy`). Guarantees acceptance by
  construction, but bypasses the model's own accept/renoise schedule and confidence
  logic for that step.
- **Masked logits** (AR-style). Ill-defined for a canvas: there is no single next
  position, and a per-position independent mask cannot encode the cross-position DFA
  coupling.

**Recommendation:** return **constrained marginals** for the *guidance* signal during
denoising (keeps DiffusionGemma's confidence/renoise semantics intact), and use a
**joint constrained sample/MAP only at the final commit** to guarantee the emitted
block is accepted. This mirrors the paper's mean-field posterior: marginals steer the
parallel refinement; the joint sample closes the acceptance guarantee.

## 2. Where should it run relative to temperature / Gumbel / entropy / accept / renoise?

From `_compiled_sample_step` (`diffusion_gemma.py:471-656`): temperature scaling and
`probs` are computed at `:539-543`; entropy/confidence at `:543-555`; accept/renoise
at `:560-583`. The constraint must run **after temperature/`probs`** (so it constrains
the actual sampling distribution) and **before the accept/renoise decision** (so
fixed/accepted positions feed back into the DFA). Concretely: replace the raw `probs`
that drive Gumbel sampling with **constrained marginals** derived from those same
`probs`, leaving entropy/confidence to operate on the constrained distribution.

## 3. Once some positions are retained, how does the DFA condition on them?

This is exactly the POC's `fixed_positions`: retained/committed canvas slots become
`{position: token}` and the forward/backward DP restricts those positions' support to
the retained token (`constrained_sampler.py:_allowed_tokens`). The DP then produces
exact marginals/samples over only the still-free positions, conditioned on the fixed
ones and on eventual acceptance. Verified by `test_fixed_token_compatible` and
`test_fixed_token_makes_impossible`.

## 4. Invariant when a constrained canvas is later renoised

**Every position that remains *fixed/committed* must keep a token for which a path to
an accepting state still exists given the other fixed positions** — i.e. the DP
partition over the fixed set must stay `> 0` (`log_partition != -inf`). Renoising may
only touch *free* positions; it must never renoise a position in a way that makes the
fixed set jointly unsatisfiable. If a renoise step would strand the constraint
(partition → 0), that is the impossibility signal (`ImpossibleConstraintError`) and
the step must be rejected/rolled back rather than committed.

## 5. Which score should acceptance confidence use?

DiffusionGemma accepts positions by entropy/confidence on the model distribution
(`:543-555`). Under constraint, confidence should be computed on the **constrained
marginals**, not the raw model probs: a position the raw model is unsure about may be
*forced* by the grammar (constrained marginal near-deterministic) and should be
accepted; conversely a raw-confident position the grammar forbids must not be. Using
constrained marginals keeps the existing entropy machinery meaningful. (Open
question: whether to expose the *original* model confidence for logging/telemetry.)

## 6. Per-request automata, heterogeneous schemas, compilation, torch.compile, GPU residency, cancellation

- **Compilation/caching:** reuse the existing structured-output backend cache
  (`StructuredOutputManager` / `XgrammarBackend.compile_grammar`,
  `structured_output/__init__.py:115-184`) to compile schema→automaton once per
  request and cache by grammar key. The DFA the POC needs can be derived from the
  compiled grammar (open question §10).
- **Per-request/heterogeneous:** each request carries its own automaton state, exactly
  as AR requests carry their own `grammar`. Batched denoising must run the DP
  per-request (heterogeneous canvases), which fights `torch.compile`'s static-shape
  assumptions in `_compiled_sample_step`.
- **torch.compile:** the DP is data-dependent (per-request state count, sparse
  transitions) and does not belong inside the compiled kernel. Candidate split: run
  the constraint DP outside the compiled region (host or a separate CUDA stream),
  passing in constrained marginals as a `[L, V]` tensor the compiled step consumes —
  keeping `_compiled_sample_step`'s shapes static.
- **GPU residency:** the POC is NumPy/CPU. For real vocab (~256k) the O(L·N·V) DP is
  the cost driver; it must exploit transition sparsity (character classes) and likely
  run as a batched GPU/Triton kernel. Correctness-first ⇒ CPU reference stays the
  oracle; GPU version validated against it.
- **Cancellation/cleanup:** mirror AR grammar cleanup — free per-request automaton
  state on request finish/abort alongside `DiffusionGemmaRequestStates.remove_request`
  (`diffusion_gemma.py:747-757`).

## 7. Stop tokens and Gemma turn-boundary tokens

The committed block includes a trailing turn-end token (noted on the issue and
consistent with `num_sampled = valid_canvas_len`, `:609-611`). The automaton must
either (a) explicitly model an optional trailing turn-end/stop token as an accepting
self-transition, or (b) the adapter strips known turn/end suffix tokens before the DFA
sees the block (the option-a validate-only approach already does this). Getting this
boundary wrong is the exact false-rejection #45468 guards against, so it must be
handled explicitly, not implicitly.

## 8. How does automaton state carry across committed blocks?

DiffusionGemma is *block-autoregressive*: blocks commit in order, conditioned on
history. So the automaton advances **at block boundaries** — after a block commits,
run the DFA across the committed block (turn-end stripped) to reach the entry state
for the next block, then re-seed the next canvas's DP from that state. This reuses the
existing `scheduler.py:1745` advance point conceptually, but the advance unit is a
*block* and the per-step guidance is the canvas DP, not a next-token mask.

## 9. Regular-only first, or a restricted JSON subset?

Start **regular-only** (DFA), which the POC fully supports and which covers a useful
slice (enums, fixed keys, bounded strings, simple patterns). A restricted JSON subset
that is regular (fixed schema, no unbounded nesting/recursion) fits. **Unbounded /
recursive JSON is context-free, not regular** — that is the LAVE (arXiv:2602.00612)
territory and out of scope for a finite-automaton-first milestone.

## 10. Can an existing xgrammar representation be reused?

Open question, and the highest-leverage one. xgrammar compiles to a grammar
matcher, not obviously to an explicit DFA with enumerable states/transitions that the
POC's DP needs. Options: (a) drive the DP off xgrammar's `fill_next_token_bitmask` as
a per-(state,position) transition oracle — reuse compilation, avoid re-deriving a DFA,
but state identity/rollback semantics must be mapped carefully; (b) compile the
regular subset to an explicit DFA independently (e.g. from the regex/`outlines`-style
FSM path) and keep xgrammar for the AR path only. Deciding this needs maintainer
input and a closer read of xgrammar internals.

---

## Candidate interface (derived, not final)

Not the brief's `constrain_or_sample(logits, canvas, fixed_positions, ...)` verbatim.
Splitting *guidance* (marginals, every denoise step) from *acceptance* (joint sample,
at commit) matches DiffusionGemma's two-phase structure better:

```python
class DiffusionConstraint:            # model-agnostic; lives near v1/structured_output
    """Compiled per request from a regular schema; wraps a DFA + the DP."""

    def constrained_marginals(
        self, log_probs: Tensor,          # [L, V] per-position log-probs this step
        fixed_positions: Mapping[int, int],  # retained/committed canvas slots
    ) -> Tensor:                          # [L, V] constrained marginals; steers denoise
        ...

    def commit_block(
        self, log_probs: Tensor, fixed_positions: Mapping[int, int],
        greedy: bool, rng,
    ) -> list[int]:                       # joint accepted block; guarantees acceptance
        ...

    def advance_block(self, committed_tokens: list[int]) -> None:
        """Advance automaton state across a committed block (turn-end stripped)."""

    def is_satisfiable(self, fixed_positions) -> bool:  # partition > 0
        ...
```

A **DiffusionGemma adapter** owns canvas lifecycle and calls `constrained_marginals`
inside the denoise step, `commit_block` at commit, `advance_block` at block boundary,
and `is_satisfiable` before accepting a renoise. The four POC functions
(`marginals`, `sample`/`greedy`, `log_partition`) already implement these primitives
on CPU.

---

## Guardrail check

- The §2 guard (`sampling_params.py:915`) is **not** weakened by any of this; it stays
  until an opt-in constrained path is separately proven. This design does not route
  DiffusionGemma through the AR xgrammar next-token path.
- Scope is deliberately regular-constraints / one canvas / fixed length — no full JSON
  Schema, CFG, tool-calls, multi-canvas, or arbitrary-length generation.
