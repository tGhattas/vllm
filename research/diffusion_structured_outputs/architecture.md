# Architecture & Failure-Path Map: Structured Outputs for Diffusion LMs (issue #45572)

Research note (private, non-production). Repo base: `upstream/main @ 1a659a0c3`,
branch `research/diffusion-structured-outputs`. All line numbers verified against
this commit. Where a claim is derived rather than directly read, it is marked
**(inference)**.

---

## 0. TL;DR

vLLM's guided-decoding pipeline is built on one hard assumption: tokens are
committed **left-to-right, one at a time**, so an xgrammar FSM can be advanced by
each committed token and a bitmask can constrain the single "next" position before
sampling. DiffusionGemma denoises an entire `canvas_length` block **in parallel**
and only surfaces a finalized block on *commit* steps — there is no single next
position for the FSM/bitmask to constrain. Because of that, structured outputs are
**rejected up-front with an HTTP 400** in `vllm/sampling_params.py:915-925`
(guard from PR #45468, issue #45436). The narrow, honest first slice discussed on
the issue is a *post-hoc validate-only* mode; true guided decoding needs the
constraint applied **inside the denoising loop**, which is what the standalone POC
in this directory prototypes at the algorithm level.

> **Correction to the task brief.** The brief pointed at
> `vllm/v1/structured_output/__init__.py:214` and `:325` as the guard. Those lines
> are *not* the rejection — they are latent diffusion-*handling* scaffolding inside
> the bitmask builder that is unreachable in production because the real guard
> rejects such requests earlier (§2).

---

## 1. Autoregressive structured-output flow (the path that works)

```text
OpenAI request (response_format / structured_outputs)
  → SamplingParams._validate_structured_outputs        vllm/sampling_params.py:906
  → StructuredOutputManager.grammar_init (compile)      vllm/v1/structured_output/__init__.py:115
  → per-request grammar on request.structured_output_request.grammar
  → scheduler: get_grammar_bitmask (per step)           vllm/v1/core/sched/scheduler.py:1565
       → StructuredOutputManager.grammar_bitmask         .../structured_output/__init__.py:204
            → grammar.fill_bitmask(...)                   backend_xgrammar.py:195
            → grammar.accept_tokens(...) + rollback       __init__.py:315, 339 (non-destructive peek)
  → gpu_model_runner: apply_grammar_bitmask(logits)      gpu_model_runner.py:4526
  → _sample(logits)                                      gpu_model_runner.py:4532
  → scheduler.update_from_output: grammar.accept_tokens  scheduler.py:1745 (authoritative advance)
```

**Key symbols (all verified):**

| Concern | Symbol | Location |
| --- | --- | --- |
| Compile grammar | `StructuredOutputManager.grammar_init` / `_create_grammar` | `structured_output/__init__.py:115, 173` |
| xgrammar compile | `XgrammarBackend.compile_grammar` | `backend_xgrammar.py:78` |
| **Advance FSM (mutating)** | `XgrammarGrammar.accept_tokens` → `matcher.accept_token` loop | `backend_xgrammar.py:152-171` |
| **Non-advancing check** | `XgrammarGrammar.validate_tokens` → accept-then-`rollback` | `backend_xgrammar.py:173-187` |
| Build per-step bitmask | `StructuredOutputManager.grammar_bitmask` | `structured_output/__init__.py:204` |
| Fill one row | `grammar.fill_bitmask` → `matcher.fill_next_token_bitmask` | `backend_xgrammar.py:195` |
| **Apply mask to logits** | `apply_grammar_bitmask` → `xgr.apply_token_bitmask_inplace` | `structured_output/utils.py:86, 161/171/175`; called `gpu_model_runner.py:4526` |
| **Authoritative advance** | `grammar.accept_tokens(req_id, advance_token_ids)` | `scheduler.py:1745` |

The invariant that makes this work: **AR masks are applied to the single next-token
logits row (plus bonus/spec rows) right before `_sample`**, and the FSM is advanced
**once per committed token, in generation order** (`scheduler.py:1745`). The bitmask
builder even *peeks* the FSM forward for speculative rows and then rolls back
(`__init__.py:315, 339`) so the mask reflects each hypothetical position without
corrupting state.

---

## 2. The diffusion guard (PR #45468) — the operative behavior

**Operative rejection** — `SamplingParams._validate_structured_outputs`,
`vllm/sampling_params.py:915-925`:

```python
if model_config.is_diffusion:
    # Diffusion LLMs denoise a whole canvas of tokens in parallel rather than
    # sampling left-to-right, which the grammar FSM requires. Without this
    # check, requests fail mid-generation with an FSM rejection (HTTP 500).
    # See issue #45436.
    raise ValueError("Structured outputs are not yet supported for diffusion ...")
```

This `ValueError` surfaces as a **clean 400** at request-validation time (before
scheduling), replacing the pre-#45468 mid-generation xgrammar 500. A companion
guard `_validate_diffusion` (`sampling_params.py:884-904`) separately rejects
per-request sampling params (temperature≠1.0, min_p, seed, …) for diffusion.

**Detection** — `ModelConfig.is_diffusion` (`config/model.py:1597-1600`) is a
`cached_property`: `getattr(self.hf_config, "canvas_length", None) is not None`.
`canvas_length` lives on `DiffusionConfig` (`config/diffusion.py:20`).

**The latent scaffolding (not the guard).** Inside `grammar_bitmask`:

- `__init__.py:214-215` — `max_num_spec_tokens = num_speculative_tokens`, with a
  comment noting this "covers both speculative decoding and diffusion LLMs
  (canvas_length)". `VllmConfig.num_speculative_tokens` (`config/vllm.py:535-547`)
  **overloads** the spec-token count with `diffusion_config.canvas_length`.
- `__init__.py:323-325` — skips the bonus-token bitmask row for a diffusion request
  that already has `req_tokens` (diffusion samples no bonus token after the
  scheduled canvas positions).

These branches would *adapt* the mask machinery to a canvas, but they are
**unreachable in production** because §2's guard rejects the request first. They are
scaffolding for a future in-loop implementation, not current policy. **(inference)**

---

## 3. DiffusionGemma denoise/commit (the path that must change)

File: `vllm/model_executor/models/diffusion_gemma.py` (1401 lines).

**Where per-position logits appear:** `compute_logits` (`:334-338`) — softcapped
logits for every canvas position, consumed inside the compiled sampler.

**The compiled step:** `_compiled_sample_step` (`:471-656`) —
`temperature → Gumbel sample → probs/confidence → accept/renoise → convergence`:

- entropy/confidence: `token_entropy = -(probs*log_probs).sum(-1)` (`:543`),
  `mean_entropy < confidence_threshold` (`:549`), entropy-bound accept mask over
  sorted cumulative entropy (`:552-555`);
- commit vs denoise: `is_commit = is_encoder_phase[decode_slots]` (`:560`),
  `is_denoise = ~is_commit` (`:561`);
- renoise: `denoise_canvas = where(eb_mask, new_tokens, random_tokens)` (`:577-578`);
- **commit emission**: `sampled[...] = argmax_canvas ...` (`:603-606`) and
  **`num_sampled = is_commit * valid_canvas_len`** (`:609-611`) — the committed
  block size;
- convergence flips `is_encoder_phase` so a denoise-converged canvas commits next
  step (`:613-626`), then the canvas is overwritten with its argmax (`:649-652`);
- canvas transported via the spec-decode `draft_tokens` buffer:
  `draft_tokens[:, :CL] = canvas` (`:655-656`).

**Sampler entry:** `DiffusionSampler.__call__` (`:1223`) computes `valid_canvas_len`
(`:1257-1263`), snapshots `is_committing` *before* the kernel mutates phase
(`:1285-1287`), runs `_compiled_sample_step`, and builds `SamplerOutput(num_sampled,
num_rejected)` via `_build_output` (`:1186-1216`). Per-request GPU state
(`DiffusionGemmaRequestStates`, `:661`) holds `canvas`, `argmax_canvas`,
`accepted_canvas_history`, `self_conditioning_embeds`; `init_canvas` (`:736`) seeds
random tokens.

**Speculative-decoding-shaped accounting reused by diffusion:**

- `num_sampled` / `num_rejected` (`SamplerOutput`) carry committed-block accounting
  (`:1211-1216`); `_compute_num_rejected` (`:459`).
- `draft_tokens` buffer reused as the canvas transport (`:655-656`).
- `num_speculative_tokens ≡ canvas_length` (`config/vllm.py:542-546`).
- Scheduler suppresses the AR bonus slot for diffusion (`scheduler.py:123`).
- Metrics remap onto spec counters: `spec_decode/metrics.py:_log_diffusion`
  (`:139-174`) maps `num_drafts→denoising steps`, `num_draft_tokens→canvas
  positions`, `num_accepted_tokens→committed tokens`; Prometheus counters
  `vllm:diffusion_num_denoising_steps/_num_canvas_positions/_num_committed_tokens`
  (`:215-226`).

**What the scheduler sees — committed block only.** During a denoise step
`num_sampled == 0` and `sampled` is zeroed (`:603-611`); intermediate canvas tokens
never leave GPU model-runner state. Only on a commit step does the `SamplerOutput`
carry the finalized `valid_canvas_len` block, which the scheduler appends via
`_update_request_with_output(request, new_token_ids)` (`scheduler.py:1724`) and only
*then* would advance a grammar (`scheduler.py:1745`).

---

## 4. The exact AR-vs-diffusion mismatch

| Aspect | Autoregressive (works) | DiffusionGemma (rejected) |
| --- | --- | --- |
| Emission order | strictly left-to-right, one token | whole `canvas_length` block, parallel |
| FSM advance unit | 1 committed token, in order (`scheduler.py:1745`) | a multi-token block, all at once |
| Constraint point | mask the single next-token row pre-`_sample` (`gpu_model_runner.py:4526`) | denoising steps are **never** bitmask-constrained |
| What scheduler sees per step | the sampled token(s) | nothing on denoise; a full block on commit |
| Intermediate state | none — prefix is committed | a mutable canvas that is renoised/replaced |
| Extra tokens | none | committed block includes a trailing turn-end token |

**Why advancing the existing FSM with a committed block is insufficient:** the FSM
is designed to *mask* the next position so an invalid token is never sampled. Handing
it a block after the fact only lets it *reject* — and because the denoising steps
that produced the block were unconstrained, a schema-valid block plus its trailing
turn-end token is a multi-token chunk the FSM never masked, so it can spuriously
reject even valid JSON. Left-to-right advancement over a block also cannot express
the constraint *during* the parallel refinement where it would actually change what
the model samples.

---

## 5. Sequence diagram: prefill → denoise → denoise → commit

Canvas length L=4; a DFA constraining the block. Shows canvas + (would-be) grammar
state at each stage. Today the grammar column is empty (guarded off); the POC shows
what a canvas-aware constraint would compute.

```text
Stage      Scheduler sees     Canvas (GPU state)         DFA / constraint (POC)
--------   ----------------   ------------------------   ----------------------------
prefill    (prompt tokens)    init_canvas → [r r r r]    start state q0; β precomputed
                              (random seed, :736)        over all 4 positions

denoise#1  num_sampled=0      argmax→[a ? b ?]           per-position marginals P_D(x_i)
           (nothing)          low-confidence slots       conditioned on acceptance;
                              renoised (:577)            fixed slots = accepted tokens

denoise#2  num_sampled=0      [a c b ?] converging       re-solve DP with newly fixed
           (nothing)          (:613 convergence)         positions {0:a,1:c,2:b} → tighten
                                                         marginals on the last free slot

commit     num_sampled=L      [a c b d] + <end>          block accepted by DFA by
           block [a c b d]    (:603-611)                 construction; FSM (if carried)
           appended :1724     draft_tokens copy :655     advances across the block for
                                                         the *next* committed block
```

The POC (`constrained_sampler.py`) implements the "DFA / constraint" column for a
single denoising step: given the per-position logits and the set of already-fixed
positions, it computes exact constrained marginals / a jointly-sampled accepted
canvas via a log-space forward/backward DP.

---

## 6. Where a constraint component could be inserted

Two seams, matching the issue's option ladder:

- **Post-hoc validate-only (option a).** At commit, strip the turn-end token and call
  the existing non-advancing `validate_tokens` (`backend_xgrammar.py:173`) on the
  block; accept if fully valid, else clean error. Smallest change; **no guidance**.
  Insertion: the commit path in `DiffusionSampler` / the scheduler's post-commit
  advance (`scheduler.py:1732-1756`). This is what maintainers were asked to bless.

- **In-denoising-loop guidance (option c).** Insert a canvas-aware constraint
  *inside* `_compiled_sample_step` between `probs` computation (`:539-543`) and the
  accept/renoise decision (`:560-583`), replacing per-position independent sampling
  with a constrained joint sample / constrained marginals over the canvas. This is
  where the POC's algorithm would live. It is the design-heavy path and is exactly
  what the issue proposes to gate behind an RFC.

**Generic vs DiffusionGemma-specific:**

- *Generic* (could live in `v1/structured_output`): the DFA/regex → automaton
  compilation, the constrained forward/backward DP, fixed-position conditioning,
  impossibility detection. The POC here is deliberately model-agnostic.
- *DiffusionGemma-specific*: canvas length/lifecycle, `is_encoder_phase`
  commit/denoise phase logic, confidence/entropy acceptance, renoise schedule,
  turn-end token handling, the `draft_tokens`/`num_sampled` transport.

---

## 7. Research-note conclusion

The defensible seam is a **model-agnostic constrained-sampling component** (the POC)
invoked from a **DiffusionGemma-specific adapter** at the denoise step, with the
post-hoc validate-only mode as an intermediate honest milestone. Nothing here
weakens the §2 guard: the guard stays until an opt-in constrained path is proven
correct. See `design.md` for the interface analysis and open questions.
