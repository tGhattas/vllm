# Phase 2 Plan: the log-depth parallel constrained sampler (paper novelty)

Plan for implementing the core novelty of Dang & Ermon (arXiv:2607.07026):
sampling the constrained mean-field posterior in **O(log L) parallel depth**
instead of the O(L) sequential ancestral pass. Grounded in the paper's method
(extracted from the arXiv HTML), mapped to concrete data structures. CPU-first,
correctness-verified against the existing sequential sampler and brute force.

## What the paper does (confirmed from the method section)

- **Chain graphical model over automaton states.** Per position a transition
  matrix `M_i(s, s') = Σ_v pᵢ(v)·1[δ(s, v) = s']` (paper Eq. for `M_t`). Backward
  messages `β_{i}(s) = Σ_{s'} M_{i+1}(s, s') β_{i+1}(s')`, `β_L(s) = 1[s ∈ F]`.
  (This is exactly our `constrained_sampler.backward_messages`, matrix form.)
- **Sequential sampler (baseline, already implemented):** `O(L)` depth,
  `O(L·|S|²)` work.
- **Log-depth sampler (Algorithm 1, the novelty):**
    - *Conditional independence / midpoint recursion* (paper Eq. 6):
    `p(x_{ℓ:r-1} | z_ℓ, z_r) ∝ Σ_{z_m} p(z_m | z_ℓ, z_r)·p(x_{ℓ:m-1} | z_ℓ, z_m)·p(x_{m:r-1} | z_m, z_r)`.
    - *Midpoint state law* (Eq. 7): `p(z_m | z_ℓ, z_r) ∝ p(z_m | z_ℓ)·p(z_r | z_m)`,
    where `p(· | ·)` are **multi-step transition probabilities** = products of the
    per-position transition matrices over a range.
    - *Bottom-up (O(log L) depth):* precompute range-products of transition matrices
    (all `2^k`-step / dyadic-segment transitions) via a reduction tree.
    - *Top-down (O(log L) depth):* recursively sample the midpoint automaton state of
    each segment using the segment products, spawning the two halves in parallel.
- **Complexity:** FLOPs `O(L(|S|³ + |ℰ||V|))`, **parallel depth `O(log L)`**,
  memory `O(L·|S|²)`.

## Concrete implementation (this repo, CPU reference)

`parallel_sampler.py`:

1. **Transition matrices.** `M[i]` is an `[N, N]` **log-space** matrix,
   `M[i][s, s'] = logsumexp_{v: δ(s,v)=s'} log pᵢ(v)`; `-inf` where no token drives
   `s → s'`. `fixed_positions` restrict position `i` to its single committed token.
2. **Log-semiring matmul.** `logmm(A, B)[i,k] = logsumexp_j (A[i,j] + B[j,k])` —
   the associative operator over which we build range products.
3. **Segment tree of range products (bottom-up).** A balanced binary tree over
   positions `[0, L)`; node `(l, r)` stores `P[l,r) = M[l] ⊗ … ⊗ M[r-1]` (⊗ = logmm),
   with children split at `m = (l+r)//2`. Filled bottom-up in `O(log L)` dependency
   depth (each level's nodes are independent).
4. **Top-down midpoint sampling (`O(log L)` depth).**
   - Sample the end state `z_L ∝ P[0,L)[start, ·]` restricted to accepting states.
   - Recurse `sample_states(l, r, z_l, z_r)`: if `r-l ≤ 1` return; else at `m`,
     sample `z_m ∝ P[l,m)[z_l, ·] + P[m,r)[·, z_r]` (log-space), then recurse on
     `(l, m, z_l, z_m)` and `(m, r, z_m, z_r)` — the two calls are independent
     (parallelizable). Tree height = `O(log L)`.
5. **Token emission (`O(1)` depth).** Given the full boundary-state vector
   `z_0..z_L`, sample every `xᵢ ∝ pᵢ(·)` restricted to `{v : δ(zᵢ, v) = z_{i+1}}`
   independently, in parallel.

Same output distribution as the sequential ancestral sampler — by construction the
midpoint recursion is an exact re-factoring of the chain, so no approximation.

## Verification strategy (Phase 3 tests)

- **Distributional equivalence to brute force:** enumerate `P_D`, draw many
  parallel-sampler samples, assert the empirical distribution matches (and every
  sample is DFA-accepted).
- **Exact agreement of partition/marginals with the sequential module:** the root
  product `P[0,L)[start, s]` over accepting `s`, log-sum-exp'd, must equal
  `constrained_sampler.log_partition` to ~1e-9.
- **Same-seed structural checks:** single-string DFA → always that string; fixed
  positions honored; impossible constraint detected (`-inf` partition).
- **Depth measurement:** instrument the tree to count sequential dependency levels;
  assert it grows like `⌈log₂ L⌉` while `L` scales (e.g. L = 2,4,8,16,32), vs. the
  sequential sampler's `L` levels — demonstrating the asymptotic depth win even
  though pure-Python wall-clock won't show it.

## Scope / honesty

- This is a **CPU reference** for the log-depth *algorithm and its depth*, not a GPU
  kernel; it proves correctness and the `O(log L)` dependency structure. A real
  vLLM kernel would batch the independent tree nodes / recursion branches on-device.
- Still regular constraints, one canvas, fixed length; guard untouched.
