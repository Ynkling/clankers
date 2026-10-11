# Distant context cues: what twelve papers say, and what we should build

October 2026. Companion to `docs/audit_2026-10.md`. The twelve papers were read in full (text extracted with `pdftotext -layout`, every page including appendices), one reader per paper working from a shared brief, followed by a spot-check of the load-bearing quotes against the texts. The per-paper notes — equations copied in each paper's notation, verbatim quotes with page numbers, loose threads, cited works to pull — are in this directory (`docs/reading/<arXiv id>.md`, about 93,000 words). This file is the synthesis. Everything marked **[inference]** is ours, not the papers'.

The question, as asked: what could Pathway's proprietary "context system" in BDH-CQ be, what have other people built to carry a context cue across distance, and what should we implement.

---

## 0. The short version

1. **BDH-CQ's memory rule is almost certainly a member of the delta-rule family**, and its reasoning loop is almost certainly a TRM/recurrent-depth-style refinement with the frozen memory re-injected at every step. Neither is a new mechanism for distant cues. The phrase "linear correction rules on S" (BDH-CQ §3.2) picks out, in the taxonomy of Wang, Shi & Fox (2501.12352) and of Nested Learning, exactly the rules that are affine in S *and read S before writing*: delta / Longhorn / NLMS / Gated DeltaNet / DeltaProduct / DGD. BDH's own published rule, ρ_t = ρ_{t−1} + LN(E y_t) x_tᵀ U (2509.26507, eq. 7–8), is the pure additive special case Pathway names, and the BDH paper itself lists "selective forgetting, state compression, or other forms of state optimization" as things to add (§6.1). The refinement loop H_{r+1} = F_θ(H_r, S_K) matches TRM's `z ← net(x + y + z)` with x re-injected and Geiping et al.'s s_i = R(e, s_{i−1}), and Geiping et al. explain *why* the conditioning object must enter every step (the iteration cannot be a stable, data-dependent operator otherwise). **Correction to what was said earlier in this conversation:** the 97.4% Sudoku figure is from Pathway's blog ("Pathway internal data", "roughly 250,000 of the toughest Sudoku puzzles", top-1, protocol otherwise unstated), not from the BDH-CQ paper, and the test set differs from HRM/TRM's 1K-train / 423K-test Sudoku-Extreme; it is not a like-for-like 87.4 → 97.4 comparison.

2. **No paper in the set carries an inferred context cue across distance label-free.** Every one either hands the context over (Active Dendrites: task id or prototype; Flesch: a cue on every trial; Context-Gated Associative Retrieval: a clamped context layer; HM-RNN: boundaries inferred, but from a dense character-level loss), carries it with softmax attention (TRM, recurrent depth, the routing paper's preprocessing), or routes by token type (MoM, Raven). The distant-cue problem as we have posed it — a header token must govern the routing of the following 2P tokens, discovered from an end-of-sequence loss alone — is open in this literature. That is the honest headline.

3. **The papers converge on a diagnosis of why our header-layout runs failed, and the diagnosis says they were expected failures, not bugs.** (a) A router cannot use a cue it is not given: the routing paper's routers reach 0.9–29% on raw embeddings, unidirectional recurrent summaries, and mean/max-pooled summaries *even with the correct route supervised*, and 98–99.7% as soon as an upstream, jointly-trained mixing step writes the context into each token's vector. Our window gate reads three raw embeddings. (b) Uniform pooling destroys the cue: an EMA over a block mixes one header with 2P key/value tokens, which is why our fading-memory gates (0/10) and the window gate's key splits are the same failure; Flesch et al.'s EMA works only because it is applied to the cue channel alone, and their own result is that a slow integrator *merges* interleaved contexts (our grouped layout). (c) Inside a block the loss exerts no pressure to use the cue (Flesch: "the network failed to utilise the task units … as the task signal was not required to solve individual tasks in isolation"), and a Hebbian single channel gets partial credit by superposition at P = 4, so the gradient toward stream routing is weak. (d) At fixed P and S the header layout is solvable by position (TRM's purely positional [L, L] mixer is the *best* Sudoku solver and fails the moment the layout varies) — a position-counting recurrent gate could pass it without inferring anything.

4. **What to build, in order of cost.** Each item has a precedent and a falsifiable prediction (§4):
   - **Delta-rule channel memory** (Gated DeltaNet eq. 10, keys L2-normalised): a single channel then provably holds one value per key, so it *cannot* bind conflicting streams — the clean negative control the audit asked for — and the gradient toward routing sharpens.
   - **A "latest-header" register**: one layer-1 channel with a constant key and a learned write strength that overwrites on CTX tokens (delta rule) or latches (HM-RNN's COPY/FLUSH with a straight-through estimator). Overwrite semantics are what a Hebbian memory with decay lacks: it sums headers, weighting the latest only 1/0.95^(2P+1) ≈ 1.6× above the previous at P = 4.
   - **Per-layer gate over the residual stream**, with the raw embedding concatenated through a learned adapter (Geiping's design; MoM, Raven and Gated DeltaNet all compute their gates from the layer input as a matter of course). Before building it, run the diagnostic the routing paper and the Hopfield paper both use: a linear probe for stream identity on the residual stream at K/V positions, per layer. If the probe fails, no gate on top can succeed.
   - **Resettable integrator** (Flesch's "sluggish" cue with a token-conditioned α; the same object as Gated DeltaNet's α_t → 0 "clear at context switches"). A fixed α cannot serve both layouts: grouped needs α ≈ 0, header needs α ≥ 0.92–0.96.
   - **Task redesign**: randomise block lengths, raise S·P past single-channel capacity, keep the oracle ceiling. A delta-rule single channel makes "bound ⇒ routed" true by construction.
   - **Forget gate**: Gated DeltaNet's α_t = exp(−A·softplus(w·x_t + b_Δ)) per channel, log-uniform time-scale init, no weight decay on A, b_Δ; or Raven's routed decay exp(a_t g_t[c]) — a channel decays only in proportion to how much is written to it, which is the mechanism behind Raven's 16× length extrapolation.
   - **Anti-collapse alternatives to the plateau split**: Gumbel(0,1) noise on the gate logits, train-only (Raven's only device); a Switch-style balance loss at 1e-3 (MoM; without it 9 of 24 MoM layers collapse to a fixed pair); kWTA/absolute-max hardness (Active Dendrites). Two warnings: routers learn in a phase transition after a long plateau (routing paper: 9 epochs flat, then 2.8% → 99.4% in one epoch), so "flat" is not "dead"; and a self-reinforcing gate (recurrence, or feedback from what the memory retrieved) goes winner-take-all onto content-free directions when the context drive is weak (Context-Gated Associative Retrieval, Theorem 3.3/C.2) — a clean theory of our position and key splits.

---

## 1. One line per paper

| Paper | What it is | What it gives us | Caveat |
|---|---|---|---|
| **Test-time regression** (Wang, Shi & Fox, 2501.12352) | Theory: every associative-recall layer = weighted regression solved at test time; linear attention = one unpreconditioned GD step; delta rule = SGD; softmax with QK-norm = Nadaraya–Watson | The vocabulary for BDH-CQ's rule; our k-channel memory is feature-mapped linear attention with φ(x) = x ⊗ g, "optimal only when the stored features are orthonormal" = the oracle gate; channel merging is its "ignores the covariance" failure; the header problem is a *key-construction* problem | "Correction" never appears in the text; no training recipes; the experiments are two figures without numbers |
| **Gated DeltaNet** (Yang, Kautz & Hatamizadeh, 2412.06464) | S_t = S_{t−1}(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ; chunkwise WY algorithm; 1.3B/100B results | The write rule to transplant (overwrites a key's value; L2 keys; sigmoid β); the forget-gate parameterisation and init (from the official code); S-NIAH evidence that a learned α erases lone distant facts | Its answer to "same key, two contexts" is *overwrite*, not coexistence; heads are never treated as routed memories |
| **TRM** (Jolicoeur-Martineau, 2510.04871) | 7M-param recursive refinement: z ← net(x + y + z) ×6, y ← net(y + z), T = 3, deep supervision ×16, EMA; Sudoku-Extreme 87.4 (HRM 55.0, HRM best 62.3) | Template for BDH-CQ's F_θ, G_θ and training loop; "no fixed point is reached" (1-step IFT gradient 56.5 vs full backprop 87.4); a positional [L, L] mixer wins on fixed layouts and fails on varying ones | No memory, no in-context learning (ARC demonstrations are training data with a per-puzzle ID token); "more recursion" untested; several internal typos (listed in the notes) |
| **Recurrent depth** (Geiping et al., 2502.05171) | 3.5B LM with a weight-shared block iterated r ~ log-normal Poisson times, s_i = R(e, s_{i−1}), e concatenated in at every step, truncated backprop k = 8 | Why S_K must be re-injected every step (monotone-operator argument, fn. 1); the "model learns to ignore its state" failure (Bad Run 2) and its cure (sandwich norm, learned concat adapter, 10× lower lr); health metrics to copy | Carries context with softmax attention; most of its fixes "only bite at scale" |
| **Flesch et al.** (2203.11560) | 2-task MLP; "sluggish" EMA on the *cue* reproduces human blocked > interleaved; Oja's rule on input weights discovers the context as PC1 and sign-splits hidden units by context | The clean statement of our two missing ingredients and their preconditions: integrator must receive only the cue; slow integration merges interleaved contexts; loss gives no pressure to use a constant cue; label-free partition needs the cue to dominate variance (only works for 2 contexts) | Cue present on every trial, never missing; EMA is fixed preprocessing; two contexts; code and paper disagree on which weights Oja touches (notes §1.4) |
| **Active Dendrites** (Iyer et al., Numenta, 2201.00042) | Per-unit multiplicative gate σ(max_j u_jᵀc) driven by a *separate* context vector + layer kWTA; MT10 87.5 vs 76.6; pMNIST 81.4 (100 tasks), 76.9 fully unsupervised | Given a clean context vector, backprop alone learns a near-orthogonal partition with no STE or auxiliary loss; decouple the gate's input from the content path; hardness (kWTA) is the component whose removal causes the "sharp drop" | Context is a task id or readable off every input (permutation signature); feedforward; relies on thousands of units so losers never matter |
| **Context-Gated Associative Retrieval** (Choraria et al., 2605.10970) | Theory (cond-mat.dis-nn): softmax Hopfield read with an additive per-memory gate bias λs_μ; gate = context drive + competition ∝ memory overlap + λ·retrieval feedback; fixed point unique iff βλ²/(2η_min) < 1 | Theorem 3.1: the gate contrast needed grows only as (1/β) ln((1−ε)(N−1)/ε); a theory of why self-reinforcing gates collapse onto content-free directions; linear-probe diagnostic (App. D.4) | Nothing is trained; context clamped in every theorem; the LLM "model" is h_zero + λ·task-vector decoded through W_U |
| **MoM** (Du et al., 2502.13685 v4) | M = 4 linear-attention memories + shared memory, per-layer softmax top-2 router on the layer input, Gated DeltaNet update, Switch aux loss 1e-3 | Closest large-scale precedent; mixture > one expanded memory at equal capacity; **Fig. 9: without the aux loss 21/24 layers have an always-selected memory, ~9 collapse to a fixed pair**; routers specialise by token type (= our key splits); shared channel worth +2.1 recall | Never asks whether routing can depend on context; shared-memory combination rule never written; a collapsed MoM still beats GDN by 2.5 (so part of the gain is per-memory K/V projections, not routing) |
| **Raven** (Afzal, Bick et al., 2607.25357) | 256 slots/head, sigmoid + Top-K write router on the layer input, Mamba-2 decay applied only to written slots, dense softmax read, Gumbel(0,1) logit noise as the sole anti-collapse device, no balance loss | Routed decay exp(a_t r_t[i]) (frozen slots → 16× extrapolation); "intentional imbalance"; existence proof that a per-layer residual router trains stably with no tricks | Write-only routing, dense read → cannot disambiguate same-key slots; **headline 400M row matches the Top-128 config, not the Top-32 "default"**; loses to GDN on NIAH-3 and at 800M LM |
| **When Does Content-Based Routing Work?** (Basu, 2603.20997) | 20+ router variants on a distant-retrieval task; routing *supervised* throughout; raw/pooled/causal-summary routers 0.9–29%, one mixing layer before the router 98.4–99.7% | The sharpest diagnosis of our header failures; phase-transition learning dynamics; "routing signal lives in a ~34-dim subspace, invisible to cosine similarity" → probe, don't eyeball | Single author, 486K-param core model, inconsistent tables; "distant" is the target not the cue; never label-free |
| **HM-RNN** (Chung, Ahn & Bengio, 1609.01704) | Per-layer binary boundary detector; COPY (lossless hold) / UPDATE / FLUSH; straight-through estimator, slope annealing 1 → 5; boundaries learned from char-level NLL | The template and recipe for a boundary-latched register; the one-sentence case against leaky integration for holding a cue | No appendix, no init details, no anti-collapse device beyond FLUSH's implicit penalty; boundaries only roughly linguistic; annealing "unusable" at scale |
| **Nested Learning / HOPE** (Behrouz et al., 2512.24695) | Everything (layers, optimisers, fast weights) as associative memories updated by GD on an inner loss, ordered by update frequency; CMS = MLPs test-time-trained every C^(ℓ) tokens; self-modifying Titans | The rule taxonomy (state-independent Hebbian vs state-dependent delta/DGD); the diagnosis that a one-layer projection "is a function of the token itself and its position"; retention/delta/momentum ablations (Table 6) | CMS periods are fixed positional constants, no boundary inference; several equations inconsistent with the ablations (momentum never written); no hyperparameters, no code |

---

## 2. BDH-CQ: what the proprietary system most plausibly is

### 2.1 The memory update

Pathway (2608.09888 §3.2): "its recurrent memory evolves as S_t = U_θ(S_{t−1}, D_t) … with linear attention being the conceptually simplest standalone realization of linear correction rules on S, capturing the special case S_t = S_{t−1} + U_θ(D_t)." §3.3: "Dimensions, exact update rules, and implementation details remain proprietary."

Three papers in the set each lay out the same ladder of rules, independently, and the ladder is the reconstruction:

| Rule | Form S_t = A(D_t) S_{t−1} + B(D_t) | Reads S before writing? | Named in |
|---|---|---|---|
| Linear attention (BDH's eq. 7–8) | A = I, B = v kᵀ | no — Pathway's "special case" | TTR Vignette 1; GDN Table 1; NL eq. 18; MoM Table 1 |
| Gated LA / Mamba-2 / GLA / RetNet | A = α_t I (or diagonal), B = v kᵀ | no | all four |
| DeltaNet / Longhorn / NLMS | A = I − β_t k kᵀ, B = β_t v kᵀ | **yes**: B − (I − A)S = β_t(v − S k)kᵀ is a prediction-error correction | TTR Vignette 3 ("Widrow–Hoff / LMS"); GDN eq. 3; NL eq. 65 |
| Gated DeltaNet / "Leaky LMS" / DGD with weight decay | A = α_t(I − β_t k kᵀ) | yes | GDN eq. 10; TTR eq. 41; NL eq. 88 |
| DeltaProduct | A = Π_i (I − β^{(i)} k kᵀ) | yes | TTR p. 13 |
| Exact RLS / Mesa layer | needs a second (d×d) covariance state | — | TTR eq. 11; GDN §Related |
| Titans / TTT (MLP memory) | nonlinear in S | — | all |

**[inference]** "Linear correction rule" read literally — U linear in S with input-dependent coefficients, and a *correction* — selects rows 3–5. Row 1 is Pathway's own special case, rows 6–7 are not linear in S. The Test-time regression paper never uses the word "correction" (checked: zero occurrences), but its vocabulary for row 3 is "rewriting the value that was bound to key k_t", "deletes an old value", "SGD on the least-squares objective"; Gated DeltaNet's is "dynamically erases the value (v_t^old) associated with the current input key (k_t) and writes a new value". That is what a correction rule is. Gated DeltaNet is the hardware-efficient, scale-tested member; MoM and Raven both build on it; Pathway's BDH paper already lists "selective forgetting" as the next addition. Confidence: moderate-to-high that U_θ is in this family; low on which member.

Two further hints, neither decisive:
- Raven credits the general two-sided linear update S_t = D_t S_{t−1} A_t + U_t to Zhong et al. (2505.19488, "Understanding transformer from the perspective of associative memory") — a candidate source for the phrase "linear rules on S".
- Nested Learning's self-modifying Titans makes the *key projection itself* a fast weight written by earlier tokens (k_t = M_{k,t−1}(x_t)), so the same token is projected differently after different contexts. Pathway's "structured latent workspace" could include something of this kind; there is no textual evidence either way.

### 2.2 The refinement loop

Pathway: H_0 = E_θ(x*, S_K), H_{r+1} = F_θ(H_r, S_K), ŷ = G_θ(H_R).

TRM (verbatim, p. 5–6): `z ← net(x + y + z)` repeated n = 6 times, then `y ← net(y + z)`, T = 3 such processes per supervision step with the first T − 1 under `no_grad`, up to 16 supervision steps with (y, z) detached between them, `ŷ = argmax(output_head(y))`, a BCE halting head, EMA 0.999, AdamW β₂ = 0.95, weight decay 1.0, stable-max loss. Geiping et al.: e = P(x), s_0 ~ N(0, σ²I), s_i = R(e, s_{i−1}) with e concatenated through a learned adapter ("concatenation works best at scale"), r ~ log-normal Poisson around 32, backprop through the last 8 iterations, no step index fed to the block (it "interacts badly with path independence").

The mapping is structural and strong: H ↔ TRM's (y, z) or Geiping's s; S_K ↔ TRM's x / Geiping's e (the fixed conditioning object re-injected every step); G_θ ↔ output_head(y) (decode a sub-part of the state); E_θ ↔ TRM's first recursion process from (y_init, z_init). Geiping et al. supply the reason S_K must be in every step: "If e was provided only at the start … the iterative process would not be stable, as its solution would depend only on its boundary conditions" (fn. 1: R "cannot be a monotone operator if it does not depend on e"). **[inference]** Pathway's formula has x* entering only through H_0; by Geiping's argument either x* is also ingested into S_K, or H_r is position-resolved over the query so x* persists in it. The Hopfield paper adds one more template: its self-consistent map p* = softmax(b + λA⁻¹u + λ²A⁻¹p*) is literally "read the fixed memory with a query that includes a feedback term from the latent, normalise, repeat", with a contraction condition βλ²/(2η_min) < 1 and the remark that feed-forward transformers "lack single-layer recurrence" and so cannot implement the feedback term.

What the loop is **not**: a fixed-point/DEQ solve. TRM's ablation (1-step IFT gradient 56.5 vs full backprop 87.4, and TorchDEQ "slower and worse") and Geiping's orbits and sliders both say the iteration need not converge, and the gradient should go through the whole short chain.

### 2.3 The Sudoku number

The 97.4% is from Pathway's blog post "Beyond Transformers: A New Architecture for Solving Sudoku" (pathway.com/research/beyond-transformers-sudoku-bench), sourced to "Pathway internal data", described as top-1 without chain-of-thought or backtracking on "roughly 250,000 of the toughest Sudoku puzzles", compared only to LLMs at 0%. The BDH-CQ paper cites that post but does not contain the number. TRM's 87.4 is on HRM's protocol (1K training puzzles with 1,000 rule-preserving shuffles, 423K test puzzles, full-grid exact match). Training-set size, test set, and voting are unstated for Pathway's run, so the two numbers should not be placed on one ladder. For scale, TRM's notes recall that Palm et al. 2018 (Recurrent Relational Networks) reached ≈96–97% on 17-given Sudokus with ~180K training puzzles — mid-90s is reachable by relational recursion when data is not scarce.

### 2.4 What none of this gives Pathway

Nothing in §2.1–2.2 is a distant-cue mechanism. In ARC, every demonstration is written into S before the query arrives, and the whole demonstration set is the context; "distance" never has to be bridged by anything other than the memory's persistence. Our question — which of several conflicting contexts governs *this* token — is not one BDH-CQ poses.

---

## 3. Why the header layout has failed: the unified diagnosis

Our record (Rev. 7 §Distant cues; audit §3.1): recurrent gate 0/10, window gate 2/10 (key splits), multi-scale EMA gates 0/10, selective gate 0/10; 5% labelled nudge 10/10 with slow memory; a single channel binds the header layout at P = 4 (3/10, 7/10 with slow memory); at P = 8 the oracle did not bind within 24,000 updates.

Four independent lines in the papers explain each entry.

**(a) The router is not given the cue.** The routing paper holds everything fixed except the representation fed to the router and finds: raw embeddings 1.2%, unidirectional recurrent state 2.2% (end-to-end) or 22–29% (frozen pretrained), global mean + broadcast 1.9%, segment mean/max pooling 24–26%, Fourier mixing 0.9%; one attention layer (causal or bidirectional) 98.4–99.2%, bidirectional Mamba 99.5%, a single Perceiver inducing point 98.8%, sparse attention with 16 random global tokens 99.7% — *all with the correct route supervised by a cross-entropy on the answer position*. Its conclusion: "routing requires relational information to be written into per-token representations" by an upstream step trained jointly with the router; "the match result must become part of the representation, not something the router must independently discover." Our window gate reads three raw embeddings, the Raven/MoM/GDN-style layer-1 gate reads one. Nothing we have tried gives the gate a vector into which the header has been written.

**(b) Uniform summaries destroy the cue.** Flesch et al.'s "sluggish" signal x̃_t = (1−α)x_t + αx̃_{t−1} is applied to the *two task units only* (Methods p. 25; confirmed in their code), as fixed preprocessing, and never tested with a missing cue; with x_t = cue at t = 1 and 0 afterwards it decays as α^{t−1}, needing α ≥ 0.917 to keep half the amplitude over 8 tokens and α ≥ 0.958 over 16. Fed every token, an integrator accumulates K/V content against which the fading header must be decoded — exactly the regime of our EMA gates. HM-RNN states the general point: leaky integration dilutes "the contents to memorize for a long-term … at every time step" (p. 3), which is why it holds a slow state by lossless COPY and resets it by FLUSH rather than by any decay rate.

**(c) The loss does not push toward the cue inside a block, and superposition gives partial credit.** Flesch (p. 9): "Under blocked training, the network failed to utilise the task units to implement this gating scheme, as the task signal was not required to solve individual tasks in isolation." Within a header-layout block the stream is constant, so no write-time decision is rewarded for depending on it; only the end-of-sequence queries carry the signal, and at P = 4 a Hebbian single channel answers a good fraction of them by superposition (the Test-time regression paper's "crude memory that sums" — exact only for orthonormal features, but useful well before that). The gradient toward stream routing is therefore small and competes with cheaper correlates (key identity, position). The routing paper's learning curves add that when routing *does* become learnable it happens as a phase transition after a long flat plateau (9 epochs at chance, then 2.8% → 99.4% in one epoch; failing conditions stay flat forever) — a plateau is not a verdict.

**(d) At fixed layout, position is a valid solution.** TRM-MLP, a purely positional [L, L] mixer with no content routing, is the best Sudoku-Extreme solver (87.4 vs 74.7 for attention) and scores 0.0 on Maze and 29.6 on ARC where layouts vary. In our header layout with fixed P and S, block membership is a deterministic function of position; a recurrent gate that counts can route the writes correctly without inferring the stream, and a single channel can exploit the same regularity through the conv. Any distant-cue task meant to test *inference* must randomise block lengths (and ideally the header's position within a block).

Two theoretical companions. The Hopfield paper's Theorem 3.3(ii)/C.2: with a competitive or self-reinforcing gate and an uninformative context drive, the winner is set by the projection of the initial state onto the null space of the content Gram matrix — a content-free direction — and the uniform state is unstable for all α > 0 once λ²β ≥ N. **[inference]** That is a formal account of position and key splits: an unsupervised competitive gate latches onto whatever is most linearly salient, not onto what the task needs. And Nested Learning's bottleneck statement (p. 31): a one-layer projection of a token "is a function of the token itself and its position; therefore … it can miss the diverse possible encodings of words whose meaning depend on the context" — which is BindTask in one sentence.

---

## 4. What to build

Notation: v_t raw embedding; x_t the BDH activations used as keys (score = x_t·x_s); r_t^{(ℓ)} the residual stream entering layer ℓ; g_t ∈ Δ^k the gate; channel memory S^c.

### 4.1 Delta-rule channels (Gated DeltaNet eq. 10, per channel)

    S^c_t = S^c_{t−1} ( α^c_t ( I − g^c_t β_t x̂_t x̂_tᵀ ) ) + g^c_t β_t v_t x̂_tᵀ,     x̂_t = x_t / ‖x_t‖₂
    read:  o_t = Σ_c g^c_t S^c_t x̂_t   (as now, with the read gate)

Equivalently S^c_t = α^c_t S^c_{t−1} + g^c_t β_t (v_t − α^c_t S^c_{t−1} x̂_t) x̂_tᵀ: the write is the per-channel prediction error. β_t = σ(w_b·v_t + b_b) ∈ (0,1) (GDN); keep α as our decay for now (0.95, or §4.6).

Precedent: GDN (eq. 10; L2 keys "essential for optimal performance", Table S.1; naive delta without gate costs 3.5 ppl); TTR Vignette 3 (the delta rule is SGD on the objective our Hebbian write crudely solves; "forget only the values whose keys are most similar to the new key"); NL Table 6 (DGD term, retention, momentum each worth > 1 ppl). MoM and Raven both run on it.

What it buys us, **[inference]** but exact algebra: with β = 1 and unit keys a single channel stores exactly one value per key direction — writing (KEY_i → VAL_{i,2}) erases the component along x̂_i that encoded VAL_{i,1}. So **a single delta channel provably fails on all but the last-written stream, in both layouts, while the oracle partition binds trivially**: the "task a single channel provably cannot solve" the audit asked for, without pushing P to where the oracle stalls. Within a correctly routed channel, non-orthogonal keys no longer cross-talk (least-squares fit), which should raise the oracle ceiling at P = 8. Side effect: the superposition that gave a single channel partial credit at P = 4 is gone, so the routing gradient is sharper. Cost: one read S^c x̂_t per channel per token, which we already do.

Caveats: the write/read offset must be preserved (the key used in the erase at the VAL write is the preceding KEY token's x; the query presents the same vector). Keep decay and β decoupled (TTR p. 14 recommends decoupling GDN's coupled λ, β). Keep LayerNorm after attention (TTR eq. 13: Hebbian outputs grow with t without it; the delta rule's self-normalisation is partial).

Prediction: single-channel BOUND on the header layout at P = 4 falls from 3/10 (7/10 with slow memory) to 0/10 at β ≈ 1; oracle BOUND at P = 8 reaches budget where the Hebbian oracle did not; grouped-task results under the window gate unchanged or better.

### 4.2 A "latest-header" register (two implementations of one idea)

The idea, from GDN's overwrite semantics and HM-RNN's COPY/FLUSH: a small state that is *cleared and rewritten on a cue and held exactly otherwise*. A decaying Hebbian memory cannot do this: with constant key e₀ it returns Σ_b decay^{t−t_b} header_b, so at P = 4 the current header outweighs the previous by 1/0.95⁹ ≈ 1.6; with the delta rule it returns the latest header exactly, independent of P.

**(i) Delta register (differentiable, no STE).** Append a constant feature e₀ to the key of one layer-1 channel (or use a dedicated register channel), with its own write strength:

    R_t = R_{t−1}(1 − β^R_t) + β^R_t · W_R v_t,      β^R_t = σ(w_R·v_t + b_R)
    read at every position: ρ_t = R_t  (the constant key makes the read trivial); concatenate ρ_t into the layer-2 gate input (and optionally into the residual stream)

This is Longhorn's "when δ_t = 0 the latest association is excluded completely" (TTR p. 13) plus the delta rule's "rewriting the value that was bound to key k_t" (p. 5), on a one-dimensional key. β^R → 1 on CTX tokens and → 0 elsewhere is the target behaviour; it must be *learned*. Initialise b_R so β^R ≈ 0.1 (mostly hold) and w_R small.

**(ii) HM-RNN latch (hard).** From eqs. 2–3 and 8–9 of HM-RNN with the roles inverted so the register is FLUSH-written and otherwise COPIED:

    u_t  = w_z·v_t + b_z            z̃_t = hard-sigm(u_t) = max(0, min(1, (a·u_t + 1)/2))
    z_t  = 1[z̃_t > 0.5] forward;   ∂z_t/∂u_t := (a/2)·1[|u_t| < 1/a] backward (straight-through)
    r_t  = z_t · W_r v_t + (1 − z_t) · r_{t−1}
    gate input: [Σ_{j<3} w_j ⊙ v_{t−j} ; r_t]

Recipe from the paper: step function forward (beats Bernoulli sampling and soft gating, 1.24–1.25 vs 1.27 BPC), slope a annealed 1 → 5 linearly over training (PTB: a = min(5, 1 + 0.04·epoch)), Adam, gradient clipping at 1, LayerNorm. **[inference]** The STE gradient is exactly zero outside |u| < 1/a, so initialise so every token's u lies inside the band (b_z = 0, small w_z, normalised inputs); a detector initialised confidently in either direction is dead (always-0 = constant gate = single channel; always-1 = the window gate). The paper has no anti-collapse device beyond FLUSH's "reward … and a penalty"; if needed, add a firing-rate prior λ(mean_t z_t − 1/(2P+1))² (Skip-RNN's budget term, 1708.06834). Kádár et al. 2018 (1807.03595) report HM-LSTM's boundaries are only weakly linguistic — temper expectations.

Both implementations are the same mechanism as a **resettable integrator** (Flesch's sluggish cue with token-conditioned α; GDN's α_t → 0 "to promptly clear memory"): c_t = α_t ⊙ c_{t−1} + (1 − α_t) ⊙ P v_t with α_t = σ(a·v_t + b). The three differ only in how hard the clear is. Prefer (i) for the first screen (no STE, no annealing), (ii) if (i) learns a leaky compromise.

What the papers do **not** give us: evidence that β^R or z_t will be discovered label-free under our sparse loss. HM-RNN's boundaries emerged from ~10⁸ character predictions; Raven's and GDN's gates from LM loss at 15–100B tokens. Our 5% nudge → 10/10 is the local analogue of "it can express it". So: train the register first with the labelled nudge to confirm the mechanism and the read path, then remove the nudge on the redesigned task (§4.5). Prediction: with the nudge, header-layout BOUND ROUTED ≥ 9/10 at P = 4 and P = 8 under the delta oracle budget; without it, strictly better than the window gate's 2/10 or we learn that the discovery signal, not the architecture, is the bottleneck.

### 4.3 Per-layer gate over the residual stream, with the embedding re-injected

    u_t^{(ℓ)} = A [ RMSNorm(r_t^{(ℓ)}) ; v_t ]          A learned, (N + d_v) → N
    g_t^{(ℓ)} = softmax( W_g tanh(u_t^{(ℓ)}) ),       layer 1 keeps the window gate (no residual yet)

Precedent: Geiping et al. — the recurrent block sees the state *and* the raw embedding, concatenated through a learned adapter ("concatenation works best at scale"; the parameter-free sum A(s, e) = s + e was part of the collapsed run), normalised before and after; MoM, Raven, GDN compute every gate (router, α, β) from the layer input as the default design, and Raven's heads/layers specialise differently (Fig. 7); the routing paper's bridge result (frozen Pythia 35% → 99.4% with one trainable layer inserted before the router) says a bare linear read-out of features not shaped for routing is not enough — give the gate a trainable nonlinearity and let gradient flow into layer ℓ−1. TRM supports one shared gate network across layers ("the task … is directly specified by the inclusion or lack of x in the inputs"). Init A so that u^{(ℓ)} ≈ W_in v_t at step 0 (the residual path starts as a perturbation of the token path — Geiping's small out-projection init; TRM's W_r = 0 idea).

For this to solve the header layout, layer 1 must write the block header into every position of the block. **[inference]** With a content-addressed Hebbian read that requires x_K·x_CTX ≠ 0 for K/V tokens — a shared "header-seeking" component in the activations — and even then the read is a decay-weighted sum of all headers (§4.2). The delta register of §4.2(i) is the clean way to do layer 1's job; design 4.3 then only needs to read it. The two designs are complementary, not alternatives.

**Diagnostic to run first** (routing paper §4.2; Hopfield paper App. D.4): on models trained on the header layout, fit a logistic-regression probe from r_t^{(ℓ)} at K and V positions to the true stream, per layer. If stream identity is not linearly decodable at layer 2, no gate reading layer 2 can route by it and the fix is upstream (register / embeddings / decay). Do not use cosine similarity between stream-mates: the routing signal "occupies roughly a quarter of the representation space — invisible to full-space cosine similarity" (matching tokens had a *negative* cosine gap, −0.154).

Risk already on record: `test_readout_path` showed gradient coupling from the gate's state into the residual harmed the gate; the reverse coupling is untested — run with and without a stop-gradient from the gate into the stack (audit §3.3).

### 4.4 Routing by what the memory returned (two-pass), with a sign warning

The Hopfield paper's fixed point s* = A(α)⁻¹(u + λp*) — gate = stable linear filter of (context drive + λ·current retrieval distribution) — and the routing paper's "value aggregation writes the match results back into each position's representation" both support a two-pass token step: probe-read every channel, then route by what came back. The paper's contraction condition (βλ²/(2η_min) < 1; with A = I and softmax temperature 1, λ² < 2) bounds the feedback gain, and its Theorem C.2 says that above it the loop performs WTA on its own, with the winner set by initialisation.

**[inference, the sign warning]** The paper's feedback is positive (retrieved → gate up → retrieve more). Transplanted naively — "write the token where its key is already known" — a familiarity-driven gate would send all S values of the same key into the same channel, reinforcing key splits. For conflicting bindings the useful run-time signal at a VAL position is *disagreement*: e^c_t = v_t − S^c_{t−1} x̂_t is small in the channel that already holds a consistent binding for this key and large in a channel holding a conflicting one; route the write away from large-error channels. This exists in no paper; GDN's per-channel prediction error makes it available for free once §4.1 is in. It helps only the write side; the read at the query still needs CTX_q, which it has (one token back).

### 4.5 Task redesign: a distant-cue task that is informative

From §3(d), TTR §6 ("memory capacity, not sequence length, limits model performance"), and the audit's own requirement:
- Delta-rule channels (§4.1) so that a single channel cannot bind conflicting streams at any P.
- Random block length per block (P_b ∈ {2, …, P_max}) and, optionally, the header at a random position within the first few tokens of its block, so that position cannot route.
- S·P large enough that the Hebbian single channel is also hopeless, but with the oracle's budget verified first (the P = 8 oracle stall at 24,000 updates is an optimisation/readout issue per TTR's capacity view — N = 256 ≫ P·S — and should be re-measured with delta channels and a longer budget before anything is read from it).
- Report BOUND ROUTED with the routing margin measured at VAL positions (audit §3.2), and the CTX/KEY/VAL × stream contingency table of g_t (MoM's heat-map diagnostic) so key/position/stream splits are visible at a glance.

### 4.6 Memory horizon: forget gates with precedent

Two published forms, both learned from the loss with no auxiliary signal:

- **Gated DeltaNet / Mamba-2 scalar decay per channel** (from the official GDN code; the paper only says "Mamba2's parameterization"):

      α^c_t = exp( − A_c · softplus( w_a^c · x_t + b_Δ^c ) ),   A_c = exp(A_log_c) > 0 learnable
      init:  A_c ~ U(0, 16);  b_Δ^c = softplus⁻¹(dt_c), dt_c log-uniform in [0.001, 0.1];  no weight decay on A_log, b_Δ
      half-life = ln 2 / (A_c · Δ);  our 0.95 ↔ A·Δ ≈ 0.051

  The log-uniform init spreads channels across time scales from under a token to effectively unbounded. GDN's own S-NIAH-1 result is the warning: the learned gate erases lone distant facts (Mamba2 65.4/30.4 at 4K/8K on pass-key; GDN 91.4/91.8; ungated DeltaNet 99.0/98.8). Persistence should come from the delta rule's content-addressed semantics (an association lives until its key is rewritten), not from α → 1 alone.

- **Raven's routed decay**: exp(a_t · g_t[c]) — a channel decays only in proportion to how much is written to it; unselected slots "experience no decay regardless of a_t". The pairwise score becomes (x_t·x_s) Σ_c g_t[c] g_s[c] Π_{u=s+1}^{t} exp(a_u g_u[c]): a stream's binding survives untouched while other streams' blocks are routed elsewhere. This is the mechanism behind Raven's 16× extrapolation and it directly enlarges the gap between oracle partition and single channel. To reproduce current behaviour at g = 1: with w = 0, softplus(0) = 0.693, set ∆ = ln(−ln 0.95/0.693) ≈ −2.6.

Flesch et al. add the argument that no fixed time constant serves both layouts and that the integration window should "adapt to the volatility of the environment" (their stated future work) — i.e. the forget gate should be token-conditioned, which both forms above are. Nested Learning's CMS (fixed update periods C^{(ℓ)}) is the one multi-timescale design in the set that is *not* data-dependent; its only use for us is a cheap variant — log-spaced fixed decays across channels (half-lives 4, 14, 50, 200) — which is untried and separable from routing.

### 4.7 Training and anti-collapse: what to borrow

- **Gumbel(0,1) noise on the gate logits, train only, no annealing** (Raven eq. 23; its sole anti-collapse device; no balance loss, "intentional imbalance"). For a softmax gate this is a Gumbel-softmax sample at temperature 1. **[inference]** Put it on the write gate only: in Raven a mis-routed write is still readable (dense read), in our model a noisy write to c′ is read only if the query is also routed to c′. Raven's gain is not universal cell-by-cell (Table 6) — measure on the grouped task before adopting.
- **Switch-style balance loss** (MoM; Fedus et al. form L_aux = α·k·Σ_c f_c P_c, α = 1e-3 best, non-monotonic in α): continuous, cheap, attacks merges; uniform usage is neither necessary nor sufficient for stream routing.
- **Hardness** (Active Dendrites: kWTA is the component whose removal causes the "sharp drop"; absolute-max gating so a unit can be switched off; losers get zero gradient and stay put — they survive on thousands of units, we would need boosting (Cui et al. 2017) or our split). Anneal softmax → top-m late in training rather than only penalising early.
- **Spare channels and many slots, no balancing** — supported by Raven (M = 256 ≫ K) and Active Dendrites; Raven's Table 7 warns that too many slots with too little capacity each collapses some tasks.
- **Health metrics** (Geiping): cross-token correlation of g_t (all tokens routed identically = the collapse signature); ‖h_t − h_{t−1}‖/‖h_t‖ for any recurrent gate; loss at R = 1 vs R = max if we ever iterate in depth.
- **Stability kit** (TRM): EMA 0.999 of weights for evaluation, AdamW β₂ = 0.95, stable-max loss / a temperature floor on the gate softmax (Prieto et al. 2501.04697: softmax saturation kills gradients and causes plateaus — a candidate account of our plateau-triggered splits).
- **Learning rate**: Geiping's "model learns to ignore its state" minimum was cured by a 10× lower peak lr plus norms; supports the slow-memory recipe in spirit, and suggests a lower rate on the gate's *recurrent* weights specifically if we keep any recurrence.
- **Phase transitions**: the routing paper's routers sit at chance for 9 epochs then jump to 99% in one; failing conditions never move. Train long enough to tell the two apart, and treat plateau-triggered interventions as acting on the right quantity.

---

## 5. Corrections to statements made earlier in this conversation

1. "Pathway reports 97.4% on Sudoku-Extreme" → the figure is from Pathway's blog, "Pathway internal data", protocol unstated, not from the BDH-CQ paper; not comparable to TRM's 87.4 (§2.3).
2. "Test-time regression (Wang & Rush)" → Wang, Shi & Fox (Stanford); Rush is cited for a tweet.
3. "Pooled-context routers fail; a cheap pairwise fix recovers it" → the recovery is a jointly trained *mixing* step that writes context into per-token vectors (one attention layer, a bidirectional scan, or a single inducing point) plus a pairwise router; the paper's routing is supervised in every experiment, so its positive results are weaker for us and its negative results stronger.
4. "Context-Gated Associative Retrieval: the gate is a function of the retrieval result, as a fixed point" → true, but the context is clamped in every theorem and experiment, nothing is trained, and the fixed point is within one retrieval episode; its value to us is the collapse theory and the margin theorem, not a mechanism.
5. "Raven: per-token learned top-k router … Gumbel noise is its only anti-collapse device" → correct, with the addition that routing is write-only (dense read) and the headline 400M configuration matches Top-128 of 256, not the Top-32 "default".
6. "Flesch et al. inject a task cue once and let it decay" → the cue is present on every trial; the EMA blurs consecutive cues and is applied to the cue channel only, as preprocessing; the paper never tests a missing cue.

---

## 6. Loose threads worth pulling (ranked)

Papers we do not have, surfaced by the readers (IDs as given by the citing paper unless marked †, from memory — verify before citing):
1. **Podlaski, Agnes & Vogels, "High capacity and dynamic accessibility in associative memory networks with context-dependent neuronal and synaptic gating", Phys. Rev. X 15, 011057 (2025)** — "context-modular networks that partition binary memories across contexts" with capacity claims; cited by the Hopfield paper as the partition-by-context precedent. The one paper that addresses conflicting bindings with gating directly. Highest priority. (bioRxiv 10.1101/2020.01.08.898528.)
2. **Zhang, Nolte, Sadhukhan, Chen & Bottou, "Memory Mosaics" (ICLR 2025, †2405.06394)** — networks of many associative-memory units with kernel smoothing; the closest published relative of a multi-memory question. Cited by TTR only in passing.
3. **Bietti et al., "Birth of a Transformer: A Memory Viewpoint" (†2306.00802)** — induction heads as associative memories built across two layers: the mechanism by which "layer 1 copies the header" would arise.
4. **Zhong, Xu, Ao & Shi, "Understanding transformer from the perspective of associative memory" (2505.19488)** — the two-sided linear update Raven credits; a candidate source for "linear rules on S".
5. **Heald, Lengyel & Wolpert, "Contextual inference underlies the learning of sensorimotor repertoires", Nature 600 (2021)** — the COIN model: Bayesian inference of a latent context from a stream, with new memories created when a new context is inferred. The principled version of Active Dendrites' Algorithm 1 and directly about inferring context over time.
6. **Russin et al., "A Neural Network Model of Continual Learning with Cognitive Control" (2202.04773)** — Flesch's closest sibling (sluggish task signals + gating).
7. **Kimi Linear (2510.26692) Table 7 and Raven Table 10** — the catalogue of test-time-training objectives ↔ update rules, for the BDH-CQ menu.
8. **Kádár et al., "Revisiting the Hierarchical Multiscale LSTM" (†1807.03595); Shen et al., ON-LSTM (†1810.09536); Campos et al., Skip RNN (†1708.06834)** — how much to expect from label-free boundaries; the soft and the budget-regularised alternatives to the STE latch.
9. **Xu et al., "KV Shifting Attention" (2411.19574); Arora et al., Based (2402.18668) and Zoology (2312.04927); Su Jianlin, "Why add short conv to linear attention?" (kexue.fm/archives/11320)** — what a short convolution buys a linear-attention layer; our window gate is this idea applied to the routing variable.
10. **Trockman et al., "Mimetic Initialization Helps State Space Models Learn to Recall" (2410.11135)** — initialisation for recall, relevant to our recipe problems.
11. **Grazzi et al. (†2411.12537) and DeltaProduct (2502.10297)** — β ∈ (0, 2) negative eigenvalues for state tracking, if the memory ever needs to track parity-like state.
12. **Masse, Grant & Freedman 2018 (XdG, PNAS)** — the hard-coded oracle gate; gating alone insufficient as tasks grow, synaptic stabilisation needed on top (relevant to k ≫ S).

Inside the twelve papers, the threads the notes flag as unresolved: MoM's shared-memory combination rule (never written); Raven's actual K at 400M and its undiscussed NIAH-3 regression; GDN's chunkwise output mask (the extraction shows ⊙ M where the derivation needs Γ — check the FLA kernel before transcribing); Nested Learning's momentum term (ablated but never written) and CMS chunk sizes (never given); TRM's "7M parameters" (the per-puzzle embedding table must be excluded); the routing paper's causal/bidirectional labels swapped between its §4.6 table and Table 2; Flesch's α convention flipped between main text and supplement, and the fitted human α never reported; the Hopfield paper's Eq. (1) — a context-*inference* equation, τ_c ċ = W_{h1}ᵀŝ − c, that is written down and never used (precisely the direction we need).

---

## 7. Proposed screens (for batch 19; nothing here changes batch 18 or `test_window_gate`)

All EXPLORATORY, on E, seeds disjoint from the main line, the usual labels. Each screen's reading is fixed before any run.

- **S53 delta channels, controls.** Grouped and header layouts at P = 4, S = 2: single channel Hebbian vs single channel delta (β fixed 1, L2 keys) vs oracle-gate delta. Reading: "delta makes the single channel fail" if single-delta BOUND ≤ 1/10 on both layouts where single-Hebbian ≥ 3/10; "delta raises the oracle" if oracle-delta at P = 8 header binds within 24,000 where Hebbian oracle did not.
- **S54 window gate on delta channels.** LOCAL3 + SLOW + KEYMASS on delta channels, grouped layout, S ∈ {2, 4, 8}, k = 16: not worse than the Hebbian version by more than 2 discordant pairs (a bound). Then the header layout with random block lengths (P_b ∈ {2..6}): the baseline that §4.2–4.3 must beat.
- **S55 the probe.** On S54's header runs, logistic-regression probes for stream identity on r^{(2)} and r^{(3)} at K/V positions. Reading: "the cue is present at layer ℓ" if held-out probe accuracy ≥ 0.9.
- **S56 delta register + layer-2 residual gate**, with the 5% nudge (validity) and without (the test), on the randomised header layout, delta channels, oracle budget. Reading: "the register carries the cue" if nudged ≥ 9/10 BOUND ROUTED; "discovered" if un-nudged beats S54's header baseline by ≥ 3 with no key splits.
- **S57 HM-RNN latch** as the alternative register (STE, a: 1 → 5), same arms as S56; run only if S56 un-nudged fails for leak reasons (β^R settles mid-range).
- **S58 anti-collapse.** Grouped S = 4, k = 16 under the window gate: Gumbel(0,1) write-gate noise vs plateau split vs Switch loss 1e-3 vs reset-only. Outcome BOUND ROUTED and time to transition.
- **S59 horizon.** Routed decay exp(a_t g_t[c]) with ∆ ≈ −2.6 init vs fixed 0.95 vs GDN per-channel α, on the header layout at P = 8 with the oracle gate first (does the gap between oracle and single channel widen?), then under S56's gate.

Main line, when S53–S56 point one way: a pre-registered `test_delta_channels` (the §4.1 swap under the current recipe, both machines, disjoint seeds) is the natural successor to `test_window_gate`, and `test_distant_cue` the one after it.

---

## Sources

Papers read (arXiv): 2501.12352 (test-time regression), 2412.06464 (Gated DeltaNet), 2510.04871 (TRM), 2502.05171 (recurrent depth), 2203.11560 (Flesch et al.), 2201.00042 (Active Dendrites), 2605.10970 (Context-Gated Associative Retrieval), 2502.13685 (MoM), 2607.25357 (Raven), 2603.20997 (content-based routing), 1609.01704 (HM-RNN), 2512.24695 (Nested Learning). Also consulted: 2608.09888 (BDH-CQ, §3.2–3.3 quoted), 2509.26507 (BDH, eq. 7–8 and §6.1), and Pathway's Sudoku post (pathway.com/research/beyond-transformers-sudoku-bench). Official code consulted by the readers for parameterisations the papers omit: NVlabs/GatedDeltaNet (`lit_gpt/gated_delta_net.py`), summerfieldlab/Flesch_Nagy_etal_HebbCL.
