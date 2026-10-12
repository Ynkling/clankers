# UNSTATED.md — details the report does not determine, and the choice made

Independent replication, EXPLORATORY, not a result.

Sources: `multichannel_hebbian_report_v8.pdf` (Revision 8 draft) and Pathway's public
`bdh.py` / `train.py` (github.com/pathwaycom/bdh, cloned 2026-10-10). Every entry was written
**before** any Phase 2 run. "Report" quotes are from the PDF. The one exception is the oracle
pilot (U-40), which was run before this file was committed.

Legend: **[A]** the report is silent; **[B]** the report is ambiguous (two or more readings);
**[C]** the report states it for another component, and I carried it over.

## Model

| # | What | Choice | Why |
|---|------|--------|-----|
| U-1 | Base architecture [A] | Pathway `bdh.py` forward: `x=LN(emb)`; per layer `xs=relu(x@enc)`, `yKV=LN(attn(xs,xs,x))`, `ys=relu(yKV@enc_v)`, `y=LN((xs*ys)@dec)`, `x=LN(x+y)`; one shared layer applied 3 times; `logits=x@lm_head`. LayerNorm without affine | The report says it extends "the public implementation, bdh.py" with Q=K, shared weights and LN after attention, which is exactly this code. |
| U-2 | Positional operator [B] | Scalar decay only: score × 0.95^(t−s); **no RoPE** | The report says "(scalar decay or RoPE)" and gives "decay 0.95, half-life about 14 tokens". I read that as the decay replacing RoPE. |
| U-3 | Strict causality [A in code, stated in report] | Mask s<t (diagonal excluded) | Eq. (1) says "strictly causal (s < t)". Pathway's `tril(diagonal=-1)` agrees. |
| U-4 | N, D, heads [stated] | N=256, D=32, one head. In Pathway's terms `mlp_internal_dim_multiplier=8` | Section 3. |
| U-5 | Dropout [A] | 0 (Pathway's default is 0.1) | Not mentioned. On a synthetic binding task dropout would only add noise. |
| U-6 | Weight init [A] | Pathway's: encoder, encoder_v, decoder, lm_head and embedding all N(0, 0.02²) | Inherit the base repo. **The embedding scale matters here.** The gate reads the *raw* (not LayerNormed) embedding, so std 0.02 makes the gate's pre-activations tiny at init (Adam's scale invariance partly compensates). |
| U-7 | Channel mixing [B] | One gate g_t for both read and write; score = (x_t·x_s)(g_t·g_s)d^(t−s); i.e. M = (G Gᵀ) ⊙ decay applied to the content scores | Eq. (1): "With one distribution for both". |
| U-8 | Where the gate applies [stated] | One g_t per token, the same at every layer | Section 1. |
| U-9 | Gate input [stated] | The model's own raw embedding `embed(idx)`, shared with the BDH path (not a separate table) | "the raw token embedding v_t (not LayerNormed …)". I read it as the same table. |
| U-10 | Window gate hidden width [C] | 32 units, as stated for the recurrent gate | The report gives 32 only for the recurrent gate. |
| U-11 | Window gate W_in, W_g init [C] | N(0, 0.1²), as stated for the recurrent gate | Same. |
| U-12 | Window gate biases [A] | None, in the conv, W_in or W_g | Eqs. (2)–(3) show none. The report says "no bias" for the conv. |
| U-13 | Window gate conv [stated + A] | `nn.Conv1d(D, D, 3, groups=D, bias=False)` with PyTorch's default init (= U(±1/√3)), left-padded with zeros | "weights drawn uniformly in ±1/√3" is PyTorch's default for this layer. Zero padding: the first tokens see zeros for v_{t−1}, v_{t−2}. |
| U-14 | "conv" placement [B] | One depthwise causal conv, width 4, over the **D-dimensional** layer input x, identity init (last tap 1). Its output feeds both the query/key path (`relu(conv(x)@enc)`) and the value (`V=conv(x)`). The residual uses unconvolved x. No bias | "feeding the queries and the values". In BDH Q=K is N-dim and V is D-dim; convolving x before both is the single filter that feeds both. |
| U-15 | "conv" across layers [B] | **Shared**: one conv module reused at all 3 layer applications | "at every layer" is ambiguous with a weight-shared stack. I followed BDH's weight sharing. |
| U-16 | Dtype/device [A] | float32, CPU, 1 thread | As instructed. |

## Task

| # | What | Choice | Why |
|---|------|--------|-----|
| U-17 | Vocabulary [A] | 16 value tokens, P key tokens, S context tokens, one `?` token (vocab 29 at S=8, P=4) | Minimal. The stated maximum sequence length of 99 = 3·8·4+3 implies the `?` token is in the sequence. |
| U-18 | Value draw [B] | Per sequence, per key: S distinct values without replacement from 16, assigned to streams in random order | "For each of P keys, S distinct values are drawn from 16 value tokens". |
| U-19 | Query draw [A] | Query stream and key are each uniform, independent | Not stated. |
| U-20 | Training loss [A] | Cross-entropy over the full vocabulary at the query's KEY position only (one target per sequence) | "the model predicts the value from the query's KEY position". §16 mentions "an end-of-sequence loss". |
| U-21 | Accuracy [A] | argmax over the **full vocabulary** at the query KEY position equals the target | Not stated. Restricting to the 16 values would be slightly more lenient. |
| U-22 | Held-out evaluation set [A] | 2048 sequences from a fixed generator (seed 123457), the same for every run and arm; one query each | "2048 held-out queries". Training draws come from a separate generator. Overlap is possible but negligible. |
| U-23 | Data / init seeding [A] | `torch.manual_seed(seed)` before init; training batches from `Generator(seed)`; probe from `Generator(seed+1000003)`; split noise from `Generator(seed+2000003)`. The arms are bit-identical up to the first split (unit-tested) | Pairing by seed. |

## Training and evaluation

| # | What | Choice | Why |
|---|------|--------|-----|
| U-24 | Optimizer [B] | `torch.optim.Adam`, default betas (0.9, 0.999), eps 1e-8, **no weight decay** (Pathway's train.py uses AdamW wd 0.1) | The report says "one Adam optimizer with two groups". |
| U-25 | SLOW group membership [stated + A] | Gate group = window-gate conv, W_in, W_g. Everything else, including the embedding the gate reads and the width-4 conv, is the memory group | "embedding and convolution included". |
| U-26 | SLOW boundary [B] | Memory lr 1e-4 for updates 1..2400 inclusive, 1e-3 from update 2401. Step change, no warmup | "10⁻⁴ for updates 1–2400 and 10⁻³ after". |
| U-27 | LR schedule otherwise [A] | Constant, no warmup or decay, no gradient clipping | Not stated. |
| U-28 | Oracle's training [A] | Same model (k=16, conv) and the SLOW schedule for its (non-gate) parameters, same budget and evaluation | The validity arm should differ from the recipe only in the gate. |
| U-29 | Evaluation timing [B] | After updates 1200, 2400, … and at the budget (43,200 = 36×1200). Early stop after three consecutive evaluations ≥ 0.95. ≥ means ≥ | Section 3. |
| U-30 | Order at a shared step [A] | Optimizer step → evaluation (and early-stop check) → split check (probe) | Not stated. If the run stops early, no split is considered at that step. |
| U-31 | Final per-stream accuracy and map [A] | Taken from a final evaluation of the final model (after early stop or at the budget) on the same 2048 held-out sequences | Not stated. |

## Outcomes

| # | What | Choice | Why |
|---|------|--------|-----|
| U-32 | Stream-to-channel map [B] | For each stream s: argmax over channels of the **mean gate vector** over all value positions of stream s in the 2048 evaluation sequences. One-to-one = 8 distinct channels | "a one-to-one stream-to-channel map, measured at value positions". The alternative is a majority vote of per-token argmaxes. |
| U-33 | Value positions [A] | The 32 VAL positions of the body (the query has no VAL; its `?` is excluded) | — |
| U-34 | Per-stream accuracy [A] | Accuracy over evaluation queries whose query stream is s (~256 each) | "every stream's accuracy is at least 0.9". |
| U-35 | Failure labels [A] | Diagnostics only (not used in the reading): collapsed if final acc < 0.15; merged(j share) if the map is not one-to-one; else unbound-routed / stream-acc<0.9. "non-stream" not implemented | — |

## SPLIT

| # | What | Choice | Why |
|---|------|--------|-----|
| U-36 | Probe [A] | 64 sequences from the task distribution, drawn once per run from a seed-derived generator (not the eval set). Accuracy on its 64 queries as above | "a fixed 64-sequence probe". |
| U-37 | First check's "previous check" [B] | The probe is also read at update 2400 (no split possible there), so the check at 4800 has a reference | The report says a split *can* fire at 4800 ("its split at 4800 included"), so the first check needs a previous value. |
| U-38 | Key positions for "read-gate mass" [B] | All KEY positions in the probe (32 body keys + the query key per sequence). Mass = mean of g[c] over them. c* = argmax, c0 = argmin (ties → lowest index) | "mean read-gate mass at key positions". |
| U-39 | Copy, noise, reset [B] | Row std σ = std of W_g[c*] before the copy (unbiased). W_g[c0] ← W_g[c*] + 0.1σ·ε₁; W_g[c*] ← W_g[c*] + 0.1σ·ε₂, with independent Gaussian ε. Zero W_g's Adam `exp_avg`, `exp_avg_sq` **and `step`** (bias correction restarts). Other parameters' Adam state untouched | "add noise of 0.1 times the row's standard deviation to both, and zero W_g's Adam state". W_g has no bias (U-12), so there is nothing else to copy. |
| U-40 | Spacing and cap [stated] | A split at u blocks splits until u+4800; at most 3. "Rose by less than 0.02 since the previous check" compares with the probe read 2400 updates earlier, including after a split | Section 3. |
| U-41 | Guard "no fire when the target carries key mass" [A] | **Not implemented** | §6 says "should", not that the recipe has it. k=16 > S=8 leaves spare channels. |

## Process

| # | What | Choice | Why |
|---|------|--------|-----|
| U-42 | Oracle pilot (disclosed) | Before pre-registering, one ORACLE run on seed 1 (outside 900–919, budget 12,000), to catch bugs. No learned-gate run was piloted | Validity check only; it cannot inform the gate comparison. |
| U-43 | Checkpoint/resume | Each run checkpoints after every evaluation and resumes bit-for-bit (unit-tested) after a container restart | Long runs on an ephemeral container. |
| U-44 | Statistics | Exact one-sided McNemar p = P[Bin(b+c, ½) ≥ b]; Wilson 95% (z = 1.96) | As instructed. |

---

# Phase 5 — details the spec does not determine, and the choice made

Independent replication, EXPLORATORY, not a result.

Sources: `specs/test_split_copy.md` and the docstring of `test_split_copy.py` at `ffb0771`, plus the "Phase 4, amended" section of `docs/reading/parallel_sessions_prompts.md`. No code below that docstring was read, and nothing from `explore_delta_mem.py`, `explore_f_delta_fast.py` or Session F's branch. Every entry was written **before** any Phase 5 run.

Legend as in Phase 2: **[A]** silent; **[B]** ambiguous; **[C]** carried over from another component; **[S]** stated in the spec, listed for completeness.

## The reset-only arm

| # | What | Choice | Why |
|---|------|--------|-----|
| U-50 | What RESET zeroes [B] | **W_g's Adam state only** (`exp_avg`, `exp_avg_sq`, `step`). W_in, the window and the rest are untouched, and W_g itself is untouched | The spec says "W_g's Adam state zeroed, W_g untouched". The user's prompt says "the gate group's Adam state". I followed the spec, so that RESET differs from COPY only by the copy, since COPY zeroes W_g's state only. |
| U-51 | RESET's trigger [S] | The copy arm's rule exactly: the same probe (`Generator(12345)`), body keys, checks, threshold, rise, cap and gap. c*/c0 are computed and recorded but not used | "the same trigger". |

## The delta-rule channel

| # | What | Choice | Why |
|---|------|--------|-----|
| U-52 | Key and query [B] | Key k_t = L2-normalised x_sparse_t (the Q=K activation at that layer). The query stays raw x_sparse_t | "L2-normalised keys". The CHECK "at β = 0 (erase off, write 1, **raw keys**) gives the Hebbian logits" implies only the keys change, and that the Hebbian query is the raw activation. |
| U-53 | Where the decay sits [A] | S_t = d·S_{t−1}(I − β k_t k_tᵀ) + w·v_t k_tᵀ; read y_t = d·S_{t−1} q_t (read before write) | This is the form whose β = 0, w = 1, raw-key case is exactly the Hebbian score d^{t−s}(x_t·x_s), which the spec's CHECK 149 requires. It is also the report's Session F equation, with α = d. |
| U-54 | "Tied write" [B] | Write strength w = β = 1 | At the fixed β = 1 both readings give w = 1. |
| U-55 | Values, LN, layers [C] | v = the conv'd layer input (BDH's V). LayerNorm after the read as in BDH. The same delta memory at each of the 3 applications of the shared layer | The spec: "no parameter added"; everything else as the Hebbian model. |
| U-56 | Computation [A] | Exact parallel form: pseudo-values U from (I + β (K Kᵀ ⊙ M)) U = w V (unit lower-triangular solve), y = (Q Kᵀ ⊙ M) U | Equal to the step-by-step recurrence to 1e-10 in float64 (unit-tested). The main line used a fused form "equal to the parallel form to 1e-5 relative". |
| U-57 | Normalisation eps [A] | `F.normalize` default (eps 1e-12). An all-zero activation gives a zero key: nothing is erased or written there, and nothing is read from it | — |
| U-58 | DELTA1's width [A] | N = 256, D = 32, 3 layers, conv, k = 1, no gate | The spec names "test_short_conv's B_conv" without its width. I took the same model as every other arm, so that only the memory differs. |
| U-59 | Soft gates [A] | The delta memory is implemented only for k = 1 and one-hot gates (asserted) | Only those are run. The parallel form with G Gᵀ is exact only for them. |

## Recipes, oracles, outcomes

| # | What | Choice | Why |
|---|------|--------|-----|
| U-60 | DELTA1 and ORACLE_DELTA recipe [S] | Single Adam group at 1e-3 on every parameter throughout (no SLOW) | Stated (F's S68 recipe). |
| U-61 | Validity oracles [S/A] | ORACLE_ADAM (Hebbian perfect gate, k = 16, single-rate Adam) and ORACLE_DELTA (perfect gate on delta channels, k = 16), each on 940–941 | The spec: CEIL_C and CEIL_DELTA_D8 on the first two seeds. The k = 16 with 8 unused channels is as the spec says. |
| U-62 | DELTA1's outcome [S] | BOUND (a transition held to the end). Per-stream accuracy is recorded; there is no routing | Stated. |
| U-63 | COPY's last check [A] | As in Phases 2–4: checks may run through 43,200 (the main line stops at 40,800, DIFF.md D-5) | The copy arm is "my Phase 4 recipe, kept". It was inert in all 60 earlier split runs, whose latest split was at 36,000. |
| U-64 | Arms not run [A] | WIN3_SLOW_D8 (needed only for C2) and SINGLE_HEBB_D8 are not run on 940–959 | The user asked for the first and third results only. |
| U-65 | Job order [A] | Oracles first, then per seed DELTA1, RESET, COPY (longest first), 4 workers × 1 thread | Runtime. |

## Reading

| # | What | Choice | Why |
|---|------|--------|-----|
| U-66 | The rule [B] | The user's statement: per arm, "consistent" if x/20 lies in the main line's pooled Wilson interval for that arm, endpoints included; otherwise "outside the main line's interval". No "replicates" label | The "Phase 4, amended" text in the file states a different rule (fresh count ≤ 27/32 reads "below 0.95"). That rule is for the rate test, not for this three-arm comparison. I applied the user's statement and recorded the difference in phase5.py. |
| U-67 | The intervals [A] | From the main line's pooled bands (docstring): COPY [0.699, 0.972], RESET [0.145, 0.519], DELTA1 [0, 0.161] | Copied as printed, not recomputed (they agree with Wilson z = 1.96 for 18/20, 6/20, 0/20). |
| U-68 | Edge case [A] | A 20/20 copy count reads "outside" (1.0 > 0.972) | The rule is applied as written; the case is named beforehand. |
| U-69 | Discordant counts [A] | COPY vs RESET (BOUND ROUTED both) and COPY (BOUND ROUTED) vs DELTA1 (BOUND), over the 20 pairs, with the exact one-sided McNemar p labelled descriptive | "the discordant counts printed". |
| U-70 | Pilot runs [A] | None. Before the protocol commit, the only Phase 5 executions were unit tests (≤ 60 updates on toy sizes), a 2400-update regression of the recorded Phase 4 run WIN3_SPLIT_D8_ML\|920 (reproduced exactly), and a 30-update timing of the delta model | Disclosure. |
