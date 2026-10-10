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
