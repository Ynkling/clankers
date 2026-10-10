# Multi-Channel Hebbian Plasticity in Multilayer BDH

**Solves context-conditional binding by partitioning memory.** A window gate, slow memory and a targeted channel split find an eight-way partition without labels or restarts, on two machines with disjoint seeds.

This repository is a research fork of Pathway's [BDH (Dragon Hatchling)](https://github.com/pathwaycom/bdh). It extends BDH's Hebbian working memory with several memory channels and a gate that spreads each token's write over the channels and blends its read from them. The research sections below follow Revision 8.1 (draft) of the project's technical report: the PDF is [`multichannel_hebbian_report_v8.pdf`](multichannel_hebbian_report_v8.pdf), with its LaTeX source alongside. Revision 7's account of Phase V is kept in [`docs/revision7.md`](docs/revision7.md) and Revision 6's account of Phases I–IV in [`docs/revision6.md`](docs/revision6.md). The October 2026 audit of the project's records is [`docs/audit_2026-10.md`](docs/audit_2026-10.md); the twelve-paper review of context routing and memory is in [`docs/reading/`](docs/reading/); an independent referee's report on Revision 8, with its recomputation of every number from the records, is in [`docs/review_2026-10/`](docs/review_2026-10/), and Revision 8.1 corrects the statements it found wrong. The original BDH README from Pathway follows the research part, [unchanged](#about-bdh-upstream-readme-from-pathway).

---

## Contents

- [Summary](#summary)
- [Status, machines and the Phase VI tests](#status-machines-and-the-phase-vi-tests)
- [1. Background](#1-background)
- [2. Earlier phases in brief](#2-earlier-phases-in-brief)
- [3. The task, the outcomes and the recipes](#3-the-task-the-outcomes-and-the-recipes)
- [4. The window gate](#4-the-window-gate)
- [5. The main result: `test_window_gate`](#5-the-main-result-test_window_gate)
- [6. The split: the copy, not the reset](#6-the-split-the-copy-not-the-reset)
- [7. Muon is not an ingredient](#7-muon-is-not-an-ingredient)
- [8. The recurrent gate's eight-stream failure, closed](#8-the-recurrent-gates-eight-stream-failure-closed)
- [9. Two-stream key splits](#9-two-stream-key-splits)
- [10. Scale](#10-scale)
- [11. Distant cues and the memory rule](#11-distant-cues-and-the-memory-rule)
- [12. Where the mechanism stands](#12-where-the-mechanism-stands)
- [13. Retrospective](#13-retrospective)
- [14. Methodological findings](#14-methodological-findings)
- [15. Limitations](#15-limitations)
- [16. Distance to a language model](#16-distance-to-a-language-model)
- [17. Conclusion and next steps](#17-conclusion-and-next-steps)
- [Repository layout and how to run the tests](#repository-layout-and-how-to-run-the-tests)
- [Provenance](#provenance)
- [References](#references)
- [About BDH (upstream README from Pathway)](#about-bdh-upstream-readme-from-pathway)

---

## Summary

Revision 7 left eight streams unsolved: no learned-gate run trained on all eight from the start had bound, and screens traced the failure to the gate's recurrent state, which stops carrying the context before any routing forms. This revision replaces the gate.

**A window gate.** A width-3 causal convolution of the token embeddings with no recurrence was screened in the first exploratory batch, led the recurrent gate there (15/20 against 7/20; the screen's verdict was "promising"), and was dropped because it cannot see a distant cue. An audit of the project's records brought it back. Combined with slow memory for the first 2400 updates and a plateau-triggered *channel split* that copies the busiest channel's gate row onto the idlest, it binds eight streams with the streams on disjoint sets of channels, one per stream by largest gate mass, in **38/40** runs of a pre-registered test:
- against 0/40 for the recurrent-gate recipe (38 vs 0 discordant pairs, p = 3.6×10⁻¹²);
- against 15/40 for the window gate without the split (23 vs 0, p = 1.2×10⁻⁷);
- the two machines ran disjoint seeds, so the pooled test is a test;
- the recipe uses no stream labels, no restarts and no hinge (the split reads the task's key positions); its reading, fixed before the run, was "eight streams bind without labels or restarts".

At four streams with sixteen channels the window gate binds 38/40 against 27/40 for the recurrent gate (13 vs 2, p = 0.004); at two streams it is within the pre-registered bound of the previous recipe, and better by two pairs.

**The split's mechanism.** Three exploratory looks at eight streams, on 30 seeds (one of them the test's own printed control), agree that the split's row copy is its active part and that the optimizer reset alone is not enough; whether the reset adds anything to the copy is shown by none of them, and in a later ten-seed screen (session H's S61) the copy alone bound as often as the full split. Two anti-collapse devices borrowed from the literature did not help: Gumbel noise on both gates collapses routing and on the write gate alone binds 0/10 at eight streams against the split's 10/10, and a load-balancing loss changed the shape of the routing at four streams without changing its count.

**Scale, Muon.** At four streams the recipe's constants survive a fourfold change of width and a doubling of depth without retuning, on ten seeds per size. Muon is not an ingredient: the recipe binds eight streams at 9/10 under either optimizer, and a pooled test of Muon's four-stream speed, shown on one machine in Revision 7, was not shown.

**Distant cues and the memory rule.** None of twelve papers reviewed carries an inferred context cue across distance without labels under a sparse end-of-sequence loss; our own distant-cue task remains unsolved. Screens found that a single memory channel with a delta-rule write binds the two-stream header layout by routes that need no partition (post-hoc diagnostics: context-tagged keys at a fixed decay; a learned forget gate that partly clears at the context token), so the partition's claim is about the number of contexts: at eight streams the gated recipe binds 38/40 and one delta channel 0/10 (at four streams 2/10 against 19/20).

**Corrections.** Among them: the gate does not "need only one token of memory" (one at the read, two at the write), and the claim that Muon "binds four streams sooner" is not shown on disjoint seeds. Revision 8.1 corrects statements of the first draft that the referee found wrong: the premise test had been run; the Muon correction had the machines the wrong way round; the convolution argument compared two models; the reset's contribution is not shown by the looks cited; and several provenance figures (see "Revision 8.1 (corrections)" below; the referee's check of the first pass is `docs/review_2026-10/revision_8_1_check.md`).

## Status, machines and the Phase VI tests

**Revision 8 (draft).** This revision adds Phase VI: two pre-registered tests run on two machines with disjoint seeds, an audit of the project's records (`docs/audit_2026-10.md`), five exploratory batches on the side branch, three further exploratory sessions run in parallel on their own branches, and a reading of twelve papers on context routing and memory (`docs/reading/`). Two machines ran every main-line test: **X**, a cloud container on Intel Xeon hosts at 2.10 and 2.80 GHz, and **L**, an Intel i7-12650H. From the Muon-recipe test on, X and L run disjoint seeds, and the pooled claim is primary. The exploratory sessions are **E** (branch `claude/outside-ideas`, batches 1–19), and **F**, **G** and **H** (branches `claude/explore-F/G/H`); F and H ran on 2.10 GHz Xeons; E's batches ran mostly on 2.10 GHz Xeons; 60 of batch 16's 165 runs and batch 19 ran on 2.80 GHz hosts, and batch 15's store names a 2.80 GHz host without recording the CPU per run. Every verdict was fixed before its run and printed mechanically. The revision is a draft: E's batch 19 is running; session F's premise test (S68) and lite task (S70), session G's first screen (S55) and session H's S61 are cited below, while F's S69, G's S56–S57, H's S62–S63 and the follow-ups are on their branches and will be in Revision 9; and everything remains one small model on a synthetic task.

**Revision 8.1 (corrections).** A referee session (V; its report, recomputation scripts and git audit are in `docs/review_2026-10/`) recomputed every count, discordant pair, *p*-value, interval and band of Tables 2 and 3 from the records (all reproduce), reproduced one run of each Phase VI test bit for bit on a third CPU model, and found statements that were wrong when the first draft was committed. This revision corrects them, cites three results that were on the branches and not in the first draft (H's S61, G's S55, F's S70), and changes nothing else; a second pass applied the referee's check of the first (`revision_8_1_check.md`). The corrections: the premise test had been run (Sections [11](#11-distant-cues-and-the-memory-rule) and [12](#12-where-the-mechanism-stands)); the Muon correction had the two machines the wrong way round (Sections [7](#7-muon-is-not-an-ingredient) and [13](#13-retrospective)); the convolution argument compared two different models (Section [4](#4-the-window-gate)); the reset's contribution is not shown by the looks cited (Section [6](#6-the-split-the-copy-not-the-reset)); "one channel per stream" is by largest mass (Section [3](#3-the-task-the-outcomes-and-the-recipes)); batch 16's hardware, the single-evaluation count, the specification files, Revision 7's routing on L, the one-channel column and the branch heads were misstated (this section, Sections [3](#3-the-task-the-outcomes-and-the-recipes) and [12](#12-where-the-mechanism-stands), the appendix); two literature sentences were overstated (Section [11](#11-distant-cues-and-the-memory-rule)). The referee's structural requests (scope, intervals for the paired differences, external baselines, the list of claims) are for Revision 9.

**Table 1.** Phase VI. Main-line tests are files `test_<name>.py` whose docstring records the pre-registered design and, after the run, the result on each machine and pooled; `test_window_gate`'s specification was committed to `specs/` before the test was written; the earlier tests fix their designs in the docstring. Exploratory batches and sessions write only new files on their own branches and label every output "exploratory, not a result". Appendix [Provenance](#provenance) lists the commits.

| test or batch                | question                                                                                  | verdict                                                                                    |                                                                                                                                                                        § |
|:-----------------------------|:------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `muon_recipe` (X, L, pooled) | the early hinge window under Muon; Muon's speed at four streams and eight keys            | M1 SHOWN pooled (and on L); M2, M3 NOT SHOWN pooled; reading not met                       |                                                                                            [7](#7-muon-is-not-an-ingredient) |
| `window_gate` (X, L, pooled) | the window gate with slow memory, no hinge, at 2, 4 and 8 streams; the split at 8         | G1, G2, G4 SHOWN pooled; G3 HOLDS; reading "eight streams bind without labels or restarts" |                                                                                            [5](#5-the-main-result-test_window_gate) |
| batch 15 (E)                 | a cap on the recurrent gate's gain                                                        | no arm bound: not                                                                          |                                                                                  [8](#8-the-recurrent-gates-eight-stream-failure-closed) |
| batch 16 (E)                 | the window gate with slow memory at 2, 4, 8 streams; a reservoir gate; eight-stream knobs | 20/20 at four streams; 3/10 at eight, the first ever                                   |                                                                                        [4](#4-the-window-gate) |
| batch 17 (E)                 | the targeted split on the window gate at eight streams                                    | 9/10 vs 3/10 (6 vs 0, *p* = 0.031)                                                     |                                                                                        [4](#4-the-window-gate) |
| batch 18 (E)                 | reset-only control; Muon; the split at *k* = 4; two-stream key splits                     | copy does the work; recipe carries to Muon; key splits unresolved                          | [6](#6-the-split-the-copy-not-the-reset), [9](#9-two-stream-key-splits) |
| batch 19 (E)                 | width and depth                                                                           | scale-free to *N* = 1024, *D* = 128, 4 layers                                              |                                                                                         [10](#10-scale) |
| session H                    | Gumbel noise, balance loss, reset-only at 4 and 8 streams; gate temperature               | nothing replaces the split; the copy, not the reset                                        |                                                                                          [6](#6-the-split-the-copy-not-the-reset) |
| session F                    | delta-rule channels; learned forget gates; the header layout                              | a single delta channel binds the two-stream header layout; worse at eight streams          |                                                                                             [11](#11-distant-cues-and-the-memory-rule) |

## 1. Background

BDH \[1\] stores working memory in synaptic state updated by a Hebbian rule. In its GPU form the operation is linear attention: the score between a query position *t* and a past position *s* is a content match $x_t \cdot x_s$ between sparse positive neuron activations, weighted by a positional operator (scalar decay or RoPE). The public implementation, `bdh.py`, ties queries to keys (Q = K), shares one layer's weights across depth, and LayerNorms the attention output. Its state update, in the paper's notation, is $\rho_{t} = \rho_{t-1} + \mathrm{LN}(E y_{t})\, x_{t}^{\top} U$: a pure additive write.

The extension gives every synapse *k* channels and every token a gate $g_t \in \Delta^{k-1}$ that distributes its write over the channels and blends its read from them. With one distribution for both, the effective score becomes

$$
\underbrace{(x_t \cdot x_s)}_{\text{content match}} \times \underbrace{(g_t \cdot g_s)}_{\text{context-gate match}} \times \mathrm{decay}^{\,t-s} \qquad (1)
$$

strictly causal (*s* < *t*), so tokens sharing a context can read each other's memory while tokens in different contexts are isolated. The score has the three-factor form that Triadic Linear Attention \[3\] describes (cited for its form; the paper is not among the twelve read in full), with g as the second key; there the second key is a learned projection of the token, here it is a routing distribution inferred from context, and the subject is its label-free discovery. With *k* = 1 the model is single-channel BDH. One gate $g_t$ is computed per token from the raw token embedding $v_t$ (not LayerNormed, not convolved, not the residual stream) and applied at every layer. The hand-set *perfect gate* sends every token of stream *s* to channel *s*; it is used only as a validity ceiling.

**Two learned gates.** The *recurrent gate* of Phases II–V,

$$
h_t = \tanh\bigl(W_{\mathrm{in}}\,v_t + W_h\,h_{t-1}\bigr), \qquad g_t = \mathrm{softmax}(W_g\,h_t) \qquad (2)
$$

with 32 units in $h_t$ and $W_{\mathrm{in}}, W_h, W_g \sim \mathcal{N}(0, 0.1^2)$ (so that $\rho(W_h) \approx 0.6$ and $\sigma_{\max}(W_h) \approx 1.08$ at initialization). The *window gate* of this revision,

$$
h_t = \tanh\Bigl(W_{\mathrm{in}} \sum_{j<3} w_j \odot v_{t-j}\Bigr), \qquad g_t = \mathrm{softmax}(W_g\,h_t) \qquad (3)
$$

a depthwise causal convolution of width 3 over the embeddings (weights drawn uniformly in $\pm 1/\sqrt{3}$, no bias), with no recurrence. It sees the current token and the two before it, which in the grouped layout is enough to see each key's stream token at the key, the value and the query. It cannot see anything further back, and it has no notion of position.

**The uniform gate is a stationary point.** Writing each gate as a deviation from uniform, $g_t = \tfrac{1}{k}\mathbf{1} + \delta_t$, gives $g_t \cdot g_s = \tfrac{1}{k} + \delta_t \cdot \delta_s$. The constant is cancelled by the LayerNorm after attention, so routing enters only through products of deviations, and at the uniform gate the task loss exerts no first-order pull toward any routing. This is why discovery is a race between the gate's first commitment and whatever the memory learns first, and why the recipes below act in the first few thousand updates.

**A short causal convolution.** Most arms include the short convolution that Mamba and most linear-attention models place before their token mixer \[4\]: a depthwise causal filter of width 4 at every layer, initialized to the identity, feeding the queries and the values. Without it, the perfect gate of this model (*N* = 256) binds the two-stream configuration on every seed recorded (4/4 in the main line's `CEIL_A`, 10/10 in session F). The first draft attributed four failures to the missing convolution; those runs used a smaller model (`test_multilayer_binding`'s *N* = 64 ceiling, 7,552 parameters against 25,984), as the referee's recount found. Whether the convolution carries part of the binding is open.

## 2. Earlier phases in brief

**Phases I–IV (Revisions 1–6).** A gate that was a function of token identity could not express the routing (Phase I); a recurrent gate could (Phase II). On the binding task of Section [3](#3-the-task-the-outcomes-and-the-recipes), two channels with the perfect gate bind two streams where no single-channel model binds, and a label-free learned gate finds the routing on about half its runs (Phase III). A restart rule made two streams reliable; four streams with four channels mostly failed by merging streams into a shared channel, and spare channels removed most merges (Phase IV).

**Phase V (Revision 7).** An early-training recipe replaced restarts at two streams and at four with sixteen channels: for the first 2400 updates the memory trains at a tenth of the gate's rate (**SLOW**), and a hinge penalizes the gate when its channel choice at key positions is explained by position (**HINGE**; **WINDOW** switches the hinge off after update 2400). Eight streams failed in every arm (none of more than 100 runs), and screens located the failure in the recurrent gate: its state carries the stream at initialization and loses it within the first 600–1200 updates, as its recurrent term grows and, in seven of ten runs, the spectral radius of $W_h$ rises above one. Revision 7 ended with a cap on that recurrence being screened.

## 3. The task, the outcomes and the recipes

**Task.** There are *S* streams, each with a context token $\mathrm{CTX}_s$. For each of *P* keys, *S* distinct values are drawn from 16 value tokens, so every key is bound to a different value in every stream. In the *grouped* layout the body is *S* *P* triples $[\mathrm{CTX}_s, \mathrm{KEY}_i, \mathrm{VAL}_{i,s}]$, grouped by key, keys in random order and stream order random within each key's group. One query $[\mathrm{CTX}_q, \mathrm{KEY}_q, ?]$ ends the sequence; the model predicts the value from the query's KEY position, where $\mathrm{CTX}_q$ is one token back. The write that must carry the stream is at the VAL position, where CTX is two tokens back. Chance among values is 1/16; a model that binds keys to values but cannot tell streams apart scores 1/S. The *header* layout, used in screens, gives each stream one block $[\mathrm{CTX}_s, \mathrm{KEY}, \mathrm{VAL}, \mathrm{KEY}, \mathrm{VAL}, \ldots]$ with the stream token once at the start, so most writes are far from their cue. The model has three layers, decay 0.95 (half-life about 14 tokens), *N* = 256, embedding width 32 and one head, and trains at batch 32 for a budget set per test (24,000, 28,800 or 43,200 updates), with evaluation every 1200 updates on 2048 held-out queries; a run stops early after three consecutive evaluations at or above 0.95.

**Outcomes.**

- **Bound:** an evaluation reaches 0.95 and every later one stays there; the *transition* is the first such step. A run that reaches its budget counts as bound on a single final evaluation; this has happened in 14 of 1869 bound runs in `results/` at this revision's commit (the audit's 11 reproduces on X's files, its 1028 does not; the recount is the referee's, `docs/review_2026-10/`).

- **Discovered** (*k* = *S* = 2): bound, with the streams' read gates separated at value positions.

- **Bound routed** (*k* ≥ *S* > 2): bound, every stream has its own channel (a one-to-one *stream-to-channel map*, measured at value positions, where the writes are), and every stream's accuracy is at least 0.9. The map takes each stream's largest-mass channel; in the 38 headline runs the streams' channel sets are disjoint (largest pairwise overlap 0.047) while in 18 of them some stream's largest channel holds under 0.9 of its gate mass (the referee's recount).

- **Failure classes:** POSITION or KEY when the gate's channel choice at key positions is explained (η² ≥ 0.5) by position or by key identity; STREAM-PARTIAL; OTHER. For *S* > 2: MERGED (j share) when j streams share a channel, non-stream when the gate does not split by stream, *collapsed* when final accuracy is below 0.15.

**Recipes.**

- **SLOW:** one Adam optimizer with two groups. The gate's parameters train at 10⁻³ throughout; every other parameter, embedding and convolution included, at 10⁻⁴ for updates 1–2400 and 10⁻³ after.

- **HINGE, WINDOW:** Revision 7's position hinge, on the recurrent gate; kept here only as the comparison.

- **SPLIT (KEYMASS):** at checks every 2400 updates from 4800, on a fixed 64-sequence probe, if the probe's accuracy is below 0.95 and rose by less than 0.02 since the previous check, copy the gate row $W_g[c^*]$ of the channel with the largest mean read-gate mass at key positions onto the row $W_g[c_0]$ of the channel with the smallest, add noise of 0.1 times the row's standard deviation to both, and zero $W_g$'s Adam state; at most three splits per run, at least 4800 updates apart. It uses no stream labels; it needs to know where the key positions are.

- **MUON:** Muon at learning rate 0.005 on every 2-D weight (momentum 0.95, Nesterov), with Adam on the rest, and SLOW's schedule.

**Statistics.** Paired comparisons in the main-line tests use exact one-sided McNemar tests on the discordant pairs ("b vs c"), unpaired ones Fisher's exact test; E's exploratory batches report two-sided McNemar *p*-values and session H one-sided ones, as marked where quoted; bands are RELIABLE at ≥ 90% and MAJORITY at ≥ 50% of runs, with Wilson 95% intervals. From the Muon-recipe test on, X and L run disjoint seeds (X 400–559, L 1400–1559), the pooled claim over both machines' pairs is primary and each machine's is secondary. About sixty pre-registered one-sided claims have been made per machine at $\alpha = 0.05$ over the project, so two or three false positives are expected under the global null; the conclusions that matter below are those replicated on both machines or pooled over disjoint seeds.

## 4. The window gate

**History.** The window gate was the first screen of the first exploratory batch (S1, batch 1): on seeds 160–179 with plain Adam and no recipe it discovered 15/20 against the recurrent gate's 7/20 (12 vs 4, two-sided *p* = 0.077: "promising" by the screen's rule) and bound sooner on average (mean transition 2720 against 3429; equal medians); its five failures were three key splits and two stream-partial runs under the main-line classifier, four key splits under the key-first rule (it cannot position-split, having no position). Batch 2 ran it on the header layout, where it bound 2/10, both by key splits, and it was dropped: a gate that cannot see a cue three tokens back seemed the wrong direction for a language model. It was never combined with slow memory and never run at four or eight streams. The October audit of the project's records (`docs/audit_2026-10.md`, Section 3.1) noticed that every obstacle of Phase V was a property of the recurrent parameterization: position splits are the recurrence counting triples, and the eight-stream loss of the stream is the recurrence becoming expansive. A window gate can do neither. The audit fixed a prediction before any run: with slow memory, at two streams ≥ 36/40 with key failures gone; at eight streams, binding on some seeds where every recurrent arm bound none.

**Batch 16 (E).** The window gate with SLOW and no hinge (`LOCAL3_SLOW`): at two streams 31/40 on seeds 160–199 (STREAM-PARTIAL 5, KEY 4, no POSITION), inconclusive against the hinge recipe's 34/40 (5 vs 8). At four streams with sixteen channels, 20/20 bound routed against X's recorded `HINGE4k16` 15/20 (5 vs 0, two-sided *p* = 0.0625). At eight streams, 3/10 under Adam and 3/10 under Muon, where every recurrent arm in batches 13–16 had bound 0/10; the seven unbound Adam runs ended with two or three streams sharing a channel at accuracy 0.73–0.88. The window gate's state decodes the stream at both key and value positions (1.00 at four streams at update 4800, the two runs still training at 9600 reading 0.70–0.72; at eight, 0.93–1.00 under Muon and, under Adam, 0.55–0.67 at updates 2400–4800 recovering to 0.87–0.88 by 9600), where the recurrent gate's reads 0.14–0.28. So the audit's first prediction was partly met (no position failures; key failures reduced, not gone, and 31/40 short of 36), and its second was met.

**Batch 17 (E).** The failures at eight streams were merges, so batch 17 added the targeted split of Section [3](#3-the-task-the-outcomes-and-the-recipes) (S37's KEYMASS rule, at most three splits). `W_SPLIT` bound routed 9/10 against 3/10 for the same seeds without it (6 vs 0, two-sided *p* = 0.031); every one of its 19 splits was on target by the labels (the busy channel held two to five streams, the idle one none), and in six of nine bound runs the transition came at the evaluation right after the last split. Decay 0.98 alone bound 1/10 (it settled the merges early) and a longer budget alone 4/10: the splits, not the decay or the time, do the work. The screen handed `W_SPLIT` back for a pre-registered test.

## 5. The main result: `test_window_gate`

The test's specification was committed before the test was written (`specs/test_window_gate.md`) and amended, before any Part D run, to add the split. Both machines ran it on their own seeds (X 500–559, L 1500–1559). Part A (*S* = 2, *P* = 4, *k* = 2, no convolution, 24,000 updates, 40 seeds per machine): the window gate with one learning rate (`WIN3`), with slow memory (`WIN3_SLOW`), and the recurrent-gate recipe (`HINGE0`). Part B (*S* = 4, *P* = 4, *k* = 16, convolution, 28,800, 20 seeds): `WIN3_SLOW16` against `WIN16_A`, the recurrent gate under Adam with the early hinge window. Parts C and D (*S* = 8, *P* = 4, *k* = 16, convolution, 43,200, 20 seeds): `WIN3_SLOW_D8`, `HINGE_D8` (the recurrent recipe) and `WIN3_SPLIT_D8` (the window gate with the split), with a reset-only control (`WIN3_RESET_D8`: the Adam-state reset at the same trigger, no copy) on five seeds per machine, printed and not claimed. The perfect gate on two seeds per part was the validity arm, and bound on every one (Part C at 4800 and 3600–6000). Every CHECK passed on X, including bit-for-bit reproductions of batch 1's window gate, batch 16's `LOCAL3_SLOW` and batch 17's `W_SPLIT` (its split at 4800 included); L's file carries the metadata the test writes only after its verification passes. Nothing was cut; X ran its 225 learned-gate records in 30.9 h of training (its six perfect-gate records came from an earlier start that a container restart ended), L its 231 records in one start (projected 17.3 h).

**Table 2.** `test_window_gate`. Pooled claims (primary, exact one-sided McNemar over both machines' pairs): G1, `WIN3_SLOW16` beats `WIN16_A`: 13 vs 2 (X 5 vs 1, L 8 vs 1), *p* = 0.0037, SHOWN. G2, `WIN3_SPLIT_D8` beats `HINGE_D8`: 38 vs 0, *p* = 3.6×10⁻¹², SHOWN. G3, `WIN3_SLOW` not worse than `HINGE0` by more than two discordant pairs: `HINGE0` only 9, `WIN3_SLOW` only 11, *d* = -2, HOLDS (a bound, no *p*-value). G4, `WIN3_SPLIT_D8` beats `WIN3_SLOW_D8`: 23 vs 0 (X 12 vs 0, L 11 vs 0), *p* = 1.2×10⁻⁷, SHOWN. Per machine every claim is SHOWN or HOLDS on both, except G1 on X (5 vs 1, *p* = 0.11; on L 8 vs 1, *p* = 0.020). Bands, pooled: `WIN3_SPLIT_D8` RELIABLE, Wilson [0.835, 0.986]; `WIN3_SLOW16` RELIABLE [0.835, 0.986]; `WIN3_SLOW` MAJORITY [0.770, 0.921]; `WIN3_SLOW_D8` MINORITY [0.242, 0.530]; `HINGE_D8` NEVER [0, 0.088]. The pre-registered reading, "eight streams bind without labels or restarts" (`WIN3_SPLIT_D8` RELIABLE pooled and G2, G4 SHOWN), applies.

| part                                     | arm                                     |    X    |    L    |      pooled      |
|:-----------------------------------------|:----------------------------------------|:-------:|:-------:|:----------------:|
| A: *S*=2, *P*=4, *k*=2, discovered | `WIN3`                                  | 33/40 | 32/40 |     65/80      |
|                                          | `WIN3_SLOW`                             | 33/40 | 36/40 |     69/80      |
|                                          | `HINGE0` (recurrent, Rev. 7's recipe)   | 33/40 | 34/40 |     67/80      |
| B: *S*=4, *k*=16, bound routed       | `WIN3_SLOW16`                           | 19/20 | 19/20 |     38/40      |
|                                          | `WIN16_A` (recurrent, early window)     | 15/20 | 12/20 |     27/40      |
| C, D: *S*=8, *k*=16, bound routed    | `WIN3_SPLIT_D8` (the recipe)            | 18/20 | 20/20 | **38/40** |
|                                          | `WIN3_SLOW_D8` (no split)               | 6/20  | 9/20  |     15/40      |
|                                          | `HINGE_D8` (recurrent, Rev. 7's recipe) | 0/20  | 0/20  |      0/40      |
|                                          | `WIN3_RESET_D8` (control, of 5)         |  0/5  |  3/5  |      3/10      |
|                                          | perfect gate (of 2)                     |  2/2  |  2/2  |      4/4       |

**What the runs show (diagnostics, not claims).** The recurrent-gate recipe bound none of 40 eight-stream runs (15 collapsed, 17 non-stream, 8 merged). The window gate alone bound 15/40, and every one of its 25 failures was a merge of two, three or four streams on one channel. The split raised that to 38/40: the trigger fired 85 splits in 40 runs (none to three per run), every target labelled on target, and brought the median time to bind forward (X: 21,600 against 27,600; L: 18,600 against 32,400; medians over bound runs). The two failures (X's seeds 555 and 556) ended with two streams sharing a channel. The reset-only control matched the window gate without the split seed for seed on L (3/5 and 3/5, 0 vs 0 discordant) and bound 0/5 against 1/5 on X: on these seeds the row copy, not the Adam reset, made the difference (Section [6](#6-the-split-the-copy-not-the-reset)). At four streams, `WIN3_SLOW16`'s two failures were one unrouted bind (X) and one merge (L); `WIN16_A`'s thirteen were four position splits, three merges, four unrouted binds and two others. At two streams no arm failed by position; `WIN3`'s failures were key splits (12 of 15), `WIN3_SLOW`'s key splits and partial streams (6 and 5 of 11), and `HINGE0`'s key splits (10 of 13), with two OTHER and one partial. Slow memory on the window gate at two streams is printed, not claimed: 69/80 against 65/80 (7 vs 3, *p* = 0.17).

**What is and is not established.** The result is a pre-registered, two-machine, disjoint-seed test of a recipe that uses no stream labels, no restarts and no hinge, and that finds an eight-way partition of the memory, the streams on disjoint channel sets with one channel per stream by largest mass, in 38 of 40 runs where the previous recipe found none. The split needs to know where the key positions are (as the hinge did); the window gate needs the cue within two tokens; and the task is the grouped layout, where it is. Neither limitation is new. The audit's over-prediction for two streams ("≥ 36/40 with KEY failures gone") was not met on X (33/40) and KEY failures were reduced, not removed; what was claimed, G3, holds.

## 6. The split: the copy, not the reset

The split bundles two actions, a row copy and an optimizer-state reset, and Revision 7's screens had no reset-only control (audit, Section 2.3). Phase VI ran four looks at it, on disjoint seeds and with two different implementations of the reset.

**Table 3.** The split's two actions. In the eight-stream looks the reset-only arm fired under the live rule (S58b, the test's control) or at the recorded split updates (S50), on targets labelled on target where labelling was recorded (S58b, X's control), and, where a no-intervention arm exists (S50, the test's control), bound no more often than it. At four streams with sixteen channels nothing is needed, so S58 cannot tell the actions apart.

| where                 | configuration                                                                           |  split  | reset only | discordant, *p*                                     |
|:----------------------|:----------------------------------------------------------------------------------------|:-------:|:----------:|:----------------------------------------------------|
| batch 18, S50 (E)     | *S*=8, *k*=16, 260–269; the Adam-state reset replayed at the recorded split updates | 9/10  |   2/10   | 8 vs 1, two-sided *p* = 0.039                         |
| session H, S58b       | *S*=8, *k*=16, 460–469; live trigger                                                | 10/10 |   2/10   | 8 vs 0, one-sided *p* = 0.0039                        |
| `window_gate` control | *S*=8, *k*=16, 540–544 and 1540–1544; live trigger                                  |    —    |   3/10   | vs no split 4/10 on the same seeds, 0 vs 1        |
| session H, S58        | *S*=4, *k*=16, 420–459                                                              | 39/40 |  39/40   | 0 vs 0; uninformative (no split needed: NONE 38/40) |

Pooled over the two ten-seed screens, which differ in budget and in how the reset was applied, 19/20 against 4/20. The copy puts the idle channel's gate logits in a near-tie with the busy channel's, so the next gradient step can separate two merged streams; a reset leaves the idle channel where it was. These looks show that the reset alone is not enough; none of them has a copy-without-reset arm, so they do not show that the reset adds nothing. Session H's S61 (seeds 480–489, run after S50 and S58b and alongside the test's Part D; not cited in the first draft) has one: the exact copy without reset or noise bound 9/10, on the same seeds as the full split (9/10) and the copy with noise (9/10), against 4/10 for the reset alone and 3/10 with no trigger. Its pre-fixed readings read "the copy alone suffices" and "the reset adds nothing" (both by a one-pair bound), and its "the noise matters" reading, which needed ≥ 4 vs 0, did not apply (0 vs 0). "The copy, not the reset" remains printed, not claimed; the copy against the reset is a pre-registered claim of the next main-line test (`specs/test_split_copy.md`), and the copy alone against the full split is to follow.

**What does not replace it (session H).** Two devices the literature uses against router collapse were run beside the split and the reset at four streams, *k* = 16 (40 paired seeds), where every arm but one sat at the ceiling (NONE 38/40) and no reading could apply; a rule fixed beforehand then carried the two best-scoring arms, which were the reset and write-gate noise, to eight streams (10 seeds). Gumbel(0,1) noise on the gate's logits during training, Raven's sole anti-collapse device \[13\]: on both the read and write gates it collapsed routing even at four streams (33 of 40 runs merged), consistent with our read being routed where Raven's is dense; on the write gate alone it bound 37/40 at four streams with a later transition (median 6000 against 3600) and a lower margin, and at eight streams it bound 0/10, with up to six streams sharing a channel (10 vs 0 against the split; the screen had no no-device arm there). A Switch-style balance loss, as Mixture-of-Memories uses \[12, 17\]: 36/40 at four streams, routed by stream, but in 34 of those 36 runs the streams were mapped to different channels at key positions than at value positions and the routing margin fell from 0.87 to 0.46; it changed the routing's shape, not its count, and was not carried to eight streams. At two streams (*k* = 2, no convolution, ten seeds), a learned gate temperature with a floor ran to the floor in every run, and a stable-max softmax changed nothing: the gate wants to be sharper, not softer, and in those runs a saturated gate marked a committed wrong partition, not a plateau before binding.

**A guard the split needs at *k* = *S*.** With no spare channel (batch 18, S51, *S* = *k* = 4), the trigger once fired on a correctly routed map, found no empty channel, overwrote a live one and merged the run; in seven of its eight bound runs it never fired. Spare channels remain the default (Phase IV), and the trigger should not fire when its target channel already carries key mass.

## 7. Muon is not an ingredient

Revision 7 reported, from one machine, that under Muon the recipe "binds four streams sooner". The Muon-recipe test (`test_muon_recipe`, the first on disjoint seeds) asked three questions. M1, the early hinge window under Muon at two streams: `WIN_M` 79/80 against `SLOW_M` 69/80 pooled, 10 vs 0, *p* = 0.001, SHOWN (on L 6 vs 0, SHOWN; on X 4 vs 0, *p* = 0.0625). M2, Muon binds four streams by update 4800 more often than Adam: 12 vs 8 pooled, *p* = 0.25, NOT SHOWN; per machine it was shown on X (7 vs 1, *p* = 0.035) and reversed on L (5 vs 7). Revision 7's result had come from L (7 vs 0 on its seeds; X's cut arm gave 5 vs 1), so on fresh seeds it was confirmed on the other machine and reversed on its own; the machines disagree, by seed block or by CPU (the design cannot separate them), and pooled it is not shown. M3, the same at eight keys: 7 vs 5 over 30 pairs (X's time rule cut its Adam arm to ten seeds), *p* = 0.39, NOT SHOWN. The test's reading, "the Muon recipe is reliable at the working configurations", was not met: `WIN16_M` was MAJORITY pooled (33/40). Batch 18 then ran the window recipe with the split under Muon at eight streams: 9/10, the same as under Adam (1 vs 1), and 9/10 against a fresh Muon reference without the split at 2/10 (7 vs 0, two-sided *p* = 0.016). The eight-stream result is the recipe's, not the optimizer's. Muon's bf16 Newton–Schulz step does not reproduce across processors, which cost the project paired comparisons; nothing in the main line now needs it.

## 8. The recurrent gate's eight-stream failure, closed

Revision 7's reading was that the recurrent gate's state stops carrying the stream when its recurrence becomes expansive, and it left a cap on that recurrence being screened. Batch 15 (S41) ran the cap at 0.5 on the spectral norm of $W_h$ under Muon, alone, with a previous-token input, and with the previous-token input and no recurrence, and the capped previous-token input under Adam: the three Muon arms bound 0/10 and the Adam arm 0/5. With the look-back the capped gate's state kept the stream at key positions throughout (decodability 0.93–1.00) and still ended with the eight streams on three to seven channels; final accuracy rose in every pair (median 0.61 against 0.36) but no run bound. Batch 16 added a frozen reservoir recurrence (0/10), decay 0.98, a tenth of the rate on $W_h$, and a higher learning rate: none bound (the perfect gate binds at decay 0.98, so the decay is valid). The window gate then bound where all of these had not. Two corrections to Revision 7 follow: the gate's state must carry the stream at the *write* position, where CTX is two tokens back, not only at the read, where it is one back, so the sentence "the gate needs only one token of memory" was wrong (the audit noted it; batch 16's S47 measured the write positions, which S38–S41 had not); and S41's no-recurrence arm, whose input is $(v_t, v_{t-1})$, cannot by construction see the stream token at value positions, so its ten key splits were a property of that window and not evidence about recurrence (audit, Section 2.3). The cap, as specified, also acted from update 1 ($\sigma_{\max}(W_h) \approx 1.08$ at initialization), not only after the crossing. The diagnosis stands as a description of the recurrent gate; the remedy was to remove the recurrence.

## 9. Two-stream key splits

At two streams with two channels and no convolution, the window gate fails by routing on key identity: its key splits form within the first 600 updates, while the memory is still at 10⁻⁴ (batch 17, S49). Slow memory reduces them (`WIN3` KEY 12 of 15 failures pooled; `WIN3_SLOW` KEY 6 of 11), and batch 18 screened two further devices on seeds 160–199: two spare channels (*k* = 4: 36/40 bound routed, the outcome at *k* > *S*, with four key splits still formed by update 325) and the hinge's key term on updates 1–2400 (36/40 discovered, one KEY failure, but three "STREAM-PARTIAL" failures at η² = 0.5 by key (two at 0.5, one at 0.4999), the signature of a hard two-of-four key pattern, which relabelled would make the reading "neither"). Neither is carried to the main line. The classifier's handling of that boundary case was fixed after the first draft (`fail_class_v2`, the key rule ahead of the margin rule, commit `ba8901a`); recorded verdicts keep the rule they were read under. The configuration is the project's hardest relative to its size and the one with the least headroom (65–69/80 for every arm); it is also the only one where the window recipe's margin over the hinge recipe is small (two pairs pooled) and where the hinge recipe won a screen (batch 16: 34/40 against 31/40 on seeds 160–199).

## 10. Scale

Batch 19 (E, running) asked whether the recipe's constants are tuned to *N* = 256, *D* = 32, three layers. At four streams with sixteen channels (seeds 270–289, the oracle binding at every size and depth), with every constant unchanged: *N* = 512 and 1024 at *D* = 32, and *D* = 64 and 128 at *N* = 256, bound routed 10/10 each against the reference's 9/10, all with median transition 3600; 2, 4 and 6 applications of the shared layer bound 10/10, 10/10 and 9/10 against 10/10 for three, with six layers slower (median 4800). Both readings were "scale-free" by their pre-fixed rule (every *d* ≥ -1), at four streams only; with ten seeds the screen rules out a large drop, not a small one. Eight streams and larger key loads at the larger widths are running.

## 11. Distant cues and the memory rule

**What the literature holds.** Twelve papers on routing, memory and context were read in full (`docs/reading/`): the test-time-regression framework \[5\], Gated DeltaNet \[6\], the Tiny Recursive Model and recurrent-depth language models \[7, 8\], Flesch et al.'s Hebbian context gating \[9\], Active Dendrites \[10\], Context-Gated Associative Retrieval \[11\], Mixture-of-Memories \[12\], Raven \[13\], a study of content-based routing \[14\], the Hierarchical Multiscale RNN \[15\] and Nested Learning \[16\]. None of the twelve carries an inferred context cue across distance without labels under a sparse end-of-sequence loss (the Hierarchical Multiscale RNN infers its boundaries from a dense likelihood, as our note records): each hands the context over (a task identifier, a prototype readable off every input, a clamped context layer), carries it with softmax attention, or routes by token type. The study of content-based routing is the sharpest statement of why our header-layout runs failed: routers fed raw embeddings, recurrent summaries or pooled summaries reach 0.9–29% even with the correct route supervised, and 98–99.7% once an upstream, jointly trained mixing step has written the context into each token's representation. Flesch et al. supply the rest: a slow integrator must receive only the cue, slow integration merges interleaved contexts, and inside a block the loss exerts no pressure to use a constant cue. Our window gate reads three raw embeddings; our fading-memory gates pooled everything; our header layout is positionally solvable at fixed block length. Those are the failures the papers' analyses, read as analogies, predict.

**BDH-CQ.** Pathway's follow-up \[2\] describes its memory as "linear correction rules on *S*" with linear attention as the special case $S_t = S_{t-1} + U_\theta(D_t)$, and its reasoning as $H_{r+1} = F_\theta(H_r, S_K)$ over a frozen memory; "dimensions, exact update rules, and implementation details remain proprietary". Our reading (`docs/reading/`, an inference from the sentences quoted above, not a finding, and one reading among several, since the quoted sentence counts plain linear attention, which does not read *S*, among such rules): in the taxonomy of \[5\] and \[16\], the rules that are affine in *S* and read *S* before writing are the delta-rule family (DeltaNet, Longhorn, Gated DeltaNet, DeltaProduct), and the loop matches the recursive-refinement models \[7, 8\], of which \[8\] explains why the frozen memory must enter every step. Neither is a distant-cue mechanism. The 97.4% Sudoku figure sometimes attached to BDH is from Pathway's blog, on internal data with an unstated protocol, and is not comparable to the Sudoku-Extreme numbers of \[7\].

**Session F: delta-rule channels.** Session F replaced the channel write by Gated DeltaNet's rule, $S^c_t = S^c_{t-1}\bigl(\alpha^c_t(I - g^c_t \beta_t \hat{x}_t \hat{x}_t^\top)\bigr) + g^c_t \beta_t v_t \hat{x}_t^\top$ with unit keys; at $\beta = 0$ its logits agree with the Hebbian model's to 10⁻⁷, and with the new path disabled it reproduces a recorded run bit for bit. We had predicted that a single delta channel would be the negative control the project lacks, since the rule overwrites a key's value. It was not: with $\beta = 1$ and the fixed decay, on the two-stream header layout at *P* = 4 a single delta channel bound 8/10 (Hebbian 1/10), and on the grouped layout 4/10 against 0/10; a post-hoc look at one rerun (seed 307) found the keys at the third layer tagged by stream and nearly orthogonal across streams. With a learned forget gate (Raven's routed decay, or Gated DeltaNet's) a single delta channel bound the *P* = 8 header layout 9/10 and 8/10 against 3/10 with the fixed decay, and, post hoc, had learned to partly clear its memory at the context token (median $\alpha$ of 0.18–0.26 at CTX in the first layer, 0.72–0.74 in the third, 1.00 at keys and values). Under the window gate, with a learned $\beta$, the delta rule was ahead at two streams (36/40 against the recorded 31/40 on other seeds; Fisher *p* = 0.11, a bound by the screen's rule), equal at four (20/20 against 19/20) and worse at eight (3/10 against 9/10, 0 vs 6, *p* = 0.016), where even the delta perfect gate takes 22,800–28,800 updates against 2400–4800 for Hebbian, at about three times the compute per step. A learned write strength $\beta$ collapsed toward "do not write" whenever it was the only change. The Hebbian write stays on the main line.

**What this changes.** At two streams the memory has routes to conflicting bindings that need no partition: make the key context-dependent through the stack, so that one delta memory holds both (the route Nested Learning's self-modifying key projections describe \[16\]; Gated DeltaNet itself overwrites), or clear the memory at the context switch (Gated DeltaNet's $\alpha_t \to 0$ \[6\]). So "the partition is necessary" is a claim about the number of contexts, not about two: at eight streams the gated Hebbian recipe binds 38/40, the same recipe on delta channels stalls at 3/10, and one delta channel binds 2/10 at four streams and 0/10 at eight (session F's premise test S68: seeds 300–309, 43,200 updates, both delta perfect gates bound at 1200; the gated recipe on other seeds 19/20 and 9/10). The first draft of this revision said the premise test had not been run; it had, on F's branch, before the draft was committed. If the one-seed observation of stream-tagged keys at the third layer holds up, the context *is* reaching depth in the header layout, which is what the per-layer residual-stream gate of the audit (Section 3.3) requires. Session G's first screen (S55) probed the window-gate stacks on the *P* = 8 header layout and found the stream linearly undecodable from the residual stream after training (0.55–0.58 at layers 2 and 3 and at the output, from 0.70 at initialisation); the delta stacks are probed in G's and F's follow-ups.

**The distant-cue task.** A header layout with random block lengths and eight keys per stream exceeded the oracle's budget (neither perfect gate bound within 24,000 updates), so a lighter version with four keys per stream in two blocks of random length and order replaced it (F's S70): both perfect gates bind it (five of six seeds by update 1200), and the window-gate recipe discovers 0/20 on Hebbian memory and 0/20 on delta memory, by key splits; session G's register screens run on it (Revision 9). What a distant-cue test needs is now clear: random block lengths (so that position cannot route), a budget the oracle meets, and routing reported at write positions, so that "bound" is not mistaken for "routed".

## 12. Where the mechanism stands

**Table 4.** The main configurations in `test_window_gate` (pooled over X and L, disjoint seeds; bound routed, or discovered at *S* = 2). ʷWith the early hinge window (`WIN16_A`); Revision 7's full-hinge arm bound 35/40 on shared seeds, routed 15/20 on X and 16/20 on L. ᵉSession H's S58, exploratory, 40 seeds. ᶜOne Hebbian channel without the convolution, every such run on the grouped layout in `results/` (the blocked and shuffled layouts add 0/30); with the convolution at learning rate 4×10⁻³ one channel binds 18/40 and 10/20 at two streams (Phase IV's `conv_lr`; Revision 6 reported 34/80). At four streams one Hebbian channel with the convolution bound 0/20 (Phase IV's `test_scale_axes`, arm B4s, X and L); at eight it has not been run; one delta channel binds 2/10 and 0/10 there (Section [11](#11-distant-cues-and-the-memory-rule)).

| setting                           | perfect gate | recurrent gate, Rev. 7 recipe | window gate + SLOW | window gate + SLOW + split | one channel |
|:----------------------------------|:-------------|:------------------------------|:-------------------|:---------------------------|:------------|
| *S*=2, *P*=4, no conv.        | 4/4        | 67/80                       | 69/80            | —                          | 0/60ᶜ  |
| *S*=4, *P*=4, *k*=16, conv. | 4/4        | 27/40ʷ                   | 38/40            | 39/40ᵉ                | —           |
| *S*=8, *P*=4, *k*=16, conv. | 4/4        | 0/40                        | 15/40            | **38/40**           | —           |

**The partition does what one Hebbian channel cannot, and the gate finds it.** Given the partition, every grouped-layout configuration tried with the *N* = 256 model binds; one Hebbian channel does not without the convolution (0/60 at two streams in `results/`), with the convolution at a higher learning rate binds a minority at two streams (18/40, 10/20) and none at four (0/20). A gate that reads three tokens, trained with the memory held slow for 2400 updates and with a channel split when binding stalls, finds the partition at two, four and eight streams without labels or restarts.

**Discovery is a race, and the recipe's parts each remove one way of losing it.** Slow memory reduces the key splits (12 of 15 failures to 6 of 11 at two streams; the count itself, 7 vs 3, is not shown); the window gate cannot count positions, so position splits are gone; the split breaks the merges that remain at eight streams by giving an idle channel the busy channel's logits. The recurrent gate's pathologies (position splits, the expansive recurrence) were not obstacles to the mechanism; they were obstacles of that gate.

**Two routes, and the partition's claim.** A single delta-rule channel binds the two-stream header layout (8/10 at a fixed decay; 9/10 and 8/10 with a learned forget gate at eight keys) and the grouped two-stream layout on 4/10. The partition's advantage is in the number of contexts: one delta channel binds 2/10 at four streams and 0/10 at eight on the grouped layout (S68).

## 13. Retrospective

### 13.1 Revision 7's claims

**"At eight streams the gate forgets the context" (subtitle).** **Superseded.** True of the recurrent gate; the window gate carries the context at every write position by construction and binds 38/40.

**Revision 7's recipe (slow memory and a hinge) binds without restarts at two streams and at four with sixteen channels.** **Stands**, and is replaced by a recipe without the hinge, which equals it at two streams (within the bound), beats its early-window form at four (G1) and beats it at eight (G2).

**"The gate needs only one token of memory" (Section 11 of Revision 7).** **Corrected.** One at the read, two at the write; the write-position decoders were not measured in Revision 7's screens S38–S41 (batch 16, S47, measured them).

**"Under Muon the recipe binds four streams sooner."** **Not shown** on disjoint seeds pooled (12 vs 8, *p* = 0.25), and the machines disagreed: confirmed on X's fresh seeds (7 vs 1), reversed on L's (5 vs 7), where Revision 7's result had been shown. The first draft of this revision called it a seed effect, which the data do not establish.

**"A cap on the gate's recurrence is being screened."** **Resolved: not.** No capped arm bound (0/10, 0/10, 0/10 and 0/5); the cap acted from update 1.

**The row copy breaks merges (Section 10 of Revision 7, from S36/S37, which scored BOUND).** **Supported** with the right outcome (bound routed) and a reset-only control, at eight streams, in three exploratory looks; printed, not yet claimed.

### 13.2 Errors made during Phase VI

**Predictions that the data overturned.** We predicted that a single delta channel could not bind conflicting streams (it binds the two-stream header layout 8/10); that Gumbel noise on the gate's logits could replace the split (on the read gate it collapsed routing; on the write gate it worsened eight-stream merges); that a balance loss would attack merges (where it was run there were no merges to attack, and it changed the key- and value-position maps instead); that a learned forget gate would widen the gap between the oracle and one channel (it shrank it, on the eight-key header layout); and, in the audit, that the window gate with slow memory would reach ≥ 36/40 on seeds 160–199 with no key failures (it reached 31/40 there, and 33 and 36 of 40 on the main test's seeds, with key failures reduced).

**A distant-cue task that exceeded its oracle.** Session F's randomized header layout did not bind under the perfect gate within its budget, so it gave Session G no baseline. The specification fixed the block lengths and left the key load open; F's resolution, eight keys per stream, matched the eight-key header layout that the oracle already binds slowly.

**A classifier at its own threshold.** Three batch-18 runs sit at exactly η² = 0.5 by key and are labelled STREAM-PARTIAL; the hard two-of-four key pattern should be KEY. The reading that depended on them is withdrawn until the classifier is fixed.

**Attribution of a number.** An earlier working note put Pathway's 97.4% Sudoku figure beside TRM's 87.4% as if on one protocol; the figure is from Pathway's blog, on internal data, and the protocols differ.

**Process.** Container restarts cost in-flight runs on X (the test's first start), on F (twice, while the session was idle with only detached processes running) and on E in several batches (batch 16's segments 2, 6 and 7 among them); on F a harness-tracked keepalive stopped it, and some form of it should be in every long run. Every resumed segment began by reproducing a recorded run bit for bit. Batch 18's Muon reference did not reproduce across hosts and was rerun fresh, as the rules require.

## 14. Methodological findings

Revisions 5–7 recorded sixteen lessons; Phase VI adds five.

**Disjoint seeds make the pooled test a test.** From the Muon-recipe test on, each machine has its own seeds and the pooled McNemar is the primary claim. The window-gate test's G1 is the case this was for: not shown on X alone, shown on L and pooled.

**Audit the records before the next screen.** The window gate had been screened, had won, and had been dropped for a reason unrelated to the obstacle at hand; one reading of the records found it. A fixed prediction before the rerun made the audit falsifiable.

**Control for the bundled action.** A split that copies and resets needed a reset-only arm; so does every intervention with two parts.

**Score routing at the write positions.** A stream's channel at key positions can differ from its channel at value positions in a bound run (session H: 34 of the balance-loss arm's 36 bound-routed runs, and two to five runs per arm elsewhere); the writes are at the values.

**Borrowed devices need the model's read path.** Gumbel noise works for a dense read and fails for a routed one; a balance loss balances whatever is cheapest to balance.

## 15. Limitations

**The task.** Synthetic, with explicit context tokens; in the grouped layout each key's stream token is within the window gate's reach. Distant cues are unresolved, here and in the literature we read.

**The recipe needs task structure.** The split reads the key positions (as the hinge did). A language model has no marked keys; the analogue is untested.

**The model.** Three layers (two to six screened), one head, *d* = 32 (to 128 screened), *N* = 256 (to 1024 screened), sequences of at most 99 tokens, a fixed memory half-life of about 14 tokens.

**Two streams.** The hardest configuration relative to its size; every arm is at 65–69/80 and key splits remain.

**Screens.** Exploratory sessions use 10–40 runs per arm and lenient rules; their readings are labelled, and nothing from them is a claim until a main-line test repeats it.

## 16. Distance to a language model

**What Phase VI adds to the case.** The gate is now a convolution: parallel, cheap, the form every modern linear-attention model already uses. The recipe is two learning-rate groups and a plateau rule. It finds an eight-way partition without labels, and its constants hold to four times the width and twice the depth.

**What stands in the way,** in order:

1.  *Context cues at a distance.* Open in this project and in the literature. The next steps are a probe of where the cue is in the residual stream, a register that overwrites on the cue and holds otherwise, and a per-layer gate over the residual stream (audit, Section 3.3; `docs/reading/`, Section 4). The risk is that an end-of-sequence loss gives too little pressure to discover the cue label-free; language supplies a denser loss.

2.  *The split's nuisance variable.* The key positions come from the task.

3.  *Memory horizon.* A fixed decay; session F's learned forget gates are the first step, and at eight streams the memory rule that carries them is slower.

4.  *Baselines.* Mixture-of-Memories, Raven and Gated DeltaNet route or forget at scale; the comparison that matters is whether their routers discover *contexts* rather than token types, which none of their papers tests.

**A staged path.** \(i\) On CPUs: the distant-cue task, eight streams at the larger widths (the premise test is done: Section [11](#11-distant-cues-and-the-memory-rule)). (ii) A GPU port as a small package, with a chunkwise delta memory, the recall benchmarks \[18\] and the baselines above, on disjoint seeds. (iii) Small language models, asking one question: do the channels specialize by context or by token type.

## 17. Conclusion and next steps

Multi-channel Hebbian plasticity in multilayer BDH solves context-conditional binding by partitioning memory, and a label-free gate finds the partition at eight streams: a window gate, slow memory for 2400 updates, and a channel split when binding stalls, in 38 of 40 runs on two machines with disjoint seeds, where the previous recipe bound none. In every look so far the split's row copy, not its reset, is what works. The obstacle Revision 7 named, the gate's recurrence, was removed rather than fixed.

Next:

- A pre-registered "copy, not reset" claim and the classifier fix, in the next main-line test.

- Session G's register screens on the header layout; session F's S69–S70 and follow-up; batch 19's eight-stream width runs.

- Revision 9 after those, then the GPU package.

## Repository layout and how to run the tests

- **`bdh.py`:** Pathway's BDH model, used unmodified. The multi-channel model subclasses it:
  - it replaces the attention module, multiplying each score by the gate match at every layer;
  - it adds the gate's parameters (the recurrent gate of Phases II–V, or the window gate of Phase VI, converted in place);
  - with the convolution, it runs a copy of BDH's layer loop with the causal convolution inserted.
- **`test_*.py`:** one file per experiment.
  - Each test's docstring holds its pre-registered design (background, arms, seeds, claims and thresholds), and after the run, its recorded result on each machine and pooled.
  - Every Phase III–VI test first runs its inherited verification chain and its own numbered CHECKs, then prints a runtime projection, then trains.
  - Phase VI's tests are `test_muon_recipe.py`, `test_window_gate.py` and `test_split_copy.py`; Phase V's are `test_stream_recipe.py`, `test_stream_curriculum.py`, `test_slow_start.py`, `test_recipe_scope.py` and `test_early_recipe.py`.
  - Tests from Phases I–II (for example `test_interference*.py`, `test_asymmetric_routing.py`, `test_instrument_v2.py`) and earlier modules (`bdh_multichannel.py`, `bdh_mc.py`, `bdh_recurrent.py`) are kept for the record.
- **`longrun.py`:** the detached run, its watcher and the resume and projection helpers (below). **`fail_class_v2.py`:** the failure classes with the key rule ahead of the margin rule, used by new tests; `docs/fail_class_v2.md` reclassifies the recorded two-stream runs (no recorded verdict changes).
- **`specs/`:** each Phase VI test's specification, committed before the test was written (and amended, with the amendment dated, before any affected run).
- **`results/X/`, `results/L/`:** copies of each machine's result JSON files, with `results/README.md` recording each file's machine, commit and start time. Results files are gitignored where they are written. Both machines' files for every Phase V and VI test are recorded.
- **`multichannel_hebbian_report_v8.pdf`** and **`multichannel_hebbian_report_v8.tex`:** Revision 8 of the technical report, which this README follows. `multichannel_hebbian_report_v7.pdf`, `multichannel_hebbian_report.pdf` and `multichannel_hebbian_report_v2.pdf` are earlier revisions.
- **`docs/audit_2026-10.md`:** the October 2026 audit of the project's records (code, beliefs, literature, process), whose Section 3.1 brought the window gate back.
- **`docs/reading/`:** notes on twelve papers on context routing and memory, read in full, with the synthesis `distant_cues_2026-10.md` and the prompts for the parallel exploratory sessions.
- **`docs/review_2026-10/`:** an independent referee's report on Revision 8 (session V, on branch `claude/review-V`, merged here): `review.md`, with `recompute.py`/`recompute.md` (every number of Tables 1–6 recomputed from the records), `preregistration.md` (the git history of every claim), `claims.md` (116 statements with verdicts), `statistics.md`, `literature.md`, `repro.py` (one run per Phase VI test reproduced bit for bit on a third CPU) and `revision_8_1_check.md` (its check of the corrections). Revision 8.1 corrects the statements it found wrong; its structural requests are for Revision 9.
- **`docs/revision7.md`**, **`docs/revision6.md`:** this README's research text as of Revisions 7 and 6, kept for their full accounts of Phase V and of Phases I–IV.
- **Branch `claude/outside-ideas`:** the exploratory screens (`explore_*.py`, outputs and per-batch logs in `explore_out/`, verdicts in `explore_out/README.md`). **Branches `claude/explore-F`, `claude/explore-G`, `claude/explore-H`:** the parallel exploratory sessions (delta-rule channels and forget gates; distant cues; anti-collapse devices), each with its reports under `explore_out/<session>/`. Nothing on these branches is a result.

**Running a test** (it runs the CHECKs, the projection, then the full test; runs are cached, so an interrupted test resumes). Phase VI tests take a required `--machine X|L` flag, and each machine runs its own seed block:

```bash
pip install -r requirements.txt
python3 -u test_window_gate.py --machine L --workers 6
```

**To compute the pooled claims** once both machines' files are complete (the pooled block is identical from either side):

```bash
python3 -u test_window_gate.py --machine L --report --also results/X/window_gate_results.json
```

**Tests that pair with earlier recorded runs** read *this machine's* own earlier results files: by default from the working directory, or from flags such as `--early`, `--slow`, `--recipe` or `--short`. Before pairing, they reproduce a recorded run from each file bit for bit. If a file does not reproduce, the test stops before training, or reports the paired claims as UNTESTED.

Training is deterministic for a given processor, kernel path and thread count; workers run one thread each.

**Long runs** (`longrun.py`). Long runs in cloud containers have been lost to container restarts. `longrun.py start` launches the test detached (its own session, stdin closed, output appended to `longrun/<name>.log`, gitignored) and records each start's process, boot id and commit. `longrun.py watch --resume` prints the new log lines and the results file's record count for up to a few minutes. It relaunches the same command when the run has stopped without an exit code (a container restart or a kill), and the test then resumes from its cached records. A run that exits with an error is reported, not relaunched. Tests resume the way `test_window_gate`'s Part D did: runs already in the results file are skipped, and the meta keeps the first start's commit and time with every later start listed under `starts` (`longrun.resume_meta`). Each test prints its runtime projection, of all runs and of the runs still to do, before the first run (`longrun.projection`).

```bash
python3 longrun.py start window_gate_L --results window_gate_results.json --total 231 -- \
    python3 -u test_window_gate.py --machine L --workers 6
python3 longrun.py watch window_gate_L --minutes 9 --resume    # repeat until FINISHED
```

Without `longrun.py`, `setsid nohup python3 -u test_window_gate.py --machine L --workers 6 > window_gate.log 2>&1 &` detaches a run, and rerunning the same command resumes it.

## Provenance

**Table 5.** Phase VI main-line tests on branch `claude/bdh-growth-hebbian-inference-w90069`. L's results files for the Phase V tests are now in `results/L/` (`00ac0f2`, `d0e6c85`; the Phase IV scale-axes file at `42af095`). X: Intel Xeon at 2.80 GHz for both tests; L: Intel i7-12650H; PyTorch 2.14.0.

| test          | X: test commit / result commit                                                  | L: results file (commit recorded)                            | runs, X; L |
|:--------------|:--------------------------------------------------------------------------------|:-------------------------------------------------------------|:-----------|
| `muon_recipe` | `74f5907` / `3e2d171`; pooled `2a6ba3d`                                         | `L/muon_recipe_results.json` (`74f5907`)                     | 156; 166      |
| `window_gate` | spec `9a769c7`, amended `d05f0ca`; test `2f98f06`, Part D `886a668` / `628eb7c` | `L/window_gate_results.json` (`886a668`; recorded `1cd20a1`) | 231; 231      |

**Table 6.** Exploratory work in Phase VI, one thread per run, each segment opened by a bit-for-bit reproduction of a recorded run. Commits are the last on each branch before Revision 8's commit (`4a3eebc`), and the run counts are theirs; screens pushed after that commit are noted with their own counts. L's recorded start times are local time (UTC-<!-- -->6); X's are UTC. The audit is `docs/audit_2026-10.md` (`b22da04`); the reading notes and synthesis are `docs/reading/` (`ad8a92e`).

| batch or session       | screens                                                                                                                                                                                           | code commit                    | runs |
|:-----------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------|-----:|
| 15 (E)                 | S41 gate cap, S42 $W_h$ spectrum                                                                                                                                                                  | `90c3a28`                      |   45 |
| 16 (E)                 | S43–S44 window gate + slow memory, S45 reservoir, S46 knobs, S47 decoders                                                                                                                         | `93bdfe6` (verdicts `092f355`) |  165 |
| 17 (E)                 | S48 split on the window gate, S49 key splits                                                                                                                                                      | `2531480` (verdicts `992e320`) |   49 |
| 18 (E)                 | S50 reset control and Muon, S51 split at *k*=4, S52 key splits                                                                                                                                  | `7ef4e15` (verdicts `ac2d3ba`) |  120 |
| 19 (E, running)        | S64 width, S65 depth; S66 eight streams and S67 capacity under way                                                                                                                                | `8cff21e` (count at `5dfc2b5`) |  118 |
| H (`claude/explore-H`) | S58 anti-collapse, S58b at eight streams, S60 temperature; S61 copy without reset (Section [6](#6-the-split-the-copy-not-the-reset)); S61–S63 add 300 runs | `7e9197f`                      |  318 |
| F (`claude/explore-F`) | delta memory, S53, S54, S59; S68 premise test; S69–S70 add 78 runs (`9a6c2bf`)                                                                                                                    | `a2e55ed`                      |  322 |
| G (`claude/explore-G`) | S55 probe (Section [11](#11-distant-cues-and-the-memory-rule)); S56–S57 register and latch add 139 runs (`427db36`; Revision 9)                                 | `514d17f`                      |   22 |

## References

1. A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz. *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain.* [arXiv:2509.26507](https://arxiv.org/abs/2509.26507), 2025.
2. B. Engdahl, A. Kosowski, J. Chorowski, Z. Stamirowska, P. Uznański, J. Jiang, R. Phadke, R. Kinas, R. Zhong. *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning.* [arXiv:2608.09888](https://arxiv.org/abs/2608.09888), 2026.
3. O. Sieberling et al. *Triadic Linear Attention.* [arXiv:2609.36529](https://arxiv.org/abs/2609.36529), 2026.
4. A. Gu, T. Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* [arXiv:2312.00752](https://arxiv.org/abs/2312.00752), 2023.
5. K. A. Wang, J. Shi, E. B. Fox. *Test-time regression: a unifying framework for designing sequence models with associative memory.* [arXiv:2501.12352](https://arxiv.org/abs/2501.12352), 2025.
6. S. Yang, J. Kautz, A. Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* [arXiv:2412.06464](https://arxiv.org/abs/2412.06464), 2025.
7. A. Jolicoeur-Martineau. *Less is More: Recursive Reasoning with Tiny Networks.* [arXiv:2510.04871](https://arxiv.org/abs/2510.04871), 2025.
8. J. Geiping et al. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.* [arXiv:2502.05171](https://arxiv.org/abs/2502.05171), 2025.
9. T. Flesch, D. G. Nagy, A. Saxe, C. Summerfield. *Modelling continual learning in humans with Hebbian context gating and exponentially decaying task signals.* [arXiv:2203.11560](https://arxiv.org/abs/2203.11560), 2022.
10. A. Iyer et al. *Avoiding Catastrophe: Active Dendrites Enable Multi-Task Learning in Dynamic Environments.* [arXiv:2201.00042](https://arxiv.org/abs/2201.00042), 2022.
11. M. Choraria et al. *Context-Gated Associative Retrieval: From Theory to Transformers.* [arXiv:2605.10970](https://arxiv.org/abs/2605.10970), 2026.
12. J. Du et al. *MoM: Linear Sequence Modeling with Mixture-of-Memories.* [arXiv:2502.13685](https://arxiv.org/abs/2502.13685), 2025.
13. A. Afzal, A. Bick et al. *Raven: High-Recall Sequence Modeling with Sparse Memory Routing.* [arXiv:2607.25357](https://arxiv.org/abs/2607.25357), 2026.
14. A. Basu. *When Does Content-Based Routing Work? Representation Requirements for Selective Attention in Hybrid Sequence Models.* [arXiv:2603.20997](https://arxiv.org/abs/2603.20997), 2026.
15. J. Chung, S. Ahn, Y. Bengio. *Hierarchical Multiscale Recurrent Neural Networks.* [arXiv:1609.01704](https://arxiv.org/abs/1609.01704), 2017.
16. A. Behrouz et al. *Nested Learning: The Illusion of Deep Learning Architectures.* [arXiv:2512.24695](https://arxiv.org/abs/2512.24695), 2025.
17. W. Fedus, B. Zoph, N. Shazeer. *Switch Transformers.* JMLR 23(120), 2022.
18. S. Arora et al. *Zoology: Measuring and Improving Recall in Efficient Language Models.* [arXiv:2312.04927](https://arxiv.org/abs/2312.04927), 2023.

---

*On the upstream README that follows:* its Sudoku Extreme figure (97.4%) is Pathway's own, reported on internal data with an unstated protocol, and is not a result of this project nor comparable to the Sudoku-Extreme numbers of [7] (Section 11). The section is reproduced unchanged.

# About BDH (upstream README from Pathway)

*The remainder of this file is the original README of Pathway's BDH repository, unchanged apart from heading levels.*

## BDH (Dragon Hatchling)

### **Bridging the Gap Between Transformers and the Brain**

**BDH (Dragon Hatchling)** is a biologically inspired large language model architecture that connects principles of deep learning with the foundations of neuroscience. Developed by researchers at [Pathway](https://pathway.com), BDH provides a theoretical and practical framework for understanding the emergence of reasoning and generalization in artificial systems.

This repository contains the official implementation from the paper:
> *A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz.*
> [_The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain_](https://doi.org/10.48550/arXiv.2509.26507), arXiv (2025).


### Overview

BDH represents a **scale-free, locally interacting network of neurons** capable of intrinsic reasoning dynamics. BDH scales like a Transformer on performance benchmarks—yet retains full interpretability and theoretical grounding in the fine-grained dynamics of neuron interactions.

**Key properties:**

- **Scale-free network topology** mimicking biological connectivity
- **Locally interacting neuron particles** with excitatory/inhibitory dynamics
- **Hebbian working memory** based on synaptic plasticity, displaying monosemanticity
- **GPU-friendly state-space formulation** for efficient implementation
- **Interpretable activations** that are sparse and positive

BDH formalizes a bridge between **neural computation and machine-based language understanding**. It shows how **macro reasoning behavior** in large AI models emerges from **micro-level neuron dynamics**, guided by principles of graph theory and local computation.

Empirically, BDH matches **GPT-2–scale Transformers** across language and translation tasks at equivalent parameter scales (10M–1B).


***

### Architecture

<img src="figs/architecture.png" width="600"/>

***

### Relation to Transformers

<img src="figs/vocab.png" width="600"/>

BDH and the Transformer share attention-inspired computation; however, BDH’s graph-based architecture makes its attention **emerge naturally from neuron-level interactions**, reflecting attention as seen in biological systems.

***

### Scaling Laws

<img src="figs/bdh_scaling.png" width="600"/>

BDH follows **Transformer-like scaling laws**, maintaining parameter efficiency while achieving interpretability at any scale.

***

### Latest research update: Sudoku Benchmark

Note: The Sudoku Extreme result refers to Pathway’s internal BDH implementation, not to the current open-source repository. This repository contains the implementation of the baseline variant as described in our [public paper](https://arxiv.org/abs/2509.26507) and does not reproduce the 97.4% benchmark result out of the box. See the dedicated Extreme Sudoku research blog post for additional benchmark context and the reported results.

On Sudoku Extreme, BDH reaches 97.4% accuracy across roughly 250,000 difficult puzzles, without chain-of-thought, solution backtracking, or external tool use, while leading LLMs struggle to perform on the benchmark at all.

Language is not enough for intelligence. Transformers process information token by token with limited internal state, which makes search-heavy, non-linguistic reasoning tasks like Sudoku awkward. BDH uses a larger latent reasoning space with intrinsic memory that supports learning and adaptation during use.

We believe that the future of AI will belong to systems that can reason natively across domains, that can hold multiple possibilities in a rich latent space, and that can converge on solutions without needing to verbalize every step. BDH is our answer to that challenge. It is designed to be a universal reasoning system that can speak our language without being trapped inside it. And yes, it solves Sudoku.

Read more: [Post-transformers: Sudoku Bench](https://pathway.com/research/beyond-transformers-sudoku-bench)

#### Performance Comparison

| Model | Sudoku Extreme Accuracy | Relative Cost |
|------|------------------------|--------------|
| Pathway BDH | 97.4% | 10× lower, No chain-of-thought |
| Leading LLMs (O3-mini, DeepSeek R1, Claude 3.7 8K) | ~0% | High (chain-of-thought) |

*Table 1: Performance comparison on extreme Sudoku benchmarks (~250,000 difficult puzzles).*  
*Source: Pathway internal data and https://arxiv.org/pdf/2506.21734 for the Leading LLMs’ accuracy score. Pathway’s approach reflects top-1 accuracy and does not rely on chain-of-thought nor solution backtracking.*


### Installation and Training

```bash
# install dependencies
pip install -r requirements.txt

# train BDH on a toy dataset
python train.py
```

<!--For visualization and interpretability analysis, explore the example notebooks in `notebooks/`.-->



### Learn and Discuss

- Watch the *SuperDataScience podcast* [▶️ *Dragon Hatchling: The Missing Link Between Transformers and the Brain*](https://www.youtube.com/watch?v=mfV44-mtg7c) (72 min.) featuring Adrian Kosowski in conversation with Jon Krohn, unpacking BDH’s neuron-level architecture and sparse reasoning dynamics.

- Read about BDH in
[*Forbes*](https://www.forbes.com/sites/victordey/2025/10/08/can-ai-learn-and-evolve-like-a-brain-pathways-bold-research-thinks-so/),
[*Semafor*](https://www.semafor.com/article/10/01/2025/new-ai-research-claims-to-be-getting-closer-to-modeling-human-brain),
[*The Turing Post*](https://www.turingpost.com/p/fod-121-300-million-to-start-a-big-promise-for-science#the-freshest-research-papers-catego),
[*Quantum Zeitgeist*](https://quantumzeitgeist.com/palo-alto-ai-firm-pathway-unveils-post-transformer-architecture-for-autonomous-ai/),
[*Golem*](https://www.golem.de/news/neue-ki-architektur-was-ist-baby-dragon-hatchling-2510-201047-2.html),
and elsewhere in the media.

- Discuss and share the BDH paper on:
[*Hugging Face Papers*](https://huggingface.co/papers/2509.26507), 
[*Alphaxiv*](https://alphaxiv.org/abs/2509.26507),
and [*EmergentMind*](https://emergentmind.com/papers/2509.26507).

### Community Projects

- [adamskrodzki/bdh](https://github.com/adamskrodzki/bdh): dynamic vocabulary, stateful attention
- [mosure/burn_dragon_hatchling](https://github.com/mosure/burn_dragon_hatchling): Burn port
- [severian42/bdh](https://github.com/severian42/bdh): MLX port
- [Git-Faisal/bdh](https://github.com/Git-Faisal/bdh)
- [GrahLnn/bdh](https://github.com/GrahLnn/bdh)

### Acknowledgements
We thank Andrej Karpathy for the [nanoGPT](https://github.com/karpathy/nanoGPT/) code and the tiny Shapespeare dataset used in this demonstration.

BDH research stands at the intersection of **AI architecture**, **biological learning models**, and **theoretical computer science**—an effort to map the *equations of reasoning* between artificial and biological intelligence.
## Acknowledgements
We thank Andrej Karpathy for the [nanoGPT](https://github.com/karpathy/nanoGPT/) code and the tiny Shapespeare dataset used in this demonstration.

BDH research stands at the intersection of **AI architecture**, **biological learning models**, and **theoretical computer science**—an effort to map the *equations of reasoning* between artificial and biological intelligence.
