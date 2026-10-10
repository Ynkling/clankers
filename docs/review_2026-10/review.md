# Referee report: "Multi-Channel Hebbian Plasticity in Multilayer BDH Solves Context-Conditional Binding by Partitioning Memory", Revision 8 (draft)

*Referee V, 10 October 2026. Branch `claude/review-V`, off the main line at `8be3f98`; the report reviewed is `4a3eebc`. Supporting files in this directory: `recompute.md` (every number recomputed from the records), `preregistration.md` (git history), `claims.md` (116 statements with verdicts), `statistics.md`, `literature.md`. I changed nothing outside this directory. Besides my own checking scripts I ran one recorded run per Phase VI test (1 thread, Xeon @ 2.10 GHz, torch 2.14.0); both reproduced X's records bit for bit.*

## 1. The contribution as I understand it

BDH's Hebbian memory is split into k channels. A per-token softmax gate g_t distributes each token's write and read over the channels, so the attention score becomes (x_t·x_s)(g_t·g_s)·decay^(t−s). On a synthetic task, S streams bind every key to a different value, cued by a stream token. One Hebbian channel cannot hold the conflicting bindings at four or eight streams; a hand-set gate, one channel per stream, can. The question is whether a gate trained only on the task loss finds that partition. Revision 8's answer is a recipe of three parts:

- a width-3 causal convolution of the embeddings as the gate ("window gate"), replacing a recurrent gate;
- slow memory (other parameters at a tenth of the gate's rate for 2400 updates);
- a plateau-triggered "split" that copies the busiest channel's gate row onto the idlest, aimed by gate mass at key positions.

In a pre-registered test on two machines with disjoint seeds, the recipe binds eight streams with a stream-to-channel partition in 38/40 runs. The previous recipe binds 0/40, the same recipe without the split 15/40. The rest is exploratory: the split's row copy, not its optimizer reset, does the work; borrowed anti-collapse devices do not replace it; width and depth do not matter at four streams; and, unexpectedly, a single delta-rule channel binds the two-stream *header* layout without any partition.

## 2. Strongest points

1. **The headline is real, reproducible and pre-registered.** Every count, discordant pair, p-value, Wilson interval and band in the main table and in the copy/reset table reproduces from the JSON records under outcome definitions I re-implemented independently (`recompute.md`, 140 matching rows). The stored transitions, maps and flags agree with the recomputation in all 462 records of `test_window_gate`. The spec preceded the code, which preceded every run. No claim or reading in any of the three main-line docstrings I audited was edited after a run; later commits only appended (`preregistration.md` §1). Two recorded runs, one per Phase VI test, reproduce bit for bit on a third CPU model. The effect is large: 23 vs 0 discordant pairs against the no-split arm, and it survives any multiplicity correction and any choice of test (`statistics.md` §1).
2. **The process is unusually auditable and self-correcting.** Disjoint seeds per machine make the pooled test valid. Readings are fixed in committed docstrings before runs, and on four exploratory branches 21 audited screens show no post-run edits to a reading. The report retracts its own earlier claims (the "one token of memory", the Muon speed claim, a classifier boundary case), and an audit of the records produced a falsifiable prediction that partly failed and is reported as failing.
3. **The controls are the right ones and they are reported even when they cut against the authors.** There is a reset-only control for the split, and the perfect gate serves as a validity arm in every part. The single-channel delta result undercuts "the partition is necessary" at two streams, and it is in the abstract.

## 3. Weaknesses, each with the evidence that would resolve it

**W1. Statements that were false when the report was committed.**
- (a) "No single-channel run has yet been made" at eight streams (tex 35, 226, 254). Session F's premise test had ten single-delta-channel S = 8 runs on its branch, none bound, by 18:33 UTC. The report was committed at 19:55 (`recompute.md`, row tex 35).
- (b) Revision 7's Muon result "did not replicate on the other machine's seeds" and was "a seed effect" (tex 200, 267). Revision 7's E2 was shown on L. On fresh seeds it was confirmed on X (7 vs 1) and reversed on L (5 vs 7) (`statistics.md` §6a).
- (c) Routing for Revision 7's HINGE4k16 "recorded only on X" (tex 246). L's records carry it: 16/20.
- *Resolve:* correct the sentences; cite S68 (2/10 at four streams, 0/10 at eight). S68 *supports* the report's "number of contexts" reading.

**W2. The Section 1 argument about the convolution compares two different models.** "Without [the convolution] the perfect gate failed to bind on the four seeds tried in batch 18 and session H, while session F found it binding 10/10 ...: seed-dependent" (tex 98). The failing gate is `test_multilayer_binding`'s ceiling at N = 64 (7,552 parameters). F's, and the main line's own CEIL_A, use the N = 256 model the report describes (25,984 parameters) and bound 10/10 and 4/4 (`recompute.md`, row tex 98). The inference ("the convolution carries part of the binding") is unsupported, and it contradicts Section 12's "given the partition, every grouped-layout configuration tried binds" (tex 250). *Resolve:* run the N = 256 perfect gate without the convolution on 160–161 and 470–471, or delete the sentence.

**W3. "The optimizer reset ... adds nothing" (abstract) is not shown by the looks the report cites.** All three compare copy + reset with reset alone. They establish that the copy is needed (pooled 16 vs 1), not that the reset is not. The one copy-without-reset arm in the project, session H's S61 (COPY 9/10 = SPLIT 9/10), was complete fourteen hours before the report and is uncited. Its pre-registered noise reading was "the noise matters"; the branch's "the noise does not matter" was written after the run (`preregistration.md` §2). *Resolve:* the pre-registered copy-only test already specified in `specs/test_split_copy.md`; until then, say "the reset alone is not enough".

**W4. One task, one model, and a recipe that reads the task's structure.** Every main-line result is the grouped layout, where each key's stream token sits inside the gate's three-token window. The split is aimed by gate mass at the task's key positions. "Label-free" holds for stream labels, not for task structure, and the abstract omits the caveat (`claims.md` A7). The title's "solves context-conditional binding" generalises from this one configuration. Distant cues remain unsolved: F's randomised header layout exceeded its oracle, and session G's probe (reported 4.5 h before the report) found the stream absent from the trained window stack. *Resolve:* (i) a split trigger aimed without key positions (all positions, or the loss's gradient) on the same 40 seeds; (ii) a second task family with a dense loss (the planned session I); (iii) title and conclusion scoped to "grouped-layout binding with a local cue".

**W5. The comparison class is internal.** Every baseline is a recipe from the project's own earlier revisions. The literature section argues that Mixture-of-Memories, Raven and Gated DeltaNet "route or forget at scale" but never runs any of them on the task. Session F's finding that one delta channel binds the two-stream header layout 8/10 shows that a published memory rule can rival the partition on a variant of this task. *Resolve:* at least a Gated DeltaNet single memory and an MoM-style router (per-layer, on the residual stream) on the grouped layout at 2, 4 and 8 streams, same budgets, same seeds.

**W6. Weak secondary claims presented beside the strong one.**
- G1 (four streams, p = 0.0037) is not shown on X alone and does not survive Bonferroni over the project's ~60 claims (0.22).
- G3 at two streams is a count margin that holds 75% of the time at exact equality and 40% of the time if the window recipe is five points worse; its 95% interval is [−7.9, +12.1] points (`statistics.md` §3).
- The "scale-free" rule on ten seeds reads "scale-free" 39% of the time when the rate drops from 0.9 to 0.7.

*Resolve:* report intervals for paired differences; a non-inferiority margin on the rate scale; state which error rate is controlled, and publish the list of ~60 claims.

**W7. "One channel per stream" is an argmax.** In 18 of the 38 headline runs some stream's largest channel holds < 0.9 of its value-position gate mass. At four streams, streams are often spread over several channels (top mass as low as 0.15), though never shared: the largest pairwise overlap is 0.047. One S58 run is routed only through a near-tie. *Resolve:* report the overlap and top-mass distributions; say "disjoint channel sets". Under an overlap criterion G4 is unchanged and G1 strengthens to 16 vs 2.

**W8. Literature.**
- The two references the paper positions itself against were never read: Triadic Linear Attention (the "three-factor form" framing) and BDH-CQ. Neither has a note in `docs/reading/`; BDH-CQ is known only through "the brief".
- The BDH-CQ reconstruction reads "correction rule" as "reads S before writing", but the quoted sentence calls linear attention, which does not, a realisation of such rules.
- "None of twelve papers carries an inferred cue across distance without labels" is contradicted by the project's own HM-RNN note.
- See `literature.md` for the full list. arxiv.org was blocked from my container, so I could not open any abstract.
- *Resolve:* read TLA and BDH-CQ; restate the reconstruction as one of several readings; add "from a sparse end-of-sequence loss" to the "none of twelve" sentence.

**W9. Provenance and bookkeeping.**
- "Branch head" is not a commit; H's and F's heads now contain screens the report does not cite.
- "Most of batch 16 ... on 2.80 GHz hosts": the records say 105 of 165 ran at 2.10 GHz.
- "11 of 1028" single-evaluation binds is the audit's X-only count, not updated (14 of 1869 in `results/` at the report's commit).
- L's `meta.started` is local time (UTC−6 by every consistency check), and no record states its timezone.
- test_muon_recipe has no `specs/` file, although Table 1's caption says every main-line test does.
- *Resolve:* pin every provenance row to a commit; write UTC start times and the CPU into every record.

## 4. The five fixes that matter most, ranked

1. **Correct the false statements (W1) and the convolution argument (W2).** They are cheap, and a reader who finds one will distrust the rest. Cite S68: it strengthens the paper.
2. **Run the copy-only arm as a pre-registered claim (W3).** It is the paper's second mechanistic claim and is currently supported by one uncited ten-seed screen. The spec exists.
3. **Remove the task-structure dependence of the split, or scope every claim to it (W4).** A trigger aimed without key positions on the same seeds would show whether "label-free" survives without the task's annotation.
4. **Add one external baseline family on the same task (W5)**, Gated DeltaNet and a residual-stream router at minimum. Without it, "partitioning memory" is shown to beat earlier versions of itself, not the field.
5. **Fix the statistical reporting of the secondary claims (W6, W7):** paired-difference intervals, a stated error rate over the claim list, a rate-scale non-inferiority margin for G3, and overlap-based routing.

## 5. Questions I could not answer from the repository

1. Did any learned-gate result from X's first start (23:30 UTC, 7 October) exist before the amendment at 00:48? Only the six perfect-gate records survive; the start's log is not in the repository.
2. What were the verification logs of X's and L's runs? CHECKs 135–144 are asserted in the docstring; no log is committed.
3. What does BDH-CQ (2608.09888) actually say in §3.2–3.3, and what is Triadic Linear Attention's gating? Neither was read, and arXiv was blocked from my container.
4. Where does Table 4's "0 of >200" come from? `results/` holds 60 qualifying single-channel runs, all unbound.
5. What is the ~60-claim list behind the multiplicity statement?
6. What was in the "+dirty" working trees of the exploratory runs? G's stores hash every imported module, so G's are verifiable; E's, F's and H's are not.
7. Why do X and L disagree on Muon's four-stream speed (heterogeneity p = 0.070)? Is it the CPU's bf16 Newton–Schulz kernel, or the seed block? Swapping seed blocks between machines would tell.
8. Does the split separate merged streams, or move them? At X 555's third split the merged pair moved together to the new channel. Per-split maps one evaluation later would show which is typical.

## 6. Recommendation

**Major revision.** The central experiment is sound: pre-registered before its runs, reproducible bit for bit, and decisive at eight streams (38/40 against 15/40 without the split, 23 vs 0). I would accept a paper that claimed exactly that, scoped as a grouped-layout result with a split aimed at the task's key positions.

The manuscript claims more. Its title and conclusion generalise to "context-conditional binding". The abstract states a mechanism ("the reset adds nothing") that its cited evidence cannot show, and a fact ("no single-channel run has yet been made") that was false when it was committed. It contains a wrong attribution of a correction (Muon), an argument built on mismatched models (the convolution), a literature claim contradicted by its own notes, and no external baseline.

None of these needs new theory. W1, W2, W7 and the wording of W3, W4 and W8 are edits. The copy-only test and an external baseline are each about a day of the project's compute. With those, I would expect to recommend acceptance.
