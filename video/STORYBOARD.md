# Storyboard: "Partitioning memory" — an explainer of Ynkling/clankers and Revision 7

The video explains the repository and its technical report, *Multi-Channel Hebbian Plasticity in
Multilayer BDH Solves Context-Conditional Binding by Partitioning Memory* (Revision 7, preliminary).
It is built with ManimGL (github.com/3b1b/manim). Narration is synthesized offline (Kokoro, voice `af_heart`).

The narration below is the script. Scene authors keep it **verbatim** (they may split a paragraph
into several `voiceover` blocks at sentence boundaries, and may supply a `spoken=` form for TTS),
and change wording only to fix a factual error, which must be reported. Every number on screen
carries a small source tag (`source_note`) naming the report section/table or the repo file.

Sources: report = `report_v7.pdf` (§ = section). Code facts: `notes/facts_*.md`.

---------------------------------------------------------------------------------------------------

## Global visual language

- Background `BG`; text in `INK`; secondary `MUTED`. Serif (`T`) for statements, sans (`L`) for labels.
- **Streams have fixed colors** `STREAM_COLORS[s]` everywhere (stream 0 blue, 1 yellow, 2 green, 3 red,
  4 purple, 5 orange, 6 teal, 7 pink). A channel that holds stream s is drawn in stream s's color.
- **Concept colors:** gate = `GATE_COLOR` (purple), memory = `MEMORY_COLOR` (teal), hinge = `HINGE_COLOR`
  (orange), slow phase = `SLOW_COLOR` (blue), Muon = `MUON_COLOR` (gold), Adam = `ADAM_COLOR` (grey),
  good = `GOOD`, bad = `BAD`. **Input term** of the gate = `MEMORY_COLOR`-ish teal; **recurrent term** = `GATE_COLOR` purple
  (introduced in Ch. 3, reused in Ch. 19).
- Machines: badges X (blue), L (yellow), E (purple). Every result shown per machine.
- Tokens: `token()` / `triple()` from `common/style.py`: CTX filled with stream color, KEY grey outline,
  VAL outlined in its stream color, query `?` yellow.
- Each chapter opens with `self.chapter_card(n, title)` (about 2 s), then the card fades and a
  `section_title` stays top-left. Keep content inside x ∈ [−6.7, 6.7], y ∈ [−3.7, 3.4] (below the title).
- Pace: something should move at least every ~4 s of narration; never leave a static frame for more
  than ~6 s. Use `vo.wait_until_sentence(i)` to land a visual on the sentence that names it.
- Minimum on-screen text size 20 (labels) / 24 (statements). No overlapping text. Nothing off-frame.

---------------------------------------------------------------------------------------------------

## Ch. 0 — `s00_intro.py` / `Intro` — "Prologue" (~75 s)

**N0.1** Imagine reading two conversations at once, interleaved line by line. In one, the code for the door is seven. In the other, the code for the door is three. Later, someone from the first conversation asks: what was the code?

*Visual:* two chat columns (blue = conversation 0, yellow = conversation 1) whose lines slide into a single
interleaved stream of tokens: `CTX0 door 7`, `CTX1 door 3`, … then the query `CTX0 door ?`.

**N0.2** A memory that simply links "door" to whatever followed it has stored both answers in the same place. The best it can do is guess, and it will be right half the time.

*Visual:* one memory grid; both writes land on the same cells, the blue and yellow fills mix; the
read returns a 50/50 bar "7 or 3".

**N0.3** One fix is obvious. Give the memory separate compartments, and file each conversation in its own. Then the question reads only from the compartment it belongs to, and the conflict is gone.

*Visual:* the grid splits into two channels (blue, yellow); writes re-route; the query reads the blue
channel only → "7".

**N0.4** The hard part is the word "give". Can a network discover, from the task alone, that it should split its memory this way, and which tokens belong in which compartment?

*Visual:* the routing arrows dissolve into a question mark over a small purple "gate" box.

**N0.5** That is the question behind the repository Ynkling slash clankers, and its technical report: Multi-Channel Hebbian Plasticity in Multilayer BDH Solves Context-Conditional Binding by Partitioning Memory. This video covers all of it: the model, the task, the code, the experiments, what works, what still fails, and how the work was checked.
(spoken: "Inkling slash clankers")

*Visual:* title card with the full report title, "A technical report · Revision 7 (preliminary)",
"github.com/Ynkling/clankers". Then a quick roadmap strip of chapter icons.

---------------------------------------------------------------------------------------------------

## Ch. 1 — `s01_bdh.py` / `BDHModel` — "The Dragon Hatchling" (~95 s)

**N1.1** The model underneath is BDH, the Dragon Hatchling, introduced by Pathway in 2025. BDH keeps its working memory in synapses. Each token activates a sparse set of neurons, and a Hebbian rule strengthens the connections between neurons that are active together. Reading the memory means asking which stored patterns overlap with what is active now.

*Visual:* two rows of neuron dots; a token lights a few (sparse); co-active pairs get a thickening synapse
line ("fire together, wire together"). Then the same as a grid σ accumulating outer products
(`S ← S + x_t ⊗ v_t`, from `bdh_recurrent.py`), and a read `x_t · S`.

**N1.2** On a GPU this becomes linear attention. The score between the current position t and an earlier position s is a content match: the dot product of their sparse, positive neuron codes. In this project a positional decay of 0.95 per step weights it, so a memory fades to half its strength in about fourteen tokens.

*Visual:* `M(R"\text{score}(t,s) = (x_t\cdot x_s)\,\times\,0.95^{\,t-s}")`; small plot of 0.95^Δ vs Δ with the
half-life marker at 13.5. Source: `test_instrument_v2.py: decay_mask`, report §16.

**N1.3** The public implementation, bdh dot py, is short. Each layer encodes the residual stream into N neurons and keeps the positive part. It attends with queries tied to keys and no softmax, normalizes the result, passes it through a second sparse code, multiplies the two codes, and decodes back. The same weights serve every layer.

*Visual:* `code_block` of the layer loop (bdh.py:122-144, from facts_model §1.4); highlight lines in step
with the sentences (encoder+relu → attn → ln → encoder_v+relu → product → decoder → residual).
A side flow-diagram may mirror it.

**N1.4** The repository uses that file unmodified, at a tiny size: three layers, two hundred fifty-six neurons, an embedding width of thirty-two, one head. About twenty-eight thousand parameters in all.

*Visual:* spec card: layers 3 · N 256 · d 32 · heads 1 · decay 0.95 · ≈28k parameters. Source: report §3, facts_model §2.11.

---------------------------------------------------------------------------------------------------

## Ch. 2 — `s02_task.py` / `BindingTask` — "Context-conditional binding" (~100 s)

**N2.1** Here is the task the whole project is built on. There are S streams, each announced by its own context token, and P keys. For every key, each stream gets its own value, drawn from sixteen value tokens. So the same key means something different in every stream.

*Visual:* legend: CTX0, CTX1 (stream colors), keys K0–K3, values V0–V15.

**N2.2** The sequence is a list of triples: context, key, value. In the grouped layout, the triples for one key sit together; the order of the keys is random, and so is the order of the streams within each key's group. At the end comes a query: a context token, a key, and a blank to fill in.

*Visual:* build the **real** sequence (facts_task §2.1: the first training sequence of seed 280, slow-start
Part 0's first seed; S=2, P=4), triple by triple, with a brace under each key's group:
`CTX1 K0 V13 | CTX0 K0 V2 | CTX0 K3 V11 | CTX1 K3 V2 | CTX0 K2 V10 | CTX1 K2 V6 | CTX0 K1 V12 | CTX1 K1 V6 || CTX0 K0 ?`
(token ids `[1,2,19, 0,2,8, 0,5,17, 1,5,8, 0,4,16, 1,4,12, 0,3,18, 1,3,12, 0,2,8]`: ctx 0–1, keys 2–5 → K0–K3,
values 6–21 → V0–V15). Answer: V2. There is no query-marker token; "?" is the answer slot (the model reads its
prediction at the query's key position). Tag: "seed 280, first training batch, row 0".

**N2.3** With two streams and four keys that is twenty-seven tokens. Here the query asks for key zero in stream zero. Key zero appears twice: value thirteen in stream one, and value two in stream zero. The answer is value two.

*Visual:* highlight both K0 triples, connect the query to the stream-0 one, reveal V2.

**N2.4** There are three reference levels. Guessing among the values gives one in sixteen. A model that binds keys to values but cannot tell the streams apart has S candidates and scores one over S: here, one half. Only a model that resolves the conflict by stream reaches one.

*Visual:* accuracy number line 0 → 1 with ticks at 1/16, 1/S = 0.5, 1. Labels "guess", "stream-blind", "solved".

**N2.5** That middle level is a trap. A single memory stores both bindings of a key in the same place, and in the basic setting, not one of more than two hundred single-channel runs escaped it.

*Visual:* single grid with superimposed K0 bindings (V13 and V2); tag "one channel: 0 of > 200 bound (S=2, P=4)". Source: report §13, Table 8.
Optional small table: S,P → length: (2,4) 27 · (2,8) 51 · (4,4) 51 · (8,4) 99.

---------------------------------------------------------------------------------------------------

## Ch. 3 — `s03_channels.py` / `ChannelsAndGate` — "Channels and a gate" (~120 s)

**N3.1** The extension gives every synapse k channels instead of one: k separate Hebbian memories that share all the network's weights. Every token also gets a gate, a probability distribution over the k channels. The gate spreads the token's write across the channels and blends its read from them.

*Visual:* channel_stack(2); a token with a gate bar (e.g. 0.8/0.2) writes into both channels with
opacities 0.8 and 0.2; a later token's gate weights the read.

**N3.2** With one distribution doing both jobs, the attention score gains a second factor. The first is the content match from before. The second is the match between the two tokens' gates.

*Visual:* equation (1): `(x_t·x_s) × (g_t·g_s)` with underbraces "content match", "context-gate match".
Code tag: `scores = scores * self.G` and `G = einsum("btk,bsk->bts", gr, gw)` (test_multilayer_binding.py).

**N3.3** If two tokens put their gates on the same channel, that factor is one and they see each other's memory. If they use different channels it is zero, and they are invisible to each other. So tokens that share a context share memory, and tokens from different contexts are isolated. With k equal to one, the factor is always one, and the model is plain BDH.

*Visual:* two gate one-hots → dot = 1 (same) and = 0 (different) computed on screen.

**N3.4** A hand-set perfect gate sends every token of stream s to channel s. With it, the conflict disappears, and the model binds: ten of ten runs in the basic setting.

*Visual:* replay the Ch. 2 sequence; each token's gate one-hot in its stream color; writes land in their
channel; query reads its channel → V2. Tag "perfect gate: 10/10 (S=2, P=4)". Source: Table 8.

**N3.5** The real question is whether a gate can learn this. The learned gate, arm A, is a small recurrent network over the token embeddings, with a thirty-two unit state h. At each step, h is the hyperbolic tangent of two terms: the input term, which looks at the current token, and the recurrent term, which carries the previous state forward. A linear map and a softmax turn h into the gate.

*Visual:* equation (2): `h_t = tanh(W_in v_t + W_h h_{t-1})`, `g_t = softmax(W_g h_t)`; color `W_in v_t` teal
("input term") and `W_h h_{t-1}` purple ("recurrent term"). An unrolled RNN diagram over 4 tokens.
Optional code_block `gate_recurrent` (test_instrument_v2.py:250-258).

**N3.6** Nothing tells this gate which stream a token belongs to. It has to find that out from the task loss alone. Keep its two terms in mind; they come back at the end.

*Visual:* a "no labels" crossed tag; the two colored terms shrink to a corner badge.

---------------------------------------------------------------------------------------------------

## Ch. 4 — `s04_flat_start.py` / `FlatStart` — "A flat start" (~80 s)

**N4.1** Learning that is harder than it looks, and a short calculation shows why. Write each gate as the uniform distribution plus a deviation, delta, whose entries sum to zero. The gate match becomes one over k, plus the dot product of the two deviations.

*Visual:* `g_t = \tfrac1k\mathbf 1 + \delta_t,\ \sum_c\delta_t[c]=0` → `g_t\cdot g_s = \tfrac1k + \delta_t\cdot\delta_s` (eq. 3) via
TransformMatchingTex. A k=3 simplex triangle with the uniform point at the centroid and δ as an arrow.

**N4.2** The constant one over k scales every score alike, and the layer norm after attention cancels it. So routing enters only through products of deviations. The gradient for one token's deviation is proportional to the other tokens' deviations, and at the uniform gate those are all zero. The task loss exerts no first-order pull toward any routing.

*Visual:* strike out 1/k with label "LayerNorm cancels"; `∂/∂δ_t ∝ δ_s` → `= 0 at δ = 0`; a landscape
curve flat at the center.

**N4.3** And the gate does start almost exactly uniform, because the embeddings are initialized small. At the first step, the gradient reaching the gate has a norm below ten to the minus five; the rest of the model's is about 2.6. One more detail: the gate ends in a softmax, so the k rows of its gradient sum to zero, and that gradient has rank at most k minus one.

*Visual:* log-scale bar pair: gate < 1e-5 vs rest ≈ 2.6 (five orders of magnitude). Source: report §1.

**N4.4** So the gate sits on a flat spot, and whatever pushes it off first decides where it goes.

*Visual:* a ball balanced at the flat center, nudged; it rolls toward one of the simplex corners.

---------------------------------------------------------------------------------------------------

## Ch. 5 — `s05_rig.py` / `ConvAndMuon` — "Two more parts" (~95 s)

**N5.1** Two more parts of the setup. Most runs since Phase Four include a short causal convolution, the kind Mamba and most modern linear-attention models place before their token mixer. It is a depthwise filter four tokens wide, applied at every layer and initialized to the identity, so at first it changes nothing. It feeds the queries and the values; the residual stream and the gate's input are not convolved.

*Visual:* `x̃_t = Σ_{j=0}^{3} w_j ⊙ x_{t−j}` (eq. 4); a 4-wide window sliding over tokens; weights shown as
[1, 0, 0, 0] at init. Optional code_block `CausalConv` (test_multilayer_binding.py:225-242).

**N5.2** In this task each context token sits two positions before its value, so the convolution's lag-two weight can carry the context right onto the value's position. That gives one channel a second route to binding, which is why results with the convolution are scored with care.

*Visual:* in a triple CTX·KEY·VAL, a curved arrow from CTX to VAL labelled "lag 2"; the w_2 entry glows.

**N5.3** Second, the optimizer. Before Phase Five every main-line test used Adam. Phase Five adds Muon. For each weight matrix, Muon takes the momentum and replaces it by an approximately orthogonalized version, computed with five Newton–Schulz steps. As a result, the size of an update no longer depends on the size of the gradient.

*Visual:* singular-value bars of a momentum matrix (spread 0.1 … 20); apply σ ↦ aσ + bσ³ + cσ⁵ (a, b, c =
3.4445, −4.775, 2.0315) after normalizing, five times: bars converge into a band near 1. Then "gradient × 10
→ same update" demo. Source: test_early_recipe.py MUON_KW; torch.optim._muon.

**N5.4** Muon handles every matrix, and each head slice of a three-dimensional weight. Adam keeps the embedding, the convolution and the one-dimensional parameters.

*Visual:* parameter chips sorted into two bins: Muon {encoder, encoder_v, decoder, lm_head, W_in, W_h, W_g},
Adam {embed, conv}.

---------------------------------------------------------------------------------------------------

## Ch. 6 — `s06_scoring.py` / `Scoring` — "How a run is scored" (~110 s)

**N6.1** How is a run scored? A model trains at batch thirty-two for a fixed budget, usually twenty-four thousand or twenty-eight thousand eight hundred steps. Every twelve hundred steps it is evaluated on two thousand and forty-eight held-out queries, and it stops early after three evaluations in a row at or above 0.95.

*Visual:* accuracy-vs-step axes with evaluation dots every 1200; a 0.95 dashed line; a real curve if
available from `video/data` (else schematic); three dots above the line → "stop".

**N6.2** A run is bound if an evaluation reaches 0.95 and every later one stays there; the first such step is its transition. With two streams and two channels, a run is discovered if it is bound and the streams' read gates are separated at the value positions. With more channels than streams, a run is bound and routed if every stream has a channel of its own and every stream is at least ninety percent accurate.

*Visual:* three definition cards: BOUND (with transition marker on the curve), DISCOVERED (k = S = 2:
two separated gate bars), BOUND ROUTED (k ≥ S > 2: one-to-one stream→channel map drawn as arrows).

**N6.3** The failures have names. The code looks at the gate's channel choice at key positions, and asks how much of its variance is explained by position, either the triple's index or which half of the sequence it is in, or by the key's identity. That fraction is eta squared. Above one half, the run is a POSITION or a KEY failure.

*Visual:* a "gate map": rows = sequences of a batch, columns = key positions, cell color = channel chosen.
Show (a) stream split (colors follow the stream), (b) POSITION split (left half one channel, right half the other),
(c) KEY split (columns by key). η² meter for each.

**N6.4** With more than two streams, a gate can also put several streams in one channel: a merge. And a run whose final accuracy stays below 0.15 has collapsed. Keep these in mind; most of the story is about which of them a gate falls into.

*Visual:* four streams → channels map with two streams sharing a channel ("MERGED, 2 share"); then
everything into one channel ("collapsed, accuracy < 0.15"). Source: report §3 Outcomes.

---------------------------------------------------------------------------------------------------

## Ch. 7 — `s07_history.py` / `History` — "Phases I to IV" (~110 s)

**N7.1** This report is Revision Seven, and it stands on four earlier phases.

*Visual:* horizontal timeline: Phase I (Rev 1–3) · II (Rev 4) · III (Rev 5) · IV (Rev 6) · V (Rev 7).

**N7.2** In Phase One the gate was a function of token identity alone, and it could not express the routing, because which stream a key belongs to is not a property of the key. The lesson: check that a target is in the model's function class before reading anything into a failure to learn it. In Phase Two a recurrent gate could express it, and it separated two streams.

*Visual:* Phase I card: token → gate (crossed: "K2 belongs to which stream?"). Phase II card: recurrent gate ✓.

**N7.3** Phase Three built the binding task. Two channels with the perfect gate bound two streams where no single-channel model did, and the learned gate found the routing on about half its runs, beating every single-channel reference on both machines.

*Visual:* Phase III card: two-channel ✓ vs one channel ✗; learned gate ≈ ½.

**N7.4** Phase Four ran ten pre-registered tests. A readout from the gate's state into the residual stream harmed the gate, through its gradient. At two streams, more channels did not raise the gate's rate, but a restart rule that reads only held-out accuracy at step twenty-four hundred succeeded in sixty of sixty trials.

*Visual:* Phase IV card, bullet 1 (readout ✗ via gradient: A 41/60 vs A_ro 2/60, README §4), bullet 2 (restart 60/60).

**N7.5** With eight keys per stream, the gate bound from scratch in forty-nine of seventy runs, while one channel with the same fast-weight memory and nearly twice the parameters bound none of twenty-four. The advantage is the partition, not the memory. At four streams with four channels the gate bound only fourteen of forty, mostly failing by merging streams. Spare channels removed most merges. What remained was an early commitment to a split by something other than the stream.

*Visual:* bullets 3–4 with counts (49/70 vs 0/24; 14/40, merges); final line emphasised: "early commitment to
a non-stream split". Source: report §2.

---------------------------------------------------------------------------------------------------

## Ch. 8 — `s08_method.py` / `Method` — "How the work is checked" (~105 s)

**N8.1** Before the new results, a word on how they were produced, because the repository is organized around it. Every confirmatory experiment is one file: test underscore name dot py. Its docstring is written before any run: the background, the arms, the seeds, the claims and their thresholds, the checks, and a runtime projection with rules for what to cut if time runs short.

*Visual:* a document panel titled `test_early_recipe.py` with sections BACKGROUND · OPTIMIZERS · ARMS ·
CLAIMS · DIAGNOSTICS · CHECKS · RUNTIME · OUTPUT · RESULT appearing; zoom on the real CLAIMS block
(E1, E2, bands, readings) as a code_block (language "text").

**N8.2** The test prints its verdicts mechanically, and after the run the result is appended to the same docstring. Before training, each test runs a chain of numbered checks inherited from earlier tests, more than a hundred by now, including reproducing a recorded run bit for bit.

*Visual:* pipeline: docstring → CHECKs 1…128 (counter ticking) → projection → training → verdict line
`E1 ... NOT SHOWN` / `SHOWN` → RESULT appended.

**N8.3** Two machines run every test: X, a cloud container, and L, an Intel laptop processor. They run the same seeds, so every verdict is given per machine, and pooled counts are descriptive, not independent evidence. A third container, E, runs quick exploratory screens on a separate branch, and nothing there counts as a result until a pre-registered test confirms it.

*Visual:* machine badges X, L, E with their CPUs (X: Xeon 2.10/2.80 GHz; L: i7-12650H; E: Xeon, branch
`claude/outside-ideas`). Same seed list feeding X and L.

**N8.4** Most comparisons are paired. A seed fixes the initial weights and the stream of training batches, so two arms on the same seed differ only in the thing under test. The statistic is an exact McNemar test on the discordant seeds, where one arm succeeded and the other failed. When you hear "eleven versus zero", that is what it counts. Results are also put in bands: reliable at ninety percent or more, majority at half, minority for at least one success.

*Visual:* a 2×2 McNemar table (arm 1 ✓/✗ × arm 2 ✓/✗); discordant cells b and c glow; seeds as dots sorted
into the cells. Then a 0–100 % ruler with band thresholds (NEVER 0 · MINORITY ≥1 · MAJORITY ≥50 % · RELIABLE ≥90 %).

---------------------------------------------------------------------------------------------------

## Ch. 9 — `s09_repo.py` / `RepoTour` — "Inside the repository" (~90 s)

**N9.1** Here is the repository itself. bdh dot py is Pathway's model, unmodified. The multi-channel model, MultiBDH, lives in test multilayer binding dot py. It subclasses BDH and replaces only the attention module, multiplying every score by the gate match at every layer. With the convolution on, it runs a copy of BDH's layer loop with the convolution inserted. The gate's own code is imported from the Phase Two instrument.

*Visual:* file tree; highlight bdh.py, test_multilayer_binding.py (MultiBDH, GatedAttention, CausalConv),
test_instrument_v2.py (Instrument.gates, decay_mask, perfect gate).

**N9.2** Then come the experiments, about forty files in the order the project ran them: the Phase One and Two instrument tests, the Phase Three binding tests, the ten Phase Four tests, and the five Phase Five tests this report adds. Each test imports its predecessors, so the chain of checks grows with the project.

*Visual:* test files grouped into phase bands (from facts_repo), Phase V five highlighted:
stream_recipe, stream_curriculum, slow_start, recipe_scope, early_recipe. Import arrows forming a chain.

**N9.3** Results live in results slash X and results slash L: one JSON file per test per machine, and a table of the commit, processor and start time behind each one. Earlier revisions of the report sit beside them.

*Visual:* results/X, results/L folders; results/README.md table row; a JSON snippet showing meta.git, CPU.

**N9.4** Running a test is one command. It runs its checks, prints the projection, then trains on four workers with one thread each. Finished runs are cached, so an interrupted test resumes. Training is deterministic for a given machine and thread count, which is what makes bit-for-bit pairing possible. A flag can pool another machine's results, and tests that pair with earlier runs refuse to pair unless a recorded run reproduces exactly.

*Visual:* terminal: `pip install -r requirements.txt` / `python3 test_stream_recipe.py --workers 4` /
`--also results/X/stream_recipe_results.json`; output lines CHECK ... ok, projection, progress.

---------------------------------------------------------------------------------------------------

## Ch. 10 — `s10_four_streams.py` / `FourStreams` — "Phase V: four streams" (~120 s)

**N10.1** Phase Five adds five pre-registered tests, each run on both machines, and thirteen batches of screens. The first asked about four streams.

*Visual:* Table 1 condensed (test · question · verdict on X · verdict on L) — five rows; highlight row 1.

**N10.2** Revision Six had noticed, after the fact, that at four streams with spare channels, a held-out accuracy of at least 0.4 at step forty-eight hundred had picked the eventual binders with thirty-eight of thirty-eight precision. The stream recipe test made that a restart rule on fresh seeds: four streams, four keys, the convolution; continue a run if it clears 0.4 at step forty-eight hundred, otherwise restart from a new seed, up to five attempts. It ran with sixteen channels, and with four as a control.

*Visual:* flowchart: attempt → step 4800 check "acc ≥ 0.4?" → yes: continue · no: new seed (≤ 5 attempts).

**N10.3** With sixteen channels and restarts, seventeen of twenty bound on X and nineteen of twenty on L. Without restarts, twelve and nine. With four channels, restarts barely helped: ten and eight, against ten and five.

*Visual:* Table 2 as paired bars (X, L): A4k16 12, 9 · A4k16_R 17, 19 · A4k4 10, 5 · A4k4_R 10, 8 · perfect gate 3/3, 3/3.

**N10.4** The reason is merges. If two of the four streams share a channel, those two are each right half the time, and the run's accuracy sits near three quarters. Three streams in one channel gives about one half. Both clear the 0.4 bar, and then stall. With four channels, every attempt that passed the check and then failed was a merge: nine of nine on X, twelve of twelve on L.

*Visual:* the arithmetic: (1 + 1 + ½ + ½)/4 = 0.75; (1 + ⅓·3)/4 = 0.5, shown with four stream chips over
channels; then accuracy axis with plateau lines at 0.75 and 0.5 above the 0.4 check line.

**N10.5** So the check works only with spare channels. The recipe was reliable on L and a majority on X, and spare channels making restarts work was shown on both machines. L also tried eight streams: the perfect gate bound both its runs, and the restart arm none of five.

*Visual:* verdict badges R1 (X MAJORITY, L RELIABLE), R2 (X NOT PRECISE, L PRECISE), R3 SHOWN both, R4 NOT SHOWN
both; eight-stream footnote 2/2 vs 0/5. Source: report §4, Table 2.

---------------------------------------------------------------------------------------------------

## Ch. 11 — `s11_curriculum.py` / `StreamCurriculum` — "Eight streams by curriculum" (~70 s)

**N11.1** From scratch, eight streams with sixteen channels did not bind. So the stream curriculum test trained on two of the eight streams per sequence for the first forty-eight hundred updates, then four, then all eight. The perfect gate bound on both machines, so the setup was valid.

*Visual:* three-stage timeline (2 → 4 → 8 streams per sequence; updates 1–4800, 4801–9600, then 8).

**N11.2** The curriculum bound six of sixteen runs on X and four of twenty on L; training on all eight from the start bound none, zero of six and zero of ten. With so few runs, neither difference was significant, so "the curriculum beats training from scratch" was not shown. The scratch runs mostly collapsed below 0.15. The curriculum's binders appeared soon after the switch to eight streams, but several of them bound without one channel per stream, and its failures were collapses and merges.

*Visual:* bars SC8 6/16 (X), 4/20 (L) vs D8 0/6, 0/10; verdicts C1 NOT SHOWN both, C2 MINORITY both, C3 UNTESTABLE.
Source: report §5.

---------------------------------------------------------------------------------------------------

## Ch. 12 — `s12_screens.py` / `Screens` — "The race" (~120 s)

**N12.1** Where did the new recipe come from? From screens: cheap exploratory runs, mostly at two streams and four keys, each paired by seed with a recorded run. A screen that looks promising goes back to the main line for a proper test.

*Visual:* container E badge; screen cards S4 … S26 fanning out; one card "promising → main line".

**N12.2** Two observations shaped everything. First, routing at step twelve hundred predicts routing at the end: the outcome is decided early. Second, early training looks like a race. The gate is searching for the stream split. Meanwhile the memory is learning too, and it can start exploiting some other split, early positions against late ones, which already relieves part of the key confusion. Whichever settles first wins.

*Visual:* **the race**: two runners on parallel tracks — purple "gate finds the stream split" and teal
"memory exploits a position split" — with a finish line at "commitment". The teal runner usually wins.

**N12.3** So, slow the memory down. For the first twenty-four hundred updates, everything except the gate trains at a tenth of the learning rate. Over forty seeds, that lifted discovery from twelve to twenty-four, and twenty-three runs routed at step twelve hundred instead of eleven.

*Visual:* the teal runner slows (lr/10 tag); purple wins more often. Bars: A 12/40 → slow memory 24/40 (15 vs 3).

**N12.4** The failures that remained were mostly position splits, so the next screen went after exactly those, with a hinge penalty. It removed them. On the same seeds, it discovered on ten where slow memory alone did not, and on none where slow memory won.

*Visual:* Table 3 as bars: A 12/40 · kWTA 7/20 · slow 24/40 · gate reset 10/20 · position penalty 15/20 ·
hinge 34/40 (vs slow 24/40, 10 vs 0) · random kick 12/20 · untimed kick 27/40 · gate noise 27/40. Hinge bar in orange.

**N12.5** And the hinge acts rarely. On twenty of forty seeds it never fired. Where it did, it usually fired on one to three of the roughly five thousand batches before binding. A random push of the same size, on the same batches, did most of what the hinge did. A single random push at a fixed update did not. Timing appears to do most of the work.

*Visual:* a 0 → 5000 batch timeline per seed (a few rows), with sparse orange ticks early; comparison chips
"random push, same batches: 12/20 vs hinge 14/20" and "one push at update 120: 27/40 vs 34/40 (0 vs 7)".
Source: report §6, Table 3.

---------------------------------------------------------------------------------------------------

## Ch. 13 — `s13_recipe.py` / `Recipe` — "The recipe" (~100 s)

**N13.1** Here is the recipe precisely. SLOW is one Adam optimizer with two groups. The gate's three weight matrices train at ten to the minus three throughout. Every other parameter, the embedding and the convolution included, trains at ten to the minus four for updates one to twenty-four hundred, and at ten to the minus three after that.

*Visual:* log-lr vs update plot: gate line flat at 1e-3; "everything else" at 1e-4 until 2400 then 1e-3.
code_block of `param_groups` (test_slow_start.py:418-423) optional.

**N13.2** HINGE adds a penalty. On each training batch, take the read gate's channel probabilities at the key positions. Ask how much of their variance is explained by the triple's index, and how much by which half of the sequence the key is in. Those are two eta-squared values between zero and one.

*Visual:* batch matrix (rows = sequences, columns = 8 key positions) colored by p(channel 0); group means
per column (index) and per half; "η² = between-group variance / total variance" (pooled over channels).

**N13.3** The penalty is the part of each value above 0.2, added together. It is zero while the gate's choice has little to do with position, and it fires when the gate starts splitting by position. Its gradient reaches only the gate and the embedding.

*Visual:* `\text{HINGE} = 1.0\times[\mathrm{relu}(\eta^2_{\text{index}}-0.2)+\mathrm{relu}(\eta^2_{\text{half}}-0.2)]`;
relu plot with the 0.2 kink. Show the two example gates from facts_model §5.2: position split η² = 1.0/1.0 →
penalty 1.6 (fires); random split η² = 0.03/0.003 → 0.

**N13.4** Notice what it uses and what it doesn't. It uses no stream labels; it never says which channel any stream should go to. But it does need to know where the keys are, and that is task structure.

*Visual:* two tags: "no stream labels ✓", "needs key positions !".

**N13.5** Two variants. WINDOW applies the hinge only during updates one to twenty-four hundred. MUON runs the same two-speed schedule under Muon, at a learning rate of 0.005.

*Visual:* hinge weight step function (1 until 2400, then 0); Muon schedule: gate 0.005 throughout, other
matrices ×0.1 until 2400, Adam groups 1e-4 → 1e-3. Source: report §3 Recipes; test_early_recipe.py.

---------------------------------------------------------------------------------------------------

## Ch. 14 — `s14_slow_start.py` / `SlowStart` — "Slow memory and a hinge, confirmed" (~120 s)

**N14.1** The slow start test took the recipe to fresh seeds and to the working configurations, with no restarts anywhere.

*Visual:* Table 4 layout appears (parts 0, A, B, C).

**N14.2** At two streams and four keys, the plain gate discovered the routing on eight of twenty seeds on each machine. Slow memory alone, fifteen of twenty on each. Slow memory plus the hinge, nineteen of twenty on X, twenty of twenty on L.

*Visual:* Part 0 grouped bars (X, L): A0 8, 8 · SLOW0 15, 15 · HINGE0 19, 20.

**N14.3** The recipe beat the plain gate on both machines, eleven to zero and twelve to zero in discordant seeds, and slow memory alone beat it on both as well. The hinge's gain over slow memory alone was shown on L, five to zero. On X it was four to zero, a p-value of 0.0625, just short.

*Visual:* verdict table: H0 SHOWN (11 vs 0; 12 vs 0) · S0 SHOWN (9 vs 2; 10 vs 3) · H0S NOT SHOWN on X (4 vs 0,
p = 0.0625) / SHOWN on L (5 vs 0, p = 0.031).

**N14.4** At eight keys per stream, with the convolution, the recipe bound twenty and seventeen of twenty, against thirteen and twelve for the recorded plain runs, shown on both machines. At four streams with sixteen channels, it bound seventeen and eighteen of twenty, without restarts, against twelve and nine: shown on L; on X it was seven to two, not significant.

*Visual:* Part A bars DIRECT8 13, 12 vs HINGE8 20, 17 (HA SHOWN both); Part B A4k16 12, 9 vs HINGE4k16 17, 18
(HB NOT SHOWN on X p = 0.09, SHOWN on L p = 0.011).

**N14.5** Look at how the failures change. The plain gate failed mostly by position splits; slow memory alone failed almost only by position splits; the hinge removes those. And it fired rarely: never on eleven of X's twenty two-stream runs. At eight streams, though, nothing bound, zero of ten on each machine, and four of ten collapsed.

*Visual:* failure composition chips (A: 9 of 12 failures position on X; SLOW: 4 of 5 position); hinge firings
count strip; Part C 0/10, 0/10 in red. Source: report §7, Table 4.

---------------------------------------------------------------------------------------------------

## Ch. 15 — `s15_recipe_scope.py` / `RecipeScope` — "Which part matters" (~75 s)

**N15.1** Which part of the recipe does the work? The recipe scope test removed the slow phase and kept the hinge, at one learning rate throughout. At two streams the position splits still disappeared, but key splits took their place: five of eight failures on X and five of seven on L were key splits. Slowing the memory prevents those.

*Visual:* failure-type morph: POSITION split gate map → KEY split gate map; HONLY0 12/20, 13/20.

**N15.2** The full recipe beat the hinge alone on both machines, eight to one and seven to zero. The pre-registered reading: both parts are needed.

*Visual:* Q2 SHOWN both (8 vs 1, p = 0.020; 7 vs 0, p = 0.008); quote card "both parts are needed".

**N15.3** With sixteen channels at four streams, the hinge alone did about as well as the full recipe. The slow phase may matter only when channels are few; that is an observation, not a tested claim. On the eight-stream curriculum, the hinge removed the collapses, but merges took their place: on X, nine of sixteen gates held all eight streams in just two channels of four.

*Visual:* HONLY4k16 16/20, 18/20 vs HINGE4k16 17, 18 (Q4 NOT SHOWN); SC8_H 8/16, 7/20; eight stream chips
packed 4 + 4 into two channels. Source: report §8, Table 5.

---------------------------------------------------------------------------------------------------

## Ch. 16 — `s16_kick.py` / `EarlyKick` — "The hinge is an early kick" (~110 s)

**N16.1** The screens then asked what the hinge actually does, and whether the recipe survives a different optimizer. Under Muon, slow memory plus the hinge discovered twenty of twenty, all routed by step twelve hundred. Without either, nine of twenty, and all eleven failures were position splits. With slow memory but no hinge, thirty-one of forty; adding the hinge made it thirty-nine, eight seeds rescued and none lost. On every rescued seed, the hinge first fired between updates fifty and one hundred sixty-eight.

*Visual:* Muon chips: S25 20/20 vs 9/20; S32 31/40 → 39/40 (8 vs 0); first-firing range 50–168 marked on an
update axis.

**N16.2** Here is what a firing is. At that moment the hinge's gradient on the gate was a median of eight hundred fifty times the task gradient under Muon, and seven hundred twenty-six times under Adam.

*Visual:* two arrows on a log scale: task gradient (×1) vs hinge gradient (×850 Muon, ×726 Adam).

**N16.3** Under Muon, the gate's momentum then stayed aligned with that push for a median of fifty-nine updates, and since Muon's step size does not depend on the gradient's size, the gate keeps moving in the hinge's direction the whole time. Under Adam, a gradient far above its running average produces a sign-like, full-size step. Either way: a large, directional push, delivered early, while the gate is still near its flat uniform start.

*Visual:* the simplex/flat-landscape ball from Ch. 4 receives a big orange kick and travels; a cosine trace
staying above 0.5 for ~59 updates under Muon. Callback to Ch. 4's "whatever pushes it off first".

**N16.4** That suggests the hinge is only needed early. With the hinge switched off after update twenty-four hundred, Muon discovered thirty-eight of forty, against thirty-nine with the full hinge, and Adam thirty-four against thirty-four. One loss in eighty runs. And at four streams with sixteen channels, Muon with the recipe bound seventeen of twenty, fifteen of them by update twenty-four hundred.

*Visual:* S34 window vs full: Muon 38/40 vs 39/40, Adam 34/40 vs 34/40; S35 17/20 (15 by 2400). Source: report §9.1.

---------------------------------------------------------------------------------------------------

## Ch. 17 — `s17_early_recipe.py` / `EarlyRecipeTest` — "The early-recipe test" (~95 s)

**N17.1** The early recipe test checked both findings on fresh seeds. At two streams, Muon with slow memory alone discovered thirty-six of forty on X and thirty-four on L. Adding the early hinge window made it forty of forty, on both.

*Visual:* Table 6 part A bars: SLOW_M 36/40, 34/40 · WIN_M 40/40, 40/40.

**N17.2** At four streams with sixteen channels, it compared the window under Adam with the window under Muon, scored by whether a run had bound by update forty-eight hundred. On L both claims were shown: the early window works under Muon, six to zero, and Muon binds four streams sooner, seven to zero. On X both pointed the same way without reaching significance: four to zero, and five to one. Under Muon the four-stream binders bound at a median of update twenty-four hundred on both machines, against forty-eight hundred under Adam.

*Visual:* Part B: WIN16_A 10/12 (6 by 4800) X, 17/20 (12) L; WIN16_M 11/12 (10) X, 19/20 (19) L. Transition
strip plot from data if available; median markers 2400 (Muon) vs 4800 (Adam). Badges E1, E2 per machine.

**N17.3** Why do the machines differ? X's fresh baseline was higher than in the screens, leaving less room, and its time rule cut the four-stream part to twelve seeds. And the two machines are not independent replicates. They ran the same seeds, so most seed-level outcomes agree, yet no run reproduced across machines, and Muon's low-precision Newton–Schulz step even differs between X's two Xeon hosts.

*Visual:* two seed columns (X, L) with mostly matching ✓/✗; a "≠ bits" tag; bf16 chip. Source: report §9.2, Table 6.

---------------------------------------------------------------------------------------------------

## Ch. 18 — `s18_merges.py` / `Merges` — "Breaking merges" (~90 s)

**N18.1** At four streams with only four channels, the failures are merges, and spare channels are the main fix. Can a merge, once formed, be broken? Screens took eight recorded four-channel runs that were merged at update ninety-six hundred and continued each one.

*Visual:* 4 streams over 4 channels: two share channel 1, channel 3 idle.

**N18.2** Repeated large pushes on the gate split none of eight. A router z-loss split two: it did lower the logits' scale, but by lowering them all together, along the one direction the softmax ignores.

*Visual:* S21 0/8; S27 2/8; logits bar group shifting down uniformly, softmax output unchanged.

**N18.3** What worked was surgery: copy the busiest channel's gate row onto the idlest one, add a little noise, and reset that weight's optimizer state. Eight of eight split, against one of eight for the noise and the reset alone. The merged streams' gate states were nearly identical, but once the idle channel was level with the shared one, small differences could grow.

*Visual:* W_g rows as colored strips; row of channel 1 copied onto row of channel 3 (+ε); the two merged
streams then diverge to separate channels. S24 8/8 vs 1/8.

**N18.4** Triggering the copy without labels, when accuracy stalls, worked less well, because the rule aimed at the wrong channel about half the time. Aiming by gate mass at key positions fixed the aim under Muon, ten of eleven on target, and bound six of ten with one channel per stream, against three without splitting. Why a split sometimes fails even on target is still open.

*Visual:* S36 5/10 vs 4/10 (aim wrong ~½ of 31 splits); S37 Muon 10/11 on target vs 5/11; bound 6/10 vs 3/10.
Source: report §10.

---------------------------------------------------------------------------------------------------

## Ch. 19 — `s19_eight_streams.py` / `EightStreams` — "The gate forgets the context" (~170 s)

**N19.1** Eight streams. Every learned-gate arm trained on all eight from the start has failed: under Adam and Muon, with and without the hinge, with splits on plateaus, with restarts. None of more than a hundred runs bound, while the perfect gate binds. So instead of another fix, the screens probed the mechanism.

*Visual:* a grid of ~100 small red ✗ cells vs a green ✓ "perfect gate".

**N19.2** Does the gate's state even carry the stream? A simple decoder, reading the gate's hidden state at key positions, recovered the stream at 0.13 under Muon and 0.14 under Adam, against a chance level of 0.125. At four streams the same decoder read 0.84, or 1.00 with sixteen channels. That is striking, because each key's stream token sits just one position earlier. The gate needs one token of memory, and it doesn't have it.

*Visual:* bars: S=8 Muon 0.13, Adam 0.14 vs chance 0.125 line; S=4 0.84 (k=4), 1.00 (k=16). A key token with an arrow
to the CTX token one step back: "1 token of memory needed".

**N19.3** When is it lost? At initialization the decoder reads 0.91: with small weights, the state still carries the previous token. Training removes this memory within the first six hundred to twelve hundred updates. The loss coincides with the recurrent term overtaking the input term. Under Muon the recurrent term passes the input term by update four hundred, and by update twenty-four hundred it is near seven, while the input term stays around 0.1.

*Visual:* **Table 7 as line charts** (real numbers, updates 0, 200, 400, 600, 1200, 2400):
decodability Muon S8 0.91 0.70 0.39 0.18 0.15 0.13; Adam S8 0.91 0.88 0.79 0.50 0.24 0.14; Muon S4 0.95 0.84 0.62 0.61 0.85 0.83;
recurrent term (log y) Muon S8 0.04 0.08 0.20 1.91 4.02 6.93; Adam S8 0.04 0.05 0.06 0.14 0.32 5.08; Muon S4 0.04 0.10 0.49 0.58 1.67 2.27;
input term Muon S8 0.06 0.07 0.10 0.11 0.15 0.13. Colors: recurrent = purple, input = teal (as Ch. 3).

**N19.4** It is mostly not saturation of the tanh: at update twenty-four hundred, the median share of saturated units is at most fifteen percent. At four streams, by contrast, the input term grows and the memory survives. At eight streams the gate's output stays nearly uniform, so the task gives its input weights little reason to grow.

*Visual:* S=4 curve stays high; entropy gauge 2.70 of max 2.77 (uniform over 16).

**N19.5** Here is the picture. The gate's state is a small dynamical system driven by its input. If the recurrent weights contract, the state forgets its past and follows the current token, so every stream token leaves a recognizable mark. If they expand, the state sustains itself and ignores its input.

*Visual:* **phase-plane animation**: 2-D state h; input pulses colored by stream. Contracting W_h (ρ < 1):
the state jumps to a stream-specific region each pulse (colored clusters, decodable). Expanding W_h (ρ > 1):
the state runs along its own orbit/attractor regardless of pulses (colors mixed). Show ρ on a dial.

**N19.6** A later screen measured exactly this: the spectral radius of the recurrent weights, every fifty updates. It started between 0.52 and 0.63. In seven of ten runs it rose clearly above one, and in each of those the stream's decodability fell to between 0.13 and 0.26 within fifty updates of the crossing. Where it peaked near one, the stream was only partly lost; where it fell back below one, decodability partly recovered. Ten runs give a correlation, not a proof.

*Visual:* ρ(t) schematic crossing 1 (dashed) between updates 350 and 1400, decodability dropping right after;
counts 7/10 crossed (peaks 1.20–2.13), 3/10 peaked 0.88–1.03; 2 recovered (0.50, 0.69). Source: report §11 (S42).

**N19.7** Feeding the gate the previous token directly did not solve it either: its input then decodes the stream almost perfectly, but the recurrent term dilutes it, and no run bound. The intervention now running is a cap on the recurrent weights' spectral norm.

*Visual:* S40 chip (input decodes 0.97–1.00, state 0.53–0.64, 0/10); S41 "cap ‖W_h‖ ≤ 0.5 — running".

---------------------------------------------------------------------------------------------------

## Ch. 20 — `s20_distant_cues.py` / `DistantCues` — "Distant cues" (~60 s)

**N20.1** In a language model, the cues that set a context are usually far from the tokens they govern. The header layout tests that: each stream's context token appears once, at the start of its block. The plain gate discovered the routing on none of ten, while the perfect gate bound three of three.

*Visual:* header layout: `CTX0 K V K V K V … | CTX1 K V K V …`; distance arrows growing.

**N20.2** But a screen found a deeper problem. A single channel can bind the header layout too, three of ten, and seven of ten with slow memory. So binding there is no evidence that a gate routes. What is still needed is a distant-cue task that one channel provably cannot solve.

*Visual:* single channel 3/10, 7/10 (slow) with a "binding ≠ routing" stamp. Source: report §12.

---------------------------------------------------------------------------------------------------

## Ch. 21 — `s21_standing.py` / `WhereItStands` — "Where the mechanism stands" (~90 s)

**N21.1** Here is where things stand, counted over both machines. Given the partition, every grouped-layout configuration tried binds, and one channel with the same memory does not. The partition does what one channel cannot.

*Visual:* Table 8 (setting · perfect gate · plain gate · recipe, no restarts · restarts · one channel), built row by row.

**N21.2** Discovery is decided in the first few hundred updates, and it can be steered. Slowing the memory shifts the race toward the stream split, and a hinge that fires when the gate's choice turns positional delivers a few large, well-timed pushes. Together they bind without restarts at two streams, and at four streams with sixteen channels. The recipe's work is done by update twenty-four hundred.

*Visual:* highlight the "recipe, no restarts" column: HINGE 39/40, Muon+WINDOW 80/80; HINGE 37/40 (P=8); HINGE 35/40,
hinge only 34/40, Muon+WINDOW 30/32 (S=4, k=16).

**N21.3** At eight streams the obstacle moves upstream. Before any routing can form, the gate's recurrent state stops reflecting its input. No change to the gate's readout, whether a hinge, a split, or an optimizer, can route by a stream its state does not carry. And merges remain at four channels: spare channels prevent most of them, and a targeted copy breaks some.

*Visual:* S=8 row: HINGE 0/20; curriculum 15/36 (descriptive). Four claim cards summarizing. Source: report §13, Table 8.

---------------------------------------------------------------------------------------------------

## Ch. 22 — `s22_retrospective.py` / `Retrospective` — "Keeping score on itself" (~110 s)

**N22.1** The report also grades its own earlier claims. The title claim stands. "A learned gate finds the partition, reliably with restarts, at two streams; four are the open problem": superseded, since the recipe now binds two streams, and four with sixteen channels, without restarts. The early check at four streams: partly confirmed, precise on L but not on X. "It depends on what the memory learns first": supported by a pre-registered test. Restarts as the working recipe: replaced. And Revision Six's pooled p-values across the machines: corrected, because both machines ran the same seeds.

*Visual:* claim list with badges STANDS · SUPERSEDED · PARTLY CONFIRMED · SUPPORTED · REPLACED · CORRECTED.

**N22.2** It lists its own errors in this phase too: calling the two machines independent replicates; reading the hinge's rarity as a sign it wasn't needed, when a few well-timed firings were the mechanism; a merge breaker claimed before its trigger existed, then specified wrongly; a confirmation planned from the screens' numbers that met a higher baseline; machine differences blamed on Muon's kernels alone; specifications lost between sessions; a validity arm without the budget to bind; and process slips, caught by the tests' own pairing checks.

*Visual:* six error cards with short titles (from report §14.2).

**N22.3** Five lessons come out of it. Give each machine its own seeds when you want independent confirmation. Bit-for-bit determinism depends on the processor and its kernels, so record the kernel path. Probe the mechanism before designing the fix. Score routing, not binding, wherever the memory has another route. And power a confirmation for the baseline it will actually meet.

*Visual:* five numbered lesson cards. Source: report §14–15.

---------------------------------------------------------------------------------------------------

## Ch. 23 — `s23_outlook.py` / `Outlook` — "Limits and next steps" (~140 s)

**N23.1** The limitations are stated plainly. The task is synthetic, with explicit context tokens, and in the grouped layout each key's stream token is the previous token. The hinge needs the key positions, and a language model has no marked keys. The model is tiny: three layers, one head, sequences of at most ninety-nine tokens, a memory half-life of about fourteen tokens. Eight streams have no recipe. The second machine replicates the procedure, not independent samples. And screens are screens.

*Visual:* limitation tiles (task · recipe needs structure · model · streams · replication · screens).

**N23.2** How far is this from a language model? Phase Five removes per-run restarts, and the recipe works under Muon, an optimizer already used to pretrain large language models. What stands in the way, roughly in order of risk: many contexts, where the gate's recurrence needs redesign or constraint; the hinge's nuisance variables, which need a generic version; context cues at a distance; the gate's token-by-token compute; the memory horizon; and prior art, since Mixture-of-Memories already routes tokens among linear-attention memories at the billion-parameter scale.

*Visual:* risk ladder 1–6 (report §17).

**N23.3** The plan is staged. On CPUs: eight streams first, then a distant-cue task that one channel provably cannot solve. Then a GPU port with Muon and a parallel or contractive gate, on the multi-query associative recall benchmark, against linear-attention baselines including a Mixture-of-Memories router. Then small language models, and then scaling. The first two stages are cheap, and they are where the idea is most likely to fail.

*Visual:* four-stage path (i) CPU → (ii) GPU + MQAR → (iii) small LMs → (iv) scaling; stages i–ii flagged "cheap; most likely to fail".

**N23.4** To sum up. Multi-channel Hebbian plasticity in multilayer BDH solves context-conditional binding by partitioning memory, and a label-free gate can now find that partition without restarts, at two streams and at four with sixteen channels. Two changes do it, both confined to the first twenty-four hundred updates: slow the memory, and push the gate away from positional splits the moment it starts to make them. At eight streams the obstacle comes earlier still: the gate's recurrent state loses the context before any routing forms.

*Visual:* recap montage: channels with perfect routing; SLOW + HINGE icons; ρ > 1 warning.

**N23.5** Everything you saw is in the repository: the model, every test with its pre-registered design, and every result file. Thanks for watching.
(spoken: "Everything you saw is in the repository, Inkling slash clankers: the model, every test with its pre-registered design, and every result file. Thanks for watching.")

*Visual:* end card: "github.com/Ynkling/clankers", report title, "Made with ManimGL (3b1b/manim) · narration: Kokoro TTS".
