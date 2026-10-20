#!/usr/bin/env python
"""
explore_f_s89_tasks.py — EXPLORATORY, not a result. Session F, round 6 (S89-F): the lite task without marked keys. The
task module is explore_f_tasks.py at 50dc8e4 (RandHeaderTaskPool, RandHeaderTaskAllPool; the commit Sessions G and H
vendor). This screen checks the two new tasks for positional solvability, pilots their perfect gates for validity, and
records the baselines G and H will pair against on their own seeds.

REVISION (20 October, after the runs; referee V's review of 8ba99bb at claude/review-V 3ae815b, verdict "fix first"). The
readings, the rules and the phases are unchanged, and the pilots and the post-hoc diagnostic stand as run: all S89 records
are under code SHA ea9728c636c6 (RECORDED_SHA; this module reads them there, and its queue is closed: phase 2 never ran,
since both tasks read "untested"). Taken from the review's last section, in order:
  (1) the routing statistics on ALLPOOL invariant to a per-sequence channel swap (OUTCOMES below; explore_f_tasks'
      appended pool_stats_inv / pool_outcome, with the synthetic-gate CHECK);
  (2) ALLPOOL's routing cap stated (TASKS below) and the type-level perfect gate run as a fifth pilot on 2800-2801,
      descriptive (explore_f_s89_post's O_TYPE_HEBB_ALLPOOL);
  (3) a CHECK tying this screen's own child path to a recorded run, and G's reference read at a commit (CHECKS below).
  The review's other fixes concern phase 2 (the baselines' role, failed records, label-free gates, v2 classes, lite-id
  statistics), which did not run; they are not taken here. Wording corrected: the gradient norms at updates 10 and 100
  are functions of the first 10 and 100 batches, not of the first batch.

CPU: an Intel Xeon @ 2.80GHz (Session F's screens before S88 ran on a Xeon @ 2.10GHz, whose records reproduce bit for bit
here). Every run of this screen is on this CPU. 1 torch thread per run; Session F's harness (explore_f_common).

SEEDS: the assigned block 2800-2839 appears in no record key ("arm|seed|"), no "seed" field and no "seed N" log line in
explore_out/, results/ or records/ on any branch. Phase 0 and phase 1 on 2800-2801; phase 2 on 2802-2811.

TASKS (explore_f_tasks; frequencies over S87's sample, 4096 sequences from torch.Generator(87_000), input tokens; these
numbers were computed before this docstring was written and are printed again by the CHECKs):
  LITE     S70's RandHeaderTask-lite (reference only): CTX 0.114 each (2 types), KEY 0.102 (4), VAL 0.023 (16). The
           largest-ratio cut selects the 2 CTX + 4 KEY types (G's S87 set); the half-mass cut 2 CTX + 3 KEY.
  POOL     RandHeaderTask-pool: keys from a pool of 32: CTX 0.114 (2), VAL 0.023 (16), KEY 0.013 (32). The largest-ratio
           cut selects ONLY THE 2 CTX TYPES (the expectation: confirmed); the half-mass cut 2 CTX + 12 VAL types (the
           expectation that it too selects only the CTX types: NOT confirmed).
  ALLPOOL  RandHeaderTask-allpool: keys from 32, the 2 contexts from 16: VAL 0.023 (16), CTX 0.014 (16), KEY 0.013 (32).
           The largest-ratio cut selects THE 16 VALUE TYPES (ratio 1.46 between the rarest value and the commonest CTX
           type); the half-mass cut 16 VAL + 10 CTX types. Nothing is frequent BY ROLE in the sense of the CTX/KEY roles;
           the values remain the commonest types (8 value tokens per sequence over 16 types). The two-document
           associative-recall layout (MQAR with a context per document) at our vocabulary size; see explore_f_tasks.
  ALLPOOL'S ROUTING CAP (stated in the revision; referee V): no fixed map from context type to channel separates a
           sequence's two contexts in more than 64/120 = 0.533 of sequences (16 types split a / 16 - a over two
           channels: a·(16 - a) of the C(16, 2) = 120 pairs, at most 64). A gate that reads token identity (LOCAL3, the
           register gates) can route the rest only by comparing the contexts within the sequence. The perfect gate is
           handed that comparison (task.perfect_gate, the index by first appearance). So "the task is valid" would mean
           the memory can use a relative routing, not that the arms under test can express one, and a null on ALLPOOL
           does not separate "keys are not frequent" from "contexts have no fixed identity". The type-level perfect gate
           (a fixed balanced map, explore_f_tasks.BALANCED_MAP) is printed as a fifth pilot (explore_f_s89_post).
  Both: every key occurs twice, once under each of the sequence's two contexts (one block of each stream), with two
  distinct values; the query names a (context, key) pair answered by exactly one body pair. On ALLPOOL the stream index
  is by first appearance (stream 0 = the first block's context) and the perfect gate is task.perfect_gate.

PHASE 0, POSITIONAL SOLVABILITY (seeds 2800-2801, each task, LITE as reference): a model with no memory, i.e. the
  embedding, the three-layer stack and a short convolution (K = 4, depthwise, causal; test_short_conv's), one channel, with
  the memory's read removed, in two forms (with the convolution 'layer' and a zero read the convolution would reach nothing,
  since it feeds only the memory's input; so the two forms are):
    NOMEM       test_short_conv's B_conv_in: the convolution on the embeddings, entering the residual stream; the read
                returns zeros ("no channels": each layer adds nothing but its LayerNorm). A position's logits are a
                function of its last 4 tokens.
    NOMEM_SELF  test_short_conv's B_conv: the convolution on each layer's input to the memory; the read returns that
                position's own conv'd write vector V (no position reads any other; the stack's MLP path is kept). A
                position's logits are a function of its last 10 tokens. The stronger of the two.
  Adam 1e-3 on every parameter, 24000 updates, evaluation every 1200 on 2048 held-out sequences, early stop after 3
  evaluations >= 0.95. Printed: the held-out accuracy at the end. RULE (fixed now): a task is SOLVABLE WITHOUT MEMORY if
  either model's final held-out accuracy exceeds 0.6 on either seed; such a task is reported first and STOPPED (no pilot,
  no baseline).
PHASE 1, VALIDITY PILOTS (seeds 2800-2801; each task not stopped): the perfect gate (k = 2, test_channel_binding's
  'ceiling'; on ALLPOOL task.perfect_gate), on the Hebbian memory and on the delta memory (β = 1, L2 keys, tied write,
  decay 0.95; explore_f_delta_fast). S70's recipe with SLOW as the round's prompt states it: every trainable parameter at
  1e-4 for updates 1-2400 and 1e-3 after (the perfect gate has no parameter of its own, so the gate group is empty; S70's
  own oracle used one Adam at 1e-3, which binds sooner, so this is the stricter pilot), 24000 updates, evaluation every
  1200, bound = the first evaluation >= 0.95 that holds to the end (early stop after 3). Printed: the transitions.
  If a task's perfect gate does not bind both seeds on both memories within 24000, both memories are run once more at
  48000 (arms *_48K, the same seeds), and the report says so.
READINGS (fixed now, per task, on the pilots):
  "the task is valid"              if the perfect gate binds 2/2 on Hebbian and 2/2 on delta within 24000;
  "the task needs the longer budget"  if it does so only at 48000 (2/2 on each memory at 48000);
  "untested"                       otherwise (that task is UNTESTED for this round; G and H run the other only);
  "solvable without memory"        if phase 0 stopped it.
  Descriptive, not a reading: which cut selects what (above, and printed again).
PHASE 2, BASELINES (seeds 2802-2811, 10, paired across arms; each task read "valid" or "needs the longer budget", at
  that budget): no reading; the references G and H pair against on their own seeds.
    BASELINE     LOCAL3 + SLOW, Hebbian, k = 2, no conv: S70's arm (explore_f_followup_child's S70_HEBB path on the new
                 task: the window gate on the task's vocabulary, gate W_in, W_g, window at 1e-3 throughout, the rest 1e-4 for
                 updates 1-2400, 1e-3 after).
    SINGLE_HEBB  one Hebbian channel (test_short_conv's arm B on the task's vocabulary, k = 1, no conv), SLOW.
    SINGLE_DELTA one delta channel (the same, delta β = 1, L2 keys), SLOW.
  OUTCOMES (as S70 / S87; ROUTED*@end and ROUTED-BOUND are the branches' exploratory definitions, not a main-line
  outcome; revised, referee V's fix 1): BOUND (a transition); for BASELINE also ROUTED*@end (end margin >= 0.9 and η² by
  stream at KEY positions > 0.9), ROUTED-BOUND = BOUND and ROUTED*@end, and DISCOVERED (bound and end VAL cos < 0.5).
    - On ALLPOOL, where the stream index is a per-sequence convention, both statistics are computed within each sequence
      and averaged over the probe: η²_seq at the 8 key positions, and the cosine between the two streams' mean VAL read
      gates per sequence. explore_f_tasks.pool_routing_stats_inv records them as eta_key_by_stream_inv and val_cos_inv,
      and ROUTED*@end and DISCOVERED read these fields there (explore_f_tasks.pool_outcome(rec, invariant=True)).
    - On POOL both forms are recorded and printed; they agree for a fixed CTX -> channel map.
    - CHECK: on synthetic gates, a context-type gate that separates the streams in a sequence scores η²_seq = 1 and
      cos_seq = 0 whichever channel the first block's context takes, and the pooled forms score 0 and 1 on ALLPOOL
      (explore_f_tasks.check_inv_stats).
  Failure classes of unbound BASELINE runs under v1 and v2. Wilson 95% and bands with every count; median transitions.
CHECKS (asserted before any run; explore_f_delta_checks (26) at the start of every segment; then the child's):
  - RandHeaderTask-lite's batches at seed 2700 equal G's S87 record's first batch: G's S87 records hold no tokens, so
    (i) the first training batch at seed 2700 (onset_run's generator, seed + 10000) from this module equals the one from
    the copy G vendored (explore_f_tasks.py at claude/explore-G e46799a), and (ii) S70's own path (S70_HEBB = G's BASELINE)
    on seed 2700 reproduces G's S87 BASELINE|2700 record bit for bit through 2400: the curve and the gradient norms at
    updates 1, 10 and 100 (functions of the first 1, 10 and 100 batches).
  - (Revision, referee V's fix 3) This screen's own path: explore_f_s89_tasks_child.run(dict(arm="BASELINE_LITE",
    seed=2700, iters=2400)) reproduces G's S87 BASELINE|2700|f205db31dfed|Intel(R) Xeon(R) Processor @ 2.10GHz, read at
    e46799a (G_REF = G_SHA; the record is identical there and was first held by 464f7e4), bit for bit through 2400: the
    curve, the gradient norms at updates 1, 10 and 100, and every statistic this path records at 0, 1200 and 2400 (G's
    record also holds G's own additions, beta and layers, which this path does not compute; listed).
  - (Revision) The revised path (statistics pool_stats_inv) reproduces the S89 records O_HEBB_POOL|2800 and
    O_HEBB_ALLPOOL|2800 (read at 52e2ea7) bit for bit through 2400: the curve and every recorded statistic.
  - The S70 reproduction: S70_DELTA|330 (explore_out/F/rh_lite_results.json, read at 61ba46e) through 2400 bit for bit.
  - explore_f_tasks.py up to the round-6 section is byte-identical to a199d5e's file; check_pool_task on both tasks:
    lite's batch relabelled; the key types distinct and from the pool; every key in one block of each stream, with two
    distinct values; the query answered by exactly one pair (the answer is determined); one CTX per block; the stream
    index; the perfect gate; -pool: no key type above any value type's frequency; -allpool: no type above 1/8 (printed).
  - The frequency tables and both cuts printed for LITE, POOL, ALLPOOL; LITE's equals G's published S87 table.
  - The perfect gates: one-hot, equal to task.perfect_gate; on ALLPOOL the Hebbian model's logits equal those of the
    built-in perfect gate over the 16 CTX types bit for bit (the same same-stream matrix), and the delta memory receives
    the k = 2 gate.
  - The no-memory models (convolution weights randomised for the check: CausalConv starts at the identity): the query
    position's logits do not change when the token 4 (NOMEM) / 10 (NOMEM_SELF) positions back is changed and do change for
    3 / 9 back; NOMEM's read is zero.
  - BASELINE's and the single channels' groups and lrs at updates 1, 2400, 2401 through the real path.
"""
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore_common16 as c16
import explore_f_common as fc

NAME = fc.name("s89_tasks")
CHILD = "explore_f_s89_tasks_child"
MAIN = False
LR = 1e-3
P0_SEEDS, P1_SEEDS, P2_SEEDS = (2800, 2801), (2800, 2801), tuple(range(2802, 2812))
TASKS = ("POOL", "ALLPOOL")
NOMEM_THR = 0.6
DECISION = os.path.join(fc.F_DIR, "s89_decision.json")


def decision():
    return json.load(open(DECISION)) if os.path.exists(DECISION) else {}


def _arms():
    d = decision()
    arms = {}
    for t in ("LITE",) + TASKS:
        for v in ("NOMEM", "NOMEM_SELF"):
            arms[f"{v}_{t}"] = dict(opt="adam", lr=LR, iters=24000, seeds=P0_SEEDS, sched="Adam 1e-3 throughout",
                                    label=f"no memory ({v.lower()}), {t.lower()}")
    for t in TASKS:
        if t not in d.get("phase0", {}) or d["phase0"][t]["stopped"]:
            continue
        for m in ("HEBB", "DELTA"):
            arms[f"O_{m}_{t}"] = dict(opt="adam", lr=LR, iters=24000, seeds=P1_SEEDS, sched="SLOW (rest 1e-4 to 2400)",
                                      label=f"perfect gate, {m.lower()}, {t.lower()}")
            if d.get("pilots", {}).get(t, {}).get("double"):
                arms[f"O_{m}_{t}_48K"] = dict(opt="adam", lr=LR, iters=48000, seeds=P1_SEEDS, sched="SLOW (rest 1e-4 to 2400)",
                                              label=f"perfect gate, {m.lower()}, {t.lower()}, 48000")
        b = d.get("budget", {}).get(t)
        if isinstance(b, int):
            for arm, lab in (("BASELINE", "LOCAL3 + SLOW, Hebbian, k 2"), ("SINGLE_HEBB", "one Hebbian channel, SLOW"),
                             ("SINGLE_DELTA", "one delta channel (beta 1, L2), SLOW")):
                arms[f"{arm}_{t}"] = dict(opt="adam", lr=LR, iters=b, seeds=P2_SEEDS, sched="SLOW", label=f"{lab}, {t.lower()}")
    return arms


ARMS = _arms()


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ARMS[arm]["iters"]))


def timing(arm, steps):
    return child("timing", dict(arm=arm, steps=steps))


PARTS = ("tasks", "g87", "s70", "models", "inv")
RECORDED_SHA = "ea9728c636c6"          # every S89 record; the revision changes the code SHA, not these records


def pin():
    """Read S89's records under RECORDED_SHA (the revised code has another SHA; S89's queue is closed)."""
    c16.STATE["shas"][NAME] = RECORDED_SHA


def check():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(fc.WORKERS) as ex:
        res = list(ex.map(lambda part: child("checks", dict(part=part)), PARTS))
    ok = True
    for rows in res:
        for nm, v in rows:
            print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
            ok &= bool(v)
    return ok


def line(arm, r):
    if not r.get("ok"):
        return f"FAILED {r.get('error')}"
    e = r.get("end") or {}
    rt = f"  margin {c16.fmt2(e.get('margin'))} etaS {c16.fmt2(e.get('eta_key_by_stream'))}" if "margin" in e else ""
    return f"acc {r['acc']:.3f}  trans {str(r['transition']):>5}{rt}"


def _bound(r):
    return bool(r and r.get("ok") and r["transition"] is not None)


def decide():
    """Apply the phase rules to the stored runs and update the decision file (never revises a recorded phase)."""
    me = sys.modules[__name__]
    fc.setup([me])
    pin()
    d = decision()
    runs = lambda arm: fc.runs(me, arm) if arm in ARMS else {}
    if "phase0" not in d:
        p0 = {}
        for t in TASKS:
            rs = {v: runs(f"{v}_{t}") for v in ("NOMEM", "NOMEM_SELF")}
            if not all(len(x) == 2 for x in rs.values()):
                print(f"  phase 0 incomplete for {t}", flush=True)
                return d
            accs = {v: [x[s]["acc"] for s in P0_SEEDS] for v, x in rs.items()}
            p0[t] = dict(acc=accs, stopped=any(a > NOMEM_THR for v in accs.values() for a in v))
        d["phase0"] = p0
    elif "pilots" not in d or any(t not in d["pilots"] for t in TASKS if not d["phase0"][t]["stopped"]):
        pil = d.get("pilots", {})
        for t in TASKS:
            if d["phase0"][t]["stopped"] or t in pil:
                continue
            o = {m: runs(f"O_{m}_{t}") for m in ("HEBB", "DELTA")}
            if not all(len(x) == 2 for x in o.values()):
                print(f"  phase 1 incomplete for {t}", flush=True)
                return d
            both = all(sum(_bound(r) for r in x.values()) == 2 for x in o.values())
            pil[t] = dict(at24=both, double=not both)
        d["pilots"] = pil
    budget = d.get("budget", {})
    for t in TASKS:
        if t in budget:
            continue
        if d.get("phase0", {}).get(t, {}).get("stopped"):
            budget[t] = "STOPPED (solvable without memory)"
        elif t in d.get("pilots", {}):
            if d["pilots"][t]["at24"]:
                budget[t] = 24000
            elif f"O_HEBB_{t}_48K" in ARMS:
                o = {m: runs(f"O_{m}_{t}_48K") for m in ("HEBB", "DELTA")}
                if all(len(x) == 2 for x in o.values()):
                    budget[t] = 48000 if all(sum(_bound(r) for r in x.values()) == 2 for x in o.values()) else "UNTESTED"
    d["budget"] = budget
    json.dump(d, open(DECISION, "w"), indent=1)
    print(f"  decision: {json.dumps(d)}", flush=True)
    return d


def report():
    import explore_f_reports6 as fr
    return fr.report_s89()


if __name__ == "__main__":
    import explore_f_s89_tasks as me
    if len(sys.argv) > 1 and sys.argv[1] == "--decide":
        decide()
    elif len(sys.argv) > 1 and sys.argv[1] == "--check":
        fc.setup([me])
        assert check(), "an S89 CHECK failed"
    else:
        # the queue is closed (revision): S89's pilots stand as run under RECORDED_SHA; the report reads them there
        fc.setup([me])
        pin()
        report()
