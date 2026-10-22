#!/usr/bin/env python
"""
explore_f_s90_ctx.py — EXPLORATORY, not a result. Session F, round 7 (S90-F): the context-pool family. Round 6 showed that
on RandHeaderTask-allpool the contexts' missing identity, not the keys, is what no learned gate crossed; -allpool changes
the keys' frequency and the contexts' identity together (V's screen review of S89-F, fix 2; G's caveat). This screen
validates two tasks that change only the contexts, and asks whether a single channel binds documents without any routing
when the contexts are not fixed types. Task module: explore_f_tasks.py at 9017b82 (RandHeaderTaskCtxPool,
RandHeaderTaskDocs; appended, b0c6332's file unchanged above; the commit G vendors).

CPU and harness: Session F's (explore_f_common), 1 torch thread per run, the CPU and the code SHA in every record key.
This container has alternated between a Xeon @ 2.10GHz and a Xeon @ 2.80GHz between sessions. Records from another CPU
are accepted for a stage only after one of them reproduces bit for bit on the current CPU through 2400 (check()); the
report prints every run's CPU, and pairs are bit for bit only within one CPU.

SEEDS: the assigned 3060-3071 appear in no record key ("arm|seed|"), no "seed" field and no "seed N" log line in
explore_out/, results/, records/ or specs/ on any of the 13 branches (scanned 22 October). Phases 0 and 1 on 3060-3061;
phase 2 on 3062-3071, paired across arms.

TASKS (explore_f_tasks; S87's sample, 4096 sequences from torch.Generator(87_000); printed by the CHECKs):
  CTXPOOL  RandHeaderTask-ctxpool: lite (S = 2, P = 4, NB = 2 blocks of 1-3 pairs, one query, L = 23) with lite's 4 key
           types kept and the 2 contexts of each sequence drawn from 16 types. KEY 0.102 each (4 types), VAL 0.022-0.024
           (16), CTX 0.013-0.015 (16). The largest-ratio cut selects the 4 KEY types; the half-mass cut the 4 KEY + 4 VAL
           types (mass 0.502). The batch is lite's with only the contexts renamed.
  DOCS3, DOCS4  RandHeaderTask-docs at D = 3 / 4 documents (S = D streams, L = 33 / 43): the same, D contexts from the 16.
           KEY 0.101-0.102, VAL 0.023-0.024, CTX 0.013-0.015; the same cuts as CTXPOOL (4 KEY; 4 KEY + 4 VAL).
  Stream index: by first appearance (stream 0 = the first block's context), any S; the perfect gate is its one-hot.
  THE CAP (printed): a fixed type -> channel map at k = D channels separates a sequence's D contexts in at most
    D = 2: 64/120 = 0.533; D = 3: 150/560 = 0.268; D = 4: 256/1820 = 0.141 of sequences (explore_f_tasks.type_map_cap,
    exact; the balanced split); at k = 2D: 0.800 / 0.664 / 0.615; at k = 16: 1. A gate that reads context identity at
    k = D can route the rest only by comparing contexts within the sequence; the perfect gate is handed that comparison.

PHASE 0, NO-MEMORY SOLVABILITY (3060-3061, each task): S89-F's NOMEM (conv on the embeddings, zero read; reach 4) and
  NOMEM_SELF (conv into the memory, own-V read; reach 10), Adam 1e-3, 24000. RULE (fixed now): a task is SOLVABLE WITHOUT
  MEMORY if either model's final held-out accuracy exceeds 0.6 on either seed; it is then reported first and STOPPED.
  Printed with it: P(the target pair, its KEY and VAL, inside the reach) and the approximate positional bound
  P + (1 - P)/(n_vals - S + 1) (V's fix 5; the rule is expected not to fire, so "not stopped" is no evidence beyond that
  bound).
PHASE 1, VALIDITY PILOTS (3060-3061; each task not stopped), S89-F's rules: the perfect gate handed the per-sequence index
  (k = S; task.perfect_gate), on two memories:
    HEBBCONV   Hebbian + the main line's conv 'layer' (K = 4) in the memory's input (round 6's valid Hebbian
               configuration);
    DELTA      the delta memory (beta 1, L2 keys, tied write, decay 0.95; explore_f_delta_fast), no conv;
  SLOW (every parameter at 1e-4 for updates 1-2400, 1e-3 after; the perfect gate has no parameter), 24000 updates,
  evaluation every 1200 on 2048 held-out sequences, bound = the first evaluation >= 0.95 that holds to the end, early stop
  after 3. FALLBACK (per task and memory): a memory that does not bind 2/2 within 24000 is run once more at 48000 on the
  same seeds (O_<M>_<T>_48K); a memory that bound 2/2 at 24000 is not rerun.
  Also, DESCRIPTIVE (no reading): the type-level perfect gate (a fixed balanced map of the 16 types to k = S channels,
  explore_f_tasks.type_level_gate / balanced_map) on both memories, 24000 (TL_<M>_<T>).
READINGS (fixed now, per task T and memory M, on the pilots):
  "the task is valid"               if O_M_T binds 2/2 within 24000;
  "the task needs the longer budget" if not, and O_M_T_48K binds 2/2 within 48000;
  "untested"                        otherwise;
  "solvable without memory"         if phase 0 stopped T.
  A run that fails (ok False) is retried by the harness (3 attempts in all); still failed, it counts as not bound (in
  phase 0 as accuracy 0) and the report prints it. If each pilot run binds independently with probability p, P(valid) =
  p^2 per memory (0.81 at p = 0.9): the rule is fixed and conservative.
PHASE 2, THE SCREEN (3062-3071, 10 seeds paired across arms; each (T, M) read "valid" or "needs the longer budget", at
  that budget): one channel, no gate:
    SINGLE_DELTA_<T>     one delta channel (test_short_conv's arm B, k = 1; beta 1, L2 keys), SLOW;
    SINGLE_HEBBCONV_<T>  one Hebbian channel + conv 'layer' (test_short_conv's B_conv, k = 1), SLOW.
  The question: does a single channel bind documents without any routing when the contexts are not fixed types? (On lite's
  header layout one delta channel bound 8/10, Revision 10 §11.) The perfect gate it is read against is the pilot (2/2 at
  the task's budget, which is its validity); it is not rerun on 3062-3071.
  OUTCOME: BOUND (a transition). Wilson 95% and bands with every count; median transitions; per run the final accuracy
  and the corrected write-order split (explore_f_tasks.write_order_acc_idx: last write / not last, the last-written
  value's share among the misses).
SCREEN READING (fixed now; on CTXPOOL, the delta channel):
  "one channel binds without a context identity"    if SINGLE_DELTA_CTXPOOL binds >= 7/10;
  "the partition's claim survives the context pool" if it binds <= 3/10 while the delta perfect gate is 2/2 on CTXPOOL
                                                    (its validity at the screen's budget);
  "no reading applies"                              otherwise (4-6/10);
  "untested"                                        if CTXPOOL is not valid on the delta memory.
  Descriptive, no reading: SINGLE_HEBBCONV on every valid task; both single channels on DOCS3 and DOCS4; the exact
  one-sided McNemar between SINGLE_DELTA and SINGLE_HEBBCONV on the same seeds where both ran.
CHECKS (asserted before every stage): explore_f_delta_checks (26); the child's:
  tasks   explore_f_tasks.py equals 9017b82's file and starts with b0c6332's (round 6's classes byte-identical); for each
          new task check_ctx_task (lite's layout at S = D (check_rand_header); the batch is lite's with only the contexts
          renamed; the index by first appearance, at S = 2 -allpool's convention; every key once under each stream with S
          distinct values; the query answered by exactly one pair; the perfect gate; no CTX type as frequent as a value
          type; the cap, exact and against the balanced map's empirical rate), check_inv_stats_k (synthetic gates: the
          relative and per-sequence-permuted perfect gates score separated 1 and VAL cos 0; the type-level gate's
          separated fraction equals the probe's), check_write_order_idx; the frequency tables and both cuts; the write
          order's ERRATUM (write_order_acc_idx equals write_order_acc on -pool; on -allpool the query's CTX token is not
          its stream index).
  ties    this module's run() on recorded arms with the new tasks' code disabled, bit for bit through 2400 on this CPU
          (curve and every recorded statistic): S89's NOMEM_SELF_LITE|2800 and O_DELTA_ALLPOOL|2800 (read at 52e2ea7),
          S89-post's O_HEBB_CONV_ALLPOOL|2801 (read at 3e3b1bf, as O_HEBBCONV_ALLPOOL). No SLOW single-channel record
          exists; the single channels are S89's single-channel construction on the new task (CHECKed below).
  models  each new task: the perfect and type-level gates one-hot at k = S on both memories (the delta memory receives
          the gate); the single channels k = 1 with the right memory and conv; the no-memory models' reach (4 / 10, conv
          weights randomised); groups and lrs at updates 1, 2400, 2401 (the conv in the rest group); the positional bound.
STAGES (explore_out/F/run_s90.sh): phase0, phase1, phase1b, phase2, each followed by --decide (the decision file
explore_out/F/s90_decision.json; a recorded phase is never revised), a commit and a push.
"""
import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore_common16 as c16
import explore_f_common as fc

NAME = fc.name("s90_ctx")
CHILD = "explore_f_s90_ctx_child"
MAIN = False
LR = 1e-3
P0_SEEDS = P1_SEEDS = (3060, 3061)
P2_SEEDS = tuple(range(3062, 3072))
TASKS = ("CTXPOOL", "DOCS3", "DOCS4")
MEMS = ("HEBBCONV", "DELTA")
NOMEM_THR = 0.6
SCREEN_HI, SCREEN_LO = 7, 3
DECISION = os.path.join(fc.F_DIR, "s90_decision.json")
SLOW = "SLOW (rest 1e-4 to 2400)"


def decision():
    return json.load(open(DECISION)) if os.path.exists(DECISION) else {}


def _arms():
    d = decision()
    arms = {}
    for t in TASKS:
        for v in ("NOMEM", "NOMEM_SELF"):
            arms[f"{v}_{t}"] = dict(opt="adam", lr=LR, iters=24000, seeds=P0_SEEDS, sched="Adam 1e-3 throughout",
                                    label=f"no memory ({v.lower()}), {t.lower()}")
    for t in TASKS:
        if t not in d.get("phase0", {}) or d["phase0"][t]["stopped"]:
            continue
        for m in MEMS:
            arms[f"O_{m}_{t}"] = dict(opt="adam", lr=LR, iters=24000, seeds=P1_SEEDS, sched=SLOW,
                                      label=f"perfect gate, {m.lower()}, {t.lower()}")
            arms[f"TL_{m}_{t}"] = dict(opt="adam", lr=LR, iters=24000, seeds=P1_SEEDS, sched=SLOW,
                                       label=f"type-level perfect gate, {m.lower()}, {t.lower()}")
            if d.get("pilots", {}).get(t, {}).get(m, {}).get("double"):
                arms[f"O_{m}_{t}_48K"] = dict(opt="adam", lr=LR, iters=48000, seeds=P1_SEEDS, sched=SLOW,
                                              label=f"perfect gate, {m.lower()}, {t.lower()}, 48000")
            b = d.get("budget", {}).get(t, {}).get(m)
            if isinstance(b, int):
                arms[f"SINGLE_{m}_{t}"] = dict(opt="adam", lr=LR, iters=b, seeds=P2_SEEDS, sched=SLOW,
                                               label=f"one {m.lower()} channel, no gate, {t.lower()}")
    return arms


ARMS = _arms()


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def run_job(arm, seed):
    return child("run", dict(arm=arm, seed=seed, iters=ARMS[arm]["iters"]))


def timing(arm, steps):
    return child("timing", dict(arm=arm, steps=steps))


PARTS = ("tasks", "ties", "models")


def accept_other_cpus():
    """Records of this screen on another CPU: reproduce one (the first stored ok record of that CPU) through 2400 bit for
    bit on this CPU; accept that CPU's records only if it does. [(name, ok)]."""
    import explore_common as ec
    st = fc.load(sys.modules[__name__])
    cpu = c16.STATE["batch_cpu"]
    sha = c16.STATE["shas"][NAME][:12]
    rows = []
    for other in sorted({r.get("cpu") for r in st["runs"].values() if r.get("ok")} - {cpu, None}):
        keys = [k for k, r in st["runs"].items() if r.get("ok") and r.get("cpu") == other and k.split("|")[2] == sha]
        if not keys:
            continue
        k = sorted(keys)[0]
        arm, seed = k.split("|")[0], int(k.split("|")[1])
        ref = st["runs"][k]
        r = child("run", dict(arm=arm, seed=seed, iters=2400))
        want = [c for c in ref["curve"] if c[0] <= 2400]
        so = [x for x in ref["stats"] if x["step"] != "end" and x["step"] <= 2400]
        sn = [x for x in r["stats"] if x["step"] != "end" and x["step"] <= 2400]
        same = lambda a, b: all((b.get(f) == v) or (v != v and b.get(f) != b.get(f)) for f, v in a.items())
        ok = r["curve"] == want and len(so) == len(sn) and all(same(a, b) for a, b in zip(so, sn))
        if ok and other not in c16.STATE["other_cpus"]:
            c16.STATE["other_cpus"].append(other)
        rows.append((f"records on {other}: {k} reproduces through 2400 bit for bit on this CPU ({cpu}): curve {r['curve']} vs "
                     f"{want}; that CPU's records accepted: {ok}", ok))
    return rows


def check():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(fc.WORKERS) as ex:
        res = list(ex.map(lambda part: child("checks", dict(part=part)), PARTS))
    ok = True
    for rows in res:
        for nm, v in rows:
            print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
            ok &= bool(v)
    for nm, v in accept_other_cpus():
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


def line(arm, r):
    if not r.get("ok"):
        return f"FAILED {r.get('error')}"
    e = r.get("end") or {}
    rt = f"  margin {c16.fmt2(e.get('margin'))} sep {c16.fmt2(e.get('seq_separated_frac'))}" if "margin" in e else ""
    return f"acc {r['acc']:.3f}  trans {str(r['transition']):>5}{rt}"


def outcomes(arm, seeds):
    """{seed: True (bound) / False (not bound, or failed after the harness's attempts) / None (no record yet)}; and the
    final accuracies (failed: 0)."""
    me = sys.modules[__name__]
    st = fc.load(me)
    ok = fc.runs(me, arm) if arm in ARMS else {}
    out, acc = {}, {}
    for s in seeds:
        if s in ok:
            out[s], acc[s] = ok[s]["transition"] is not None, ok[s]["acc"]
            continue
        failed = [r for k, r in st["runs"].items() if k.startswith(f"{arm}|{s}|") and not r.get("ok")
                  and r.get("attempts", 1) >= c16.MAX_ATTEMPTS]
        out[s], acc[s] = (False, 0.0) if failed else (None, None)
    return out, acc


def decide():
    """Apply the phase rules to the stored runs and update the decision file (never revises a recorded phase)."""
    me = sys.modules[__name__]
    fc.setup([me])
    for nm, v in accept_other_cpus():
        print(f"  {nm}", flush=True)
    d = decision()
    if "phase0" not in d:
        p0 = {}
        for t in TASKS:
            accs = {}
            for v in ("NOMEM", "NOMEM_SELF"):
                o, a = outcomes(f"{v}_{t}", P0_SEEDS)
                if any(x is None for x in o.values()):
                    print(f"  phase 0 incomplete for {t}", flush=True)
                    return d
                accs[v] = [a[s] for s in P0_SEEDS]
            p0[t] = dict(acc=accs, stopped=any(x > NOMEM_THR for v in accs.values() for x in v))
        d["phase0"] = p0
    elif "pilots" not in d:
        pil = {}
        for t in TASKS:
            if d["phase0"][t]["stopped"]:
                continue
            pil[t] = {}
            for m in MEMS:
                o, _ = outcomes(f"O_{m}_{t}", P1_SEEDS)
                if any(x is None for x in o.values()):
                    print(f"  phase 1 incomplete for {t} {m}", flush=True)
                    return d
                both = sum(o.values()) == 2
                pil[t][m] = dict(at24=both, double=not both)
        d["pilots"] = pil
    budget = d.get("budget", {})
    for t in TASKS:
        budget.setdefault(t, {})
        for m in MEMS:
            if m in budget[t]:
                continue
            if d.get("phase0", {}).get(t, {}).get("stopped"):
                budget[t][m] = "STOPPED (solvable without memory)"
            elif t in d.get("pilots", {}):
                if d["pilots"][t][m]["at24"]:
                    budget[t][m] = 24000
                elif f"O_{m}_{t}_48K" in ARMS:
                    o, _ = outcomes(f"O_{m}_{t}_48K", P1_SEEDS)
                    if all(x is not None for x in o.values()):
                        budget[t][m] = 48000 if sum(o.values()) == 2 else "UNTESTED"
        if not budget[t]:
            budget.pop(t)
    d["budget"] = budget
    json.dump(d, open(DECISION, "w"), indent=1)
    print(f"  decision: {json.dumps(d)}", flush=True)
    return d


def reading_task(d, t, m):
    b = d.get("budget", {}).get(t, {}).get(m)
    return ("solvable without memory" if isinstance(b, str) and b.startswith("STOPPED") else "the task is valid" if b == 24000
            else "the task needs the longer budget" if b == 48000 else "untested" if b == "UNTESTED" else
            "INCOMPLETE, no reading")


def reading_screen(d):
    b = d.get("budget", {}).get("CTXPOOL", {}).get("DELTA")
    if not isinstance(b, int):
        return "untested" if b is not None else "INCOMPLETE, no reading"
    o, _ = outcomes("SINGLE_DELTA_CTXPOOL", P2_SEEDS)
    if any(x is None for x in o.values()):
        return "INCOMPLETE, no reading"
    n = sum(o.values())
    return ("one channel binds without a context identity" if n >= SCREEN_HI else
            "the partition's claim survives the context pool" if n <= SCREEN_LO else "no reading applies")


def report():
    import explore_f_reports7 as fr
    return fr.report_s90()


if __name__ == "__main__":
    import explore_f_s90_ctx as me
    if len(sys.argv) > 1 and sys.argv[1] == "--decide":
        decide()
    else:
        fc.drive("s90", [me], sys.argv[1:])
