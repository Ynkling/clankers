#!/usr/bin/env python
"""
explore_f_s89_post.py — EXPLORATORY, not a result. Session F, round 6: two follow-up pilots to S89-F (explore_f_s89_tasks),
written on 20 October after report 9 and referee V's review of S89-F (claude/review-V 3ae815b). They change no reading of
S89-F: its pilots and its post-hoc diagnostic stand as run (both tasks "untested"). The task module is explore_f_tasks.py
as revised for fixes 1 and 2 (appended only; RandHeaderTaskPool and RandHeaderTaskAllPool are 50dc8e4's).

CPU: this container is an Intel Xeon @ 2.10GHz again (the S89 pilots ran on a Xeon @ 2.80GHz). The revised S89 path
reproduces two S89 pilot records bit for bit through 2400 on this CPU (CHECK), but pairs are bit for bit only within one
CPU, so the readings below are counts per arm, not paired; the S89 pilots are printed beside them for reference. 1 torch
thread per run; Session F's harness (explore_f_common). SEEDS: 2800-2801, the S89 pilots' seeds (the block 2800-2839 is
F's; no other record uses it): each arm has an S89 pilot's batches and, apart from the convolution's weights, its initial
parameters (CHECKed).

ARMS (child explore_f_s89_post_child: explore_f_s89_tasks_child's O_HEBB_<T> pilot with one change; Hebbian memory, k = 2,
SLOW as in the S89 pilots (every parameter at 1e-4 for updates 1-2400, 1e-3 after; the perfect gate has no parameter),
24000 updates, evaluation every 1200 on 2048 held-out sequences, bound = the first evaluation >= 0.95 that holds to the
end, early stop after 3; statistics explore_f_tasks.pool_stats_inv, both forms recorded):
  O_TYPE_HEBB_ALLPOOL   (referee V's fix 2, the fifth pilot on ALLPOOL; DESCRIPTIVE, no reading) the type-level perfect
        gate: CTX types 0-7 -> channel 0, 8-15 -> channel 1 (explore_f_tasks.BALANCED_MAP), in place of the per-sequence
        index. It separates a sequence's two streams in 64/120 = 0.533 of sequences (the cap for any fixed map) and puts
        both streams on one channel in the rest. Stated beforehand: on the Hebbian memory the relative perfect gate is
        itself unbound on ALLPOOL (S89: 0.33 / 0.50 at 24000), so this pilot cannot separate the cap from the memory; it
        prints what the type-level gate reaches and its within-sequence statistics (eta^2 by stream should equal the
        separated fraction).
  O_HEBB_CONV_POOL, O_HEBB_CONV_ALLPOOL   (POST HOC, 4 runs) the Hebbian perfect gate with the short causal convolution
        in the memory's input (test_short_conv.LAYER: conv 'layer', K = 4, depthwise causal, on each layer's input to the
        memory, i.e. what it writes and reads with; the main line's conv as the grouped-layout arms carry it:
        test_stream_recipe's ceiling8k16 = test_scale_axes' ceiling4k4 + LAYER). The conv weights train in the rest group.
HYPOTHESIS (stated before the runs): H's Hebbian perfect gate bound 2/2 on the eight-stream grouped layout with the same
  32-key pool (claude/explore-H, explore_out/H/s89_results.json: ORACLE|2900 bound at 16800, ORACLE|2901 at 10800), and
  that layout's arms carry the convolution; Revision 10 §11 records that one Hebbian channel binds the grouped layout 0/60
  without it. The S89 pilots had no convolution and bound 0/4 on Hebbian. If the convolution is what lets a Hebbian
  channel use pooled keys, O_HEBB_CONV_* binds. Differences from H's oracle that remain: H's ran one Adam at 1e-3 for
  43200 updates on S = 8, k = 16, the grouped layout; these run SLOW for 24000 on S = 2, k = 2, the lite layout.
READINGS (fixed now, on O_HEBB_CONV_POOL and O_HEBB_CONV_ALLPOOL together, 4 runs):
  "the convolution restores the Hebbian oracle on pooled keys"   if 4/4 bind;
  "partly"                                                        if 3/4 bind;
  "it does not"                                                   if 2 or fewer bind.
  A failed run (ok False) is retried by the harness (explore_common16.MAX_ATTEMPTS = 3 attempts in all); a run still
  failed after that counts as not bound, and the report prints it.
  Printed beside the reading (no reading): per task the counts, the transitions, the final accuracies of unbound runs,
  the routing statistics (margin; eta^2 by stream at KEY and VAL cos, pooled and within-sequence).
NO OTHER RUNS. The S89 baselines stay unrun: no Hebbian configuration is valid under S89's rule, whose pilots are fixed.
  If this reading is "restores", the report says so and the choice of a conv configuration for a later round is the
  user's.
CHECKS (asserted before any run): explore_f_delta_checks (26) at the start of the segment; S89-F's CHECKs, all five parts
  as revised (tasks, g87 with this screen's own path tied to G's S87 record at e46799a, s70 at 61ba46e, models, inv: the
  synthetic-gate CHECK and two S89 pilots reproduced bit for bit through 2400 by the revised path); the child's: the type
  gate is BALANCED_MAP's one-hot at every position and the model's parameters are the S89 pilot's; the conv arms' gate is
  task.perfect_gate, the conv is 'layer' with K = 4 and is the only parameter beyond the S89 pilot's, the shared ones equal
  the pilot's, and at initialisation (CausalConv starts at the identity) the logits equal the pilot's bit for bit, so
  each conv arm is its S89 pilot plus a convolution that starts as a no-op; groups and lrs at
  updates 1, 2400, 2401 through the real path (the conv in the rest group).

ROUND-6 FALLBACK ON ALLPOOL (added 21 October, after the readings above came in; the conv pilot read "partly", 3/4). The
  user's decision: the conv configuration (Hebbian, k = 2, the main line's conv 'layer' in the memory's input, SLOW) is this
  round's valid Hebbian configuration on POOL (O_HEBB_CONV_POOL 2/2 at 24000). On ALLPOOL (1/2 at 24000) S89's own fallback
  applies, as S89-F's rule ran it for the S89 pilots:
    O_HEBB_CONV_ALLPOOL_48K   O_HEBB_CONV_ALLPOOL unchanged (the same child arm, the same code SHA 267d08456c38) at 48000
                              updates, seeds 2800-2801 (2 runs). SLOW does not depend on the budget, so a run equals the
                              24000 run through the 24000 run's end (the 2800 run stops early, after its bind, as at 24000).
  READING (fixed now, before the 2 runs): "ALLPOOL: the conv configuration needs the longer budget" if both bind (2/2) at
  48000; "ALLPOOL: untested" otherwise (then G and H run POOL only). A failed run is retried by the harness (3 attempts);
  still failed, it counts as not bound.
  CPU: this container is a Xeon @ 2.80GHz again; the 24000 conv runs ran on a Xeon @ 2.10GHz. CHECK (asserted before the
  runs): O_HEBB_CONV_ALLPOOL|2801 (the 2.10GHz record) through 2400 reproduces bit for bit on this CPU (curve and every
  recorded statistic); only then does the harness accept the 2.10GHz records as this batch's (explore_common16.cpus_for),
  so the 6 recorded runs are not re-queued. No other run: nothing else is queued.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import explore_common16 as c16
import explore_f_common as fc

NAME = fc.name("s89_post")
CHILD = "explore_f_s89_post_child"
MAIN = False
LR = 1e-3
SEEDS = (2800, 2801)
CONV_ARMS = ("O_HEBB_CONV_POOL", "O_HEBB_CONV_ALLPOOL")
ARMS = {"O_TYPE_HEBB_ALLPOOL": dict(opt="adam", lr=LR, iters=24000, seeds=SEEDS, sched="SLOW (rest 1e-4 to 2400)",
                                    label="type-level perfect gate (balanced map), hebb, allpool"),
        "O_HEBB_CONV_POOL": dict(opt="adam", lr=LR, iters=24000, seeds=SEEDS, sched="SLOW (rest 1e-4 to 2400)",
                                 label="perfect gate + conv 'layer', hebb, pool"),
        "O_HEBB_CONV_ALLPOOL": dict(opt="adam", lr=LR, iters=24000, seeds=SEEDS, sched="SLOW (rest 1e-4 to 2400)",
                                    label="perfect gate + conv 'layer', hebb, allpool"),
        "O_HEBB_CONV_ALLPOOL_48K": dict(opt="adam", lr=LR, iters=48000, seeds=SEEDS, sched="SLOW (rest 1e-4 to 2400)",
                                        label="perfect gate + conv 'layer', hebb, allpool, 48000 (fallback)")}
OLD_CPU = "Intel(R) Xeon(R) Processor @ 2.10GHz"          # the 24000 runs


def child(func, payload):
    return c16.run_child(CHILD, func, payload, main=MAIN)


def child_arm(arm):
    """The child's arm: the 48000 fallback is O_HEBB_CONV_ALLPOOL with a longer budget."""
    return arm[:-4] if arm.endswith("_48K") else arm


def run_job(arm, seed):
    return child("run", dict(arm=child_arm(arm), seed=seed, iters=ARMS[arm]["iters"]))


def timing(arm, steps):
    return child("timing", dict(arm=child_arm(arm), steps=steps))


def repro_old_cpu():
    """CHECK: the 2.10GHz record O_HEBB_CONV_ALLPOOL|2801 through 2400 on this CPU, bit for bit; if it holds, accept the
    2.10GHz records as this batch's."""
    import explore_common as ec
    cpu = ec.cpu_model()
    st = fc.load(sys.modules[__name__])
    key = c16.key("O_HEBB_CONV_ALLPOOL", 2801, c16.STATE["shas"][NAME], OLD_CPU)
    ref = st["runs"].get(key)
    if cpu == OLD_CPU:
        return True, f"this CPU is the records' ({cpu}); nothing to reproduce"
    if ref is None:
        return False, f"{key} not found"
    r = child("run", dict(arm="O_HEBB_CONV_ALLPOOL", seed=2801, iters=2400))
    want = [c for c in ref["curve"] if c[0] <= 2400]
    so = [x for x in ref["stats"] if x["step"] != "end" and x["step"] <= 2400]
    sn = [x for x in r["stats"] if x["step"] != "end" and x["step"] <= 2400]
    ok = r["curve"] == want and so == sn
    if ok and OLD_CPU not in c16.STATE["other_cpus"]:
        c16.STATE["other_cpus"].append(OLD_CPU)
    return ok, (f"{key} through 2400 reproduces bit for bit on this CPU ({cpu}): curve {r['curve']} vs {want}; statistics "
                f"at {[x['step'] for x in so]} equal {so == sn}; the 2.10GHz records are accepted as this batch's: {ok}")


def check():
    import explore_f_s89_tasks as s89
    ok = s89.check()
    for nm, v in child("checks", {}):
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    v, nm = repro_old_cpu()
    print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
    return ok and v


def line(arm, r):
    if not r.get("ok"):
        return f"FAILED {r.get('error')}"
    e = r.get("end") or {}
    return (f"acc {r['acc']:.3f}  trans {str(r['transition']):>5}  margin {c16.fmt2(e.get('margin'))} "
            f"etaS {c16.fmt2(e.get('eta_key_by_stream'))}/{c16.fmt2(e.get('eta_key_by_stream_inv'))}")


def reading(n_bound, n_failed, n):
    """n_bound of the conv runs bound, n_failed failed after MAX_ATTEMPTS (not bound), n = ok records."""
    if n + n_failed < len(CONV_ARMS) * len(SEEDS):
        return "INCOMPLETE, no reading"
    return ("the convolution restores the Hebbian oracle on pooled keys" if n_bound == 4 else "partly" if n_bound == 3
            else "it does not")


def reading_48k(n_bound, n_failed, n):
    """The fallback's reading on O_HEBB_CONV_ALLPOOL_48K (2 runs)."""
    if n + n_failed < len(SEEDS):
        return "INCOMPLETE, no reading"
    return "ALLPOOL: the conv configuration needs the longer budget" if n_bound == 2 else "ALLPOOL: untested"


def report():
    from explore_f_reports6 import report_s89_post
    return report_s89_post()


if __name__ == "__main__":
    import explore_f_s89_post as me
    fc.drive("s89post", [me], sys.argv[1:])
