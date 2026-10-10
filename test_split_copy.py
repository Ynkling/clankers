#!/usr/bin/env python
"""
test_split_copy.py — at eight streams, is it the split's row copy or its Adam reset that binds the window gate, and
does a single delta-rule channel do without the partition? Everything is fixed here before any run (specification:
specs/test_split_copy.md, committed before this file).

    python3 -u test_split_copy.py --machine X --workers 4            # CHECKs, projection, all runs, report
    python3 -u test_split_copy.py --machine L --report --also results/X/split_copy_results.json
    python3 longrun.py start split_copy_X --results split_copy_results.json --total 54 -- \\
        python3 -u test_split_copy.py --machine X --workers 4       # detached, resumable

BACKGROUND
- test_window_gate Part D (pooled X + L): WIN3_SPLIT_D8 BOUND ROUTED 38/40; the split beats no trigger (G4, 23 vs 0).
  The split copies W_g's row c* onto c0 (noise 0.1 std) and zeroes W_g's Adam state. The reset-only control (the same
  trigger, the state zeroed, W_g untouched) ran on five seeds per machine: bound 0/5 on X, 3/5 on L (printed only).
- Session F's S68 (claude/explore-F, EXPLORATORY): one delta-rule channel (beta 1, L2 keys, tied write, decay 0.95) on
  the grouped eight-stream task with conv, Adam 1e-3, 43200 updates: bound 0/10 (seeds 300-309); its perfect gate
  (k = 8) on the same memory 2/2 at 1200.

CONFIGURATION (test_window_gate's Part C/D path throughout): test_stream_curriculum.run_sc with D8's settings (all eight
streams from update 1): task_for(4, 8) (S = 8, P = 4, n_vals 16, n_q 1, L = 99), conv 'layer' K = 4, batch 32, 43200
updates, evaluation every 1200 on 2048 held-out sequences, run_one's early stop (three evaluations >= 0.95).

SEEDS: X 560-569, L 1560-1569 for every arm; the perfect gates on the first two (560-561, 1560-1561). Pairs by seed.

ARMS
Part A, the split's ingredient (test_window_gate's arms and one_run, imported unchanged):
- WIN3_SPLIT_D8  LOCAL3 window gate + SLOW (gate W_in, W_g, window at 1e-3; the rest 1e-4 for updates 1-2400, 1e-3
                 after; W_h frozen), no hinge, with batch 17's KEYMASS trigger: checks every 2400 from 4800 (2400 is the
                 reference) to 40800 on a 64-sequence probe; fires if the probe accuracy < 0.95 and its rise < 0.02;
                 copies W_g[c*] onto W_g[c0] with noise 0.1 std and zeroes W_g's Adam state; at most 3 splits, >= 4800
                 apart.
- WIN3_RESET_D8  the same trigger; W_g's Adam state zeroed, W_g untouched (all ten seeds).
- WIN3_SLOW_D8   no trigger.
- CEIL_C         the perfect gate (SC8_ceil: k = 16, ctx_pad 16, no curriculum), Hebbian, Adam 1e-3 throughout (validity).
Outcome: BOUND ROUTED (test_stream_recipe's: bound, one-to-one map, every stream >= 0.9 at the end), with the transition.
Part B, the premise (explore_delta_mem.py and explore_f_delta_fast.py copied byte for byte from claude/explore-F at
9a6c2bf; F's S68 ran the fused form, impl "fast", equal to the parallel form to 1e-5 relative):
- SINGLE_DELTA_D8 test_short_conv's B_conv (k = 1, gate 'none', conv 'layer' K = 4) on task_for(4, 8), converted by
                 explore_f_delta_fast.to_delta_fast(m, seed=seed, beta=1.0): beta 1 fixed, L2 keys, tied write, decay
                 0.95, no gate; no parameter added. F's S68_SINGLE_S8 arm.
- SINGLE_HEBB_D8 the same k = 1 model, Hebbian (unconverted).
- CEIL_DELTA_D8  CEIL_C's model (perfect gate, k = 16) converted by to_delta_fast(beta=1.0) (validity).
Recipe of all three: Adam 1e-3 on every parameter throughout (F's S68 recipe). Outcome: BOUND (a transition held to the
end; k = 1 has no routing), with the transition.

VALIDITY (test_window_gate's rule, 2 of 2): Part A is VALID on a machine if CEIL_C binds on both of its seeds, Part B
if CEIL_DELTA_D8 does. A claim is UNTESTED on a machine where a part it uses is not VALID (Part B: UNTESTED, not a
negative); a pooled claim is UNTESTED unless its parts are VALID on both machines.

CLAIMS (exact one-sided McNemar on seed pairs, SHOWN if p < 0.05; POOLED over both machines PRIMARY, b and c summed;
per machine SECONDARY):
- C1  WIN3_SPLIT_D8 beats WIN3_RESET_D8 (BOUND ROUTED).
- C2  WIN3_RESET_D8 is not better than WIN3_SLOW_D8 by more than 2 discordant pairs pooled: d = RESET only - SLOW only
      <= 2 HOLDS, else FAILS (a bound, no p-value).
- C3  WIN3_SPLIT_D8 (BOUND ROUTED) beats SINGLE_DELTA_D8 (BOUND).
READINGS (pooled only):
- "the copy, not the reset" if C1 is SHOWN and C2 HOLDS.
- "the partition is necessary at eight streams against a single delta channel" if C3 is SHOWN, SINGLE_DELTA_D8 is
  below MAJORITY (bound on fewer than ceil(0.5 n)) and CEIL_DELTA_D8 bound on both machines (Part B VALID on X and L).
PRINTED, NOT CLAIMS: SINGLE_DELTA_D8 vs SINGLE_HEBB_D8 (BOUND, descriptive p both ways); every split's update, c* -> c0
and whether its target is labelled ok (c* holds >= 2 streams and c0 none on the labelled map before it); the control's
resets; CEIL_DELTA_D8's and CEIL_C's transitions; the k = 1 arms' final held-out accuracy by stream; bands with
Wilson 95% intervals for every arm (RELIABLE >= ceil(0.9 n), MAJORITY >= ceil(0.5 n)); failure classes of unbound
k = 16 runs by fail_class_v2.outcome_v2.

CHECKS (before any run unless stated)
- The inherited chain: test_window_gate's verification (135-144; below it test_muon_recipe's, test_early_recipe's, ...
  down to test_multilayer_binding's) on this machine's own earlier results files. test_muon_recipe's CHECK 130 (its
  Muon reproductions of this machine's test_early_recipe file) is asserted when that file was written on this CPU and
  printed otherwise: the keyword early_assert (default True, as recorded), passed through test_window_gate.verify.
- 145 that knob is inert: both verify functions at the knob's default have the syntax tree they had at fb3acb7.
- 146 WIN3_SPLIT_D8 and WIN3_RESET_D8 at a Part D seed (the first of the machine's Part D range on which both recorded
      arms fired by 7200: X 540, L 1541) equal this machine's test_window_gate records bit for bit through 7200 (curve,
      gradient norms, statistics, the trigger's rows); asserted on the records' CPU, printed elsewhere.
- 147 CHECK 142 rerun (the trigger's mechanics). 148 CHECK 143 rerun (threshold 0: both Part D arms equal WIN3_SLOW_D8,
      seed 1, through 6000).
- 149 the delta memory at beta 0 (erase off, write 1, raw keys) gives the Hebbian logits to 1e-7 on SINGLE's and
      CEIL_DELTA's models, impl parallel and fast; impl "hebb" bit-identical (F's C2).
- 150 SINGLE_DELTA_D8 at seed 300 on this path equals F's S68_SINGLE_S8|300 (explore_out/F/premise_results.json at
      9a6c2bf, Xeon @ 2.10GHz) through 2400; asserted on a matching CPU, printed otherwise.
- 151 fail_class_v2: (a) outcome_v2 equals test_stream_recipe.outcome on every Part C/D record of this machine's
      test_window_gate file (asserted); (b) in the report, on every Part A record of this test (printed: v1 and v2 also
      differ at eight streams where eta^2 by key >= 0.5 and margin >= 0.25, as at test_recipe_scope's SC8_H s264, so
      any disagreement is printed with eta^2 and margin).
- 152 the copied modules equal F's files at 9a6c2bf byte for byte; F's fast_checks pass here.
- 153 the Part B models (k = 1 B_conv on task_for(4, 8) with a FastDeltaMemory, beta 1, L2 keys, tied write, decay
      0.95, no parameter added; SINGLE_HEBB's initial parameters equal SINGLE_DELTA's; CEIL_DELTA's equal CEIL_C's and
      its gate is one-hot on each position's stream).
- 154 the Part B recipe: one Adam group at 1e-3 holding every trainable parameter at updates 1, 2400, 2401.
- 155 the seeds; 156 a worker's run is bit-identical to the same run here; 157 (a resume) cached records reproduce.

RUNTIME: projection with pool-load timing before the first run (longrun.projection: all runs and the runs still to
do). Nothing is cut. Perfect gates first, then longest first. Runs are cached in split_copy_results.json (gitignored,
written atomically after each run); a restart resumes from the cached records (longrun.resume_meta).
"""

import argparse
import ast
import math
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

import test_binding_onset as tbo
from test_binding_onset import EVAL_EVERY, fmt_step
from test_router_curriculum import get, load_store, init_worker, cpu_model, save_results
from test_router_reliability import strip_all
from test_conv_lr import bound_r, mcnemar_greater
import test_short_conv as tsc
import test_stream_recipe as tsr
from test_stream_channels import med_int, mstr, shared_max
import test_stream_curriculum as tscur
import test_slow_start as tss
from test_slow_start import ok_r, subset_equal, trunc, curve_str
from test_recipe_scope import groups_str
import test_muon_recipe as tmr
import test_window_gate as tgw
import explore_delta_mem as dm
import explore_f_delta_fast as ff
import fail_class_v2 as fc2
import longrun

# ── Settings (fixed before any run) ──────────────────────────────────────────
MACHINES = ("X", "L")
RANGES = {"X": range(560, 570), "L": range(1560, 1570)}
ITERS = tgw.MAX_C                          # 43200
VALID_N = tgw.VALID_N                      # 2 of 2
C2_BOUND = 2
LR = tgw.LR                                # 1e-3: Part B's single rate (F's S68 recipe)
BETA = 1.0
F_SHA = "9a6c2bf"                          # claude/explore-F: explore_delta_mem.py, explore_f_delta_fast.py, S68's records
F_FILES = ("explore_delta_mem.py", "explore_f_delta_fast.py")
F_RECORD = ("explore_out/F/premise_results.json", "S68_SINGLE_S8|300")
F_SEED, F_THROUGH = 300, 2400
KNOB_SHA = "fb3acb7"                       # the head before the early_assert knob (CHECK 145)
D_THROUGH = 7200                           # CHECK 146
D_SEED = {"X": 540, "L": 1541}             # the rule's seed per machine (CHECK 146 recomputes it from the file)
BETA0_TOL = 1e-7
TIME_ITERS = 1200
WORKERS = min(4, os.cpu_count() or 1)
RESULTS_FILE = "split_copy_results.json"

SINGLE_ARM = dict(tsc.ARM["B_conv"], key="SINGLE_D8", task="P4S8", cur=False)
ARMS = [
    dict(key="CEIL_C", part="A", ceil=True, src="tgw",
         label="CEIL_C          perfect gate SC8_ceil (k=16), Hebbian, Adam 1e-3 (validity, Part A)"),
    dict(key="CEIL_DELTA_D8", part="B", ceil=True, src="b", model=tgw.CEIL_C_ARM, delta=True,
         label="CEIL_DELTA_D8   perfect gate (k=16) on the delta memory (beta 1), Adam 1e-3 (validity, Part B)"),
    dict(key="WIN3_SPLIT_D8", part="A", ceil=False, src="tgw",
         label="WIN3_SPLIT_D8   LOCAL3 + SLOW + the split trigger (copy + Adam reset)"),
    dict(key="WIN3_RESET_D8", part="A", ceil=False, src="tgw",
         label="WIN3_RESET_D8   LOCAL3 + SLOW + the trigger, Adam reset only"),
    dict(key="WIN3_SLOW_D8", part="A", ceil=False, src="tgw",
         label="WIN3_SLOW_D8    LOCAL3 + SLOW, no trigger"),
    dict(key="SINGLE_DELTA_D8", part="B", ceil=False, src="b", model=SINGLE_ARM, delta=True,
         label="SINGLE_DELTA_D8 one delta channel (k=1, beta 1, L2 keys), Adam 1e-3"),
    dict(key="SINGLE_HEBB_D8", part="B", ceil=False, src="b", model=SINGLE_ARM, delta=False,
         label="SINGLE_HEBB_D8  one Hebbian channel (k=1), Adam 1e-3"),
]
ARM = {a["key"]: a for a in ARMS}
KEYS = tuple(ARM)
ARMS_RUN = tuple(k for k in KEYS if not ARM[k]["ceil"])
CEILS = (("A", "CEIL_C"), ("B", "CEIL_DELTA_D8"))
OUTCOME = {"A": "BOUND ROUTED", "B": "BOUND"}
CLAIMS = (("C1", "WIN3_SPLIT_D8", "WIN3_RESET_D8", "A"), ("C3", "WIN3_SPLIT_D8", "SINGLE_DELTA_D8", "AB"))
C2 = ("C2", "WIN3_RESET_D8", "WIN3_SLOW_D8", "A")
PRINTED = (("one delta channel vs one Hebbian channel", "SINGLE_DELTA_D8", "SINGLE_HEBB_D8", "B"),)
READING_1 = "the copy, not the reset"
READING_2 = "the partition is necessary at eight streams against a single delta channel"


def seeds_for(machine):
    s = tuple(RANGES[machine])
    return {k: (s[:VALID_N] if ARM[k]["ceil"] else s) for k in KEYS}


SEEDS = seeds_for("X")                     # main() sets the machine's


def outcome2(r):
    """test_stream_recipe.outcome under fail_class_v2 (k = 16 runs)."""
    return fc2.outcome_v2(r, tsr.routed)


def success(key, r):
    """The arm's outcome: BOUND ROUTED for Part A's learned gates, BOUND otherwise."""
    if not ok_r(r):
        return False
    if ARM[key]["part"] == "A" and not ARM[key]["ceil"]:
        return outcome2(r) == "BOUND ROUTED"
    return bound_r(r)


# ── Runs ─────────────────────────────────────────────────────────────────────
def b_recipe(delta, keep=None):
    """run_sc's recipe for Part B: the builder's model converted to the delta memory (to_delta_fast, beta 1) when
    delta; Adam 1e-3 single rate as run_sc gives it."""
    def rc(kw):
        if delta:
            b0 = kw["builder"]

            def builder(aa, seed):
                make = b0(aa, seed)
                return lambda: ff.to_delta_fast(make(), seed=seed, beta=BETA)
            kw = dict(kw, builder=builder)
        if keep is not None:
            kw = dict(kw, keep=keep)
        return kw
    return rc


def run_b(key, seed, iters, keep=None):
    a = ARM[key]
    rec = tscur.run_sc(a["model"], seed, dict(tscur.REAL, total=iters), recipe=b_recipe(a["delta"], keep))
    if rec.get("ok"):
        rec.update(arm=key, cpu=cpu_model(), k=a["model"]["n_ch"],
                   delta_cfg=dict(beta=BETA, write="tied", normalize=True, decay="fixed", impl="fast") if a["delta"] else None)
    return tss.norm(rec)


def one_run(key, seed, iters):
    """A run of arm key: Part A's arms through test_window_gate.one_run (its REAL schedule: SLOW's switch at 2400, the
    trigger's horizon 43200 - 2400), Part B's through run_b."""
    if ARM[key]["src"] == "tgw":
        return tgw.one_run(tgw.ARM[key], seed, iters, tgw.REAL)[0]
    return run_b(key, seed, iters)


def run_job(sp):
    return one_run(sp["arm"], sp["seed"], sp["iters"])


def spec(key, seed, iters):
    return dict(arm=key, seed=seed, iters=iters)


def time_job(key):
    """Pool-load timing: the arm's real path for TIME_ITERS steps (one evaluation included); seconds per step."""
    t0 = time.time()
    one_run(key, 0, TIME_ITERS)
    return (time.time() - t0) / TIME_ITERS


# ── CHECK jobs ───────────────────────────────────────────────────────────────
def check_a(key, seed, iters, thr=tgw.ACC_THR):
    return tgw.check_job(key, seed, iters, thr)


def check_b(key, seed, iters):
    return dict(rec=run_b(key, seed, iters))


def beta0_job():
    """CHECK 149: logits of the delta memory at beta 0 (erase off, write 1, raw keys) vs the unconverted model."""
    import copy
    out = []
    for name, a in (("SINGLE (k=1)", SINGLE_ARM), ("CEIL_DELTA (perfect gate, k=16)", tgw.CEIL_C_ARM)):
        task = tsr.TASKS[a["task"]]
        if a.get("ctx_pad"):
            task = tsr.padded_task(task, a["ctx_pad"])
        m0 = tsr.builder_for(a)(a, RANGES["X"][0])()
        x = task.make_batch(16, torch.Generator().manual_seed(3))[0][:, :-1]
        m0.eval()
        with torch.no_grad():
            l0 = m0(x)[0]
        for impl in ("parallel", "fast", "hebb"):
            if impl == "fast":
                m = ff.to_delta_fast(copy.deepcopy(m0), beta=0.0, write=1.0, normalize=False)
            else:
                m = dm.to_delta(copy.deepcopy(m0), beta=0.0, write=1.0, normalize=False, impl=impl)
            m.eval()
            with torch.no_grad():
                lg = m(x)[0]
            out.append(dict(name=name, impl=impl, d=float((lg - l0).abs().max()), equal=torch.equal(lg, l0),
                            cls=type(m.attn).__name__, erase_off=m.attn.erase_off()))
    return out


def fast_job():
    """CHECK 152: F's fast_checks (fused form = parallel form)."""
    return [[n, bool(v)] for n, v in ff.fast_checks()]


def models_job():
    """CHECK 153: the Part B models as their run path builds them (seed 560)."""
    s = RANGES["X"][0]
    hebb = tsr.builder_for(SINGLE_ARM)(SINGLE_ARM, s)()
    delta = b_recipe(True)(dict(builder=tsr.builder_for(SINGLE_ARM)))["builder"](SINGLE_ARM, s)()
    ph, pd = dict(hebb.named_parameters()), dict(delta.named_parameters())
    same_single = set(ph) == set(pd) and all(torch.equal(ph[n], pd[n]) for n in ph)
    at = delta.attn
    single_ok = (delta.n_ch == 1 and delta.gate_kind == "none" and delta.conv == "layer" and delta.short_conv.conv_w.shape[1] == tsc.CONV_K
                 and type(at) is ff.FastDeltaMemory and at.impl == "fast" and at.beta_fixed == BETA and at.normalize
                 and at.write == "tied" and at.decay_mod is None and at.decay == 0.95 and not at.erase_off())
    a = tgw.CEIL_C_ARM
    c0 = tsr.builder_for(a)(a, s)()
    cd = b_recipe(True)(dict(builder=tsr.builder_for(a)))["builder"](a, s)()
    pc, pcd = dict(c0.named_parameters()), dict(cd.named_parameters())
    same_ceil = set(pc) == set(pcd) and all(torch.equal(pc[n], pcd[n]) for n in pc)
    task = tsr.padded_task(tsr.TASKS[a["task"]], a["ctx_pad"])
    x = task.make_batch(4, torch.Generator().manual_seed(3))[0][:, :-1]
    cd.eval()
    with torch.no_grad():
        g = cd(x)[2]
    lab = task.stream_labels(x)
    onehot = bool((g.argmax(-1) == lab).all()) and bool(((g == 0) | (g == 1)).all())
    ceil_ok = (cd.n_ch == 16 and cd.gate_kind == "perfect" and type(cd.attn) is ff.FastDeltaMemory
               and cd.attn.beta_fixed == BETA and cd.attn.normalize)
    return dict(single_ok=single_ok, same_single=same_single, n_single=len(ph), task_L=tsr.TASKS["P4S8"].L,
                same_ceil=same_ceil, onehot=onehot, ceil_ok=ceil_ok, n_ceil=len(pc), g_shape=list(g.shape),
                cls=type(delta).__name__, cls_ceil=type(cd).__name__)


def lr_job(key):
    """CHECK 154: each group's lr at every update through the arm's real path (forward and evaluation stubbed), 2401
    updates; the groups' parameters."""
    lrs, ids = [], {}

    def hook(opt, args, kwargs):
        if not ids:
            ids["g"] = [[id(p) for p in g["params"]] for g in opt.param_groups]
        lrs.append([g["lr"] for g in opt.param_groups])

    saved = (tbo.evaluate, tbo.logits_of)
    h = register_optimizer_step_pre_hook(hook)
    tbo.evaluate = lambda model, task, data: (0.5, 1.0)
    tbo.logits_of = lambda model, x: (torch.zeros(x.shape[0], x.shape[1], model.lm_head.shape[1])
                                      + model.lm_head.sum() * 0.0)
    keep = {"at": 1}
    try:
        rec = run_b(key, 3, tss.WARM + 1, keep=keep)
    finally:
        tbo.evaluate, tbo.logits_of = saved
        h.remove()
    m = keep["model"]
    flat = [i for g in ids["g"] for i in g]
    train = {id(p) for p in m.parameters() if p.requires_grad}
    return dict(ok=rec["ok"], n=len(lrs), groups=len(ids["g"]),
                at={str(u): lrs[u - 1] for u in (1, tss.WARM, tss.WARM + 1) if u <= len(lrs)},
                cover=len(flat) == len(set(flat)) and set(flat) == train, n_params=len(train))


# ── CHECK 145: the knob's syntax tree ────────────────────────────────────────
class _Default(ast.NodeTransformer):
    """The knob at its default: drop the parameter, keep the branch it selects at True, drop it from calls."""

    def __init__(self, name):
        self.name = name

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        a = node.args
        names = [x.arg for x in a.args]
        if self.name in names:
            i = names.index(self.name)
            j = i - (len(a.args) - len(a.defaults))
            del a.args[i]
            del a.defaults[j]
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if isinstance(node.test, ast.Name) and node.test.id == self.name:
            return node.body
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        node.keywords = [k for k in node.keywords if k.arg != self.name]
        return node


def fn_tree(src, name, knob=None):
    tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(fn) == 1, name
    node = fn[0]
    if knob is not None:
        node = _Default(knob).visit(node)
    return ast.dump(node, include_attributes=False)


def git_show(sha, path):
    return subprocess.run(["git", "-C", HERE, "show", f"{sha}:{path}"], capture_output=True, text=True, check=True).stdout


def knob_default(src, name, knob):
    """(present, default) of the keyword `knob` of function `name` in source src."""
    fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name][0]
    a = fn.args
    names = [x.arg for x in a.args]
    if knob not in names:
        return False, None
    d = a.defaults[names.index(knob) - (len(a.args) - len(a.defaults))]
    return True, ast.literal_eval(d)


def knob_check():
    out = []
    for path in ("test_muon_recipe.py", "test_window_gate.py"):
        src = open(os.path.join(HERE, path)).read()
        new = fn_tree(src, "verify", "early_assert")
        old = fn_tree(git_show(KNOB_SHA, path), "verify")
        has, default = knob_default(src, "verify", "early_assert")
        out.append((path, new == old, has, default))
    return out


# ── Verification ─────────────────────────────────────────────────────────────
def d_seed(window, machine):
    """CHECK 146's seed: the first of the machine's Part D range on which both recorded arms fired by 7200."""
    for s in tgw.RANGES[machine]["C"]:
        rs = [get(window, k, s) for k in ("WIN3_SPLIT_D8", "WIN3_RESET_D8")]
        if all(ok_r(r) and any(c["fired"] and c["step"] <= D_THROUGH for c in r.get("checks", [])) for r in rs):
            return s
    return None


def f_record():
    import json
    runs = json.loads(git_show(F_SHA, F_RECORD[0]))["runs"]
    ks = [k for k in runs if k == F_RECORD[1] or k.startswith(F_RECORD[1] + "|")]
    assert len(ks) == 1, ks
    return ks[0], runs[ks[0]]


def earlier_seeds():
    return tgw.earlier_seeds() | {s for m in MACHINES for p in "ABC" for s in tgw.RANGES[m][p]}


def other_of(machine):
    return "L" if machine == "X" else "X"


def check_also(machine, also):
    m = also.get("meta", {})
    oth = other_of(machine)
    seeds = m.get("seeds") or {}
    in_range = all(s in RANGES[oth] for k in KEYS for s in seeds.get(k, []))
    mine = {s for k in KEYS for s in SEEDS[k]}
    theirs = {s for k in KEYS for s in seeds.get(k, [])}
    good = m.get("machine") == oth and bool(seeds) and in_range and m.get("cpu") != cpu_model() and not (mine & theirs)
    return good, (f"--also: machine {m.get('machine')} (should be {oth}); CPU {m.get('cpu')} (this machine {cpu_model()}; "
                  f"differs: {m.get('cpu') != cpu_model()}); its seeds {tsr.ranges(sorted(theirs)) if theirs else None} in "
                  f"{oth}'s range: {in_range}; overlap with this machine's: {sorted(mine & theirs) or 'none'}")


def v12_rows(store, keys, seeds):
    """(arm, seed, v1, v2, eta^2 by key (multichannel), margin) for every ok record."""
    out = []
    for k in keys:
        for s in seeds.get(k, ()):
            r = get(store, k, s)
            if ok_r(r):
                e = r.get("end") or {}
                out.append((k, s, tsr.outcome(r), outcome2(r), e.get("etak_key_by_key"), e.get("margin")))
    return out


def verify(pool, files, machine, early, also, window, cached):
    f = {}
    ds = d_seed(window, machine)
    for k in ("WIN3_SPLIT_D8", "WIN3_RESET_D8"):
        if ds is not None:
            f["146", k] = pool.submit(check_a, k, ds, D_THROUGH)
    f["150"] = pool.submit(check_b, "SINGLE_DELTA_D8", F_SEED, F_THROUGH)
    for k in ("WIN3_SPLIT_D8", "WIN3_RESET_D8", "WIN3_SLOW_D8"):
        f["148", k] = pool.submit(check_a, k, 1, 6000, 0.0)
    for op in ("split", "reset"):
        f["147", op] = pool.submit(tgw.trigger_job, op)
    f["149"] = pool.submit(beta0_job)
    f["152"] = pool.submit(fast_job)
    f["153"] = pool.submit(models_job)
    for k in ("SINGLE_DELTA_D8", "SINGLE_HEBB_D8", "CEIL_DELTA_D8"):
        f["154", k] = pool.submit(lr_job, k)
    f["156", "B"] = pool.submit(check_b, "SINGLE_DELTA_D8", 1, EVAL_EVERY)
    f["156", "A"] = pool.submit(check_a, "WIN3_SPLIT_D8", 1, EVAL_EVERY)
    for k, (s0, _) in cached.items():
        f["157", k] = pool.submit(check_a if ARM[k]["src"] == "tgw" else check_b, k, s0, tss.WARM)

    early_assert = early.get("meta", {}).get("cpu") == cpu_model()
    print("test_window_gate.py's verification (which runs test_muon_recipe's, and so on down to")
    print("test_multilayer_binding's); test_muon_recipe's CHECK 130 is "
          + ("asserted (this machine's test_early_recipe file was written on this CPU)" if early_assert else
             f"printed, not asserted (the file was written on {early.get('meta', {}).get('cpu')}, this machine is "
             f"{cpu_model()})") + ":")
    tgw.SEEDS = tgw.seeds_for(machine)
    tgw.verify(pool, files, machine, early, None, None, early_assert=early_assert)
    print("=" * 100)
    print("VERIFICATION — this test's own CHECKs")
    print("=" * 100)
    ok = True

    print(f"CHECK 145 the early_assert knob is inert: verify at its default has the syntax tree it had at {KNOB_SHA}:")
    good = True
    for path, same, has, default in knob_check():
        g = same and has and default is True
        good &= g
        print(f"     {path}: verify(..., early_assert={default}) syntax tree equal to {KNOB_SHA}'s {same}  -> "
              f"{'INERT' if g else 'DIFFERS'}")
    ok &= good
    print()

    wm = window.get("meta", {})
    match = wm.get("cpu") == cpu_model()
    print(f"CHECK 146 WIN3_SPLIT_D8 and WIN3_RESET_D8 at seed {ds} (the first of machine {machine}'s Part D range on which "
          f"both recorded arms fired by {D_THROUGH}; fixed: {D_SEED[machine]}) vs this machine's test_window_gate records "
          f"({files['window_path']}, written on {wm.get('cpu')}) through {D_THROUGH} (asserted on a matching CPU; this "
          f"machine {cpu_model()}):")
    good = ds == D_SEED[machine]
    for k in ("WIN3_SPLIT_D8", "WIN3_RESET_D8"):
        if ds is None:
            print(f"     {k}: no seed")
            good = False
            continue
        r = f["146", k].result()
        old = get(window, k, ds)
        c, g_, s_, shared = tgw.same_through(r["rec"], old, D_THROUGH)
        rn = [x for x in r["checks"] if x["step"] <= D_THROUGH]
        ro = [x for x in old["checks"] if x["step"] <= D_THROUGH]
        rows_eq = len(rn) == len(ro) > 0 and all(all(a[q] == b[q] for q in set(a) & set(b)) for a, b in zip(rn, ro))
        g = c and g_ and s_ and rows_eq and r["rec"]["ok"]
        good &= g
        print(f"     {k}|{ds}: curve {r['rec']['curve']} vs {[x for x in old['curve'] if x[0] <= D_THROUGH]}; curve equal {c}; "
              f"gradient norms equal {g_}; statistics equal {s_} ({shared}); the trigger's rows at {[x['step'] for x in ro]} "
              f"equal {rows_eq} (fired at {[x['step'] for x in rn if x['fired']]} vs {[x['step'] for x in ro if x['fired']]}"
              f"{'; targets ' + str([(x['key_cs'], x['key_c0'], x['key_ok']) for x in rn if x['fired']]) if k == 'WIN3_SPLIT_D8' else ''})"
              f"  -> {'IDENTICAL' if g else 'DIFFERS'}")
    if match:
        ok &= good
    else:
        print(f"     (the records' CPU differs from this machine's: printed, not asserted)")
    print()

    print("CHECK 147 (CHECK 142 rerun) the trigger's mechanics (Part C's LOCAL3 model, seed 260, after one Adam step; checks "
          "2400..26400 with no training between; threshold above 1):")
    good = True
    for op in ("split", "reset"):
        r = f["147", op].result()
        zeroed = all(v == 0.0 for v in r["after"].values()) and any(v > 0 for q, v in r["before"].items() if q != "step")
        g = (r["fired"] == [4800, 9600, 14400] and r["gap"] == [7200, 12000] and r["cap"] == [16800, 19200, 21600, 24000, 26400]
             and zeroed and (all(d > 0 for d in r["dist"]) and not r["wg_same"] if op == "split" else r["wg_same"]))
        good &= g
        print(f"     {op:<5}: fired at {r['fired']}, blocked by the gap at {r['gap']}, by the cap at {r['cap']}; W_g's Adam state "
              f"zeroed {zeroed}; W_g unchanged {r['wg_same']}"
              + (f"; rows c*, c0 apart after each split {[round(d, 4) for d in r['dist']]}" if op == "split" else "")
              + f"  -> {'OK' if g else 'WRONG'}")
    ok &= good
    print()

    print("CHECK 148 (CHECK 143 rerun) with the threshold at 0 the Part D arms equal WIN3_SLOW_D8 (seed 1) bit for bit "
          "through 6000:")
    base = f["148", "WIN3_SLOW_D8"].result()["rec"]
    good = True
    for k in ("WIN3_SPLIT_D8", "WIN3_RESET_D8"):
        r = f["148", k].result()
        same = subset_equal(trunc(r["rec"], 6000), trunc(base, 6000)) and subset_equal(trunc(base, 6000), trunc(r["rec"], 6000))
        g = same and r["rec"]["ok"] and not any(x["fired"] for x in r["checks"]) and len(r["checks"]) == 2
        good &= g
        print(f"     {k}: curve, statistics and gradient norms equal {same}; trigger checks at {[x['step'] for x in r['checks']]}, "
              f"none fired  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()

    print(f"CHECK 149 the delta memory at beta 0 (erase off, write 1, raw keys) vs the Hebbian model's logits (to "
          f"{BETA0_TOL:g}; impl hebb bit-identical):")
    good = True
    for x in f["149"].result():
        g = (x["equal"] if x["impl"] == "hebb" else x["d"] <= BETA0_TOL and x["erase_off"])
        good &= g
        print(f"     {x['name']:<32} impl {x['impl']:<8} ({x['cls']}): logits max |diff| {x['d']:.2e}, bit-identical "
              f"{x['equal']}  -> {'OK' if g else 'WRONG'}")
    ok &= good
    print()

    k_f, rec_f = f_record()
    match = rec_f.get("cpu") == cpu_model()
    r = f["150"].result()["rec"]
    c, g_, s_, shared = tgw.same_through(r, rec_f, F_THROUGH)
    g = c and g_ and s_ and r["ok"]
    print(f"CHECK 150 SINGLE_DELTA_D8 seed {F_SEED} on this test's path vs F's {k_f} ({F_RECORD[0]} at {F_SHA}, written on "
          f"{rec_f.get('cpu')}) through {F_THROUGH} (asserted on a matching CPU; this machine {cpu_model()}):")
    print(f"     curve {r['curve']} vs {[x for x in rec_f['curve'] if x[0] <= F_THROUGH]}; curve equal {c}; gradient norms "
          f"equal {g_}; statistics equal {s_} ({shared})  -> {'IDENTICAL' if g else 'DIFFERS'}")
    if match:
        ok &= g
    else:
        print("     (F's record was written on another CPU: printed, not asserted)")
    print()

    print(f"CHECK 151(a) fail_class_v2.outcome_v2 equals test_stream_recipe.outcome on every Part C/D record of this machine's "
          f"test_window_gate file ({files['window_path']}):")
    wseeds = wm.get("seeds") or {}
    rows = v12_rows(window, ("WIN3_SLOW_D8", "HINGE_D8", "WIN3_SPLIT_D8", "WIN3_RESET_D8"), wseeds)
    diff = [x for x in rows if x[2] != x[3]]
    g = bool(rows) and not diff
    print(f"     {len(rows)} records; disagreements {len(diff)}" + (f": {diff}" if diff else "") + f"  -> {'OK' if g else 'WRONG'}")
    ok &= g
    print()

    print(f"CHECK 152 the copied modules equal claude/explore-F's at {F_SHA} byte for byte; F's fast_checks here:")
    good = True
    for p in F_FILES:
        same = open(os.path.join(HERE, p)).read() == git_show(F_SHA, p)
        good &= same
        print(f"     {p}: equal {same}")
    for n, v in f["152"].result():
        good &= v
        print(f"     {n}  -> {'OK' if v else 'WRONG'}")
    ok &= good
    print()

    r = f["153"].result()
    g = r["single_ok"] and r["same_single"] and r["same_ceil"] and r["onehot"] and r["ceil_ok"] and r["task_L"] == 99
    print(f"CHECK 153 the Part B models (seed {RANGES['X'][0]}): SINGLE_DELTA_D8 {r['cls']} k=1, gate none, conv layer "
          f"K={tsc.CONV_K} on task_for(4, 8) (L {r['task_L']}), FastDeltaMemory beta {BETA:g}, L2 keys, tied write, decay "
          f"0.95 ({r['single_ok']}); its {r['n_single']} initial parameters equal SINGLE_HEBB_D8's ({r['same_single']}); "
          f"CEIL_DELTA_D8 ({r['cls_ceil']}, k=16 perfect gate on the delta memory: {r['ceil_ok']}) has CEIL_C's "
          f"{r['n_ceil']} initial parameters ({r['same_ceil']}) and a gate one-hot on each position's stream ({r['onehot']}, "
          f"gate {r['g_shape']})  -> {'OK' if g else 'WRONG'}")
    ok &= g
    print()

    print(f"CHECK 154 the Part B recipe: one Adam group at {LR:g} with every trainable parameter, at updates 1, {tss.WARM}, "
          f"{tss.WARM + 1} (real paths, forward stubbed):")
    good = True
    for k in ("SINGLE_DELTA_D8", "SINGLE_HEBB_D8", "CEIL_DELTA_D8"):
        r = f["154", k].result()
        g = (r["ok"] and r["groups"] == 1 and r["cover"] and r["n"] == tss.WARM + 1
             and all(r["at"][str(u)] == [LR] for u in (1, tss.WARM, tss.WARM + 1)))
        good &= g
        print(f"     {k:<16} groups {r['groups']}, every trainable parameter ({r['n_params']}) once {r['cover']}; lrs at 1 "
              f"{r['at'].get('1')}, {tss.WARM} {r['at'].get(str(tss.WARM))}, {tss.WARM + 1} {r['at'].get(str(tss.WARM + 1))}"
              f"  -> {'OK' if g else 'WRONG'}")
    ok &= good
    print()

    print(f"CHECK 155 the seeds (machine {machine}):")
    rg = tuple(RANGES[machine])
    arms_ok = all(SEEDS[k] == (rg[:VALID_N] if ARM[k]["ceil"] else rg) for k in KEYS)
    mine = {s for k in KEYS for s in SEEDS[k]}
    other = set(RANGES[other_of(machine)])
    earlier = earlier_seeds()
    g = arms_ok and not (mine & earlier) and not (mine & other)
    print(f"     this machine's seeds {tsr.ranges(sorted(mine))}: every arm on the range, the perfect gates on its first "
          f"{VALID_N} {arms_ok}; disjoint from the other machine's range {not (mine & other)} and from every earlier "
          f"main-line seed {not (mine & earlier)} (earlier: {tsr.ranges(sorted(earlier))})")
    if also is not None:
        ga, line = check_also(machine, also)
        print(f"     {line}  -> {'OK' if ga else 'WRONG'}")
        g &= ga
    else:
        print("     no --also file")
    ok &= g
    print()

    print("CHECK 156 a worker's run is bit-identical to the same run here:")
    good = True
    for which, key, fn in (("B", "SINGLE_DELTA_D8", check_b), ("A", "WIN3_SPLIT_D8", check_a)):
        rw = f["156", which].result()["rec"]
        rh = fn(key, 1, EVAL_EVERY)["rec"]
        same = subset_equal(strip_all(rw), strip_all(rh)) and subset_equal(strip_all(rh), strip_all(rw))
        g = same and rh["ok"]
        good &= g
        print(f"         {key} seed 1, {EVAL_EVERY} steps: records equal {same}   acc {rh['curve'][-1][1]}  -> "
              f"{'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()

    print(f"CHECK 157 (a resume) records cached by an earlier start reproduce under this code through {tss.WARM}:")
    good = True
    if not cached:
        print("     no cached record (a first start)")
    for k, (s0, old) in cached.items():
        r = f["157", k].result()["rec"]
        g = subset_equal(trunc(r, tss.WARM), trunc(old, tss.WARM)) and subset_equal(trunc(old, tss.WARM), trunc(r, tss.WARM))
        good &= g
        print(f"     {k}|{s0}: curve {r['curve']} vs {[x for x in old['curve'] if x[0] <= tss.WARM]}; curve, statistics and "
              f"gradient norms equal {g}  -> {'IDENTICAL' if g else 'DIFFERS'}")
    ok &= good
    print()
    print(f"{'ALL VERIFICATION CHECKS PASSED' if ok else 'SOME VERIFICATION CHECKS FAILED'}\n")
    assert ok, "verification failed — do not trust the results below"


# ── Reporting ────────────────────────────────────────────────────────────────
def paired2(new, old, seeds, fnew, fold):
    """Seed pairs with both records ok: b = new only, c = old only; one-sided exact McNemar (new higher)."""
    both = [s for s in seeds if ok_r(new.get(s)) and ok_r(old.get(s))]
    b = sum(1 for s in both if fnew(new[s]) and not fold(old[s]))
    c = sum(1 for s in both if fold(old[s]) and not fnew(new[s]))
    return dict(n=len(both), new=sum(bool(fnew(new[s])) for s in both), old=sum(bool(fold(old[s])) for s in both), b=b, c=c,
                p=mcnemar_greater(b, c))


def validity(store, seeds):
    out = {}
    for part, k in CEILS:
        n = sum(bound_r(get(store, k, s)) for s in seeds[k])
        out[part] = (n, n >= VALID_N)
    return out


def tally(store, seeds):
    seeds = {k: tuple(seeds.get(k, ())) for k in KEYS}
    rA = {k: tss.runs_of(store, k, seeds[k]) for k in KEYS}
    valid = validity(store, seeds)
    out = dict(rA=rA, valid=valid, seeds=seeds, claims={}, bands={}, printed={})
    for name, new, old, parts in CLAIMS + (C2,):
        d = paired2(rA[new], rA[old], seeds[new], lambda r, k=new: success(k, r), lambda r, k=old: success(k, r))
        out["claims"][name] = dict(d, valid=all(valid[p][1] for p in parts))
    for k in ARMS_RUN:
        out["bands"][k] = dict(c=sum(success(k, r) for r in rA[k].values()), n=len(seeds[k]), valid=valid[ARM[k]["part"]][1])
    for name, new, old, part in PRINTED:
        out["printed"][name] = paired2(rA[new], rA[old], seeds[new], lambda r, k=new: success(k, r),
                                       lambda r, k=old: success(k, r))
    return out


def verdict(d):
    return tgw.verdict(d)


def c2_verdict(d):
    return "UNTESTED" if not d["valid"] or d["n"] == 0 else ("HOLDS" if d["b"] - d["c"] <= C2_BOUND else "FAILS")


def band_of(b):
    return "UNTESTED" if not b["valid"] else tscur.band_n(b["c"], b["n"])


def band_str(b):
    lo, hi = tsr.wilson(b["c"], b["n"])
    return (f"{b['c']}/{b['n']} -> {band_of(b)}   Wilson 95% [{lo:.3f}, {hi:.3f}]  (RELIABLE >= {math.ceil(0.9 * b['n'])}, "
            f"MAJORITY >= {math.ceil(0.5 * b['n'])})")


def out_of(k):
    return OUTCOME["A"] if ARM[k]["part"] == "A" else OUTCOME["B"]


def print_claims(cl, who):
    for name, new, old, parts in sorted(CLAIMS + (C2,)):
        d = cl[name]
        nv = "" if d["valid"] else f"   (a part it uses is not valid{who})"
        if name == "C2":
            print(f"  {name}  {new} not better than {old} by more than {C2_BOUND} discordant pairs (BOUND ROUTED): "
                  f"{d['new']}/{d['n']} vs {d['old']}/{d['n']}; {new} only {d['b']}, {old} only {d['c']}; d = {d['b'] - d['c']} "
                  f"(bound: d <= {C2_BOUND}; no p-value)" + nv)
            print(f"     *** {name}{' (pooled)' if who == ' on a machine' else ''}: {c2_verdict(d)} ***")
            continue
        print(f"  {name}  {new} ({out_of(new)}) beats {old} ({out_of(old)}): {d['new']}/{d['n']} vs {d['old']}/{d['n']}; "
              f"{new} only {d['b']}, {old} only {d['c']}; p = {d['p']:.4g}" + nv)
        print(f"     *** {name}{' (pooled)' if who == ' on a machine' else ''}: {verdict(d)} ***")


def print_printed(pr, label):
    for name, new, old, part in PRINTED:
        d = pr[name]
        print(f"  Printed, not a claim{label}: {name}, {new} vs {old} (BOUND): {d['new']}/{d['n']} vs {d['old']}/{d['n']}; "
              f"{new} only {d['b']}, {old} only {d['c']}; one-sided p = {d['p']:.4g} ({new} higher), "
              f"{mcnemar_greater(d['c'], d['b']):.4g} ({old} higher) (descriptive)")


def print_bands(bands, label):
    for k in ARMS_RUN:
        print(f"  band {k:<16} {label}({out_of(k)}): {band_str(bands[k])}")


def readings(pc, pb, valid_both_b):
    """The pooled readings: (holds, why) for each."""
    c1, c2 = verdict(pc["C1"]), c2_verdict(pc["C2"])
    r1 = (c1 == "SHOWN" and c2 == "HOLDS", f"C1 {c1}; C2 {c2}")
    b = pb["SINGLE_DELTA_D8"]
    below = b["c"] < math.ceil(0.5 * b["n"])
    c3 = verdict(pc["C3"])
    r2 = (c3 == "SHOWN" and below and valid_both_b,
          f"C3 {c3}; SINGLE_DELTA_D8 {b['c']}/{b['n']} pooled, below MAJORITY (< {math.ceil(0.5 * b['n'])}) {below}; "
          f"CEIL_DELTA_D8 bound on both machines {valid_both_b}")
    return r1, r2


def pool_claim(ds):
    return tgw.pool_claim(ds)


def complete(store):
    seeds = store.get("meta", {}).get("seeds") or {}
    missing = [f"{k}|{s}" for k in KEYS for s in seeds.get(k, []) if not ok_r(get(store, k, s))]
    return bool(seeds) and set(seeds) == set(KEYS) and not missing, missing


def stream_str(r):
    sa = (r.get("end") or {}).get("stream_acc")
    return "/".join(f"{v:.2f}" for v in sa) if sa else "--"


def raw_rows(k, runs, seeds):
    a = ARM[k]
    if a["part"] == "A" and not a["ceil"]:
        print(f"  {'arm':<16} {'seed':>4} {'acc':>7} {'transition':>10} {'map':>23} {'1:1':>3} {'sh':>2}  {'per-stream acc':<39} "
              f"{'ROUTED':>6}  {'fired':<18} outcome (v2)")
    else:
        print(f"  {'arm':<16} {'seed':>4} {'acc':>7} {'transition':>10}  {'per-stream acc (end)':<39} outcome")
    for s in seeds:
        r = runs.get(s)
        if not ok_r(r):
            print(f"  {k:<16} {s:>4}  {'NOT RUN' if r is None else 'FAILED — ' + str(r.get('error'))}")
            continue
        if a["part"] == "A" and not a["ceil"]:
            e = r["end"]
            fired = [c["step"] for c in r.get("checks", []) if c["fired"]] if r.get("checks") is not None else None
            print(f"  {k:<16} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10} {mstr(e['ch_map']):>23} "
                  f"{'y' if e['one_to_one'] else 'n':>3} {shared_max(r):>2}  {stream_str(r):<39} "
                  f"{'Y' if tsr.routed(r) else '-':>6}  {str(fired) if fired is not None else '--':<18} {outcome2(r)}")
        else:
            print(f"  {k:<16} {s:>4} {r['acc']:>7.4f} {fmt_step(r['transition']):>10}  {stream_str(r):<39} "
                  f"{'BOUND' if bound_r(r) else 'not bound'}")
    print()


def report(store, seeds, machine, wall, path, also):
    t = tally(store, seeds)
    rA, valid = t["rA"], t["valid"]
    print()
    print("#" * 100)
    print(f"PER-SEED RAW RESULTS — every value, before any aggregate (machine {machine}, {store.get('meta', {}).get('cpu')})")
    print("#" * 100)
    print("  (Part A: map = stream -> channel argmax at the end; 1:1 = one-to-one; sh = most streams on one channel; ROUTED = "
          "one-to-one and every stream >= 0.9; fired = the trigger's firing updates; outcome = fail_class_v2.outcome_v2. "
          "Part B and the perfect gates: BOUND = a transition held to the end)")
    print()
    for k in KEYS:
        print(f"  {ARM[k]['label']}   (outcome {'perfect gate: BOUND' if ARM[k]['ceil'] else out_of(k)})")
        raw_rows(k, rA[k], t["seeds"][k])

    print("=" * 100)
    print("VALIDITY (the perfect gate binds on both of its seeds)")
    print("=" * 100)
    for part, k in CEILS:
        print(f"  Part {part}: {k} bound {valid[part][0]}/{len(t['seeds'][k])} "
              f"({', '.join(f's{s} ' + (('BOUND at ' + str(r['transition'])) if bound_r(r) else 'not bound') for s, r in sorted(rA[k].items()))})"
              f"  -> {'VALID' if valid[part][1] else 'NOT VALID: the claims using Part ' + part + ' are UNTESTED here'}")
    print()

    print("=" * 100)
    print("COUNTS")
    print("=" * 100)
    for k in ARMS_RUN:
        done = [r for r in rA[k].values() if ok_r(r)]
        trs_ = [r["transition"] for r in done if r["transition"] is not None]
        print(f"  {ARM[k]['label']:<74} {out_of(k)} {sum(success(k, r) for r in done):>2}/{len(t['seeds'][k])}   bound "
              f"{len(trs_):>2}   collapsed {sum(bool(r.get('collapsed')) for r in done):>2}   completed {len(done)}/"
              f"{len(t['seeds'][k])}   transition {med_int(trs_) if trs_ else '--'}")
    print()

    other = None
    if also:
        other = load_store(also)
        okc, missing = complete(other)
        om = other.get("meta", {})
        if om.get("machine") != other_of(machine) or not okc:
            print(f"  --also {also}: machine {om.get('machine')} (should be {other_of(machine)}), complete {okc} "
                  f"({len(missing)} runs missing or failed: {missing[:6]}{' ...' if len(missing) > 6 else ''}): NOT POOLED")
            other = None
    print("#" * 100)
    print("PRE-REGISTERED CLAIMS (exact McNemar, one-sided, first arm higher; SHOWN if p < 0.05; C2 a bound)")
    print("#" * 100)
    to = None
    if other is not None:
        om = other.get("meta", {})
        to = tally(other, {k: tuple(v) for k, v in om["seeds"].items()})
        both = {machine: t, other_of(machine): to}
        print(f"POOLED (PRIMARY): machine {machine} ({store.get('meta', {}).get('cpu')}) + machine {other_of(machine)} "
              f"({om.get('cpu')}, {also}); b and c summed over both machines' pairs")
        pc = {name: pool_claim([both[m_]["claims"][name] for m_ in MACHINES]) for name, *_ in CLAIMS + (C2,)}
        print("  (per machine: " + "; ".join(f"{name} " + ", ".join(f"{m_} {both[m_]['claims'][name]['b']} vs "
                                                                     f"{both[m_]['claims'][name]['c']}" for m_ in MACHINES)
                                             for name, *_ in CLAIMS + (C2,)) + ")")
        print_claims(pc, " on a machine")
        pb = {k: dict(c=sum(both[m_]["bands"][k]["c"] for m_ in MACHINES), n=sum(both[m_]["bands"][k]["n"] for m_ in MACHINES),
                      valid=all(both[m_]["bands"][k]["valid"] for m_ in MACHINES)) for k in ARMS_RUN}
        print_bands(pb, "pooled ")
        vb = all(both[m_]["valid"]["B"][1] for m_ in MACHINES)
        (h1, w1), (h2, w2) = readings(pc, pb, vb)
        print("  READINGS (pooled):")
        print(f"     {'' if h1 else 'not '}\"{READING_1}\" ({w1})")
        print(f"     {'' if h2 else 'not '}\"{READING_2}\" ({w2})")
        print_printed({name: pool_claim([both[m_]["printed"][name] for m_ in MACHINES]) for name, *_ in PRINTED}, " (pooled)")
        print()
    else:
        print("POOLED (PRIMARY): not computed here — it needs --also with the other machine's complete file")
        print()
    print(f"PER MACHINE (SECONDARY): machine {machine}")
    print_claims(t["claims"], " on this machine")
    print_bands(t["bands"], "")
    print("  (the readings are pooled only)")
    print_printed(t["printed"], "")
    for k in ARMS_RUN:
        trs_ = [r["transition"] for r in rA[k].values() if ok_r(r) and r["transition"] is not None]
        print(f"     median transition {k:<16} {med_int(trs_) if trs_ else 'none bound'} ({len(trs_)} bound)")
    print()
    if to is not None:
        print(f"  Machine {other_of(machine)} (from --also, the same tests on its pairs):")
        print_claims(to["claims"], f" on {other_of(machine)}")
        print_bands(to["bands"], "")
        print()

    print("=" * 100)
    print("DIAGNOSTICS — NOT part of the verdict")
    print("=" * 100)
    print("  outcomes (Part A: fail_class_v2.outcome_v2; Part B: BOUND / not bound):")
    for k in ARMS_RUN:
        cls = {}
        for s, r in sorted(rA[k].items()):
            if ok_r(r):
                lab = outcome2(r) if ARM[k]["part"] == "A" else ("BOUND" if bound_r(r) else "not bound")
                cls.setdefault(lab + ("  collapsed" if r.get("collapsed") else ""), []).append(s)
        print(f"    {k:<16} " + "   ".join(f"{c_} {len(v)}" for c_, v in sorted(cls.items())))
        for c_, v in sorted(cls.items()):
            print(f"      {c_:<34} {' '.join('s' + str(x) for x in v)}")
    print()
    rows = v12_rows(store, [k for k in ARMS_RUN if ARM[k]["part"] == "A"], t["seeds"])
    diff = [x for x in rows if x[2] != x[3]]
    print(f"  CHECK 151(b) fail_class_v2 vs v1 on every Part A record of this test: {len(rows)} records, disagreements "
          f"{len(diff)}  -> {'PASS' if not diff else 'FAIL'}")
    for k, s, a1, a2, eta, mg in diff:
        print(f"     {k}|{s}: v1 {a1}, v2 {a2}; eta^2 by key (multichannel) {eta}, margin {mg} "
              f"({'both >= the thresholds: the definitions differ here by design' if eta is not None and mg is not None and eta >= fc2.ETA_SPLIT and mg >= fc2.MARGIN_PARTIAL else 'not that case: a bug'})")
    print()
    print("  Part A per seed: WIN3_SLOW_D8 | WIN3_RESET_D8 | WIN3_SPLIT_D8 (transition); Part B: SINGLE_DELTA_D8 | "
          "SINGLE_HEBB_D8:")

    def oc(k, r):
        if not ok_r(r):
            return "--"
        lab = outcome2(r) if ARM[k]["part"] == "A" else ("BOUND" if bound_r(r) else f"not bound ({r['acc']:.2f})")
        return f"{lab} ({fmt_step(r['transition'])})"
    for s in t["seeds"]["WIN3_SPLIT_D8"]:
        print(f"    s{s}  " + " | ".join(f"{oc(k, rA[k].get(s)):<30}" for k in ("WIN3_SLOW_D8", "WIN3_RESET_D8", "WIN3_SPLIT_D8"))
              + " || " + " | ".join(f"{oc(k, rA[k].get(s)):<26}" for k in ("SINGLE_DELTA_D8", "SINGLE_HEBB_D8")))
    print()
    for k in ("WIN3_SPLIT_D8", "WIN3_RESET_D8"):
        rows_all = [(s, r) for s, r in sorted(rA[k].items()) if ok_r(r)]
        fired = [c for _, r in rows_all for c in r.get("checks", []) if c["fired"]]
        print(f"  {k}: the trigger's {'splits' if k == 'WIN3_SPLIT_D8' else 'resets'} — {len(fired)} in {len(rows_all)} runs "
              f"(per run {[sum(1 for c in r.get('checks', []) if c['fired']) for _, r in rows_all]}); targets labelled ok "
              f"{sum(1 for c in fired if c['key_ok'])}/{len(fired)}")
        for s, r in rows_all:
            ch = r.get("checks", [])
            ev = [f"{c['step']}: {c['key_cs']}->{c['key_c0']} {'ok' if c['key_ok'] else 'NOT labelled'} "
                  f"(map {groups_str(c['map_before'])} -> {groups_str(c.get('map_after'))})" for c in ch if c["fired"]]
            bl = [f"{c['step']} ({c.get('blocked_by')})" for c in ch if c["blocked"]]
            print(f"    s{s}  {outcome2(r)} (transition {fmt_step(r['transition'])}); fired: {'; '.join(ev) or 'none'}; "
                  f"blocked: {', '.join(bl) or 'none'}")
            print(f"           maps at the checks (step: distinct channels / probe acc): "
                  + " ".join(f"{c['step']}:{len(set(c['map_before']))}/{c['acc']:.2f}{'*' if c['fired'] else ''}" for c in ch))
        print()
    for k in ("CEIL_C", "CEIL_DELTA_D8"):
        print(f"  {k} transitions: " + ", ".join(f"s{s} {fmt_step(r['transition']) if ok_r(r) else '--'}"
                                               for s, r in sorted(rA[k].items())))
    print("  the k = 1 arms' final held-out accuracy by stream (streams 0-7):")
    for k in ("SINGLE_DELTA_D8", "SINGLE_HEBB_D8"):
        for s in t["seeds"][k]:
            r = rA[k].get(s)
            if ok_r(r):
                print(f"    {k:<16} s{s}  overall {r['acc']:.4f}  by stream {stream_str(r)}  "
                      f"{'BOUND at ' + str(r['transition']) if bound_r(r) else 'not bound'}")
    cpus = {}
    for k in KEYS:
        for r in rA[k].values():
            if ok_r(r):
                cpus.setdefault(r.get("cpu"), set()).add(k)
    print(f"  CPU labels in the records: {{{', '.join(f'{c!r}: {len(v)} arms' for c, v in cpus.items())}}}")
    print()

    if to is not None:
        print("=" * 100)
        print("X AND L SIDE BY SIDE (each machine's own seeds; pooled counts are the sums)")
        print("=" * 100)
        both = {machine: t, other_of(machine): to}
        print(f"  {'':44} {'X':>14} {'L':>14} {'pooled':>14}")
        for k in KEYS:
            cs = [(sum((bound_r(r) if ARM[k]['ceil'] else success(k, r)) for r in both[m_]["rA"][k].values()),
                   len(both[m_]["seeds"][k])) for m_ in MACHINES]
            cells = [f"{c}/{n}" for c, n in cs] + [f"{sum(c for c, _ in cs)}/{sum(n for _, n in cs)}"]
            lab = "perfect gate BOUND" if ARM[k]["ceil"] else out_of(k)
            print(f"  {k:<16} {lab:<27} " + " ".join(f"{c_:>14}" for c_ in cells))
        for name, *_ in CLAIMS + (C2,):
            ds = [both[m_]["claims"][name] for m_ in MACHINES]
            cells = [f"{d_['b']} vs {d_['c']}" for d_ in ds] + [f"{sum(d_['b'] for d_ in ds)} vs {sum(d_['c'] for d_ in ds)}"]
            print(f"  {name + ' discordant (first only vs second only)':<44} " + " ".join(f"{c_:>14}" for c_ in cells))
        print()

    print("=" * 100)
    print(f"CURVES — held-out accuracy x100 at every evaluation ({EVAL_EVERY} steps)")
    print("=" * 100)
    for k in KEYS:
        for s in t["seeds"][k]:
            r = get(store, k, s)
            if ok_r(r):
                lab = outcome2(r) if (ARM[k]["part"] == "A" and not ARM[k]["ceil"]) else ("BOUND" if bound_r(r) else "not bound")
                print(f"  {k:<16} s{s} {curve_str(r['curve'])}  -> {lab}")
    print()
    print(f"  total wall clock after verification: {wall / 60:.1f} min ({wall:.0f}s)   raw records: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    global SEEDS
    ap = argparse.ArgumentParser(description="The split's ingredient (copy vs reset) and a single delta channel at eight streams.")
    ap.add_argument("--machine", required=True, choices=MACHINES)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--results", default=RESULTS_FILE)
    ap.add_argument("--window", default=tgw.RESULTS_FILE, help="this machine's test_window_gate results (CHECKs 146, 151)")
    ap.add_argument("--early", default=None, help="this machine's test_early_recipe results (inherited CHECK 130)")
    ap.add_argument("--slow", default=tss.RESULTS_FILE, help="this machine's test_slow_start results (inherited)")
    ap.add_argument("--recipe", default=tsr.RESULTS_FILE, help="this machine's test_stream_recipe results (inherited)")
    ap.add_argument("--curriculum", default=tscur.RESULTS_FILE, help="this machine's test_stream_curriculum results (inherited)")
    ap.add_argument("--scale", default=tgw.tsa.RESULTS_FILE, help="this machine's test_scale_axes results (inherited)")
    ap.add_argument("--short", default=tsc.RESULTS_FILE, help="this machine's test_short_conv results (inherited)")
    ap.add_argument("--also", default=None, help="the other machine's split_copy_results.json (pooled claims)")
    ap.add_argument("--report", action="store_true", help="only the report, from a finished results file")
    args = ap.parse_args()
    torch.set_num_threads(1)
    SEEDS = seeds_for(args.machine)

    if args.report:
        store = load_store(args.results)
        m = store.get("meta", {})
        assert m.get("machine") == args.machine, f"{args.results} was written by machine {m.get('machine')}"
        SEEDS = {k: tuple(v) for k, v in m["seeds"].items()}
        report(store, SEEDS, args.machine, 0.0, args.results, args.also)
        return

    head = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    early_path = args.early or tgw.ter.RESULTS_FILE
    early = load_store(early_path)
    window = load_store(args.window)
    also = load_store(args.also) if args.also else None
    files = {"early": early, "early_path": early_path, "window_path": args.window}
    for k in ("slow", "recipe", "curriculum", "scale", "short"):
        files[k], files[k + "_path"] = load_store(getattr(args, k)), getattr(args, k)
    files["tss_files"] = {k: files[k] for k in ("scale", "recipe", "curriculum", "short")}
    files["tss_files"].update({k + "_path": files[k + "_path"] for k in ("scale", "recipe", "curriculum", "short")})
    if window.get("meta", {}).get("machine") != args.machine:
        print(f"STOP: {args.window} holds machine {window.get('meta', {}).get('machine')}'s test_window_gate runs")
        sys.exit(2)

    print("=" * 100)
    print("Split copy: the split's ingredient (copy vs reset) and a single delta channel, at eight streams")
    print(f"  machine {args.machine}: {cpu_model()}")
    print(f"  test_window_gate's Part C/D path: run_sc D8, S=8, P=4, conv, {ITERS} updates, all eight streams from update 1")
    for a in ARMS:
        s = SEEDS[a["key"]]
        print(f"  {a['label']:<100} seeds {s[0]}-{s[-1]} ({len(s)})")
    print(f"  --window     {args.window} (CPU {window['meta'].get('cpu')}, git {window['meta'].get('git')}, "
          f"{len(window.get('runs', {}))} records)")
    print(f"  --early      {early_path} (CPU {early['meta'].get('cpu')}, git {early['meta'].get('git')})")
    for k in ("slow", "recipe", "curriculum", "scale", "short"):
        m_ = files[k].get("meta", {})
        print(f"  --{k:<11} {files[k + '_path']} (CPU {m_.get('cpu')}, git {m_.get('git')}, {len(files[k].get('runs', {}))} records)")
    if also is not None:
        print(f"  --also       {args.also} (machine {also.get('meta', {}).get('machine')}, CPU {also.get('meta', {}).get('cpu')})")
    print(f"  torch {torch.__version__}   CPU {cpu_model()}   {args.workers} worker processes x 1 thread   git {head}")
    print("=" * 100)
    print()

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=init_worker) as pool:
        store = load_store(args.results)
        old_m = store.get("meta", {})
        if old_m and old_m.get("machine") not in (None, args.machine):
            print(f"STOP: {args.results} holds machine {old_m.get('machine')}'s runs")
            sys.exit(2)
        cached = {}
        if not args.force:
            for k in ("WIN3_SLOW_D8", "SINGLE_DELTA_D8"):
                hit = [(s, get(store, k, s)) for s in SEEDS[k] if ok_r(get(store, k, s))]
                if hit:
                    cached[k] = hit[0]
        verify(pool, files, args.machine, early, also, window, cached)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        meta = dict(machine=args.machine, torch=torch.__version__, cpu=cpu_model(), workers=args.workers, git=head, lr=LR,
                    beta=BETA, iters=ITERS, f_sha=F_SHA, window=dict(path=args.window, git=window["meta"].get("git")),
                    seeds={k: list(v) for k, v in SEEDS.items()}, started=now)
        store["meta"] = longrun.resume_meta(old_m, meta, head, args.workers, KEYS, force=args.force)
        t0 = time.time()

        print("=" * 100)
        print(f"PROJECTION — pool-load timing: each arm's real path for {TIME_ITERS} steps (one evaluation and its statistics "
              f"included), {args.workers} simultaneous copies on the {args.workers} workers, the median; worst case = every "
              f"run to its last step")
        print("=" * 100)
        cost = {}
        for k in KEYS:
            cost[k] = statistics.median(pool.map(time_job, [k] * args.workers))
        done = {(k, s) for k in KEYS for s in SEEDS[k] if get(store, k, s) is not None and not args.force}
        h_all, h_rest = longrun.projection([(k, cost[k], ITERS, SEEDS[k]) for k in KEYS], args.workers, done)
        print(f"  nothing is cut")
        store["meta"].update(projected_wall_h=h_all, projected_rest_h=h_rest, ms_per_step={k: cost[k] * 1000 for k in KEYS})
        save_results(args.results, store)
        print()

        def record(sp, rec):
            store["runs"][f"{sp['arm']}|{sp['seed']}"] = rec
            save_results(args.results, store)
            k = sp["arm"]
            if not rec.get("ok"):
                line = f"*** FAILED: {rec.get('error')}"
            else:
                tr = rec["transition"]
                lab = (outcome2(rec) if (ARM[k]["part"] == "A" and not ARM[k]["ceil"])
                       else ("BOUND" if bound_r(rec) else "not bound"))
                fired = ([c["step"] for c in rec["checks"] if c["fired"]] if rec.get("checks") is not None else None)
                line = (f"{'BOUND at ' + str(tr) if tr else 'not bound':<16} acc={rec['acc']:.4f} @{rec['stopped_at']:<5} "
                        f"{lab}" + (f"  {rec.get('trigger_op')}s at {fired}" if fired is not None else ""))
            print(f"   {k:<16} seed {sp['seed']}  {line}  {rec.get('secs', 0.0):.0f}s  (elapsed {(time.time() - t0) / 60:.1f}m)",
                  flush=True)

        specs = []
        for k in sorted(KEYS, key=lambda k: (not ARM[k]["ceil"], -cost[k])):
            specs += [spec(k, s, ITERS) for s in SEEDS[k]]
        todo = []
        for sp in specs:
            if get(store, sp["arm"], sp["seed"]) is not None and not args.force:
                print(f"   {sp['arm']:<16} seed {sp['seed']}  cached")
            else:
                todo.append(sp)
        print("=" * 100)
        print(f"RUNS — {len(todo)} runs, the perfect gates first, then longest first")
        print("=" * 100)
        pending, inflight = list(todo), {}
        while pending or inflight:
            while pending and len(inflight) < args.workers:
                sp = pending.pop(0)
                inflight[pool.submit(run_job, sp)] = sp
            done_, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fu in done_:
                sp = inflight.pop(fu)
                try:
                    rec = fu.result()
                except Exception as e:
                    rec = dict(ok=False, seed=sp["seed"], error=f"worker {type(e).__name__}: {e}")
                record(sp, rec)
        print()

    report(store, SEEDS, args.machine, time.time() - t0, args.results, args.also)


if __name__ == "__main__":
    main()
