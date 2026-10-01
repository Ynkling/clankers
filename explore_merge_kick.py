#!/usr/bin/env python
"""
explore_merge_kick.py — EXPLORATORY, not a result. Screen S21 (batch 7): can large gate kicks split
two streams merged in one channel at S=4, k=4?

BACKGROUND (post hoc)
- Main line, X's recorded S=4, k=4 runs (main branch, results/X/stream_recipe_results.json and
  scale_axes_results.json): of 22 runs with streams sharing a channel at step 9600, 10 split later
  and bound (as late as 27600), 12 never did.
- S16: random kicks sized as the hinge gradient did about as well as the hinge on the grouped
  layout (12/20 vs 14/20; SLOW_MEM 4/20). Under Adam, one gradient far above a parameter's running
  gradient moves it by roughly the same amount whatever its size.

THE CODE PATH. The runs are main-branch runs. test_scale_axes.py, test_stream_channels.py and
test_stream_recipe.py exist only on the main branch (claude/bdh-growth-hebbian-inference-w90069);
they are loaded READ-ONLY from its commit MAIN_SHA (git show; no file is added to or changed on this
branch) by an import hook, and so are the recorded results. Every other module they import is this
branch's: identical to main's at MAIN_SHA except test_binding_capacity.py (main adds the opt-in
s_active knob; the default path is unchanged) and test_curriculum_confirm.py (docstring only); the
CHECK lists the differences, and the reruns must reproduce the recorded curves.

RUNS: X's recorded A4k4 runs (arm A, k=4, conv 'layer' width 4, S=4, P=4, task_for(4, 4), lr 1e-3)
merged at 9600 that stayed merged to the end, two streams on one channel at 0.74-0.77:
  stream_recipe (test_stream_recipe's plain A4k4, its run path run_attempt):  247, 248, 251, 258
  scale_axes    (test_scale_axes's A4k4, its run path run_one as run_job):    221, 223, 225, 226
"The merged pair" = the two streams sharing a channel in the recorded end map (the CHECK derives
the list from the records: A4k4, exactly two streams sharing at 9600 and at the end, not bound,
final accuracy 0.74-0.77 to two decimals).

PROTOCOL, per run: re-run to step T0 = 9600 through the recorded run's own path (the model kept by
run_one's keep, the optimizer caught by a step pre-hook, the batch generator rebuilt by replaying
its T0 batches from seed + 10_000), then continue T_CONT = 9600 updates twice from that state, each
with its own deep copy of the model and of the optimizer state (copied, not shared) and the same
batch stream (onset_run's step, line for line):
  (a) CONTROL: plain;
  (b) KICKS: every 600 updates (16 kicks), after loss.backward(), add to the .grad of W_in, W_h and
      W_g a Gaussian tensor (dedicated generator seeded KICK_BASE + 100_000 * seed + update; the
      three tensors drawn in that order) scaled to 100x that tensor's gradient norm on that batch.
At every evaluation (every 1200): the stream -> channel map (test_scale_axes.routing_k), the merged
pair's channels, per-stream (test_scale_axes.stream_acc) and overall held-out accuracy.
Per-run reading: "split" if at some evaluation up to 19200 the merged pair is on different
channels and the held-out accuracy is >= 0.95; otherwise "merged". (The state at 19200 is printed
too.)

RULE (fixed before any run): "kicks split merges" if >= 4/8 split under KICKS; "they do not" if
<= 1/8; otherwise neither.

CHECKS: before the batch: the 8 runs are the derived list; the main modules load from MAIN_SHA and
the shared modules differ from main's only as stated; the machinery on one run of each path (a
rerun to 1200 reproduces the record; CONTROL's 1200 continued updates reproduce the recorded
evaluation at 2400; KICKS logs 2 kicks at 100x the batch gradient norm; identical batches; the
continuations start from optimizer states equal to the run's, stored apart from it, and the run's
optimizer is left unchanged; a load without a deep copy, as in batch 5's S14, is caught). In each job: the rerun reproduces its recorded curve to 9600; CONTROL reproduces it to
19200; the continuations see identical batches (digests) and start from equal, separately stored
optimizer states; every kick's norm equals 100x the batch gradient norm.
"""

import ast
import copy
import hashlib
import importlib.abc
import importlib.util
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F
from torch.optim.optimizer import register_optimizer_step_pre_hook

import explore_common as ec
import explore_common2 as c2
import test_binding_onset as tbo
from test_binding_onset import BATCH, EVAL_EVERY
from test_multilayer_binding import probe_batch
from test_channel_binding import eval_batch
from test_instrument_v2 import TAU_END

NAME = "merge_kick"
IDEA = "large gate kicks (100x the batch gradient, every 600 updates) on S=4 runs with two streams merged"
SOURCE = ("the main line's S=4, k=4 runs (merges at 9600: 10 of 22 split later); batch 6's S16 (a kick's "
          "size, not its direction, did most of the hinge's work)")
CHANGE = ("none to the recorded runs up to 9600; then 9600 updates twice from that state: CONTROL plain, "
          "KICKS with a Gaussian kick of 100x the gradient norm on W_in, W_h, W_g every 600 updates")
PAIRING = "each run's KICKS against its own CONTROL (same state at 9600, same batches) and its recorded curve"
MAIN_REF = "claude/bdh-growth-hebbian-inference-w90069"
MAIN_SHA = "520fe510cace8e4f698c1dfd390438281163e6df"
MAIN_MODULES = ("test_scale_axes", "test_stream_channels", "test_stream_recipe")
T0, T_CONT = 9600, 9600
KICK_EVERY, KICK_MULT = 600, 100.0
KICK_BASE = 8_000_000
KICK_PARAMS = ("W_in", "W_h", "W_g")
BIND = 0.95
SPLIT_MIN, NOT_MAX = 4, 1
SEEDS_SR, SEEDS_SA = (247, 248, 251, 258), (221, 223, 225, 226)
ACC_LO, ACC_HI = 0.74, 0.77
DIFF_EXPECTED = {"test_binding_capacity.py", "test_curriculum_confirm.py"}


# ── read-only access to the main branch ─────────────────────────────────────
def _git(*a, text=True):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=text, check=True).stdout


def ensure_main():
    if subprocess.run(["git", "-C", HERE, "cat-file", "-e", f"{MAIN_SHA}^{{commit}}"],
                      capture_output=True).returncode != 0:
        subprocess.run(["git", "-C", HERE, "fetch", "-q", "origin", MAIN_REF], check=True)


def main_blob(path):
    ensure_main()
    return _git("show", f"{MAIN_SHA}:{path}")


class _MainLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        src = main_blob(module.__name__ + ".py")
        module.__file__ = os.path.join(HERE, module.__name__ + ".py")   # main's file name, not on disk
        module.__main_sha__ = MAIN_SHA
        exec(compile(src, f"<{MAIN_SHA[:7]}:{module.__name__}.py>", "exec"), module.__dict__)


class _MainFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name in MAIN_MODULES:
            return importlib.util.spec_from_loader(name, _MainLoader(), origin=f"git {MAIN_SHA[:7]}:{name}.py")
        return None


if not any(isinstance(f, _MainFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _MainFinder())

import test_scale_axes as tsa                                      # noqa: E402  (main, read-only)
import test_stream_recipe as tsr                                   # noqa: E402  (main, read-only)
from test_router_reliability import run_one                        # noqa: E402
from test_short_conv import conv_grad_norms                        # noqa: E402

TASK = tsa.TASK44
_REC = {}


def recorded_store(name):
    if name not in _REC:
        _REC[name] = json.loads(main_blob(f"results/X/{name}_results.json"))["runs"]
    return _REC[name]


SRC = {"SR_A4k4": dict(store="stream_recipe", key="A4k4", path="test_stream_recipe.run_attempt (plain A4k4)"),
       "SA_A4k4": dict(store="scale_axes", key="A4k4", path="test_scale_axes.run_job (run_one)")}

_A = dict(tsa.ARM["A4k4"], lr=tsa.LR, iters=T0 + 2 * T_CONT, prio=1,
          sched=f"lr {tsa.LR:g} throughout; from {T0}: CONTROL plain, KICKS + Gaussian kick of {KICK_MULT:g}x the "
                f"batch gradient norm on W_in, W_h, W_g every {KICK_EVERY} updates")
ARMS = {
    "SR_A4k4": dict(_A, key="SR_A4k4", seeds=SEEDS_SR, label="test_stream_recipe's A4k4 (X), merged pair stalled"),
    "SA_A4k4": dict(_A, key="SA_A4k4", seeds=SEEDS_SA, label="test_scale_axes's A4k4 (X), merged pair stalled"),
}


def recorded(arm, seed):
    return recorded_store(SRC[arm]["store"])[f"{SRC[arm]['key']}|{seed}"]


def stat_at(r, step):
    return next((s for s in r["stats"] if s["step"] == step), None)


def derived_runs():
    out = {}
    for arm, src in SRC.items():
        seeds = []
        for k, r in recorded_store(src["store"]).items():
            if not k.startswith("A4k4|"):
                continue
            s96 = stat_at(r, T0)
            if (r.get("ok") and r["transition"] is None and s96 and s96.get("shared") == 2
                    and r["end"].get("shared") == 2 and ACC_LO <= round(r["acc"], 2) <= ACC_HI):
                seeds.append(int(k.split("|")[1]))
        out[arm] = tuple(sorted(seeds))
    return out


def merged_pair(r):
    cm = r["end"]["ch_map"]
    pair = [s for s in range(len(cm)) if cm.count(cm[s]) > 1]
    assert len(pair) == 2, cm
    return pair


# ── the protocol ─────────────────────────────────────────────────────────────
def rerun(arm, seed, t0):
    """The recorded run's own path to step t0; returns the run's record, model, optimizer, rng."""
    holder, keep = {}, {"at": t0}

    def hook(opt, args, kwargs):
        holder["opt"] = opt

    h = register_optimizer_step_pre_hook(hook)
    try:
        if arm == "SR_A4k4":
            rec = tsr.run_attempt(tsr.ARM["A4k4"], seed, dict(tsr.REAL, total=t0), keep=keep)
        else:
            a = tsa.ARM["A4k4"]
            rec = run_one(a, seed, t0, task=tsa.TASKS[a["task"]], stats_fn=tsa.scale_stats,
                          grad_fn=conv_grad_norms, lr=tsa.LR, builder=tsa.builder_for(a), keep=keep)
    finally:
        h.remove()
    g = torch.Generator()
    g.manual_seed(seed + 10_000)                                     # onset_run's batch generator
    for _ in range(t0):
        TASK.make_batch(BATCH, g)
    return rec, keep["model"], holder["opt"], g


def opt_state(opt):
    return [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in opt.state[p].items()}
            for gr in opt.param_groups for p in gr["params"]]


def same_state(a, b):
    return len(a) == len(b) and all(
        x.keys() == y.keys() and all(torch.equal(x[k], y[k]) if torch.is_tensor(x[k]) else x[k] == y[k]
                                     for k in x) for x, y in zip(a, b))


def ptrs(opt):
    return {v.data_ptr() for gr in opt.param_groups for p in gr["params"] for v in opt.state[p].values()
            if torch.is_tensor(v)}


def continuation(m0, opt0, rng0, seed, t0, steps, kicks, data, probe, lr, ref):
    """onset_run's step, line for line, from (m0, opt0, rng0), on deep copies; with kicks, every
    KICK_EVERY updates a Gaussian kick of KICK_MULT x the batch gradient norm on W_in, W_h, W_g.
    ref: the run's optimizer state snapshotted before any continuation. The copy's state must equal
    it and share no storage with the run's optimizer (alive throughout); the two continuations never
    coexist, so they cannot share storage with each other."""
    m = copy.deepcopy(m0)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    opt.load_state_dict(copy.deepcopy(opt0.state_dict()))
    start = dict(equal=same_state(ref, opt_state(opt)), separate=not (ptrs(opt) & ptrs(opt0)))
    g = torch.Generator()
    g.set_state(rng0.get_state())
    named = dict(m.named_parameters())
    dig, curve, evals, klog = hashlib.sha256(), [], [], []
    for step in range(1, steps + 1):
        tokens, _ = TASK.make_batch(BATCH, g)
        dig.update(tokens.numpy().tobytes())
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        ql, qt, _ = TASK.select(m(inp, TAU_END)[0], tgt, None)
        loss = F.cross_entropy(ql, qt)
        opt.zero_grad(); loss.backward()
        if kicks and step % KICK_EVERY == 0:
            kg = torch.Generator().manual_seed(KICK_BASE + 100_000 * seed + step)
            e = dict(update=t0 + step, grad_norm=[], kick_norm=[])
            for n in KICK_PARAMS:
                p = named[n]
                gn = p.grad.norm()
                z = torch.randn(p.shape, generator=kg, dtype=p.dtype)
                k = z * (KICK_MULT * gn / z.norm())
                p.grad.add_(k)
                e["grad_norm"].append(float(gn))
                e["kick_norm"].append(float(k.norm()))
            klog.append(e)
        opt.step()
        if step % EVAL_EVERY == 0:
            acc, el = tbo.evaluate(m, TASK, data)
            curve.append([t0 + step, acc, el])
            rk = tsa.routing_k(m, TASK, probe)
            evals.append(dict(step=t0 + step, acc=acc, ch_map=rk["ch_map"], shared=rk["shared"],
                              stream_acc=tsa.stream_acc(m, TASK, data)))
    return dict(curve=curve, evals=evals, digest=dig.hexdigest(), kicks=klog, start_equal=start["equal"],
                start_separate=start["separate"]), m


def kick_diff(klog):
    d = 0.0
    for e in klog:
        for gn, kn in zip(e["grad_norm"], e["kick_norm"]):
            want = KICK_MULT * gn
            d = max(d, abs(kn - want) / want if want > 0 else abs(kn))
    return d


def split_at(evals, pair):
    for e in evals:
        if e["ch_map"][pair[0]] != e["ch_map"][pair[1]] and e["acc"] >= BIND:
            return e["step"]
    return None


def run_job(arm, seed):
    a = ARMS[arm]
    dry = a["iters"] < T0 + 2 * T_CONT
    t0, cont = (2 * EVAL_EVERY, EVAL_EVERY) if dry else (T0, T_CONT)
    rec0 = recorded(arm, seed)
    pair = merged_pair(rec0)
    want0 = [c for c in rec0["curve"] if c[0] <= t0]
    want1 = [c for c in rec0["curve"] if t0 < c[0] <= t0 + cont]
    rec, m, opt, rng = rerun(arm, seed, t0)
    data, probe = eval_batch(TASK), probe_batch(TASK, seed)
    snap = opt_state(opt)
    ctrl, _ = continuation(m, opt, rng, seed, t0, cont, False, data, probe, a["lr"], snap)
    kick, mk = continuation(m, opt, rng, seed, t0, cont, True, data, probe, a["lr"], snap)
    end = tsa.scale_stats(mk, TASK, probe, "end")
    out = dict(ok=True, seed=seed, k=4, dry=dry, t0=t0, cont=cont, merged_pair=pair,
               recorded_end_map=rec0["end"]["ch_map"], recorded_acc=rec0["acc"],
               rerun_curve=rec["curve"], rerun_map=(stat_at(rec, t0) or {}).get("ch_map"),
               reproduces=rec["curve"] == want0,
               control=ctrl, kicks=kick,
               control_matches=ctrl["curve"] == want1 and len(want1) == len(ctrl["curve"]),
               batches_identical=ctrl["digest"] == kick["digest"],
               opt_separate=(ctrl["start_equal"] and kick["start_equal"] and ctrl["start_separate"]
                             and kick["start_separate"]),
               opt_unchanged=same_state(snap, opt_state(opt)),
               n_kicks=len(kick["kicks"]), kick_norm_diff=kick_diff(kick["kicks"]),
               split_control=split_at(ctrl["evals"], pair), split_kicks=split_at(kick["evals"], pair))
    # the record's summary fields describe KICKS's end state (the progress line and tag)
    out.update(curve=[c for c in rec["curve"]] + kick["curve"], acc=kick["curve"][-1][1] if kick["curve"] else None,
               val_cos=end["val_cos"], key_cos=end["key_cos"], ctx_cos=end["ctx_cos"], end=end, stats=[])
    out["transition"] = tbo.transition(out["curve"])
    out["reading"] = "split" if out["split_kicks"] is not None else "merged"
    return out


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], TASK, tsa.builder_for(tsa.ARM["A4k4"])(tsa.ARM["A4k4"], a["seeds"][0]), a["lr"],
             tsa.scale_stats)]


# ── CHECKs ───────────────────────────────────────────────────────────────────
def shared_diffs():
    ensure_main()
    names = _git("diff", "--name-only", MAIN_SHA, "--", "test_*.py").split()
    here = [n for n in names if os.path.exists(os.path.join(HERE, n))]
    doc_only = {}
    for n in here:
        a, b = ast.parse(open(os.path.join(HERE, n)).read()), ast.parse(main_blob(n))
        strip = lambda t: ast.dump(ast.Module(body=t.body[1:] if (t.body and isinstance(t.body[0], ast.Expr)
                                                                   and isinstance(t.body[0].value, ast.Constant))
                                              else t.body, type_ignores=[]))
        doc_only[n] = strip(a) == strip(b)
    return sorted(here), doc_only


def check():
    ok = True
    der = derived_runs()
    runs_ok = der["SR_A4k4"] == SEEDS_SR and der["SA_A4k4"] == SEEDS_SA
    pairs = {f"{arm[:2]} {s}": merged_pair(recorded(arm, s)) for arm, d in der.items() for s in d}
    loaded = all(getattr(sys.modules[n], "__main_sha__", None) == MAIN_SHA for n in MAIN_MODULES)
    here, doc_only = shared_diffs()
    diffs_ok = set(here) == DIFF_EXPECTED and doc_only.get("test_curriculum_confirm.py") and \
        not doc_only.get("test_binding_capacity.py")
    rows = [(f"the runs are the recorded A4k4 runs with exactly two streams sharing a channel at {T0} and at the "
             f"end, not bound, final accuracy {ACC_LO}-{ACC_HI}: stream_recipe {der['SR_A4k4']}, scale_axes "
             f"{der['SA_A4k4']}; merged pairs {pairs}", runs_ok),
            (f"{', '.join(MAIN_MODULES)} load read-only from main at {MAIN_SHA[:7]} (git show; not on this branch)",
             loaded),
            (f"this branch's test modules that differ from main's at {MAIN_SHA[:7]}: {here} "
             f"(test_curriculum_confirm.py docstring only: {doc_only.get('test_curriculum_confirm.py')}; "
             f"test_binding_capacity.py: main adds the opt-in s_active knob)", diffs_ok)]
    for arm, seed in (("SA_A4k4", SEEDS_SA[0]), ("SR_A4k4", SEEDS_SR[0])):
        rec0 = recorded(arm, seed)
        t0, cont = EVAL_EVERY, EVAL_EVERY
        rec, m, opt, rng = rerun(arm, seed, t0)
        data, probe = eval_batch(TASK), probe_batch(TASK, seed)
        snap = opt_state(opt)
        ctrl, _ = continuation(m, opt, rng, seed, t0, cont, False, data, probe, ARMS[arm]["lr"], snap)
        kick, _ = continuation(m, opt, rng, seed, t0, cont, True, data, probe, ARMS[arm]["lr"], snap)
        want0 = [c for c in rec0["curve"] if c[0] <= t0]
        want1 = [c for c in rec0["curve"] if t0 < c[0] <= t0 + cont]
        good = (rec["curve"] == want0 and ctrl["curve"] == want1 and ctrl["digest"] == kick["digest"]
                and ctrl["start_equal"] and kick["start_equal"] and ctrl["start_separate"]
                and kick["start_separate"] and same_state(snap, opt_state(opt))
                and len(kick["kicks"]) == cont // KICK_EVERY and kick_diff(kick["kicks"]) < 1e-5)
        rows.append((f"{SRC[arm]['path']} seed {seed}: the rerun to {t0} reproduces the record "
                     f"({rec['curve'] == want0}); CONTROL's {cont} continued updates reproduce the recorded evaluation "
                     f"at {t0 + cont} ({ctrl['curve']} vs {want1}); identical batches ({ctrl['digest'] == kick['digest']}); "
                     f"optimizer states equal to the run's ({ctrl['start_equal']}, {kick['start_equal']}) and stored apart "
                     f"from it ({ctrl['start_separate']}, {kick['start_separate']}), the run's left unchanged "
                     f"({same_state(snap, opt_state(opt))}); KICKS logged {len(kick['kicks'])} kicks at "
                     f"{KICK_MULT:g}x the batch gradient norm (max relative difference {kick_diff(kick['kicks']):.1e})",
                     good))
        if arm == "SA_A4k4":
            shared = torch.optim.Adam(m.parameters(), lr=ARMS[arm]["lr"])
            shared.load_state_dict(opt.state_dict())                    # S14's load: shares the tensors
            teeth = bool(ptrs(shared) & ptrs(opt))
            rows.append(("the separation test has teeth: a state loaded without a deep copy (batch 5's S14 "
                         f"load) shares storage with the run's optimizer ({teeth})", teeth))
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def rule(n):
    return "kicks split merges" if n >= SPLIT_MIN else ("they do not" if n <= NOT_MAX else "neither")


def ev_str(evals, pair):
    return " ".join(f"{e['acc']:.2f}:{''.join(str(c) for c in e['ch_map'])}"
                    + ("|" if e["ch_map"][pair[0]] != e["ch_map"][pair[1]] else "") for e in evals)


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    print(f"  per run (map = stream -> channel at each evaluation, as acc:map; '|' = the merged pair on different "
          f"channels; continuations from {T0} to {T0 + T_CONT}):")
    n_split, n_ctrl, n_done = 0, 0, 0
    for arm, a in ARMS.items():
        for s in a["seeds"]:
            r = store["runs"].get(f"{arm}|{s}")
            if r is None or not r.get("ok"):
                print(f"    {arm:<8} {s:>4} {ec.tag(r)}")
                continue
            n_done += 1
            pair = r["merged_pair"]
            chk = (f"rerun to {r['t0']} reproduces: {'yes' if r['reproduces'] else 'NO'}; CONTROL = record to "
                   f"{r['t0'] + r['cont']}: {'yes' if r['control_matches'] else 'NO'}; batches identical: "
                   f"{'yes' if r['batches_identical'] else 'NO'}; optimizer states equal and separate: "
                   f"{'yes' if r['opt_separate'] else 'NO'} (run's unchanged: {'yes' if r['opt_unchanged'] else 'NO'}); "
                   f"kicks {r['n_kicks']}, max norm diff {r['kick_norm_diff']:.1e}")
            print(f"    {arm:<8} {s:>4}  merged pair streams {pair} (recorded end map {r['recorded_end_map']}, rec acc "
                  f"{r['recorded_acc']:.3f}; map at {r['t0']} {r['rerun_map']})")
            print(f"    {'':<8} {'':>4}  {chk}")
            for lab, c in (("CONTROL", r["control"]), ("KICKS", r["kicks"])):
                last = c["evals"][-1] if c["evals"] else None
                sa = "/".join(f"{v:.2f}" for v in last["stream_acc"]) if last else "--"
                print(f"    {'':<8} {'':>4}  {lab:<7} {ev_str(c['evals'], pair)}")
                print(f"    {'':<8} {'':>4}  {'':<7} at {last['step'] if last else '--'}: acc {last['acc'] if last else float('nan'):.3f}, "
                      f"per stream {sa}, merged pair on channels "
                      f"{[last['ch_map'][pair[0]], last['ch_map'][pair[1]]] if last else '--'}")
            gn = [e["grad_norm"] for e in r["kicks"]["kicks"]]
            if gn:
                print(f"    {'':<8} {'':>4}  KICKS' batch gradient norms W_in/W_h/W_g at the first kick "
                      f"{'/'.join(f'{v:.1e}' for v in gn[0])}, at the last {'/'.join(f'{v:.1e}' for v in gn[-1])}")
            print(f"    {'':<8} {'':>4}  READING: KICKS {r['reading']}"
                  + (f" (at {r['split_kicks']})" if r["split_kicks"] else "")
                  + f"; CONTROL {'split' if r['split_control'] else 'merged'}"
                  + (f" (at {r['split_control']})" if r["split_control"] else ""))
            n_split += r["split_kicks"] is not None
            n_ctrl += r["split_control"] is not None
    rd = rule(n_split)
    print(f"  SPLIT under KICKS {n_split}/{n_done}; under CONTROL {n_ctrl}/{n_done}")
    print(f"  RULE S21 ('kicks split merges' if >= {SPLIT_MIN}/8 split under KICKS, 'they do not' if <= {NOT_MAX}/8): {rd}")
    return dict(n=n_done, split=n_split, split_control=n_ctrl, rule=rd)
