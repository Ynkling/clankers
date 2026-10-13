#!/usr/bin/env python
"""
explore_gpu/s75b_checks.py — the CHECKs of explore_gpu/s75b.py (the S75b spec at 114cd08; the changes since, not blind,
are s75b.CHANGES_SINCE_SPEC). EXPLORATORY tooling.

    python -m explore_gpu.s75b_checks --device cpu     (this container; output explore_gpu/s75b_checks_cpu_output.txt / .json)
    python -m explore_gpu.s75b_checks --device cuda    (each card, before its first stage; output
                                                        explore_gpu/s75b_checks_cuda_<GPU>_output.txt / .json: commit it,
                                                        stage 1 and stage 2 refuse without it)

  B1  the task CHECK at configuration B: explore_i_task.check_task at its default n = 4096 (S75's n = 512 fails at B on a
      tolerance).                                                                                                  cpu
  B2  the guarded trigger's mechanics (threshold 1.01, rise 1.0; the source -> channel map scripted per check through
      explore_i_common.routing_layer): a check on a one-to-one map is blocked "one-to-one" and leaves the fire count and
      the last-fire step unchanged and W_g and its Adam state untouched; a run blocked at 4800 whose map is merged at 7200
      fires at 7200; gap and cap come before one-to-one (9600 and 21600 one-to-one, blocked by gap and cap); every row
      records map_eta / map_margin / map_one_to_one (the last as scripted); with every map merged the guarded trigger's
      rows equal explore_i_common.make_split's on the same model (the fields both record), the two models end bitwise
      equal and so do W_g's Adam states (the second optimizer loaded from a deep copy of the first's state).          cpu
  B3  the routing margin in chunks of 16 sequences equals the 64-sequence chunks to 1e-12 (relative), at L = 1536.  cpu
  B4  the rules on constructed inputs: proceed iff mean >= 0.912 and each >= 0.9 (edges included: mean exactly 0.912,
      a run exactly 0.9); SPLIT_THR = min - 0.03 unrounded; a missing or failed record never proceeds; the stop's
      wording; register_rule (scan fits / does not fit / failed); held_evals, workers_rule and ram_budget.            cpu
  B5  refusals (stage 1 and stage 2) in a temporary output directory: no record; the record, the pilot results file
      not committed (a temporary git repository: untracked, committed, modified; the committed pilot records disagreeing);
      record != pilot; an incomplete pilot; pilot records under another code SHA; the pilot stopped; records of one seed
      from two cards; a machine's arms with records from another card; a card that is not the machine's; on a (faked)
      CUDA card: no benchmark, a failed benchmark, a refusing worker rule, no CUDA CHECK output; pilot records from
      another card at stage 1; the machine -> arms allocation; git_committed itself.                                 cpu
  B6  S75's code SHA is still 5fb120a92d69 (S75b imports S75's modules and changes none).                            cpu
  B7  LOCK-STEP at L = 1536, R = 1 (as the runs): batched (explore_gpu.batched) vs sequential (the project's model) on
      the same device, explore_gpu.checks.Judge at same-device tolerances, each seed alone; per update the batched
      forward and backward first (graph freed), then the sequential one, then the float64 arbitration if needed.
      ORACLE, SINGLE, WIDE (WIDE_SPLIT_G's and WIDE_SLOW's model), REG3 with the loop, REG3 with the scan (against the
      sequential loop). cpu: 2 updates on seed 850 at batch 32, every update within the FAST tolerance (the sign test
      cannot reject in 2 updates). cuda: 20 updates on seeds 850 and 851 each, batch 8 (the batch-32 float64
      arbitration does not fit either card). A control: the same lock-step with the batched gradient of W_g doubled
      must fail (small configuration).                                                                      cpu and cuda
  B8  the whole driver end to end on a small configuration (S = 4, NB = 3, M = 30; 1200 updates; evaluation every 400;
      the split's schedule 400 / 400 / 800, rise 1.0; the proceed bar at 0 so stage 2 runs): stage 1, the stage-1 record,
      stage 2 (fallback: every arm; every WIDE_SPLIT_G row records the map), the report (its sections); the guarded
      trigger inside run_batched_icmc (split schedule 2 / 2 / 4, threshold 1.01, R = 2): each run's eligible check
      fires (W_g's Adam state zeroed, the rows moved) or is blocked one-to-one on a one-to-one map; the benchmark path
      on the small configuration (both register modes, the host-memory measurements and the worker rule).            cpu
"""

import argparse
import contextlib
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F

import explore_i_task as it
import explore_i_common as ic
from explore_gpu import device as dv
from explore_gpu import run as gr
from explore_gpu import s75b as sb
from explore_gpu import checks as ck
from explore_gpu.batched import BatchedBDH

SMALL = dict(S=4, V=4, NB=3, lmin=6, lmax=14, M=30)


def b1_task(dev):
    if dev.type != "cpu":
        return dict(ok=True, skipped="cpu only")
    r = it.check_task(sb.task())
    return dict(ok=all(r.values()), failed=[k for k, v in r.items() if not v], n=4096)


@contextlib.contextmanager
def scripted_routing(script):
    """explore_i_common.routing_layer -> the real statistics with ch_map / one_to_one replaced by script[step]."""
    real = ic.routing_layer
    cur = {}

    def rl(g, x, task):
        out = real(g, x, task)
        cmap = script[cur["step"]]
        out.update(ch_map=list(cmap), one_to_one=len(set(cmap)) == task.S)
        return out
    ic.routing_layer = rl
    try:
        yield cur
    finally:
        ic.routing_layer = real


def _wide_model(task, seed=7):
    torch.manual_seed(seed)
    m = gr.make_model(gr.arm_spec("WIDE_SPLIT", task.S), task, seed)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = task.make_batch(8, torch.Generator().manual_seed(11))[0]
    lo = m(x[:, :-1])[0]
    mask = gr.icmc_loss_mask(task, x)
    loss = torch.nn.functional.cross_entropy(lo[mask], x[:, 1:][mask])
    opt.zero_grad()
    loss.backward()
    opt.step()
    return m, opt


def b2_guard(dev):
    if dev.type != "cpu":
        return dict(ok=True, skipped="cpu only")
    task = it.ICMCTask(**SMALL)
    one, merged = [0, 1, 2, 3], [0, 0, 2, 3]
    steps = list(range(2400, 26400 + 1, 2400))
    script = {2400: one, 4800: one, 7200: merged, 9600: one, 12000: one, 14400: merged, 16800: merged,
              19200: merged, 21600: one, 24000: merged, 26400: merged}
    out = {}
    with scripted_routing(script) as cur:
        m, opt = _wide_model(task)
        trig = sb.make_split_guarded(task, 7, 26400, 1.01, rise=1.0)
        snap = None
        for t in steps:
            cur["step"] = t
            if t == 4800:
                snap = (m.W_g.detach().clone(), {k: v.clone() for k, v in opt.state[m.W_g].items() if torch.is_tensor(v)})
            trig(m, t, opt)
            if t == 4800:
                out["w_g_untouched_at_4800"] = torch.equal(snap[0], m.W_g.detach()) and all(
                    torch.equal(v, opt.state[m.W_g][k]) for k, v in snap[1].items())
        rows = {r["step"]: r for r in trig.rows}
    fired = [t for t, r in rows.items() if r["fired"]]
    by = {t: r.get("blocked_by") for t, r in rows.items() if r.get("blocked")}
    out.update(fired=fired, blocked=by)
    want_fired = [7200, 14400, 19200]
    want_blocked = {4800: "one-to-one", 9600: "gap", 12000: "one-to-one", 16800: "gap", 21600: "cap", 24000: "cap",
                    26400: "cap"}
    fields = all({"map_eta", "map_margin", "map_one_to_one"} <= set(r) for r in rows.values())
    scripted = all(r["map_one_to_one"] == (len(set(script[t])) == task.S) for t, r in rows.items())
    out.update(map_fields_every_row=fields, map_one_to_one_as_scripted=scripted)
    ok = fired == want_fired and by == want_blocked and out["w_g_untouched_at_4800"] and fields and scripted
    # every map merged: the guarded trigger's rows equal explore_i_common.make_split's, the models and W_g's Adam
    # states end bitwise equal (the second optimizer from a deep copy of the first's state: no shared tensors)
    allm = {t: merged for t in steps}
    with scripted_routing(allm) as cur:
        m1, o1 = _wide_model(task)
        m2 = ck._copy_model(m1)
        o2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
        o2.load_state_dict(copy.deepcopy(o1.state_dict()))
        shared = any(a.data_ptr() == b.data_ptr() for a, b in zip(
            [v for v in o1.state[m1.W_g].values() if torch.is_tensor(v)],
            [v for v in o2.state[m2.W_g].values() if torch.is_tensor(v)]))
        tg = sb.make_split_guarded(task, 7, 26400, 1.01, rise=1.0)
        tm = ic.make_split(task, 7, 26400, 1.01, rise=1.0)
        for t in steps:
            cur["step"] = t
            tg(m1, t, o1)
            tm(m2, t, o2)
    same_rows = len(tg.rows) == len(tm.rows) and all(all(a.get(k) == v for k, v in b.items())
                                                       for a, b in zip(tg.rows, tm.rows))
    same_model = all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))
    s1, s2 = o1.state[m1.W_g], o2.state[m2.W_g]
    same_adam = set(s1) == set(s2) and all(torch.equal(s1[k], s2[k]) if torch.is_tensor(s1[k]) else s1[k] == s2[k]
                                           for k in s1)
    out.update(all_merged_rows_equal=same_rows, all_merged_models_equal=same_model, all_merged_adam_equal=same_adam,
               adam_state_shared=shared, all_merged_fired=[r["step"] for r in tg.rows if r["fired"]])
    return dict(ok=bool(ok and same_rows and same_model and same_adam and not shared), **out)


def b3_margin(dev):
    if dev.type != "cpu":
        return dict(ok=True, skipped="cpu only")
    task = sb.task()
    x = task.make_batch(64, torch.Generator().manual_seed(3))[0][:, :-1]
    g = torch.softmax(torch.randn(64, x.shape[1], 4, generator=torch.Generator().manual_seed(4)), -1)
    src = task.stream_labels(x)
    word = x >= task.S
    orig = sb._MARGIN.get("orig", ic.margin_of)
    a = orig(g, src, word, chunk=64)
    b = orig(g, src, word, chunk=sb.MARGIN_CHUNK)
    rel = abs(a - b) / max(abs(a), 1e-30)
    return dict(ok=rel <= 1e-12, margin_64=a, margin_16=b, rel=rel)


def _rec(v, ok=True):
    return dict(ok=ok, end=dict(set=v))


def b4_rules(dev):
    if dev.type != "cpu":
        return dict(ok=True, skipped="cpu only")
    cases = {
        "0.915 / 0.910 -> proceed, thr 0.880": ({840: _rec(0.915), 841: _rec(0.910)}, True, 0.910 - 0.03),
        "mean exactly 0.912 -> proceed": ({840: _rec(0.913), 841: _rec(0.911)}, True, 0.911 - 0.03),
        "a run exactly 0.9, mean 0.915 -> proceed": ({840: _rec(0.9), 841: _rec(0.93)}, True, 0.9 - 0.03),
        "0.930 / 0.899 -> stop (one < 0.9)": ({840: _rec(0.930), 841: _rec(0.899)}, False, None),
        "0.911 / 0.912 -> stop (mean 0.9115)": ({840: _rec(0.911), 841: _rec(0.912)}, False, None),
        "one record missing -> not complete": ({840: _rec(0.95)}, False, None),
        "one record failed -> not complete": ({840: _rec(0.95), 841: _rec(0.95, ok=False)}, False, None),
    }
    out = {}
    for name, (pilot, proceed, thr) in cases.items():
        d = sb.pilot_decision(pilot)
        good = d["proceed"] == proceed and (thr is None or abs(d["split_thr"] - thr) < 1e-15)
        out[name] = dict(ok=good, proceed=d["proceed"], thr=d.get("split_thr"), why=d.get("why"))
    d = sb.pilot_decision({840: _rec(0.9), 841: _rec(0.93)})
    out["a run exactly 0.9 is not 'not usable'"] = dict(ok=not d["not_usable"])
    d = sb.pilot_decision({840: _rec(0.89), 841: _rec(0.95)})
    w = sb.stop_words(d)
    out["0.89 / 0.95 -> stop, both stop phrases"] = dict(
        ok=(not d["proceed"]) and d["not_usable"] and "did not clear the proceed rule" in w and "not usable at B" in w,
        said=w)
    d = sb.pilot_decision({840: _rec(0.905), 841: _rec(0.91)})
    w = sb.stop_words(d)
    out["0.905 / 0.91 -> stop, 'did not clear', not 'not usable'"] = dict(
        ok=(not d["proceed"]) and not d["not_usable"] and "did not clear the proceed rule" in w and "not usable" not in w,
        said=w)
    # register_rule
    mk = lambda ok_, peak: dict(arms={"REG3 scan": dict(ok=ok_, peak_reserved_gb=peak)}, context_gb=0.4, total_gb=5.6)
    out["register_rule: scan fits -> scan"] = dict(ok=sb.register_rule(mk(True, 4.0)) == "scan")
    out["register_rule: scan does not fit -> loop"] = dict(ok=sb.register_rule(mk(True, 5.3)) == "loop")
    out["register_rule: scan failed -> loop"] = dict(ok=sb.register_rule(mk(False, None)) == "loop")
    # held_evals / workers_rule / ram_budget
    out["held_evals: keeping up holds the workers' jobs"] = dict(ok=sb.held_evals(2, 20, 600, 300) == 2)
    out["held_evals: half speed holds half the run"] = dict(ok=sb.held_evals(1, 20, 60, 120) == 11)
    out["held_evals: no workers holds none"] = dict(ok=sb.held_evals(0, 20, 60, 120) == 0)
    w, needed, need, why = sb.workers_rule(10, 2, 0.4, 1.5, 0.27, 600, 30, 20)
    out["workers_rule: fast statistics -> 1"] = dict(ok=(w, needed) == (1, 1), why=why)
    w, needed, need, why = sb.workers_rule(10, 2, 0.4, 1.5, 0.27, 100, 250, 20)
    out["workers_rule: needs 3, fits -> 3"] = dict(ok=(w, needed) == (3, 3) and why == "keeps up", why=why)
    w, needed, need, why = sb.workers_rule(7.2, 2, 0.4, 1.5, 0.27, 100, 250, 20)
    out["workers_rule: needs 3, 3 does not fit -> 2 (behind; 1 holds more evaluations than 2 costs)"] = dict(
        ok=w == 2 and why.startswith("falls behind") and need[1] > need[2], need=need, why=why)
    w, needed, need, why = sb.workers_rule(4.0, 2, 0.4, 1.5, 0.27, 60, 300, 20)
    out["workers_rule: no worker fits -> 0 (in line)"] = dict(ok=w == 0, need=need, why=why)
    w, needed, need, why = sb.workers_rule(3.0, 2, 0.4, 1.5, 0.27, 60, 300, 20)
    out["workers_rule: nothing fits -> refuse"] = dict(ok=w is None and why.startswith("refuse"), why=why)
    bs = dict(hwm_gb=5.0, rss_gb=4.0)
    out["ram_budget: total bound"] = dict(ok=sb.ram_budget(15.2, dict(MemTotal=15.6, MemAvailable=12.0), bs)
                                          == (15.2 - 5.0 - sb.OS_RESERVE_GB, "total"))
    b, how = sb.ram_budget(15.2, dict(MemTotal=15.6, MemAvailable=7.0), bs)
    out["ram_budget: available bound"] = dict(ok=how == "available" and abs(b - (7.0 - 1.0 - sb.AVAIL_RESERVE_GB)) < 1e-12)
    out["ram_budget: MemTotal below the user's figure"] = dict(
        ok=sb.ram_budget(16.0, dict(MemTotal=7.6, MemAvailable=7.5), dict(hwm_gb=0.0, rss_gb=0.0))[0]
        == 7.6 - sb.OS_RESERVE_GB)
    return dict(ok=all(v["ok"] for v in out.values()), cases=out)


@contextlib.contextmanager
def temp_out():
    old = sb.OUT, sb.CHECKS_DIR
    with tempfile.TemporaryDirectory() as d:
        sb.OUT = d
        sb.CHECKS_DIR = d
        try:
            yield d
        finally:
            sb.OUT, sb.CHECKS_DIR = old


@contextlib.contextmanager
def fake_card(name):
    """dv.setup / dv.device_info -> a CUDA device named `name` (the refusals come before anything touches the device)."""
    old = dv.setup, dv.device_info
    dv.setup = lambda device: torch.device("cuda")
    dv.device_info = lambda device: dict(device="cuda:0", gpu=name, cpu="fake cpu", threads=1)
    try:
        yield
    finally:
        dv.setup, dv.device_info = old


def _git(d, *args):
    return subprocess.run(["git", *args], cwd=d, capture_output=True, text=True, check=True).stdout.strip()


def _git_repo(d):
    _git(d, "init", "-q")
    _git(d, "config", "user.email", "checks@example.invalid")
    _git(d, "config", "user.name", "s75b checks")
    _git(d, "config", "commit.gpgsign", "false")


def _commit(d, *paths):
    _git(d, "add", *[os.path.relpath(p, d) for p in paths])
    _git(d, "commit", "-q", "-m", "c")


def b5_refusals(dev):
    if dev.type != "cpu":
        return dict(ok=True, skipped="cpu only")
    out = {}
    sha = sb.code_sha()[0]
    hw = dv.device_info("cpu")["cpu"]
    G1660, G3070 = "FAKE GeForce GTX 1660 SUPER (checks)", "FAKE GeForce RTX 3070 Laptop GPU (checks)"

    def expect_exit(fn, needle):
        try:
            fn()
            return False, "no refusal"
        except SystemExit as e:
            return needle in str(e), str(e)

    def pilot_store(sets, card=hw, sh=sha, seeds=(840, 841)):
        st = {"meta": {}, "runs": {f"ORACLE_PILOT|{s}|{sh}|{card}": dict(_rec(v), arm="ORACLE_PILOT", hw=card)
                                   for s, v in zip(seeds, sets)}}
        sb.save_json(st, sb.store_path(card))
        return st

    def record(**kw):
        d = sb.pilot_decision(sb.runs_of(sb.load_all(), "ORACLE_PILOT", sha))
        rec = dict(sha=sha, decision=d, machine=hw, results_file=os.path.relpath(sb.store_path(hw), HERE), **kw)
        sb.save_json(rec, sb.record_paths()[0])
        return rec

    def s2(machine="fallback", rc=False, card=None):
        if card is None:
            return lambda: sb.stage2("cpu", machine, require_commit=rc)

        def f():
            with fake_card(card):
                sb.stage2("cpu", machine, require_commit=rc)
        return f

    def s1(card, rc=False):
        def f():
            with fake_card(card):
                sb.stage1("cpu", require_commit=rc)
        return f

    # git_committed itself
    with tempfile.TemporaryDirectory() as d:
        _git_repo(d)
        p = os.path.join(d, "x.json")
        with open(p, "w") as f:
            f.write("{}\n")
        a = sb.git_committed(p)
        _commit(d, p)
        b = sb.git_committed(p)
        with open(p, "a") as f:
            f.write(" ")
        c = sb.git_committed(p)
        out["git_committed: untracked / committed / modified"] = (
            a == (None, False) and b[0] is not None and b[1] and c == (b[0], False), f"{a} {b} {c}")
    out["git_committed: outside a repository"] = (sb.git_committed("/nonexistent/x.json") == (None, False), "")
    real_dirty = sb.dirty_code
    out["dirty_code on this checkout (description)"] = (True, str(real_dirty())[:150])
    sb.dirty_code = lambda: []            # the refusals below do not depend on this checkout's state
    try:
        with temp_out() as d:
            _b5_temp(d, out, expect_exit, pilot_store, record, s2, s1, sha, hw, G1660, G3070)
    finally:
        sb.dirty_code = real_dirty
    out["gpu-name rule"] = (all(sb.MACHINE_GPU[m] in g for m, g in (("1660", "NVIDIA GeForce GTX 1660 SUPER"),
                                                                     ("3070", "NVIDIA GeForce RTX 3070 Laptop GPU"),
                                                                     ("fallback", "NVIDIA GeForce GTX 1660 SUPER")))
                            and "RTX 3070" not in "NVIDIA GeForce GTX 1660 SUPER", "")
    alloc = sb.MACHINE_ARMS
    out["allocation"] = (set(alloc["1660"]) == {"ORACLE", "SINGLE", "WIDE_SLOW"} and set(alloc["3070"]) == {"REG3", "WIDE_SPLIT_G"}
                         and set(alloc["fallback"]) == set(sb.STAGE2) and not set(alloc["1660"]) & set(alloc["3070"]), "")
    return dict(ok=all(v[0] for v in out.values()), cases={k: dict(ok=v[0], said=v[1][:200]) for k, v in out.items()})


def _b5_temp(d, out, expect_exit, pilot_store, record, s2, s1, sha, hw, G1660, G3070):
    out["no record"] = expect_exit(s2(), "no stage-1 record")
    pilot_store((0.93, 0.92))
    record()
    sb.save_json(dict(sha=sha, decision=dict(sb.pilot_decision(sb.runs_of(sb.load_all(), "ORACLE_PILOT", sha)),
                                              split_thr=0.5)), sb.record_paths()[0])
    out["record != pilot"] = expect_exit(s2(), "does not match")
    record()
    # the commit checks, in a temporary git repository
    _git_repo(d)
    out["record not committed (untracked)"] = expect_exit(s2(rc=True, card=G1660), "the stage-1 record (")
    _commit(d, sb.record_paths()[0])
    out["pilot results file not committed"] = expect_exit(s2(rc=True, card=G1660), "pilot results file")
    _commit(d, sb.store_path(hw))
    out["both committed -> past the commit checks (to the benchmark)"] = expect_exit(s2(rc=True, card=G1660),
                                                                                     "no benchmark")
    with open(sb.record_paths()[0], "a") as f:
        f.write("\n")
    out["record modified after its commit"] = expect_exit(s2(rc=True, card=G1660), "local changes")
    _commit(d, sb.record_paths()[0])
    good = sb.load(hw)
    pilot_store((0.95, 0.95))
    _commit(d, sb.store_path(hw))
    sb.save_json(good, sb.store_path(hw))
    out["committed pilot records disagree"] = expect_exit(s2(rc=True, card=G1660), "committed pilot records")
    _commit(d, sb.store_path(hw))
    # the pilot
    pilot_store((0.93,), seeds=(840,))
    out["incomplete pilot"] = expect_exit(s2(), "both pilot records")
    os.remove(sb.store_path(hw))
    pilot_store((0.93, 0.92), sh="000000000000")
    out["pilot records under another code SHA"] = expect_exit(s2(), "both pilot records")
    os.remove(sb.store_path(hw))
    pilot_store((0.93, 0.89))
    record()
    out["pilot stopped"] = expect_exit(s2(), "did not clear the proceed rule")
    out["pilot stopped: 'not usable' wording"] = expect_exit(s2(), "not usable at B")
    pilot_store((0.93, 0.92))
    record()
    # two cards
    dup = pilot_store((0.93,), card="another card", seeds=(840,))
    out["duplicate pilot records (stage 2)"] = expect_exit(s2(), "two cards")
    out["duplicate pilot records (stage 1)"] = expect_exit(lambda: sb.stage1("cpu", require_commit=False), "two cards")
    txt = sb.report(log=lambda *a: None, sha=sha, out=os.path.join(d, "summary.md"))
    out["report: DUPLICATE and incomplete"] = ("DUPLICATE RECORDS" in txt and "incomplete: records from two cards" in txt,
                                               "")
    os.remove(sb.store_path("another card"))
    st = {"meta": {}, "runs": {f"REG3|853|{sha}|{G3070}": dict(_rec(0.6), arm="REG3", hw=G3070)}}
    sb.save_json(st, sb.store_path(G3070))
    out["fallback with the 3070's records"] = expect_exit(s2("fallback", card=G1660), "another card")
    out["1660 with the 3070's REG3 records: not its arms -> past (to the benchmark)"] = expect_exit(
        s2("1660", card=G1660), "no benchmark")
    os.remove(sb.store_path(G3070))
    # the card
    out["machine 1660 on a 3070"] = expect_exit(s2("1660", card=G3070), "expects a GTX 1660")
    out["machine 3070 on a 1660"] = expect_exit(s2("3070", card=G1660), "expects a RTX 3070")
    out["no benchmark (stage 2, CUDA)"] = expect_exit(s2("3070", card=G3070), "no benchmark")
    bench = dict(sha=sha, all_ok=True, stats_workers=1, register_reg3="loop", arms={})
    sb.save_json(dict(bench, all_ok=False, arms={"REG3 scan": dict(ok=False)}), sb.bench_path(G3070))
    out["failed benchmark"] = expect_exit(s2("3070", card=G3070), "failed cases")
    sb.save_json(dict(bench, stats_workers=None, workers_why="refuse: test"), sb.bench_path(G3070))
    out["refusing worker rule"] = expect_exit(s2("3070", card=G3070), "host-memory rule refuses")
    sb.save_json(bench, sb.bench_path(G3070))
    out["no CUDA CHECK output"] = expect_exit(s2("3070", card=G3070), "CUDA CHECK output")
    res = dict(meta=dict(s75b_sha=sha), checks={f"B{i}": dict(ok=True) for i in range(1, 9)})
    res["checks"]["B7"] = dict(ok=False)
    sb.save_json(res, sb.checks_output_path(G3070))
    out["failed CUDA CHECK output"] = expect_exit(s2("3070", card=G3070), "failed: ['B7']")
    res["checks"]["B7"] = dict(ok=True)
    sb.save_json(dict(res, meta=dict(s75b_sha="000000000000")), sb.checks_output_path(G3070))
    out["CUDA CHECK output under another SHA"] = expect_exit(s2("3070", card=G3070), "written under code SHA")
    # stage 1 (no pilot records yet: with them, stage 1 on another card refuses first)
    os.remove(sb.store_path(hw))
    out["stage 1: no benchmark (CUDA)"] = expect_exit(s1(G1660), "no benchmark")
    sb.save_json(bench, sb.bench_path(G1660))
    out["stage 1: no CUDA CHECK output"] = expect_exit(s1(G1660), "CUDA CHECK output")
    sb.save_json(res, sb.checks_output_path(G1660))
    out["stage 1: benchmark not committed"] = expect_exit(s1(G1660, rc=True), "benchmark")
    pilot_store((0.93, 0.92))
    out["stage 1: pilot records from another card"] = expect_exit(s1(G3070), "from another card")
    os.remove(sb.store_path(hw))
    sb.dirty_code = lambda: ["explore_gpu/s75b.py"]
    out["stage 1: the driver's code not committed"] = expect_exit(s1(G1660, rc=True), "code is not committed")
    pilot_store((0.93, 0.92))
    _commit(d, sb.store_path(hw), sb.record_paths()[0])
    out["stage 2: the driver's code not committed"] = expect_exit(s2(rc=True, card=G1660), "code is not committed")


def b6_s75_sha(dev):
    if dev.type != "cpu":
        return dict(ok=True, skipped="cpu only")
    from explore_gpu import s75_s4
    sha = s75_s4.code_sha()[0]
    return dict(ok=sha == "5fb120a92d69", s75_sha=sha, s75b_sha=sb.code_sha()[0])


def lockstep_r1(task, spec, seed, iters, dev, tol, register="auto", batch=None, tamper=None):
    """R = 1 lock-step on one device: before each update the batched model takes the sequential model's parameters; the
    batched forward, loss (the runs' ICMC loss) and backward run first and keep only the logits and gradients (the graph
    is freed), then the sequential forward and backward (the model that trains, Adam as the runs: SLOW's two groups for
    a SLOW arm), then explore_gpu.checks.Judge (float64 arbitration on the device where FAST fails). tamper(grads) -> the
    batched gradients as judged (the control). Returns Judge.result() and the device's peak memory."""
    torch.manual_seed(seed)
    m = gr.make_model(spec, task, seed)
    slow = ic.gate_names(m) if spec.get("recipe") == "slow" else None
    bm = BatchedBDH([copy.deepcopy(m)], dev, register=register)
    m = dv.to_device(m, dev, cpu_io=False)
    opt = ck._optimizer(m, slow)
    rng = torch.Generator().manual_seed(seed + 10_000)
    lossf = ck._loss_fn(task, "icmc")
    judge = ck.Judge(tol)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    named = dict(m.named_parameters())
    for step in range(1, iters + 1):
        tok = task.make_batch(batch or gr.BATCH, rng)[0]
        with torch.no_grad():
            for n in bm.names:
                bm.P(n).copy_(named[n].detach().to(bm.device).unsqueeze(0))
        tb = tok.unsqueeze(0).to(dev)
        lb, _ = bm(tb[..., :-1])
        mf = gr.icmc_loss_mask(task, tok).unsqueeze(0).to(dev).to(lb.dtype)
        V = lb.shape[-1]
        ce = F.cross_entropy(lb.reshape(-1, V), tb[..., 1:].reshape(-1), reduction="none").view(mf.shape)
        loss_b = ((ce * mf).sum((1, 2)) / mf.sum((1, 2))).sum()
        for p in bm.ps:
            p.grad = None
        loss_b.backward()
        lt = lb[0].detach().clone()
        gt = {n: (None if bm.P(n).grad is None else bm.P(n).grad[0].detach().clone()) for n in bm.names}
        del lb, ce, loss_b, mf
        for p in bm.ps:
            p.grad = None
        if tamper is not None:
            gt = tamper(gt)
        t = tok.to(dev)
        lo = m(t[:, :-1])[0]
        opt.zero_grad()
        lossf(lo, t).backward()
        ls = lo.detach()
        del lo
        gs = {n: p.grad for n, p in m.named_parameters() if p.grad is not None}
        judge.check(step, 0, lt, ls, {n: gt.get(n) for n in gs}, gs, ck._ref64(m, t, lossf))
        opt.step()
    r = judge.result()
    if dev.type == "cuda":
        r["peak_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
        torch.cuda.empty_cache()
    return r


def b7_lockstep(dev):
    cpu = dev.type == "cpu"
    iters, seeds, batch = (2, (850,), gr.BATCH) if cpu else (20, (850, 851), 8)
    task = sb.task()
    out = {}
    cases = [("ORACLE", "ORACLE", "auto"), ("SINGLE", "SINGLE", "auto"), ("WIDE (WIDE_SPLIT_G / WIDE_SLOW)", "WIDE_SPLIT", "auto"),
             ("REG3 (loop)", "REG3", "loop"), ("REG3 (scan vs the sequential loop)", "REG3", "scan")]
    with sb.runtime():
        for name, key, reg in cases:
            spec = gr.arm_spec(key, task.S)
            for s in seeds:
                r = lockstep_r1(task, spec, s, iters, dev, ck.TOL_SAME, register=reg, batch=batch)
                if cpu:                       # 2 updates: the sign test cannot reject; every update must pass FAST
                    r["ok"] = bool(r["ok"] and r["arbitrated"] == 0)
                out[f"{name} seed {s}"] = r
        # the control: W_g's batched gradient doubled must fail (small configuration, 20 updates)
        st = it.ICMCTask(**SMALL)
        spec = gr.arm_spec("WIDE_SPLIT", st.S)
        ctl = lockstep_r1(st, spec, 850, 20, dev, ck.TOL_SAME, batch=8,
                          tamper=lambda g: {n: (2 * v if n == "W_g" and v is not None else v) for n, v in g.items()})
    out["control: W_g gradient doubled (must fail)"] = dict(ctl, ok=not ctl["ok"], failed_as_it_should=not ctl["ok"])
    return dict(ok=all(v["ok"] for v in out.values()), tol=ck.TOL_SAME, updates=iters, seeds=list(seeds), batch=batch,
                R=1, cases=out)


@contextlib.contextmanager
def small_driver():
    keep = dict(CONFIG=sb.CONFIG, ITERS=sb.ITERS, EVAL_EVERY=sb.EVAL_EVERY, PROCEED_MEAN=sb.PROCEED_MEAN,
                PROCEED_EACH=sb.PROCEED_EACH, PILOT_SEEDS=sb.PILOT_SEEDS, ORACLE_SEEDS=sb.ORACLE_SEEDS, SEEDS=sb.SEEDS,
                ARMS=dict(sb.ARMS))
    sched = (ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST, ic.SPLIT_RISE)
    sb.CONFIG, sb.ITERS, sb.EVAL_EVERY = dict(SMALL), 1200, 400
    sb.PROCEED_MEAN = sb.PROCEED_EACH = 0.0
    sb.PILOT_SEEDS, sb.ORACLE_SEEDS, sb.SEEDS = (840, 841), (850, 851), (850, 851)
    sb.ARMS = {a: (k, sb.PILOT_SEEDS if a == "ORACLE_PILOT" else sb.ORACLE_SEEDS if a == "ORACLE" else sb.SEEDS)
               for a, (k, _) in keep["ARMS"].items()}
    ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST, ic.SPLIT_RISE = 400, 400, 800, 1.0
    try:
        yield
    finally:
        for k, v in keep.items():
            setattr(sb, k, v)
        ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST, ic.SPLIT_RISE = sched


def _guard_in_run():
    """The guarded trigger inside run_batched_icmc (R = 2, split schedule 2 / 2 / 4, threshold 1.01, rise 1.0)."""
    task = it.ICMCTask(**SMALL)
    sched = (ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST)
    ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST = 2, 2, 4
    try:
        with sb.runtime(), sb.guarded(rise=1.0):
            recs = gr.run_batched_icmc(task, gr.arm_spec("WIDE_SPLIT", task.S), [850, 851], 6, device="cpu",
                                       split_thr=1.01, eval_every=2, stats_workers=0, prefetch=False, log=lambda *a: None)
    finally:
        ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST = sched
    out, ok = {}, True
    for r in recs:
        el = [c for c in r.get("checks", []) if c.get("eligible")]
        good = bool(r.get("ok")) and len(el) == 1 and all(
            (c["fired"] and not c["map_one_to_one"] and c.get("state_zeroed") and "map_after" in c
             and c["row_dist_after"] != c["row_dist_before"])
            or (c.get("blocked_by") == "one-to-one" and c["map_one_to_one"]) for c in el)
        ok = ok and good
        out[r["seed"]] = dict(ok=good, error=r.get("error"), eligible=[(c["step"], c["fired"], c.get("blocked_by"),
                                                                         c.get("map_one_to_one"), c.get("state_zeroed"))
                                                                        for c in el])
    return ok, out


def b8_end_to_end(dev):
    if dev.type != "cpu":
        return dict(ok=True, skipped="cpu only")
    out = {}
    logs = []
    with temp_out() as d, small_driver():
        d1, miss1 = sb.stage1("cpu", log=logs.append, require_commit=False)
        rec1 = json.load(open(sb.record_paths()[0]))
        out["stage1"] = dict(proceed=d1["proceed"], split_thr=d1.get("split_thr"), missing=miss1,
                             record=os.path.exists(sb.record_paths()[0]) and os.path.exists(sb.record_paths()[1]),
                             record_machine=rec1.get("machine"), record_changes=bool((rec1.get("changes") or {}).get("items")),
                             record_blind=(rec1.get("changes") or {}).get("blind"))
        miss2 = sb.stage2("cpu", "fallback", log=logs.append, require_commit=False)
        st = sb.load_all()
        sha = sb.code_sha()[0]
        R = {a: sb.runs_of(st, a, sha) for a in sb.ARMS}
        rows = [c for r in R["WIDE_SPLIT_G"].values() for c in r.get("checks", [])]
        meta2 = st["meta"][next(iter(st["meta"]))].get("stage2") or {}
        out["stage2"] = dict(missing=miss2, records={a: sorted(v) for a, v in R.items()},
                             split_checks=[(c["step"], c.get("eligible"), c.get("blocked_by"), c["fired"],
                                            c.get("map_one_to_one")) for c in rows],
                             rows_record_map=all("map_one_to_one" in c and "map_eta" in c for c in rows),
                             meta={k: v for k, v in meta2.items() if k != "changes"}, meta_changes=bool(meta2.get("changes")),
                             reg3_register=sorted({r.get("register") for r in R["REG3"].values()}),
                             host=[r.get("host", {}).get("main_peak_gb") is not None for rr in R.values() for r in rr.values()])
        txt = sb.report(log=logs.append)
        need = ["Changes since the approved spec", "Cards per arm", "VALIDITY", "median margin (layers)", "READING",
                "DIAGNOSTICS", "probe L1", "beta", "train", "PER RUN", "split checks", "paired by seed"]
        out["report"] = dict(lines=len(txt.splitlines()), missing_sections=[n for n in need if n not in txt])
        b = sb.bench("cpu", host_ram_gb=12.0, beside="none", iters=4, every=2, log=logs.append)
        out["bench"] = {k: b.get(k) for k in ("stats_workers", "workers_needed", "workers_why", "budget_gb", "budget_bound",
                                              "main_gb", "worker_gb", "data_gb", "data_gb_source", "payload_gb",
                                              "stats_secs", "secs_per_update", "register_reg3", "all_ok")}
        out["bench"]["arms"] = {k: v.get("ok") for k, v in b["arms"].items()}
        out["bench"]["worker_ru_maxrss_gb"] = b["arms"]["ORACLE"].get("worker_ru_maxrss_gb")
        out["bench"]["split_checks"] = b["arms"]["WIDE_SPLIT_G"].get("checks")
    g_ok, out["guard_in_run"] = _guard_in_run()
    bn = out["bench"]
    ok = (out["stage1"]["record"] and not miss1 and not miss2 and d1["proceed"] and out["stage1"]["record_changes"]
          and out["stage1"]["record_blind"] is False and out["stage1"]["record_machine"]
          and all(len(v) == 2 for v in out["stage2"]["records"].values())
          and len(out["stage2"]["split_checks"]) > 0 and out["stage2"]["rows_record_map"] and out["stage2"]["meta_changes"]
          and all(out["stage2"]["host"]) and not out["report"]["missing_sections"]
          and bn["all_ok"] and bn["stats_workers"] is not None and bn["worker_gb"] and bn["payload_gb"]
          and bn["stats_secs"] is not None and bn["secs_per_update"] and g_ok)
    return dict(ok=bool(ok), **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    sb.one_thread()
    dev = dv.setup(a.device)
    info = dv.device_info(dev)
    tag = "cpu" if dev.type == "cpu" else "cuda_" + sb.hw_slug(info["gpu"])
    out_txt = os.path.join(HERE, "explore_gpu", f"s75b_checks_{tag}_output.txt")
    out_json = os.path.join(HERE, "explore_gpu", f"s75b_checks_{tag}_output.json")
    plan = [("B1", "task CHECK at B", b1_task), ("B2", "guarded trigger mechanics", b2_guard),
            ("B3", "margin in 16-sequence chunks", b3_margin), ("B4", "rules", b4_rules),
            ("B5", "refusals and allocation", b5_refusals), ("B6", "S75's code SHA unchanged", b6_s75_sha),
            ("B7", "lock-step at L = 1536, R = 1", b7_lockstep), ("B8", "end to end (small configuration)", b8_end_to_end)]
    only = None if a.only is None else set(a.only.split(","))
    res = dict(meta=dict(info, started=time.strftime("%Y-%m-%d %H:%M:%S"), s75b_sha=sb.code_sha()[0],
                         only=sorted(only) if only else None), checks={})
    lines = [f"explore_gpu.s75b CHECKs on {info}; S75b code SHA {res['meta']['s75b_sha']}", ""]
    print(lines[0], flush=True)
    for key, name, f in plan:
        if only and key not in only:
            continue
        t0 = time.time()
        try:
            r = f(dev)
        except Exception as e:
            r = dict(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
        r["secs"] = round(time.time() - t0, 1)
        res["checks"][key] = r
        status = "SKIP" if r.get("skipped") else ("ok  " if r["ok"] else "FAIL")
        line = f"{status} {key} {name} ({r['secs']:.0f} s): " + json.dumps(
            {k: v for k, v in r.items() if k not in ("traceback", "secs", "cases")}, default=str)[:900]
        print(line, flush=True)
        lines.append(line)
        if "traceback" in r:
            lines.append(r["traceback"])
            print(r["traceback"], flush=True)
        for cn, c in (r.get("cases") or {}).items():
            cl = f"       {cn}: " + json.dumps({k: v for k, v in c.items() if k != "first_failures"}, default=str)[:300]
            print(cl, flush=True)
            lines.append(cl)
        with open(out_json, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
    failed = [k for k, r in res["checks"].items() if not r["ok"]]
    lines += ["", "ALL CHECKS PASSED" if not failed else f"FAILED: {failed}"]
    print(lines[-1], flush=True)
    with open(out_txt, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
