#!/usr/bin/env python
"""
explore_gpu/s75b.py — EXPLORATORY, not a result. S75b: S75 at S = 4 on a configuration where the bar is reachable. The
specification is explore_out/I/s75b_spec_draft.md at 114cd08, approved by the user on 2026-10-12. Its design was fixed
blind to S75's learned arms; this driver implements it, and nothing below changes a constant of it.

CONFIGURATION B: ICMC S = 4, V = 4, blocks 6..14, NB = 32, M = 352, L = 1536 (the model reads 1535); the model, B = 32,
  Adam 1e-3, evaluation every 1200 updates on 1024 held-out sequences, the probe and the routing statistics as S75;
  24000 updates per run.
STAGE 1, the pilot: ORACLE_PILOT (perfect gate, k = 4, ADAM), seeds 840 and 841; its own arm and records, never in VALID,
  GAP or any stage-2 aggregate.
  PROCEED (fixed): both pilot records exist and ok, mean(end.set) >= 0.912 and each end.set >= 0.9 (end.set = the
  final evaluation's overall SET at 24000 updates on the 1024 held-out sequences, unrounded).
  SPLIT_THR := min(end.set of ORACLE_PILOT 840, 841) - 0.03, unrounded.
  STOP otherwise: S75b stops; "S = 4 is not usable at B" only if a pilot run ended below 0.9; no other configuration.
  The stage-1 record (s75b_stage1_record.json / .md, written by --stage 1) names the machine the pilot ran on and must be
  committed before stage 2: --stage 2 refuses unless the record file is committed and clean, both pilot records exist
  under the driver's code SHA and the proceed rule holds; it writes SPLIT_THR and the record's commit into its meta.
STAGE 2, seeds 850-859: ORACLE (850-851, validity), SINGLE, WIDE_SPLIT_G (WIDE window gate, k = 4, SLOW, the split with
  the one-to-one guard), REG3 (as S75), WIDE_SLOW (WIDE_SPLIT_G without the split; printed, never read; runs
  unconditionally). Every arm runs all its seeds whatever ORACLE 850-851 show.
  THE GUARDED TRIGGER (make_split_guarded): explore_i_common.make_split transcribed, installed in this process by
  assigning ic.make_split for WIDE_SPLIT_G's runs (run_batched_icmc looks it up at run time); every helper is called
  through explore_i_common, so the CUDA patches of explore_gpu.run.eval_scope apply. Precedence at a check: (1) not
  eligible: as S75; (2) eligible and capped or gapped: blocked by "cap" / "gap", as S75; (3) eligible, not capped or
  gapped, and the source -> channel map at word positions one-to-one (routing_layer's own one_to_one): blocked by
  "one-to-one", the fire count and the last-fire step unchanged; (4) otherwise the split fires. prev is updated at
  every check. Every row also records the map's eta, margin and one_to_one.
OUTCOMES, VALID, READING (S75's): ROUTED := at every gate layer eta^2 by source at word positions >= 0.9 and a one-to-one
  map, at the end. GAP := (SET(arm) - SET(SINGLE, same seed)) / (mean SET(ORACLE 850-851) - SET(SINGLE, same seed)),
  overall and by bin. VALID := both ORACLE 850-851 end with SET >= 0.9. READING: "carries to four sources" if the better
  of WIDE_SPLIT_G and REG3 is ROUTED on >= 8/10, "does not carry" otherwise; UNTESTED if not VALID (then ROUTED counts,
  GAP, eta^2, margin and the probe are descriptions only, never re-read under a later bar, and the pilot is never pooled
  with 850-851).
MACHINES (fixed): the GTX 1660 SUPER runs ORACLE 850-851, SINGLE and WIDE_SLOW; the RTX 3070 Laptop runs REG3 and
  WIDE_SPLIT_G; if the 3070 is unavailable when stage 2 starts, every arm runs on the 1660 (--machine fallback) and
  nothing is dropped. R = 1 for every arm on both cards. REG3's register mode by rule from this card's benchmark: "scan" if
  the benchmark's peak reserved memory with the scan, plus the context (memory in use beyond this process's reserve),
  is below the card's total; otherwise "loop". --stats-workers by rule from the benchmark: max(1, min(3, floor((host RAM
  for S75b - the main process's peak RSS - 0.5 GB for the data process) / one worker's peak RSS))).

CHANGES SINCE 114cd08 (made after S75's learned-arm records were read; none is a constant of the design):
  - The pilot may run on the RTX 3070 if it is free before the GTX 1660 (the user's message of 12 October, round 3); the
    stage-1 record names the machine.
  - Host memory: in S75b's processes the routing margin (explore_i_common.margin_of) runs in chunks of MARGIN_CHUNK = 16
    sequences instead of 64 (the same value up to float64 summation order: CHECK B3), and the evaluation statistics'
    worker pool is this module's StatsPool (explore_gpu.run.AsyncStats with that patch applied in each worker).
  - The host RAM S75b may use on a machine is the user's input at the benchmark (--host-ram-gb): the laptop has 15.2 GB
    and runs test_delta_recipe beside it; the server's figure comes from the user.

RUN (from the repository root; every step writes into explore_out/I/gpu/s75b/):
  python -m explore_gpu.s75b_checks --device cpu            CHECKs (this container); --device cuda on each card
  python -m explore_gpu.s75b --bench --device cuda --host-ram-gb G      each card, before stage 1 (commit the bench file)
  python -m explore_gpu.s75b --stage 1 --device cuda        the pilot (either card); then commit and push the record
  python -m explore_gpu.s75b --stage 2 --device cuda --machine 1660|3070|fallback
  python -m explore_gpu.s75b --report                       the counts table -> s75b_summary.md
"""

import argparse
import contextlib
import json
import math
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch

import explore_i_task as it
import explore_i_common as ic
import explore_i_dense as idense
from explore_gpu import device as dv
from explore_gpu import run as gr

NAME = "s75b"
OUT = os.path.join(HERE, "explore_out", "I", "gpu", "s75b")
CONFIG = dict(S=4, V=4, NB=32, lmin=6, lmax=14, M=352)
ITERS = 24000
EVAL_EVERY = 1200                          # S75's (explore_gpu.run.EVAL_EVERY); the split trigger runs after evaluations
PILOT_SEEDS = (840, 841)
ORACLE_SEEDS = (850, 851)
SEEDS = tuple(range(850, 860))
BAR = 0.9                                  # VALID (S74's registered bar)
PROCEED_MEAN, PROCEED_EACH = 0.912, 0.9    # 0.9 + S75's ORACLE seed spread (0.892 - 0.880); each run >= 0.9
THR_MARGIN = 0.03                          # SPLIT_THR := min(pilot) - 0.03 (S75's rule)
READ_ARMS = ("WIDE_SPLIT_G", "REG3")
HI = 8
ARMS = {                                   # arm -> (explore_gpu.run.S75_ARMS key, seeds)
    "ORACLE_PILOT": ("ORACLE", PILOT_SEEDS),
    "ORACLE": ("ORACLE", ORACLE_SEEDS),
    "SINGLE": ("SINGLE", SEEDS),
    "WIDE_SPLIT_G": ("WIDE_SPLIT", SEEDS),
    "REG3": ("REG3", SEEDS),
    "WIDE_SLOW": ("WIDE_SLOW", SEEDS),
}
STAGE2 = ("ORACLE", "SINGLE", "WIDE_SPLIT_G", "REG3", "WIDE_SLOW")
MACHINE_ARMS = {"1660": ("ORACLE", "SINGLE", "WIDE_SLOW"), "3070": ("REG3", "WIDE_SPLIT_G"), "fallback": STAGE2}
MACHINE_GPU = {"1660": "GTX 1660", "3070": "RTX 3070", "fallback": "GTX 1660"}
MARGIN_CHUNK = 16
DATA_PROC_GB = 0.5
MAX_WORKERS = 3


def task():
    return it.ICMCTask(**CONFIG)


def arm_spec(arm):
    return gr.arm_spec(ARMS[arm][0], CONFIG["S"])


def seeds_of(arm):
    return ARMS[arm][1]


def code_sha():
    return gr.code_sha("explore_gpu.s75b")


def hw_slug(hw):
    return "".join(c if c.isalnum() else "_" for c in hw)


# ── host memory and the routing margin ───────────────────────────────────────
_MARGIN = {}


def small_margin():
    """explore_i_common.margin_of in chunks of MARGIN_CHUNK sequences (routing_layer looks it up at call time)."""
    if "orig" not in _MARGIN:
        _MARGIN["orig"] = ic.margin_of

        def margin_of(g, src, word, chunk=MARGIN_CHUNK):
            return _MARGIN["orig"](g, src, word, chunk=min(chunk, MARGIN_CHUNK))
        ic.margin_of = margin_of


def rss_gb():
    """This process's peak resident memory in GB (None where it cannot be read)."""
    try:
        import resource
        k = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return k / 2 ** 30 if sys.platform == "darwin" else k / 2 ** 20
    except Exception:
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 2 ** 30
        except Exception:
            return None


_WRSS = []


def _stats_init_b(task_, ev, pdata):
    small_margin()
    gr._stats_init(task_, ev, pdata)


def _stats_job_b(lg, gates, feats):
    out = gr._stats_job(lg, gates, feats)
    return dict(out, _worker_rss_gb=rss_gb())


class StatsPool(gr.AsyncStats):
    """explore_gpu.run.AsyncStats with the small-margin patch in each worker; records the workers' peak RSS."""

    def __init__(self, task_, ev, pdata, workers):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                        initializer=_stats_init_b, initargs=(task_, ev, pdata))
        self.jobs = []

    def submit(self, key, extra, lg, gates, feats):
        self.jobs.append((key, extra, self.pool.submit(_stats_job_b, lg, gates, feats)))

    def collect(self):
        out = []
        for key, extra, f in self.jobs:
            r = dict(f.result())
            w = r.pop("_worker_rss_gb", None)
            if w is not None:
                _WRSS.append(w)
            out.append((key, dict(r, **extra)))
        self.pool.shutdown()
        return out


@contextlib.contextmanager
def runtime():
    """S75b's process-level patches: the small routing margin, the statistics pool. Restored on exit."""
    small_margin()
    old = gr.AsyncStats
    gr.AsyncStats = StatsPool
    try:
        yield
    finally:
        gr.AsyncStats = old


# ── the guarded split trigger ─────────────────────────────────────────────────
def make_split_guarded(task_, seed, last, thr, rise=None, max_splits=None, gap=None):
    """explore_i_common.make_split, transcribed, with the one-to-one guard (precedence in the module docstring)."""
    rise = ic.SPLIT_RISE if rise is None else rise
    max_splits = ic.SPLIT_MAX if max_splits is None else max_splits
    gap = ic.SPLIT_GAP if gap is None else gap
    pe = ic.eval_set(task_, ic.SPLIT_PROBE_N, ic.SPLIT_PROBE_SEED)
    px = pe["tokens"][:, :-1]
    st = dict(prev=None, n=0, last=None)
    rows = []

    def trig(m, step, opt):
        if step % ic.SPLIT_EVERY or step < ic.SPLIT_REF or step > last:
            return
        lg = ic.logits_chunked(m, px)
        acc = float(it.set_hits(task_, lg, pe["sets"])[pe["ann"]["mask"]].float().mean())
        g = ic.model_gates(m, px)[0]
        word = px >= task_.S
        km = g[word].mean(0)
        cs, c0 = int(km.argmax()), int(km.argmin())
        rl = ic.routing_layer(g, px, task_)
        cmap = rl["ch_map"]
        row = dict(step=step, acc=acc, prev=st["prev"], map_before=cmap, masses=[round(float(v), 5) for v in km],
                   cs=cs, c0=c0, on_target=cmap.count(cs) >= 2 and cmap.count(c0) == 0, fired=False, blocked=False,
                   map_eta=rl["eta"], map_margin=rl["margin"], map_one_to_one=bool(rl["one_to_one"]))
        eligible = step >= ic.SPLIT_FIRST and st["prev"] is not None and acc < thr and acc - st["prev"] < rise
        capped = st["n"] >= max_splits or (st["last"] is not None and step - st["last"] < gap)
        row["eligible"] = eligible
        if eligible and capped:
            row.update(blocked=True, blocked_by="cap" if st["n"] >= max_splits else "gap")
        elif eligible and rl["one_to_one"]:
            row.update(blocked=True, blocked_by="one-to-one")
        elif eligible:
            res = ic.split_op(m, opt, cs, c0, ic.NOISE_BASE + 1000 * seed + st["n"])
            st["n"] += 1
            st["last"] = step
            g2 = ic.model_gates(m, px)[0]
            row.update(fired=True, map_after=ic.routing_layer(g2, px, task_)["ch_map"], **res)
        st["prev"] = acc
        rows.append(row)
    trig.rows = rows
    return trig


@contextlib.contextmanager
def guarded(rise=None):
    """ic.make_split -> make_split_guarded in this process (run_batched_icmc looks ic.make_split up at run time)."""
    old = ic.make_split
    ic.make_split = lambda t, s, last, thr: make_split_guarded(t, s, last, thr, rise=rise)
    try:
        yield
    finally:
        ic.make_split = old


# ── stores ─────────────────────────────────────────────────────────────────────
def store_path(hw):
    return os.path.join(OUT, f"{NAME}_{hw_slug(hw)}_results.json")


def load(hw):
    p = store_path(hw)
    if not os.path.exists(p):
        return {"meta": {}, "runs": {}}
    with open(p) as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)


def load_all():
    import glob
    out = {"meta": {}, "runs": {}}
    for p in sorted(glob.glob(os.path.join(OUT, f"{NAME}_*_results.json"))):
        with open(p) as f:
            st = json.load(f)
        out["meta"][os.path.basename(p)] = st["meta"]
        out["runs"].update(st["runs"])
    return out


def runs_of(st, arm, sha):
    out = {}
    for k in sorted(st["runs"]):
        r = st["runs"][k]
        a, s, sh = k.split("|")[:3]
        if a == arm and r.get("ok") and sh == sha and int(s) not in out:
            out[int(s)] = r
    return out


# ── the stage-1 rules ─────────────────────────────────────────────────────────
def pilot_decision(pilot):
    """pilot: {seed: record}. Returns the stage-1 decision (the module docstring's rules), all values unrounded."""
    have = {s: pilot.get(s) for s in PILOT_SEEDS}
    if any(r is None or not r.get("ok") for r in have.values()):
        return dict(complete=False, proceed=False, why="a pilot record is missing or failed",
                    sets={s: (r or {}).get("end", {}).get("set") for s, r in have.items()})
    sets = {s: float(r["end"]["set"]) for s, r in have.items()}
    mean = statistics.fmean(sets.values())
    each = all(v >= PROCEED_EACH for v in sets.values())
    proceed = mean >= PROCEED_MEAN and each
    thr = min(sets.values()) - THR_MARGIN
    why = (f"mean {mean:.6f} {'>=' if mean >= PROCEED_MEAN else '<'} {PROCEED_MEAN}; each >= {PROCEED_EACH}: {each}")
    return dict(complete=True, proceed=proceed, sets=sets, mean=mean, each_ok=each, split_thr=thr if proceed else None,
                not_usable=any(v < BAR for v in sets.values()), why=why)


def record_paths():
    return os.path.join(OUT, "s75b_stage1_record.json"), os.path.join(OUT, "s75b_stage1_record.md")


def git_committed(path):
    """(commit, clean): the last commit touching path, and whether the working tree's file equals it."""
    rel = os.path.relpath(path, HERE)
    try:
        c = subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=HERE, capture_output=True, text=True,
                           check=True).stdout.strip()
        s = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=HERE, capture_output=True, text=True,
                           check=True).stdout.strip()
        return c or None, not s
    except Exception:
        return None, False


# ── the benchmark and the run-mechanics rules ─────────────────────────────────
def bench_path(hw):
    return os.path.join(OUT, f"{NAME}_bench_{hw_slug(hw)}.json")


def workers_rule(host_ram_gb, main_gb, worker_gb):
    if host_ram_gb is None or main_gb is None or not worker_gb:
        return None
    return max(1, min(MAX_WORKERS, int((host_ram_gb - main_gb - DATA_PROC_GB) // worker_gb)))


def register_rule(bench):
    s = bench["arms"].get("REG3 scan") or {}
    if not s.get("ok") or s.get("peak_reserved_gb") is None:
        return "loop"
    return "scan" if s["peak_reserved_gb"] + bench["context_gb"] < bench["total_gb"] else "loop"


def bench(device, host_ram_gb, iters=9, every=3, log=print):
    """Each S75b arm at R = 1 on this card for `iters` updates with an evaluation every `every` (device part here, the
    statistics in one StatsPool worker) and, for WIDE_SPLIT_G, one guarded split check (the trigger's schedule scaled
    to `every`, threshold 1.01, rise 1.0); REG3 in both register modes. Reads peak device memory and host RSS."""
    dev = dv.setup(device)
    T = task()
    out = dict(device=dv.device_info(dev), host_ram_gb=host_ram_gb, iters=iters, eval_every=every, arms={})
    sched = (ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST)
    cases = [("ORACLE", "ORACLE", "auto"), ("SINGLE", "SINGLE", "auto"), ("WIDE_SLOW", "WIDE_SLOW", "auto"),
             ("WIDE_SPLIT_G", "WIDE_SPLIT_G", "auto"), ("REG3 scan", "REG3", "scan"), ("REG3 loop", "REG3", "loop")]
    with runtime():
        for name, arm, reg in cases:
            _WRSS.clear()
            ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST = every, every, 2 * every
            try:
                with guarded(rise=1.0) if arm == "WIDE_SPLIT_G" else contextlib.nullcontext():
                    t0 = time.time()
                    recs = gr.run_batched_icmc(T, arm_spec(arm), [PILOT_SEEDS[0]], iters, device=dev, split_thr=1.01,
                                               eval_every=every, register=reg, stats_workers=1, prefetch=True)
                r = recs[0]
                row = dict(ok=bool(r.get("ok")), error=r.get("error"), secs=time.time() - t0, register=r.get("register"),
                           secs_per_update=(r.get("train_secs") or 0) / iters, worker_rss_gb=max(_WRSS) if _WRSS else None,
                           checks=[(c["step"], c.get("blocked_by"), c["fired"]) for c in r.get("checks", [])])
                if dev.type == "cuda":
                    free, total = torch.cuda.mem_get_info(dev)
                    row.update(peak_alloc_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                               peak_reserved_gb=torch.cuda.max_memory_reserved(dev) / 2 ** 30,
                               device_used_gb=(total - free) / 2 ** 30, reserved_now_gb=torch.cuda.memory_reserved(dev) / 2 ** 30)
            except Exception as e:
                row = dict(ok=False, error=f"{type(e).__name__}: {e}")
            finally:
                ic.SPLIT_EVERY, ic.SPLIT_REF, ic.SPLIT_FIRST = sched
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            out["arms"][name] = row
            log(f"  bench {name}: {json.dumps(row, default=str)[:300]}")
    out["main_rss_gb"] = rss_gb()
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info(dev)
        out["total_gb"] = total / 2 ** 30
        ctx = [r["device_used_gb"] - r["reserved_now_gb"] for r in out["arms"].values() if "device_used_gb" in r]
        out["context_gb"] = max(ctx) if ctx else 0.0
        out["register_reg3"] = register_rule(out)
    wr = [r["worker_rss_gb"] for r in out["arms"].values() if r.get("worker_rss_gb")]
    out["worker_rss_gb"] = max(wr) if wr else None
    out["stats_workers"] = workers_rule(host_ram_gb, out["main_rss_gb"], out["worker_rss_gb"])
    out["sha"] = code_sha()[0]
    return out


def load_bench(hw, sha):
    p = bench_path(hw)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        b = json.load(f)
    return b if b.get("sha") == sha else None


# ── running arms ──────────────────────────────────────────────────────────────
def run_arm(arm, seeds, dev, hw, sha, st, meta_extra, register, workers, split_thr=None, log=print):
    """Each seed at R = 1; a failed run is retried once. Records keyed arm|seed|sha|hw. Returns the seeds still missing."""
    T = task()
    missing = []
    for s in seeds:
        key = f"{arm}|{s}|{sha}|{hw}"
        if (st["runs"].get(key) or {}).get("ok"):
            continue
        for attempt in (1, 2):
            log(f"  {arm} seed {s} (attempt {attempt}) ...")
            ctx = guarded() if arm == "WIDE_SPLIT_G" else contextlib.nullcontext()
            with ctx:
                rec = gr.run_batched_icmc(T, arm_spec(arm), [s], ITERS, device=dev, split_thr=split_thr, eval_every=EVAL_EVERY,
                                          register=register if arm == "REG3" else "auto", stats_workers=workers, tag=arm)[0]
            rec.update(arm=arm, sha=sha, hw=hw, attempt=attempt, **meta_extra)
            st["runs"][key] = rec
            save_json(st, store_path(hw))
            if rec.get("ok"):
                e = rec["end"]
                log(f"    {arm} {s}: SET {e['set']:.4f} ROUTED {idense.routed(rec)}"
                    + (f" splits {rec.get('splits')}" if "splits" in rec else ""))
                break
            log(f"    {arm} {s}: FAILED {rec.get('error')}")
        if not (st["runs"].get(key) or {}).get("ok"):
            missing.append(s)
    return missing


def stage1(device, log=print):
    dev = dv.setup(device)
    info = dv.device_info(dev)
    hw = info["gpu"] or info["cpu"]
    sha = code_sha()[0]
    b = load_bench(hw, sha)
    if dev.type == "cuda" and b is None:
        raise SystemExit(f"no benchmark for {hw} under code SHA {sha}: run --bench on this card first")
    workers = (b or {}).get("stats_workers") or (1 if dev.type == "cuda" else 0)
    T = task()
    chk = it.check_task(T)
    assert all(chk.values()), f"task CHECK failed: {chk}"
    st = load(hw)
    st["meta"].update(sha=sha, config=CONFIG, iters=ITERS, device=info, stage1=dict(workers=workers, started=time.strftime("%Y-%m-%d %H:%M:%S")))
    save_json(st, store_path(hw))
    log(f"[S75b stage 1] {T.label}; code SHA {sha}; {info}; task CHECK ok; stats workers {workers}")
    with runtime():
        missing = run_arm("ORACLE_PILOT", PILOT_SEEDS, dev, hw, sha, st, dict(stage=1), "auto", workers, log=log)
    pilot = runs_of(load_all(), "ORACLE_PILOT", sha)
    d = pilot_decision(pilot)
    rec = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), sha=sha, machine=hw, device=info, config=CONFIG, iters=ITERS,
               pilot_seeds=list(PILOT_SEEDS), decision=d, missing=missing,
               note=("commit and push this record before stage 2; --stage 2 refuses otherwise. The pilot ran on " + hw +
                     " (the spec at 114cd08 assigned the 1660; the user allowed the 3070 if free first)."))
    jp, mp_ = record_paths()
    save_json(rec, jp)
    sets = d.get("sets") or {}
    lines = [f"# S75b stage-1 record (EXPLORATORY), {rec['written']}", "",
             f"- Pilot: ORACLE_PILOT, seeds {PILOT_SEEDS}, 24,000 updates, on **{hw}** (code SHA {sha}).",
             f"- end.set: " + ", ".join(f"{s}: {v}" for s, v in sets.items()),
             f"- Proceed rule (mean >= {PROCEED_MEAN} and each >= {PROCEED_EACH}): {d.get('why')} -> "
             f"**{'PROCEED' if d['proceed'] else ('STOP' if d['complete'] else 'INCOMPLETE')}**"]
    if d["proceed"]:
        lines.append(f"- SPLIT_THR = min(end.set) - {THR_MARGIN} = {d['split_thr']!r}")
    elif d["complete"]:
        lines.append("- S75b stops. " + ("A pilot run ended below 0.9: S = 4 is not usable at B." if d["not_usable"] else
                                          "Both pilot runs reached 0.9; B's pilot did not clear the proceed rule."))
    lines += ["", rec["note"]]
    with open(mp_, "w") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines))
    return d, missing


def stage2(device, machine, log=print, require_commit=True):
    dev = dv.setup(device)
    info = dv.device_info(dev)
    hw = info["gpu"] or info["cpu"]
    sha = code_sha()[0]
    if dev.type == "cuda" and MACHINE_GPU[machine] not in hw:
        raise SystemExit(f"--machine {machine} expects a {MACHINE_GPU[machine]}, this card is {hw}")
    jp, _ = record_paths()
    if not os.path.exists(jp):
        raise SystemExit("no stage-1 record: run --stage 1 first")
    commit, clean = git_committed(jp)
    if require_commit and (commit is None or not clean):
        raise SystemExit(f"the stage-1 record {jp} is not committed (or has local changes): commit and push it first")
    pilot = runs_of(load_all(), "ORACLE_PILOT", sha)
    d = pilot_decision(pilot)
    with open(jp) as f:
        rec1 = json.load(f)
    if not d["complete"]:
        raise SystemExit("both pilot records under this code SHA are needed (" + d["why"] + ")")
    if rec1.get("sha") != sha or rec1["decision"].get("split_thr") != d["split_thr"]:
        raise SystemExit("the stage-1 record does not match the pilot records under this code SHA")
    if not d["proceed"]:
        raise SystemExit("S75b stops at stage 1 (" + d["why"] + ")")
    b = load_bench(hw, sha)
    if dev.type == "cuda" and b is None:
        raise SystemExit(f"no benchmark for {hw} under code SHA {sha}: run --bench on this card first")
    register = (b or {}).get("register_reg3") or ("loop" if dev.type == "cpu" else "loop")
    workers = (b or {}).get("stats_workers") or (1 if dev.type == "cuda" else 0)
    st = load(hw)
    meta2 = dict(machine=machine, split_thr=d["split_thr"], stage1_commit=commit, register_reg3=register, workers=workers,
                 started=time.strftime("%Y-%m-%d %H:%M:%S"))
    st["meta"].update(sha=sha, config=CONFIG, iters=ITERS, device=info, stage2=meta2)
    save_json(st, store_path(hw))
    log(f"[S75b stage 2] machine {machine} ({hw}); arms {MACHINE_ARMS[machine]}; SPLIT_THR {d['split_thr']!r}; "
        f"stage-1 commit {commit}; REG3 register {register}; stats workers {workers}")
    missing = {}
    with runtime():
        for arm in MACHINE_ARMS[machine]:
            m = run_arm(arm, seeds_of(arm), dev, hw, sha, st, dict(stage=2, machine=machine), register, workers,
                        split_thr=d["split_thr"] if arm == "WIDE_SPLIT_G" else None, log=log)
            if m:
                missing[arm] = m
    if missing:
        log(f"[S75b stage 2] runs without an ok record: {missing}")
    return missing


# ── report ────────────────────────────────────────────────────────────────────
def report(log=print, sha=None, out=None):
    st = load_all()
    sha = sha or code_sha()[0]
    R = {a: runs_of(st, a, sha) for a in ARMS}
    L = [f"# S75b (EXPLORATORY): {task().label}; {ITERS} updates; code SHA {sha}", ""]
    pil = R["ORACLE_PILOT"]
    d = pilot_decision(pil)
    L.append(f"Stage 1 (pilot, never pooled): end.set {[(s, round(r['end']['set'], 4)) for s, r in sorted(pil.items())]}; "
             f"{d['why']}; {'PROCEED, SPLIT_THR ' + format(d['split_thr'], '.6f') if d['proceed'] else 'not proceeding'}")
    orc = [R["ORACLE"].get(s) for s in ORACLE_SEEDS]
    complete_orc = all(r is not None for r in orc)
    valid = complete_orc and all(r["end"]["set"] >= BAR for r in orc)
    oset = {f: statistics.fmean(r["end"][f] for r in orc) for f in ["set"] + [f"set_{b}" for b in it.BIN_NAMES]} \
        if complete_orc else None
    L.append(f"VALIDITY: ORACLE 850-851 final SET {[round(r['end']['set'], 4) for r in orc if r]} -> "
             f"{'VALID' if valid else ('UNTESTED' if complete_orc else 'incomplete')}")
    sing = R["SINGLE"]
    L += ["", "| arm | ROUTED (Wilson; band) | median GAP | GAP by bin 1 / 2 / 3-4 / 5+ | median SET | splits fired / blocked "
              "one-to-one | median eta (layers) | margin |", "|---|---|---|---|---|---|---|---|"]
    for a in STAGE2:
        rr = R[a]
        if not rr:
            L.append(f"| {a} | - | | | | | | |")
            continue
        _, _, cl = idense.count_line(a, R, idense.routed)
        g = [idense.gap(r, sing[s], oset) for s, r in rr.items() if s in sing and oset] if oset else []
        gb = [idense.med([idense.gap(r, sing[s], oset, f"set_{b}") for s, r in rr.items() if s in sing]) if oset else None
              for b in it.BIN_NAMES]
        fired = sum(c["fired"] for r in rr.values() for c in r.get("checks", []))
        oto = sum(c.get("blocked_by") == "one-to-one" for r in rr.values() for c in r.get("checks", []))
        lay = sorted(next(iter(rr.values()))["end"].get("routing") or {})
        etas = [idense.med([r["end"]["routing"][y]["eta"] for r in rr.values()]) for y in lay]
        mg = idense.med([statistics.fmean(v["margin"] for v in r["end"]["routing"].values()) for r in rr.values()
                         if r["end"].get("routing")])
        L.append(f"| {a} | {cl} | {idense.fmt(idense.med(g))} | " + " / ".join(idense.fmt(x) for x in gb) +
                 f" | {idense.fmt(idense.med([r['end']['set'] for r in rr.values()]), 3)} | "
                 f"{(str(fired) + ' / ' + str(oto)) if a == 'WIDE_SPLIT_G' else '-'} | "
                 f"{' / '.join(idense.fmt(x) for x in etas) or '-'} | {idense.fmt(mg)} |")
    ks = {a: sum(idense.routed(r) for r in R[a].values()) for a in READ_ARMS}
    done = complete_orc and all(len(R[a]) == len(SEEDS) for a in READ_ARMS)
    if not done:
        tag = "incomplete"
    elif not valid:
        tag = "UNTESTED (descriptions only; never re-read under a later bar)"
    else:
        tag = "carries to four sources" if max(ks.values()) >= HI else "does not carry"
    L += ["", f"READING (pre-fixed): {tag}  (ROUTED {ks}; the better arm needs >= {HI}/10)"]
    if oset and sing:
        us = [s for s, r in sing.items() if r["end"]["set"] <= oset["set"] - 0.2]
        L.append(f"Printed: S74's usability count: ORACLE mean {oset['set']:.4f}; SINGLE <= ORACLE - 0.2 on {len(us)}/{len(sing)}")
    L += ["", "WIDE_SPLIT_G's split checks (every eligible check): seed step probe-SET map-before one-to-one outcome"]
    for s, r in sorted(R["WIDE_SPLIT_G"].items()):
        for c in r.get("checks", []):
            if c.get("eligible"):
                L.append(f"  {s} {c['step']} {c['acc']:.3f} {c['map_before']} {c.get('map_one_to_one')} "
                         f"{'fired' if c['fired'] else 'blocked: ' + str(c.get('blocked_by'))}")
    L += ["", "WIDE_SPLIT_G beside WIDE_SLOW, paired by seed: seed ROUTED(split_g) ROUTED(slow) SET(split_g) SET(slow)"]
    for s in SEEDS:
        a, b_ = R["WIDE_SPLIT_G"].get(s), R["WIDE_SLOW"].get(s)
        if not (a or b_):
            continue
        cells = [("R" if idense.routed(r) else ".") if r else "-" for r in (a, b_)]
        cells += [f"{r['end']['set']:.4f}" if r else "-" for r in (a, b_)]
        L.append(f"  {s} " + " ".join(cells))
    txt = "\n".join(L) + "\n"
    with open(out or os.path.join(OUT, "s75b_summary.md"), "w") as f:
        f.write(txt)
    log(txt)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stage", type=int, choices=(1, 2))
    ap.add_argument("--machine", choices=tuple(MACHINE_ARMS))
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--host-ram-gb", type=float, default=None,
                    help="host RAM S75b may use on this machine (the user's figure; recorded)")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.report:
        report()
        return
    if a.bench:
        if a.host_ram_gb is None:
            raise SystemExit("--bench needs --host-ram-gb (the RAM S75b may use on this machine)")
        b = bench(a.device, a.host_ram_gb)
        hw = b["device"]["gpu"] or b["device"]["cpu"]
        save_json(b, bench_path(hw))
        print(json.dumps({k: v for k, v in b.items() if k != "arms"}, indent=1, default=str))
        return
    if a.stage == 1:
        d, missing = stage1(a.device)
        sys.exit(1 if missing else 0)
    if a.stage == 2:
        if a.machine is None:
            raise SystemExit("--stage 2 needs --machine 1660, 3070 or fallback")
        missing = stage2(a.device, a.machine)
        sys.exit(1 if missing else 0)
    ap.print_help()


if __name__ == "__main__":
    main()
