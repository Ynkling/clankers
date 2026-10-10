#!/usr/bin/env python
"""
explore_gpu/s75_s4.py — EXPLORATORY, not a result. Session I, S75 at S = 4 on a GPU (the batched path of explore_gpu.run).
Everything below is fixed and committed before any S75 run. Readings are S75's (explore_i_dense.py's docstring, the user's
specification of 9-10 October); what S75 left to the pilot and what the pilot could not supply is fixed here, with the
reason.

TASK: ICMC (explore_i_task.ICMCTask) S = 4 sources, V = 4 words, blocks of LMIN..LMAX = 6..14 words, NB = 16 blocks per
  source, M = 176 words per source, L = 768 (the model reads 767). Per source this is S74's C1 exactly (C1: S = 2, same
  V, block lengths, NB and M at L = 384), whose Bayes bound on SET accuracy is 0.932 and where ORACLE reached
  0.910-0.918 on 4/4 seeds at 12000 updates; at S = 4 the same per-source chain needs L = 768 (S74's bound table: "S = 4
  needs L ≈ 768"). Usability at S = 4 was not piloted (S74 excluded S = 4 on CPU cost). VALIDITY below guards only the
  ORACLE half of S74's usability rule (ORACLE >= 0.9); the SINGLE half (SINGLE <= ORACLE - 0.2 on >= 3/4 seeds), which
  C1 failed at S = 2 (2/4), is printed beside the reading, not a condition of it: S75's reading is about ROUTED, which
  does not depend on how close SINGLE comes, and GAP is reported as it falls.
MODEL: the project's BDH (N = 256, D = 32, 3 layers, decay 0.95, conv "layer" width 4), k = S = 4 channels for the gated
  arms. B = 32, Adam 1e-3, evaluation every 1200 updates on 1024 held-out sequences (explore_i_common.evaluate: masked loss,
  SET accuracy overall / by bin / by source, routing per gate layer, the source probe at every layer).
ARMS (seeds 830-839; ORACLE on 830-831; 24000 updates):
  ORACLE      perfect gate, k = 4, ADAM (validity)
  SINGLE      one channel, no gate, ADAM
  WIDE_SPLIT  WIDE window gate (width LMAX + 2 = 16) + SLOW + SPLIT (KEYMASS on all word positions)
  REG3        Session G's register gate at all three layers + SLOW, no nudge, no split
  S75 said: "the two learned arms with the highest ROUTED counts at S = 2 (decided by that count; say which)". S75 at
  S = 2 was never run (S74 found no usable configuration), so that count does not exist. Fixed here instead: WIDE_SPLIT and
  REG3, the two label-free learned arms whose gates can see a block's source token at every word of the block (the WIDE
  window reaches LMAX + 1 tokens back; REG3 reads a register and the residual stream). WIN3_SPLIT's window sees the SRC
  token only at the first two words of a block; at a block's later words its gate is a function of three word tokens,
  which carry no source, so it cannot reach eta^2 >= 0.9 over all word positions. WIDE_SLOW is WIDE_SPLIT without the
  split. --extended adds WIN3_SPLIT, WIDE_SLOW and REG3_NUDGE (10 seeds each), printed, never
  read. REG3_NUDGE's nudge runs on CPU while the model is built (300 Adam steps of a full REG3 forward / backward at
  B = 64, L = 767: some minutes per run before its GPU training).
  SPLIT_THR (the trigger's probe SET accuracy threshold; rise 0.02; checks every 2400 from 4800 to 21600; <= 3 splits,
  >= 4800 apart): 0.88 = C1's lowest ORACLE final SET (0.910) minus 0.03, so a run routed as well as the oracle does not
  fire and a merged run does. At k = S there is no spare channel: a split that fires on a routed run overwrites a live
  channel (batch 18, S51); kept, as S75 specifies k = S.
OUTCOMES (explore_i_dense's definitions, imported): ROUTED := at every gate layer eta^2 by source at word positions >= 0.9 and
  a one-to-one source -> channel map at word positions, at the end. GAP := (SET(arm) - SET(SINGLE, same seed)) /
  (mean SET(ORACLE) - SET(SINGLE, same seed)), and per position bin. VALID := both ORACLE runs end with SET >= 0.9.
READING (fixed): "carries to four sources" if the better of WIDE_SPLIT and REG3 is ROUTED on >= 8/10; otherwise "does not
  carry". UNTESTED if not VALID. Printed beside it (not readings): each arm's ROUTED with Wilson interval and band, median
  GAP and GAP by bin, SET, splits (and on target), eta^2 per layer, margin, probe accuracy per layer, REG3's beta at SRC /
  word; S74's usability count at S = 4 (ORACLE >= 0.9 and SINGLE <= ORACLE - 0.2, per seed, ORACLE's mean).
RUNNING: runs of one arm are stacked R at a time on the GPU (explore_gpu.run.run_batched_icmc; --runs-per-batch, default from
  the memory estimate in explore_gpu/runtime_estimate.md: 1 on a 6 GB card, 2 on an 8 GB card). Records cached in
  explore_out/I/gpu/s75_S4_<GPU model>_results.json (one file per machine; --arms shares the arms between machines; the
  report merges every machine's file), key "arm|seed|code SHA|GPU model"; a failed batch (e.g. out of memory) is retried
  with half as many runs per batch, a single failed run once more. Every record carries the device meta (GPU model, CUDA,
  torch). Pairs (GAP) are by seed across machines: a CUDA run is not bit for bit any other device's, so pairing is of
  outcomes, as between the main line's X and L.

Run:  python -m explore_gpu.s75_s4 --device cuda [--runs-per-batch 2] [--extended]      (from the repository root)
      python -m explore_gpu.s75_s4 --report-only
"""

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import explore_i_task as it
import explore_i_common as ic
import explore_i_dense as idense
from explore_gpu import device as dv
from explore_gpu import run as gr

NAME = "s75_S4"
OUT = os.path.join(HERE, "explore_out", "I", "gpu")
CONFIG = dict(S=4, V=4, NB=16, lmin=6, lmax=14, M=176)
ITERS = 24000
SPLIT_THR = 0.88
SEEDS = tuple(range(830, 840))
ORACLE_SEEDS = (830, 831)
READ_ARMS = ("WIDE_SPLIT", "REG3")
EXTENDED = ("WIN3_SPLIT", "WIDE_SLOW", "REG3_NUDGE")
HI = 8
RPB_6GB, RPB_8GB = 1, 2               # default runs per batch: half the smallest R that fits by explore_gpu/runtime_estimate.md, rounded down (3 -> 1 on 6 GB, 4 -> 2 on 8 GB)


def task():
    return it.ICMCTask(**CONFIG)


def arms(extended=False):
    return ("ORACLE", "SINGLE") + READ_ARMS + (EXTENDED if extended else ())


def seeds_of(arm):
    return ORACLE_SEEDS if arm == "ORACLE" else SEEDS


def hw_slug(hw):
    return "".join(c if c.isalnum() else "_" for c in hw)


def store_path(hw):
    """One results file per GPU model, so the two machines can share the arms and push their files side by side."""
    return os.path.join(OUT, f"{NAME}_{hw_slug(hw)}_results.json")


def load(hw):
    p = store_path(hw)
    if not os.path.exists(p):
        return {"meta": {}, "runs": {}}
    with open(p) as f:
        return json.load(f)


def save(st, hw):
    os.makedirs(OUT, exist_ok=True)
    p = store_path(hw)
    with open(p + ".tmp", "w") as f:
        json.dump(st, f)
    os.replace(p + ".tmp", p)


def load_all():
    """Every machine's file merged: {"meta": {hw: meta}, "runs": {key: rec}} (keys carry the GPU model)."""
    import glob
    out = {"meta": {}, "runs": {}}
    for p in sorted(glob.glob(os.path.join(OUT, f"{NAME}_*_results.json"))):
        with open(p) as f:
            st = json.load(f)
        hw = (st["meta"].get("device") or {}).get("gpu") or os.path.basename(p)
        out["meta"][hw] = st["meta"]
        out["runs"].update(st["runs"])
    return out


def code_sha():
    return gr.code_sha("explore_gpu.s75_s4")


def runs_of(st, arm, sha):
    """{seed: record} of the arm's ok records made with code SHA `sha`, from any machine (one per seed: the first GPU model
    in sorted order if two machines ran the same seed)."""
    out = {}
    for k in sorted(st["runs"]):
        r = st["runs"][k]
        a, s, sh = k.split("|")[:3]
        if a == arm and r.get("ok") and sh == sha and int(s) not in out:
            out[int(s)] = r
    return out


def drive(device, rpb, extended, only=None, log=print, stats_workers=None):
    """Runs every planned (arm, seed) without an ok record under this code SHA on this GPU; returns the list of planned
    runs still without one (main exits non-zero if any)."""
    dev = dv.setup(device)
    info = dv.device_info(dev)
    sha, mods = code_sha()
    hw = info["gpu"] or info["cpu"]
    st = load(hw)
    st["meta"].setdefault("started", time.strftime("%Y-%m-%d %H:%M:%S"))
    st["meta"] = {k: v for k, v in st["meta"].items() if not k.startswith("retried_")}
    T = task()
    t0 = time.time()
    chk = it.check_task(T, n=512)                       # the task's CHECKs at the screen's configuration (explore_i_task)
    assert all(chk.values()), f"task CHECK failed: {chk}"
    st["meta"].update(sha=sha, modules=mods, device=info, config=CONFIG, iters=ITERS, split_thr=SPLIT_THR,
                      task_check=dict(ok=True, n=512, secs=round(time.time() - t0, 1)))
    save(st, hw)
    log(f"[S75 S=4] {T.label}; code SHA {sha}; {info}; task CHECK ok")
    for arm in arms(extended):
        if only and arm not in only:
            continue
        todo = [s for s in seeds_of(arm) if not (st["runs"].get(f"{arm}|{s}|{sha}|{hw}") or {}).get("ok")]
        r = rpb
        while todo:
            batch, todo = todo[:r], todo[r:]
            log(f"  {arm}: seeds {batch} (R = {len(batch)}) ...")
            recs = gr.run_batched_icmc(T, gr.arm_spec(arm, T.S), batch, ITERS, device=dev, split_thr=SPLIT_THR,
                                       tag=arm, stats_workers=stats_workers)
            st = load(hw)
            for rec in recs:
                rec.update(arm=arm, sha=sha, hw=hw)
                st["runs"][f"{arm}|{rec['seed']}|{sha}|{hw}"] = rec
            save(st, hw)
            bad = [rec["seed"] for rec in recs if not rec.get("ok")]
            for rec in recs:
                if rec.get("ok"):
                    e = rec["end"]
                    log(f"    {arm} {rec['seed']}: SET {e['set']:.3f} ROUTED {idense.routed(rec)} "
                        f"train {rec['train_secs'] / 60:.0f} min per run, peak {rec.get('peak_mem_gb_batch')} GB (R = {rec['R']})")
                else:
                    log(f"    {arm} {rec['seed']}: FAILED {rec.get('error')}")
            if bad:
                if len(batch) > 1 and r > 1:
                    r = max(1, r // 2)
                    log(f"    retrying {bad} with {r} run(s) per batch")
                    todo = bad + todo
                elif not st["meta"].get(f"retried_{arm}_{bad}"):
                    st["meta"][f"retried_{arm}_{bad}"] = True
                    save(st, hw)
                    todo = bad + todo
    st = load(hw)
    missing = [(a, s_) for a in arms(extended) if not only or a in only for s_ in seeds_of(a)
               if not (st["runs"].get(f"{a}|{s_}|{sha}|{hw}") or {}).get("ok")]
    report(log, sha, summary=f"{NAME}_{hw_slug(hw)}_summary.md")
    if missing:
        log(f"[S75 S=4] {len(missing)} planned runs have no ok record: {missing}")
    return missing


def report(log=print, sha=None, summary=None):
    st = load_all()
    sha = sha or code_sha()[0]
    T = task()
    R = {a: runs_of(st, a, sha) for a in arms(True)}
    R = {a: v for a, v in R.items() if v or a in arms(False)}
    gpus = sorted({r.get("hw") for rr in R.values() for r in rr.values()})
    L = [f"S75 at S = 4 (EXPLORATORY): {T.label}; {ITERS} updates; split threshold {SPLIT_THR}; code SHA {sha}; "
         f"GPUs {gpus}", ""]
    other = {}
    for k, r in st["runs"].items():
        a, s_, sh = k.split("|")[:3]
        if sh != sha:
            other[(sh, r.get("hw"))] = other.get((sh, r.get("hw")), 0) + 1
    if other:
        L.append("NOT COUNTED (another code SHA): " + "; ".join(f"{n} records with SHA {sh} on {hw}"
                                                              for (sh, hw), n in sorted(other.items())))
    failed = [(k.split("|")[0], k.split("|")[1], r.get("error")) for k, r in st["runs"].items()
              if not r.get("ok") and k.split("|")[2] == sha]
    if failed:
        L.append("FAILED RUNS (last error; a later ok record for the same seed supersedes): " +
                 "; ".join(f"{a} {s_}: {e}" for a, s_, e in failed))
    L.append("")
    orc = list(R["ORACLE"].values())
    valid = len(orc) == 2 and all(r["end"]["set"] >= 0.9 for r in orc)
    oset = {f: (statistics.fmean(r["end"][f] for r in orc) if orc else None)
            for f in ["set"] + [f"set_{b}" for b in it.BIN_NAMES]}
    L.append(f"VALIDITY: ORACLE final SET {[round(r['end']['set'], 3) for r in orc]} -> "
             f"{'VALID' if valid else 'UNTESTED' if len(orc) == 2 else 'incomplete'}")
    sing = R["SINGLE"]
    L += ["", "| arm | ROUTED (Wilson; band) | median GAP | GAP by bin 1 / 2 / 3-4 / 5+ | median SET | splits (on target) | "
              "median eta (layers) | margin |", "|---|---|---|---|---|---|---|---|"]
    gaps = {}
    for a, rr in R.items():
        if not rr:
            L.append(f"| {a} | - | | | | | | |")
            continue
        _, _, cl = idense.count_line(a, R, idense.routed)
        g = {s: idense.gap(r, sing[s], oset) for s, r in rr.items() if s in sing and oset["set"] is not None}
        gaps[a] = g
        gb = [idense.med([idense.gap(r, sing[s], oset, f"set_{b}") for s, r in rr.items() if s in sing])
              for b in it.BIN_NAMES] if oset["set"] is not None else [None] * 4
        sp = [r.get("splits") for r in rr.values() if "splits" in r]
        on = sum(c["on_target"] for r in rr.values() for c in r.get("checks", []) if c["fired"])
        lay = sorted((next(iter(rr.values()))["end"].get("routing") or {}))
        etas = [idense.med([r["end"]["routing"][Ly]["eta"] for r in rr.values()]) for Ly in lay]
        mg = idense.med([statistics.fmean(v["margin"] for v in r["end"]["routing"].values()) for r in rr.values()
                         if r["end"].get("routing")])
        L.append(f"| {a} | {cl} | {idense.fmt(idense.med(g.values()))} | " + " / ".join(idense.fmt(x) for x in gb) +
                 f" | {idense.fmt(idense.med([r['end']['set'] for r in rr.values()]), 3)} | "
                 f"{(str(sum(sp)) + ' (' + str(on) + ')') if sp else '-'} | "
                 f"{' / '.join(idense.fmt(x) for x in etas) or '-'} | {idense.fmt(mg)} |")
    ks = {a: sum(idense.routed(r) for r in R[a].values()) for a in READ_ARMS}
    best = max(ks.values()) if ks else 0
    done = all(len(R[a]) == len(SEEDS) for a in READ_ARMS) and len(orc) == 2
    tag = ("UNTESTED" if not valid else ("carries to four sources" if best >= HI else "does not carry")) if done else "incomplete"
    L += ["", f"READING (pre-fixed): {tag}  (ROUTED {ks}; the better arm needs >= {HI}/10)"]
    if orc and sing:
        om = oset["set"]
        us = [s for s, r in sing.items() if r["end"]["set"] <= om - 0.2]
        L.append(f"Printed: S74's usability count at S = 4: ORACLE mean {om:.3f} (>= 0.9: {om >= 0.9}); SINGLE <= ORACLE - 0.2 "
                 f"on {len(us)}/{len(sing)} seeds")
    L += ["", "DIAGNOSTICS (medians at the end): eta^2 by source at word index 1-2 / 5+ (layer 1), probe L1 / L2 / L3 / final, "
              "REG3 beta^R at SRC / word, train minutes, peak GB"]
    for a, rr in R.items():
        if not rr:
            continue
        rt = [r for r in rr.values() if r["end"].get("routing")]
        e12 = idense.med([r["end"]["routing"]["L1"]["eta_pos12"] for r in rt])
        e5 = idense.med([r["end"]["routing"]["L1"]["eta_pos5p"] for r in rt])
        pr = [idense.med([r["end"]["probe"][k]["acc"] for r in rr.values() if r["end"].get("probe")])
              for k in ("L1", "L2", "L3", "final")]
        bt = (idense.fmt(idense.med([r.get("beta_src") for r in rr.values()])),
              idense.fmt(idense.med([r.get("beta_word") for r in rr.values()])))
        L.append(f"  {a:<11} eta 1-2 {idense.fmt(e12)} / 5+ {idense.fmt(e5)}; probe " + " / ".join(idense.fmt(v) for v in pr) +
                 (f"; beta {bt[0]} / {bt[1]}" if a.startswith("REG3") else "") +
                 f"; train {idense.fmt(idense.med([r.get('train_secs', 0) / 60 for r in rr.values()]), 0)} min per run; "
                 f"peak {idense.med([r.get('peak_mem_gb_batch') for r in rr.values()])} GB per batch of "
                 f"{idense.med([r.get('R') for r in rr.values()])}")
    L += ["", "PER RUN: arm seed ROUTED SET GAP maps / eta / margin per layer, splits"]
    for a, rr in R.items():
        for s, r in sorted(rr.items()):
            rt = r["end"].get("routing") or {}
            g = idense.gap(r, sing[s], oset) if s in sing and oset["set"] is not None else None
            L.append(f"  {a:<11} {s} {'R' if idense.routed(r) else '.'} {r['end']['set']:.3f} {idense.fmt(g)} "
                     + " ".join(f"{Ly}:{v['ch_map']}/{v['eta']:.2f}/{v['margin']:.2f}" for Ly, v in rt.items())
                     + (f" splits {r.get('splits')} at {[c['step'] for c in r.get('checks', []) if c['fired']]}"
                        if "splits" in r else ""))
    txt = "\n".join(L)
    log(txt)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, summary or f"{NAME}_summary.md"), "w") as f:
        f.write(txt + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--runs-per-batch", type=int, default=None)
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--arms", default=None, help="comma-separated subset (to share the arms between two machines)")
    ap.add_argument("--stats-workers", type=int, default=None,
                    help="CPU processes for the evaluation statistics (default 3 on CUDA)")
    a = ap.parse_args()
    if a.report_only:
        report()
        return
    only = None
    if a.arms is not None:
        only = {x.strip() for x in a.arms.split(",") if x.strip()}
        bad = only - set(arms(True))
        assert not bad, f"unknown arms {bad}; arms are {arms(True)}"
    rpb = a.runs_per_batch
    if rpb is None:
        dev = dv.setup(a.device)
        mem = dv.device_info(dev).get("gpu_mem_gb") or 0
        rpb = RPB_8GB if mem >= 7.5 else RPB_6GB
    missing = drive(a.device, rpb, a.extended, only=only, log=lambda s: print(s, flush=True),
                    stats_workers=a.stats_workers)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
