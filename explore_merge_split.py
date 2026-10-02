#!/usr/bin/env python
"""
explore_merge_split.py — EXPLORATORY, not a result. Screen S24 (batch 8): does breaking the symmetry
of the read gate's rows un-merge two streams that share a channel at S=4, k=4?

BACKGROUND (post hoc, from batch 7)
- S21: in the 8 merged runs the merged pair's mean read gates at value positions are identical (e.g.
  0.94/0.06 on the same two channels); the spare channel keeps 3-7% of their mass. A merge is a
  symmetric point: the task gradient cannot split two streams whose gates are equal. 100x kicks on
  the gate's gradient split 0/8 (Adam: a 100x kick moves a parameter about as far as 3 normal steps).

THE RUNS AND THE CODE PATH: S21's (explore_merge_kick): X's recorded A4k4 runs (arm A, k=4, conv
'layer' width 4, S=4, P=4, task_for(4, 4), lr 1e-3) merged at 9600 that stayed merged:
stream_recipe 247, 248, 251, 258 and scale_axes 221, 223, 225, 226, re-run to T0 = 9600 through the
recorded run's own path (main-branch modules loaded read-only at S21's MAIN_SHA; S21's rerun).

AT 9600, before any intervention (probe batch test_multilayer_binding.probe_batch(task, seed), model in
eval mode; the gate state h_t = tanh(W_in v_t + W_h h_{t-1}) recomputed from the model's own embedding
exactly as test_instrument_v2.Instrument.gates computes it, and softmax(h W_g^T) checked against the
model's read gate), for the merged pair (the two streams sharing a channel in the stream -> channel map
at 9600, test_scale_axes.routing_k) and for the non-merged pair (the other two streams):
  - the mean gate state h at value positions (3j + 2, j < S*P) per stream, h_a and h_b;
  - ||h_a - h_b|| / mean ||h|| (the mean over every value position of the probe batch);
  - the cosine between W_g's row difference and (h_a - h_b): rows (shared channel, spare channel) for
    the merged pair, (channel of a, channel of b) for the non-merged pair; and the largest |cosine|
    over all 6 row pairs;
  - the mean read gate per stream at value positions, and the mean read-gate mass per channel over
    all positions.
Then T_CONT = 9600 updates three ways from that state, each on its own deep copy of the model and of
the optimizer state (copied, not shared; the batch stream replayed identically, onset_run's step line
for line, as S21's continuation):
  (a) CONTROL: plain.
  (b) SPLIT: c* = the channel with the largest mean read-gate mass over all positions of the probe
      batch, c0 = the smallest; W_g[c0] = W_g[c*], then Gaussian noise of std 0.1 x std(W_g[c*])
      (torch.std, unbiased, of the row before the operation) added to both rows (a dedicated
      torch.Generator seeded with the run's seed; the row-c* noise drawn first, then row c0's);
      Adam's state for W_g removed (its moments and step count restart at the next update).
  (c) NOISE: the same two noise vectors added to rows c* and c0 without the copy; the same Adam reset.
At every evaluation (every 1200): the stream -> channel map, the per-stream mean read gates at value
positions, per-stream and overall held-out accuracy (S21's).
Per-run reading as S21's: "split" if at some evaluation up to 19200 the merged pair (the recorded end
map's) is on different channels and held-out accuracy is >= 0.95; otherwise "merged".

RULES (fixed before any run)
- "splitting un-merges" if SPLIT splits >= 4/8 and NOISE <= 1/8; "noise suffices" if NOISE >= 4/8;
  "neither works" if SPLIT <= 1/8 (every reading that applies is printed; none otherwise).
- Separately: "the gate state tells the merged streams apart" if ||h_a - h_b|| / mean ||h|| for the
  merged pair is >= 0.5 x that of the non-merged pair in at least 6/8 runs.

CHECKS: before the batch, the machinery on one run of each path (rerun to 1200, continuations of 1200):
the reruns reproduce the record, CONTROL reproduces the recorded evaluation at 2400, the three
continuations see identical batches and start from separately stored optimizer states (CONTROL's
equal to the run's, SPLIT's and NOISE's equal to it except W_g's, which is absent; W_g's Adam step
count is 1 after the first update), the run's optimizer and model are left unchanged, SPLIT's and
NOISE's W_g rows after the operation match the spec (recomputed from a fresh generator; the other rows
unchanged), and the recomputed gate equals the model's. In each job: the rerun reproduces its
recorded curve to 9600; CONTROL reproduces it to 19200; the same checks on the continuations.
"""

import copy
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F

import explore_common as ec
import explore_common2 as c2
import explore_merge_kick as s21
import test_binding_onset as tbo
from test_binding_onset import BATCH, EVAL_EVERY
from test_multilayer_binding import probe_batch
from test_channel_binding import eval_batch
from test_instrument_v2 import TAU_END

tsa = s21.tsa
NAME = "merge_split"
IDEA = "break the symmetry of a merged pair's read-gate rows: copy the busiest channel's W_g row to the idlest, plus noise"
SOURCE = ("batch 7's S21 (merged pairs have identical gates: a symmetric point the task gradient cannot "
          "leave; 100x kicks split 0/8)")
CHANGE = ("none to the recorded runs up to 9600; then 9600 updates three ways: CONTROL; SPLIT (W_g[c0] = "
          "W_g[c*] + noise, W_g[c*] + noise, W_g's Adam state reset); NOISE (the same noise, no copy, same reset)")
PAIRING = "each run's SPLIT and NOISE against its own CONTROL (same state at 9600, same batches) and its recorded curve"
TASK = s21.TASK
T0, T_CONT = s21.T0, s21.T_CONT
NOISE_REL = 0.1
BIND = s21.BIND
SPLIT_MIN, NOT_MAX = 4, 1
GATE_REL, GATE_RUNS = 0.5, 6
KINDS = ("CONTROL", "SPLIT", "NOISE")

_A = dict(tsa.ARM["A4k4"], lr=tsa.LR, iters=T0 + 3 * T_CONT, prio=1,
          sched=(f"lr {tsa.LR:g} throughout; from {T0}: CONTROL plain; SPLIT W_g[c0] = W_g[c*] + noise and "
                 f"W_g[c*] + noise (std {NOISE_REL:g} x std(W_g[c*])), W_g's Adam state reset; NOISE the same "
                 f"noise without the copy, same reset"))
ARMS = {
    "SR_A4k4": dict(_A, key="SR_A4k4", seeds=s21.SEEDS_SR, label="test_stream_recipe's A4k4 (X), merged pair stalled"),
    "SA_A4k4": dict(_A, key="SA_A4k4", seeds=s21.SEEDS_SA, label="test_scale_axes's A4k4 (X), merged pair stalled"),
}


# ── the gate state ───────────────────────────────────────────────────────────
@torch.no_grad()
def gate_states(m, x):
    """h_t for every position (Instrument.gates' recurrence, on the model's own embedding), the read
    gate recomputed from it, and the model's read gate."""
    was = m.training
    m.eval()
    try:
        assert m.gate_kind == "recurrent" and not m.gate_ln and m.mode == "sym"
        v = m.embed(x)
        h = torch.zeros(v.shape[0], m.h_gate, dtype=v.dtype)
        hs = []
        for t in range(v.shape[1]):
            h = torch.tanh(v[:, t] @ m.W_in.T + h @ m.W_h.T)
            hs.append(h)
        H = torch.stack(hs, dim=1)
        gr_m = m(x, TAU_END)[2]
    finally:
        m.train(was)
    return H, F.softmax(H @ m.W_g.detach().T, dim=-1), gr_m


def pairs_from_map(cmap, rec_pair):
    """The merged pair (two streams sharing a channel in cmap; else the recorded pair), the other two
    streams, and the channels not in the map."""
    S = len(cmap)
    sh = [s for s in range(S) if cmap.count(cmap[s]) > 1]
    pair = sh if len(sh) == 2 else list(rec_pair)
    other = [s for s in range(S) if s not in pair]
    return pair, other, len(sh) == 2


def rows_for(a, b, cmap, spare, k):
    if cmap[a] != cmap[b]:
        return cmap[a], cmap[b]
    return cmap[a], (spare[0] if spare else (cmap[a] + 1) % k)


def gate_measures(m, probe, cmap, rec_pair):
    x = probe[0]
    S, P = TASK.S, TASK.P
    j = torch.arange(S * P)
    ctx, vp = x[:, 3 * j], 3 * j + 2
    H, gr, gr_m = gate_states(m, x)
    Hv = H[:, vp]                                                       # (B, n, h_gate)
    hbar = [Hv[ctx == s].mean(0) for s in range(S)]
    mn = Hv.norm(dim=-1).mean()
    W = m.W_g.detach()
    k = W.shape[0]
    spare = [c for c in range(k) if c not in cmap]
    pair, other, merged = pairs_from_map(cmap, rec_pair)

    def ratio(a, b):
        return float((hbar[a] - hbar[b]).norm() / mn)

    def cos(c1, c2, a, b):
        return float(F.cosine_similarity((W[c1] - W[c2]).unsqueeze(0), (hbar[a] - hbar[b]).unsqueeze(0)))

    def maxcos(a, b):
        return max(abs(cos(c1, c2, a, b)) for c1 in range(k) for c2 in range(c1 + 1, k))

    rm, ro = rows_for(*pair, cmap, spare, k), rows_for(*other, cmap, spare, k)
    gv = gr_m[:, vp]
    mass = gr_m.mean((0, 1))
    return dict(map=list(cmap), pair=pair, other=other, map_merged=merged, spare=spare,
                ratio_merged=ratio(*pair), ratio_other=ratio(*other),
                rows_merged=list(rm), rows_other=list(ro),
                cos_merged=cos(*rm, *pair), cos_other=cos(*ro, *other),
                maxcos_merged=maxcos(*pair), maxcos_other=maxcos(*other),
                h_norm=float(mn), h_diff_merged=float((hbar[pair[0]] - hbar[pair[1]]).norm()),
                h_diff_other=float((hbar[other[0]] - hbar[other[1]]).norm()),
                stream_gate=[[round(v, 4) for v in gv[ctx == s].mean(0).tolist()] for s in range(S)],
                mass=[round(v, 5) for v in mass.tolist()], cstar=int(mass.argmax()), c0=int(mass.argmin()),
                gate_recompute_diff=float((gr - gr_m).abs().max()))


# ── the operations ───────────────────────────────────────────────────────────
def noise_for(w, seed):
    sd = NOISE_REL * w.std()
    g = torch.Generator().manual_seed(seed)
    n1 = torch.randn(w.shape, generator=g, dtype=w.dtype) * sd                   # row c*
    n2 = torch.randn(w.shape, generator=g, dtype=w.dtype) * sd                   # row c0
    return sd, n1, n2


def expected_rows(W0, kind, seed, cs, c0):
    """The spec, recomputed from a fresh generator (for the check)."""
    W = W0.clone()
    w = W0[cs].clone()
    _, n1, n2 = noise_for(w, seed)
    if kind == "SPLIT":
        W[c0] = w + n2
        W[cs] = w + n1
    elif kind == "NOISE":
        W[cs] = W0[cs] + n1
        W[c0] = W0[c0] + n2
    return W


def apply_op(m, opt, kind, seed, cs, c0):
    if kind == "CONTROL":
        return {}
    W = m.W_g
    with torch.no_grad():
        w = W[cs].clone()
        before = float((W[cs] - W[c0]).norm())
        sd, n1, n2 = noise_for(w, seed)
        if kind == "SPLIT":
            W[c0] = w + n2
            W[cs] = w + n1
        else:
            W[cs] += n1
            W[c0] += n2
    opt.state.pop(W, None)                                              # Adam's state for W_g reset
    return dict(noise_sd=float(sd), n1_std=float(n1.std()), n2_std=float(n2.std()),
                row_dist_before=before, row_dist_after=float((W[cs] - W[c0]).detach().norm()))


def continuation(m0, opt0, rng0, seed, t0, steps, kind, cs, c0, data, probe, lr, ref, rec_pair):
    """S21's continuation (onset_run's step, line for line, on deep copies), with the operation applied
    to the copy before the first update. ref: the run's optimizer state snapshotted at t0."""
    m = copy.deepcopy(m0)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    opt.load_state_dict(copy.deepcopy(opt0.state_dict()))
    names = [n for n, _ in m.named_parameters()]
    iw = names.index("W_g")
    W0 = m.W_g.detach().clone()
    info = apply_op(m, opt, kind, seed, cs, c0)
    st = s21.opt_state(opt)
    if kind == "CONTROL":
        equal = s21.same_state(ref, st)
    else:
        equal = (st[iw] == {} and s21.same_state([x for i, x in enumerate(ref) if i != iw],
                                                 [x for i, x in enumerate(st) if i != iw]))
    rows_ok = torch.equal(m.W_g.detach(), expected_rows(W0, kind, seed, cs, c0))
    start = dict(equal=equal, separate=not (s21.ptrs(opt) & s21.ptrs(opt0)), rows_ok=rows_ok)
    g = torch.Generator()
    g.set_state(rng0.get_state())
    dig, curve, evals = hashlib.sha256(), [], []
    step_after1 = None
    for step in range(1, steps + 1):
        tokens, _ = TASK.make_batch(BATCH, g)
        dig.update(tokens.numpy().tobytes())
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        ql, qt, _ = TASK.select(m(inp, TAU_END)[0], tgt, None)
        loss = F.cross_entropy(ql, qt)
        opt.zero_grad(); loss.backward()
        opt.step()
        if step == 1:
            po = next(p for n, p in m.named_parameters() if n != "W_g" and "step" in opt.state[p])
            step_after1 = [float(opt.state[m.W_g]["step"]), float(opt.state[po]["step"])]
        if step % EVAL_EVERY == 0:
            acc, el = tbo.evaluate(m, TASK, data)
            curve.append([t0 + step, acc, el])
            rk = tsa.routing_k(m, TASK, probe)
            evals.append(dict(step=t0 + step, acc=acc, ch_map=rk["ch_map"], shared=rk["shared"],
                              stream_gate=rk["stream_gate"], stream_acc=tsa.stream_acc(m, TASK, data)))
    end_map = evals[-1]["ch_map"] if evals else None
    gate_end = gate_measures(m, probe, end_map, rec_pair) if end_map else None
    return dict(curve=curve, evals=evals, digest=dig.hexdigest(), op=info, start_equal=start["equal"],
                start_separate=start["separate"], rows_ok=rows_ok, step_after1=step_after1,
                gate_end=gate_end), m


def split_at(evals, pair):
    return s21.split_at(evals, pair)


def params_snapshot(m):
    return {n: p.detach().clone() for n, p in m.named_parameters()}


def same_params(a, m):
    return all(torch.equal(a[n], p.detach()) for n, p in m.named_parameters())


def protocol(arm, seed, t0, cont, lr):
    """Rerun to t0, the gate measures, and the three continuations. Returns the pieces for the job
    record and the check."""
    rec0 = s21.recorded(arm, seed)
    rec_pair = s21.merged_pair(rec0)
    rec, m, opt, rng = s21.rerun(arm, seed, t0)
    data, probe = eval_batch(TASK), probe_batch(TASK, seed)
    snap, psnap = s21.opt_state(opt), params_snapshot(m)
    rk = tsa.routing_k(m, TASK, probe)
    gm = gate_measures(m, probe, rk["ch_map"], rec_pair)
    cs, c0 = gm["cstar"], gm["c0"]
    conts, models = {}, {}
    for kind in KINDS:
        conts[kind], models[kind] = continuation(m, opt, rng, seed, t0, cont, kind, cs, c0, data, probe, lr,
                                                 snap, rec_pair)
    want0 = [c for c in rec0["curve"] if c[0] <= t0]
    want1 = [c for c in rec0["curve"] if t0 < c[0] <= t0 + cont]
    ctl = conts["CONTROL"]
    flags = dict(
        reproduces=rec["curve"] == want0,
        control_matches=ctl["curve"] == want1 and len(want1) == len(ctl["curve"]),
        batches_identical=len({c["digest"] for c in conts.values()}) == 1,
        opt_separate=all(c["start_equal"] and c["start_separate"] for c in conts.values()),
        rows_ok=all(c["rows_ok"] for c in conts.values()),
        adam_reset=all(conts[k]["step_after1"][0] == 1.0 and conts[k]["step_after1"][1] == t0 + 1
                       for k in ("SPLIT", "NOISE")) and conts["CONTROL"]["step_after1"] == [t0 + 1.0, t0 + 1.0],
        run_unchanged=s21.same_state(snap, s21.opt_state(opt)) and same_params(psnap, m),
        gate_recompute=gm["gate_recompute_diff"] < 1e-6,
        cstar_ne_c0=cs != c0)
    return dict(rec0=rec0, rec_pair=rec_pair, rec=rec, gm=gm, rk=rk, conts=conts, models=models, flags=flags,
                probe=probe, want1=want1)


def run_job(arm, seed):
    a = ARMS[arm]
    dry = a["iters"] < T0 + 3 * T_CONT
    t0, cont = (2 * EVAL_EVERY, EVAL_EVERY) if dry else (T0, T_CONT)
    p = protocol(arm, seed, t0, cont, a["lr"])
    conts, rec0, pair = p["conts"], p["rec0"], p["rec_pair"]
    end = tsa.scale_stats(p["models"]["SPLIT"], TASK, p["probe"], "end")
    out = dict(ok=True, seed=seed, k=4, dry=dry, t0=t0, cont=cont, merged_pair=pair,
               recorded_end_map=rec0["end"]["ch_map"], recorded_acc=rec0["acc"],
               rerun_curve=p["rec"]["curve"], map_t0=p["rk"]["ch_map"], stream_gate_t0=p["rk"]["stream_gate"],
               gate_t0=p["gm"], cstar=p["gm"]["cstar"], c0=p["gm"]["c0"],
               control=conts["CONTROL"], split=conts["SPLIT"], noise=conts["NOISE"], **p["flags"])
    for kind in KINDS:
        out[f"split_{kind.lower()}"] = split_at(conts[kind]["evals"], pair)
    # the record's summary fields describe SPLIT's end state (the progress line and tag)
    sp = conts["SPLIT"]
    out.update(curve=[c for c in p["rec"]["curve"]] + sp["curve"], acc=sp["curve"][-1][1] if sp["curve"] else None,
               val_cos=end["val_cos"], key_cos=end["key_cos"], ctx_cos=end["ctx_cos"], end=end, stats=[])
    out["transition"] = tbo.transition(out["curve"])
    out["reading"] = "split" if out["split_split"] is not None else "merged"
    return out


def segments(arm):
    a = ARMS[arm]
    return [(a["iters"], TASK, tsa.builder_for(tsa.ARM["A4k4"])(tsa.ARM["A4k4"], a["seeds"][0]), a["lr"],
             tsa.scale_stats)]


# ── CHECKs ───────────────────────────────────────────────────────────────────
def check():
    ok = True
    der = s21.derived_runs()
    rows = [(f"the runs are S21's derived list (stream_recipe {der['SR_A4k4']}, scale_axes {der['SA_A4k4']})",
             der["SR_A4k4"] == s21.SEEDS_SR and der["SA_A4k4"] == s21.SEEDS_SA)]
    for arm, seed in (("SA_A4k4", s21.SEEDS_SA[0]), ("SR_A4k4", s21.SEEDS_SR[0])):
        t0 = cont = EVAL_EVERY
        p = protocol(arm, seed, t0, cont, ARMS[arm]["lr"])
        f, gm, cc = p["flags"], p["gm"], p["conts"]
        rows.append((f"{s21.SRC[arm]['path']} seed {seed}: rerun to {t0} reproduces the record ({f['reproduces']}); "
                     f"CONTROL's {cont} updates reproduce the recorded evaluation at {t0 + cont} ({cc['CONTROL']['curve']} "
                     f"vs {p['want1']}); identical batches in the 3 continuations ({f['batches_identical']}); start states "
                     f"equal to the run's (W_g's absent for SPLIT/NOISE) and stored apart from it ({f['opt_separate']}); "
                     f"W_g's Adam step after the first update CONTROL/SPLIT/NOISE "
                     f"{[cc[k]['step_after1'] for k in KINDS]} (W_g, another) ({f['adam_reset']}); the run's optimizer "
                     f"and model unchanged ({f['run_unchanged']}); W_g rows after the operation match the spec "
                     f"({f['rows_ok']}; c* {gm['cstar']}, c0 {gm['c0']}, mass {gm['mass']}, noise sd "
                     f"{cc['SPLIT']['op'].get('noise_sd', float('nan')):.2e}); recomputed gate = model's (max diff "
                     f"{gm['gate_recompute_diff']:.1e})", all(f.values())))
    for nm, v in rows:
        print(f"  CHECK {NAME}: {nm}: {'ok' if v else 'FAIL'}", flush=True)
        ok &= bool(v)
    return ok


# ── Report ───────────────────────────────────────────────────────────────────
def readings(n_split, n_noise):
    out = []
    if n_split >= SPLIT_MIN and n_noise <= NOT_MAX:
        out.append("splitting un-merges")
    if n_noise >= SPLIT_MIN:
        out.append("noise suffices")
    if n_split <= NOT_MAX:
        out.append("neither works")
    return out or ["no reading applies"]


def gate_reading(n):
    return "the gate state tells the merged streams apart" if n >= GATE_RUNS else "rule not met"


def ev_str(evals, pair):
    return s21.ev_str(evals, pair)


def report(store):
    me = sys.modules[__name__]
    c2.print_screen_header2(me)
    done = [r for r in store["runs"].values() if r.get("ok")]
    t0, t1 = (done[0]["t0"], done[0]["t0"] + done[0]["cont"]) if done else (T0, T0 + T_CONT)
    print(f"  per run (map = stream -> channel at each evaluation, as acc:map; '|' = the merged pair on different "
          f"channels; continuations from {t0} to {t1}; gate state at the start of the continuations: "
          f"ratio = ||h_a - h_b|| / mean ||h|| at value positions, cos = cos(W_g[r1] - W_g[r2], h_a - h_b)):")
    n = {k: 0 for k in KINDS}
    n_done, n_gate = 0, 0
    for arm, a in ARMS.items():
        for s in a["seeds"]:
            r = store["runs"].get(f"{arm}|{s}")
            if r is None or not r.get("ok"):
                print(f"    {arm:<8} {s:>4} {ec.tag(r)}")
                continue
            n_done += 1
            pair = r["merged_pair"]
            g = r["gate_t0"]
            yn = lambda v: "yes" if v else "NO"
            print(f"    {arm:<8} {s:>4}  merged pair streams {pair} (recorded end map {r['recorded_end_map']}, rec acc "
                  f"{r['recorded_acc']:.3f}; map at {r['t0']} {r['map_t0']}; c* {r['cstar']}, c0 {r['c0']}, mass "
                  f"{g['mass']})")
            print(f"    {'':<8} {'':>4}  rerun reproduces: {yn(r['reproduces'])}; CONTROL = record to {r['t0'] + r['cont']}: "
                  f"{yn(r['control_matches'])}; batches identical: {yn(r['batches_identical'])}; start states equal and "
                  f"separate: {yn(r['opt_separate'])}; W_g Adam reset: {yn(r['adam_reset'])}; rows as spec: "
                  f"{yn(r['rows_ok'])}; run unchanged: {yn(r['run_unchanged'])}; gate recompute: {yn(r['gate_recompute'])}")
            sg = g["stream_gate"]
            print(f"    {'':<8} {'':>4}  at {r['t0']}: merged {g['pair']} ratio {g['ratio_merged']:.3f}, cos rows "
                  f"{g['rows_merged']} {g['cos_merged']:+.3f} (max |cos| {g['maxcos_merged']:.3f}); non-merged "
                  f"{g['other']} ratio {g['ratio_other']:.3f}, cos rows {g['rows_other']} {g['cos_other']:+.3f} (max "
                  f"|cos| {g['maxcos_other']:.3f}); mean ||h|| {g['h_norm']:.3f}; read gates at values "
                  + " ".join(f"s{i}:" + "/".join(f"{v:.2f}" for v in sg[i]) for i in range(len(sg))))
            ok_g = g["ratio_merged"] >= GATE_REL * g["ratio_other"]
            n_gate += ok_g
            for lab in KINDS:
                c = r[lab.lower()]
                last = c["evals"][-1] if c["evals"] else None
                sa = "/".join(f"{v:.2f}" for v in last["stream_acc"]) if last else "--"
                print(f"    {'':<8} {'':>4}  {lab:<7} {ev_str(c['evals'], pair)}")
                ge = c.get("gate_end") or {}
                extra = (f"; op: noise sd {c['op']['noise_sd']:.2e}, ||W_g[c*] - W_g[c0]|| "
                         f"{c['op']['row_dist_before']:.3f} -> {c['op']['row_dist_after']:.3f}" if c.get("op") else "")
                print(f"    {'':<8} {'':>4}  {'':<7} at {last['step'] if last else '--'}: acc "
                      f"{last['acc'] if last else float('nan'):.3f}, per stream {sa}, merged pair on channels "
                      f"{[last['ch_map'][pair[0]], last['ch_map'][pair[1]]] if last else '--'}, merged ratio "
                      f"{ge.get('ratio_merged', float('nan')):.3f}{extra}")
            print(f"    {'':<8} {'':>4}  READING: " + "; ".join(
                f"{lab} {'split' if r['split_' + lab.lower()] else 'merged'}"
                + (f" (at {r['split_' + lab.lower()]})" if r["split_" + lab.lower()] else "") for lab in KINDS)
                  + f"; gate state merged ratio >= {GATE_REL} x non-merged: {'yes' if ok_g else 'no'}")
            for lab in KINDS:
                n[lab] += r["split_" + lab.lower()] is not None
    rd = readings(n["SPLIT"], n["NOISE"])
    gr = gate_reading(n_gate)
    print(f"  SPLIT {n['SPLIT']}/{n_done}, NOISE {n['NOISE']}/{n_done}, CONTROL {n['CONTROL']}/{n_done} split by "
          f"{t1}")
    print(f"  RULE S24 ('splitting un-merges' if SPLIT >= {SPLIT_MIN}/8 and NOISE <= {NOT_MAX}/8; 'noise suffices' if "
          f"NOISE >= {SPLIT_MIN}/8; 'neither works' if SPLIT <= {NOT_MAX}/8): {'; '.join(rd)}")
    print(f"  RULE S24 gate state (merged ratio >= {GATE_REL} x non-merged in {n_gate}/{n_done}; needs >= {GATE_RUNS}/8): {gr}")
    return dict(n=n_done, split=n["SPLIT"], noise=n["NOISE"], control=n["CONTROL"], rule=rd, gate_runs=n_gate,
                gate_rule=gr)
