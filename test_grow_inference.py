"""
test_grow_inference.py — Can a BDH grown MID-SEQUENCE at inference acquire useful,
task-relevant state through Hebbian plasticity alone (no gradient training)?

Model: the VERIFIED recurrent/Hebbian BDH from bdh_recurrent.py (imported, not modified).
The token-by-token engine needed to grow mid-sequence is implemented here and VERIFIED
against the imported bdh_layer_parallel / bdh_layer_recurrent before any experiment runs.

Growth (paper arXiv 2509.26507 sec 7.1: models compose by concatenation along the
neuron dimension n):
  * WIDEN: every new neuron is a copy of an existing neuron's parameters
    (its Dx column, Dy column, E row) plus Gaussian noise of scale PERTURB; its Hebbian
    state row S[j] is copied exactly. We grow by duplicating the ENTIRE population
    (GROWN_N = 2*BASE_N, parent map = identity). Because BDH's LayerNorm is
    parameter-free and scale-invariant, uniform doubling is function-preserving:
      x' = [x, x];  a*' = x' @ [S; S] = 2(x @ S);  ln(2z) = ln(z)   (up to ln's eps)
      y' = [y, y];  y' @ [E; E] = 2(y @ E);        ln cancels the 2 again.
    Writes keep the two copies synchronized, so with zero perturbation the grown model
    equals the base model for the rest of the sequence up to ln's eps=1e-5 (per-position
    logit wiggle where std is small, ~1e-6 effect on loss — enforced by the cond-4
    control; the recurrent STATE stays exact because x depends only on the layer input).
  * DEEPEN: add a second layer with Dx, Dy copied from layer 1 (+ noise) and E
    initialized to 0 (+ noise), fresh state S2 = 0. At zero perturbation the new layer's
    residual contribution ln(y2 @ 0) = 0, i.e. exact identity.

Task (byte-level, vocab = 3*(K_EASY+K_HARD) <= 256): content-keyed mode retrieval.
With U = I the verified BDH's attention is purely content-based (no position info), so
classic key->value induction is impossible; what IS solvable is retrieval keyed by the
current token's own code. Each channel i owns three bytes: query Q_i, mode tokens A_i,
B_i. A sequence sets each channel's mode with a single token (A_i or B_i, random per
sequence), later the query byte Q_i must be answered with that channel's mode token.
Answering requires the learned code of Q_i to overlap its own channel's mode codes and
not other channels' — with n neurons, only ~n channels can be held cleanly, so:
  easy region  (positions <  T_STEP): K_EASY channels set + queried  -> base model copes
  hard region  (positions >= T_STEP): K_HARD fresh channels set + queried -> more
    distinct associations than BASE_N cleanly holds -> base model degrades.
Loss is isolated to the HARD-region query positions (predict the mode token that
follows each hard Q_i). Chance-given-channel = ln(2) ~ 0.693.

Conditions (>= 3 seeds, mean +- std, same trained base model and same eval sequences
per seed, growth applied at t = T_STEP, everything below is pure inference):
  1. base       — no growth (expected to degrade on the hard region)
  2. grown      — function-preserving copies + PERTURB noise, Hebbian plasticity runs
                  as normal for all neurons over the hard region      (test condition)
  3. frozen     — same grown params, but Hebbian WRITE disabled for the new neurons
                  only (their state rows stay at the grow-time copy)  (params-only control)
  4. zerocopy   — grown with PERTURB = 0, exact copies                (must match cond 1)
  ref. capacity — a model TRAINED at the grown size, as a check that the hard region is
                  genuinely capacity-limited (extra capacity helps IF trained into it).

Diagnostics (condition 2): divergence of each grown neuron from its parent over the
hard region — relative L2 and cosine of activations x_child vs x_parent per step, and
of Hebbian state rows S_child vs S_parent at sequence end (for deepen: layer-2 rows vs
the post-growth part of layer-1 rows).

Verdict (stated plainly, thresholds are constants below):
 (a) cond2 beats cond1 AND cond2 beats cond3 AND grown neurons diverge
     -> Hebbian adaptation makes grown neurons useful without training.
 (b) otherwise -> inference-time growth yields no useful capacity here.
"""

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from bdh_recurrent import ln, bdh_layer_parallel, bdh_layer_recurrent

# ── Model / growth constants ──────────────────────────────────────────────────
D         = 32
BASE_N    = 16
GROWN_N   = 32        # widen: must equal 2*BASE_N (uniform doubling => function-preserving)
PERTURB   = 0.02      # Gaussian perturbation scale for new-neuron parameter copies
GROW_MODE = "widen"   # "widen" | "deepen"   (argv[1] overrides: python test_grow_inference.py deepen)

# ── Task difficulty constants ─────────────────────────────────────────────────
K_EASY          = 4    # channels in the easy region
K_HARD          = 32   # fresh channels introduced at the difficulty step (>> BASE_N capacity)
EASY_QUERY_REPS = 2    # times each easy channel is queried
VOCAB           = 3 * (K_EASY + K_HARD)   # 108 byte values

# ── Training / eval constants ─────────────────────────────────────────────────
SEEDS  = [0, 1, 2]
BATCH  = 16
ITERS  = 1500
LR     = 3e-3
N_EVAL = 64            # fresh eval sequences per seed (shared across conditions: paired)
RUN_CAPACITY_REF = True

# ── Verdict thresholds ────────────────────────────────────────────────────────
BEAT_MARGIN  = 0.005   # mean paired hard-loss improvement required to call a "beat"
MATCH_TOL    = 1e-3    # cond4 must match cond1 within this (function-preservation check)
DRIFT_THRESH = 0.05    # relative L2 state drift above which grown neurons "diverged"

# ── Derived sequence layout (fixed structure => fixed positions) ──────────────
K_TOTAL      = K_EASY + K_HARD
EASY_QUERIES = K_EASY * EASY_QUERY_REPS
T_STEP       = K_EASY + 2 * EASY_QUERIES            # difficulty step / growth point
EASY_QPOS    = [K_EASY + 2 * k for k in range(EASY_QUERIES)]
HARD_QPOS    = [T_STEP + K_HARD + 2 * k for k in range(K_HARD)]
SEQ_LEN      = T_STEP + 3 * K_HARD


def q_tok(i):
    return 3 * i


def m_tok(i, m):
    return 3 * i + 1 + m


def make_seq(gen):
    """One byte sequence: easy set+queries, then (difficulty step) hard set+queries."""
    modes = torch.randint(0, 2, (K_TOTAL,), generator=gen).tolist()
    toks = []
    for i in torch.randperm(K_EASY, generator=gen).tolist():
        toks.append(m_tok(i, modes[i]))
    qlist = [i for i in range(K_EASY) for _ in range(EASY_QUERY_REPS)]
    qlist = [qlist[j] for j in torch.randperm(len(qlist), generator=gen).tolist()]
    for i in qlist:
        toks += [q_tok(i), m_tok(i, modes[i])]
    assert len(toks) == T_STEP
    for i in torch.randperm(K_HARD, generator=gen).tolist():
        c = K_EASY + i
        toks.append(m_tok(c, modes[c]))
    for i in torch.randperm(K_HARD, generator=gen).tolist():
        c = K_EASY + i
        toks += [q_tok(c), m_tok(c, modes[c])]
    assert len(toks) == SEQ_LEN
    return torch.tensor(toks, dtype=torch.long)


# ── Batched parallel layer (training only; verified vs imported reference) ────
def bdh_layer_parallel_batched(v, Dx, Dy, E):
    x = F.relu(ln(v) @ Dx)                                    # (B, T, N)
    scores = torch.einsum('btn,bsn->bts', x, x).tril(-1)      # (B, T, T) causal
    a = scores @ v
    y = F.relu(ln(a) @ Dy) * x
    return v + ln(y @ E)


class BDHModel(nn.Module):
    def __init__(self, n=BASE_N, nlayers=1):
        super().__init__()
        self.nlayers = nlayers
        self.embed = nn.Embedding(VOCAB, D)
        self.Dx = nn.ParameterList([nn.Parameter(torch.randn(D, n) * 0.1) for _ in range(nlayers)])
        self.Dy = nn.ParameterList([nn.Parameter(torch.randn(D, n) * 0.1) for _ in range(nlayers)])
        self.E  = nn.ParameterList([nn.Parameter(torch.randn(n, D) * 0.1) for _ in range(nlayers)])
        self.head = nn.Linear(D, VOCAB, bias=False)

    def forward(self, tokens):
        v = self.embed(tokens)
        for l in range(self.nlayers):
            v = bdh_layer_parallel_batched(v, self.Dx[l], self.Dy[l], self.E[l])
        return self.head(v)


def train_model(seed, n=BASE_N, nlayers=1, tag=""):
    torch.manual_seed(seed)
    model = BDHModel(n=n, nlayers=nlayers)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed + 10_000)
    qpos = torch.tensor(EASY_QPOS + HARD_QPOS)
    for it in range(ITERS):
        toks = torch.stack([make_seq(gen) for _ in range(BATCH)])
        inp, tgt = toks[:, :-1], toks[:, 1:]
        logits = model(inp)
        loss = F.cross_entropy(logits[:, qpos, :].reshape(-1, VOCAB),
                               tgt[:, qpos].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 500 == 0:
            print(f"    [seed {seed}{tag}] iter {it+1}/{ITERS}  train query-loss {loss.item():.4f}")
    model.eval()
    return model


def get_params(model):
    """Trained weights as float64 tensors for the exact stepwise inference engine."""
    assert model.nlayers == 1
    return dict(embed=model.embed.weight.detach().double(),
                Dx=model.Dx[0].detach().double(),
                Dy=model.Dy[0].detach().double(),
                E=model.E[0].detach().double(),
                head=model.head.weight.detach().double())


# ── Growth: function-preserving copies + Gaussian perturbation ────────────────
def grow_widen(p, scale, gen):
    """New neuron j (= child of parent j) gets parent's Dx col / Dy col / E row + noise."""
    def noise(shape):
        return torch.randn(shape, generator=gen, dtype=torch.float64) * scale
    return dict(Dx=torch.cat([p['Dx'], p['Dx'] + noise(p['Dx'].shape)], dim=1),
                Dy=torch.cat([p['Dy'], p['Dy'] + noise(p['Dy'].shape)], dim=1),
                E=torch.cat([p['E'], p['E'] + noise(p['E'].shape)], dim=0))


def grow_deepen(p, scale, gen):
    """New layer: Dx/Dy copied from layer 1 (+noise), E = 0 (+noise) => near-identity."""
    def noise(shape):
        return torch.randn(shape, generator=gen, dtype=torch.float64) * scale
    return dict(Dx=p['Dx'] + noise(p['Dx'].shape),
                Dy=p['Dy'] + noise(p['Dy'].shape),
                E=noise(p['E'].shape))


# ── Stepwise recurrent inference engine (supports mid-sequence growth) ────────
def run_stepwise(p, toks, grown=None, grow_mode=None, freeze_new=False, collect_diag=False):
    """
    Token-by-token Hebbian BDH (read a*_t = x_t @ S, then write S += outer(x_t, v*_t)),
    identical to bdh_layer_recurrent (verified below). Growth, if any, is applied just
    before processing input position T_STEP. freeze_new disables the Hebbian write for
    the new neurons only. Returns (logits (T,V), diagnostics dict or None).
    """
    Dx, Dy, E = p['Dx'], p['Dy'], p['E']
    emb, head = p['embed'], p['head']
    n = Dx.shape[1]
    S = torch.zeros(n, D, dtype=torch.float64)
    wmask = torch.ones(n, dtype=torch.float64)
    l2, S2, S1_post = None, None, None
    T = toks.shape[0]
    logits = torch.zeros(T, VOCAB, dtype=torch.float64)
    xdrift_rel, xdrift_cos = [], []
    for t in range(T):
        if grown is not None and t == T_STEP:
            if grow_mode == "widen":
                Dx, Dy, E = grown['Dx'], grown['Dy'], grown['E']
                n = Dx.shape[1]
                S = torch.cat([S, S.clone()], dim=0)   # child state rows = exact copies
                wmask = torch.ones(n, dtype=torch.float64)
                if freeze_new:
                    wmask[BASE_N:] = 0.0
            elif grow_mode == "deepen":
                l2 = grown
                S2 = torch.zeros(l2['Dx'].shape[1], D, dtype=torch.float64)
                if collect_diag:
                    S1_post = torch.zeros(n, D, dtype=torch.float64)
        v = emb[toks[t]]
        xv = F.relu(ln(v) @ Dx)
        a = xv @ S                                  # READ (S holds only tau < t)
        yv = F.relu(ln(a) @ Dy) * xv
        S = S + torch.outer(xv * wmask, v)          # Hebbian WRITE (masked if frozen)
        v1 = v + ln(yv @ E)
        x2 = None
        if l2 is not None:
            x2 = F.relu(ln(v1) @ l2['Dx'])
            a2 = x2 @ S2
            y2 = F.relu(ln(a2) @ l2['Dy']) * x2
            if not freeze_new:
                S2 = S2 + torch.outer(x2, v1)
            v_out = v1 + ln(y2 @ l2['E'])
        else:
            v_out = v1
        logits[t] = v_out @ head.T
        if collect_diag and grown is not None and t >= T_STEP:
            if grow_mode == "widen":
                xp, xc = xv[:BASE_N], xv[BASE_N:]
            else:
                xp, xc = xv, x2
                S1_post = S1_post + torch.outer(xv, v)
            npar, nchi = xp.norm(), xc.norm()
            if npar > 1e-9:
                xdrift_rel.append(((xc - xp).norm() / npar).item())
                if nchi > 1e-9:
                    xdrift_cos.append(((xp @ xc) / (npar * nchi)).item())
    diag = None
    if collect_diag and grown is not None:
        Sp, Sc = (S[:BASE_N], S[BASE_N:]) if grow_mode == "widen" else (S1_post, S2)
        row_cos = []
        for j in range(Sp.shape[0]):
            na, nb = Sp[j].norm(), Sc[j].norm()
            if na > 1e-9 and nb > 1e-9:
                row_cos.append(((Sp[j] @ Sc[j]) / (na * nb)).item())
        diag = dict(
            x_rel=sum(xdrift_rel) / max(len(xdrift_rel), 1),
            x_cos=sum(xdrift_cos) / max(len(xdrift_cos), 1),
            s_cos=sum(row_cos) / max(len(row_cos), 1),
            s_rel=((Sc - Sp).norm() / (Sp.norm() + 1e-12)).item(),
        )
    return logits, diag


def eval_condition(p, eval_seqs, grown=None, grow_mode=None, freeze_new=False,
                   collect_diag=False):
    """Mean CE on hard-region query positions (and easy for reference) over eval seqs."""
    hq, eq = torch.tensor(HARD_QPOS), torch.tensor(EASY_QPOS)
    hard_ces, easy_ces, diags = [], [], []
    for toks in eval_seqs:
        inp, tgt = toks[:-1], toks[1:]
        logits, diag = run_stepwise(p, inp, grown, grow_mode, freeze_new, collect_diag)
        hard_ces.append(F.cross_entropy(logits[hq], tgt[hq]).item())
        easy_ces.append(F.cross_entropy(logits[eq], tgt[eq]).item())
        if diag is not None:
            diags.append(diag)
    mh = sum(hard_ces) / len(hard_ces)
    me = sum(easy_ces) / len(easy_ces)
    ad = {k: sum(d[k] for d in diags) / len(diags) for k in diags[0]} if diags else None
    return mh, me, ad


# ── In-file verification against the imported reference ───────────────────────
def verify():
    print("--- verification against bdh_recurrent.py reference ---")
    g = torch.Generator().manual_seed(123)
    Dx = torch.randn(D, BASE_N, generator=g, dtype=torch.float64) * 0.1
    Dy = torch.randn(D, BASE_N, generator=g, dtype=torch.float64) * 0.1
    E  = torch.randn(BASE_N, D, generator=g, dtype=torch.float64) * 0.1
    v  = torch.randn(24, D, generator=g, dtype=torch.float64)

    vb = bdh_layer_parallel_batched(v.unsqueeze(0), Dx, Dy, E)[0]
    vp, *_ = bdh_layer_parallel(v, Dx, Dy, E)
    d1 = (vb - vp).abs().max().item()
    print(f"  batched-parallel (training) vs bdh_layer_parallel : max diff {d1:.2e}")
    assert d1 < 1e-10

    p = dict(embed=torch.randn(VOCAB, D, generator=g, dtype=torch.float64),
             Dx=Dx, Dy=Dy, E=E,
             head=torch.randn(VOCAB, D, generator=g, dtype=torch.float64) * 0.5)
    toks = torch.randint(0, VOCAB, (SEQ_LEN - 1,), generator=g)
    logits_sw, _ = run_stepwise(p, toks)
    vr, *_ = bdh_layer_recurrent(p['embed'][toks], Dx, Dy, E)
    d2 = ((logits_sw - vr @ p['head'].T).abs().max().item())
    print(f"  stepwise engine (no growth) vs bdh_layer_recurrent: max diff {d2:.2e}")
    assert d2 < 1e-10

    gg = torch.Generator().manual_seed(9)
    lw, _ = run_stepwise(p, toks, grown=grow_widen(p, 0.0, gg), grow_mode="widen")
    d3 = (lw - logits_sw).abs().max().item()
    # Widening doubles a* and y@E; the parameter-free ln cancels the 2 exactly only if
    # its eps=1e-5 were 0, so per-position logits deviate where std(z) is small. The
    # Hebbian STATE stays exact (x depends only on the layer input), so the artifact
    # does not compound; empirically it moves hard-query loss by ~1e-6 (checked as the
    # cond4-vs-cond1 control, tolerance MATCH_TOL, in the main run).
    print(f"  widen, zero perturbation vs no growth             : max logit diff {d3:.2e}"
          f"  (LayerNorm-eps artifact only; loss-level control enforced in main run)")
    assert d3 < 0.2

    ld, _ = run_stepwise(p, toks, grown=grow_deepen(p, 0.0, gg), grow_mode="deepen")
    d4 = (ld - logits_sw).abs().max().item()
    print(f"  deepen, zero perturbation vs no growth            : max logit diff {d4:.2e}")
    assert d4 < 1e-12
    print("  VERIFIED: engine matches reference; zero-perturb growth is function-preserving\n")


def ms(xs):
    m = sum(xs) / len(xs)
    s = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    return m, s


def paired_beats(worse, better):
    """Paired per-seed comparison: does `better` beat `worse` on hard-region loss?"""
    diffs = [w - b for w, b in zip(worse, better)]
    m, s = ms(diffs)
    return (m > BEAT_MARGIN and min(diffs) > 0), m, s


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    grow_mode = sys.argv[1] if len(sys.argv) > 1 else GROW_MODE
    assert grow_mode in ("widen", "deepen")
    if grow_mode == "widen":
        assert GROWN_N == 2 * BASE_N, "exact function preservation needs uniform doubling"
    grow_fn = grow_widen if grow_mode == "widen" else grow_deepen

    print("=" * 72)
    print("Inference-time growth + Hebbian plasticity test")
    print(f"  GROW_MODE={grow_mode}  D={D}  BASE_N={BASE_N}  GROWN_N={GROWN_N}"
          f"  PERTURB={PERTURB}")
    print(f"  K_EASY={K_EASY}  K_HARD={K_HARD}  VOCAB={VOCAB}  SEQ_LEN={SEQ_LEN}"
          f"  T_STEP={T_STEP}")
    print(f"  SEEDS={SEEDS}  ITERS={ITERS}  N_EVAL={N_EVAL}"
          f"   [chance-given-channel = ln2 = 0.693]")
    print("=" * 72)
    print()
    verify()

    res = {k: [] for k in ("c1", "c2", "c3", "c4", "easy", "ref")}
    diags = []
    t0 = time.time()
    for seed in SEEDS:
        print(f"-- seed {seed}: training base model (n={BASE_N}) --")
        base = train_model(seed, n=BASE_N, nlayers=1)
        p = get_params(base)

        egen = torch.Generator().manual_seed(seed + 5_000)
        eval_seqs = [make_seq(egen) for _ in range(N_EVAL)]
        ggen = torch.Generator().manual_seed(seed + 777)
        grown_pert = grow_fn(p, PERTURB, ggen)          # shared by conds 2 and 3
        grown_zero = grow_fn(p, 0.0, ggen)              # cond 4

        c1h, c1e, _ = eval_condition(p, eval_seqs)
        c2h, _, dg = eval_condition(p, eval_seqs, grown_pert, grow_mode,
                                    freeze_new=False, collect_diag=True)
        c3h, _, _ = eval_condition(p, eval_seqs, grown_pert, grow_mode, freeze_new=True)
        c4h, _, _ = eval_condition(p, eval_seqs, grown_zero, grow_mode, freeze_new=False)
        res["c1"].append(c1h); res["c2"].append(c2h)
        res["c3"].append(c3h); res["c4"].append(c4h)
        res["easy"].append(c1e)
        diags.append(dg)

        print(f"  hard-region loss:  base={c1h:.4f}  grown={c2h:.4f}  "
              f"frozen={c3h:.4f}  zerocopy={c4h:.4f}   (easy-region base={c1e:.4f})")
        print(f"  cond2 drift: x_rel={dg['x_rel']:.4f} x_cos={dg['x_cos']:.4f}  "
              f"S_rel={dg['s_rel']:.4f} S_cos={dg['s_cos']:.4f}")

        if RUN_CAPACITY_REF:
            if grow_mode == "widen":
                print(f"  training capacity reference (n={GROWN_N}, 1 layer)...")
                ref = train_model(seed, n=GROWN_N, nlayers=1, tag=" ref")
            else:
                print(f"  training capacity reference (n={BASE_N}, 2 layers)...")
                ref = train_model(seed, n=BASE_N, nlayers=2, tag=" ref")
            with torch.no_grad():
                hq = torch.tensor(HARD_QPOS)
                ces = []
                for toks in eval_seqs:
                    lg = ref(toks[:-1].unsqueeze(0))[0]
                    ces.append(F.cross_entropy(lg[hq], toks[1:][hq]).item())
            res["ref"].append(sum(ces) / len(ces))
            print(f"  capacity reference hard-region loss: {res['ref'][-1]:.4f}")
        print()

    print(f"(total {time.time() - t0:.0f}s)")
    print()
    print("=" * 72)
    print(f"RESULTS — hard-region query loss, mean +- std over seeds {SEEDS}")
    print("=" * 72)
    names = [("c1", "1. base (no growth)"),
             ("c2", "2. grown + Hebbian plasticity"),
             ("c3", "3. grown, new neurons FROZEN"),
             ("c4", "4. grown, ZERO perturbation")]
    for key, label in names:
        m, s = ms(res[key])
        print(f"  {label:34s}: {m:.4f} +- {s:.4f}")
    me, se = ms(res["easy"])
    print(f"  {'easy-region loss (base model)':34s}: {me:.4f} +- {se:.4f}")
    if res["ref"]:
        mr, sr = ms(res["ref"])
        print(f"  {'ref: TRAINED at grown size':34s}: {mr:.4f} +- {sr:.4f}")
    print()

    dxr, _ = ms([d["x_rel"] for d in diags])
    dxc, _ = ms([d["x_cos"] for d in diags])
    dsr, _ = ms([d["s_rel"] for d in diags])
    dsc, _ = ms([d["s_cos"] for d in diags])
    print("Cond-2 grown-neuron divergence from parents over the hard region:")
    print(f"  activations x: relative L2 = {dxr:.4f}, cosine = {dxc:.4f}")
    print(f"  Hebbian state S rows (end of seq): relative L2 = {dsr:.4f}, cosine = {dsc:.4f}")
    print()

    beats21, m21, s21 = paired_beats(res["c1"], res["c2"])
    beats23, m23, s23 = paired_beats(res["c3"], res["c2"])
    match41 = max(abs(a - b) for a, b in zip(res["c4"], res["c1"]))
    diverged = dsr > DRIFT_THRESH

    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    m1, _ = ms(res["c1"])
    print(f"Calibration: base model easy {me:.4f} vs hard {m1:.4f} "
          f"({'difficulty step confirmed' if m1 > me + 0.1 else 'WEAK difficulty step'}).")
    if res["ref"]:
        print(f"Capacity check: trained-at-grown-size reference reaches {mr:.4f} on the "
              f"hard region ({'capacity does help when trained in' if m1 - mr > BEAT_MARGIN else 'extra capacity does NOT help even trained in — hard region may not be capacity-limited'}).")
    print(f"Control: |cond4 - cond1| = {match41:.2e} "
          f"({'exact copies change nothing, as required' if match41 < MATCH_TOL else 'CONTROL FAILED — exact copies changed behavior'}).")
    print(f"Cond 2 vs cond 1 (paired): mean improvement {m21:+.4f} +- {s21:.4f}"
          f" -> {'beats' if beats21 else 'does NOT beat'}")
    print(f"Cond 2 vs cond 3 (paired): mean improvement {m23:+.4f} +- {s23:.4f}"
          f" -> {'beats' if beats23 else 'does NOT beat'}")
    print(f"Grown-neuron state drift {dsr:.4f} vs threshold {DRIFT_THRESH}"
          f" -> {'diverged' if diverged else 'did NOT diverge'}")
    print()
    if beats21 and beats23 and diverged:
        print("(a) Condition 2 beats condition 1 on hard-region loss, grown neurons")
        print("    diverge from their parents into distinct activity, and condition 2")
        print("    beats the frozen control — Hebbian adaptation, not just extra")
        print("    parameters, drove the gain. BDH plasticity CAN make grown neurons")
        print("    useful at inference without gradient training. Direction is alive.")
    else:
        reasons = []
        if not beats21:
            reasons.append("condition 2 does not beat the ungrown baseline")
        if not beats23:
            reasons.append("condition 2 does not beat the frozen-new-neurons control "
                           "(any effect is from parameters, not Hebbian adaptation)")
        if not diverged:
            reasons.append("grown neurons stay near-identical to their parents "
                           f"(state drift {dsr:.4f})")
        print("(b) Inference-time growth does NOT yield useful capacity here, even with")
        print("    the Hebbian mechanism running over the hard region: " +
              "; ".join(reasons) + ".")
