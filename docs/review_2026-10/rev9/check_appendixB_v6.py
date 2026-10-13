#!/usr/bin/env python
"""check_appendixB_v6.py -- Referee V, sixth pass: Appendix B ("What a replication needs") of the SIXTH Revision 9
draft (origin/claude/rev9-draft @ 42c4557, 23:14:04 UTC on 12 October, 568 lines) and the body sentences on it, against
the main line's code and records as of that time (origin/claude/bdh-growth-hebbian-inference-w90069 at 80bfc1d; the
Part C/D path is unchanged since db22857: since 8ae660d only test_split_copy.py's docstring changed) and R's branch as of
that time (550a736, the Phase 5 protocol).

A copy of check_appendixB_v5.py (fifth pass, draft c5e4508) re-anchored on the sixth draft:
  * every item's fragment is located in 42c4557 by text search (tex6:NNN; an assertion fails if it is missing); 42c4557
    replaces lines in place, so tex5 and tex6 line numbers agree; Appendix B (tex6:526-549) is unchanged;
  * R's UNSTATED.md at 550a736 has 21 Phase 5 entries (U-50 to U-70, added with the Phase 5 protocol at 22:47:46 UTC)
    after the 44 Phase 2 entries the appendix compiles: the label counts are recomputed on the Phase 2 section only (as
    in v5), and a new item records the file's total against the draft's "UNSTATED.md (44 entries)";
  * the 'none is needed' item's supporting count is the sixth draft's (126 of 126 key-position firings, tex6:441);
  * every fifth-pass item's fragment (read from check_appendixB_v5.py's own source with ast) is asserted in c5e4508 and
    searched in 42c4557;
  * the record checks are those of the earlier passes, unchanged.

Usage (from the repository root):
    python3 docs/review_2026-10/rev9/check_appendixB_v6.py [--code DIR] > docs/review_2026-10/rev9/check_appendixB_v6.md
--code DIR: an extracted copy of the main line's *.py (git archive 80bfc1d), run with the venv's python; read-only
builds of the Part C model at seed 540 and of CEIL_C, and one draw of the held-out batch with Generator(424242). No
training.
"""
import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rlib  # noqa: E402

OLD_REF, V2_REF, V3_REF, V4_REF, V5_REF, NEW_REF, DRAFT = ("9bfa85a", "f67bc07", "a5f67a2", "c66cc76", "c5e4508", "42c4557",
                                                           "multichannel_hebbian_report_v9.tex")
CUTOFF = rlib.git("show", "-s", "--format=%cI", NEW_REF).strip()          # the sixth draft's commit time
MAIN = "origin/claude/bdh-growth-hebbian-inference-w90069"
BR = rlib.git("rev-list", "-1", f"--before={CUTOFF}", MAIN).strip()[:7]    # the main line as of the sixth draft
R_BR = rlib.git("rev-list", "-1", f"--before={CUTOFF}", "origin/claude/replicate-R").strip()[:7]
FILES = dict(twg="test_window_gate.py", tmb="test_multilayer_binding.py", bdh="bdh.py", tbo="test_binding_onset.py",
             tbc="test_binding_capacity.py", trr="test_router_reliability.py", tsa="test_scale_axes.py",
             tsr="test_stream_recipe.py", tscur="test_stream_curriculum.py", tss="test_slow_start.py",
             tiv="test_instrument_v2.py", tcb="test_channel_binding.py", tsc="test_short_conv.py",
             tint="test_interference.py", tcl="test_conv_lr.py", tdr="test_delta_recipe.py", tsplit="test_split_copy.py",
             trc="test_router_confirm.py", trd="test_router_discovery.py")
SRC = {k: rlib.git("show", f"{BR}:{v}").split("\n") for k, v in FILES.items()}
tex1 = rlib.git("show", f"{OLD_REF}:{DRAFT}").split("\n")
tex2 = rlib.git("show", f"{V2_REF}:{DRAFT}").split("\n")
tex3 = rlib.git("show", f"{V3_REF}:{DRAFT}").split("\n")
tex4 = rlib.git("show", f"{V4_REF}:{DRAFT}").split("\n")
tex5 = rlib.git("show", f"{V5_REF}:{DRAFT}").split("\n")
tex6 = rlib.git("show", f"{NEW_REF}:{DRAFT}").split("\n")
HEAD = rlib.git("rev-parse", "--short", BR).strip()
PY_DIFF = rlib.git("diff", "--name-only", "db22857", BR, "--", "*.py").split()


def ev(abbr, snippet):
    """file:line of the first line holding snippet, at the main line's head."""
    for i, ln in enumerate(SRC[abbr]):
        if snippet in ln:
            return f"{abbr}:{i + 1}"
    raise AssertionError((abbr, snippet))


def locate(lines, phrases):
    for i, ln in enumerate(lines):
        if all(p in ln for p in phrases):
            return i + 1
    return None


# ── records ──────────────────────────────────────────────────────────────────
WG = {m: rlib.load(BR, f"results/{m}/window_gate_results.json") for m in "XL"}
SC = rlib.load("592c574", "results/X/split_copy_results.json")


def probe_vs_eval():
    """The first pass's helper, kept: every WIN3_SPLIT_D8 / WIN3_RESET_D8 check row against the curve at its step."""
    out = dict(rows=0, ref_rows=0, hi=[], reset_rows=0, reset_hi=0, sg_every_eval=True, fields=set(), runs=0)
    for m in "XL":
        for arm in ("WIN3_SPLIT_D8", "WIN3_RESET_D8"):
            runs = rlib.runs(WG[m], arm)
            for s, r in sorted(runs.items()):
                curve = {c[0]: c[1] for c in r["curve"]}
                st_steps = {x["step"] for x in r["stats"] if "stream_gate" in x}
                out["sg_every_eval"] &= set(curve) <= st_steps and "stream_gate" in r["end"]
                n_split, last = 0, None
                chk = r["checks"]
                for i, c in enumerate(chk):
                    capped = n_split >= 3 or (last is not None and c["step"] - last < 4800)
                    rise_ok = c["prev"] is not None and c["acc"] - c["prev"] < 0.02
                    out["fields"] |= set(c)
                    e = curve.get(c["step"])
                    if arm == "WIN3_SPLIT_D8":
                        out["rows"] += 1
                        out["ref_rows"] += c["step"] == 2400
                    else:
                        out["reset_rows"] += 1
                    if e is not None and c["acc"] >= 0.95 and e < 0.95:
                        if arm == "WIN3_RESET_D8":
                            out["reset_hi"] += 1
                        else:
                            nxt = chk[i + 1] if i + 1 < len(chk) else None
                            out["hi"].append(dict(m=m, s=s, step=c["step"], probe=c["acc"], ev=e, capped=capped,
                                                  rise_ok=rise_ok, alone=c["step"] >= 4800 and not capped and rise_ok,
                                                  next_fired=bool(nxt and nxt["fired"]), next_step=nxt and nxt["step"],
                                                  br=rlib.bound_routed(r), map=c["map_before"]))
                    if c["fired"]:
                        n_split, last = n_split + 1, c["step"]
            if arm == "WIN3_SPLIT_D8":
                out["runs"] += len(runs)
    return out


PV = probe_vs_eval()
ALONE = [h for h in PV["hi"] if h["alone"]]
EV_LO, EV_HI = min(h["ev"] for h in PV["hi"]), max(h["ev"] for h in PV["hi"])
starts = [WG["X"]["meta"]["started"], WG["L"]["meta"]["started"], SC["meta"]["started"]] + \
         [x["started"] for x in WG["X"]["meta"].get("starts", [])]
ZONE_FREE = all(len(x) == 19 and x[10] == " " and "+" not in x and "Z" not in x and "UTC" not in x for x in starts)
SC_SPLIT = rlib.runs(SC, "WIN3_SPLIT_D8")
SC_SG = all({c[0] for c in r["curve"]} <= {x["step"] for x in r["stats"] if "stream_gate" in x} for r in SC_SPLIT.values())
REC_FIELDS = {"acc", "prev", "step", "key_cs", "key_c0", "map_before", "fired"}
PART_CD = ("WIN3_SLOW_D8", "HINGE_D8", "WIN3_SPLIT_D8", "WIN3_RESET_D8", "CEIL_C")
CPU_GIT = all("cpu" in d["meta"] and "git" in d["meta"] for d in (WG["X"], WG["L"], SC)) and \
    all("cpu" in r for d in (WG["X"], WG["L"]) for k, r in d["runs"].items() if r.get("ok") and k.split("|")[0] in PART_CD) \
    and all("cpu" in r for r in SC["runs"].values() if r.get("ok"))
NO_CPU = sorted({k.split("|")[0] for d in (WG["X"], WG["L"]) for k, r in d["runs"].items() if r.get("ok") and "cpu" not in r})
NO_CPU_N = sum(1 for d in (WG["X"], WG["L"]) for k, r in d["runs"].items() if r.get("ok") and "cpu" not in r)


def r914():
    out = None
    for ln in rlib.git("show", f"{R_BR}:records/phase2.jsonl").split("\n"):
        if ln.strip():
            r = json.loads(ln)
            if r.get("seed") == 914 and r.get("arm") == "WIN3_SPLIT_D8":
                pr = [p for p in r["probes"] if p[0] >= 16800]
                evd = dict((e[0], e[1]) for e in r["evals"])
                evs = [evd[p[0]] for p in pr if p[0] in evd]
                top = max(pr, key=lambda p: p[1])
                out = (len(pr), min(p[1] for p in pr), max(p[1] for p in pr), min(evs), max(evs), r["failure"], top[0])
    return out


R914 = r914()
UNSTATED = rlib.git("show", f"{R_BR}:UNSTATED.md")
DIFFMD = rlib.git("show", f"{R_BR}:DIFF.md")
UNSTATED_ALL = UNSTATED
N_U_ALL = sum(1 for ln in UNSTATED_ALL.split("\n") if ln.startswith("| U-"))
P5_HDR = "# Phase 5"
UNSTATED = UNSTATED_ALL.split(P5_HDR)[0]          # the Phase 2 list (U-1 to U-44), the one the appendix compiles
P5_IDS = [ln.split("|")[1].strip() for ln in UNSTATED_ALL.split(P5_HDR)[1].split("\n") if ln.startswith("| U-")] if P5_HDR in UNSTATED_ALL else []
P5_COMMIT = rlib.git("log", "-1", "--format=%h %cI %s", R_BR, "--", "UNSTATED.md").strip()
N_U = sum(1 for ln in UNSTATED.split("\n") if ln.startswith("| U-"))
N_D = sum(1 for ln in DIFFMD.split("\n") if ln.startswith("| D-"))
matched = DIFFMD.split("## Points where my reading of the report matched theirs")[1].split("##")[0]
N_MATCHED = sum(1 for ln in matched.split("\n") if ln.startswith("- "))

# ── items ────────────────────────────────────────────────────────────────────
ITEMS = []   # (tex6 line, item, verdict, evidence, note, old item numbers)


def item(phrases, what, verdict, evid=(), note="", old=()):
    n = locate(tex6, phrases)
    assert n is not None, (what, phrases)
    ITEMS.append((n, what, verdict, ", ".join(ev(a, s) for a, s in evid), note, tuple(old)))


A = "\\paragraph{Model.}"
T = "\\paragraph{Task.}"
TR = "\\paragraph{Training and evaluation.}"
O = "\\paragraph{Outcomes.}"
SP = "\\paragraph{The split.} The probe: 64 sequences"
RC = "\\paragraph{Records.}"
STT = "\\paragraph{Statistics.} Exact one-sided McNemar"

# intro
item(("compiled from session~R's \\texttt{UNSTATED.md} (44 entries), its \\texttt{DIFF.md} (17)",),
     "sources: UNSTATED.md 44 entries, DIFF.md 17", "TRUE" if (N_U, N_D) == (44, 17) else "FALSE", (),
     f"R's branch: {N_U} U-rows, {N_D} D-rows.")
item(("compiled from session~R's \\texttt{UNSTATED.md} (44 entries), its \\texttt{DIFF.md} (17)",),
     "sources: UNSTATED.md '44 entries' against the file at R's tip as of the sixth draft (new)",
     "TRUE" if N_U_ALL == N_U == 44 else ("PARTIAL" if N_U == 44 else "FALSE"), (),
     f"UNSTATED.md at {R_BR}: {N_U_ALL} U-rows in all; the Phase 2 list (before '{P5_HDR}') {N_U}; the Phase 5 section "
     f"{len(P5_IDS)} ({P5_IDS[0] if P5_IDS else '-'} to {P5_IDS[-1] if P5_IDS else '-'}), last changed by '{P5_COMMIT[:110]}'. The appendix "
     "compiles the Phase 2 list, and its entries are those 44; but the file the sentence names has held 65 entries since R's "
     "Phase 5 protocol, 26 minutes before the draft. 'UNSTATED.md's Phase 2 list (44 entries)' would be exact. Wording only.")
# Model (tex6:530)
item((A, "in float32"), "float32", "TRUE",
     (("tmb", "assert self.freqs.dtype == torch.float32"), ("bdh", "torch.zeros((nh * N, D)).normal_(std=0.02)")),
     "No set_default_dtype or .double() on the path; every parameter is created at the default dtype. --code: every "
     "parameter of the built Part C model is float32. (Missing in the first draft: U-16.)")
item((A, "$x = \\mathrm{LN}(\\mathrm{emb})$"), "x = LN(emb)", "TRUE", (("tmb", "x = self.embed(idx).unsqueeze(1)"), ("tmb", "x = self.ln(x)")), "", (9,))
item((A, "the sparse code $\\sigma = \\mathrm{ReLU}(xE)$"), "sigma = ReLU(xE)", "TRUE",
     (("tmb", "x_latent = xc @ self.encoder"), ("tmb", "x_sparse = F.relu(x_latent)")),
     "With the width-4 conv, x here is conv(x) (stated later in the same paragraph).", (10,))
item((A, "$y = \\mathrm{LN}(\\mathrm{attn}(\\sigma, \\sigma, x))$"), "y = LN(attn(sigma, sigma, x))", "TRUE",
     (("tmb", "yKV = self.attn(Q=x_sparse, K=x_sparse, V=xc)"), ("tmb", "yKV = self.ln(yKV)")), "V = conv(x) with the conv.", (11,))
item((A, "$y_\\sigma = \\mathrm{ReLU}(yE_v)$"), "y_sigma = ReLU(y E_v)", "TRUE",
     (("tmb", "y_latent = yKV @ self.encoder_v"), ("tmb", "y_sparse = F.relu(y_latent)")), "", (12,))
item((A, "$z = \\mathrm{LN}((\\sigma \\odot y_\\sigma)D)$"), "z = LN((sigma * y_sigma) D)", "TRUE",
     (("tmb", "xy_sparse = x_sparse * y_sparse"), ("tmb", "@ self.decoder"), ("tmb", "y = self.ln(yMLP)")),
     "New in the second draft (first pass row 4: the layer stopped at y_s).", (4,))
item((A, "$x \\gets \\mathrm{LN}(x + z)$"), "x <- LN(x + z) (residual not convolved)", "TRUE",
     (("tmb", "x = self.ln(x + y)"), ("bdh", "x = self.ln(x + y)")), "New (row 4).", (4,))
item((A, "every LayerNorm without affine parameters"), "LayerNorm without affine", "TRUE",
     (("bdh", "self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)"),), "New (row 4).", (4,))
item((A, "logits $x W_{\\mathrm{head}}$ after the last layer"), "logits = x W_head after the last layer", "TRUE",
     (("tmb", "return x.view(B, T, D_) @ self.lm_head"),), "New (row 4).", (4,))
item((A, "one head, $N = 256$, $d = 32$"), "one head, N = 256, d = 32", "TRUE",
     (("tmb", "cfg = BDHConfig(n_layer=n_layer, n_embd=D, dropout=0.0, n_head=1,"), ("tcb", "SUB_MULT = 8"), ("tint", "D     = 32")),
     "--code: encoder (1, 32, 256).", (13, 14, 15))
item((A, "three applications of one shared layer"), "three applications of one shared layer", "TRUE",
     (("tcb", 'ARCH = dict(n_layer=3, positional="decay")'), ("tmb", "for level in range(C.n_layer):")), "", (16,))
item((A, "scalar decay $0.95^{t-u}$ in place of RoPE"), "scalar decay 0.95^(t-u) instead of RoPE", "TRUE",
     (("tiv", "OPERATING_DECAY = 0.95"), ("tmb", "scores = (Q @ K.mT) * decay_mask(T, self.decay, Q.dtype)")), "", (17,))
item((A, "strictly causal ($u < t$)"), "strictly causal u < t", "TRUE", (("tmb", "scores = scores.tril(diagonal=-1)"),), "", (18,))
item((A, "no dropout"), "no dropout", "TRUE", (("tmb", "dropout=0.0, n_head=1"),), "", (19,))
item((A, "Pathway's initialisation (encoder, decoder, head and embedding at $\\mathcal{N}(0, 0.02^2)$)"),
     "init N(0, 0.02^2) for encoder, decoder, head, embedding", "TRUE",
     (("bdh", "self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))"),
      ("bdh", "self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))"),
      ("bdh", "nn.init.normal_(module.weight, mean=0.0, std=0.02)")),
     "encoder_v (E_v, named in the forward pass) is also 0.02 but is not in the parenthesis; 'Pathway's initialisation' "
     "covers it (R's U-6 lists it).", (20,))
item((A, "which sets the scale of the gate's input"), "which sets the scale of the gate's input", "TRUE",
     (("twg", "gr, gw = self.local_gates(self.embed(tokens))"),), "", (21,))
item((A, "$(\\sigma_t \\cdot \\sigma_u)(g_t \\cdot g_u)\\,0.95^{t-u}$ on the sparse codes"),
     "score (sigma_t . sigma_u)(g_t . g_u) 0.95^(t-u) on the sparse codes", "TRUE",
     (("tmb", "scores = (Q @ K.mT) * decay_mask"), ("tmb", "scores = scores * self.G"),
      ("twg", 'self.attn.G = torch.einsum("btk,bsk->bts", gr, gw).unsqueeze(1)')),
     "Now on the sparse codes, with u for the earlier position (first pass row 5: the score was on 'x' and x_s was "
     "overloaded).", (5,))
item((A, "with one gate per token for read and write, the same at every layer"), "one gate per token, read = write, same at every layer",
     "TRUE", (("twg", "return g, g"), ("twg", 'self.attn.G = torch.einsum("btk,bsk->bts", gr, gw).unsqueeze(1)')),
     "G is set once per forward and read by every layer.", (22, 23))
item((A, "the gate reads the model's own raw embedding (the same table)"), "gate reads the raw embedding, same table", "TRUE",
     (("twg", "gr, gw = self.local_gates(self.embed(tokens))"),), "No LayerNorm on the gate input.", (24,))
item((A, "a width-3 depthwise causal convolution"), "width-3 depthwise causal conv", "TRUE",
     (("twg", "WIDTH, GCONV_OFFSET = 3, 31_337"), ("twg", "m.gate_conv = CausalConv(D, WIDTH)")), "", (25,))
item((A, "weights uniform in $\\pm 1/\\sqrt{3}$ from \\texttt{Generator(seed + 31337)}"), "window weights U(+-1/sqrt 3) from Generator(seed + 31337)",
     "TRUE", (("twg", "g = torch.Generator().manual_seed(int(seed) + GCONV_OFFSET)"),
              ("twg", "return (torch.rand(D, WIDTH, generator=g) * 2 - 1) * (1.0 / math.sqrt(WIDTH))")),
     "New (missing in the first draft). --code: the built window equals window_init(540). For bit-for-bit, the draw's "
     "layout is not stated: torch.rand(D, 3) with column j the lag-j tap (column 0 = the current token; R's D-12).", (26,))
item((A, "no bias, left-padded with zeros"), "no bias, left-padded with zeros", "TRUE",
     (("tmb", "self.conv_w = nn.Parameter(w)"), ("tmb", "xp = F.pad(x, (0, 0, K - 1, 0))")), "", (27, 28))
item((A, "a hidden layer of 32 units with $\\tanh$"), "hidden layer of 32 units, tanh", "TRUE",
     (("tiv", "H_GATE      = 32"), ("twg", "h = torch.tanh(u @ self.W_in.T)")), "", (29,))
item((A, "$W_{\\mathrm{in}}, W_g \\sim \\mathcal{N}(0, 0.1^2)$, no biases"), "W_in, W_g ~ N(0, 0.1^2), no biases", "TRUE",
     (("tmb", "self.W_in = nn.Parameter(torch.randn(hg, D) * 0.1)"), ("tmb", "self.W_g = nn.Parameter(torch.randn(n_ch, hg) * 0.1)"),
      ("twg", "g = F.softmax(h @ self.W_g.T, dim=-1)")), "", (30, 31))
item((A, "the model is built as the recurrent-gate model and converted"), "built as the recurrent-gate model, converted", "TRUE",
     (("twg", 'assert type(m) is MultiBDH and m.gate_kind == "recurrent"'), ("twg", "m.__class__ = LocalBDH")), "")
item((A, "its recurrent weight $W_h$ is drawn (which fixes the draw order of the parameters after it)"),
     "W_h is drawn, which fixes the draw order after it", "TRUE",
     (("tmb", "self.W_in = nn.Parameter(torch.randn(hg, D) * 0.1)"), ("tmb", "self.W_h = nn.Parameter(torch.randn(hg, hg) * 0.1)"),
      ("tmb", "self.W_g = nn.Parameter(torch.randn(n_ch, hg) * 0.1)")),
     "New (missing in the first draft; R's D-8, D-13). W_g is drawn from the global RNG after W_h. The order of "
     "Pathway's own draws (decoder, encoder, encoder_v, lm_head, embedding) is covered by 'Pathway's initialisation'.", (30,))
item((A, "stays frozen, unused and in no optimizer group"), "W_h frozen, unused, in no optimizer group", "TRUE",
     (("twg", "m.W_h.requires_grad_(False)"), ("twg", "named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]"),
      ("twg", "h = torch.tanh(u @ self.W_in.T)")),
     "--code: W_h requires_grad False; the SLOW groups are [W_in, W_g, gate_conv.conv_w] and the rest without W_h; "
     "local_gates never reads W_h.", (30, 83))
item((A, "one shared depthwise causal filter of width 4 on the $d$-dimensional layer input, initialised to the identity"),
     "conv: one shared width-4 depthwise causal filter on the d-dim layer input, identity init", "TRUE",
     (("tsc", "CONV_K = 4"), ("tmb", "xc = self.short_conv(x) if self.conv == \"layer\" else x"), ("tmb", "w[:, 0] = 1.0")),
     "", (32, 33, 34))
item((A, "whose output feeds the keys (equal to the queries, through $E$) and the values"), "conv output feeds K = Q (through E) and V",
     "TRUE", (("tmb", "yKV = self.attn(Q=x_sparse, K=x_sparse, V=xc)"),), "", (35,))
item((A, "the perfect gate's model is \\texttt{test\\_stream\\_recipe}'s \\texttt{ceiling8k16}"),
     "perfect gate's model: tsr ceiling8k16 (k = 16, stream s -> channel s, ctx list padded with 16 - S never-occurring ids), curriculum off",
     "TRUE", (("twg", 'CEIL_C_ARM = dict(tscur.ARM["SC8_ceil"], key="CEIL_C", cur=False)'),
              ("tscur", 'dict(tsr.ARM["ceiling8k16"], key="SC8_ceil", cur=True'),
              ("tsr", 'dict(CEIL, key="ceiling8k16", n_ch=16, ctx_pad=16, task="P4S8",'),
              ("tsr", "t.ctx_tokens = tuple(task.ctx_tokens) + tuple(-(i + 1) for i in range(k - task.S))")),
     "New (missing in the first draft; R's U-28/D-3). --code: CEIL_C builds a 16-channel perfect gate with ctx_tokens "
     "(0..7, -1..-8).")
# Task (tex6:532, unchanged from tex:519)
item((T, "28 tokens at eight streams and four keys"), "28 tokens; 16 values, P keys, S contexts, no query token", "TRUE",
     (("tbc", "self.vocab = S + P + n_vals"), ("tbc", "queries = torch.stack([qs, S + qk, S + P + qv], dim=-1)")), "", (36,))
item((T, "the sequence of 99 tokens ends with the query's value"), "99 tokens ending with the query's value; reads 98, predicts the 99th from the query's KEY",
     "TRUE", (("tbc", "self.L = 3 * (Z * A + n_q)"), ("tbc", "self.qpos = [3 * (Z * A + j) + 1 for j in range(n_q)]"),
              ("tbo", "inp, tgt = tokens[:, :-1], tokens[:, 1:]")), "", (37, 38))
item((T, "$S$ distinct values without replacement, assigned to streams in random order"), "S distinct values per key, random stream order",
     "TRUE", (("tbc", "vals = torch.rand(B, P, nv, generator=gen).argsort(-1)[..., :S]"), ("tbc", "sorder = torch.rand(B, P, S, generator=gen).argsort(-1)")),
     "", (39,))
item((T, "the query's stream and key uniform and independent"), "query stream and key uniform, independent", "TRUE",
     (("tbc", "q = torch.rand(B, S * P, generator=gen).argsort(-1)[:, :nq]"),), "One draw from the S*P pairs.", (40,))
item((T, "cross-entropy over the full vocabulary at the query's KEY position"), "CE over the full vocabulary at the query's KEY", "TRUE",
     (("tbo", "loss = F.cross_entropy(ql, qt)"), ("tbc", "ql = logits[:, self.qpos, :].reshape(-1, logits.shape[-1])")), "", (41,))
item((T, "accuracy is argmax over the full vocabulary"), "accuracy argmax over the full vocabulary", "TRUE",
     (("tbo", "acc = (ql.argmax(-1) == qt).float().mean().item()"),), "", (42,))
item((T, "2048 held-out sequences from a fixed generator (\\texttt{Generator(424242)}), the same for every run and arm"),
     "2048 held-out, Generator(424242) (re-anchored)", "TRUE",
     (("tbo", "EVAL_SEED = 424_242"), ("tbo", "g = torch.Generator().manual_seed(EVAL_SEED)"),
      ("tbo", "return task.make_batch(EVAL_QUERIES // task.n_q, g)[0]")),
     "Reworded in a5f67a2: the seed is now stated (R's D-7; the second pass's bit-for-bit omission). eval_batch draws "
     "with a fresh Generator seeded 424242 for every evaluation; 424_242 == 424242.", (43,))
item((T, "\\texttt{torch.manual\\_seed(seed)} before initialisation"), "torch.manual_seed(seed) before init", "TRUE",
     (("tbo", "torch.manual_seed(seed)"), ("tmb", "torch.manual_seed(seed)")), "", (44,))
item((T, "the training batches from \\texttt{Generator(seed + 10000)}"), "training batches from Generator(seed + 10000)", "TRUE",
     (("tbo", "rng = torch.Generator(); rng.manual_seed(seed + 10_000)"),), "", (45,))
# Training (tex6:534, unchanged)
item((TR, "Adam with default betas, no weight decay"), "Adam, default betas, no weight decay", "TRUE",
     (("tbo", "opt = torch.optim.Adam(param_groups(model), lr=lr)"), ("twg", "def factory(params, lr=1e-3, **kw):")), "", (46,))
item((TR, "batch 32"), "batch 32", "TRUE", (("tint", "BATCH = 32"), ("tbo", "tokens, _ = bt.make_batch(BATCH, rng)")), "", (47,))
item((TR, "constant rates, no warm-up, no clipping"), "constant rates, no warm-up, no clipping", "TRUE",
     (("tbo", "data=None, early_stop=True, lr=LR, warmup=0"),), "No clip_grad anywhere on the path; only SLOW's switch changes a rate.", (48, 49))
item((TR, "SLOW's groups and its switch after the evaluation at 2400"), "SLOW groups; switch after the evaluation at 2400", "TRUE",
     (("twg", 'holder["groups"] = [dict(params=[nm[n] for n in gate], lr=LR), dict(params=[nm[n] for n in rest], lr=LR_WARM)]'),
      ("twg", 'holder["groups"][1]["lr"] = LR'), ("tss", "WARM = 2400")), "", (50, 84))
item((TR, "evaluation every 1200 updates and at the budget"), "evaluation every 1200 and at the budget", "TRUE",
     (("tbo", "EVAL_EVERY = 1200"), ("tbo", "if data is not None and step % eval_every == 0:")), "43200 = 36 x 1200.", (51,))
item((TR, "early stop after three consecutive evaluations"), "early stop after three evaluations >= 0.95", "TRUE",
     (("tbo", "STOP_AFTER = 3"), ("tbo", "if early_stop and streak >= STOP_AFTER:")), "", (52,))
item((TR, "evaluation, statistics, the path's own check, the learning-rate switch, the split trigger, the early-stop test"),
     "order at a shared step", "TRUE",
     (("tbo", "on_eval(model, step)"), ("trr", "stats.append(stats_fn(m, task, probe, step))"), ("twg", "c0(m, step, data)"),
      ("twg", "trig(m, step)")), "", (53,))
item((TR, "the perfect gate trains at the single rate $10^{-3}$ without SLOW"), "perfect gate at 1e-3, no SLOW", "TRUE",
     (("twg", "return tscur.run_sc(CEIL_C_ARM, seed, dict(tscur.REAL, total=iters), recipe=rc)"), ("tcb", "SUB_LR = 1e-3")),
     "rc is None for the perfect gate (one_run).", (54,))
item((TR, "the final per-stream accuracy and map come from a final evaluation of the final model"), "final per-stream accuracy and map from a final evaluation",
     "TRUE", (("trr", 'end = stats_fn(m, task, probe, "end")'), ("tsa", 'out["stream_acc"] = stream_acc(model, task, eval_batch(task))')), "", (55,))
# Outcomes (tex6:536, changed)
item((O, "Bound: the first evaluation $\\geq 0.95$ that holds to the end (a run can bind on its last evaluation)"), "bound; can bind on the last evaluation",
     "TRUE", (("tbo", "def transition(curve):"),), "", (56, 87))
item((O, "the argmax over channels of the mean gate at the body's value positions on a 256-sequence probe (\\texttt{Generator(seed + 99000)})"),
     "map: argmax of the mean gate at body value positions, 256-sequence probe Generator(seed + 99000)", "TRUE",
     (("tsa", "means = [gv[ctx == s].mean(0) for s in range(S)]"), ("tmb", "prng = torch.Generator(); prng.manual_seed(seed + 99_000)"),
      ("tiv", "PROBE_B     = 256")), "", (57, 58, 86))
item((O, "ties on unrounded values to the lowest channel index (\\texttt{torch.argmax})"), "map ties: unrounded, lowest index (torch.argmax)",
     "TRUE", (("tsa", "cmap = [int(m_.argmax()) for m_ in means]"),),
     "New (the first draft said only 'ties on unrounded values'). torch.argmax returns the first maximal index (--code "
     "checks it on a tie).", (59,))
item((O, "bound routed: bound, the map one-to-one, every stream's accuracy $\\geq 0.9$ on the held-out queries of that stream"),
     "bound routed = bound, one-to-one, every stream >= 0.9 on its held-out queries", "TRUE",
     (("tsr", 'return bool(e["one_to_one"] and min(e["stream_acc"]) >= ROUTED_ACC)'), ("tsa", "return [hit[qs == s].mean().item() for s in range(task.S)]")),
     "", (60,))
item((O, "failure labels as Section"), "failure labels as Section 3", "TRUE", (("tsr", "def outcome(r):"),), "")
# The split (tex6:538, changed)
item((SP, "\\texttt{BindTask.make\\_batch(64, Generator(12345))}, the same for every seed and arm"), "probe: 64 sequences, Generator(12345), same for every seed and arm",
     "TRUE", (("twg", "PROBE_N, PROBE_SEED = 64, 12345"), ("twg", "return task.make_batch(PROBE_N, torch.Generator().manual_seed(PROBE_SEED))[0]")),
     "", (61,))
item((SP, "scored on their queries (5, 5, 6, 8, 10, 12, 5 and 13 per stream at eight streams)"), "scored on its queries; 5,5,6,8,10,12,5,13",
     "TRUE", (("twg", "acc, _ = tbo.evaluate(m, task, data)"), ("tdr", "PROBE_COUNTS = (5, 5, 6, 8, 10, 12, 5, 13)")),
     "--code recomputes the counts.", (62, 63))
item((SP, "read at 2400 as the first reference and at every check from 4800 in steps of 2400 up to the budget minus 2400"),
     "reference at 2400; checks 4800..budget-2400 every 2400", "TRUE",
     (("twg", "EVERY, FIRST, REF_AT = 2400, 4800, 2400"), ("twg", "if step % EVERY or step < REF_AT or step > last:")), "", (64,))
item((SP, "fire if the probe's accuracy is below $0.95$ and rose by less than $0.02$ since the previous check"), "fire rule", "TRUE",
     (("twg", "eligible = step >= FIRST and st[\"prev\"] is not None and acc < thr and acc - st[\"prev\"] < rise"),
      ("twg", 'st["prev"] = acc')), "prev is updated at every check, blocked ones included.", (65,))
item((SP, "$c^*$ and $c_0$ the argmax and argmin of the mean read-gate mass over the probe's 32 body key positions (not the query's), ties to the lowest index"),
     "c*, c0: argmax/argmin of key mass over the 32 body keys; ties to the lowest index", "TRUE",
     (("twg", "j = torch.arange(S * P)"), ("twg", "km = gr[:, 3 * j + 1].mean((0, 1))"),
      ("twg", "return (int(km.argmax()), int(km.argmin()), km), (int(am.argmax()), int(am.argmin()), am)")),
     "Tie rule new (missing in the first draft; R's U-38). torch.argmax/argmin return the first index (--code).", (66,))
item((SP, "$W_g[c_0] \\gets W_g[c^*]$"), "W_g[c0] <- W_g[c*]", "TRUE", (("twg", "W[c0] = w + n2"),), "", (67,))
item((SP, "adds $0.1\\,\\sigma$ noise to both rows, $\\sigma$ the copied row's standard deviation by \\texttt{torch.std} with Bessel's correction"),
     "noise 0.1 sigma to both rows; sigma = copied row's torch.std (Bessel)", "TRUE",
     (("twg", "NOISE_REL, NOISE_BASE = 0.1, 12_000_000"), ("twg", "sd = NOISE_REL * w.std()"), ("twg", "w = W[cs].clone()")),
     "Estimator new (missing in the first draft; R's U-39). Tensor.std() defaults to the unbiased estimator (--code).", (68,))
item((SP, "two draws of the row's shape from \\texttt{Generator(12\\_000\\_000 + 1000*seed + n\\_splits)}, the first added to $c^*$'s row and the second to $c_0$'s"),
     "two draws of the row's shape from Generator(12_000_000 + 1000 seed + n), first to c*, second to c0", "TRUE",
     (("twg", "n1 = torch.randn(w.shape, generator=g, dtype=w.dtype) * sd                   # row c*"),
      ("twg", "n2 = torch.randn(w.shape, generator=g, dtype=w.dtype) * sd                   # row c0"),
      ("twg", "NOISE_BASE + 1000 * seed + st[\"n\"]"), ("twg", "W[cs] = w + n1")), "", (69,))
item((SP, "zeroes $W_g$'s Adam state including its step counter"), "zeroes W_g's Adam state incl. step", "TRUE",
     (("twg", "for k, v in st.items():"), ("twg", "zeroed = zero_state(o, W)")),
     "Records: every fired row has state_zeroed = [exp_avg, exp_avg_sq, step] (first pass).", (70,))
item((SP, "a split blocks further splits for 4800 updates; at most three"), "gap 4800, cap 3", "TRUE",
     (("twg", "MAX_SPLITS, GAP = 3, 4800"), ("twg", "capped = st[\"n\"] >= max_splits or (st[\"last\"] is not None and step - st[\"last\"] < gap)")),
     "", (71, 72))
item((SP, "no guard against a target that carries key mass"), "no guard", "TRUE", (("twg", "elif eligible:"),), "", (73,))
item((SP, "(none is needed at $k = 16$, $S = 8$)"), "'none is needed at k = 16, S = 8'", "UNVERIFIABLE", (),
     "A claim about need, not code. Supporting: 126 of 126 key-position firings on labelled targets in the 60 recorded split runs "
     "(tex6:441; check_split_copy_v6 recomputes it with split_copy L added), and none on a map already one-to-one.", (6,))
# Records (tex6:540, changed)
rf_ok = REC_FIELDS <= PV["fields"]
item((RC, "the curve at every evaluation"), "records: curve at every evaluation", "TRUE", (("tbo", "curve.append([step, acc, el])"),), "", (74,))
item((RC, "the per-stream gate mass at value positions at every evaluation"), "records: per-stream gate mass at value positions at every evaluation",
     "TRUE" if PV["sg_every_eval"] and SC_SG else "FALSE",
     (("trr", "stats.append(stats_fn(m, task, probe, step))"), ("tsa", 'out["stream_gate"] = [[round(v, 4) for v in m_.tolist()] for m_ in means]')),
     f"window_gate X/L split and reset runs: stats[*].stream_gate at every curve step and at the end: {PV['sg_every_eval']}; "
     f"split_copy split runs: {SC_SG}. 'at every evaluation' is new (first draft: 'at the end').", (75,))
item((RC, "the per-stream accuracy at the end"), "records: per-stream accuracy at the end", "TRUE",
     (("tsa", 'out["stream_acc"] = stream_acc(model, task, eval_batch(task))'),), "", (75,))
item((RC, "each check's probe reading beside the evaluation's reading at the same step"), "records: probe reading beside the evaluation",
     "TRUE" if rf_ok else "FALSE", (("twg", "row = dict(step=step, acc=acc, prev=st[\"prev\"], map_before=cmap,"), ("twg", 'info["checks"].append(row)')),
     f"check-row fields {sorted(PV['fields'])}", (76,))
item((RC, "each split's update, $c^*$, $c_0$ and the map before it"), "records: split update, c*, c0, map before", "TRUE" if rf_ok else "FALSE",
     (("twg", "key_cs=kc, key_c0=k0, all_cs=ac, all_c0=a0,"),), "", (77,))
item((RC, "the CPU, the code commit"), "records: CPU and code commit", "TRUE" if CPU_GIT else "FALSE",
     (("twg", 'rec.update(arm=a["key"], cpu=cpu_model())'), ("twg", "meta = dict(machine=args.machine, torch=torch.__version__, cpu=cpu_model(), workers=args.workers, git=head,")),
     f"Every Part C/D record (window_gate X and L) and every split_copy record carries its CPU; every file's meta its CPU "
     f"and commit. Outside Part C/D, {NO_CPU_N} window_gate records ({NO_CPU}, Part B, built by test_early_recipe's path) "
     "carry no per-record CPU; the file's meta.cpu covers them.", (78, 79))
item((RC, "The main line's records hold all of these except the zone (its start times are local to the machine, without one)"),
     "the main line's records hold all but the zone; start times local, no zone", "TRUE" if ZONE_FREE and rf_ok and CPU_GIT else "FALSE",
     (("twg", 'now = time.strftime("%Y-%m-%d %H:%M:%S")'), ("tsplit", 'now = time.strftime("%Y-%m-%d %H:%M:%S")')),
     f"time.strftime formats local time with no zone field; recorded start strings {starts}. The time zone is now a "
     "prescription ('a replication should record it'), and the main line is said not to meet it (first pass row 1 FALSE).",
     (1,))
# Statistics (tex6:542, unchanged)
item((STT, "$p = P[\\mathrm{Bin}(b + c, \\tfrac12) \\geq b]$"), "exact one-sided McNemar", "TRUE",
     (("tcl", "def mcnemar"),), "", (80,))
item((STT, "Wilson 95\\% intervals with $z = 1.96$"), "Wilson 95%, z = 1.96", "TRUE", (("tsr", "def wilson(k, n, z=1.959963984540054):"),), "", (81,))
item((STT, "a one-sided exact test against the reported rate with its size and power stated in advance"),
     "replication prescription (exact test vs reported rate, Fisher, paired arms)", "UNVERIFIABLE", (),
     "A prescription, not a description of the code.", (7,))
# body items
item(("$W_{\\mathrm{in}}, W_h, W_g \\sim \\mathcal{N}(0, 0.1^2)$", "$\\rho(W_h) \\approx 0.6$"), "tex:87 recurrent gate init; rho ~ 0.6, sigma_max ~ 1.08",
     "TRUE", (("tmb", "self.W_h = nn.Parameter(torch.randn(hg, hg) * 0.1)"),),
     "Unchanged; first pass measured rho 0.606, sigma_max 1.083 over seeds 540-559.", (82,))
item(("a depthwise causal convolution of width 3 over the embeddings (weights drawn uniformly in $\\pm 1/\\sqrt{3}$, no bias), with no recurrence",),
     "tex:92 window gate", "TRUE", (("twg", "u = self.gate_conv(v)"),), "Unchanged.", (83,))
item(("A run that reaches its budget counts as bound on a single final evaluation",), "tex:112 bound on a single final evaluation", "TRUE",
     (("tbo", "def transition(curve):"),), "Unchanged; the '14 of 1869' count not checked here.", (87,))
item(("The map takes each stream's largest-mass channel, from a 256-sequence probe, ties resolved on unrounded values",),
     "tex:114 map from a 256-sequence probe, ties on unrounded values", "TRUE", (("tsa", "cmap = [int(m_.argmax()) for m_ in means]"),),
     "Sentence unchanged (the item's line was edited elsewhere).", (86,))
item(("The gate's parameters train at $10^{-3}$ throughout; every other parameter, embedding and convolution included",),
     "tex:120 SLOW", "TRUE", (("twg", 'GATE_L = ("W_in", "W_g", "gate_conv.conv_w")'),), "Unchanged.", (84,))
item(("copy the gate row $W_g[c^*]$ of the channel with the largest mean read-gate mass at the body's key positions",),
     "tex:122 SPLIT description", "TRUE", (("twg", "km = gr[:, 3 * j + 1].mean((0, 1))"),), "Unchanged.", (85,))
item(("session~H's \\emph{exact copy} does neither",), "tex:122 'session H's exact copy does neither'", "UNVERIFIABLE", (),
     "Unchanged; H's code is on another branch, outside this scope.", (8,))
# tex6:207 and tex6:395 (first pass rows 2-3)
item(("The main line's records store the probe's reading at every check beside the evaluation's curve",),
     "tex6:207 records store the probe's reading beside the curve", "TRUE" if rf_ok else "FALSE",
     (("twg", "row = dict(step=step, acc=acc, prev=st[\"prev\"], map_before=cmap,"),),
     "Reverses the first draft's false 'store neither ...' (first pass row 2).", (2,))
hi_txt = "; ".join(f"{h['m']} s{h['s']}@{h['step']}: probe {h['probe']:.3f} eval {h['ev']:.4f}" for h in PV["hi"])
HI_CHECKS = all(h["step"] >= 4800 for h in PV["hi"])
N_CHECKS = PV["rows"] - PV["ref_rows"]
item(("6 of the 353 checks in the split arm's 40 runs read the probe at or above $0.95$ with the evaluation at $0.865$--$0.88$",),
     "tex6:207 '6 of the 353 checks ... evaluation at 0.865-0.88' (re-anchored)",
     "TRUE" if (len(PV["hi"]) == 6 and N_CHECKS == 353 and HI_CHECKS and PV["runs"] == 40
                and f"{EV_LO:.3f}" == "0.865" and f"{EV_HI:.2f}" == "0.88") else "FALSE",
     (("twg", "eligible = step >= FIRST and st[\"prev\"] is not None and acc < thr and acc - st[\"prev\"] < rise"),),
     f"{len(PV['hi'])} of {PV['rows']} probe readings in {PV['runs']} runs ({hi_txt}); {PV['ref_rows']} of the readings are "
     f"the 2400 references, so {N_CHECKS} checks (from 4800); all six at a check step (>= 4800): {HI_CHECKS}; evaluation "
     f"{EV_LO:.4f}-{EV_HI:.4f}. Reworded in a5f67a2 from '393' (the second pass's wording point, from the referee's own A1).",
     (2,))
item(("in 2 of those the probe alone blocked a split, and both fired at the next check and bound routed; the reset arm had none",),
     "tex6:207 '2 blocked by the probe alone, both fired next and bound routed; reset arm none'",
     "TRUE" if (len(ALONE) == 2 and all(h["next_fired"] and h["br"] for h in ALONE) and PV["reset_hi"] == 0) else "FALSE",
     (), "blocked by the probe alone (eligible but for the probe: not gap/cap-blocked, rise < 0.02): "
     + "; ".join(f"{h['m']} s{h['s']}@{h['step']} (next check {h['next_step']} fired {h['next_fired']}; run bound routed {h['br']})" for h in ALONE)
     + f"; the other {len(PV['hi']) - len(ALONE)}: gap/cap or rise >= 0.02. WIN3_RESET_D8: {PV['reset_hi']} of {PV['reset_rows']} rows.", (2,))
item(("5, 5, 6, 8, 10, 12, 5 and 13 queries for streams 0--7", "identical for every seed"), "tex6:207 probe composition, same for every seed",
     "TRUE", (("twg", "PROBE_N, PROBE_SEED = 64, 12345"),), "--code recomputes the counts.", ())
item(("costs the probe about $0.08$ where the balanced evaluation loses $0.125$",), "tex6:207 0.08 vs 0.125", "TRUE", (),
     f"Two 5-query streams halved: (5 + 5)/2/64 = {10 / 2 / 64:.4f}; balanced: (2/8)/2 = 0.125.", ())
item(("A 64-query probe near $0.9$ has a standard error of about $0.04$",), "tex6:207 SE ~ 0.04", "TRUE", (),
     f"sqrt(0.9 * 0.1 / 64) = {(0.9 * 0.1 / 64) ** 0.5:.4f}.", ())
item(("on its seed 914 the 64-query probe read $0.953$--$1.000$ at every check from 16{,}800 on while the 2048-query evaluation sat at $0.871$--$0.887$",),
     "tex6:207 R's seed 914 (re-anchored)", "TRUE" if R914 and (f"{R914[1]:.3f}", f"{R914[2]:.3f}") == ("0.953", "1.000") and
     (f"{R914[3]:.3f}", f"{R914[4]:.3f}") == ("0.871", "0.887") else "FALSE", (),
     f"R records/phase2.jsonl at {R_BR} (WIN3_SPLIT_D8, 914): {R914[0]} probe readings from 16800, probe "
     f"{R914[1]:.4f}-{R914[2]:.4f} (the maximum at update {R914[6]}), evaluation at those steps {R914[3]:.4f}-{R914[4]:.4f}; "
     f"failure {R914[5]}. Reworded in a5f67a2 from 0.984 (the second pass's FALSE item).", ())
item(("The next main-line test prints every check's probe reading beside the held-out accuracy",),
     "tex6:207 the next test prints probe vs held-out", "TRUE",
     (("tdr", "def probe_blind(t, k):"), ("tdr", "print(f\"  probe-blind checks, {k} (checks at update >= {tgw.FIRST}): {blind} of {len(rows_pb)} read >= {tgw.ACC_THR} on the \"")),
     "test_delta_recipe.py at the main line as of the sixth draft (the X run used a581027; 2225f00 changed the print, as Amendment 2 says). It has now printed it on X (RESULT (X): 'Probe-blind (checks from 4800)').", ())
item(("Both could have been read from the project's own records, which store the probe's reading at every check beside the evaluation's",),
     "tex6:395 records store the probe beside the evaluation", "TRUE" if rf_ok else "FALSE",
     (("twg", 'info["checks"].append(row)'),), "Reverses the first draft's false tex:384 (first pass row 3).", (3,))
item(("nobody read the two against each other until the referee did (6 of 353 checks;",), "tex6:395 'nobody read them until the referee did' (re-anchored)",
     "UNVERIFIABLE", (), f"Not a code question. '6 of 353 checks' is now right ({len(PV['hi'])} of {N_CHECKS}; see the tex6:207 row).", (3,))
# tex6:432 (rewritten in a5f67a2)
TDR_TRIG = any("tgw.make_trigger" in ln or "make_trigger(" in ln for ln in SRC["tdr"])
item(("The records keep every check's probe reading beside the evaluation's",), "tex6:432 records keep the probe beside the evaluation",
     "TRUE" if rf_ok else "FALSE", (("twg", "row = dict(step=step, acc=acc, prev=st[\"prev\"], map_before=cmap,"),
                                     ("tbo", "curve.append([step, acc, el])")),
     "Rewritten in a5f67a2 (the second pass's item 3: tex2:427 said the records do not keep the readings).", ())
item(("A test should print them side by side, with the count of checks at which the probe read $\\geq 0.95$ while the held-out accuracy was below it, as \\texttt{test\\_delta\\_recipe} prints",),
     "tex6:432 test_delta_recipe prints them side by side, with the count of checks where probe >= 0.95 and held-out < 0.95 (re-anchored)", "TRUE",
     (("tdr", "def probe_blind(t, k):"), ("tdr", "for c in r.get(\"checks\", []):"),
      ("tdr", "if h is not None and c[\"acc\"] >= tgw.ACC_THR and h < tgw.ACC_THR:"),
      ("twg", 'info["checks"].append(row)'), ("tdr", "if c[\"step\"] >= tgw.FIRST:")),
     "Wording unchanged since c66cc76. The count is as the test computes it (probe >= 0.95 while held-out < 0.95). The "
     "fourth pass's residual (the printed denominator included the 2400 reference rows) is gone: since Amendment 2 "
     f"(55dac50, 2225f00) probe_blind counts checks at update >= FIRST = 4800 (test uses test_window_gate's trigger: {TDR_TRIG}).", ())
item(("weights the streams unequally, in the same way for every seed",), "tex6:395 unbalanced, same for every seed", "TRUE",
     (("twg", "return task.make_batch(PROBE_N, torch.Generator().manual_seed(PROBE_SEED))[0]"),), "", ())
item(("the main line's records store neither the probe's reading nor the per-stream masses at each check (they store both;",),
     "tex6:349 the referee's finding restated: records store both", "TRUE" if rf_ok and PV["sg_every_eval"] else "FALSE",
     (("twg", 'info["checks"].append(row)'), ("trr", "stats.append(stats_fn(m, task, probe, step))")),
     "Probe reading in checks[*].acc; per-stream masses in stats[*].stream_gate at every evaluation, which includes every "
     "check step; per-channel key and all-position masses in the check row.", (2,))
item(("among them that the records do not store the probe's reading",), "tex6:387 the first draft's false statement listed", "TRUE",
     (), "Listed as false; consistent with the records (rows above).", (3,))
item(("Appendix~\\ref{app:repro} gained the six entries the check found missing",), "tex6:349 'gained the six entries'", "TRUE", (),
     "See the tex6:332 row.")
# tex6:332 and tex6:36 on the appendix
item(("Appendix~\\ref{app:repro} states them, with the entries the referee found missing from this draft's first version",),
     "tex6:332 the six missing entries now stated", "TRUE", (),
     "window generator (tex6:530), frozen W_h (tex6:530), dtype (tex6:530), perfect gate's model (tex6:530), tie rules "
     "(tex6:536, 538), noise estimator (tex6:538): all present and true (rows above).")
item(("R lists eleven guesses that matched our choices",), "tex6:332 'eleven guesses that matched'", "TRUE" if N_MATCHED == 11 else "FALSE",
     (), f"DIFF.md 'Points where my reading ... matched theirs': {N_MATCHED} bullets. Replaces 'Nine of the 44 guesses were "
     "material and all nine matched'.")
D10 = [ln for ln in DIFFMD.split("\n") if ln.startswith("| D-10 |")]
item(("is filed by R as a difference (D-10) that matched in statistic only, since R measured it on the 2048 held-out sequences and the main line on a 256-sequence probe",),
     "tex6:332 the map: R's D-10, matched in statistic only (re-anchored)",
     "TRUE" if D10 and "256-sequence probe" in D10[0] and "2048 held-out" in D10[0] else "FALSE",
     (("tmb", "prng = torch.Generator(); prng.manual_seed(seed + 99_000)"),),
     "Reworded in a5f67a2: no longer listed among the eleven matched guesses (the second pass's point). DIFF.md: "
     + (D10[0][:160] if D10 else "no D-10 row"))
LAB = {}
for ln in UNSTATED.split("\n"):
    if ln.startswith("| U-"):
        what = ln.split("|")[2]
        lab = re.findall(r"\[(A|B|C|stated|stated \+ A|A in code, stated in report)\]", what)
        LAB[ln.split("|")[1].strip()] = lab[0] if lab else "unlabelled"
from collections import Counter  # noqa: E402
LC = Counter(LAB.values())
n_stated = LC["stated"] + LC["stated + A"] + LC["A in code, stated in report"]
n_abc = LC["A"] + LC["B"] + LC["C"]
UNL = sorted((k for k, v in LAB.items() if v == "unlabelled"), key=lambda x: int(x[2:]))
PROC = [k for k in UNL]
NAMED = {k: LAB.get(k) for k in ("U-6", "U-11", "U-16", "U-23")}
item(("the text stated four outright and three only in part (the gate's input among the four)",),
     "tex6:332 R's labels: four stated outright, three in part, the gate's input among the four (re-anchored)",
     "TRUE" if (LC["stated"], LC["stated + A"] + LC["A in code, stated in report"]) == (4, 3) and LAB.get("U-9") == "stated" else "FALSE", (),
     f"UNSTATED.md: [stated] {LC['stated']}, [stated + A] {LC['stated + A']}, [A in code, stated in report] "
     f"{LC['A in code, stated in report']}; the gate's input is U-9 [{LAB.get('U-9')}]. Reworded in c66cc76 from 'seven outright or in part'.")
item(("left 34 to be guessed (the initialisation scales, the parameter dtype, the evaluation's sampling and the trigger's details among them): 20 on which it was silent, 12 on which it was ambiguous and 2 carried over from what the report states for another component, with 3 further entries of process",),
     "tex6:332 R's labels: 34 guessed = 20 + 12 + 2 ([C] as R defines it), 3 process (re-anchored)",
     "TRUE" if (n_abc, LC["A"], LC["B"], LC["C"], len(UNL)) == (34, 20, 12, 2, 3) else "FALSE", (),
     f"[A] {LC['A']}, [B] {LC['B']}, [C] {LC['C']} = {n_abc}; unlabelled (process) {len(UNL)}: {', '.join(UNL)}; "
     f"4 + 3 + {n_abc} + {len(UNL)} = {4 + 3 + n_abc + len(UNL)}. The third pass's PARTIAL (37 against 20 + 12 + 2; [C] "
     "misdescribed) is resolved. Re-anchored on c5e4508 ('3 further entries of process').")
NAMED = {k: LAB.get(k) for k in ("U-6", "U-11", "U-16", "U-22", "U-36", "U-32", "U-37", "U-38", "U-39", "U-41")}
GUESSED = {"A", "B", "C"}
item(("left 34 to be guessed (the initialisation scales, the parameter dtype, the evaluation's sampling and the trigger's details among them):",),
     "tex6:332 the parenthesis, now after '34 to be guessed' (re-anchored)", "TRUE" if all(v in GUESSED for v in NAMED.values()) else "PARTIAL", (),
     f"Moved in fabf505 on the fourth pass's item 9. Every entry it names is a guess: the initialisation scales U-6 "
     f"[{NAMED['U-6']}], U-11 [{NAMED['U-11']}]; the dtype U-16 [{NAMED['U-16']}]; the evaluation's sampling U-22 "
     f"[{NAMED['U-22']}], U-32 [{NAMED['U-32']}], U-36 [{NAMED['U-36']}]; the trigger's details U-37 [{NAMED['U-37']}], "
     f"U-38 [{NAMED['U-38']}], U-39 [{NAMED['U-39']}], U-41 [{NAMED['U-41']}]. The process entries ({', '.join(UNL)}) follow "
     "as '3 further entries of process'.")
item(("the 44 details it logged (34 of them guesses) are stated in an appendix",),
     "tex6:36 'the 44 details it logged (34 of them guesses) are stated in an appendix' (re-anchored)", "TRUE", (),
     "Reworded in c66cc76; it now agrees with tex6:332 (34 guesses of 44). Every U-entry's main-line choice is given, "
     "U-22's held-out generator included (tex6:532, Generator(424242)). Bit for bit, the window's draw layout "
     "(torch.rand(D, 3), column 0 = current token; D-12) is still not stated (an omission, section 5). R's UNSTATED.md has since "
     f"grown by {len(P5_IDS)} Phase 5 entries ({R_BR}); the sentence is about the replication's first phases, so the 44 stand.")

# ── fifth-pass items (read from check_appendixB_v5.py's source) ────────────
import ast  # noqa: E402

V2_ITEMS = []   # here: the FIFTH pass's items (check_appendixB_v5.py), located in the sixth draft
for node in ast.walk(ast.parse(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_appendixB_v5.py")).read())):
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "item":
        a0 = node.args[0]
        consts = {"A": "\\paragraph{Model.}", "T": "\\paragraph{Task.}", "TR": "\\paragraph{Training and evaluation.}",
                  "O": "\\paragraph{Outcomes.}", "SP": "\\paragraph{The split.} The probe: 64 sequences",
                  "RC": "\\paragraph{Records.}", "STT": "\\paragraph{Statistics.} Exact one-sided McNemar"}
        ph = tuple(consts[e.id] if isinstance(e, ast.Name) else ast.literal_eval(e) for e in a0.elts)
        V2_ITEMS.append((ast.literal_eval(node.args[1]) if isinstance(node.args[1], ast.Constant) else "?", ph))
for w_, ph_ in V2_ITEMS:
    assert locate(tex5, ph_) is not None, (w_, ph_)          # every fifth-pass fragment is in c5e4508

# ── the --code checks ────────────────────────────────────────────────────────


def code_checks(d):
    out = {}
    sys.path.insert(0, d)
    import torch
    import test_window_gate as twg
    import test_stream_curriculum as tscur
    import test_stream_recipe as tsr
    task = tscur.TASK48
    x, qs = task.make_batch(64, torch.Generator().manual_seed(12345))
    out["probe counts"] = [int((qs.reshape(-1) == s).sum()) for s in range(task.S)]
    a = tscur.ARM["D8"]
    torch.manual_seed(540)
    m = twg.to_local(tsr.builder_for(a)(a, 540)(), 540)
    out["dtypes"] = sorted({str(p.dtype) for p in m.parameters()})
    out["default dtype"] = str(torch.get_default_dtype())
    out["W_h requires_grad"] = m.W_h.requires_grad
    h = {}
    twg.local_groups(h)(m)
    out["SLOW groups"] = h["names"]
    out["window == window_init(540)"] = bool(torch.equal(m.gate_conv.conv_w.data, twg.window_init(540)))
    out["encoder shape"] = tuple(m.encoder.shape)
    c = twg.CEIL_C_ARM
    torch.manual_seed(540)
    mc = tsr.builder_for(c)(c, 540)()
    out["CEIL_C"] = dict(n_ch=mc.n_ch, gate=mc.gate_kind, ctx_tokens=mc.ctx_tokens, cur=c["cur"])
    out["argmax/argmin on ties"] = (int(torch.tensor([0.1, 0.3, 0.3]).argmax()), int(torch.tensor([0.2, 0.0, 0.0]).argmin()))
    w = torch.tensor([1.0, 2.0, 3.0, 4.0])
    import test_binding_onset as tbo_
    eb = tbo_.eval_batch(task)
    eb2 = task.make_batch(tbo_.EVAL_QUERIES // task.n_q, torch.Generator().manual_seed(424242))[0]
    out["eval_batch == make_batch(.., Generator(424242))"] = bool(torch.equal(eb, eb2))
    out["eval_batch sequences"] = int(eb.shape[0])
    out["Tensor.std() unbiased"] = bool(torch.isclose(w.std(), w.std(unbiased=True)) and not torch.isclose(w.std(), w.std(unbiased=False)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default=None)
    args = ap.parse_args()
    CC = code_checks(args.code) if args.code else None
    if CC:
        assert CC["probe counts"] == [5, 5, 6, 8, 10, 12, 5, 13] and CC["dtypes"] == ["torch.float32"]
        assert CC["W_h requires_grad"] is False and all("W_h" not in g for g in CC["SLOW groups"])
        assert CC["window == window_init(540)"] and CC["argmax/argmin on ties"] == (1, 1) and CC["Tensor.std() unbiased"]
        assert CC["eval_batch == make_batch(.., Generator(424242))"]
        assert CC["CEIL_C"]["n_ch"] == 16 and CC["CEIL_C"]["ctx_tokens"] == tuple(range(8)) + tuple(-(i + 1) for i in range(8))
    ORDER = {"FALSE": 0, "PARTIAL": 1, "UNVERIFIABLE": 2, "TRUE": 3}
    P = print
    P("# Revision 9, sixth draft: Appendix B and its body sentences against the main line's code and records\n")
    APPB6 = locate(tex6, ("\\label{app:repro}",)) or locate(tex6, ("What a replication needs",))
    P(f"Sixth draft: `origin/claude/rev9-draft` at `{NEW_REF}` (committed {CUTOFF}), `{DRAFT}` (`tex6:NNN`, "
      f"{len(tex6) - (tex6[-1] == '')} lines); fifth draft `{V5_REF}` (`tex5:NNN`, {len(tex5) - (tex5[-1] == '')} lines: 42c4557 "
      f"replaces lines in place, so the numbers agree; the lines it changed: "
      f"{[i + 1 for i in range(min(len(tex5), len(tex6))) if tex5[i] != tex6[i]]}; Appendix B, tex6:526-549, unchanged: "
      f"{tex5[525:549] == tex6[525:549]}); first draft `{OLD_REF}` (`tex:NNN`). "
      f"Code and records: the main line as of the sixth draft, `{HEAD}` (read with `git show`; its *.py differ from db22857's "
      f"only in {PY_DIFF}; since 8ae660d only test_split_copy.py's docstring); R's branch as of the sixth draft, `{R_BR}`. The path "
      "is test_window_gate's Part C/D path: `one_run` -> `run_path` -> `test_slow_start.part_run(\"C\")` -> "
      "`test_stream_curriculum.run_sc(ARM[\"D8\"])` -> `test_router_reliability.run_one` -> `test_binding_onset.onset_run`, the "
      "model `test_multilayer_binding.MultiBDH` (on `bdh.BDH`) converted by `test_window_gate.to_local`; the perfect gate "
      f"`run_sc(CEIL_C_ARM)`. Generated by `check_appendixB_v6.py{' --code DIR' if CC else ''}` (a copy of v5 re-anchored on "
      "42c4557); each item's fragment is located in the sixth draft by text search and each piece of evidence is a code "
      "snippet resolved to its line at the head.\n")
    P("Abbreviations: " + ", ".join(f"{k} = {v}" for k, v in FILES.items()) + ".\n")
    cnt = {k: sum(1 for x in ITEMS if x[2] == k) for k in ORDER}
    P("## Counts\n")
    P("| verdict | items |\n|---|---|")
    for k, v in cnt.items():
        P(f"| {k} | {v} |")
    v2_gone = [(w, ph) for w, ph in V2_ITEMS if locate(tex6, ph) is None]
    P(f"\n({len(ITEMS)} atomic items against the sixth draft. Fifth-pass items: {len(V2_ITEMS) - len(v2_gone)} of "
      f"{len(V2_ITEMS)} fragments found verbatim in the sixth draft, {len(v2_gone)} reworded and re-anchored or REMOVED; "
      f"{len(ITEMS) - len(V2_ITEMS)} items are new.)\n")
    P("## 1. Items (FALSE and PARTIAL first)\n")
    P("| # | tex6 line | item | verdict | code evidence (file:line at the main line as of the sixth draft) | note | first-pass items |")
    P("|---|---|---|---|---|---|---|")
    srt = sorted(ITEMS, key=lambda x: (ORDER[x[2]], x[0]))
    for i, (n, w, v, e, note, old) in enumerate(srt):
        P(f"| {i + 1} | tex6:{n} | {w.replace('|', '/')} | {v} | {e} | {note.replace('|', '/')} | "
          f"{', '.join(str(o) for o in old) or 'new'} |")
    P("\n## 2. Fifth-pass items located in the sixth draft\n")
    P("Each fifth-pass fragment (from `check_appendixB_v5.py`'s `item(...)` calls, read with `ast`) was asserted in "
      "`c5e4508` and searched verbatim in `42c4557`.\n")
    P("| # | fifth-pass item | tex5 | tex6 |\n|---|---|---|---|")
    for i, (w, ph) in enumerate(V2_ITEMS):
        m6 = locate(tex6, ph)
        l5 = locate(tex5, ph)
        ra = [x[0] for x in ITEMS if x[1].endswith("(re-anchored)") and x[0] == l5] if not m6 else []
        P(f"| {i + 1} | {w.replace('|', '/')} | tex5:{l5} | "
          f"{f'tex6:{m6}' if m6 else (f'REWORDED; re-anchored at tex6:{ra[0]}' if ra else 'REMOVED')} |")
    P("\n## 3. First-pass items (1-87) in the sixth draft\n")
    P("| first-pass items | first verdict | now |\n|---|---|---|")
    mp = {}
    for n, w, v, e, note, old in ITEMS:
        for o in old:
            mp.setdefault(o, []).append(f"tex6:{n} {v}")
    first = {1: "FALSE", 2: "FALSE", 3: "FALSE", 4: "PARTIAL", 5: "PARTIAL", 6: "UNVERIFIABLE", 7: "UNVERIFIABLE",
             8: "UNVERIFIABLE"}
    for o in range(1, 88):
        P(f"| {o} | {first.get(o, 'TRUE')} | {'; '.join(sorted(set(mp.get(o, ['not mapped']))))} |")
    unm = [o for o in range(1, 88) if o not in mp]
    P(f"\nFirst-pass items not mapped to a sixth-draft item: {unm or 'none'}.\n")
    P("## 4. Record and code facts\n")
    P(f"- WIN3_SPLIT_D8 check rows, window_gate X+L: {PV['rows']} in {PV['runs']} runs, of which {PV['ref_rows']} are the "
      f"2400 reference readings, so {N_CHECKS} checks; probe >= 0.95 with the evaluation < 0.95 at the same step: {len(PV['hi'])}, all at checks (>= 4800): {HI_CHECKS} "
      f"(evaluation {EV_LO:.4f}-{EV_HI:.4f}); blocked by the probe alone: {len(ALONE)}. WIN3_RESET_D8: {PV['reset_hi']} of "
      f"{PV['reset_rows']} rows.")
    for h in PV["hi"]:
        P(f"  - {h['m']} s{h['s']}@{h['step']}: probe {h['probe']:.4f}, eval {h['ev']:.4f}, gap/cap block {h['capped']}, "
          f"rise < 0.02 {h['rise_ok']}, blocked by the probe alone {h['alone']}, next check {h['next_step']} fired "
          f"{h['next_fired']}, run bound routed {h['br']}, map_before {h['map']}")
    P(f"- stream_gate at every evaluation: window_gate split/reset runs {PV['sg_every_eval']}; split_copy split runs {SC_SG}.")
    P(f"- Recorded start strings (no zone field): {starts}; zone-free: {ZONE_FREE}.")
    P(f"- R: UNSTATED.md {N_U} Phase 2 entries ({N_U_ALL} in the file at {R_BR}, Phase 5's {len(P5_IDS)} included), DIFF.md {N_D} rows, {N_MATCHED} 'matched' bullets; labels {dict(LC)}; seed 914 "
      f"(n, probe min, probe max, eval min, eval max, failure, step of the max): {R914}.")
    if CC:
        P("- --code (read-only builds, no training): " + "; ".join(f"{k} = {v}" for k, v in CC.items()) + ".")
    else:
        P("- --code not given: the build checks were skipped.")
    P("\n## 5. Omissions noticed (not verdicts)\n")
    P("- The held-out set's generator seed (tbo:" + ev("tbo", "EVAL_SEED = 424_242").split(":")[1]
      + ", R's D-7) is stated (tex6:532); this second-pass omission stays closed.")
    P("- The window's draw layout, `torch.rand(D, 3, generator=g)` with column j the lag-j tap (twg:"
      + ev("twg", "return (torch.rand(D, WIDTH, generator=g) * 2 - 1)").split(":")[1] + "; R's D-12), is not stated. Bit-for-bit only.")
    P("- encoder_v's N(0, 0.02^2) is covered by 'Pathway's initialisation' but missing from the parenthesis that lists the others.")
    P("- The perfect gate's gate before any context token is uniform over the 16 padded channels (tiv perfect_gate_general); "
      "every sequence starts with a context token, so this never applies here.")


if __name__ == "__main__":
    main()
