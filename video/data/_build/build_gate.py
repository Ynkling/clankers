"""video/data/gate_dynamics.json — per-run gate statistics at every evaluation (step 0 = initialization)."""
from common import *

SRC = {"slow_start": "results/X/slow_start_results.json", "early_recipe": "results/X/early_recipe_results.json",
       "stream_recipe": "results/X/stream_recipe_results.json",
       "stream_curriculum": "results/X/stream_curriculum_results.json"}
D = {k: load(v) for k, v in SRC.items()}

K2_FIELDS = ["margin", "eta_key_by_stream", "eta_key_by_index", "eta_key_by_half", "eta_key_by_key",
             "hs_eta_stream", "hs_eta_index", "hs_eta_half", "hs_eta_key", "val_cos", "key_cos", "ctx_cos",
             "eff_ch", "spread", "sep"]
KN_FIELDS = ["margin", "eff_ch", "spread", "etak_key_by_stream", "etak_key_by_index", "etak_key_by_block",
             "etak_key_by_key", "etak_val_by_stream", "hs_eta_stream", "hs_eta_index", "hs_eta_half", "hs_eta_key"]

K2 = [("slow_start", "A0", 294, "plain gate, POSITION split"), ("slow_start", "SLOW0", 294, "SLOW alone, POSITION split"),
      ("slow_start", "HINGE0", 294, "HINGE, discovered (hinge fired once in 1-1200)"),
      ("slow_start", "A0", 281, "plain gate, fast discovery (0.91 at 1200)"),
      ("early_recipe", "SLOW_M", 304, "Muon without hinge, POSITION split"),
      ("early_recipe", "WIN_M", 304, "Muon + hinge window, discovered (fired at 180, 230)")]
KN = [("stream_recipe", "A4k16", 246, "S=4, k=16 plain: bound routed at 2400"),
      ("stream_recipe", "A4k16", 256, "S=4, k=16 plain: MERGED (3 share), plateau 0.5"),
      ("stream_recipe", "A4k4", 258, "S=4, k=4 plain: MERGED (2 share), plateau 0.75"),
      ("stream_recipe", "A4k16", 241, "S=4, k=16 plain: non-stream, all streams on one channel"),
      ("slow_start", "HINGE4k16", 240, "S=4, k=16 HINGE: bound routed at 3600"),
      ("early_recipe", "WIN16_M", 340, "S=4, k=16 Muon + window: bound at 2400"),
      ("stream_curriculum", "SC8", 261, "S=8, k=16 stream curriculum (2 -> 4 -> 8 streams): bound routed at 10800"),
      ("stream_curriculum", "D8", 260, "S=8, k=16 all eight streams from step 1: collapsed"),
      ("slow_start", "HINGE_D8", 265, "S=8, k=16 HINGE from step 1: best HINGE_D8 run, 0.47")]


def series(r, fields, nd=4):
    st = sorted(r["stats"], key=lambda x: x["step"])
    steps = [x["step"] for x in st]
    acc = {c[0]: round(c[1], 4) for c in r["curve"]}
    out = dict(steps=steps, acc=[acc.get(s) for s in steps])
    for f in fields:
        out[f] = [r4(x.get(f), nd) for x in st]
    return st, out


k2, kn = {}, {}
for src, arm, seed, role in K2:
    r = runs_of(D[src], arm)[seed]
    _, s = series(r, K2_FIELDS)
    s.update(file=SRC[src], role=role, outcome=tag2(r), transition=r["transition"], k=r["k"])
    k2[f"X:{arm}|{seed}"] = s
for src, arm, seed, role in KN:
    r = runs_of(D[src], arm)[seed]
    st, s = series(r, KN_FIELDS)
    s["stream_gate"] = [[[round(v, 3) for v in row] for row in x["stream_gate"]] for x in st]
    s["ch_map"] = [x["ch_map"] for x in st]
    s["one_to_one"] = [x["one_to_one"] for x in st]
    s.update(file=SRC[src], role=role, outcome=tagk(r), transition=r["transition"], k=r["k"],
             S=len(st[0]["stream_gate"]), stream_acc_end=[round(v, 3) for v in r["end"]["stream_acc"]])
    if src == "stream_curriculum" or arm == "HINGE_D8":
        s["note"] = ("probe and acc are on the 8-stream set; in SC8 the training batches hold 2 streams for updates "
                     "1-4800, 4 for 4801-9600, then 8")
    kn[f"X:{arm}|{seed}"] = s

obj = {
    "_source": {
        "files": SRC,
        "keys_used": "runs[ARM|seed].stats[i] (one entry per evaluation, step 0 = the initialized model, then every 1200 "
                     "updates) and curve",
        "built_by": "video/data/_build/build_gate.py (python3, no torch; run from any directory)",
        "all X": "every run here is from results/X; L's per-eval gate statistics are not in the repo except in "
                 "results/L/early_recipe_results.json",
    },
    "fields": {
        "stream_gate": "S x k matrix: each stream's mean read gate (channel distribution) at that stream's value "
                       "positions on the probe batch (test_scale_axes.py:298; 4 dp in the source, 3 dp here). Uniform "
                       "= 1/k everywhere; routed = one channel per stream near 1.",
        "ch_map": "argmax channel of each stream's row of stream_gate (stream 0 first)",
        "one_to_one": "every stream has a different argmax channel",
        "margin": "routing margin at the query: g_q . g_target minus the mean over the other streams' bindings of "
                  "g_q . g_distractor (test_router_layout.py:262-270); 1 = perfectly routed, 0 = uniform",
        "eta_key_by_*": "k=2: eta^2 of the read gate's channel-0 probability at key positions grouped by stream / "
                        "triple index / half / key (test_router_layout.py:241)",
        "etak_*": "k>2: multichannel eta^2 pooled over channels (test_scale_axes.py:259 eta2_multi), groups stream / "
                  "index / block / key, at key (etak_key_*) or value (etak_val_*) positions",
        "hs_eta_*": "the hinge's own pooled statistic at key positions (test_slow_start.py:428 extra_stats); the hinge "
                    "penalises hs_eta_index and hs_eta_half above 0.2",
        "val_cos": "k=2: cosine between the two streams' mean read gates at value positions; DISCOVERED needs < 0.5 at "
                   "the end (test_multilayer_binding.py:360)",
        "eff_ch": "exp(entropy) of the read gate averaged over all positions (test_router_reliability.py:262)",
        "spread": "mean normalised distance of the gate from uniform: 0 = uniform, 1 = one-hot",
        "acc": "held-out accuracy at that step (null at step 0: no evaluation before training)",
    },
    "k2_runs": k2,
    "multichannel_runs": kn,
}

if __name__ == "__main__":
    for k, v in kn.items():
        print(k, v["outcome"], v["steps"][:4], v["ch_map"][:3], v["stream_gate"][0][0][:4], v["margin"][:4])
    for k, v in k2.items():
        print(k, v["outcome"], v["margin"][:4], v["hs_eta_index"][:4], v["val_cos"][:4])
    write("gate_dynamics.json", obj)
