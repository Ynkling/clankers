"""Chapter 17 — the early-recipe test (report §9.2, Table 6; screens S32, S34, S35 in §9.1;
video/data/early_recipe.json, built from results/{X,L}/early_recipe_results.json; test_early_recipe.py)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
with open(os.path.join(DATA, "early_recipe.json")) as _fh:
    _ER = json.load(_fh)
COUNTS, CLAIMS, PER_SEED = _ER["counts"], _ER["claims"], _ER["per_seed"]
MACHINES = ("X", "L")


# ------------------------------------------------------------------------------------- real data
def _code(outcome):
    """D discovered, P position split, O any other failure (STREAM-PARTIAL)."""
    return "D" if outcome == "DISCOVERED" else ("P" if outcome == "POSITION" else "O")


# Part A (S = 2, seeds 300-339): one letter per seed, Muon + slow memory without / with the hinge window
PART_A = {m: {a: "".join(_code(o) for o in PER_SEED[m][a]["outcome"]) for a in ("SLOW_M", "WIN_M")}
          for m in MACHINES}
RESCUED = {}
for _m in MACHINES:
    for _a in ("SLOW_M", "WIN_M"):
        assert len(PART_A[_m][_a]) == COUNTS[_m][_a]["n"] == 40
        assert PART_A[_m][_a].count("D") == COUNTS[_m][_a]["success"]
    _s, _w = PART_A[_m]["SLOW_M"], PART_A[_m]["WIN_M"]
    RESCUED[_m] = [i for i in range(40) if _s[i] != "D" and _w[i] == "D"]
    _lost = [i for i in range(40) if _s[i] == "D" and _w[i] != "D"]
    assert (len(RESCUED[_m]), len(_lost)) == (CLAIMS[_m]["E1"]["b"], CLAIMS[_m]["E1"]["c"])

# Part B (S = 4, k = 16, seeds 340-359; X 340-351): update of the first evaluation at which a run counted as bound
TRANS = {m: {a: PER_SEED[m][a]["transition"] for a in ("WIN16_A", "WIN16_M")} for m in MACHINES}
MEDIAN = {}


def _by4800(t):
    return t is not None and t <= 4800


for _m in MACHINES:
    for _a in ("WIN16_A", "WIN16_M"):
        _tr = TRANS[_m][_a]
        assert len(_tr) == COUNTS[_m][_a]["n"]
        assert sum(t is not None for t in _tr) == COUNTS[_m][_a]["success"]
        assert sum(_by4800(t) for t in _tr) == COUNTS[_m][_a]["bound_by_4800"]
        MEDIAN[_m, _a] = int(np.median([t for t in _tr if t is not None]))
    _ad, _mu = TRANS[_m]["WIN16_A"], TRANS[_m]["WIN16_M"]
    _b = sum(_by4800(y) and not _by4800(x) for x, y in zip(_ad, _mu))
    _c = sum(_by4800(x) and not _by4800(y) for x, y in zip(_ad, _mu))
    assert (_b, _c) == (CLAIMS[_m]["E2"]["b"], CLAIMS[_m]["E2"]["c"])
    assert MEDIAN[_m, "WIN16_M"] == 2400 and MEDIAN[_m, "WIN16_A"] == 4800   # report §9.2

# the claims as the report prints them: "4 vs 0, p = 0.0625, NOT SHOWN"
CLAIM_TXT = {(m, e): CLAIMS[m][e]["report"].split(", ") for m in MACHINES for e in ("E1", "E2")}

# seed-level agreement of the two machines on SLOW_M (same seeds; report §9.2: 38/40)
AGREE = [(PART_A["X"]["SLOW_M"][i] == "D") == (PART_A["L"]["SLOW_M"][i] == "D") for i in range(40)]
assert sum(AGREE) == _ER["seed_level_agreement_X_vs_L"]["SLOW_M"]["same_outcome"] == 38
# held-out accuracy of SLOW_M seed 300 at the first evaluation (update 1200), per machine
ACC300 = {m: _ER["curves"][m]["SLOW_M|300"][0] for m in MACHINES}
assert ACC300["X"] != ACC300["L"]

SCREEN_S32 = (31, 40)   # Muon + slow memory without the hinge, screen S32, seeds 160-199 (report §9.1)

OUT_COLOR = {"D": GOOD, "P": BAD, "O": MUTED}
OUT_OPAC = {"D": 0.9, "P": 0.9, "O": 0.6}
OPT_COLOR = {"WIN16_A": ADAM_COLOR, "WIN16_M": MUON_COLOR}
OPT_NAME = {"WIN16_A": "Adam", "WIN16_M": "Muon"}


# ------------------------------------------------------------------------------------- helpers
def at_phrase(vo, i, phrase):
    """Approximate time (s into the block) at which `phrase` is spoken in sentence i."""
    text, a, b = vo.synth.sentences[i]
    k = text.find(phrase)
    if k < 0:
        raise ValueError(phrase)
    return a + (b - a) * k / max(len(text), 1)


def chip(text, color, size=24, padx=0.16, pady=0.09, fill=0.2):
    t = L(text, size=size, color=color, weight="BOLD")
    box = RoundedRectangle(width=t.get_width() + 2 * padx, height=t.get_height() + 2 * pady, corner_radius=0.1)
    box.set_fill(color, fill).set_stroke(color, 2)
    t.move_to(box)
    return VGroup(box, t)


def legend_item(color, text, opacity=0.9, size=20):
    sq = Square(0.2).set_fill(color, opacity).set_stroke(width=0)
    return VGroup(sq, L(text, size=size, color=MUTED)).arrange(RIGHT, buff=0.12)


def strip(codes, x0, y, cell, pitch):
    return VGroup(*[Square(cell).set_fill(OUT_COLOR[c], OUT_OPAC[c]).set_stroke(width=0)
                    .move_to([x0 + i * pitch, y, 0]) for i, c in enumerate(codes)])


def empty_strip(n, x0, y, cell, pitch):
    return VGroup(*[Square(cell).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1)
                    .move_to([x0 + i * pitch, y, 0]) for i in range(n)])


def cross_over(mob, color=BAD, width=2.5):
    return VGroup(Line(mob.get_corner(UL), mob.get_corner(DR)),
                  Line(mob.get_corner(DL), mob.get_corner(UR))).set_stroke(color, width)


def header_line(tag, color, statement, detail):
    g = VGroup(chip(tag, color, 22), T(statement, 30), L(detail, 22, MUTED)).arrange(RIGHT, buff=0.3)
    g[2].align_to(g[1], DOWN).shift(0.02 * UP)
    g.to_edge(LEFT, buff=0.55).set_y(2.25)
    return g


def claim_cell(machine, claim):
    pair, p, verdict = CLAIM_TXT[machine, claim]
    top = VGroup(L(pair, 26, INK, weight="BOLD"), verdict_badge(verdict, 20)).arrange(RIGHT, buff=0.22)
    bottom = L(p, 20, MUTED)
    g = VGroup(top, bottom).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
    g.pair, g.badge, g.p = top[0], top[1], bottom
    return g


class EarlyRecipeTest(ClankersScene):
    def construct(self):
        card = self.chapter_card(17, "The early-recipe test")
        self.wait(0.6)
        title = section_title("Two findings, fresh seeds")
        self.play(FadeOut(card, shift=0.3 * UP), FadeIn(title, shift=0.3 * UP))
        self.title_mob = title

        self.part_a()
        self.part_b()
        self.why_x()
        self.not_independent()
        self.wait(1.0)
        self.clear_all()

    def fade_all_but_title(self, run_time=0.7, keep=()):
        mobs = [m for m in self.mobjects if m is not self.title_mob and m is not self.frame and m not in keep]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    # ============================================================== N17.1  two findings; two streams
    def part_a(self):
        def finding(tag, color, statement, old, new):
            c = chip(tag, color, 26)
            st = T(statement, 34)
            top = VGroup(c, st).arrange(RIGHT, buff=0.3)
            o = L(old, 22, MUTED)
            arr = Arrow(LEFT, RIGHT, buff=0, thickness=3).set_width(0.6).set_color(GOOD)
            n = L(new, 22, GOOD)
            prov = VGroup(o, arr, n).arrange(RIGHT, buff=0.2)
            body = VGroup(top, prov).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
            prov.align_to(st, LEFT)
            panel = RoundedRectangle(width=11.6, height=body.get_height() + 0.55, corner_radius=0.15)
            panel.set_fill(PANEL, 1).set_stroke(color, 1.5)
            body.move_to(panel).align_to(panel, LEFT).shift(0.4 * RIGHT)
            g = VGroup(panel, c, st, o, arr, n)
            g.panel, g.chip, g.st, g.old, g.arr, g.new = panel, c, st, o, arr, n
            return g

        f1 = finding("E1", HINGE_COLOR, "the early hinge window helps under Muon",
                     "screens S32, S34: seeds 160–199", "fresh seeds 300–339, two streams")
        f2 = finding("E2", MUON_COLOR, "Muon binds four streams sooner",
                     "screen S35: seeds 240–259", "fresh seeds 340–359, four streams")
        VGroup(f1, f2).arrange(DOWN, buff=0.5).move_to(0.35 * DOWN)
        src0 = source_note("Report §9.1 (screens S32, S34, S35), §9.2")

        # ---- the per-seed picture of Part A
        CELL, PITCH, X0 = 0.17, 0.205, -3.95
        ROWS = {("X", "SLOW_M"): 1.05, ("X", "WIN_M"): 0.5, ("L", "SLOW_M"): -0.65, ("L", "WIN_M"): -1.2}
        COUNT_X, TAG_X = 4.55, 5.15
        badges = VGroup(*[machine_badge(m, 24).move_to([-6.3, (ROWS[m, "SLOW_M"] + ROWS[m, "WIN_M"]) / 2, 0])
                          for m in MACHINES])
        labels, empties, strips, counts = {}, {}, {}, {}
        for (m, a), y in ROWS.items():
            labels[m, a] = L("slow memory" if a == "SLOW_M" else "+ hinge window", 22,
                             SLOW_COLOR if a == "SLOW_M" else HINGE_COLOR).move_to([-5.9, y, 0], aligned_edge=LEFT)
            empties[m, a] = empty_strip(40, X0, y, CELL, PITCH)
            strips[m, a] = strip(PART_A[m][a], X0, y, CELL, PITCH)
            n = COUNTS[m][a]
            counts[m, a] = L(f"{n['success']}/{n['n']}", 24, GOOD if a == "WIN_M" else INK,
                             weight="BOLD").move_to([COUNT_X, y, 0])
        seed_lab = VGroup(L("seed 300", 20, MUTED).next_to(empties["L", "WIN_M"][0], DOWN, buff=0.18),
                          L("seed 339", 20, MUTED).next_to(empties["L", "WIN_M"][-1], DOWN, buff=0.18))
        legend = VGroup(legend_item(GOOD, "discovered"), legend_item(BAD, "position split"),
                        legend_item(MUTED, "other failure", opacity=0.6)).arrange(RIGHT, buff=0.45)
        legend.move_to([0.0, -2.45, 0])
        src_a = source_note("Report §9.2, Table 6; data/early_recipe.json (per seed)")

        # rescue outlines (the seeds the window turned from failure into discovery) and tags
        outlines, tags = {}, {}
        for m in MACHINES:
            ys, yw = ROWS[m, "SLOW_M"], ROWS[m, "WIN_M"]
            outlines[m] = VGroup(*[
                RoundedRectangle(width=PITCH + 0.02, height=(ys - yw) + CELL + 0.14, corner_radius=0.05)
                .set_stroke(HINGE_COLOR, 2.5).set_fill(opacity=0).move_to([X0 + i * PITCH, (ys + yw) / 2, 0])
                for i in RESCUED[m]])
            tags[m] = VGroup(L(f"{len(RESCUED[m])} rescued", 20, HINGE_COLOR, weight="BOLD"),
                             L("none lost", 20, MUTED)).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
            tags[m].move_to([TAG_X, (ys + yw) / 2, 0], aligned_edge=LEFT)

        with self.voiceover(
            "The early recipe test took two screened findings to fresh seeds: that the early hinge window helps "
            "under Muon, and that Muon binds four streams sooner. At two streams, Muon with slow memory alone "
            "discovered thirty-six of forty on X and thirty-four on L. Adding the early hinge window made it "
            "forty of forty, on both."
        ) as vo:
            # two screened findings ...
            self.play(LaggedStart(*[FadeIn(VGroup(f.panel, f.chip, f.old), shift=0.2 * UP) for f in (f1, f2)],
                                  lag_ratio=0.3), FadeIn(src0), run_time=1.2)
            # ... to fresh seeds
            vo.wait_until(at_phrase(vo, 0, "to fresh seeds"))
            self.play(*[AnimationGroup(GrowArrow(f.arr), FadeIn(f.new, shift=0.2 * RIGHT),
                                       f.old.animate.set_opacity(0.55)) for f in (f1, f2)], run_time=1.0)
            vo.wait_until(at_phrase(vo, 0, "that the early hinge"))
            self.play(Write(f1.st), run_time=1.3)
            vo.wait_until(at_phrase(vo, 0, "and that Muon"))
            self.play(Write(f2.st), run_time=1.1)

            # E1 becomes the header; the per-seed board for two streams
            vo.wait_until_sentence(1)
            head = header_line("E1", HINGE_COLOR, "the early hinge window helps under Muon",
                               "two streams · all under Muon · seeds 300–339")
            self.play(ReplacementTransform(f1.chip, head[0]), ReplacementTransform(f1.st, head[1]),
                      FadeTransform(f1.new, head[2]), FadeOut(VGroup(f1.panel, f1.old, f1.arr)),
                      FadeOut(f2, shift=0.4 * DOWN), FadeOut(src0), run_time=1.1)
            self.play(FadeIn(badges), *[FadeIn(labels[k]) for k in labels], *[FadeIn(empties[k]) for k in empties],
                      FadeIn(seed_lab), FadeIn(legend), FadeIn(src_a), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "discovered thirty-six") - 0.3)
            self.play(LaggedStartMap(FadeIn, strips["X", "SLOW_M"], lag_ratio=0.03), FadeIn(counts["X", "SLOW_M"]),
                      run_time=1.2)
            vo.wait_until(at_phrase(vo, 1, "and thirty-four") - 0.1)
            self.play(LaggedStartMap(FadeIn, strips["L", "SLOW_M"], lag_ratio=0.03), FadeIn(counts["L", "SLOW_M"]),
                      run_time=1.2)
            fails = VGroup(*[strips[m, "SLOW_M"][i] for m in MACHINES for i in RESCUED[m]])
            self.play(LaggedStartMap(Indicate, fails, scale_factor=1.6, color=BAD, lag_ratio=0.1), run_time=1.0)

            # adding the window: copy each row down, the failures turn into discoveries
            vo.wait_until_sentence(2)
            copies = {m: strips[m, "SLOW_M"].copy() for m in MACHINES}
            self.play(*[copies[m].animate.move_to(empties[m, "WIN_M"]) for m in MACHINES],
                      *[Indicate(labels[m, "WIN_M"], color=HINGE_COLOR, scale_factor=1.1) for m in MACHINES],
                      run_time=0.9)
            self.remove(*copies.values())
            self.add(*[strips[m, "WIN_M"] for m in MACHINES])
            for m in MACHINES:   # start the WIN_M rows from the SLOW_M colours
                for i in RESCUED[m]:
                    strips[m, "WIN_M"][i].set_fill(OUT_COLOR[PART_A[m]["SLOW_M"][i]], 0.9)
            self.play(*[strips[m, "WIN_M"][i].animate.set_fill(GOOD, 0.9) for m in MACHINES for i in RESCUED[m]],
                      *[FadeIn(counts[m, "WIN_M"], scale=1.3) for m in MACHINES],
                      *[LaggedStartMap(ShowCreation, outlines[m], lag_ratio=0.1) for m in MACHINES], run_time=0.9)
            self.play(*[FadeIn(tags[m], shift=0.15 * LEFT) for m in MACHINES], run_time=0.6)

        self.part_a_mobs = dict(head=head)

    # ============================================================== N17.2  four streams; the verdicts
    def part_b(self):
        head_a = self.part_a_mobs["head"]
        head = header_line("E2", MUON_COLOR, "Muon binds four streams sooner",
                           "four streams · 16 channels · seeds 340–359 (X: 340–351)")

        # ---- the dot plot: one dot per run at the evaluation where it first counted as bound
        CHECKS = [1200, 2400, 3600, 4800, 6000, 7200, 8400]
        X_COL0, DX = -3.85, 0.76
        COL_X = {t: X_COL0 + DX * i for i, t in enumerate(CHECKS)}
        X_LATER, X_NEVER = COL_X[8400] + 1.05, COL_X[8400] + 2.15
        X_CUT = (COL_X[4800] + COL_X[6000]) / 2
        ROWS = {("X", "WIN16_A"): 1.3, ("X", "WIN16_M"): 0.72, ("L", "WIN16_A"): -0.08, ("L", "WIN16_M"): -0.66}
        Y_TOP, Y_BOT = 1.62, -0.98
        Y_TICK = -1.25
        C_BOUND, C_BY, C_MED = 3.95, 4.95, 5.95
        R, SP = 0.058, 0.135

        def col_of(t):
            if t is None:
                return X_NEVER
            return COL_X[t] if t <= 8400 else X_LATER

        dots, row_dots = VGroup(), {}
        for (m, a), y in ROWS.items():
            groups = {}
            for t in TRANS[m][a]:
                groups.setdefault(col_of(t), []).append(t)
            rd = VGroup()
            for cx, ts in groups.items():
                n = len(ts)
                nrows, ncols = min(n, 4), int(np.ceil(n / 4))
                for k, t in enumerate(ts):
                    cc, rr = k // 4, k % 4
                    p = [cx + (cc - (ncols - 1) / 2) * SP, y + ((nrows - 1) / 2 - rr) * SP, 0]
                    if t is None:
                        d = Circle(radius=R).set_stroke(BAD, 2).set_fill(BG, 0).move_to(p)
                        d.fo, d.so = 0.0, 1.0
                    else:
                        d = Dot(p, radius=R).set_fill(OPT_COLOR[a], 1).set_stroke(width=0)
                        d.fo, d.so = 1.0, 0.0
                    d.col_x, d.t = cx, t
                    rd.add(d)
            row_dots[m, a] = rd
            dots.add(*rd)

        badges = VGroup(*[machine_badge(m, 24).move_to([-6.3, (ROWS[m, "WIN16_A"] + ROWS[m, "WIN16_M"]) / 2, 0])
                          for m in MACHINES])
        opt_labs = {k: L(OPT_NAME[k[1]], 24, OPT_COLOR[k[1]], weight="BOLD").move_to([-5.85, y, 0], aligned_edge=LEFT)
                    for k, y in ROWS.items()}
        ticks = VGroup(*[L(f"{t}", 20, MUTED).move_to([COL_X[t], Y_TICK, 0]) for t in CHECKS],
                       L("later", 20, MUTED).move_to([X_LATER, Y_TICK, 0]),
                       L("not bound", 20, BAD).move_to([X_NEVER, Y_TICK, 0]))
        tick_head = L("bound at update", 20, MUTED).move_to([-6.45, Y_TICK, 0], aligned_edge=LEFT)
        dot_key = VGroup(Dot(radius=R).set_fill(INK, 1).set_stroke(width=0), L("= one run (one seed)", 20, MUTED))
        dot_key.arrange(RIGHT, buff=0.12).move_to([-6.45, Y_TOP + 0.15, 0], aligned_edge=LEFT)
        brk_x = (COL_X[8400] + X_LATER) / 2
        brk = VGroup(*[Line([brk_x - 0.07 + dx, Y_TICK - 0.12, 0], [brk_x + 0.07 + dx, Y_TICK + 0.12, 0])
                       .set_stroke(MUTED, 2) for dx in (-0.06, 0.06)])
        guides = VGroup(*[Line([-4.35, y, 0], [X_NEVER + 0.45, y, 0]).set_stroke(FAINT, 1, opacity=0.5)
                          for y in ROWS.values()])
        col_heads = VGroup(L("bound", 20, MUTED).move_to([C_BOUND, Y_TOP + 0.15, 0]),
                           L("by 4800", 20, GOOD).move_to([C_BY, Y_TOP + 0.15, 0]),
                           L("median", 20, MUTED).move_to([C_MED, Y_TOP + 0.15, 0]))
        bound_vals, by_vals, med_vals = {}, {}, {}
        for (m, a), y in ROWS.items():
            n = COUNTS[m][a]
            bound_vals[m, a] = L(f"{n['success']}/{n['n']}", 24, INK, weight="BOLD").move_to([C_BOUND, y, 0])
            by_vals[m, a] = L(f"{n['bound_by_4800']}/{n['n']}", 24, GOOD, weight="BOLD").move_to([C_BY, y, 0])
            med_vals[m, a] = L(f"{MEDIAN[m, a]}", 24, OPT_COLOR[a], weight="BOLD").move_to([C_MED, y, 0])

        cursor_x = ValueTracker(X_COL0 - 0.6)
        cursor = Line([0, Y_TOP, 0], [0, Y_BOT, 0]).set_stroke(INK, 2, opacity=0.7)
        cursor.add_updater(lambda mob: mob.set_x(cursor_x.get_value()))

        def reveal(d):
            f = float(np.clip((cursor_x.get_value() - d.col_x + 0.25) / 0.3, 0, 1))
            d.set_fill(opacity=f * d.fo).set_stroke(opacity=f * d.so)

        cut = DashedLine([X_CUT, Y_TOP, 0], [X_CUT, Y_BOT, 0], dash_length=0.1).set_stroke(GOOD, 2.5)
        cut_lab = L("update 4800", 20, GOOD).next_to(cut, UP, buff=0.08)
        shade = Rectangle(width=X_CUT - (X_COL0 - 0.45), height=Y_TOP - Y_BOT)
        shade.set_fill(GOOD, 0.07).set_stroke(width=0).move_to([(X_CUT + X_COL0 - 0.45) / 2, (Y_TOP + Y_BOT) / 2, 0])
        src_b = source_note("Report §9.2, Table 6; data/early_recipe.json (transition per run)")

        # ---- the verdict board (E1, E2 x machine)
        CELL_X = {"X": 0.25, "L": 3.95}
        Y_BHEAD, Y_E = -1.7, {"E1": -2.31, "E2": -2.93}
        b_rule = Line([-6.45, Y_BHEAD - 0.25, 0], [6.45, Y_BHEAD - 0.25, 0]).set_stroke(FAINT, 1.5)
        row_labels = {}
        for e, col, txt, sub in (("E1", HINGE_COLOR, "the early window works under Muon", "two streams, discovered"),
                                 ("E2", MUON_COLOR, "Muon binds four streams sooner", "four streams, bound by 4800")):
            lab = VGroup(chip(e, col, 20), VGroup(L(txt, 22, INK), L(sub, 20, MUTED))
                         .arrange(DOWN, buff=0.08, aligned_edge=LEFT)).arrange(RIGHT, buff=0.2)
            lab.move_to([-6.45, Y_E[e], 0], aligned_edge=LEFT)
            row_labels[e] = lab
        cells = {}
        for m in MACHINES:
            for e in ("E1", "E2"):
                c = claim_cell(m, e)
                c.move_to([CELL_X[m] - 0.6, Y_E[e], 0], aligned_edge=LEFT)
                cells[m, e] = c
        b_badges = {m: machine_badge(m, 22).move_to([cells[m, "E1"].get_center()[0], Y_BHEAD, 0]) for m in MACHINES}

        def col_hl(m, color):
            w = max(cells[m, e].get_width() for e in ("E1", "E2")) + 0.36
            top, bot = Y_BHEAD + 0.25, Y_E["E2"] - 0.37
            r = Rectangle(width=w, height=top - bot).set_fill(color, 0.09).set_stroke(color, 1.5, opacity=0.7)
            r.move_to([cells[m, "E1"].get_left()[0] + w / 2 - 0.18, (top + bot) / 2, 0])
            return r
        src_c = source_note("Report §9.2, Table 6: E1, E2, exact one-sided McNemar on paired seeds")

        # transition: E1 header -> E2 header, Part A board away
        keep_ids = {id(self.title_mob), id(self.frame)} | {id(x) for x in head_a}
        self.play(*[FadeOut(mob) for mob in self.mobjects if id(mob) not in keep_ids],
                  FadeTransform(head_a[0], head[0]), FadeTransform(head_a[1], head[1]),
                  FadeTransform(head_a[2], head[2]), run_time=0.9)

        with self.voiceover(
            "At four streams with sixteen channels, it compared the window under Adam with the window under Muon, "
            "scored by whether a run had bound by update forty-eight hundred. On L both claims were shown: the "
            "early window works under Muon, six to zero, and Muon binds four streams sooner, seven to zero. On X "
            "both pointed the same way without reaching significance: four to zero, and five to one. Under Muon the "
            "four-stream binders bound at a median of update twenty-four hundred on both machines, against "
            "forty-eight hundred under Adam."
        ) as vo:
            self.play(FadeIn(badges), FadeIn(guides), FadeIn(ticks), FadeIn(tick_head), FadeIn(brk), FadeIn(src_b),
                      run_time=0.8)
            vo.wait_until(at_phrase(vo, 0, "the window under Adam"))
            self.play(*[FadeIn(opt_labs[m, "WIN16_A"], shift=0.15 * RIGHT) for m in MACHINES], run_time=0.6)
            vo.wait_until(at_phrase(vo, 0, "the window under Muon"))
            self.play(*[FadeIn(opt_labs[m, "WIN16_M"], shift=0.15 * RIGHT) for m in MACHINES], run_time=0.6)
            # sweep through training: each run lights up at the evaluation where it first counted as bound
            for d in dots:
                d.set_fill(opacity=0).set_stroke(opacity=0)
                d.add_updater(reveal)
            self.add(dots, cursor)
            self.play(cursor_x.animate.set_value(X_NEVER + 0.5), FadeIn(dot_key),
                      run_time=3.2, rate_func=linear)
            for d in dots:
                d.clear_updaters()
            cursor.clear_updaters()
            self.play(FadeOut(cursor), FadeIn(col_heads[0]),
                      *[FadeIn(bound_vals[k], shift=0.1 * LEFT) for k in bound_vals], run_time=0.6)
            vo.wait_until(at_phrase(vo, 0, "scored by"))
            late = VGroup(*[d for d in dots if d.t is not None and d.t > 4800])
            self.play(ShowCreation(cut), FadeIn(cut_lab), FadeIn(shade), late.animate.set_opacity(0.3),
                      run_time=0.9)
            self.play(FadeIn(col_heads[1]), *[FadeIn(by_vals[k], shift=0.1 * LEFT) for k in by_vals], run_time=0.7)

            # On L both claims were shown
            vo.wait_until_sentence(1)
            self.play(FadeIn(b_rule), *[FadeIn(b_badges[m]) for m in MACHINES],
                      *[FadeIn(row_labels[e], shift=0.15 * UP) for e in row_labels], FadeTransform(src_b, src_c),
                      run_time=0.9)
            lbox = col_hl("L", MACHINE_COLORS["L"])
            self.play(FadeIn(lbox), b_badges["L"].animate(rate_func=there_and_back).scale(1.35), run_time=0.7)
            vo.wait_until(at_phrase(vo, 1, "six to zero") - 0.2)
            self.play(FadeIn(cells["L", "E1"], shift=0.15 * LEFT), Indicate(row_labels["E1"][1][0], color=HINGE_COLOR),
                      run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "seven to zero") - 0.2)
            self.play(FadeIn(cells["L", "E2"], shift=0.15 * LEFT),
                      Indicate(VGroup(by_vals["L", "WIN16_A"], by_vals["L", "WIN16_M"]), color=GOOD),
                      run_time=0.8)

            # On X: same direction, not significant
            vo.wait_until_sentence(2)
            xbox = col_hl("X", MACHINE_COLORS["X"])
            self.play(FadeOut(lbox), FadeIn(xbox), b_badges["X"].animate(rate_func=there_and_back).scale(1.35),
                      run_time=0.7)
            vo.wait_until(at_phrase(vo, 2, "four to zero") - 0.2)
            self.play(FadeIn(cells["X", "E1"], shift=0.15 * LEFT), run_time=0.7)
            vo.wait_until(at_phrase(vo, 2, "five to one") - 0.2)
            self.play(FadeIn(cells["X", "E2"], shift=0.15 * LEFT),
                      Indicate(VGroup(by_vals["X", "WIN16_A"], by_vals["X", "WIN16_M"]), color=GOOD),
                      run_time=0.7)

            # medians: Muon 2400 on both machines, Adam 4800
            vo.wait_until_sentence(3)
            med_boxes = {}
            for (m, a), y in ROWS.items():
                cx = COL_X[MEDIAN[m, a]]
                med_boxes[m, a] = RoundedRectangle(width=0.66, height=0.54, corner_radius=0.08).move_to([cx, y, 0])
                med_boxes[m, a].set_stroke(OPT_COLOR[a], 2.5).set_fill(opacity=0)
            self.play(FadeOut(xbox), *[ShowCreation(med_boxes[m, "WIN16_M"]) for m in MACHINES], run_time=0.9)
            vo.wait_until(at_phrase(vo, 3, "twenty-four hundred") - 0.2)
            self.play(FadeIn(col_heads[2]),
                      *[FadeIn(med_vals[m, "WIN16_M"], shift=0.15 * LEFT) for m in MACHINES], run_time=0.8)
            vo.wait_until(at_phrase(vo, 3, "on both machines"))
            self.play(*[Indicate(med_vals[m, "WIN16_M"], color=MUON_COLOR, scale_factor=1.25) for m in MACHINES],
                      *[badges[i].animate(rate_func=there_and_back).scale(1.3) for i in range(len(MACHINES))],
                      run_time=0.8)
            vo.wait_until(at_phrase(vo, 3, "against forty-eight") - 0.2)
            self.play(*[ShowCreation(med_boxes[m, "WIN16_A"]) for m in MACHINES], run_time=0.6)
            self.play(*[FadeIn(med_vals[m, "WIN16_A"], shift=0.15 * LEFT) for m in MACHINES], run_time=0.7)

        self.part_b_mobs = dict(cells=cells, board=[b_rule, *b_badges.values(), *row_labels.values(), src_c],
                                warn_hl=col_hl("X", WARN))

    # ============================================================== N17.3a  why X fell short
    def why_x(self):
        cells = self.part_b_mobs["cells"]
        LX, RX = -3.4, 3.35
        e1x, e2x = cells["X", "E1"], cells["X", "E2"]

        def slot(tag, color, cell, x):
            lab = VGroup(chip(tag, color, 20), L("on X", 22, MUTED)).arrange(RIGHT, buff=0.15)
            tgt = cell.copy()
            g = VGroup(lab, tgt).arrange(RIGHT, buff=0.35).move_to([x, 2.15, 0])
            return lab, tgt

        lab1, tgt1 = slot("E1", HINGE_COLOR, e1x, LX)
        lab2, tgt2 = slot("E2", MUON_COLOR, e2x, RX)

        def reason(num, text, x):
            n = VGroup(Circle(radius=0.22).set_fill(WARN, 0.2).set_stroke(WARN, 2), L(num, 22, WARN, weight="BOLD"))
            t = T(text, 32)
            g = VGroup(n, t).arrange(RIGHT, buff=0.22).move_to([x, 1.2, 0])
            return n, t

        n1, r1 = reason("1", "less room", LX)
        n2, r2 = reason("2", "fewer seeds", RX)
        divider = DashedLine([0, 2.5, 0], [0, -3.3, 0], dash_length=0.1).set_stroke(FAINT, 1.5)

        # ---- left: baseline bars (sorted: discoveries first, failures last), 40 seeds each
        BW, BH, BX0 = 4.6, 0.36, -6.1

        def base_bar(label, num, den, y):
            lab = L(label, 20, MUTED).move_to([BX0, y + 0.42, 0], aligned_edge=LEFT)
            track = Rectangle(width=BW, height=BH).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1)
            track.move_to([BX0 + BW / 2, y, 0])
            fill = Rectangle(width=BW * num / den, height=BH).set_fill(GOOD, 0.85).set_stroke(width=0)
            fill.align_to(track, LEFT).set_y(y)
            sep = VGroup(*[Line([BX0 + BW * i / den, y - BH / 2, 0], [BX0 + BW * i / den, y + BH / 2, 0])
                           .set_stroke(BG, 1.2) for i in range(1, den)])
            val = L(f"{num}/{den}", 24, INK, weight="BOLD").next_to(track, RIGHT, buff=0.2)
            room = Rectangle(width=BW * (den - num) / den, height=BH + 0.1).set_stroke(HINGE_COLOR, 2.5).set_fill(opacity=0)
            room.align_to(track, RIGHT).set_y(y)
            room_lab = L(f"{den - num} left to rescue", 20, HINGE_COLOR).next_to(room, DOWN, buff=0.1)
            room_lab.align_to(track, RIGHT)
            g = VGroup(lab, track, fill, sep, val)
            g.room, g.room_lab, g.fill, g.track = room, room_lab, fill, track
            return g

        n_x = COUNTS["X"]["SLOW_M"]
        fresh = base_bar("fresh seeds 300–339 on X, slow memory alone", n_x["success"], n_x["n"], 0.25)
        screen = base_bar("screen S32 (seeds 160–199), slow memory alone", *SCREEN_S32, -1.15)
        best = VGroup(L("even 4 vs 0, the best X could get, gives", 20, MUTED),
                      M(R"p = \left(\tfrac{1}{2}\right)^{4} = 0.0625 > 0.05", 32, WARN)).arrange(DOWN, buff=0.18)
        best.move_to([LX - 0.2, -2.75, 0])

        # ---- right: the four-stream seeds each machine ran
        CELL, PITCH, SX0 = 0.2, 0.245, 0.95
        Y_L, Y_X = -0.2, -1.25
        seeds_l = VGroup(*[Square(CELL).set_fill(MUTED, 0.6).set_stroke(width=0).move_to([SX0 + i * PITCH, Y_L, 0])
                           for i in range(20)])
        seeds_x = VGroup(*[Square(CELL).set_fill(MUTED, 0.6).set_stroke(width=0).move_to([SX0 + i * PITCH, Y_X, 0])
                           for i in range(20)])
        bl = machine_badge("L", 22).next_to(seeds_l, LEFT, buff=0.18)
        bx = machine_badge("X", 22).next_to(seeds_x, LEFT, buff=0.18)
        cnt_l = L("20", 24, INK, weight="BOLD").next_to(seeds_l, RIGHT, buff=0.2)
        cnt_x = L("12", 24, WARN, weight="BOLD").next_to(seeds_x, RIGHT, buff=0.2)
        cnt_x0 = L("20", 24, INK, weight="BOLD").next_to(seeds_x, RIGHT, buff=0.2)
        seed_head = L("four-stream seeds 340–359", 20, MUTED).next_to(seeds_l, UP, buff=0.22).align_to(seeds_l, LEFT)
        cut_part = VGroup(*seeds_x[12:])
        crosses = VGroup(*[cross_over(s, BAD, 2) for s in cut_part])
        cut_lab = L("cut by X's time rule", 20, WARN).next_to(cut_part, DOWN, buff=0.18)
        src = source_note("Report §9.1 (S32), §9.2; p for b vs 0 is (1/2)^b (exact McNemar, one-sided)")

        # clear the run plot first, keeping the verdict board
        board = self.part_b_mobs["board"]
        keep_ids = {id(cells[m, e]) for m in MACHINES for e in ("E1", "E2")} | {id(b) for b in board}
        keep_ids |= {id(self.title_mob), id(self.frame)}
        self.play(*[FadeOut(mob) for mob in self.mobjects if id(mob) not in keep_ids], run_time=0.6)
        xbox = self.part_b_mobs["warn_hl"]
        q = T("Why did X fall short?", 40, WARN).move_to([xbox.get_center()[0], -0.55, 0])

        with self.voiceover(
            "Why did X fall short? Two things weakened its run. On fresh seeds, slow memory alone already "
            "discovered thirty-six of forty on X, against thirty-one in the screens, leaving less room; and X's "
            "time rule cut the four-stream part to twelve seeds."
        ) as vo:
            # Why did X fall short?  (the board stays; the X column is the question)
            self.play(FadeIn(xbox), FadeIn(q, shift=0.2 * UP),
                      cells["L", "E1"].animate.set_opacity(0.35), cells["L", "E2"].animate.set_opacity(0.35),
                      run_time=0.8)

            # Two things weakened its run.
            vo.wait_until_sentence(1)
            self.play(*[FadeOut(mob) for mob in board], FadeOut(cells["L", "E1"]), FadeOut(cells["L", "E2"]),
                      FadeOut(xbox), FadeOut(q),
                      ReplacementTransform(e1x, tgt1), ReplacementTransform(e2x, tgt2),
                      FadeIn(lab1), FadeIn(lab2), ShowCreation(divider), run_time=1.0)
            self.play(FadeIn(n1, scale=0.6), FadeIn(n2, scale=0.6), FadeIn(src), run_time=0.5)

            # less room
            vo.wait_until(at_phrase(vo, 2, "slow memory alone") - 0.2)
            self.play(Write(r1), FadeIn(VGroup(fresh[0], fresh.track, fresh[3])), run_time=0.8)
            fresh.fill.save_state()
            fresh.fill.stretch(1e-3, 0, about_edge=LEFT)
            self.play(Restore(fresh.fill), FadeIn(fresh[4]), run_time=1.0)
            vo.wait_until(at_phrase(vo, 2, "against thirty-one") - 0.2)
            self.play(FadeIn(VGroup(screen[0], screen.track, screen[3])), run_time=0.5)
            screen.fill.save_state()
            screen.fill.stretch(1e-3, 0, about_edge=LEFT)
            self.play(Restore(screen.fill), FadeIn(screen[4]), run_time=0.9)
            vo.wait_until(at_phrase(vo, 2, "leaving less room") - 0.3)
            self.play(ShowCreation(screen.room), FadeIn(screen.room_lab), run_time=0.6)
            self.play(ShowCreation(fresh.room), FadeIn(fresh.room_lab), run_time=0.6)
            self.play(FadeIn(best, shift=0.15 * UP), Indicate(tgt1, color=WARN, scale_factor=1.06), run_time=1.0)

            # fewer seeds
            vo.wait_until(at_phrase(vo, 2, "and X's time rule") - 0.1)
            self.play(Write(r2), FadeIn(seed_head), FadeIn(bl), FadeIn(bx),
                      LaggedStartMap(FadeIn, seeds_l, lag_ratio=0.03), LaggedStartMap(FadeIn, seeds_x, lag_ratio=0.03),
                      FadeIn(cnt_l), FadeIn(cnt_x0), run_time=1.0)
            self.play(LaggedStartMap(ShowCreation, crosses, lag_ratio=0.08), cut_part.animate.set_opacity(0.15),
                      FadeIn(cut_lab, shift=0.1 * UP), FadeTransform(cnt_x0, cnt_x), run_time=1.1)
            self.play(Indicate(tgt2, color=WARN, scale_factor=1.06), run_time=0.8)

    # ============================================================== N17.3b  the machines are not independent
    def not_independent(self):
        self.fade_all_but_title(run_time=0.7)

        q = T("two independent replicates?", 34).move_to([0, 2.3, 0])
        strike = Line(q.get_left() + 0.1 * LEFT, q.get_right() + 0.1 * RIGHT).set_stroke(BAD, 4)
        q2 = T("the same seeds on both machines", 34, color=WARN).move_to(q)
        mcards = {}
        for m, desc, x in (("X", "a cloud container, Intel Xeon", -3.3), ("L", "an Intel i7-12650H", 3.3)):
            content = VGroup(machine_badge(m, 34), L(desc, 22, MUTED)).arrange(DOWN, buff=0.25)
            c = card(content, buff=0.32)
            c.move_to([x, 0.3, 0])
            mcards[m] = c
        seed_chip = chip("seeds 300–339", INK, 24, fill=0.08).move_to([0, 0.3, 0])
        seed_arrows = VGroup(Arrow(seed_chip.get_left(), mcards["X"].get_right(), buff=0.15, thickness=3),
                             Arrow(seed_chip.get_right(), mcards["L"].get_left(), buff=0.15, thickness=3)).set_color(INK)

        CELL, PITCH, X0 = 0.19, 0.245, -5.55
        Y_X, Y_L = 1.35, 0.8
        bx = machine_badge("X", 24).move_to([-6.25, Y_X, 0])
        bl = machine_badge("L", 24).move_to([-6.25, Y_L, 0])
        ex = empty_strip(40, X0, Y_X, CELL, PITCH)
        el = empty_strip(40, X0, Y_L, CELL, PITCH)
        sx = strip(PART_A["X"]["SLOW_M"], X0, Y_X, CELL, PITCH)
        sl = strip(PART_A["L"]["SLOW_M"], X0, Y_L, CELL, PITCH)
        seed_a = L("seed 300", 20, MUTED).next_to(el[0], DOWN, buff=0.14)
        seed_b = L("seed 339", 20, MUTED).next_to(el[-1], DOWN, buff=0.14)
        arm_lab = L("Muon, slow memory alone", 20, SLOW_COLOR).next_to(ex, UP, buff=0.14).align_to(ex, LEFT)
        mism = VGroup(*[RoundedRectangle(width=PITCH + 0.03, height=Y_X - Y_L + CELL + 0.14, corner_radius=0.05)
                        .set_stroke(WARN, 2.5).move_to([X0 + i * PITCH, (Y_X + Y_L) / 2, 0])
                        for i in range(40) if not AGREE[i]])
        agree = VGroup(L(f"{sum(AGREE)} of 40", 26, INK, weight="BOLD"), L("seeds agree", 20, MUTED))
        agree.arrange(DOWN, buff=0.08).move_to([5.55, (Y_X + Y_L) / 2, 0])
        sla = _ER["seed_level_agreement_X_vs_L"]
        agree_more = L(f"also: with the window {sla['WIN_M']['same_outcome']} of {sla['WIN_M']['n']} · four streams, "
                       f"Adam {sla['WIN16_A']['same_outcome']} of {sla['WIN16_A']['n']}, "
                       f"Muon {sla['WIN16_M']['same_outcome']} of {sla['WIN16_M']['n']}", 20, MUTED)
        agree_more.next_to(seed_b, DOWN, buff=0.12).align_to(ex[-1], RIGHT).shift(0.1 * RIGHT)

        # left card: same seed, different digits
        # seed 300 (index 0): discovered on both machines
        pick_box = RoundedRectangle(width=PITCH + 0.03, height=Y_X - Y_L + CELL + 0.14, corner_radius=0.05)
        pick_box.set_stroke(INK, 2.5).move_to([X0, (Y_X + Y_L) / 2, 0])
        c1_head = L("seed 300: accuracy at update 1200", 22, MUTED)
        vx = VGroup(machine_badge("X", 22), L(f"{ACC300['X']:.4f}", 30, INK, weight="BOLD")).arrange(RIGHT, buff=0.15)
        vl = VGroup(machine_badge("L", 22), L(f"{ACC300['L']:.4f}", 30, INK, weight="BOLD")).arrange(RIGHT, buff=0.15)
        neq = M(R"\neq", 44, BAD)
        vals = VGroup(vx, neq, vl).arrange(RIGHT, buff=0.3)
        c1_foot = L("no run reproduced bit for bit across machines,", 20, INK)
        c1_foot2 = L("in any arm, Adam's included", 20, INK)
        c1 = VGroup(c1_head, vals, VGroup(c1_foot, c1_foot2).arrange(DOWN, buff=0.08)).arrange(DOWN, buff=0.28)
        c1_card = card(c1, buff=0.28)
        c1_card.move_to([-3.35, -1.75, 0])
        link = Arrow(pick_box.get_bottom(), c1_card.get_top() + 1.2 * LEFT, buff=0.08, thickness=2.5).set_color(INK)

        # right card: X's two hosts
        c2_head = L("X's two Xeon hosts", 22, MUTED)
        h1 = chip("2.10 GHz", MACHINE_COLORS["X"], 22)
        h2 = chip("2.80 GHz", MACHINE_COLORS["X"], 22)
        hosts = VGroup(h1, L("vs", 22, MUTED), h2).arrange(RIGHT, buff=0.3)
        r_adam = VGroup(L("Adam", 22, ADAM_COLOR, weight="BOLD"), M(R"=", 34, GOOD),
                        L("same bits", 22, GOOD)).arrange(RIGHT, buff=0.2)
        r_muon = VGroup(L("Muon", 22, MUON_COLOR, weight="BOLD"), M(R"\neq", 34, BAD),
                        L("bits differ", 22, BAD)).arrange(RIGHT, buff=0.2)
        bf16 = chip("Newton–Schulz in bf16, low precision", MUON_COLOR, 20)
        rows = VGroup(r_adam, r_muon).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        c2 = VGroup(c2_head, hosts, rows, bf16).arrange(DOWN, buff=0.25)
        c2_card = card(c2, buff=0.28)
        c2_card.move_to([3.4, -1.75, 0])
        src = source_note("Report p. 1 & Table 9 (hosts), §9.2, §15; data/early_recipe.json (SLOW_M per seed, seed 300)")

        with self.voiceover(
            "And the two machines are not independent replicates. They ran the same seeds, so most seed-level "
            "outcomes agree, yet no run reproduced bit for bit across machines, and Muon's low-precision "
            "Newton–Schulz step even differs between X's two Xeon hosts."
        ) as vo:
            self.play(FadeIn(mcards["X"], shift=0.3 * RIGHT), FadeIn(mcards["L"], shift=0.3 * LEFT), Write(q),
                      run_time=1.1)
            vo.wait_until(at_phrase(vo, 0, "not independent"))
            self.play(ShowCreation(strike), q.animate.set_opacity(0.45), run_time=0.6)

            # same seeds
            vo.wait_until_sentence(1)
            self.play(LaggedStart(FadeOut(VGroup(q, strike), shift=0.3 * UP), FadeIn(q2, shift=0.3 * UP),
                                  lag_ratio=0.65), FadeIn(seed_chip, scale=0.8), run_time=0.9)
            self.play(*[GrowArrow(a) for a in seed_arrows], run_time=0.6)
            self.play(ReplacementTransform(mcards["X"][1][0], bx), ReplacementTransform(mcards["L"][1][0], bl),
                      FadeOut(VGroup(mcards["X"][0], mcards["X"][1][1], mcards["L"][0], mcards["L"][1][1], seed_arrows)),
                      FadeTransform(seed_chip, VGroup(seed_a, seed_b)), FadeIn(ex), FadeIn(el), FadeIn(arm_lab),
                      FadeIn(src), run_time=0.9)
            vo.wait_until(at_phrase(vo, 1, "so most seed-level") - 0.2)
            self.play(LaggedStart(*[FadeIn(VGroup(a, b)) for a, b in zip(sx, sl)], lag_ratio=0.03), run_time=1.2)
            self.play(LaggedStartMap(ShowCreation, mism, lag_ratio=0.3), FadeIn(agree, shift=0.1 * LEFT),
                      FadeIn(agree_more), run_time=0.8)

            # ... yet not bit for bit
            vo.wait_until(at_phrase(vo, 1, "yet no run") - 0.2)
            self.play(ShowCreation(pick_box), FadeOut(seed_a), GrowArrow(link), FadeIn(c1_card, shift=0.2 * UP),
                      run_time=1.0)
            self.play(Indicate(neq, scale_factor=1.4, color=BAD), run_time=0.8)

            # ... and Muon's bf16 Newton-Schulz differs even between X's own hosts
            vo.wait_until(at_phrase(vo, 1, "and Muon's") - 0.2)
            self.play(FadeIn(VGroup(c2_card[0], c2_head, hosts), shift=0.2 * UP), run_time=0.8)
            self.play(FadeIn(r_adam, shift=0.1 * RIGHT), run_time=0.6)
            vo.wait_until(at_phrase(vo, 1, "low-precision"))
            self.play(FadeIn(bf16, shift=0.1 * UP), FadeIn(r_muon, shift=0.1 * RIGHT), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "even differs"))
            self.play(Indicate(r_muon, color=BAD, scale_factor=1.08), run_time=0.9)
