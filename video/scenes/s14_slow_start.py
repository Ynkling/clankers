"""Chapter 14 — slow memory and a hinge, confirmed (report §7, Table 4; §2 Phase IV for the restart rule;
video/data/slow_start.json for per-seed outcomes, video/data/hinge_firings.json for the hinge's firings,
video/data/report_tables.json for Table 4's counts and paired tests)."""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load(name):
    with open(os.path.join(DATA, name)) as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------------- real data
_SS = _load("slow_start.json")
_HF = _load("hinge_firings.json")
T4 = _load("report_tables.json")["table4_slow_start"]
ROWS4 = {r["arm"]: r for r in T4["rows"]}
CLAIMS4 = T4["claims"]

ARMS0 = ["A0", "SLOW0", "HINGE0"]
ARM_NAMES = {"A0": "plain gate", "SLOW0": "slow memory", "HINGE0": "slow + hinge"}
SEEDS0 = _SS["seeds"]["HINGE0"]                      # 280..299


def _code(outcome):
    """One letter per seed: D discovered, P position split, K key split, O any other failure."""
    return {"DISCOVERED": "D", "POSITION": "P", "KEY": "K"}.get(outcome, "O")


# Part 0 per seed, X from results/X/slow_start_results.json, L from results/L/recipe_scope_L_final.log
PART0 = {m: {a: "".join(_code(o) for o in _SS["per_seed"][m][a]["outcome"]) for a in ARMS0} for m in ("X", "L")}


def _frac(m, arm):
    num, den = ROWS4[arm][m].split("/")
    return int(num), int(den)


def _pairs(m, new, old):
    a, b = PART0[m][new], PART0[m][old]
    return ([j for j in range(20) if a[j] == "D" and b[j] != "D"],
            [j for j in range(20) if a[j] != "D" and b[j] == "D"])


for _m in ("X", "L"):
    for _a in ARMS0:
        assert PART0[_m][_a].count("D") == _frac(_m, _a)[0], (_m, _a)
    for _claim, (_new, _old) in {"H0": ("HINGE0", "A0"), "S0": ("SLOW0", "A0"), "H0S": ("HINGE0", "SLOW0")}.items():
        _b, _c = _pairs(_m, _new, _old)
        assert (len(_b), len(_c)) == (CLAIMS4[_claim][_m]["b"], CLAIMS4[_claim][_m]["c"]), (_m, _claim)
# failure composition quoted in §7: A0 9 of 12 (X) / 8 of 12 (L) position; SLOW0 4 of 5 / 5 of 5
assert PART0["X"]["A0"].count("P") == 9 and PART0["L"]["A0"].count("P") == 8
assert PART0["X"]["SLOW0"].count("P") == 4 and PART0["L"]["SLOW0"].count("P") == 5
assert "P" not in PART0["X"]["HINGE0"] + PART0["L"]["HINGE0"]

# hinge firings per X HINGE0 run (sum of its 1200-update windows), seeds 280..299
_W = _HF["windows_X"]["HINGE0"]["windows"]
FIRINGS_X = [sum(_W[str(s)]) for s in SEEDS0]
assert sum(1 for f in FIRINGS_X if f == 0) == _HF["windows_X"]["HINGE0"]["never_fired"] == 11
assert FIRINGS_X[SEEDS0.index(287)] == 950

# Part C (HINGE D8, S = 8, k = 16): 0/10 bound, 4 collapsed on each machine (Table 4 caption)
assert ROWS4["HINGE_D8"]["X"] == ROWS4["HINGE_D8"]["L"] == "0/10"
assert _SS["counts"]["X"]["HINGE_D8"]["collapsed"] == 4

FAILN = "#3B414E"              # a failure before its type is shown
KEY_SPLIT = "#EC92AB"
OTHER_FAIL = "#7A808C"
OUT_COLORS = {"D": GOOD, "P": BAD, "K": KEY_SPLIT, "O": OTHER_FAIL}

# Part 0 strip geometry
SQ, PITCH = 0.28, 0.37
STRIP_X0 = -3.6
ROW_Y = {"X": [1.85, 1.49, 1.13], "L": [0.45, 0.09, -0.27]}
NAME_RX, CODE_LX, COUNT_LX, CHIP_LX = -4.6, -4.45, 3.9, 4.75


def col_x(j):
    return STRIP_X0 + SQ / 2 + j * PITCH


# ------------------------------------------------------------------------------------- helpers
def at_phrase(vo, i, phrase):
    """Approximate time (s into the block) at which `phrase` is spoken in sentence i."""
    text, a, b = vo.synth.sentences[i]
    k = text.find(phrase)
    if k < 0:
        raise ValueError(phrase)
    return a + (b - a) * k / max(len(text), 1)


def chip(text, color, size=22, padx=0.2, pady=0.12, fill=0.14, weight="BOLD"):
    t = L(text, size=size, color=color, weight=weight)
    box = RoundedRectangle(width=t.get_width() + 2 * padx, height=t.get_height() + 2 * pady, corner_radius=0.1)
    box.set_fill(color, fill).set_stroke(color, 1.6)
    t.move_to(box)
    return VGroup(box, t)


def legend_item(color, text, opacity=0.9, outline=False, size=20):
    sq = Square(0.24).set_fill(color, 0.0 if outline else opacity).set_stroke(color, 2.5 if outline else 0)
    return VGroup(sq, L(text, size=size, color=MUTED)).arrange(RIGHT, buff=0.12)


def fmt_p(p):
    return f"{float(p):.4f}" if "e" in str(p) else str(p)


def pair_text(b, c, p, color_counts=True, size=22):
    """'11 vs 0  ·  p = 0.0005' with the discordant counts coloured like the highlights."""
    t = L(f"{b} vs {c}  ·  p = {fmt_p(p)}", size=size, color=INK)
    nb, nc = len(str(b)), len(str(c))
    t[nb + 2 + nc:].set_color(MUTED)
    if color_counts:
        t[:nb].set_color(WARN)
        t[nb + 2:nb + 2 + nc].set_color(BAD)
    return t


def verdict_cell(claim, m, color_counts=True):
    c = CLAIMS4[claim][m]
    badge = verdict_badge(c["verdict"], size=20)
    return VGroup(badge, pair_text(c["b"], c["c"], c["p"], color_counts)).arrange(RIGHT, buff=0.18)


def cross_mark(color=BAD, s=0.2):
    return VGroup(Line([-s / 2, -s / 2, 0], [s / 2, s / 2, 0]), Line([-s / 2, s / 2, 0], [s / 2, -s / 2, 0])
                  ).set_stroke(color, 5)


def check_mark(color=GOOD, s=0.24):
    return VMobject().set_points_as_corners(
        [[-s / 2, 0, 0], [-s / 6, -s / 2.4, 0], [s / 2, s / 2.4, 0]]).set_stroke(color, 5)


class SlowStart(ClankersScene):
    def construct(self):
        card = self.chapter_card(14, "Slow memory and a hinge, confirmed")
        self.wait(0.6)
        title = section_title("The slow-start test")
        self.play(FadeOut(card, shift=0.3 * UP), FadeIn(title, shift=0.3 * UP))
        self.title_mob = title
        self.note = None

        self.why_restarts()      # N14.1
        self.part_zero()         # N14.2 + N14.3
        self.parts_a_b()         # N14.4
        self.failures()          # N14.5
        self.wait(1.0)
        self.clear_all()

    # ------------------------------------------------------------------ small utilities
    def swap_note(self, text):
        new = source_note(text)
        anim = FadeIn(new) if self.note is None else FadeTransform(self.note, new)
        self.note = new
        return anim

    def tab_anims(self, active):
        anims = []
        for i, t in enumerate(self.tabs):
            on = i in active
            c = WARN if on else PANEL_EDGE
            anims.append(t[0].animate.set_stroke(c, 2.0 if on else 1.2).set_fill(WARN if on else PANEL, 0.16 if on else 1))
            anims.append(t[1].animate.set_color(WARN if on else MUTED))
        return anims

    # ================================================================== N14.1  why drop restarts
    def why_restarts(self):
        x0, xc, xe = -5.0, -2.5, 1.6
        ys = [1.95, 1.3, 0.65]
        head = L("restart rule (Phase IV)", 24, weight="BOLD").move_to([-6.3, 2.55, 0], aligned_edge=LEFT)
        check_line = DashedLine([xc, 2.3, 0], [xc, 0.4, 0], dash_length=0.08).set_stroke(FAINT, 2)
        check_lab = L("check at 2400", 20, MUTED).next_to(check_line, UP, buff=0.08)
        tries, marks, extras = VGroup(), VGroup(), VGroup()
        for i, y in enumerate(ys):
            lab = L(f"try {i + 1}", 20, MUTED).move_to([x0 - 0.25, y, 0], aligned_edge=RIGHT)
            bar = Rectangle(width=xc - x0 - 0.05, height=0.24).set_fill(MUTED, 0.55).set_stroke(width=0)
            bar.move_to([x0, y, 0], aligned_edge=LEFT)
            tries.add(VGroup(lab, bar))
            if i < 2:
                mk = cross_mark().move_to([xc + 0.3, y, 0])
                ex = L("restart, new seed", 20, BAD).next_to(mk, RIGHT, buff=0.2)
            else:
                mk = check_mark().move_to([xc + 0.3, y, 0])
                ex = VGroup(Rectangle(width=xe - xc - 0.6, height=0.24).set_fill(GOOD, 0.75).set_stroke(width=0)
                            .move_to([xc + 0.6, y, 0], aligned_edge=LEFT))
                ex.add(L("discovered", 22, GOOD, weight="BOLD").next_to(ex[0], RIGHT, buff=0.15))
            marks.add(mk)
            extras.add(ex)
        stat = VGroup(L("60/60", 48, GOOD, weight="BOLD"), L("trials succeeded", 22),
                      L("two streams, both machines", 20, MUTED)).arrange(DOWN, buff=0.14)
        stat.move_to([4.75, 1.35, 0])

        cheap_brace = Brace(VGroup(*[t[1] for t in tries]), DOWN, buff=0.12)
        cheap_lab = L("cheap runs", 22, WARN, weight="BOLD").next_to(cheap_brace, DOWN, buff=0.1)
        check_tag = L("a reliable early check, for every new setting", 22, WARN, weight="BOLD")
        check_tag.move_to([1.25, cheap_lab.get_y(), 0])
        check_arrow = Arrow(check_tag.get_left() + 0.05 * LEFT, [xc + 0.08, 0.42, 0], buff=0.1, thickness=3)
        check_arrow.set_color(WARN)

        big_lab = L("a large model: one training run", 22).move_to([-5.2, -1.12, 0], aligned_edge=LEFT)
        big = Rectangle(width=11.0, height=0.42).set_fill(MUTED, 0.3).set_stroke(MUTED, 1.2)
        big.move_to([-5.2, -1.65, 0], aligned_edge=LEFT)
        recipe_seg = Rectangle(width=2.2, height=0.42).set_fill(SLOW_COLOR, 0.8).set_stroke(width=0)
        recipe_seg.move_to(big.get_left(), aligned_edge=LEFT)
        ticks = VGroup(*[Line(UP * 0.21, DOWN * 0.21).set_stroke(HINGE_COLOR, 5).move_to(recipe_seg.get_left() + RIGHT * dx)
                         for dx in (0.45, 0.8, 1.55)])
        recipe_lab = L("the recipe: slow memory + hinge, inside the run", 22, SLOW_COLOR)
        recipe_lab.next_to(big, DOWN, buff=0.18).align_to(big, LEFT)
        no_restart = chip("no restarts", GOOD, 22).next_to(big, DOWN, buff=0.14).align_to(big, RIGHT)

        n1 = ("Why does dropping restarts matter, when restarts already gave sixty of sixty at two streams? "
              "A restart rule needs cheap runs and a reliable early check for every new setting, while a large "
              "model is usually trained once, so it needs a recipe that works inside a single run. The slow start "
              "test took the recipe to fresh seeds and to the working configurations, with no restarts anywhere.")
        with self.voiceover(n1) as vo:
            self.play(FadeIn(head), ShowCreation(check_line), FadeIn(check_lab),
                      self.swap_note("Report §2 (Phase IV): the restart rule, 60/60 trials"), run_time=0.7)
            for i in range(3):
                self.play(FadeIn(tries[i][0]), GrowFromEdge(tries[i][1], LEFT), run_time=0.5)
                self.play(FadeIn(marks[i], scale=0.6), FadeIn(extras[i], shift=0.1 * RIGHT) if i < 2
                          else GrowFromEdge(extras[i][0], LEFT), run_time=0.45)
            self.play(FadeIn(extras[2][1]), FadeIn(stat, shift=0.2 * UP), run_time=0.7)
            self.play(Indicate(stat[0], color=GOOD, scale_factor=1.12), run_time=0.8)

            vo.wait_until(at_phrase(vo, 1, "cheap runs"))
            self.play(GrowFromCenter(cheap_brace), FadeIn(cheap_lab, shift=0.1 * DOWN),
                      *[Indicate(t[1], color=WARN, scale_factor=1.04) for t in tries], run_time=0.9)
            vo.wait_until(at_phrase(vo, 1, "a reliable early check"))
            self.play(FadeIn(check_tag, shift=0.1 * LEFT), GrowArrow(check_arrow),
                      check_line.animate.set_stroke(WARN, 3), run_time=0.9)
            vo.wait_until(at_phrase(vo, 1, "while a large"))
            restart_part = VGroup(head, check_line, check_lab, tries, marks, extras, stat, cheap_brace, cheap_lab,
                                  check_tag, check_arrow)
            self.play(restart_part.animate.set_opacity(0.3), FadeIn(big_lab), run_time=0.6)
            self.play(GrowFromEdge(big, LEFT), run_time=1.4)
            vo.wait_until(at_phrase(vo, 1, "so it needs a recipe"))
            self.play(FadeIn(recipe_seg), LaggedStartMap(FadeIn, ticks, lag_ratio=0.3), run_time=0.8)
            self.play(FadeIn(recipe_lab, shift=0.1 * UP), FadeIn(no_restart, shift=0.1 * UP), run_time=0.8)

            # ---------------- the test's layout: Table 4's four parts
            vo.wait_until_sentence(2)
            specs = [
                ("Part 0", "two streams, four keys", "S = 2, P = 4, k = 2", "fresh seeds 280–299",
                 "A0 · SLOW0 · HINGE0", "scored: discovered"),
                ("Part A", "eight keys, convolution", "S = 2, P = 8, k = 2", "seeds 220–239",
                 "DIRECT8* · HINGE8", "scored: bound"),
                ("Part B", "four streams, 16 channels", "S = 4, P = 4, k = 16", "seeds 240–259",
                 "A4k16* · HINGE4k16", "scored: bound"),
                ("Part C", "eight streams, 16 channels", "S = 8, P = 4, k = 16", "seeds 260–269",
                 "D8* · HINGE D8", "descriptive"),
            ]
            cards = VGroup()
            for name, desc, conf, seeds, arms, scored in specs:
                body = VGroup(L(name, 26, weight="BOLD"), L(desc, 22), L(conf, 20, MUTED), L(seeds, 20, MUTED),
                              L(arms, 20), L(scored, 20, MUTED)).arrange(DOWN, buff=0.17)
                box = RoundedRectangle(width=3.1, height=body.get_height() + 0.55, corner_radius=0.12)
                box.set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1.5)
                body.move_to(box)
                c = VGroup(box, body)
                c.box, c.body = box, body
                cards.add(c)
            cards.arrange(RIGHT, buff=0.17).move_to(0.45 * UP)
            work_brace = Brace(VGroup(cards[1], cards[2]), UP, buff=0.12)
            work_lab = L("the working configurations", 22, WARN, weight="BOLD").next_to(work_brace, UP, buff=0.1)
            no_any = chip("no restarts in any arm", GOOD, 24).next_to(cards, DOWN, buff=0.4)
            foot = L("* recorded runs of the plain gate from earlier tests, on the same seeds (X's D8: 260–265 only)", 20, MUTED)
            foot.next_to(no_any, DOWN, buff=0.3)

            self.play(FadeOut(VGroup(restart_part, big_lab, big, recipe_seg, ticks, recipe_lab)),
                      FadeOut(no_restart), run_time=0.6)
            self.play(LaggedStartMap(FadeIn, cards, shift=0.2 * UP, lag_ratio=0.2),
                      self.swap_note("Report §7, Table 4"), run_time=1.3)
            vo.wait_until(at_phrase(vo, 2, "fresh seeds"))
            self.play(cards[0].body[3].animate.set_color(WARN), cards[0].box.animate.set_stroke(WARN, 2),
                      run_time=0.6)
            vo.wait_until(at_phrase(vo, 2, "working configurations"))
            self.play(GrowFromCenter(work_brace), FadeIn(work_lab, shift=0.1 * UP),
                      cards[1].box.animate.set_stroke(WARN, 2), cards[2].box.animate.set_stroke(WARN, 2),
                      FadeIn(foot), run_time=0.8)
            vo.wait_until(at_phrase(vo, 2, "with no restarts"))
            self.play(FadeIn(no_any, scale=0.8), run_time=0.6)

        # cards shrink into a row of tabs, top right, that marks the part being shown
        tabs = VGroup()
        for name, *_ in specs:
            t = L(name, 20, MUTED)
            box = RoundedRectangle(width=1.1, height=0.4, corner_radius=0.1).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1.2)
            t.move_to(box)
            tabs.add(VGroup(box, t))
        tabs.arrange(RIGHT, buff=0.12)
        tabs.move_to([6.6, 3.15, 0], aligned_edge=RIGHT)
        self.tabs = tabs
        self.play(*[FadeOut(cards[i].body[1:]) for i in range(4)],
                  FadeOut(VGroup(work_brace, work_lab, no_any, foot)), run_time=0.45)
        self.play(*[ReplacementTransform(cards[i].box, tabs[i][0]) for i in range(4)],
                  *[ReplacementTransform(cards[i].body[0], tabs[i][1]) for i in range(4)], run_time=0.85)
        self.add(tabs)   # regroup the transformed pieces under one mobject

    # ================================================================== N14.2 + N14.3  Part 0
    def build_block(self, m):
        y = ROW_Y[m]
        blk = SimpleNamespace(badge=machine_badge(m, 28).move_to([-6.35, y[1], 0]), rows=[])
        for r, arm in enumerate(ARMS0):
            name = L(ARM_NAMES[arm], 22).move_to([NAME_RX, y[r], 0], aligned_edge=RIGHT)
            code = L(arm, 20, MUTED).move_to([CODE_LX, y[r], 0], aligned_edge=LEFT)
            sqs = VGroup(*[Square(SQ).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1).move_to([col_x(j), y[r], 0])
                           for j in range(20)])
            n, den = _frac(m, arm)
            cnt = L(f"{n}/{den}", 24, GOOD, weight="BOLD").move_to([COUNT_LX, y[r], 0], aligned_edge=LEFT)
            ov = Rectangle(width=COUNT_LX + 0.7 - (NAME_RX - 1.35), height=0.35).set_fill(BG, 0.78).set_stroke(width=0)
            ov.move_to([(COUNT_LX + 0.7 + NAME_RX - 1.35) / 2, y[r], 0])
            blk.rows.append(SimpleNamespace(arm=arm, name=name, code=code, sqs=sqs, cnt=cnt, ov=ov,
                                            codes=PART0[m][arm], y=y[r]))
        blk.frame = VGroup(blk.badge, *[VGroup(r.name, r.code, r.sqs) for r in blk.rows])
        return blk

    def fill_row(self, row, typed=False, run_time=1.2):
        anims = []
        for j, sq in enumerate(row.sqs):
            c = row.codes[j]
            if c == "D":
                color = GOOD
            else:
                color = OUT_COLORS[c] if typed else FAILN
            anims.append(sq.animate.set_fill(color, 0.9).set_stroke(color, 1))
        return LaggedStart(*anims, lag_ratio=0.04, run_time=run_time)

    def disc_rects(self, m, new, old):
        rows = {r.arm: r for r in self.blocks[m].rows}
        ya, yb = rows[new].y, rows[old].y
        top, bot = max(ya, yb) + SQ / 2 + 0.05, min(ya, yb) - SQ / 2 - 0.05
        wins, losses = _pairs(m, new, old)
        g = VGroup()
        for js, color in ((wins, WARN), (losses, BAD)):
            for j in js:
                r = RoundedRectangle(width=SQ + 0.06, height=top - bot, corner_radius=0.05)
                r.set_stroke(color, 2.5).set_fill(color, 0.06).move_to([col_x(j), (top + bot) / 2, 0])
                g.add(r)
        return g

    def part_zero(self):
        self.blocks = {m: self.build_block(m) for m in ("X", "L")}
        B = self.blocks
        header = L("one square per seed, 280–299", 20, MUTED).move_to([STRIP_X0, 2.5, 0], aligned_edge=LEFT)
        legend = VGroup(legend_item(GOOD, "discovered"), legend_item(FAILN, "not discovered", opacity=1)
                        ).arrange(RIGHT, buff=0.35)
        legend.move_to([6.6, 2.5, 0], aligned_edge=RIGHT)
        self.p0_header, self.p0_legend = header, legend

        self.play(*self.tab_anims([0]),
                  LaggedStart(*[FadeIn(B[m].frame, shift=0.15 * RIGHT) for m in ("X", "L")], lag_ratio=0.3),
                  FadeIn(header), FadeIn(legend),
                  self.swap_note("Report §7, Table 4; per seed: data/slow_start.json (L from its log)"), run_time=1.2)

        n2 = ("At two streams and four keys, the plain gate discovered the routing on eight of twenty seeds on "
              "each machine. Slow memory alone, fifteen of twenty on each. Slow memory plus the hinge, nineteen "
              "of twenty on X, twenty of twenty on L.")
        with self.voiceover(n2) as vo:
            for r in range(3):
                if r == 0:
                    pass
                elif r == 1:
                    vo.wait_until_sentence(1)
                else:
                    vo.wait_until_sentence(2)
                rows = [B[m].rows[r] for m in ("X", "L")]
                self.play(*[Indicate(rw.name, color=WARN, scale_factor=1.12) for rw in rows], run_time=0.6)
                if r < 2:
                    self.play(*[self.fill_row(rw) for rw in rows])
                    if r == 0:
                        vo.wait_until(at_phrase(vo, 0, "eight of twenty"))
                    self.play(*[FadeIn(rw.cnt, shift=0.1 * LEFT) for rw in rows], run_time=0.5)
                else:
                    self.play(self.fill_row(rows[0]))
                    self.play(FadeIn(rows[0].cnt, shift=0.1 * LEFT), run_time=0.4)
                    vo.wait_until(at_phrase(vo, 2, "twenty of twenty on L"))
                    self.play(self.fill_row(rows[1]))
                    self.play(FadeIn(rows[1].cnt, shift=0.1 * LEFT), run_time=0.4)

        # ---------------- N14.3 paired tests on discordant seeds
        TX0, CW = -6.2, [3.9, 3.75, 3.75]
        xs = [TX0, TX0 + CW[0], TX0 + CW[0] + CW[1], TX0 + sum(CW)]
        ys = {"head": -1.0, "H0": -1.5, "S0": -2.0, "H0S": -2.5}
        rules = VGroup(Line([xs[0], -0.75, 0], [xs[3], -0.75, 0]).set_stroke(INK, 2),
                       Line([xs[0], -1.25, 0], [xs[3], -1.25, 0]).set_stroke(FAINT, 1.5),
                       Line([xs[0], -2.75, 0], [xs[3], -2.75, 0]).set_stroke(INK, 2))
        head_lab = L("paired test (discordant seeds)", 20, MUTED, weight="BOLD").move_to([xs[0] + 0.08, ys["head"], 0], aligned_edge=LEFT)
        head_x = machine_badge("X", 24).move_to([(xs[1] + xs[2]) / 2, ys["head"], 0])
        head_l = machine_badge("L", 24).move_to([(xs[2] + xs[3]) / 2, ys["head"], 0])
        labels = {}
        for claim, desc in (("H0", "slow + hinge beats plain gate"), ("S0", "slow memory beats plain gate"),
                            ("H0S", "slow + hinge beats slow memory")):
            g = VGroup(L(claim, 20, MUTED), L(desc, 22)).arrange(RIGHT, buff=0.15)
            g.move_to([xs[0] + 0.08, ys[claim], 0], aligned_edge=LEFT)
            labels[claim] = g
        cells = {}
        for claim in ("H0", "S0", "H0S"):
            for m, xm in (("X", (xs[1] + xs[2]) / 2), ("L", (xs[2] + xs[3]) / 2)):
                cells[claim, m] = verdict_cell(claim, m).move_to([xm, ys[claim], 0])
        table = VGroup(rules, head_lab, head_x, head_l, *labels.values())
        hl_legend = VGroup(legend_item(WARN, "only the first arm discovered", outline=True),
                           legend_item(BAD, "only the second arm discovered", outline=True)).arrange(RIGHT, buff=0.5)
        hl_legend.move_to([xs[0], -3.12, 0], aligned_edge=LEFT)
        row_hl = Rectangle(width=xs[3] - xs[0] + 0.1, height=0.46).set_fill(WARN, 0.1).set_stroke(WARN, 1, opacity=0.5)
        row_hl.move_to([(xs[0] + xs[3]) / 2, ys["H0"], 0])

        def ov(m, arm):
            return next(r.ov for r in B[m].rows if r.arm == arm)

        n3 = ("The recipe beat the plain gate on both machines, eleven to zero and twelve to zero in discordant "
              "seeds, and slow memory alone beat it on both as well. The hinge's gain over slow memory alone was "
              "shown on L, five to zero. On X it was four to zero, a p-value of 0.0625, just short.")
        with self.voiceover(n3) as vo:
            self.play(FadeIn(table), FadeIn(row_hl), FadeIn(hl_legend),
                      self.swap_note("Report Table 4 (McNemar, discordant seeds); data/slow_start.json"), run_time=0.8)
            h0 = {m: self.disc_rects(m, "HINGE0", "A0") for m in ("X", "L")}
            self.play(FadeIn(ov("X", "SLOW0")), FadeIn(ov("L", "SLOW0")),
                      *[LaggedStartMap(ShowCreation, h0[m], lag_ratio=0.08) for m in ("X", "L")], run_time=1.4)
            vo.wait_until(at_phrase(vo, 0, "eleven to zero"))
            self.play(FadeIn(cells["H0", "X"], shift=0.1 * UP), run_time=0.6)
            vo.wait_until(at_phrase(vo, 0, "twelve to zero"))
            self.play(FadeIn(cells["H0", "L"], shift=0.1 * UP), run_time=0.6)

            vo.wait_until(at_phrase(vo, 0, "and slow memory alone"))
            s0 = {m: self.disc_rects(m, "SLOW0", "A0") for m in ("X", "L")}
            self.play(FadeOut(VGroup(h0["X"], h0["L"])), FadeOut(ov("X", "SLOW0")), FadeOut(ov("L", "SLOW0")),
                      FadeIn(ov("X", "HINGE0")), FadeIn(ov("L", "HINGE0")),
                      row_hl.animate.set_y(ys["S0"]), run_time=0.7)
            self.play(*[LaggedStartMap(ShowCreation, s0[m], lag_ratio=0.08) for m in ("X", "L")], run_time=1.0)
            self.play(FadeIn(cells["S0", "X"], shift=0.1 * UP), FadeIn(cells["S0", "L"], shift=0.1 * UP),
                      run_time=0.6)

            vo.wait_until_sentence(1)
            h0s = {m: self.disc_rects(m, "HINGE0", "SLOW0") for m in ("X", "L")}
            self.play(FadeOut(VGroup(s0["X"], s0["L"])), FadeOut(ov("X", "HINGE0")), FadeOut(ov("L", "HINGE0")),
                      FadeIn(ov("X", "A0")), FadeIn(ov("L", "A0")), row_hl.animate.set_y(ys["H0S"]), run_time=0.7)
            self.play(LaggedStartMap(ShowCreation, h0s["L"], lag_ratio=0.15), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "five to zero"))
            self.play(FadeIn(cells["H0S", "L"], shift=0.1 * UP), run_time=0.6)

            vo.wait_until_sentence(2)
            self.play(LaggedStartMap(ShowCreation, h0s["X"], lag_ratio=0.15), run_time=0.8)
            vo.wait_until(at_phrase(vo, 2, "four to zero"))
            self.play(FadeIn(cells["H0S", "X"], shift=0.1 * UP), run_time=0.6)
            vo.wait_until(at_phrase(vo, 2, "a p-value"))
            self.play(Indicate(cells["H0S", "X"][1][-6:], color=WARN, scale_factor=1.2), run_time=1.0)
            self.play(FlashAround(cells["H0S", "X"][0], color=MUTED), run_time=1.0)

        # clear Part 0 (it comes back in N14.5); the tabs move on to parts A and B
        self.play(FadeOut(VGroup(h0s["X"], h0s["L"], ov("X", "A0"), ov("L", "A0"), row_hl, table, hl_legend,
                                 *cells.values())), run_time=0.6)
        self.p0_all = VGroup(header, *[B[m].frame for m in ("X", "L")],
                             *[r.cnt for m in ("X", "L") for r in B[m].rows])
        self.play(FadeOut(self.p0_all), FadeOut(legend), *self.tab_anims([1, 2]), run_time=0.8)

    # ================================================================== N14.4  Parts A and B
    def parts_a_b(self):
        panels = {}
        for key, cx, head, sub, plain_arm, rec_arm, claim in (
                ("A", -3.2, "Part A: eight keys", "S = 2, P = 8, convolution · seeds 220–239",
                 "DIRECT8", "HINGE8", "HA"),
                ("B", 3.6, "Part B: four streams, 16 channels", "S = 4, P = 4, k = 16 · seeds 240–259",
                 "A4k16", "HINGE4k16", "HB")):
            left = cx - 3.15
            h = L(head, 26, weight="BOLD").move_to([left, 2.45, 0], aligned_edge=LEFT)
            s = L(sub, 20, MUTED).move_to([left, 2.03, 0], aligned_edge=LEFT)
            p = SimpleNamespace(head=VGroup(h, s), bars={}, verdict={}, frame=VGroup(h, s))
            for m, (yp, yr, yv) in (("X", (1.2, 0.7, 0.15)), ("L", (-0.8, -1.3, -1.85))):
                badge = machine_badge(m, 28).move_to([left + 0.15, (yp + yr) / 2, 0])
                fbs = []
                for arm, lab, color, y in ((plain_arm, "plain, recorded", MUTED, yp), (rec_arm, "slow + hinge", GOOD, yr)):
                    num, den = _frac(m, arm)
                    fb = frac_bar(lab, num, den, color=color, width=2.4, height=0.38, label_width=1.45, size=22)
                    fb.shift([cx - 0.9 - fb.track.get_left()[0], y - fb.track.get_y(), 0])
                    fbs.append(fb)
                    p.frame.add(fb.name_mob, fb.track)
                p.frame.add(badge)
                p.bars[m] = fbs
                v = verdict_cell(claim, m, color_counts=False)
                v.move_to([fbs[0].name_mob.get_left()[0], yv, 0], aligned_edge=LEFT)
                p.verdict[m] = v
            panels[key] = p
        foot = L("no restarts in any arm  ·  recorded: the plain gate's earlier runs on the same seeds", 20, MUTED)
        foot.move_to([0, -2.75, 0])
        divider = DashedLine([0.25, 2.6, 0], [0.25, -2.15, 0], dash_length=0.1).set_stroke(PANEL_EDGE, 1.5)

        n4 = ("At eight keys per stream, with the convolution, the recipe bound twenty and seventeen of twenty, "
              "against thirteen and twelve for the recorded plain runs, shown on both machines. At four streams "
              "with sixteen channels, it bound seventeen and eighteen of twenty, without restarts, against twelve "
              "and nine: shown on L; on X it was seven to two, not significant.")
        A, Bp = panels["A"], panels["B"]
        with self.voiceover(n4) as vo:
            self.play(FadeIn(A.frame, shift=0.15 * UP), ShowCreation(divider), FadeIn(foot),
                      self.swap_note("Report §7, Table 4 (HA, HB: McNemar)"), run_time=0.9)
            vo.wait_until(at_phrase(vo, 0, "the recipe bound"))
            self.play(*[a for m in ("X", "L") for a in grow_bar(A.bars[m][1])], run_time=1.0)
            vo.wait_until(at_phrase(vo, 0, "against thirteen"))
            self.play(*[a for m in ("X", "L") for a in grow_bar(A.bars[m][0])], run_time=1.0)
            vo.wait_until(at_phrase(vo, 0, "shown on both"))
            self.play(FadeIn(A.verdict["X"], shift=0.1 * UP), FadeIn(A.verdict["L"], shift=0.1 * UP), run_time=0.7)

            vo.wait_until_sentence(1)
            self.play(FadeIn(Bp.frame, shift=0.15 * UP), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "it bound seventeen"))
            self.play(*[a for m in ("X", "L") for a in grow_bar(Bp.bars[m][1])], run_time=1.0)
            vo.wait_until(at_phrase(vo, 1, "without restarts"))
            self.play(Indicate(foot[:18], color=GOOD, scale_factor=1.08), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "against twelve"))
            self.play(*[a for m in ("X", "L") for a in grow_bar(Bp.bars[m][0])], run_time=1.0)
            vo.wait_until(at_phrase(vo, 1, "shown on L"))
            self.play(FadeIn(Bp.verdict["L"], shift=0.1 * UP), run_time=0.6)
            vo.wait_until(at_phrase(vo, 1, "on X it was"))
            self.play(FadeIn(Bp.verdict["X"], shift=0.1 * UP), run_time=0.6)
            self.play(FlashAround(Bp.verdict["X"], color=MUTED), run_time=1.0)

        mobs = [m for m in self.mobjects if m not in (self.title_mob, self.tabs, self.note, self.frame)]
        self.play(*[FadeOut(m) for m in mobs], *self.tab_anims([0]), run_time=0.7)

    # ================================================================== N14.5  failures, firings, eight streams
    def failures(self):
        B = self.blocks
        legend2 = VGroup(legend_item(GOOD, "discovered"), legend_item(BAD, "position split"),
                         legend_item(KEY_SPLIT, "key split"), legend_item(OTHER_FAIL, "other failure")
                         ).arrange(RIGHT, buff=0.3)
        legend2.move_to([6.6, 2.5, 0], aligned_edge=RIGHT)
        self.play(FadeIn(self.p0_all), run_time=0.8)

        def comp_chip(m, arm):
            codes = PART0[m][arm]
            fails = 20 - codes.count("D")
            pos = codes.count("P")
            if pos == 0:
                return chip("no position splits", GOOD, 20, padx=0.14, pady=0.06)
            return chip(f"{pos} of {fails}: position", BAD, 20, padx=0.14, pady=0.06)

        chips = {(m, r.arm): comp_chip(m, r.arm).move_to([CHIP_LX, r.y, 0], aligned_edge=LEFT)
                 for m in ("X", "L") for r in B[m].rows}

        def ovs(arm):
            return [next(r.ov for r in B[m].rows if r.arm == arm) for m in ("X", "L")]

        # hinge firings per X HINGE0 run, aligned with the strip's columns
        fy = -1.2
        f_name = L("hinge firings", 22).move_to([NAME_RX, fy, 0], aligned_edge=RIGHT)
        f_code = L("X, HINGE0", 20, MUTED).move_to([CODE_LX, fy, 0], aligned_edge=LEFT)
        f_code.set_x(min(f_code.get_x(), STRIP_X0 - 0.1 - f_code.get_width() / 2))
        f_name.next_to(f_code, LEFT, buff=0.15)
        digits = VGroup(*[L(str(f), 20, MUTED if f == 0 else HINGE_COLOR, weight="BOLD").move_to([col_x(j), fy, 0])
                          for j, f in enumerate(FIRINGS_X)])
        j287 = SEEDS0.index(287)
        callout = L("950: the one failed run, seed 287", 20, MUTED).move_to([col_x(j287), fy - 0.6, 0])
        call_line = Line(digits[j287].get_bottom() + 0.05 * DOWN, callout.get_top() + 0.05 * UP).set_stroke(MUTED, 1.5)
        zeros = VGroup(*[digits[j] for j, f in enumerate(FIRINGS_X) if f == 0])
        zero_boxes = VGroup(*[RoundedRectangle(width=0.26, height=0.36, corner_radius=0.06)
                              .set_stroke(WARN, 2).set_fill(WARN, 0.08).move_to([col_x(j), fy, 0])
                              for j, f in enumerate(FIRINGS_X) if f == 0])
        never = chip("never fired: 11 of 20", WARN, 20).move_to([CHIP_LX - 0.6, fy, 0], aligned_edge=LEFT)
        x_hinge = B["X"].rows[2]

        n5 = ("Look at how the failures change. The plain gate failed mostly by position splits; slow memory alone "
              "failed almost only by position splits; the hinge removes those. And it fired rarely: never on eleven "
              "of X's twenty two-stream runs. At eight streams, though, nothing bound, zero of ten on each machine, "
              "and four of ten collapsed.")
        with self.voiceover(n5) as vo:
            self.play(*[self.fill_row(r, typed=True, run_time=1.0) for m in ("X", "L") for r in B[m].rows],
                      FadeIn(legend2),
                      self.swap_note("Report §7; failure classes per seed: data/slow_start.json"), run_time=1.2)

            vo.wait_until(at_phrase(vo, 1, "The plain gate"))
            self.play(*[FadeIn(o) for o in ovs("SLOW0") + ovs("HINGE0")],
                      *[FadeIn(chips[m, "A0"], shift=0.1 * LEFT) for m in ("X", "L")], run_time=0.7)
            vo.wait_until(at_phrase(vo, 1, "slow memory alone"))
            self.play(*[FadeOut(o) for o in ovs("SLOW0")], *[FadeIn(o) for o in ovs("A0")],
                      *[FadeIn(chips[m, "SLOW0"], shift=0.1 * LEFT) for m in ("X", "L")], run_time=0.7)
            vo.wait_until(at_phrase(vo, 1, "the hinge removes"))
            self.play(*[FadeOut(o) for o in ovs("HINGE0")], *[FadeIn(o) for o in ovs("SLOW0")],
                      *[FadeIn(chips[m, "HINGE0"], shift=0.1 * LEFT) for m in ("X", "L")], run_time=0.7)
            self.play(*[Indicate(r.sqs, color=GOOD, scale_factor=1.03) for r in (B["X"].rows[2], B["L"].rows[2])],
                      run_time=1.0)

            # ---------------- the hinge fires rarely
            vo.wait_until_sentence(2)
            # A0 and SLOW0 are dimmed on both machines already; dim L's HINGE0 too, leaving X's HINGE0 lit
            self.play(FadeIn(B["L"].rows[2].ov),
                      FadeOut(VGroup(*chips.values())), FadeIn(f_name), FadeIn(f_code),
                      TransformFromCopy(x_hinge.sqs, digits),
                      self.swap_note("Report §7; hinge firings: data/hinge_firings.json"), run_time=1.0)
            self.play(ShowCreation(call_line), FadeIn(callout, shift=0.1 * DOWN), run_time=0.4)
            vo.wait_until(at_phrase(vo, 2, "never on eleven"))
            self.play(LaggedStart(*[z.animate.set_color(WARN) for z in zeros], lag_ratio=0.06),
                      LaggedStartMap(ShowCreation, zero_boxes, lag_ratio=0.06),
                      FadeIn(never, shift=0.1 * LEFT), run_time=0.9)

            # ---------------- Part C: eight streams (the firing row stays up a moment longer)
            vo.wait_until(at_phrase(vo, 3, "though"))
            mobs = [m for m in self.mobjects if m not in (self.title_mob, self.tabs, self.note, self.frame)]
            self.play(*[FadeOut(m) for m in mobs], *self.tab_anims([3]), run_time=0.6)
            sq_c, pitch_c, x0c = 0.5, 0.6, -4.4
            head = L("Part C: eight streams, 16 channels", 26, weight="BOLD").move_to([0, 2.25, 0])
            sub = L("HINGE D8 · S = 8, P = 4, k = 16 · one square per run, no restarts", 20, MUTED).next_to(head, DOWN, buff=0.18)
            rowsC, counts, colls = VGroup(), VGroup(), VGroup()
            for m, y in (("X", 0.75), ("L", -0.45)):
                badge = machine_badge(m, 30).move_to([x0c - 0.65, y, 0])
                sqs = VGroup(*[Square(sq_c).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1.2)
                               .move_to([x0c + sq_c / 2 + j * pitch_c, y, 0]) for j in range(10)])
                rowsC.add(VGroup(badge, sqs))
                num, den = map(int, ROWS4["HINGE_D8"][m].split("/"))
                cnt = L(f"{num}/{den} bound", 26, BAD, weight="BOLD").next_to(sqs, RIGHT, buff=0.35)
                counts.add(cnt)
                colls.add(L("4 of 10 collapsed", 22, BAD).next_to(cnt, RIGHT, buff=0.35))
            leg = VGroup(legend_item(FAILN, "did not bind", opacity=1),
                         legend_item(BAD, "collapsed: final accuracy below 0.15")).arrange(RIGHT, buff=0.5)
            leg.move_to([0, -1.6, 0])
            self.play(FadeIn(head, shift=0.1 * UP), FadeIn(sub), FadeIn(rowsC),
                      self.swap_note("Report §7, Table 4 (Part C); collapsed: final accuracy < 0.15"), run_time=0.7)
            vo.wait_until(at_phrase(vo, 3, "nothing bound"))
            self.play(*[LaggedStart(*[s.animate.set_fill(FAILN, 1).set_stroke(FAINT, 1.2) for s in r[1]],
                                    lag_ratio=0.06) for r in rowsC], FadeIn(leg[0]), run_time=1.0)
            vo.wait_until(at_phrase(vo, 3, "zero of ten"))
            self.play(LaggedStartMap(FadeIn, counts, shift=0.1 * LEFT, lag_ratio=0.3), run_time=0.7)
            vo.wait_until(at_phrase(vo, 3, "four of ten"))
            self.play(*[LaggedStart(*[s.animate.set_fill(BAD, 0.9).set_stroke(BAD, 1.2) for s in r[1][:4]],
                                    lag_ratio=0.12) for r in rowsC],
                      LaggedStartMap(FadeIn, colls, shift=0.1 * LEFT, lag_ratio=0.3), FadeIn(leg[1]), run_time=1.0)
