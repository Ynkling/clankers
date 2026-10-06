"""Chapter 16 — the hinge is an early kick (report §9.1: screens S25, S32, S34, S35 on machine E;
test_early_recipe.py docstring; X's HINGE4k16 transitions from video/data/slow_start.json)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# ------------------------------------------------------------------------------------- numbers
# The screens' records live on branch claude/outside-ideas (not in this checkout), so every screen
# number below is the report's (§9.1), cross-checked against test_early_recipe.py:13-22.
S25_FULL, S25_PLAIN, N25 = 20, 9, 20            # Muon + SLOW + HINGE / neither (all 11 failures POSITION)
S32_SLOW, S32_HINGE, N32 = 31, 39, 40           # Muon + SLOW without / with the hinge; 8 vs 0, p = 0.008
S32_B, S32_C, S32_P = 8, 0, "0.008"
FIRST_FIRE = (50, 168)                          # first firing on every rescued seed (S32)
RATIO_MUON, RATIO_ADAM = 850, 726               # median |hinge grad| / |task grad| on the gate (S34)
ALIGNED = 59                                    # median updates with cos(momentum, push) >= 0.5 (Muon)
S34 = {"Muon": (38, 39), "Adam": (34, 34)}      # (WINDOW, full hinge) of 40
S35_BOUND, S35_BY2400, S35_N = 17, 15, 20       # Muon + HINGE at S = 4, k = 16; all 17 bound by 3600

_SS = json.load(open(os.path.join(DATA, "slow_start.json")))
X_H16 = sorted(t for t in _SS["per_seed"]["X"]["HINGE4k16"]["transition"] if t is not None)
assert len(X_H16) == _SS["counts"]["X"]["HINGE4k16"]["success"] == S35_BOUND
assert (X_H16[0], X_H16[-1]) == (3600, 21600)   # report: "X's transitions ranged from 3600 to 21,600"

OUT_COL = {"D": GOOD, "P": BAD, "F": BAD}


# ------------------------------------------------------------------------------------- helpers
def at_phrase(vo, i, phrase, before=0.15):
    """Approximate time (s into the block) at which `phrase` is spoken in sentence i."""
    text, a, b = vo.synth.sentences[i]
    k = text.find(phrase)
    if k < 0:
        raise ValueError(phrase)
    return max(0.0, a + (b - a) * k / max(len(text), 1) - before)


def chip(text, color, size=22, padx=0.2, pady=0.11, fill=0.15, weight="BOLD"):
    t = L(text, size=size, color=color, weight=weight)
    box = RoundedRectangle(width=t.get_width() + 2 * padx, height=t.get_height() + 2 * pady, corner_radius=0.1)
    box.set_fill(color, fill).set_stroke(color, 2)
    t.move_to(box)
    return VGroup(box, t)


def question(text, color, size=34):
    t = T(text, size=size, color=INK)
    box = SurroundingRectangle(t, buff=0.32).round_corners(0.14)
    box.set_fill(PANEL, 1).set_stroke(color, 2.5)
    mark = T("?", size=size + 10, color=color).next_to(box, LEFT, buff=0.25)
    return VGroup(box, t, mark)


def run_dots(outcomes, cols=20, spacing=0.27, r=0.1):
    dots = VGroup()
    for i, o in enumerate(outcomes):
        d = Dot(radius=r).set_fill(OUT_COL[o], 1).set_stroke(width=0)
        d.move_to([(i % cols) * spacing, -(i // cols) * spacing, 0])
        dots.add(d)
    return dots


def bracket(x0, x1, y, color, h=0.12, down=True):
    """A flat brace drawn with lines (no TeX)."""
    s = -1 if down else 1
    xm = 0.5 * (x0 + x1)
    pts = [[x0, y - s * h, 0], [x0, y, 0], [xm - 0.1, y, 0], [xm, y + s * 0.1, 0], [xm + 0.1, y, 0],
           [x1, y, 0], [x1, y - s * h, 0]]
    return VMobject().set_points_as_corners(pts).set_stroke(color, 2.5)


def legend_dot(color, text):
    return VGroup(Dot(radius=0.1).set_fill(color, 1), L(text, size=20, color=MUTED)).arrange(RIGHT, buff=0.12)


# timeline of updates 0..1200 (Parts 1-3)
TL_X0, TL_X1, TL_MAX = -5.6, 4.9, 1200.0


def tl_x(u):
    return TL_X0 + (TL_X1 - TL_X0) * u / TL_MAX


# landscape from Ch. 4 (schematic task loss along one routing direction)
LCX, LTOP, LHALF, LDROP = 3.75, 0.85, 2.45, 1.5
R_BALL = 0.17


def ls_shape(u):
    return np.tanh((u / 0.62) ** 4)


def ls_point(u):
    return np.array([LCX + LHALF * u, LTOP - LDROP * ls_shape(u), 0.0])


def ball_pos(u):
    p = ls_point(u)
    slope = -LDROP * (ls_shape(u + 1e-4) - ls_shape(u - 1e-4)) / 2e-4 / LHALF
    n = np.array([-slope, 1.0, 0.0])
    return p + R_BALL * n / np.linalg.norm(n)


def ls_tangent(u):
    d = ls_point(u + 1e-3) - ls_point(u - 1e-3)
    return d / np.linalg.norm(d)


# momentum alignment after a firing (illustrative shape: momentum 0.95 decaying against steady task
# gradients, scaled so that the cosine crosses 0.5 at the reported median of 59 updates)
_C = np.sqrt(3) * 0.95 ** ALIGNED / (1 - 0.95 ** ALIGNED)


def cos_align(t):
    a = 0.95 ** t
    b = _C * (1 - 0.95 ** t)
    return a / np.sqrt(a * a + b * b)


U_PUSH = 0.42                       # where the ball is when the alignment ends


def ball_u(t):
    if t <= ALIGNED:
        return U_PUSH * t / ALIGNED   # equal steps (Muon's step size ignores the gradient's size)
    return U_PUSH + 0.5 * (1 - np.exp(-(t - ALIGNED) / 14.0))


class EarlyKick(ClankersScene):
    def construct(self):
        card = self.chapter_card(16, "The hinge is an early kick")
        self.wait(0.6)
        self.title_mob = section_title("The hinge under Muon")
        self.play(FadeOut(card, shift=0.3 * UP), FadeIn(self.title_mob, shift=0.3 * UP))
        self.src = None

        self.part_screens()
        self.part_firing()
        self.part_kick()
        self.part_window()
        self.wait(1.0)
        self.clear_all()

    # ------------------------------------------------------------------ small utilities
    def new_title(self, text):
        new = section_title(text)
        old = self.title_mob
        self.title_mob = new
        return [FadeOut(old, shift=0.25 * UP), FadeIn(new, shift=0.25 * UP)]

    def new_source(self, text):
        new = source_note(text)
        anims = [FadeIn(new)]
        if self.src is not None:
            anims.append(FadeOut(self.src))
        self.src = new
        return anims

    # ================================================================== N16.1 the Muon screens
    def part_screens(self):
        e_tag = VGroup(machine_badge("E", size=20), L("exploratory screens", size=20, color=MUTED))
        e_tag.arrange(RIGHT, buff=0.15)
        e_tag.move_to([6.5, 3.1, 0], aligned_edge=RIGHT)
        self.e_tag = e_tag

        q1 = question("What does the hinge actually do?", HINGE_COLOR)
        q2 = question("Does the recipe survive a different optimizer?", MUON_COLOR)
        qs = VGroup(q1, q2).arrange(DOWN, buff=0.5, aligned_edge=LEFT).move_to(0.75 * UP)
        adam = chip("Adam", ADAM_COLOR, size=26)
        muon = chip("Muon", MUON_COLOR, size=26)
        swap_arrow = Arrow(LEFT, RIGHT, buff=0, thickness=3, fill_color=MUTED).set_width(0.9)
        swap = VGroup(adam, swap_arrow, muon).arrange(RIGHT, buff=0.25).next_to(q2, DOWN, buff=0.5)

        # ---- S25 panel
        head_chip = chip("Muon", MUON_COLOR, size=22)
        h25 = L("screen S25 · seeds 160–179 · one dot per run, grouped by outcome", size=22, color=MUTED)
        head = VGroup(head_chip, h25).arrange(RIGHT, buff=0.25).move_to([-6.3, 2.2, 0], aligned_edge=LEFT)
        legend = VGroup(legend_dot(GOOD, "discovered"), legend_dot(BAD, "not discovered")).arrange(RIGHT, buff=0.4)
        legend.move_to([6.5, 2.2, 0], aligned_edge=RIGHT)

        XD = -3.0

        def row(name, sub, outcomes, y, num, den, cols=20):
            nm = L(name, size=24)
            lab = VGroup(nm, L(sub, size=20, color=MUTED)).arrange(DOWN, buff=0.08, aligned_edge=LEFT) if sub \
                else VGroup(nm)
            lab.move_to([-6.3, y, 0], aligned_edge=LEFT)
            dots = run_dots(outcomes, cols=cols).move_to([XD, y, 0], aligned_edge=LEFT)
            val = L(f"{num}/{den}", size=28, color=GOOD if num == den else INK, weight="BOLD")
            val.move_to([XD + 5.95, y, 0], aligned_edge=LEFT)
            return lab, dots, val

        a_lab, a_dots, a_val = row("slow memory + hinge", None, "D" * S25_FULL, 1.05, S25_FULL, N25)
        routed = chip("all routed by step 1200", GOOD, size=20, weight="NORMAL")
        routed.move_to([6.5, 1.05, 0], aligned_edge=RIGHT)
        b_lab, b_dots, b_val = row("plain gate", "no slow memory, no hinge",
                                   "D" * S25_PLAIN + "P" * (N25 - S25_PLAIN), -0.2, S25_PLAIN, N25)
        fails = VGroup(*b_dots[S25_PLAIN:])
        br = bracket(fails.get_left()[0], fails.get_right()[0], fails.get_bottom()[1] - 0.12, BAD)
        br_lab = L(f"all {N25 - S25_PLAIN} failures: position splits", size=22, color=BAD)
        br_lab.next_to(br, DOWN, buff=0.12)
        s25 = VGroup(a_lab, a_dots, a_val, routed, b_lab, b_dots, b_val, br, br_lab)

        # ---- S32 panel
        h32 = L("screen S32 · seeds 160–199 · 40 runs per arm, grouped by outcome", size=22, color=MUTED)
        h32.move_to(h25, aligned_edge=LEFT)
        out_slow = "D" * S32_SLOW + "F" * (N32 - S32_SLOW)
        c_lab, c_dots, c_val = row("slow memory, no hinge", None, out_slow, 1.2, S32_SLOW, N32)
        d_lab, d_dots, d_val = row("slow memory + hinge", None, out_slow, -0.1, S32_HINGE, N32)
        d_val.set_color(INK)
        rescued = VGroup(*d_dots[S32_SLOW:S32_SLOW + S32_B])
        rings = VGroup(*[Circle(radius=0.135).set_stroke(HINGE_COLOR, 2.5).move_to(d) for d in rescued])
        res_chip = chip(f"{S32_B} rescued", HINGE_COLOR, size=22)
        lost_chip = chip(f"{S32_C} lost", MUTED, size=22)
        chips = VGroup(res_chip, lost_chip).arrange(RIGHT, buff=0.2)
        chips.move_to([6.5, 0.65, 0], aligned_edge=RIGHT)
        mcn = L(f"McNemar {S32_B} vs {S32_C}, p = {S32_P}", size=20, color=MUTED)
        mcn.next_to(chips, DOWN, buff=0.15).align_to(chips, RIGHT)
        copy_arrow = Arrow(c_dots.get_bottom() + 0.05 * DOWN, d_dots.get_top() + 0.05 * UP, buff=0.05,
                           thickness=2.5, fill_color=HINGE_COLOR)
        copy_lab = L("add the hinge", size=20, color=HINGE_COLOR).next_to(copy_arrow, RIGHT, buff=0.12)

        # ---- timeline of updates
        tl_y = -1.75
        axis = Line([TL_X0, tl_y, 0], [TL_X1, tl_y, 0]).set_stroke(MUTED, 2)
        ticks = VGroup(*[Line([tl_x(u), tl_y - 0.07, 0], [tl_x(u), tl_y + 0.07, 0]).set_stroke(MUTED, 2)
                         for u in (0, 300, 600, 900, 1200)])
        tick_labs = VGroup(*[L(str(u), size=20, color=MUTED).next_to([tl_x(u), tl_y, 0], DOWN, buff=0.15)
                             for u in (0, 300, 600, 900, 1200)])
        ax_lab = L("update", size=20, color=MUTED).next_to(axis, RIGHT, buff=0.15)
        band = Rectangle(width=tl_x(FIRST_FIRE[1]) - tl_x(FIRST_FIRE[0]), height=0.4)
        band.set_fill(HINGE_COLOR, 0.45).set_stroke(HINGE_COLOR, 1.5)
        band.move_to([tl_x(FIRST_FIRE[0]), tl_y, 0], aligned_edge=DL)
        band_labs = VGroup(*[L(str(u), size=20, color=HINGE_COLOR, weight="BOLD")
                             .next_to([tl_x(u), tl_y, 0], DOWN, buff=0.15) for u in FIRST_FIRE])
        desc = L("first firing, on every rescued seed", size=22, color=HINGE_COLOR)
        desc.next_to(band, RIGHT, buff=0.25)
        target = band.get_top()
        links = VGroup(*[Line(d.get_bottom(), target).set_stroke(HINGE_COLOR, 1.5, opacity=0.7) for d in rescued])
        timeline = VGroup(axis, ticks, tick_labs, ax_lab, band, band_labs)

        with self.voiceover(
            "The screens then asked what the hinge actually does, and whether the recipe survives a different "
            "optimizer. Under Muon, slow memory plus the hinge discovered twenty of twenty, all routed by step "
            "twelve hundred. Without either, nine of twenty, and all eleven failures were position splits. With "
            "slow memory but no hinge, thirty-one of forty; adding the hinge made it thirty-nine, eight seeds "
            "rescued and none lost. On every rescued seed, the hinge first fired between updates fifty and one "
            "hundred sixty-eight."
        ) as vo:
            # sentence 0: the two questions
            self.play(FadeIn(e_tag), FadeIn(q1, shift=0.2 * UP), run_time=0.9)
            vo.wait_until(at_phrase(vo, 0, "whether the recipe"))
            self.play(FadeIn(q2, shift=0.2 * UP), run_time=0.8)
            vo.wait_until(at_phrase(vo, 0, "a different optimizer"))
            self.play(FadeIn(adam, shift=0.1 * RIGHT), run_time=0.5)
            self.play(GrowArrow(swap_arrow), FadeIn(muon, shift=0.3 * RIGHT), run_time=0.7)

            # sentence 1: S25, slow memory + hinge
            vo.wait_until_sentence(1)
            self.play(FadeOut(VGroup(q1, q2), shift=0.3 * UP), FadeOut(VGroup(adam, swap_arrow)),
                      ReplacementTransform(muon, head_chip), FadeIn(h25, shift=0.1 * RIGHT), FadeIn(legend),
                      *self.new_source("Report §9.1 · screens S25, S32 (machine E)"), run_time=1.0)
            self.play(FadeIn(a_lab), LaggedStartMap(FadeIn, a_dots, scale=0.5, lag_ratio=0.06), run_time=1.3)
            vo.wait_until(at_phrase(vo, 1, "twenty of twenty"))
            self.play(FadeIn(a_val, scale=1.2), Flash(a_dots[-1].get_center(), color=GOOD), run_time=0.6)
            vo.wait_until(at_phrase(vo, 1, "all routed"))
            self.play(FadeIn(routed, shift=0.2 * LEFT), run_time=0.6)

            # sentence 2: neither
            vo.wait_until_sentence(2)
            self.play(FadeIn(b_lab), LaggedStartMap(FadeIn, b_dots, scale=0.5, lag_ratio=0.06), run_time=1.3)
            self.play(FadeIn(b_val, scale=1.2), run_time=0.4)
            vo.wait_until(at_phrase(vo, 2, "all eleven failures"))
            self.play(ShowCreation(br), FadeIn(br_lab, shift=0.1 * DOWN),
                      LaggedStartMap(Indicate, fails, color=BAD, scale_factor=1.3, lag_ratio=0.05), run_time=1.2)

            # sentence 3: S32
            vo.wait_until_sentence(3)
            self.play(FadeOut(s25, shift=0.2 * DOWN), FadeTransform(h25, h32), run_time=0.8)
            self.play(FadeIn(c_lab), LaggedStartMap(FadeIn, c_dots, scale=0.5, lag_ratio=0.025), run_time=1.2)
            self.play(FadeIn(c_val, scale=1.2), run_time=0.4)
            vo.wait_until(at_phrase(vo, 3, "adding the hinge"))
            self.play(TransformFromCopy(c_dots, d_dots), GrowArrow(copy_arrow), FadeIn(copy_lab),
                      FadeIn(d_lab), run_time=1.0)
            self.play(LaggedStart(*[d.animate.set_fill(GOOD) for d in rescued], lag_ratio=0.12),
                      LaggedStartMap(ShowCreation, rings, lag_ratio=0.12), run_time=1.2)
            self.play(FadeIn(d_val, scale=1.2), run_time=0.4)
            vo.wait_until(at_phrase(vo, 3, "eight seeds rescued"))
            self.play(FadeIn(res_chip, shift=0.2 * LEFT), run_time=0.5)
            vo.wait_until(at_phrase(vo, 3, "none lost"))
            self.play(FadeIn(lost_chip, shift=0.2 * LEFT), FadeIn(mcn),
                      Indicate(VGroup(*d_dots[:S32_SLOW]), color=GOOD, scale_factor=1.05), run_time=0.8)

            # sentence 4: first firings
            vo.wait_until_sentence(4)
            self.play(ShowCreation(axis), FadeIn(ticks), FadeIn(tick_labs), FadeIn(ax_lab), run_time=0.8)
            self.play(LaggedStartMap(ShowCreation, links, lag_ratio=0.08), run_time=1.0)
            vo.wait_until(at_phrase(vo, 4, "between updates"))
            self.play(GrowFromEdge(band, DOWN), FadeIn(band_labs), FadeIn(desc, shift=0.1 * RIGHT), run_time=0.9)
            self.play(Indicate(band_labs, color=HINGE_COLOR, scale_factor=1.25), run_time=0.9)

        self.s32 = VGroup(head_chip, h32, legend, c_lab, c_dots, c_val, d_lab, d_dots, d_val, rings, chips, mcn,
                          copy_arrow, copy_lab, links)
        self.timeline = timeline
        self.tl_desc = desc
        self.tl_y = tl_y

    # ================================================================== N16.2 what a firing is
    def part_firing(self):
        shift = -2.9 - self.tl_y
        self.tl_y = -2.9
        short_desc = L("first firings", size=20, color=HINGE_COLOR)
        self.play(*self.new_title("What a firing is"), FadeOut(self.s32),
                  self.timeline.animate.shift(shift * UP),
                  FadeOut(self.tl_desc), run_time=1.0)
        band = self.timeline[4]
        short_desc.next_to(band, RIGHT, buff=0.2).shift(0.05 * UP)
        self.tl_desc = short_desc
        y0 = self.tl_y

        # one firing: a spike inside the band
        u_fire = 0.5 * (FIRST_FIRE[0] + FIRST_FIRE[1])
        spike = Line([tl_x(u_fire), y0, 0], [tl_x(u_fire), y0 + 1.05, 0]).set_stroke(HINGE_COLOR, 5)
        spike_dot = Dot([tl_x(u_fire), y0 + 1.05, 0], radius=0.07).set_fill(HINGE_COLOR, 1)
        spike_lab = L("a firing", size=22, color=HINGE_COLOR, weight="BOLD")
        spike_lab.next_to(spike_dot, LEFT, buff=0.15)

        # log scale of gradient size on the gate, relative to the task gradient
        XL, DEC = -3.0, 2.1

        def xlog(v):
            return XL + DEC * (np.log10(v) + 1)

        y_task, y_mu, y_ad, y_ax = 1.5, 0.55, -0.4, -0.95
        lax = Line([XL, y_ax, 0], [xlog(1000) + 0.15, y_ax, 0]).set_stroke(MUTED, 2)
        lticks = VGroup(*[Line([xlog(v), y_ax - 0.07, 0], [xlog(v), y_ax + 0.07, 0]).set_stroke(MUTED, 2)
                          for v in (0.1, 1, 10, 100, 1000)])
        lgrid = VGroup(*[DashedLine([xlog(v), y_ax + 0.1, 0], [xlog(v), y_task + 0.45, 0], dash_length=0.06)
                         .set_stroke(FAINT, 1, opacity=0.6) for v in (1, 10, 100, 1000)])
        ltick_labs = VGroup(*[L(s, size=20, color=MUTED).next_to([xlog(v), y_ax, 0], DOWN, buff=0.14)
                              for v, s in ((0.1, "0.1×"), (1, "1×"), (10, "10×"), (100, "100×"), (1000, "1000×"))])
        lax_lab = L("gradient on the gate, relative to the task gradient (log scale)", size=20, color=MUTED)
        lax_lab.next_to(ltick_labs, DOWN, buff=0.12).set_x(xlog(10))
        head = L("at a firing, medians over firings (screen S34)", size=22, color=MUTED)
        head.move_to([-6.4, 2.3, 0], aligned_edge=LEFT)

        def name(text, color, opt=None):
            parts = [L(text, size=24, color=color, weight="BOLD")]
            if opt:
                parts.append(chip(opt, MUON_COLOR if opt == "Muon" else ADAM_COLOR, size=20))
            g = VGroup(*parts).arrange(RIGHT, buff=0.15)
            return g

        n_task = name("task gradient", KEY_COLOR).move_to([-6.4, y_task, 0], aligned_edge=LEFT)
        n_mu = name("hinge gradient", HINGE_COLOR, "Muon").move_to([-6.4, y_mu, 0], aligned_edge=LEFT)
        n_ad = name("hinge gradient", HINGE_COLOR, "Adam").move_to([-6.4, y_ad, 0], aligned_edge=LEFT)
        a_task = Arrow([XL, y_task, 0], [xlog(1), y_task, 0], buff=0, thickness=7, fill_color=KEY_COLOR)
        v_task = L("1×", size=26, color=KEY_COLOR, weight="BOLD").next_to(a_task, RIGHT, buff=0.15)

        def growing(y, tracker, color):
            arr = always_redraw(lambda: Arrow([XL, y, 0], [xlog(10 ** tracker.get_value()), y, 0], buff=0,
                                              thickness=7, fill_color=color))
            num = DecimalNumber(1, num_decimal_places=0, font_size=28, text_config=dict(font=SANS),
                                color=color)
            times = L("×", size=28, color=color, weight="BOLD")
            lab = VGroup(num, times)

            def upd(m):
                num.set_value(10 ** tracker.get_value())
                times.next_to(num, RIGHT, buff=0.04)
                m.next_to(arr.get_end(), RIGHT, buff=0.12)

            lab.add_updater(upd)
            return arr, lab

        tr_mu, tr_ad = ValueTracker(0.0), ValueTracker(0.0)
        a_mu, v_mu = growing(y_mu, tr_mu, HINGE_COLOR)
        a_ad, v_ad = growing(y_ad, tr_ad, HINGE_COLOR)

        with self.voiceover(
            "Here is what a firing is. At that moment the hinge's gradient on the gate was a median of eight "
            "hundred fifty times the task gradient under Muon, and seven hundred twenty-six times under Adam."
        ) as vo:
            self.play(ShowCreation(spike), FadeIn(spike_dot, scale=0.5), FadeIn(spike_lab, shift=0.1 * LEFT),
                      FadeIn(short_desc), run_time=0.8)
            self.play(Flash(spike_dot.get_center(), color=HINGE_COLOR, flash_radius=0.35), run_time=0.6)
            vo.wait_until_sentence(1)
            self.play(FadeIn(head), ShowCreation(lax), FadeIn(lticks), FadeIn(ltick_labs), FadeIn(lgrid),
                      FadeIn(lax_lab), *self.new_source("Report §9.1 · screen S34 (machine E)"), run_time=0.8)
            self.play(FadeIn(n_task), GrowArrow(a_task), FadeIn(v_task), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "the hinge's gradient"))
            a_mu0 = Arrow([XL, y_mu, 0], [xlog(1), y_mu, 0], buff=0, thickness=7, fill_color=HINGE_COLOR)
            self.play(FadeIn(n_mu[0]), TransformFromCopy(spike, a_mu0), run_time=0.6)
            self.remove(a_mu0)
            self.add(a_mu, v_mu)
            self.play(tr_mu.animate.set_value(np.log10(RATIO_MUON)), run_time=2.6, rate_func=smooth)
            vo.wait_until(at_phrase(vo, 1, "the task gradient"))
            self.play(Indicate(VGroup(n_task, a_task, v_task), color=WHITE, scale_factor=1.06), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "under Muon"))
            self.play(FadeIn(n_mu[1], shift=0.1 * RIGHT), run_time=0.5)
            vo.wait_until(at_phrase(vo, 1, "and seven hundred"))
            self.add(a_ad, v_ad)
            self.play(FadeIn(n_ad[0]), tr_ad.animate.set_value(np.log10(RATIO_ADAM)), run_time=1.8)
            self.play(FadeIn(n_ad[1], shift=0.1 * RIGHT), run_time=0.5)

        # freeze the growing arrows
        for m in (a_mu, a_ad, v_mu, v_ad):
            m.clear_updaters()
        self.firing = dict(spike=VGroup(spike, spike_dot, spike_lab), log=VGroup(head, lax, lticks, lgrid, ltick_labs,
                           lax_lab), task=VGroup(n_task, a_task, v_task), mu=(n_mu, a_mu, v_mu),
                           ad=VGroup(n_ad, a_ad, v_ad))

    # ================================================================== N16.3 what a firing does
    def part_kick(self):
        f = self.firing
        n_mu, a_mu, v_mu = f["mu"]

        # ---- the landscape (Ch. 4)
        curve = ParametricCurve(ls_point, t_range=(-1, 1, 0.02)).set_stroke(INK, 3)
        base_y = LTOP - LDROP - 0.15
        hill = Polygon(*[ls_point(u) for u in np.linspace(-1, 1, 81)],
                       [LCX + LHALF, base_y, 0], [LCX - LHALF, base_y, 0])
        hill.set_fill(PANEL, 1).set_stroke(width=0)
        ls_axis = Line([LCX - LHALF - 0.1, base_y, 0], [LCX + LHALF + 0.1, base_y, 0]).set_stroke(MUTED, 2)
        ls_tick = Line([LCX, base_y - 0.08, 0], [LCX, base_y + 0.08, 0]).set_stroke(MUTED, 2)
        ls_uni = L("uniform", size=20, color=MUTED).next_to(ls_tick, DOWN, buff=0.08)
        ls_left = L("routed", size=20, color=MUTED).next_to([LCX - LHALF, base_y, 0], DOWN, buff=0.12)
        ls_right = L("routed", size=20, color=MUTED).next_to([LCX + LHALF, base_y, 0], DOWN, buff=0.12)
        ls_name = L("task loss (schematic)", size=20, color=MUTED).move_to([LCX, base_y + 0.45, 0])
        landscape = VGroup(hill, ls_axis, ls_tick, curve, ls_uni, ls_left, ls_right, ls_name)
        ball = Circle(radius=R_BALL).set_fill(GATE_COLOR, 1).set_stroke(INK, 1.5).move_to(ball_pos(0))
        gate_lab = L("the gate", size=20, color=GATE_COLOR).next_to(ball, UP, buff=0.12)
        kick = Arrow(ball_pos(0) + 1.75 * LEFT, ball_pos(0) + (R_BALL + 0.04) * LEFT, buff=0, thickness=7,
                     fill_color=HINGE_COLOR)

        # ---- left: cosine of momentum and push, under Muon
        cx0 = -5.45
        chart = line_chart([0, 120, 40], [0, 1, 0.5], width=4.6, height=2.2, x_label="updates after the firing",
                           x_ticks=[0, 40, 80, 120], y_ticks=[0, 0.5, 1])
        chart.shift([cx0 - chart.c2p(0, 0)[0], -0.55 - chart.c2p(0, 0)[1], 0])
        mu_chip = chip("Muon", MUON_COLOR, size=22)
        mu_head = L("momentum stays aligned with the push", size=22, color=INK)
        k_head = VGroup(mu_chip, mu_head).arrange(RIGHT, buff=0.2).move_to([-6.4, 2.3, 0], aligned_edge=LEFT)
        sub = L("cosine between the gate's momentum and the push (shape illustrative)", size=20, color=MUTED)
        sub.move_to([-6.4, 1.92, 0], aligned_edge=LEFT)
        half = DashedLine(chart.c2p(0, 0.5), chart.c2p(120, 0.5), dash_length=0.08).set_stroke(MUTED, 1.5)
        tt = ValueTracker(0.0)

        def trace_mob():
            t1 = tt.get_value()
            ts = np.linspace(0, max(t1, 1e-3), max(2, int(t1) + 2))
            vm = VMobject().set_points_as_corners([chart.c2p(t, cos_align(t)) for t in ts])
            return vm.set_stroke(MUON_COLOR, 4)

        trace = always_redraw(trace_mob)
        tdot = always_redraw(lambda: Dot(chart.c2p(tt.get_value(), cos_align(tt.get_value())), radius=0.07)
                             .set_fill(MUON_COLOR, 1))
        m59 = DashedLine(chart.c2p(ALIGNED, 0), chart.c2p(ALIGNED, 1.0), dash_length=0.07).set_stroke(INK, 2)
        m59_lab = L(f"median {ALIGNED} updates", size=22, color=INK, weight="BOLD")
        m59_lab.next_to(chart.c2p(ALIGNED, 0.9), RIGHT, buff=0.12)
        ts_fill = np.linspace(0, ALIGNED, 60)
        fill = Polygon(*[chart.c2p(t, cos_align(t)) for t in ts_fill], chart.c2p(ALIGNED, 0), chart.c2p(0, 0))
        fill.set_fill(HINGE_COLOR, 0.25).set_stroke(width=0)

        # footprints: equal steps while aligned
        n_steps = 6
        feet = VGroup()
        for k in range(n_steps + 1):
            tk = ALIGNED * k / n_steps
            d = Dot(ls_point(ball_u(tk)) + 0.02 * DOWN, radius=0.05).set_fill(HINGE_COLOR, 1)
            d.tk = tk
            d.add_updater(lambda m: m.set_opacity(1.0 if tt.get_value() >= m.tk - 1e-6 else 0.0))
            feet.add(d)
        steps_lab = VGroup(L("equal-size steps", size=20, color=MUON_COLOR, weight="BOLD"),
                           L("(Muon ignores the gradient's size)", size=20, color=MUTED))
        steps_lab.arrange(DOWN, buff=0.06, aligned_edge=LEFT).move_to([LCX - 0.3, LTOP + 0.95, 0], aligned_edge=LEFT)

        def mom_arrow():
            t = tt.get_value()
            u = ball_u(t)
            c = cos_align(t)
            ln = 0.95 * c
            if ln < 0.08:
                return VMobject()
            d = ls_tangent(u)
            p = ball_pos(u) + (R_BALL + 0.03) * d
            return Arrow(p, p + ln * d, buff=0, thickness=4, fill_color=HINGE_COLOR, fill_opacity=min(1, 0.3 + c))

        mom = always_redraw(mom_arrow)

        # ---- left, sentence 2: Adam
        ad_chip = chip("Adam", ADAM_COLOR, size=22)
        ad_head = L("step = momentum ÷ running average size", size=22, color=INK)
        a_head = VGroup(ad_chip, ad_head).arrange(RIGHT, buff=0.2).move_to([-6.4, 2.3, 0], aligned_edge=LEFT)
        base_ad = 0.3
        xs = [-5.6 + 0.6 * i for i in range(8)]
        heights = [1.25, -1.05, 1.4, 0.95, -1.3, 1.15, -0.9, 1.35]
        band_ad = Rectangle(width=xs[-1] - xs[0] + 0.6, height=0.36).set_fill(ADAM_COLOR, 0.3).set_stroke(width=0)
        band_ad.move_to([0.5 * (xs[0] + xs[-1]), base_ad, 0])
        zero_ad = Line([xs[0] - 0.3, base_ad, 0], [xs[-1] + 0.3, base_ad, 0]).set_stroke(FAINT, 1.5)

        def bars(hs, color, w=0.3):
            g = VGroup()
            for x, h in zip(xs, hs):
                r = Rectangle(width=w, height=max(abs(h), 1e-3)).set_fill(color, 0.9).set_stroke(width=0)
                r.move_to([x, base_ad, 0], aligned_edge=DOWN if h >= 0 else UP)
                g.add(r)
            return g

        g_bars = bars(heights, HINGE_COLOR)
        g_bars_small = bars([0.12 * np.sign(h) for h in heights], HINGE_COLOR)
        s_bars = bars([0.62 * np.sign(h) for h in heights], INK)
        leg_ad = VGroup(
            VGroup(Square(0.24).set_fill(ADAM_COLOR, 0.5).set_stroke(width=0),
                   L("running average size", size=20, color=MUTED)).arrange(RIGHT, buff=0.12),
            VGroup(Square(0.24).set_fill(HINGE_COLOR, 0.9).set_stroke(width=0),
                   L("gradient at a firing", size=20, color=MUTED)).arrange(RIGHT, buff=0.12),
        ).arrange(RIGHT, buff=0.45).move_to([0.5 * (xs[0] + xs[-1]), -1.45, 0])
        step_lab = L("step: sign-like, full size, in every weight", size=24, color=INK)
        step_lab.move_to(leg_ad)

        # ---- sentence 3: either way
        stmt = VGroup(T("Either way:", size=30, color=MUTED),
                      T("a large, directional push,", size=38),
                      T("delivered early", size=38, color=HINGE_COLOR)).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        stmt.move_to([-3.5, 0.75, 0])
        y0 = self.tl_y
        marks = VGroup(*[Circle(radius=0.09).set_stroke(GATE_COLOR, 2.5).set_fill(BG, 1).move_to([tl_x(u), y0, 0])
                         for u in (300, 600)])
        marks_lab = L("no run routed yet", size=20, color=GATE_COLOR)
        marks_lab.move_to([0.5 * (tl_x(300) + tl_x(600)), y0 + 0.55, 0])
        rband = Rectangle(width=tl_x(900) - tl_x(600), height=0.4).set_fill(GATE_COLOR, 0.3).set_stroke(GATE_COLOR, 1.5)
        rband.move_to([tl_x(600), y0, 0], aligned_edge=DL)
        rband_lab = L("routing appears", size=20, color=GATE_COLOR).move_to([rband.get_x(), y0 + 0.55, 0])
        flat = DashedLine(ls_point(-0.36) + 0.03 * UP, ls_point(0.36) + 0.03 * UP, dash_length=0.1).set_stroke(WARN, 3)
        flat_lab = L("near the flat uniform start", size=22, color=WARN).next_to(flat, UP, buff=0.5)
        ghost = Circle(radius=R_BALL).set_fill(GATE_COLOR, 0.35).set_stroke(INK, 1, opacity=0.4).move_to(ball_pos(0))
        quote = L("Ch. 4: whatever pushes it off first decides where it goes", size=20, color=MUTED)
        quote.move_to([LCX - 0.2, -1.55, 0])

        with self.voiceover(
            "Under Muon, the gate's momentum then stayed aligned with that push for a median of fifty-nine "
            "updates, and since Muon's step size does not depend on the gradient's size, the gate keeps moving in "
            "the hinge's direction the whole time. Under Adam, a gradient far above its running average produces "
            "a sign-like, full-size step. Either way: a large, directional push, delivered early, while the gate is "
            "presumably still near its flat uniform start."
        ) as vo:
            # the Muon arrow becomes the kick on the gate
            self.play(*self.new_title("What a firing does"),
                      FadeOut(VGroup(f["log"], f["task"], f["ad"], v_mu, f["spike"], n_mu[0])), run_time=0.6)
            self.play(FadeIn(landscape), FadeIn(ball, scale=0.5), FadeIn(gate_lab),
                      ReplacementTransform(a_mu, kick), ReplacementTransform(n_mu[1], mu_chip),
                      *self.new_source("Report §9.1 · S32, S34 (machine E); cosine curve: shape illustrative"),
                      run_time=1.1)
            self.play(FadeIn(mu_head), FadeIn(sub), FadeIn(chart), ShowCreation(half), run_time=0.9)
            vo.wait_until(at_phrase(vo, 0, "with that push"))
            self.play(kick.animate.shift((R_BALL + 0.02) * RIGHT).set_opacity(0.0), Flash(ball.get_center(),
                      color=HINGE_COLOR, flash_radius=0.4), FadeOut(gate_lab), run_time=0.5)
            self.remove(kick)
            ball.add_updater(lambda m: m.move_to(ball_pos(ball_u(tt.get_value()))))
            self.add(trace, tdot, feet, mom)
            t59 = at_phrase(vo, 0, "updates, and since") + 0.1
            self.play(tt.animate.set_value(ALIGNED), run_time=max(1.5, t59 - vo.elapsed()), rate_func=linear)
            self.play(ShowCreation(m59), FadeIn(m59_lab, shift=0.1 * RIGHT), tt.animate.set_value(120),
                      run_time=2.2, rate_func=linear)
            vo.wait_until(at_phrase(vo, 0, "since Muon's step size"))
            self.play(FadeIn(steps_lab, shift=0.1 * DOWN),
                      LaggedStart(*[Indicate(d, color=MUON_COLOR, scale_factor=2.0) for d in feet], lag_ratio=0.15),
                      run_time=1.6)
            vo.wait_until(at_phrase(vo, 0, "the gate keeps moving"))
            self.play(FadeIn(fill), run_time=0.8)
            self.play(Indicate(m59_lab, color=HINGE_COLOR), run_time=0.9)

            # sentence 2: Adam
            vo.wait_until_sentence(1)
            for m in (trace, tdot, mom, ball, *feet):
                m.clear_updaters()
            self.remove(mom)
            self.play(FadeOut(VGroup(mu_chip, mu_head, sub, chart, half, trace, tdot, m59, m59_lab, fill)),
                      FadeIn(a_head), FadeIn(zero_ad), FadeIn(band_ad), FadeIn(g_bars_small),
                      FadeIn(leg_ad[0]), run_time=0.9)
            vo.wait_until(at_phrase(vo, 1, "far above"))
            self.play(ReplacementTransform(g_bars_small, g_bars), FadeIn(leg_ad[1]), run_time=1.0)
            vo.wait_until(at_phrase(vo, 1, "produces"))
            self.play(ReplacementTransform(g_bars, s_bars), FadeOut(leg_ad), FadeIn(step_lab), run_time=1.1)

            # sentence 3: either way
            vo.wait_until_sentence(2)
            self.play(FadeOut(VGroup(a_head, zero_ad, band_ad, s_bars, step_lab)), FadeOut(steps_lab), FadeOut(feet),
                      FadeIn(stmt[0]), run_time=0.7)
            self.play(Write(stmt[1]), run_time=1.0)
            vo.wait_until(at_phrase(vo, 2, "delivered early"))
            self.play(Write(stmt[2]), Indicate(self.timeline[4], color=HINGE_COLOR, scale_factor=1.15), run_time=1.0)
            vo.wait_until(at_phrase(vo, 2, "while the gate"))
            self.play(FadeIn(marks, scale=0.5), FadeIn(marks_lab), FadeIn(rband), FadeIn(rband_lab),
                      self.tl_desc.animate.next_to(self.timeline[4], UP, buff=0.12), run_time=0.9)
            vo.wait_until(at_phrase(vo, 2, "near its flat"))
            self.play(ShowCreation(flat), FadeIn(flat_lab), FadeIn(ghost, scale=0.5), run_time=0.9)
            self.play(FadeIn(quote, shift=0.1 * UP), run_time=0.7)

        self.kick_mobs = VGroup(stmt, landscape, ball, flat, flat_lab, ghost, quote, marks, marks_lab, rband,
                                rband_lab, self.tl_desc)

    # ================================================================== N16.4 only needed early
    def part_window(self):
        # schedule rows: update 0..9600 -> x
        SX0, SX1, SMAX = -2.6, 5.4, 9600.0

        def sx(u):
            return SX0 + (SX1 - SX0) * u / SMAX

        y_full, y_win, y_sax, hh = 1.6, 0.8, 0.45, 0.42
        head = L("hinge weight over training", size=22, color=MUTED).move_to([-6.3, 2.3, 0], aligned_edge=LEFT)
        sax = Line([SX0, y_sax, 0], [SX1, y_sax, 0]).set_stroke(MUTED, 2)
        sticks = VGroup(*[Line([sx(u), y_sax - 0.07, 0], [sx(u), y_sax + 0.07, 0]).set_stroke(MUTED, 2)
                          for u in (0, 2400, 4800, 7200, 9600)])
        stick_labs = VGroup(*[L(f"{u}", size=20, color=HINGE_COLOR if u == 2400 else MUTED,
                                weight="BOLD" if u == 2400 else "NORMAL").next_to([sx(u), y_sax, 0], DOWN, buff=0.12)
                              for u in (0, 2400, 4800, 7200, 9600)])
        sax_lab = L("update", size=20, color=MUTED).next_to(sax, RIGHT, buff=0.15)
        n_full = L("full hinge", size=24).move_to([-6.3, y_full + hh / 2, 0], aligned_edge=LEFT)
        n_win = VGroup(L("window", size=24), L("updates 1–2400", size=20, color=MUTED)).arrange(
            DOWN, buff=0.06, aligned_edge=LEFT).move_to([-6.3, y_win + hh / 2, 0], aligned_edge=LEFT)
        base_full = Line([SX0, y_full, 0], [SX1, y_full, 0]).set_stroke(FAINT, 1.5)
        base_win = Line([SX0, y_win, 0], [SX1, y_win, 0]).set_stroke(FAINT, 1.5)
        on_full = Line([SX0, y_full + hh, 0], [SX1, y_full + hh, 0]).set_stroke(HINGE_COLOR, 5)
        on_win = VMobject().set_points_as_corners([[SX0, y_win + hh, 0], [sx(2400), y_win + hh, 0],
                                                   [sx(2400), y_win, 0], [SX1, y_win, 0]]).set_stroke(HINGE_COLOR, 5)
        w1 = VGroup(*[L("1", size=20, color=MUTED).next_to([SX0, y + hh, 0], LEFT, buff=0.15) for y in (y_full, y_win)])
        w0 = VGroup(*[L("0", size=20, color=MUTED).next_to([SX0, y, 0], LEFT, buff=0.15) for y in (y_full, y_win)])
        off_lab = L("off", size=22, color=MUTED).next_to([sx(6000), y_win, 0], UP, buff=0.1)
        cut = DashedLine([sx(2400), y_sax, 0], [sx(2400), y_full + hh + 0.15, 0], dash_length=0.07)
        cut.set_stroke(HINGE_COLOR, 1.5, opacity=0.8)
        sched = VGroup(head, sax, sticks, stick_labs, sax_lab, n_full, n_win, base_full, base_win, on_full, on_win,
                       w1, w0, off_lab, cut)

        # S34 results
        rows = []
        for opt, color in (("Muon", MUON_COLOR), ("Adam", ADAM_COLOR)):
            win, full = S34[opt]
            rows.append(frac_bar(f"{opt} · window", win, 40, color=color, width=5.0, label_width=3.0, size=24))
            rows.append(frac_bar(f"{opt} · full hinge", full, 40, color=color, width=5.0, label_width=3.0, size=24))
        res = VGroup(*rows)
        for fb, y in zip(rows, (-0.5, -1.05, -1.8, -2.35)):
            fb.move_to([0, y, 0])
        res.shift((-6.3 - res.get_left()[0]) * RIGHT)
        mu_vs = L("1 vs 0", size=22, color=INK).move_to([res.get_right()[0] + 0.8, -0.775, 0])
        ad_vs = L("0 vs 0", size=22, color=INK).move_to([res.get_right()[0] + 0.8, -2.075, 0])
        tr = rows[0].track
        gap = Rectangle(width=tr.get_width() / 40, height=tr.get_height()).set_fill(BAD, 0.0).set_stroke(BAD, 2.5)
        gap.move_to([tr.get_left()[0] + tr.get_width() * 38.5 / 40, tr.get_y(), 0])
        loss = VGroup(T("one loss in 80 runs", size=30, color=INK),
                      L("the lost Muon seed had needed 277 late firings", size=20, color=MUTED))
        loss.arrange(DOWN, buff=0.1, aligned_edge=LEFT).move_to([-6.3, -3.1, 0], aligned_edge=LEFT)

        # S35: cumulative runs bound, Muon (E) vs Adam (X, HINGE4k16)
        chart = line_chart([0, 24000, 2400], [0, 20, 5], width=9.4, height=3.9, x_label="update",
                           y_label="runs bound (of 20)", x_ticks=[0, 4800, 9600, 14400, 19200, 24000],
                           y_ticks=[0, 5, 10, 15, 20])
        chart.move_to([0.25, -0.45, 0])
        c_head = L("four streams, sixteen channels · seeds 240–259 · runs bound by each update", size=22, color=MUTED)
        c_head.move_to([-6.3, 2.3, 0], aligned_edge=LEFT)
        evals = list(range(0, 24001, 1200))
        adam_cum = [sum(1 for t in X_H16 if t <= u) for u in evals]
        adam_line = VMobject().set_points_as_corners([chart.c2p(u, c) for u, c in zip(evals, adam_cum)])
        adam_line.set_stroke(ADAM_COLOR, 4)
        adam_dots = VGroup(*[Dot(chart.c2p(u, c), radius=0.045).set_fill(ADAM_COLOR, 1)
                             for u, c in zip(evals, adam_cum)])
        mu_pre = DashedLine(chart.c2p(0, 0), chart.c2p(2400, S35_BY2400), dash_length=0.08).set_stroke(MUON_COLOR, 4)
        mu_line = VMobject().set_points_as_corners([chart.c2p(2400, S35_BY2400), chart.c2p(3600, S35_BOUND),
                                                    chart.c2p(24000, S35_BOUND)]).set_stroke(MUON_COLOR, 5)
        mu_dot = Dot(chart.c2p(2400, S35_BY2400), radius=0.09).set_fill(MUON_COLOR, 1).set_stroke(INK, 1.5)
        mu_dot2 = Dot(chart.c2p(3600, S35_BOUND), radius=0.07).set_fill(MUON_COLOR, 1)
        by_lab = L(f"{S35_BY2400} by 2400", size=24, color=MUON_COLOR, weight="BOLD")
        by_lab.next_to(mu_dot, RIGHT, buff=0.2).shift(0.3 * DOWN)
        cut2 = DashedLine(chart.c2p(2400, 0), chart.c2p(2400, 20), dash_length=0.07).set_stroke(HINGE_COLOR, 1.5)
        cut2_lab = L("2400", size=20, color=HINGE_COLOR, weight="BOLD").next_to(chart.c2p(2400, 0), DOWN, buff=0.15)
        mu_name = VGroup(L("Muon + recipe", size=22, color=MUON_COLOR, weight="BOLD"), machine_badge("E", size=20),
                         L("screen S35", size=20, color=MUTED)).arrange(RIGHT, buff=0.15)
        mu_name.next_to(chart.c2p(12000, S35_BOUND), UP, buff=0.2)
        ad_name = VGroup(L("Adam + recipe", size=22, color=ADAM_COLOR, weight="BOLD"), machine_badge("X", size=20),
                         L("HINGE4k16, bound 3600–21,600", size=20, color=MUTED)).arrange(RIGHT, buff=0.15)
        ad_name.move_to(chart.c2p(15500, 9.5))
        end_lab = L(f"{S35_BOUND}/{S35_N}", size=24, color=INK, weight="BOLD").next_to(chart.c2p(24000, S35_BOUND),
                                                                                        RIGHT, buff=0.15)

        with self.voiceover(
            "That suggests the hinge is only needed early. With the hinge switched off after update twenty-four "
            "hundred, Muon discovered thirty-eight of forty, against thirty-nine with the full hinge, and Adam "
            "thirty-four against thirty-four."
        ) as vo:
            self.play(*self.new_title("Only needed early"), FadeOut(self.kick_mobs),
                      ReplacementTransform(self.timeline[0], sax), FadeOut(VGroup(*self.timeline[1:])),
                      *self.new_source("Report §9.1 · screens S34, S35 (machine E); test_early_recipe.py; X: data/slow_start.json"),
                      run_time=1.0)
            self.play(FadeIn(head), FadeIn(sticks), FadeIn(stick_labs), FadeIn(sax_lab), FadeIn(n_full),
                      FadeIn(base_full), FadeIn(w1[0]), FadeIn(w0[0]), run_time=0.6)
            self.play(ShowCreation(on_full), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "switched off"))
            self.play(FadeIn(n_win), FadeIn(base_win), FadeIn(w1[1]), FadeIn(w0[1]), run_time=0.5)
            self.play(ShowCreation(on_win), run_time=1.2)
            self.play(ShowCreation(cut), FadeIn(off_lab), Indicate(stick_labs[1], color=HINGE_COLOR), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "Muon discovered"))
            self.play(FadeIn(VGroup(rows[0].name_mob, rows[0].track)), *grow_bar(rows[0]), run_time=1.0)
            vo.wait_until(at_phrase(vo, 1, "against thirty-nine"))
            self.play(FadeIn(VGroup(rows[1].name_mob, rows[1].track)), *grow_bar(rows[1]), run_time=1.0)
            vo.wait_until(at_phrase(vo, 1, "and Adam"))
            self.play(FadeIn(VGroup(rows[2].name_mob, rows[2].track)), *grow_bar(rows[2]), run_time=0.9)
            vo.wait_until(at_phrase(vo, 1, "against thirty-four"))
            self.play(FadeIn(VGroup(rows[3].name_mob, rows[3].track)), *grow_bar(rows[3]), run_time=0.9)

        with self.voiceover("One loss in eighty runs.") as vo:
            self.play(ShowCreation(gap), FadeIn(mu_vs), FadeIn(ad_vs), FadeIn(loss[0], shift=0.1 * UP), run_time=0.9)
            self.play(FadeIn(loss[1]), Indicate(gap, color=BAD, scale_factor=1.4), run_time=0.9)

        with self.voiceover(
            "And at four streams with sixteen channels, Muon with the recipe bound seventeen of twenty, fifteen of "
            "them by update twenty-four hundred."
        ) as vo:
            self.play(FadeOut(VGroup(sched, res, mu_vs, ad_vs, gap, loss)), FadeIn(c_head), FadeIn(chart),
                      run_time=1.0)
            self.play(ShowCreation(adam_line), FadeIn(ad_name), run_time=1.6)
            self.play(FadeIn(adam_dots), run_time=0.3)
            vo.wait_until(at_phrase(vo, 0, "Muon with the recipe"))
            self.play(ShowCreation(mu_pre), run_time=0.6)
            self.play(ShowCreation(mu_line), FadeIn(mu_dot2), FadeIn(mu_name), run_time=1.2)
            self.play(FadeIn(end_lab), run_time=0.5)
            vo.wait_until(at_phrase(vo, 0, "fifteen of"))
            self.play(ShowCreation(cut2), FadeIn(cut2_lab), FadeIn(mu_dot, scale=0.5), FadeIn(by_lab, shift=0.1 * RIGHT),
                      run_time=0.9)
            self.play(Flash(mu_dot.get_center(), color=MUON_COLOR, flash_radius=0.4), run_time=0.7)
