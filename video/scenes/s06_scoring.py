"""Chapter 6 — how a run is scored (report §3, §5, §7; facts_task §3; test_binding_onset.py,
test_router_reliability.py, test_router_layout.py, test_stream_recipe.py, test_slow_start.py)."""
import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load(name):
    with open(os.path.join(DATA, name)) as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------------- real data
_SLOW = _load("slow_start.json")
# Machine L, arm A0, seed 295: whole-percent held-out accuracy at steps 1200, 2400, ...
# (results/L/recipe_scope_L_final.log: "29 88 99 99 82 100 100 100 -> DISCOVERED", transition 7200)
CURVE = [v / 100 for v in _SLOW["curves_L_x100"]["A0|295"]]
CURVE_STEPS = [1200 * (i + 1) for i in range(len(CURVE))]
TRANSITION = _SLOW["per_seed"]["L"]["A0"]["transition"][_SLOW["seeds"]["A0"].index(295)]

_GATE = _load("gate_dynamics.json")["k2_runs"]
H294 = _GATE["X:HINGE0|294"]          # index 1 = step 1200: margin 0.9747, eta by stream 0.9998
A294 = _GATE["X:A0|294"]              # index 1 = step 1200: margin 0.0013, eta by stream 0.0014

_SR = _load("stream_recipe.json")
_SR_SEEDS = _SR["seeds"]["A4k4"]


def _sr(seed, key):
    return _SR["per_seed_X"]["A4k4"][key][_SR_SEEDS.index(seed)]


ROUTED_MAP = _sr(246, "ch_map_end")      # [1, 0, 3, 2], BOUND ROUTED
ROUTED_ACC = _sr(246, "stream_acc_end")  # all 1.0
MERGE_MAP = _sr(247, "ch_map_end")       # [3, 1, 0, 0], MERGED (2 share)
MERGE_ACC = _sr(247, "stream_acc_end")   # [1.0, 1.0, 0.504, 0.495]
MERGE_FINAL = _sr(247, "final_acc")      # 0.7544

_E8 = _load("eight_streams.json")["arms"]["SC8"]
_I8 = _E8["seeds"].index(263)
COLLAPSE_MAP = _E8["ch_map_end"][_I8]    # [9]*8
COLLAPSE_ACC = _E8["final_acc"][_I8]     # 0.124

# end-of-run eta^2 of two failed A0 runs on X (results/X/slow_start_results.json, runs[...]["end"]):
POS_ETA_INDEX = A294["eta_key_by_index"][-1]     # A0|294: 1.00 by index -> POSITION
KEY_ETA_KEY = 0.9999995341702521                 # A0|282: eta_key_by_key -> KEY

B, Y = STREAM_COLORS[0], STREAM_COLORS[1]


# ------------------------------------------------------------------------------------- helpers
def at_phrase(vo, i, phrase):
    """Approximate time (s into the block) at which `phrase` is spoken in sentence i."""
    text, a, b = vo.synth.sentences[i]
    k = text.find(phrase)
    if k < 0:
        raise ValueError(phrase)
    return a + (b - a) * k / max(len(text), 1)


def eta2(x, lab):
    """Between-group over total variance of x grouped by lab (test_router_layout.py:229-237)."""
    x, lab = np.asarray(x, float).reshape(-1), np.asarray(lab).reshape(-1)
    mu = x.mean()
    tot = ((x - mu) ** 2).sum()
    if tot <= 0:
        return 0.0
    return float(sum((lab == g).sum() * (x[lab == g].mean() - mu) ** 2 for g in np.unique(lab)) / tot)


def grouped_sequence(rng, S=2, P=4):
    """Key positions of one grouped-layout body: [(stream, key)] in order, keys random, streams random per key."""
    out = []
    for k in rng.permutation(P):
        for s in rng.permutation(S):
            out.append((int(s), int(k)))
    return out


def box_label(text, color, width=1.0, height=0.42, size=20, fill_opacity=0.0):
    box = RoundedRectangle(width=width, height=height, corner_radius=0.08)
    box.set_fill(color, fill_opacity).set_stroke(color, 1.5)
    t = L(text, size=size, color=color)
    t.move_to(box)
    return VGroup(box, t)


def dot_at(axes, step, acc, color=INK, r=0.075):
    return Dot(axes.c2p(step, acc), radius=r).set_fill(color, 1).set_stroke(BG, 1.5)


class Scoring(ClankersScene):
    def construct(self):
        card = self.chapter_card(6, "How a run is scored")
        self.wait(0.6)
        title = section_title("Scoring a run")
        self.play(FadeOut(card, shift=0.3 * UP), FadeIn(title, shift=0.3 * UP))
        self.title_mob = title

        self.part_curve_and_outcomes()
        self.part_margin()
        self.part_failures()
        self.part_merges()
        self.wait(0.6)
        self.clear_all()

    # ============================================================== N6.1 + N6.2
    def part_curve_and_outcomes(self):
        ax = Axes(x_range=[0, 28800, 1200], y_range=[0, 1, 0.25], width=10.2, height=3.8,
                  axis_config=dict(stroke_color=MUTED, stroke_width=2, include_tip=False, include_ticks=False))
        ax.move_to([0.6, -0.85, 0])
        xlabels = VGroup(*[L(f"{x:,}", 20, MUTED).next_to(ax.c2p(x, 0), DOWN, buff=0.18)
                           for x in (0, 6000, 12000, 18000)])
        ylabels = VGroup(*[L(t, 20, MUTED).next_to(ax.c2p(0, y), LEFT, buff=0.15)
                           for y, t in ((0, "0"), (0.5, "0.5"), (1, "1"))])
        xname = L("training step", 20, MUTED).next_to(xlabels, DOWN, buff=0.12).set_x(ax.get_center()[0])
        yname = L("held-out accuracy", 20, MUTED).rotate(PI / 2).next_to(ylabels, LEFT, buff=0.18)
        chart_frame = VGroup(ax, xlabels, ylabels, xname, yname)

        # top strip: a batch of 32 sequences -> model -> 2048 held-out queries
        rng = np.random.default_rng(6)
        icon = VGroup()
        for r in range(32):
            for t in range(9):
                seg = Rectangle(width=0.11, height=0.017).set_stroke(width=0)
                seg.set_fill(STREAM_COLORS[int(rng.integers(2))], 0.85)
                seg.move_to([t * 0.125, -r * 0.026, 0])
                icon.add(seg)
        icon_lab = L("batch of 32", 20, MUTED).next_to(icon, DOWN, buff=0.1)
        batch = VGroup(icon, icon_lab).move_to([-1.9, 2.25, 0])
        model = VGroup(RoundedRectangle(width=1.6, height=0.7, corner_radius=0.12)
                       .set_fill(PANEL, 1).set_stroke(MEMORY_COLOR, 2), L("model", 24))
        model[1].move_to(model[0])
        model.move_to([0.55, icon.get_center()[1], 0])
        evalq = VGroup(RoundedRectangle(width=2.75, height=0.7, corner_radius=0.12)
                       .set_fill(WARN, 0.10).set_stroke(WARN, 2), L("2048 held-out queries", 20, WARN))
        evalq[1].move_to(evalq[0])
        evalq.move_to([4.85, model.get_center()[1], 0])
        arr1 = Arrow(icon.get_right(), model.get_left(), buff=0.15, thickness=3).set_color(MUTED)
        arr2 = Arrow(model.get_right(), evalq.get_left(), buff=0.15, thickness=3).set_color(WARN)
        every_lab = L("every 1,200 steps", 20, WARN).next_to(arr2, UP, buff=0.08)

        # budget markers and the playhead
        def vline(x, color, top=1.04):
            return DashedLine(ax.c2p(x, 0), ax.c2p(x, top), dash_length=0.08).set_stroke(color, 2)

        b24, b28 = vline(24000, WARN), vline(28800, WARN)
        b24_lab = L("24,000", 20, WARN).next_to(ax.c2p(24000, 0), DOWN, buff=0.18)
        b28_lab = L("28,800", 20, WARN).next_to(ax.c2p(28800, 0), DOWN, buff=0.18)
        budget_lab = L("budget", 20, WARN).next_to(VGroup(b24, b28), UP, buff=0.08)
        step_t = ValueTracker(0)
        head = always_redraw(lambda: Line(ax.c2p(step_t.get_value(), 0), ax.c2p(step_t.get_value(), 1.04))
                             .set_stroke(INK, 2))
        ctr = VGroup(L("step", 20, MUTED), DecimalNumber(0, num_decimal_places=0, font_size=22,
                                                         text_config=dict(font=SANS)))

        def upd_ctr(g):
            v = step_t.get_value()
            g[1].set_value(v)
            g.arrange(RIGHT, buff=0.12)
            g.next_to(ax.c2p(v, 1.04), UP, buff=0.08)

        ctr.add_updater(upd_ctr)

        eval_ticks = VGroup(*[Line(ax.c2p(x, 0), ax.c2p(x, 0) + 0.13 * UP).set_stroke(WARN, 2)
                              for x in range(1200, 28801, 1200)])
        thr = DashedLine(ax.c2p(0, 0.95), ax.c2p(28800, 0.95), dash_length=0.1).set_stroke(GOOD, 2)
        thr_lab = L("0.95", 20, GOOD).next_to(ax.c2p(28800, 0.95), RIGHT, buff=0.1)

        dots = VGroup(*[dot_at(ax, s, a, GOOD if a >= 0.95 else INK) for s, a in zip(CURVE_STEPS, CURVE)])
        links = VGroup(*[Line(dots[i].get_center(), dots[i + 1].get_center()).set_stroke(MUTED, 2)
                         for i in range(len(dots) - 1)])
        pip_lab = L("at or above 0.95, in a row", 20, MUTED)
        pips = VGroup(*[Circle(radius=0.13).set_stroke(GOOD, 2).set_fill(GOOD, 0) for _ in range(3)])
        pips.arrange(RIGHT, buff=0.22)
        pip_grp = VGroup(pip_lab, pips).arrange(DOWN, buff=0.16).move_to(ax.c2p(17500, 0.62))
        run_lab = VGroup(machine_badge("L", 20), L("arm A0, seed 295", 20, MUTED)).arrange(RIGHT, buff=0.15)
        run_lab.move_to(ax.c2p(17500, 0.26))
        stop_lab = L("stop", 24, GOOD, weight="BOLD").next_to(dots[-1], RIGHT, buff=0.18)
        src1 = source_note("Report §3; results/L/recipe_scope_L_final.log (A0, seed 295)")

        def set_pips(n, reset=False):
            return [p.animate.set_fill(GOOD, 0.9 if i < n else 0).set_stroke(BAD if reset else GOOD, 2)
                    for i, p in enumerate(pips)]

        with self.voiceover(
            "How is a run scored? A model trains at batch thirty-two for a fixed budget, usually twenty-four "
            "thousand or twenty-eight thousand eight hundred steps. Every twelve hundred steps it is evaluated on "
            "two thousand and forty-eight held-out queries, and it stops early after three evaluations in a row at "
            "or above 0.95."
        ) as vo:
            self.play(FadeIn(model, scale=0.8), run_time=0.8)
            vo.wait_until_sentence(1)
            self.play(LaggedStartMap(FadeIn, icon, lag_ratio=0.002), FadeIn(icon_lab), GrowArrow(arr1),
                      run_time=1.4)
            vo.wait_until(at_phrase(vo, 1, "for a fixed budget"))
            self.play(ShowCreation(ax), FadeIn(xlabels), FadeIn(ylabels), FadeIn(xname), FadeIn(yname),
                      run_time=1.0)
            self.add(head, ctr)
            vo.wait_until(at_phrase(vo, 1, "usually"))
            self.play(step_t.animate.set_value(24000), run_time=1.6, rate_func=linear)
            ctr.clear_updaters()
            self.play(FadeOut(head), FadeOut(ctr), ShowCreation(b24), FadeIn(b24_lab), FadeIn(budget_lab),
                      run_time=0.6)
            vo.wait_until(at_phrase(vo, 1, "twenty-eight"))
            self.play(ShowCreation(b28), FadeIn(b28_lab), FadeIn(src1), run_time=0.7)

            vo.wait_until_sentence(2)
            self.play(GrowArrow(arr2), FadeIn(every_lab),
                      LaggedStartMap(ShowCreation, eval_ticks, lag_ratio=0.08), run_time=1.6)
            vo.wait_until(at_phrase(vo, 2, "two thousand"))
            self.play(FadeIn(evalq, shift=0.2 * LEFT), run_time=0.8)
            self.play(Indicate(evalq, color=WARN, scale_factor=1.05), run_time=0.8)
            # the evaluation's score drops out of the held-out box onto the chart as its first dot
            flyer = dots[0].copy().move_to(evalq.get_bottom())
            self.play(FadeIn(flyer, scale=0.3), run_time=0.25)
            self.play(ReplacementTransform(flyer, dots[0], path_arc=0.6), FadeIn(run_lab), run_time=0.8)
            self.play(FadeIn(dots[1], scale=0.5), ShowCreation(links[0]), run_time=0.5)
            vo.wait_until(at_phrase(vo, 2, "and it stops"))
            self.play(ShowCreation(thr), FadeIn(thr_lab), FadeIn(pip_grp), run_time=0.8)
            streak = 0
            for i in range(2, len(dots)):
                streak = streak + 1 if CURVE[i] >= 0.95 else 0
                anims = [FadeIn(dots[i], scale=0.5), ShowCreation(links[i - 1])] + set_pips(streak, streak == 0)
                self.play(*anims, run_time=0.5)
            self.play(Write(stop_lab), Flash(dots[-1], color=GOOD, flash_radius=0.3), run_time=0.7)

        # ---------------------------------------------------------------- N6.2 bound, discovered, routed
        t_idx = CURVE_STEPS.index(TRANSITION)
        reach_idx = next(i for i, a in enumerate(CURVE) if a >= 0.95)          # 3600
        dip_idx = next(i for i in range(reach_idx, len(CURVE)) if CURVE[i] < 0.95)  # 6000
        ring_reach = Circle(radius=0.17).set_stroke(GOOD, 2.5).move_to(dots[reach_idx])
        reach_lab = L("reaches 0.95", 20, GOOD).next_to(ring_reach, UP, buff=0.12)
        ring_dip = Circle(radius=0.17).set_stroke(BAD, 2.5).move_to(dots[dip_idx])
        dip_lab = L(f"falls back to {CURVE[dip_idx]:.2f}", 20, BAD).next_to(ring_dip, DOWN, buff=0.12)
        dip_lab.align_to(ring_dip, RIGHT)
        no_lab = L("does not count", 20, BAD).next_to(ring_reach, UP, buff=0.12)
        ring_t = Circle(radius=0.17).set_stroke(GOOD, 3).move_to(dots[t_idx])
        t_line = vline(TRANSITION, GOOD)
        t_lab = L(f"transition: step {TRANSITION:,}", 20, GOOD)
        t_lab.next_to(ax.c2p(TRANSITION, 1.04), UP, buff=0.1).align_to(t_line, LEFT).shift(0.05 * RIGHT)

        # three definition cards
        cw, ch, cy = 4.25, 5.6, -0.35
        xs = (-4.45, 0.0, 4.45)
        panels = VGroup(*[RoundedRectangle(width=cw, height=ch, corner_radius=0.15).set_fill(PANEL, 1)
                          .set_stroke(PANEL_EDGE, 1.5).move_to([x, cy, 0]) for x in xs])
        top = cy + ch / 2
        heads = VGroup(verdict_badge("BOUND", 24, GOOD), verdict_badge("DISCOVERED", 24, GOOD),
                       verdict_badge("BOUND ROUTED", 24, GOOD))
        for h, x in zip(heads, xs):
            h.move_to([x, top - 0.45, 0])

        # card 1: the mini chart the big chart turns into
        mini = Axes(x_range=[0, 10800, 1200], y_range=[0, 1, 0.25], width=3.0, height=1.75,
                    axis_config=dict(stroke_color=MUTED, stroke_width=2, include_tip=False, include_ticks=False))
        mini.move_to([xs[0] + 0.25, 0.75, 0])
        m_dots = VGroup(*[dot_at(mini, s, a, GOOD if a >= 0.95 else INK, r=0.06)
                          for s, a in zip(CURVE_STEPS, CURVE)])
        m_links = VGroup(*[Line(m_dots[i].get_center(), m_dots[i + 1].get_center()).set_stroke(MUTED, 2)
                           for i in range(len(m_dots) - 1)])
        m_thr = DashedLine(mini.c2p(0, 0.95), mini.c2p(10800, 0.95), dash_length=0.07).set_stroke(GOOD, 2)
        m_thr_lab = L("0.95", 20, GOOD).next_to(mini.c2p(0, 0.95), LEFT, buff=0.1)
        m_tline = DashedLine(mini.c2p(TRANSITION, 0), mini.c2p(TRANSITION, 1.04), dash_length=0.07)
        m_tline.set_stroke(GOOD, 2)
        m_ring = Circle(radius=0.13).set_stroke(GOOD, 2.5).move_to(m_dots[t_idx])
        m_tlab = L(f"{TRANSITION:,}", 20, GOOD).next_to(mini.c2p(TRANSITION, 0), DOWN, buff=0.12)
        m_xlab = L("step", 20, MUTED).next_to(mini.c2p(10800, 0), DOWN, buff=0.12)
        c1_text = VGroup(L("an evaluation reaches 0.95", 24), L("and every later one stays there", 24),
                         L("transition: the first such step", 24, GOOD)).arrange(DOWN, buff=0.16)
        c1_text.move_to([xs[0], -1.35, 0])
        c1_run = VGroup(machine_badge("L", 20), L("A0, seed 295", 20, MUTED)).arrange(RIGHT, buff=0.15)
        c1_run.move_to([xs[0], -2.8, 0])

        # card 2: two read gates at value positions, k = S = 2
        c2_sub = L("k = S = 2", 22, MUTED).next_to(heads[1], DOWN, buff=0.25)
        c2_cap = L("read gate at value positions", 20, MUTED).move_to([xs[1], 0.95, 0])
        g_names = VGroup(L("stream 0", 20, B), L("stream 1", 20, Y))

        def c2_bars(p0, p1):
            bars = VGroup(gate_bar([p0, 1 - p0], width=2.2, height=0.36), gate_bar([p1, 1 - p1], width=2.2, height=0.36))
            rows = VGroup(*[VGroup(n.copy(), b).arrange(RIGHT, buff=0.25) for n, b in zip(g_names, bars)])
            rows.arrange(DOWN, buff=0.3, aligned_edge=RIGHT).move_to([xs[1], 0.05, 0])
            return rows

        c2_rows = c2_bars(0.5, 0.5)
        ch_tags = VGroup(L("ch0", 20, BG, weight="BOLD"), L("ch1", 20, BG, weight="BOLD"))
        # DISC_COS = 0.5 on VAL cos, the cosine of the two streams' mean read gates at value positions
        # (test_router_reliability.py:338; facts_task §3.4)
        c2_text = VGroup(L("bound, and the two streams'", 24), L("read gates are separated", 24),
                         L("(their cosine is below 0.5)", 22, MUTED)).arrange(DOWN, buff=0.16)
        c2_text.move_to([xs[1], -1.35, 0])
        blind_lab = L("not separated: cosine = 1", 22, BAD).move_to([xs[1], -0.75, 0])

        # card 3: one-to-one stream -> channel map, k >= S > 2 (X, A4k4 seed 246)
        c3_sub = L("k ≥ S > 2", 22, MUTED).next_to(heads[2], DOWN, buff=0.25)
        rows_y = [0.95, 0.42, -0.11, -0.64]
        s_toks = VGroup(*[token(f"CTX{s}", "ctx", s, width=0.9, height=0.4, size=20).move_to([xs[2] - 1.15, y, 0])
                          for s, y in enumerate(rows_y)])
        c_boxes = VGroup(*[box_label(f"ch{c}", MUTED, width=0.9, height=0.4).move_to([xs[2] + 1.15, y, 0])
                           for c, y in enumerate(rows_y)])
        c3_arrows = VGroup(*[Arrow(s_toks[s].get_right(), c_boxes[c].get_left(), buff=0.08, thickness=2.5)
                             .set_color(STREAM_COLORS[s]) for s, c in enumerate(ROUTED_MAP)])
        c3_badge = VGroup(machine_badge("X", 20), L("A4k4, seed 246", 20, MUTED)).arrange(RIGHT, buff=0.15)
        c3_badge.move_to([xs[2], -2.8, 0])
        bar_h = 0.85
        base_y = -2.2
        acc_bars = VGroup()
        for s, a in enumerate(ROUTED_ACC):
            r = Rectangle(width=0.34, height=bar_h * a).set_fill(STREAM_COLORS[s], 0.9).set_stroke(width=0)
            r.move_to([xs[2] - 0.75 + 0.5 * s, base_y + bar_h * a / 2, 0])
            acc_bars.add(r)
        acc_base = Line([xs[2] - 1.05, base_y, 0], [xs[2] + 1.05, base_y, 0]).set_stroke(MUTED, 1.5)
        acc_thr = DashedLine([xs[2] - 1.1, base_y + 0.9 * bar_h, 0], [xs[2] + 1.1, base_y + 0.9 * bar_h, 0],
                             dash_length=0.06).set_stroke(WARN, 2)
        acc_thr_lab = L("0.9", 20, WARN).next_to(acc_thr, RIGHT, buff=0.1)
        acc_cap = L("every stream's accuracy", 20, MUTED).next_to(acc_base, DOWN, buff=0.1)
        c3_own = L("a channel of its own", 20, GOOD).next_to(c_boxes, DOWN, buff=0.18).set_x(xs[2])
        src2 = source_note("Report §3; L A0 s295 (recipe_scope_L_final.log); X A4k4 s246 (stream_recipe)")

        with self.voiceover(
            "A run is bound if an evaluation reaches 0.95 and every later one stays there; the first such step "
            "is its transition. With two streams and two channels, a run is discovered if it is bound and the "
            "streams' read gates are separated at the value positions. With more than two streams and at least "
            "as many channels as streams, a run is bound and routed if it is bound, every stream has a channel of "
            "its own, and every stream is at least ninety percent accurate."
        ) as vo:
            self.play(FadeOut(VGroup(batch, model, arr1, arr2, every_lab, evalq, pip_grp, stop_lab)),
                      ShowCreation(ring_reach), FadeIn(reach_lab), run_time=0.8)
            vo.wait_until(at_phrase(vo, 0, "and every later"))
            self.play(ShowCreation(ring_dip), FadeIn(dip_lab), run_time=0.7)
            self.play(ring_reach.animate.set_stroke(BAD), FadeTransform(reach_lab, no_lab), run_time=0.7)
            vo.wait_until(at_phrase(vo, 0, "the first such"))
            self.play(ShowCreation(t_line), ShowCreation(ring_t), FadeIn(t_lab, shift=0.1 * DOWN),
                      FadeOut(no_lab), ring_reach.animate.set_stroke(opacity=0.0), run_time=0.9)
            self.play(Indicate(ring_t, color=GOOD, scale_factor=1.3), run_time=0.8)

            # the chart shrinks into the first card
            vo.wait_until(vo.time_of(1) - 0.3)
            fades = VGroup(xlabels, ylabels, xname, yname, b24, b28, b24_lab, b28_lab, budget_lab, eval_ticks,
                           thr_lab, run_lab, ring_reach, ring_dip, dip_lab, t_lab)
            self.play(FadeIn(panels[0]), FadeOut(fades, run_time=0.5),
                      ReplacementTransform(ax, mini), ReplacementTransform(dots, m_dots),
                      ReplacementTransform(links, m_links), ReplacementTransform(thr, m_thr),
                      ReplacementTransform(t_line, m_tline), ReplacementTransform(ring_t, m_ring),
                      FadeTransform(src1, src2), run_time=1.3)
            self.play(FadeIn(heads[0], shift=0.1 * DOWN), FadeIn(m_thr_lab), FadeIn(m_tlab), FadeIn(m_xlab),
                      LaggedStartMap(FadeIn, c1_text, lag_ratio=0.25), FadeIn(c1_run), run_time=1.0)

            # card 2
            self.play(FadeIn(panels[1]), FadeIn(heads[1], shift=0.1 * DOWN), FadeIn(c2_sub), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "the streams' read gates"))
            self.play(FadeIn(c2_cap), FadeIn(c2_rows), FadeIn(blind_lab), run_time=0.7)
            vo.wait_until(at_phrase(vo, 1, "separated"))
            sep_rows = c2_bars(0.97, 0.03)
            ch_tags[0].move_to(sep_rows[0][1][0][0])
            ch_tags[1].move_to(sep_rows[1][1][0][1])
            self.play(Transform(c2_rows, sep_rows), FadeOut(blind_lab), run_time=1.0)
            self.play(LaggedStartMap(FadeIn, c2_text, lag_ratio=0.25), FadeIn(ch_tags), run_time=0.9)

            # card 3
            vo.wait_until_sentence(2)
            self.play(FadeIn(panels[2]), FadeIn(heads[2], shift=0.1 * DOWN), run_time=0.7)
            vo.wait_until(at_phrase(vo, 2, "at least as many"))
            self.play(FadeIn(c3_sub), LaggedStartMap(FadeIn, s_toks, shift=0.1 * RIGHT, lag_ratio=0.15),
                      LaggedStartMap(FadeIn, c_boxes, shift=0.1 * LEFT, lag_ratio=0.15), FadeIn(c3_badge),
                      run_time=1.2)
            vo.wait_until(at_phrase(vo, 2, "if it is bound"))
            self.play(Indicate(heads[0], color=GOOD, scale_factor=1.15),
                      FlashAround(panels[0], color=GOOD, time_width=0.6), run_time=1.1)
            vo.wait_until(at_phrase(vo, 2, "every stream has"))
            fills = [c_boxes[c][0].animate.set_fill(STREAM_COLORS[s], 0.25).set_stroke(STREAM_COLORS[s], 2)
                     for s, c in enumerate(ROUTED_MAP)]
            recolor = [c_boxes[c][1].animate.set_color(STREAM_COLORS[s]) for s, c in enumerate(ROUTED_MAP)]
            self.play(LaggedStartMap(GrowArrow, c3_arrows, lag_ratio=0.25), run_time=1.3)
            self.play(*fills, *recolor, FadeIn(c3_own), run_time=0.6)
            vo.wait_until(at_phrase(vo, 2, "every stream is at least"))
            for r in acc_bars:
                r.save_state()
                r.stretch(1e-3, 1, about_edge=DOWN)
            self.play(ShowCreation(acc_base), FadeIn(acc_cap), *[Restore(r) for r in acc_bars], run_time=1.0)
            self.play(ShowCreation(acc_thr), FadeIn(acc_thr_lab), run_time=0.6)

        self.wait(0.3)
        keep = [self.title_mob]
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep and m is not self.frame], run_time=0.7)

    # ============================================================== N6.3 routing margin
    def part_margin(self):
        # a run's timeline with a screen's probe at step 1200
        tl = NumberLine(x_range=[0, 24000, 1200], width=7.4, include_tip=False, tick_size=0.06)
        tl.set_stroke(MUTED, 2).move_to([2.2, 2.45, 0])
        tl_lab0 = L("0", 20, MUTED).next_to(tl.n2p(0), DOWN, buff=0.12)
        tl_lab1 = L("24,000 steps", 20, MUTED).next_to(tl.n2p(24000), DOWN, buff=0.12).align_to(tl, RIGHT)
        probe = Triangle(fill_color=WARN, fill_opacity=1, stroke_width=0).rotate(PI).scale(0.12)
        probe.next_to(tl.n2p(1200), UP, buff=0.04)
        probe_lab = L("a screen scores routing at a given step, here 1,200", 20, WARN)
        probe_lab.next_to(probe, RIGHT, buff=0.15).shift(0.14 * UP)

        # the tokens: K0 in stream 1 (distractor), K0 in stream 0 (target), query CTX0 K0 ?
        tri_dis = triple(1, 0, 13)
        tri_tgt = triple(0, 0, 2)
        dots_mid = L("…", 30, MUTED)
        query = VGroup(token("CTX0", "ctx", 0), token("K0"), token("?", "query")).arrange(RIGHT, buff=0.06)
        row = VGroup(tri_dis, tri_tgt, dots_mid, query).arrange(RIGHT, buff=0.45).move_to([0, 1.25, 0])
        v_dis, v_tgt, k_q = tri_dis[2], tri_tgt[2], query[1]

        g = ValueTracker(0.0)   # 0 = uniform (stream-blind) gates, 1 = routed gates

        # Illustrative gate vectors. The routed end point is chosen so that the dial reads the real margin of
        # X's HINGE0 seed 294 at step 1200 (0.97, shown below) and the printed sum is consistent:
        # g_q.g_tgt = 0.981 -> 0.98, g_q.g_dis = 0.0105 -> 0.01, m = 0.970.
        def gates():
            t = g.get_value()
            q = 0.5 + t * (0.99 - 0.5)
            tg = 0.5 + t * (0.9905 - 0.5)
            ds = 0.5 + t * (0.0005 - 0.5)
            return (q, 1 - q), (tg, 1 - tg), (ds, 1 - ds)

        def bar_under(tok, which):
            def make():
                p = gates()[which]
                return gate_bar(list(p), width=1.3, height=0.3).next_to(tok, DOWN, buff=0.35)
            return always_redraw(make)

        bar_q, bar_tgt, bar_dis = bar_under(k_q, 0), bar_under(v_tgt, 1), bar_under(v_dis, 2)
        lab_q = M(R"g_q", 34, GATE_COLOR).next_to(bar_q, DOWN, buff=0.15)
        lab_tgt = M(R"g_{\text{tgt}}", 34, B).next_to(bar_tgt, DOWN, buff=0.15)
        lab_dis = M(R"g_{\text{dis}}", 34, Y).next_to(bar_dis, DOWN, buff=0.15)
        sub_q = L("read gate of the query", 20, GATE_COLOR).next_to(lab_q, DOWN, buff=0.08)
        sub_tgt = L("target value", 20, B).next_to(lab_tgt, DOWN, buff=0.08)
        sub_dis = L("same key, other stream", 20, Y).next_to(lab_dis, DOWN, buff=0.08)
        hl_q = SurroundingRectangle(k_q, buff=0.06).set_stroke(GATE_COLOR, 2.5)
        # which colour is which channel in the gate bars
        ch_leg = VGroup(*[VGroup(Square(0.24).set_fill(STREAM_COLORS[c], 0.9).set_stroke(width=0),
                                 L(f"ch{c}", 20, MUTED)).arrange(RIGHT, buff=0.12) for c in (0, 1)])
        ch_leg.arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        ch_leg.move_to([-5.55, v_dis.get_bottom()[1] - 0.5, 0])
        hl_tgt = SurroundingRectangle(v_tgt, buff=0.06).set_stroke(B, 2.5)
        hl_dis = SurroundingRectangle(v_dis, buff=0.06).set_stroke(Y, 2.5)

        formula = M(R"m = g_q \cdot g_{\text{tgt}} - \text{mean}(g_q \cdot g_{\text{dis}})", 40,
                    t2c={"g_q": GATE_COLOR, R"g_{\text{tgt}}": B, R"g_{\text{dis}}": Y})
        formula.move_to([-2.6, -1.25, 0])

        def dot_vals():
            q, tg, ds = gates()
            dt = q[0] * tg[0] + q[1] * tg[1]
            dd = q[0] * ds[0] + q[1] * ds[1]
            return dt, dd, dt - dd

        d_tgt = DecimalNumber(0.5, num_decimal_places=2, font_size=30, text_config=dict(font=SANS), color=B)
        d_dis = DecimalNumber(0.5, num_decimal_places=2, font_size=30, text_config=dict(font=SANS), color=Y)
        d_m = DecimalNumber(0.0, num_decimal_places=2, font_size=30, text_config=dict(font=SANS), color=INK)
        nums = VGroup(L("=", 28), d_tgt, L("−", 28), d_dis, L("=", 28), d_m).arrange(RIGHT, buff=0.18)
        nums.next_to(formula, DOWN, buff=0.35).align_to(formula, LEFT).shift(0.75 * RIGHT)

        def upd_nums(grp):
            dt, dd, m = dot_vals()
            grp[1].set_value(dt)
            grp[3].set_value(dd)
            grp[5].set_value(m)
            grp[5].set_color(GOOD if m >= 0.9 else INK)

        nums.add_updater(upd_nums)

        # the dial
        dc = np.array([4.55, -1.62, 0])
        rad = 1.0
        track = Arc(start_angle=PI, angle=-PI, radius=rad, arc_center=dc).set_stroke(FAINT, 10)

        def fill_arc():
            m = max(min(dot_vals()[2], 1.0), 0.002)
            return Arc(start_angle=PI, angle=-PI * m, radius=rad, arc_center=dc).set_stroke(
                GOOD if m >= 0.9 else GATE_COLOR, 10)

        dial_fill = always_redraw(fill_arc)

        def needle_mob():
            m = max(min(dot_vals()[2], 1.0), 0.0)
            ang = PI - PI * m
            return Line(dc, dc + 0.85 * rad * np.array([np.cos(ang), np.sin(ang), 0])).set_stroke(INK, 3)

        needle = always_redraw(needle_mob)
        hub = Dot(dc, radius=0.06).set_fill(INK, 1)
        ang9 = PI - 0.9 * PI
        dir9 = np.array([np.cos(ang9), np.sin(ang9), 0])
        tick9 = Line(dc + (rad - 0.2) * dir9, dc + (rad + 0.2) * dir9).set_stroke(WARN, 3)
        tick9_lab = L("0.9", 20, WARN).move_to(dc + (rad + 0.42) * dir9)
        end0 = L("0", 20, MUTED).next_to(dc + rad * LEFT, DOWN, buff=0.12)
        end1 = L("1", 20, MUTED).next_to(dc + rad * RIGHT, DOWN, buff=0.12)
        dial_name = L("margin m", 20, MUTED).next_to(dc, DOWN, buff=0.15)
        dial = VGroup(track, tick9, tick9_lab, end0, end1, dial_name, hub)

        # eta^2 by stream meter
        e_t = ValueTracker(0.0)
        e_w = 2.3
        e_track = Rectangle(width=e_w, height=0.28).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1)
        e_track.move_to([4.75, -2.42, 0])
        e_fill = always_redraw(lambda: Rectangle(width=max(e_w * e_t.get_value(), 1e-3), height=0.28)
                               .set_fill(GOOD if e_t.get_value() > 0.9 else GATE_COLOR, 0.9).set_stroke(width=0)
                               .align_to(e_track, LEFT).set_y(e_track.get_y()))
        e_thr = Line(e_track.get_left() + 0.9 * e_w * RIGHT + 0.2 * UP,
                     e_track.get_left() + 0.9 * e_w * RIGHT + 0.2 * DOWN).set_stroke(WARN, 3)
        e_name = L("η² by stream", 20, MUTED).next_to(e_track, LEFT, buff=0.2)
        e_val = DecimalNumber(0, num_decimal_places=2, font_size=24, text_config=dict(font=SANS))
        e_val.add_updater(lambda d: d.set_value(e_t.get_value()).next_to(e_track, RIGHT, buff=0.15))
        meter = VGroup(e_track, e_thr, e_name)

        tag = VGroup(verdict_badge("ROUTED*", 22, GOOD),
                     L("m ≥ 0.9 and η² by stream > 0.9", 20, INK)).arrange(RIGHT, buff=0.2)
        tag.move_to([3.75, -3.0, 0])
        real = VGroup(
            VGroup(machine_badge("X", 22),
                   L(f"HINGE0 seed 294, step 1,200:  m {H294['margin'][1]:.2f},  η² {H294['eta_key_by_stream'][1]:.2f}", 22),
                   verdict_badge("ROUTED*", 22, GOOD)).arrange(RIGHT, buff=0.15),
            VGroup(machine_badge("X", 22),
                   L(f"A0 seed 294, step 1,200:  m {abs(A294['margin'][1]):.2f},  η² {A294['eta_key_by_stream'][1]:.2f}", 22),
                   verdict_badge("not routed", 22, BAD)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        real.to_edge(LEFT, buff=0.45).set_y(-2.85)
        src3 = source_note("Report §3; test_slow_start.py routed_star; data/gate_dynamics.json (X)")

        with self.voiceover(
            "Screens also score routing partway through a run. The routing margin compares how well the query's "
            "gate matches its target value with how well it matches the same key's values in other streams. With "
            "a margin of at least 0.9, and over ninety percent of the gate's variance explained by stream, a run "
            "counts as routed at that step."
        ) as vo:
            self.play(ShowCreation(tl), FadeIn(tl_lab0), FadeIn(tl_lab1), run_time=0.9)
            self.play(FadeIn(probe, shift=0.2 * DOWN), FadeIn(probe_lab), run_time=0.7)

            vo.wait_until_sentence(1)
            self.play(LaggedStartMap(FadeIn, row, shift=0.1 * DOWN, lag_ratio=0.15), run_time=1.0)
            vo.wait_until(at_phrase(vo, 1, "query's gate"))
            self.play(ShowCreation(hl_q), FadeIn(bar_q), FadeIn(lab_q), FadeIn(sub_q), FadeIn(ch_leg), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "its target value"))
            self.play(ShowCreation(hl_tgt), FadeIn(bar_tgt), FadeIn(lab_tgt), FadeIn(sub_tgt), run_time=0.8)
            self.play(Write(formula[:9]), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "the same key's values"))
            self.play(ShowCreation(hl_dis), FadeIn(bar_dis), FadeIn(lab_dis), FadeIn(sub_dis), run_time=0.8)
            self.play(Write(formula[9:]), run_time=1.0)
            self.play(FadeIn(nums), FadeIn(src3), run_time=0.6)

            vo.wait_until_sentence(2)
            self.play(FadeIn(dial), FadeIn(dial_fill), FadeIn(needle), run_time=0.7)
            self.play(g.animate.set_value(1.0), run_time=2.2)
            self.play(Flash(dc + rad * dir9, color=GOOD, flash_radius=0.3), run_time=0.6)
            vo.wait_until(at_phrase(vo, 2, "over ninety percent"))
            self.play(FadeIn(meter), FadeIn(e_fill), FadeIn(e_val), run_time=0.5)
            self.play(e_t.animate.set_value(H294["eta_key_by_stream"][1]), run_time=1.3)
            vo.wait_until(at_phrase(vo, 2, "a run counts"))
            self.play(FadeIn(tag, shift=0.15 * UP), run_time=0.7)
            self.play(LaggedStartMap(FadeIn, real, shift=0.1 * RIGHT, lag_ratio=0.4), run_time=1.2)
        self.wait(0.6)
        nums.clear_updaters()
        e_val.clear_updaters()
        keep = [self.title_mob]
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep and m is not self.frame], run_time=0.7)

    @staticmethod
    def _value_follower(tr, track):
        def upd(d):
            d.set_value(tr.get_value())
            d.next_to(track, RIGHT, buff=0.15)
        return upd

    # ============================================================== N6.4 failure names
    def part_failures(self):
        rng = np.random.default_rng(2026)
        batch = [grouped_sequence(rng) for _ in range(256)]
        n_rows, n_cols = 6, 8
        shown = batch[:n_rows]
        Sarr = np.array([[s for s, _ in seq] for seq in batch])
        Karr = np.array([[k for _, k in seq] for seq in batch])
        Jarr = np.tile(np.arange(n_cols), (len(batch), 1))
        Harr = (Jarr >= n_cols // 2).astype(int)
        rules = {
            "stream": lambda j, s, k: s,
            "position": lambda j, s, k: int(j >= n_cols // 2),
            "key": lambda j, s, k: int(k >= 2),
        }

        def etas(rule):
            x = np.array([[1 - rule(j, s, k) for j, (s, k) in enumerate(seq)] for seq in batch], float)
            return {"stream": eta2(x, Sarr),
                    "position": max(eta2(x, Jarr), eta2(x, Harr)),
                    "key": eta2(x, Karr)}

        vals = {name: etas(rule) for name, rule in rules.items()}

        pitch, size = 0.68, 0.6
        cells = VGroup()
        for r, seq in enumerate(shown):
            for j, (s, k) in enumerate(seq):
                ring = RoundedRectangle(width=size, height=size, corner_radius=0.08)
                ring.set_stroke(STREAM_COLORS[s], 3).set_fill(opacity=0)
                inner = RoundedRectangle(width=size - 0.16, height=size - 0.16, corner_radius=0.05)
                inner.set_fill(PANEL, 1).set_stroke(width=0)
                lab = L(f"K{k}", 20, INK)
                c = VGroup(ring, inner, lab)
                c.move_to([j * pitch, -r * pitch, 0])
                c.s, c.k, c.j = s, k, j
                cells.add(c)
        cells.move_to([-3.25, -0.62, 0])
        col_labs = VGroup(*[L(str(j), 20, MUTED).next_to(cells[j], UP, buff=0.12) for j in range(n_cols)])
        col_name = L("triple index", 20, MUTED).next_to(col_labs, RIGHT, buff=0.25)
        halves = VGroup()
        for a, b, name in ((0, 3, "first half"), (4, 7, "second half")):
            br = Brace(VGroup(col_labs[a], col_labs[b]), UP, buff=0.06).set_color(MUTED)
            halves.add(VGroup(br, L(name, 20, MUTED).next_to(br, UP, buff=0.05)))
        row_name = L("sequences in a batch", 20, MUTED).rotate(PI / 2).next_to(cells, LEFT, buff=0.2)
        swatches = VGroup(*[VGroup(Square(0.24).set_fill(STREAM_COLORS[c], 0.92).set_stroke(width=0),
                                   L(f"ch{c}", 20, MUTED)).arrange(RIGHT, buff=0.1) for c in (0, 1)])
        swatches.arrange(RIGHT, buff=0.3)
        legend = VGroup(L("ring: the token's stream", 20, MUTED),
                        VGroup(L("fill: the channel the gate chose", 20, MUTED), swatches).arrange(RIGHT, buff=0.3))
        legend.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        legend.next_to(cells, DOWN, buff=0.25).align_to(cells, LEFT)

        def fill_anims(rule):
            out = []
            for c in cells:
                ch = rule(c.j, c.s, c.k)
                out.append(c[1].animate.set_fill(STREAM_COLORS[ch], 0.92))
                out.append(c[2].animate.set_color(BG))
            return out

        # meters
        mx0, mw = 1.25, 3.6
        meter_rows = {}
        meters = VGroup()
        for name, label, y in (("stream", "η² by stream", 1.75), ("position", "η² by position (index or half)", 0.85),
                               ("key", "η² by key", -0.05)):
            lab = L(label, 20, MUTED).move_to([mx0, y, 0], aligned_edge=LEFT)
            track = Rectangle(width=mw, height=0.3).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1)
            track.move_to([mx0 + mw / 2, y - 0.38, 0])
            tr = ValueTracker(0.0)
            col = GOOD if name == "stream" else HINGE_COLOR
            fill = always_redraw(lambda tr=tr, track=track, col=col: Rectangle(width=max(mw * tr.get_value(), 1e-3), height=0.3)
                                 .set_fill(col, 0.9).set_stroke(width=0).align_to(track, LEFT).set_y(track.get_y()))
            val = DecimalNumber(0, num_decimal_places=2, font_size=24, text_config=dict(font=SANS))
            val.add_updater(self._value_follower(tr, track))
            meter_rows[name] = (VGroup(lab, track), tr, fill, val)
            meters.add(VGroup(lab, track, fill, val))
        state = verdict_badge("stream split: what we want", 22, GOOD).move_to([mx0 + 2.4, 2.45, 0])

        def set_meters(name):
            return [meter_rows[k][1].animate.set_value(vals[name][k]) for k in ("stream", "position", "key")]

        formula = M(R"\eta^2 = \frac{\text{between-group variance}}{\text{total variance}}", 30)
        formula.move_to([mx0 + 2.3, -1.45, 0])
        half_x = mx0 + 0.5 * mw
        thr_lines = VGroup(*[DashedLine(meter_rows[k][0][1].get_top() + 0.02 * UP + (half_x - meter_rows[k][0][1].get_center()[0]) * RIGHT,
                                        meter_rows[k][0][1].get_bottom() + 0.12 * DOWN + (half_x - meter_rows[k][0][1].get_center()[0]) * RIGHT,
                                        dash_length=0.05).set_stroke(BAD, 2.5) for k in ("position", "key")])
        thr_lab = L("0.5", 20, BAD).next_to(thr_lines[1], DOWN, buff=0.05)
        # the rule, not this map's verdict: "at 0.5 or more -> that failure name"
        pos_badge = VGroup(L("≥ 0.5 →", 20, BAD), verdict_badge("POSITION", 20, BAD)).arrange(RIGHT, buff=0.12)
        key_badge = VGroup(L("≥ 0.5 →", 20, BAD), verdict_badge("KEY", 20, BAD)).arrange(RIGHT, buff=0.12)
        pos_badge.next_to(meter_rows["position"][0][0], RIGHT, buff=0.25)
        key_badge.next_to(meter_rows["key"][0][0], RIGHT, buff=0.25)
        real = VGroup(
            L("real failed runs, end of training", 20, MUTED),
            VGroup(machine_badge("X", 20), L(f"A0 seed 294: η² by index {POS_ETA_INDEX:.2f}", 20),
                   verdict_badge("POSITION", 20, BAD)).arrange(RIGHT, buff=0.15),
            VGroup(machine_badge("X", 20), L(f"A0 seed 282: η² by key {KEY_ETA_KEY:.2f}", 20),
                   verdict_badge("KEY", 20, BAD)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        real.next_to(formula, DOWN, buff=0.3).set_x(mx0 - 0.1, LEFT)
        src4 = source_note("Report §3; test_router_layout.py fail_class; results/X/slow_start_results.json")

        with self.voiceover(
            "The failures have names. The code looks at the gate's channel choice at key positions, and asks how "
            "much of its variance is explained by position, either the triple's index or which half of the "
            "sequence it is in, or by the key's identity. That fraction is eta squared. At one half or more, a "
            "failed run is a POSITION or a KEY failure.",
            spoken="The failures have names. The code looks at the gate's channel choice at key positions, and asks "
            "how much of its variance is explained by position, either the triple's index or which half of the "
            "sequence it is in, or by the key's identity. That fraction is eta squared. At one half or more, a "
            "failed run is a position or a key failure.",
        ) as vo:
            self.play(LaggedStartMap(FadeIn, cells, lag_ratio=0.01), FadeIn(row_name), run_time=1.2)
            vo.wait_until_sentence(1)
            self.play(FadeIn(col_labs), FadeIn(col_name), FadeIn(legend), run_time=0.7)
            self.play(*fill_anims(rules["stream"]), FadeIn(state, shift=0.1 * DOWN), run_time=1.0)
            self.play(LaggedStartMap(FadeIn, meters, lag_ratio=0.2), FadeIn(src4), run_time=0.8)
            self.play(*set_meters("stream"), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "explained by position"))
            new_state = verdict_badge("a POSITION split", 22, HINGE_COLOR).move_to(state)
            self.play(*fill_anims(rules["position"]), *set_meters("position"),
                      FadeTransform(state, new_state), run_time=1.2)
            state = new_state
            vo.wait_until(at_phrase(vo, 1, "the triple's index"))
            self.play(LaggedStartMap(Indicate, col_labs, color=WARN, lag_ratio=0.1), run_time=1.0)
            vo.wait_until(at_phrase(vo, 1, "which half"))
            self.play(LaggedStartMap(FadeIn, halves, shift=0.1 * DOWN, lag_ratio=0.3), run_time=0.9)
            vo.wait_until(at_phrase(vo, 1, "the key's identity"))
            new_state = verdict_badge("a KEY split", 22, HINGE_COLOR).move_to(state)
            self.play(*fill_anims(rules["key"]), *set_meters("key"), FadeTransform(state, new_state),
                      run_time=1.2)
            state = new_state
            self.play(LaggedStartMap(Indicate, VGroup(*[c[2] for c in cells if c.k >= 2]), color=WHITE,
                                     scale_factor=1.25, lag_ratio=0.02), run_time=1.0)

            vo.wait_until_sentence(2)
            self.play(Write(formula), run_time=1.0)
            self.play(*[Indicate(meter_rows[k][0][0], color=INK, scale_factor=1.05)
                        for k in ("stream", "position", "key")], run_time=0.7)

            vo.wait_until_sentence(3)
            self.play(*[ShowCreation(t) for t in thr_lines], FadeIn(thr_lab), run_time=0.6)
            self.play(FadeIn(pos_badge, shift=0.1 * LEFT), FadeIn(key_badge, shift=0.1 * LEFT), run_time=0.6)
            # the map on screen is a key split: its key meter is past 0.5
            self.play(Indicate(key_badge[1], color=BAD, scale_factor=1.15),
                      FlashAround(meter_rows["key"][0][1], color=BAD, time_width=0.6), run_time=0.8)
            self.play(LaggedStartMap(FadeIn, real, shift=0.1 * UP, lag_ratio=0.3), run_time=1.2)
        self.wait(0.6)
        for _, _, _, val in meter_rows.values():
            val.clear_updaters()
        keep = [self.title_mob]
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep and m is not self.frame], run_time=0.7)

    # ============================================================== N6.5 merges and collapse
    def part_merges(self):
        # left: four streams on four channels, two of them sharing (X, A4k4 seed 247)
        lx_s, lx_c = -5.6, -3.0
        ys4 = [1.55, 0.9, 0.25, -0.4]
        head_l = L("four streams, four channels", 22, MUTED).move_to([(lx_s + lx_c) / 2, 2.35, 0])
        toks4 = VGroup(*[token(f"CTX{s}", "ctx", s, width=0.95, height=0.46).move_to([lx_s, y, 0])
                         for s, y in enumerate(ys4)])
        chans4 = VGroup(*[box_label(f"ch{c}", MUTED, width=1.0, height=0.46).move_to([lx_c, y, 0])
                          for c, y in enumerate(ys4)])
        arrows4 = VGroup(*[Arrow(toks4[s].get_right(), chans4[c].get_left(), buff=0.08, thickness=3)
                           .set_color(STREAM_COLORS[s]) for s, c in enumerate(MERGE_MAP)])
        shared = [c for c in set(MERGE_MAP) if MERGE_MAP.count(c) > 1][0]
        sharers = [s for s, c in enumerate(MERGE_MAP) if c == shared]
        share_lab = L(f"{len(sharers)} share", 22, WARN, weight="BOLD").next_to(chans4[shared], RIGHT, buff=0.2)
        merged_badge = verdict_badge(f"MERGED ({len(sharers)} share)", 22, WARN).move_to(head_l)
        run_l = VGroup(machine_badge("X", 20), L("A4k4, seed 247", 20, MUTED)).arrange(RIGHT, buff=0.15)
        run_l.next_to(chans4, RIGHT, buff=0.35).align_to(chans4, DOWN)

        bar_h, base_y = 1.25, -3.05
        acc4 = VGroup()
        acc4_vals = VGroup()
        for s, a in enumerate(MERGE_ACC):
            r = Rectangle(width=0.42, height=bar_h * a).set_fill(STREAM_COLORS[s], 0.9).set_stroke(width=0)
            r.move_to([-5.7 + 0.62 * s, base_y + bar_h * a / 2, 0])
            acc4.add(r)
            a2 = float(Decimal(str(a)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
            acc4_vals.add(L(f"{a2:.2f}", 20, STREAM_COLORS[s]).next_to(r, UP, buff=0.08))
        acc4_base = Line([-6.05, base_y, 0], [-3.5, base_y, 0]).set_stroke(MUTED, 1.5)
        overall = DashedLine([-6.05, base_y + bar_h * MERGE_FINAL, 0], [-3.5, base_y + bar_h * MERGE_FINAL, 0],
                             dash_length=0.07).set_stroke(INK, 2)
        overall_lab = VGroup(L("accuracy", 20, MUTED), L(f"{MERGE_FINAL:.2f}", 26, INK, weight="BOLD"))
        overall_lab.arrange(DOWN, buff=0.06).next_to(overall, RIGHT, buff=0.2)
        acc4_cap = L("per-stream accuracy", 20, MUTED).next_to(acc4_vals, UP, buff=0.12).align_to(acc4_base, LEFT)

        # right: eight streams, sixteen channels, all on one (X, SC8 seed 263)
        rx_s, rx_c = 1.15, 3.9
        ys8 = [1.85 - 0.45 * i for i in range(8)]
        head_r = L("eight streams, sixteen channels", 22, MUTED).move_to([(rx_s + rx_c) / 2 + 0.3, 2.35, 0])
        toks8 = VGroup(*[token(f"CTX{s}", "ctx", s, width=0.9, height=0.36, size=20).move_to([rx_s, y, 0])
                         for s, y in enumerate(ys8)])
        cells16 = VGroup(*[Square(0.18).set_stroke(MUTED, 1.2).set_fill(PANEL, 1) for _ in range(16)])
        cells16.arrange(DOWN, buff=0.03).move_to([rx_c, np.mean([ys8[0], ys8[-1]]), 0])
        target = COLLAPSE_MAP[0]
        arrows8 = VGroup(*[Arrow(toks8[s].get_right(), cells16[c].get_left(), buff=0.06, thickness=2)
                           .set_color(STREAM_COLORS[s]) for s, c in enumerate(COLLAPSE_MAP)])
        ch_lab = L(f"ch{target}", 20, BAD).next_to(cells16[target], RIGHT, buff=0.15)
        k_lab = L("16 channels", 20, MUTED).next_to(cells16, RIGHT, buff=0.15).align_to(cells16, UP)
        run_r = VGroup(machine_badge("X", 20), L("SC8, seed 263", 20, MUTED)).arrange(RIGHT, buff=0.15)
        run_r.next_to(cells16, RIGHT, buff=0.3).align_to(cells16, DOWN)

        nl = NumberLine(x_range=[0, 1, 0.25], width=4.6, include_tip=False, tick_size=0.06).set_stroke(MUTED, 2)
        nl.move_to([3.4, -2.55, 0])
        nl_labs = VGroup(L("0", 20, MUTED).next_to(nl.n2p(0), DOWN, buff=0.12),
                         L("1", 20, MUTED).next_to(nl.n2p(1), DOWN, buff=0.12))
        nl_name = L("final accuracy", 20, MUTED).next_to(nl, LEFT, buff=0.2)
        danger = Rectangle(width=nl.n2p(0.15)[0] - nl.n2p(0)[0], height=0.3).set_fill(BAD, 0.25).set_stroke(width=0)
        danger.move_to(nl.n2p(0.075))
        thr15 = Line(nl.n2p(0.15) + 0.2 * DOWN, nl.n2p(0.15) + 0.2 * UP).set_stroke(BAD, 3)
        thr15_lab = L("0.15", 20, BAD).next_to(thr15, DOWN, buff=0.08)
        acc_dot = Dot(nl.n2p(COLLAPSE_ACC), radius=0.09).set_fill(INK, 1)
        acc_lab = L(f"{COLLAPSE_ACC:.3f} ≈ 1/8", 20, INK).next_to(acc_dot, UP, buff=0.22).shift(0.35 * RIGHT)
        coll_badge = verdict_badge("collapsed", 22, BAD).next_to(nl, UP, buff=0.55).set_x(nl.n2p(0.62)[0])
        src5 = source_note("Report §3, §5; data/stream_recipe.json, data/eight_streams.json (X)")

        # summary
        gate = VGroup(RoundedRectangle(width=1.9, height=0.9, corner_radius=0.15).set_fill(GATE_COLOR, 0.15)
                      .set_stroke(GATE_COLOR, 2.5), L("gate", 28, GATE_COLOR, weight="BOLD"))
        gate[1].move_to(gate[0])
        gate.move_to([-5.0, -0.2, 0])
        good_row = VGroup(*[verdict_badge(t, 24, GOOD) for t in ("BOUND", "DISCOVERED", "BOUND ROUTED", "ROUTED*")])
        bad_row = VGroup(verdict_badge("POSITION", 24, BAD), verdict_badge("KEY", 24, BAD),
                         verdict_badge("MERGED", 24, WARN), verdict_badge("collapsed", 24, BAD))
        for rw in (good_row, bad_row):
            rw.arrange(RIGHT, buff=0.22)
        good_lab = L("success", 26, GOOD, weight="BOLD")
        bad_lab = L("failure", 26, BAD, weight="BOLD")
        g_line = VGroup(good_lab, good_row).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        b_line = VGroup(bad_lab, bad_row).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        lines = VGroup(g_line, b_line).arrange(DOWN, buff=0.9, aligned_edge=LEFT).move_to([1.5, -0.2, 0])
        a_good = Arrow(gate.get_right(), good_row[0].get_left(), buff=0.2, thickness=3).set_color(GOOD)
        a_bad = Arrow(gate.get_right(), bad_row[0].get_left(), buff=0.2, thickness=3).set_color(BAD)

        with self.voiceover(
            "With more than two streams, a gate can also put several streams in one channel: a merge. And a run "
            "whose final accuracy stays below 0.15 has collapsed. Keep these in mind; most of the story is about "
            "which of them a gate falls into."
        ) as vo:
            self.play(FadeIn(head_l), LaggedStartMap(FadeIn, toks4, lag_ratio=0.15),
                      LaggedStartMap(FadeIn, chans4, lag_ratio=0.15), FadeIn(src5), run_time=0.8)
            order = [s for s in range(4) if s not in sharers] + sharers
            for s in order[:-1]:
                self.play(GrowArrow(arrows4[s]), run_time=0.3)
            self.play(GrowArrow(arrows4[order[-1]]), run_time=0.45)
            stripes = VGroup(*[Rectangle(width=0.5, height=0.46).set_fill(STREAM_COLORS[s], 0.6).set_stroke(width=0)
                               for s in sharers]).arrange(RIGHT, buff=0).move_to(chans4[shared])
            others = [chans4[c][0].animate.set_fill(STREAM_COLORS[s], 0.25).set_stroke(STREAM_COLORS[s], 2)
                      for s, c in enumerate(MERGE_MAP) if c != shared]
            others += [chans4[c][1].animate.set_color(STREAM_COLORS[s]) for s, c in enumerate(MERGE_MAP) if c != shared]
            chans4[shared][1].set_color(INK)
            self.play(FadeIn(stripes), chans4[shared][0].animate.set_stroke(WARN, 3), *others,
                      Write(share_lab), FadeIn(run_l), run_time=0.6)
            self.add(chans4[shared][1])
            for r in acc4:
                r.save_state()
                r.stretch(1e-3, 1, about_edge=DOWN)
            self.play(ShowCreation(acc4_base), FadeIn(acc4_cap), *[Restore(r) for r in acc4],
                      FadeIn(acc4_vals), run_time=0.8)
            self.play(ShowCreation(overall), FadeIn(overall_lab), run_time=0.5)
            vo.wait_until(at_phrase(vo, 0, "a merge"))
            self.play(FadeTransform(head_l, merged_badge), run_time=0.5)

            vo.wait_until_sentence(1)
            self.play(FadeIn(head_r), LaggedStartMap(FadeIn, toks8, lag_ratio=0.08), FadeIn(cells16),
                      FadeIn(k_lab), run_time=0.8)
            self.play(LaggedStartMap(GrowArrow, arrows8, lag_ratio=0.1),
                      cells16[target].animate.set_fill(BAD, 0.6).set_stroke(BAD, 2), FadeIn(ch_lab), FadeIn(run_r),
                      run_time=1.0)
            vo.wait_until(at_phrase(vo, 1, "below 0.15"))
            self.play(ShowCreation(nl), FadeIn(nl_labs), FadeIn(nl_name), FadeIn(danger), ShowCreation(thr15),
                      FadeIn(thr15_lab), run_time=0.6)
            self.play(FadeIn(acc_dot, scale=0.5), FadeIn(acc_lab), run_time=0.5)
            self.play(FadeIn(coll_badge, shift=0.1 * DOWN), Flash(acc_dot, color=BAD, flash_radius=0.25),
                      run_time=0.6)

            # summary: the two named failures fly into a map of every outcome
            vo.wait_until_sentence(2)
            keep = [self.title_mob, merged_badge, coll_badge]
            self.play(*[FadeOut(m) for m in self.mobjects if m not in keep and m is not self.frame], run_time=0.6)
            self.play(ReplacementTransform(merged_badge, bad_row[2]), ReplacementTransform(coll_badge, bad_row[3]),
                      FadeIn(gate, scale=0.8), run_time=0.8)
            vo.wait_until(at_phrase(vo, 2, "most of the story"))
            self.play(GrowArrow(a_good), GrowArrow(a_bad), FadeIn(good_lab), FadeIn(bad_lab), run_time=0.6)
            self.play(LaggedStartMap(FadeIn, VGroup(*good_row, *bad_row[:2]), shift=0.15 * RIGHT, lag_ratio=0.12),
                      run_time=1.2)
            self.play(LaggedStart(*[b.animate(rate_func=there_and_back).scale(1.12) for b in (*good_row, *bad_row)],
                                  lag_ratio=0.1), run_time=1.5)
