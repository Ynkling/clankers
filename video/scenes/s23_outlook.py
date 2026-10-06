"""Chapter 23 — Limits and next steps (report §16 Limitations, §17 Distance to a language model, §18 Conclusion;
Table 7 (S39), Table 8, Table 9; data/report_tables.json, data/hinge_firings.json)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
with open(os.path.join(DATA, "report_tables.json")) as _fh:
    TABLES = json.load(_fh)
with open(os.path.join(DATA, "hinge_firings.json")) as _fh:
    HINGE = json.load(_fh)

# Table 8 (report §13): the recipe without restarts, counted over both machines
_T8 = TABLES["table8_where_it_stands"]["rows"]
assert "HINGE 39/40" in _T8[0]["recipe"] and "HINGE 35/40" in _T8[3]["recipe"]
# Table 7 (report §11): S39, Muon, S = 8, k = 16, HINGE; stream decodability from the gate state at keys
_T7 = TABLES["table7_gate_memory_S39"]
S39_UPDATES = _T7["updates"]
S39_DECOD = _T7["blocks"][0]["decodability"]
assert S39_DECOD[0] == 0.91 and S39_DECOD[-1] == 0.13 and _T7["blocks"][0]["chance"] == 0.125
# every hinge firing of machine X's WIN_M arm (Muon + WINDOW, S = 2, 40 runs), test_early_recipe results
_WIN = HINGE["early_recipe"]["X"]["arms"]["WIN_M"]["runs"]
FIRINGS = sorted(u for r in _WIN.values() for u in r["fired_at"])
FIRED_RUNS = sum(1 for r in _WIN.values() if r["fired_at"])
assert len(_WIN) == 40 and FIRED_RUNS == 19 and len(FIRINGS) == 35 and max(FIRINGS) <= 2400


# ------------------------------------------------------------------------------------- helpers
def at_phrase(vo, i, phrase):
    """Approximate time (s into the block) at which `phrase` is spoken in sentence i."""
    text, a, b = vo.synth.sentences[i]
    k = text.find(phrase)
    if k < 0:
        raise ValueError(phrase)
    return a + (b - a) * k / max(len(text), 1)


def check_mark(color=GOOD, s=0.2, width=4):
    pts = [np.array(p) * s for p in ([-0.55, 0.05, 0], [-0.15, -0.4, 0], [0.6, 0.5, 0])]
    return VMobject().set_points_as_corners(pts).set_stroke(color, width)


def cross_mark(color=BAD, s=0.14, width=4):
    return VGroup(Line(UL * s, DR * s), Line(DL * s, UR * s)).set_stroke(color, width)


def arc_arrow(a, b, angle, color, width=3, tip=0.2):
    arc = ArcBetweenPoints(a, b, angle=angle).set_stroke(color, width)
    arc.add_tip(width=tip, length=tip, fill_color=color)
    return arc


def frame_rect(center, w, h, edge=PANEL_EDGE):
    r = RoundedRectangle(width=w, height=h, corner_radius=0.14)
    r.set_fill(PANEL, 1).set_stroke(edge, 1.5)
    r.move_to(center)
    return r


def head_in(frame, text, color, size=24):
    t = L(text, size, color, weight="BOLD")
    t.next_to(frame.get_corner(UL), DR, buff=0.22)
    return t


def risk_color(i, n=6):
    a = i / (n - 1)
    return interpolate_color(BAD, WARN, a * 2) if a <= 0.5 else interpolate_color(WARN, MUTED, (a - 0.5) * 2)


def chip(text, color, size=20):
    t = L(text, size, color, weight="BOLD")
    box = RoundedRectangle(width=t.get_width() + 0.3, height=t.get_height() + 0.2, corner_radius=0.08)
    box.set_fill(color, 0.14).set_stroke(color, 1.5)
    t.move_to(box)
    return VGroup(box, t)


# ------------------------------------------------------------------------------------- narration
N1 = ("The limitations are stated plainly. The task is synthetic, with explicit context tokens, and in the grouped "
      "layout each key's stream token is the previous token. The hinge needs the key positions, and a language model "
      "has no marked keys. The model is tiny: three layers, one head, sequences of at most ninety-nine tokens, a "
      "memory half-life of about fourteen tokens. Eight streams have no recipe. The second machine replicates the "
      "procedure, not independent samples. And screens are screens.")
N2 = ("How far is this from a language model? Phase Five removes per-run restarts at two streams, and at four with "
      "sixteen channels, and the recipe works under Muon, an optimizer already used to pretrain large language "
      "models. What stands in the way, roughly in order of risk: many contexts, where the gate's recurrence needs "
      "redesign or constraint; the hinge's nuisance variables, which need a generic version; context cues at a "
      "distance; the gate's token-by-token compute; the memory horizon; and prior art, since Mixture-of-Memories "
      "already routes tokens among linear-attention memories at the billion-parameter scale.")
N3 = ("The plan is staged. On CPUs: eight streams first, then a distant-cue task that one channel provably cannot "
      "solve. Then a GPU port with Muon and a parallel or contractive gate, on the multi-query associative recall "
      "benchmark, against linear-attention baselines including a Mixture-of-Memories router. Then small language "
      "models, and then scaling. The first two stages are cheap, and they are where the idea is most likely to fail.")
N3_SPOKEN = ("The plan is staged. On C P U's: eight streams first, then a distant-cue task that one channel provably "
             "cannot solve. Then a G P U port with Muon and a parallel or contractive gate, on the multi-query "
             "associative recall benchmark, against linear-attention baselines including a Mixture-of-Memories "
             "router. Then small language models, and then scaling. The first two stages are cheap, and they are "
             "where the idea is most likely to fail.")
N4 = ("To sum up. Multi-channel Hebbian plasticity in multilayer BDH solves context-conditional binding by "
      "partitioning memory, and a label-free gate can now find that partition without restarts, at two streams and "
      "at four with sixteen channels. Two changes do it, both confined to the first twenty-four hundred updates: slow "
      "the memory, and push the gate away from positional splits the moment it starts to make them. At eight streams "
      "the obstacle comes earlier still: the gate's recurrent state loses the context before any routing forms.")
N5 = ("Everything you saw is in the repository and its branches: the model, every test with its pre-registered "
      "design, the screens, and the results files, apart from four of machine L's, which are not yet committed. "
      "Thanks for watching.")
N5_SPOKEN = ("Everything you saw is in the repository, Inkling slash clankers, and its branches: the model, every test "
             "with its pre-registered design, the screens, and the results files, apart from four of machine L's, "
             "which are not yet committed. Thanks for watching.")


class Outlook(ClankersScene):
    def construct(self):
        card = self.chapter_card(23, "Limits and next steps")
        self.wait(0.6)
        self.title = section_title("Limitations")
        self.play(FadeOut(card, shift=0.3 * UP), FadeIn(self.title, shift=0.3 * UP))

        self.part_limits()
        self.part_distance()
        self.part_plan()
        self.part_summary()
        self.part_end()

    def swap_title(self, text):
        new = section_title(text)
        self.play(FadeOut(self.title, shift=0.2 * UP), FadeIn(new, shift=0.2 * UP), run_time=0.8)
        self.title = new

    # ================================================================== N23.1 limitation tiles
    def part_limits(self):
        W, H = 4.2, 2.85
        centers = [np.array([x, y, 0]) for y in (1.25, -1.95) for x in (-4.45, 0.0, 4.45)]
        frames = VGroup(*[frame_rect(c, W, H) for c in centers])
        acc = [STREAM_COLORS[0], HINGE_COLOR, MEMORY_COLOR, BAD, WARN, MUTED]
        heads = VGroup(*[head_in(f, t, a) for f, t, a in zip(frames, [
            "A synthetic task", "The recipe needs structure", "A tiny model",
            "Eight streams: no recipe", "One set of seeds", "Screens are screens"], acc)])
        nums = VGroup(*[L(str(i + 1), 20, FAINT, weight="BOLD").next_to(f.get_corner(UR), DL, buff=0.2)
                        for i, f in enumerate(frames)])

        # --- tile 1: the task
        c = centers[0]
        trip = triple(1, 2, 7).move_to(c + 0.3 * DOWN)
        arc = arc_arrow(trip[0].get_top() + 0.05 * UP, trip[1].get_top() + 0.05 * UP, -0.7 * PI, STREAM_COLORS[1])
        arc_lab = L("stream token = previous token", 20, INK).next_to(arc, UP, buff=0.08)
        ctx_box = SurroundingRectangle(trip[0], buff=0.06).set_stroke(STREAM_COLORS[1], 2.5)
        cap1 = L("explicit context tokens", 20, MUTED).next_to(trip, DOWN, buff=0.28)

        # --- tile 2: the recipe needs task structure
        c = centers[1]
        kinds = [("C", "ctx", 0), ("K", "key", None), ("V", "val", 0), ("C", "ctx", 1), ("K", "key", None),
                 ("V", "val", 1)]
        row_a = VGroup(*[token(lb, k, s, width=0.4, height=0.4, size=20) for lb, k, s in kinds])
        row_a.arrange(RIGHT, buff=0.05)
        for i in (1, 4):
            row_a[i][0].set_stroke(HINGE_COLOR, 2.5)
        lab_a = L("task: key positions known", 20, MUTED, t2c={"key positions": HINGE_COLOR})
        words = ["the", "code", "is", "7"]
        row_b = VGroup(*[token(w, "key", width=0.66, height=0.4, size=20) for w in words]).arrange(RIGHT, buff=0.05)
        lab_b = L("language model: no marked keys", 20, MUTED, t2c={"no marked keys": BAD})
        grp2 = VGroup(lab_a, row_a, lab_b, row_b).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        grp2[3].shift(0.12 * DOWN)
        grp2[2].shift(0.12 * DOWN)
        grp2.next_to(heads[1], DOWN, buff=0.26, aligned_edge=LEFT)
        marks_a = VGroup(*[Triangle().set_fill(HINGE_COLOR, 1).set_stroke(width=0).set_height(0.12)
                           .next_to(row_a[i], DOWN, buff=0.04) for i in (1, 4)])
        q_b = L("?", 26, BAD, weight="BOLD").next_to(row_b, RIGHT, buff=0.2)

        # --- tile 3: the model
        c = centers[2]
        m1 = L("3 layers · 1 head", 22, INK)
        m2 = L("sequences ≤ 99 tokens", 22, INK)
        mlines = VGroup(m1, m2).arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        mlines.next_to(heads[2], DOWN, buff=0.22, aligned_edge=LEFT)
        ax = Axes(x_range=[0, 60, 10], y_range=[0, 1, 0.5], width=2.0, height=0.8,
                  axis_config=dict(stroke_color=MUTED, stroke_width=1.5, include_tip=False, tick_size=0.04))
        ax.move_to([c[0] - 1.85, c[1] - 0.55, 0], aligned_edge=LEFT)
        curve = ax.get_graph(lambda t: 0.95 ** t, x_range=[0, 60]).set_stroke(MEMORY_COLOR, 3)
        hl = np.log(0.5) / np.log(0.95)    # 13.5 tokens at decay 0.95
        hl_dot = Dot(ax.c2p(hl, 0.5), radius=0.06, fill_color=WARN)
        hl_v = DashedLine(ax.c2p(hl, 0), ax.c2p(hl, 0.5), dash_length=0.05).set_stroke(WARN, 1.5)
        hl_h = DashedLine(ax.c2p(0, 0.5), ax.c2p(hl, 0.5), dash_length=0.05).set_stroke(WARN, 1.5)
        hl_lab = VGroup(L("memory half-life", 20, WARN), L("≈ 14 tokens", 20, WARN)).arrange(DOWN, buff=0.06,
                                                                                          aligned_edge=LEFT)
        hl_lab.next_to(ax, RIGHT, buff=0.15).align_to(ax, UP).shift(0.05 * DOWN)
        ax_lab = L("tokens back", 20, FAINT).next_to(ax, DOWN, buff=0.06).align_to(ax, LEFT)

        # --- tile 4: streams
        c = centers[3]
        srows = VGroup()
        for i, (lab, n, ok) in enumerate([("S = 2", 2, True), ("S = 4, k = 16", 4, True), ("S = 8", 8, False)]):
            y = c[1] + 0.42 - i * 0.55
            t = L(lab, 20, INK).move_to([c[0] - 1.85, y, 0], aligned_edge=LEFT)
            dots = VGroup(*[Dot(radius=0.065, fill_color=STREAM_COLORS[j]) for j in range(n)]).arrange(RIGHT, buff=0.06)
            dots.move_to([c[0] - 0.15, y, 0], aligned_edge=LEFT)
            mk = (check_mark() if ok else cross_mark()).move_to([c[0] + 1.62, y, 0])
            srows.add(VGroup(t, dots, mk))
        cap4 = L("recipe without restarts", 20, MUTED).move_to([c[0], c[1] - 1.12, 0])

        # --- tile 5: replication
        c = centers[4]
        pattern = [0.85, 0.3, 0.6, 0.15, 0.7, 0.45]

        def seed_strip():
            return VGroup(*[Square(0.22).set_fill(INK, p).set_stroke(MUTED, 1) for p in pattern]).arrange(RIGHT,
                                                                                                         buff=0.05)
        rx = VGroup(machine_badge("X", 20), seed_strip()).arrange(RIGHT, buff=0.22)
        rl = VGroup(machine_badge("L", 20), seed_strip()).arrange(RIGHT, buff=0.22)
        reps = VGroup(rx, rl).arrange(DOWN, buff=0.38, aligned_edge=LEFT)
        reps.move_to([c[0] - 1.85, c[1] + 0.22, 0], aligned_edge=LEFT)
        links = VGroup(*[DashedLine(a.get_bottom(), b.get_top(), dash_length=0.04).set_stroke(WARN, 1.5)
                         for a, b in zip(rx[1], rl[1])])
        same = L("same seeds", 20, WARN).next_to(reps, RIGHT, buff=0.3)
        cap5 = VGroup(L("replicates the procedure,", 20, INK),
                      L("not independent samples", 20, MUTED)).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        cap5.move_to([c[0] - 1.85, c[1] - 0.92, 0], aligned_edge=LEFT)

        # --- tile 6: screens
        c = centers[5]
        items = ["one container", "3 to 40 runs per arm", "lenient rules", "post hoc readings, labelled"]
        blist = VGroup(*[VGroup(Dot(radius=0.045, fill_color=MUTED), L(t, 20, INK)).arrange(RIGHT, buff=0.15)
                         for t in items]).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        blist.next_to(heads[5], DOWN, buff=0.25, aligned_edge=LEFT)
        src = source_note("Report §16")

        def focus(i):
            anims = [frames[i].animate.set_stroke(acc[i], 2.5), FadeIn(heads[i], shift=0.1 * RIGHT),
                     nums[i].animate.set_color(acc[i])]
            if i > 0:
                anims.append(frames[i - 1].animate.set_stroke(PANEL_EDGE, 1.5))
                anims.append(nums[i - 1].animate.set_color(FAINT))
            return anims

        with self.voiceover(N1) as vo:
            self.play(LaggedStartMap(FadeIn, frames, shift=0.15 * UP, lag_ratio=0.12),
                      LaggedStartMap(FadeIn, nums, lag_ratio=0.12), FadeIn(src), run_time=1.6)
            # task
            vo.wait_until_sentence(1)
            self.play(*focus(0), run_time=0.7)
            self.play(FadeIn(trip, shift=0.1 * UP), run_time=0.7)
            vo.wait_until(at_phrase(vo, 1, "explicit context"))
            self.play(ShowCreation(ctx_box), FadeIn(cap1, shift=0.1 * UP), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "in the grouped"))
            self.play(ShowCreation(arc), FadeIn(arc_lab, shift=0.1 * DOWN), run_time=1.0)
            self.play(Indicate(trip[:2], color=STREAM_COLORS[1], scale_factor=1.06), run_time=1.0)
            # recipe needs structure
            vo.wait_until_sentence(2)
            self.play(*focus(1), FadeOut(ctx_box), run_time=0.7)
            self.play(FadeIn(lab_a), LaggedStartMap(FadeIn, row_a, lag_ratio=0.1), run_time=0.8)
            self.play(LaggedStartMap(GrowFromCenter, marks_a, lag_ratio=0.3), run_time=0.6)
            vo.wait_until(at_phrase(vo, 2, "a language model"))
            self.play(FadeIn(lab_b), LaggedStartMap(FadeIn, row_b, lag_ratio=0.1), run_time=0.8)
            self.play(Write(q_b), run_time=0.5)
            # the model
            vo.wait_until_sentence(3)
            self.play(*focus(2), FadeIn(m1, shift=0.1 * UP), run_time=0.8)
            vo.wait_until(at_phrase(vo, 3, "sequences"))
            self.play(FadeIn(m2, shift=0.1 * UP), run_time=0.7)
            vo.wait_until(at_phrase(vo, 3, "a memory half-life") - 0.4)
            self.play(FadeIn(ax), FadeIn(ax_lab), ShowCreation(curve), run_time=1.0)
            self.play(ShowCreation(hl_h), ShowCreation(hl_v), FadeIn(hl_dot, scale=0.5), FadeIn(hl_lab), run_time=0.8)
            # streams
            vo.wait_until_sentence(4)
            self.play(*focus(3), LaggedStart(*[FadeIn(r[:2], shift=0.1 * RIGHT) for r in srows], lag_ratio=0.25),
                      run_time=0.9)
            self.play(LaggedStart(*[ShowCreation(r[2]) for r in srows], lag_ratio=0.3), FadeIn(cap4), run_time=0.8)
            # replication
            vo.wait_until_sentence(5)
            self.play(*focus(4), FadeIn(rx, shift=0.1 * RIGHT), Indicate(srows[2][2], color=BAD), run_time=0.7)
            self.play(TransformFromCopy(rx, rl), run_time=0.8)
            self.play(LaggedStartMap(ShowCreation, links, lag_ratio=0.1), FadeIn(same), run_time=0.7)
            vo.wait_until(at_phrase(vo, 5, "not independent"))
            self.play(FadeIn(cap5, shift=0.1 * UP), run_time=0.7)
            # screens
            vo.wait_until_sentence(6)
            self.play(*focus(5), LaggedStartMap(FadeIn, blist, shift=0.1 * RIGHT, lag_ratio=0.2), run_time=1.2)
        self.play(frames[5].animate.set_stroke(PANEL_EDGE, 1.5), nums[5].animate.set_color(FAINT),
                  *[Indicate(h, color=a, scale_factor=1.05) for h, a in zip(heads, acc)], run_time=1.0)
        self.wait(0.3)

        tiles = VGroup(frames, nums, heads, trip, arc, arc_lab, cap1, grp2, marks_a, q_b, mlines, ax, ax_lab, curve,
                       hl_dot, hl_v, hl_h, hl_lab, srows, cap4, reps, links, same, cap5, blist)
        self.play(FadeOut(tiles, shift=0.3 * DOWN), FadeOut(src), run_time=0.8)
        self.swap_title("Distance to a language model")

    # ================================================================== N23.2 risk ladder
    def part_distance(self):
        TY = 2.0
        x0, x1 = -5.8, 5.8
        track = DashedLine([x0, TY, 0], [x1, TY, 0], dash_length=0.14).set_stroke(FAINT, 2.5)
        here = Dot([x0, TY, 0], radius=0.13, fill_color=MEMORY_COLOR)
        here_lab = L("this work", 22, MEMORY_COLOR).next_to(here, UP, buff=0.14)
        here_lab.add_updater(lambda m: m.next_to(here, UP, buff=0.14))
        goal = Dot([x1, TY, 0], radius=0.15, fill_color=INK)
        goal_lab = L("a language model", 22, INK).next_to(goal, UP, buff=0.14)
        goal_lab.align_to(goal, RIGHT).shift(0.15 * RIGHT)
        qmark = T("?", 52, WARN).move_to([0.6, TY + 0.5, 0])
        trail = Line([x0, TY, 0], [x0 + 0.001, TY, 0]).set_stroke(GOOD, 5)
        trail.add_updater(lambda m: m.put_start_and_end_on(np.array([x0, TY, 0]), here.get_center()))

        def add_row(text, sub=None):
            t = L(text, 24, INK)
            g = VGroup(check_mark(GOOD, 0.24), t).arrange(RIGHT, buff=0.25)
            if sub:
                s = L(sub, 20, MUTED).next_to(t, DOWN, buff=0.1, aligned_edge=LEFT)
                g.add(s)
            return g
        add1 = add_row("no per-run restarts at two streams, and at four with sixteen channels",
                       "an early phase of 2400 updates, inside a single training run")
        add2 = add_row("the recipe works under Muon",
                       "an optimizer already used to pretrain large language models")
        adds = VGroup(add1, add2).arrange(DOWN, buff=0.4, aligned_edge=LEFT).move_to([0.0, 0.25, 0])
        adds_head = L("What Phase V adds", 24, GOOD, weight="BOLD").next_to(adds, UP, buff=0.35, aligned_edge=LEFT)
        src_a = source_note("Report §17")
        compact = L("no restarts · Muon", 20, GOOD).move_to([(x0 - 2.6) / 2, TY - 0.36, 0])

        # the ladder
        rungs_data = [
            ("Many contexts", ["eight streams fail: the gate's recurrence needs redesign or constraint"]),
            ("Nuisance variables", ["the hinge's come from the task's structure; a language model needs a generic version"]),
            ("Distant cues", ["unresolved; the header layout cannot tell routing from single-channel binding"]),
            ("Gate compute", ["the recurrent gate runs token by token; a contractive or parallel gate would remove that cost"]),
            ("Memory horizon", ["a half-life of about 14 tokens, far below a language model's context"]),
            ("Prior art", ["Mixture-of-Memories already routes tokens among linear-attention memories,",
                           "at the billion-parameter scale"]),
        ]
        rungs = VGroup()
        for i, (name, notes) in enumerate(rungs_data):
            y = 0.95 - i * 0.72
            col = risk_color(i)
            circ = Circle(radius=0.24).set_fill(col, 0.2).set_stroke(col, 2).move_to([-5.0, y, 0])
            num = L(str(i + 1), 22, col, weight="BOLD").move_to(circ)
            nm = L(name, 24, INK, weight="BOLD").move_to([-4.55, y, 0], aligned_edge=LEFT)
            nt = VGroup(*[L(n, 20, MUTED) for n in notes]).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
            nt.move_to([-2.1, y, 0], aligned_edge=LEFT)
            if nt.get_right()[0] > 6.6:
                nt.set_width(6.6 + 2.1).move_to([-2.1, y, 0], aligned_edge=LEFT)
            rungs.add(VGroup(circ, num, nm, nt))
        risk_ar = Arrow([-5.85, rungs[-1].get_bottom()[1], 0], [-5.85, rungs[0].get_top()[1] + 0.05, 0], buff=0,
                        thickness=3).set_color(MUTED)
        risk_lab = L("risk", 20, MUTED).rotate(PI / 2).next_to(risk_ar, LEFT, buff=0.12)
        barriers = VGroup()
        for i in range(6):
            col = risk_color(i)
            sq = RoundedRectangle(width=0.42, height=0.42, corner_radius=0.06).set_fill(PANEL, 1).set_stroke(col, 2.5)
            sq.move_to([-1.4 + i * 1.3, TY, 0])
            barriers.add(VGroup(sq, L(str(i + 1), 20, col, weight="BOLD").move_to(sq)))
        mom = chip("Mixture-of-Memories · billion-parameter scale", MUTED, 20)
        mom.move_to([x1, TY - 0.45, 0], aligned_edge=RIGHT).shift(0.25 * RIGHT)

        with self.voiceover(N2) as vo:
            self.play(ShowCreation(track), FadeIn(here, scale=0.5), FadeIn(here_lab), run_time=1.0)
            self.add(trail)
            self.play(FadeIn(goal, scale=0.5), FadeIn(goal_lab), Write(qmark), run_time=0.9)
            # what Phase V adds
            vo.wait_until_sentence(1)
            self.play(FadeIn(adds_head, shift=0.1 * UP), FadeIn(src_a), run_time=0.6)
            self.play(FadeIn(add1, shift=0.15 * RIGHT), here.animate.move_to([-4.2, TY, 0]), run_time=1.2)
            vo.wait_until(at_phrase(vo, 1, "and the recipe works"))
            self.play(FadeIn(add2, shift=0.15 * RIGHT), here.animate.move_to([-2.6, TY, 0]), run_time=1.2)
            self.play(Indicate(add2[1], color=MUON_COLOR, scale_factor=1.04), run_time=1.0)
            # what stands in the way
            vo.wait_until_sentence(2)
            self.play(ReplacementTransform(VGroup(adds_head, add1, add2), compact), FadeOut(qmark), run_time=1.0)
            self.play(GrowArrow(risk_ar), FadeIn(risk_lab), run_time=0.6)
            for i, phrase in enumerate(["many contexts", "the hinge's nuisance", "context cues at a distance",
                                        "the gate's token-by-token", "the memory horizon", "and prior art"]):
                vo.wait_until(at_phrase(vo, 2, phrase) - 0.15)
                self.play(FadeIn(rungs[i][:3], shift=0.15 * RIGHT), FadeIn(rungs[i][3], shift=0.15 * RIGHT),
                          GrowFromCenter(barriers[i]), run_time=0.7)
                if i == 0:
                    vo.wait_until(at_phrase(vo, 2, "where the gate's recurrence"))
                    self.play(Indicate(rungs[0][3], color=BAD, scale_factor=1.03),
                              Indicate(barriers[0], color=BAD, scale_factor=1.2), run_time=1.2)
            vo.wait_until(at_phrase(vo, 2, "Mixture-of-Memories"))
            self.play(FadeIn(mom, shift=0.15 * DOWN), Indicate(rungs[5][3], color=INK, scale_factor=1.03),
                      run_time=1.2)
            vo.wait_until(at_phrase(vo, 2, "at the billion"))
            self.play(Indicate(mom, color=INK, scale_factor=1.06), Flash(goal.get_center(), color=INK), run_time=1.2)
        self.play(Indicate(VGroup(rungs[0], barriers[0]), color=BAD, scale_factor=1.04), run_time=1.0)

        trail.clear_updaters()
        here_lab.clear_updaters()
        self.stash = dict(track=track, here=here, here_lab=here_lab, trail=trail, goal=goal, goal_lab=goal_lab,
                          compact=compact)
        self.play(FadeOut(VGroup(rungs, risk_ar, risk_lab, mom)), FadeOut(barriers), FadeOut(src_a), run_time=0.8)

    # ================================================================== N23.3 staged path
    def part_plan(self):
        st = self.stash
        TY = 2.0
        self.swap_title("A staged path")
        widths = [3.4, 3.9, 2.3, 2.3]
        gap = 0.35
        left = -6.475
        cxs = []
        x = left
        for w in widths:
            cxs.append(x + w / 2)
            x += w + gap
        top, hgt = 1.3, 3.15
        cards = VGroup(*[frame_rect([cx, top - hgt / 2, 0], w, hgt) for cx, w in zip(cxs, widths)])
        spine = Line([-6.5, TY, 0], [6.55, TY, 0]).set_stroke(MUTED, 3)
        stations = VGroup()
        for cx, num in zip(cxs, ["i", "ii", "iii", "iv"]):
            circ = Circle(radius=0.3).set_fill(BG, 1).set_stroke(GATE_COLOR, 2.5).move_to([cx, TY, 0])
            stations.add(VGroup(circ, L(num, 22, GATE_COLOR, weight="BOLD").move_to(circ)))
        stems = VGroup(*[Line([cx, TY - 0.3, 0], [cx, top, 0]).set_stroke(FAINT, 2) for cx in cxs])
        heads = VGroup(*[head_in(f, t, INK) for f, t in zip(cards, ["On CPUs", "A GPU port", "Small LMs", "Scaling"])])

        # (i) CPUs
        dots8 = VGroup(*[Dot(radius=0.06, fill_color=STREAM_COLORS[j]) for j in range(8)]).arrange(RIGHT, buff=0.05)
        c1a = VGroup(dots8, L("eight streams", 22, INK)).arrange(RIGHT, buff=0.18)
        c1b = L("batch 15 first, now running", 20, MUTED)
        far = VGroup(token("CTX0", "ctx", 0, width=0.8, height=0.42, size=20),
                     L("· · · · ·", 20, FAINT),
                     token("K1", width=0.5, height=0.42, size=20),
                     token("?", "query", width=0.5, height=0.42, size=18)).arrange(RIGHT, buff=0.1)
        c1c = L("a distant-cue task", 22, INK)
        c1d = VGroup(L("one channel provably", 20, MUTED), L("cannot solve", 20, MUTED)).arrange(DOWN, buff=0.06,
                                                                                               aligned_edge=LEFT)
        col1a = VGroup(c1a, c1b).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        col1b = VGroup(far, c1c, c1d).arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        col1 = VGroup(col1a, col1b).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        col1.next_to(heads[0], DOWN, buff=0.3, aligned_edge=LEFT)
        cue_arc = arc_arrow(far[2].get_top() + 0.04 * UP, far[0].get_top() + 0.04 * UP, 0.4 * PI, STREAM_COLORS[0],
                            width=2.5, tip=0.16)

        # (ii) GPU port
        c2a = VGroup(L("with Muon and a parallel", 20, INK, t2c={"Muon": MUON_COLOR}),
                     L("or contractive gate", 20, INK, t2c={"contractive": GATE_COLOR, "parallel": GATE_COLOR}))
        c2a.arrange(DOWN, buff=0.06, aligned_edge=LEFT)
        c2b = VGroup(L("MQAR benchmark", 22, WARN, weight="BOLD"),
                     L("multi-query associative recall", 20, MUTED)).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
        c2c = VGroup(L("vs linear-attention baselines,", 20, INK),
                     L("incl. a Mixture-of-Memories router", 20, INK)).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
        col2 = VGroup(c2a, c2b, c2c).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        col2.next_to(heads[1], DOWN, buff=0.3, aligned_edge=LEFT)
        if col2.get_width() > widths[1] - 0.4:
            col2.set_width(widths[1] - 0.4, about_edge=UL)

        # (iii) small LMs
        c3a = VGroup(L("small language", 20, INK), L("models", 20, INK)).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
        lm_rows = VGroup(*[VGroup(*[token(w, "key", width=0.6, height=0.38, size=20) for w in ws]).arrange(RIGHT,
                                                                                                        buff=0.05)
                           for ws in (["the", "door", "is"], ["code", "7", "…"])]).arrange(DOWN, buff=0.08,
                                                                                          aligned_edge=LEFT)
        col3 = VGroup(c3a, lm_rows).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        col3.next_to(heads[2], DOWN, buff=0.3, aligned_edge=LEFT)

        # (iv) scaling
        bars = VGroup(*[Rectangle(width=0.32, height=h).set_fill(GATE_COLOR, 0.25 + 0.15 * i).set_stroke(GATE_COLOR, 1.5)
                        for i, h in enumerate([0.3, 0.6, 1.0, 1.5])]).arrange(RIGHT, buff=0.12, aligned_edge=DOWN)
        up_ar = Arrow(bars[0].get_top() + 0.2 * UP + 0.05 * LEFT, bars[-1].get_top() + 0.25 * UP + 0.15 * LEFT,
                      buff=0, thickness=2.5).set_color(GATE_COLOR)
        icon4 = VGroup(bars, up_ar)
        icon4.move_to([cxs[3], cards[3].get_center()[1] - 0.2, 0])

        cheap_brace = Brace(VGroup(cards[0], cards[1]), DOWN, buff=0.15).set_color(WARN)
        cheap = VGroup(L("cheap", 26, GOOD, weight="BOLD"), L("·", 26, MUTED),
                       L("most likely to fail", 26, BAD, weight="BOLD")).arrange(RIGHT, buff=0.2)
        cheap.next_to(cheap_brace, DOWN, buff=0.12)
        src_p = source_note("Report §17, §18")

        old = VGroup(st["here"], st["here_lab"], st["trail"], st["goal"], st["goal_lab"], st["compact"])
        with self.voiceover(N3, spoken=N3_SPOKEN) as vo:
            self.play(ReplacementTransform(st["track"], spine), FadeOut(old), FadeIn(src_p), run_time=1.0)
            self.play(LaggedStartMap(GrowFromCenter, stations, lag_ratio=0.2), run_time=1.0)
            # (i)
            vo.wait_until_sentence(1)
            self.play(ShowCreation(stems[0]), FadeIn(cards[0], shift=0.15 * DOWN), FadeIn(heads[0]),
                      stations[0][0].animate.set_fill(GATE_COLOR, 1), stations[0][1].animate.set_color(BG), run_time=0.8)
            self.play(LaggedStartMap(FadeIn, dots8, scale=0.5, lag_ratio=0.08), FadeIn(c1a[1]), run_time=0.8)
            self.play(FadeIn(c1b, shift=0.1 * UP), run_time=0.5)
            vo.wait_until(at_phrase(vo, 1, "then a distant-cue"))
            self.play(FadeIn(far, shift=0.1 * RIGHT), FadeIn(c1c), run_time=0.8)
            self.play(ShowCreation(cue_arc), run_time=0.7)
            vo.wait_until(at_phrase(vo, 1, "one channel provably"))
            self.play(FadeIn(c1d, shift=0.1 * UP), run_time=0.6)
            # (ii)
            vo.wait_until_sentence(2)
            self.play(ShowCreation(stems[1]), FadeIn(cards[1], shift=0.15 * DOWN), FadeIn(heads[1]),
                      stations[1][0].animate.set_fill(GATE_COLOR, 1), stations[1][1].animate.set_color(BG), run_time=0.8)
            self.play(FadeIn(c2a, shift=0.1 * UP), run_time=0.7)
            vo.wait_until(at_phrase(vo, 2, "on the multi-query"))
            self.play(FadeIn(c2b, shift=0.1 * UP), run_time=0.7)
            vo.wait_until(at_phrase(vo, 2, "against linear"))
            self.play(FadeIn(c2c[0], shift=0.1 * UP), run_time=0.6)
            vo.wait_until(at_phrase(vo, 2, "including a Mixture"))
            self.play(FadeIn(c2c[1], shift=0.1 * UP), run_time=0.6)
            self.play(Indicate(c2c[1], color=INK, scale_factor=1.05), run_time=1.0)
            # (iii), (iv)
            vo.wait_until_sentence(3)
            self.play(ShowCreation(stems[2]), FadeIn(cards[2], shift=0.15 * DOWN), FadeIn(heads[2]),
                      stations[2][0].animate.set_fill(GATE_COLOR, 1), stations[2][1].animate.set_color(BG), FadeIn(col3, shift=0.1 * UP), run_time=0.9)
            vo.wait_until(at_phrase(vo, 3, "and then scaling"))
            self.play(ShowCreation(stems[3]), FadeIn(cards[3], shift=0.15 * DOWN), FadeIn(heads[3]),
                      stations[3][0].animate.set_fill(GATE_COLOR, 1), stations[3][1].animate.set_color(BG),
                      LaggedStartMap(GrowFromEdge, bars, edge=DOWN, lag_ratio=0.15), run_time=0.9)
            self.play(GrowArrow(up_ar), run_time=0.4)
            # cheap, most likely to fail
            vo.wait_until_sentence(4)
            self.play(GrowFromCenter(cheap_brace), cards[0].animate.set_stroke(WARN, 2.5),
                      cards[1].animate.set_stroke(WARN, 2.5), FadeIn(cheap[0], shift=0.1 * UP), run_time=0.8)
            vo.wait_until(at_phrase(vo, 4, "most likely"))
            self.play(FadeIn(cheap[1:], shift=0.1 * UP), run_time=0.6)
            self.play(Indicate(cheap[2], color=BAD, scale_factor=1.08), run_time=0.9)
        self.wait(0.4)
        self.play(FadeOut(VGroup(spine, stations, stems, cards, heads, col1, cue_arc, col2, col3, icon4, cheap_brace,
                                 cheap, src_p)), run_time=0.8)
        self.swap_title("To sum up")

    # ================================================================== N23.4 recap montage
    def part_summary(self):
        W, H = 4.2, 5.5
        cxs = [-4.45, 0.0, 4.45]
        cy = -0.35
        panels = VGroup(*[frame_rect([cx, cy, 0], W, H) for cx in cxs])
        heads = VGroup(*[head_in(p, t, c) for p, t, c in zip(panels, [
            "A partitioned memory", "The first 2400 updates", "Eight streams"], [MEMORY_COLOR, INK, BAD])])

        # --- A: perfect routing into two channels
        cx = cxs[0]
        ctx0 = token("CTX0", "ctx", 0).move_to([cx - 1.0, 1.4, 0])
        ctx1 = token("CTX1", "ctx", 1).move_to([cx + 1.0, 1.4, 0])
        gate = VGroup(RoundedRectangle(width=1.3, height=0.5, corner_radius=0.1).set_fill(GATE_COLOR, 0.15)
                      .set_stroke(GATE_COLOR, 2), L("gate", 22, GATE_COLOR, weight="BOLD"))
        gate[1].move_to(gate[0])
        gate.move_to([cx, 0.65, 0])
        in0 = Arrow(ctx0.get_bottom(), gate.get_left() + 0.1 * UP, buff=0.08, thickness=2.5).set_color(STREAM_COLORS[0])
        in1 = Arrow(ctx1.get_bottom(), gate.get_right() + 0.1 * UP, buff=0.08, thickness=2.5).set_color(STREAM_COLORS[1])
        chans = channel_stack(2, rows=4, cols=4, cell=0.2)
        chans.arrange(RIGHT, buff=1.0).move_to([cx, -0.45, 0])
        out0 = Arrow(gate.get_bottom() + 0.2 * LEFT, chans[0].cells.get_top(), buff=0.08,
                     thickness=2.5).set_color(STREAM_COLORS[0])
        out1 = Arrow(gate.get_bottom() + 0.2 * RIGHT, chans[1].cells.get_top(), buff=0.08,
                     thickness=2.5).set_color(STREAM_COLORS[1])
        fills = VGroup(*[VGroup(*[cell.copy().set_fill(STREAM_COLORS[s], 0.75).set_stroke(width=0)
                                  for j, cell in enumerate(chans[s].cells) if (j * 7 + 3 * s) % 5 < 3])
                         for s in range(2)])
        res_head = L("found without restarts", 20, MUTED).move_to([cx, -1.55, 0])
        fb1 = frac_bar("S = 2", 39, 40, GOOD, width=1.25, height=0.3, label_width=1.45, size=20)
        fb2 = frac_bar("S = 4, k = 16", 35, 40, GOOD, width=1.25, height=0.3, label_width=1.45, size=20)
        fbs = VGroup(fb1, fb2).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        fbs.move_to([cx, -2.2, 0])
        fb_note = L("HINGE; S = 2 discovered, S = 4 bound", 20, FAINT).move_to([cx, -2.82, 0])
        if fb_note.get_width() > W - 0.3:
            fb_note.set_width(W - 0.3)

        # --- B: the early window
        cx = cxs[1]
        px0, px1 = cx - 1.75, cx + 1.75
        umax = 3600.0

        def ux(u):
            return px0 + (px1 - px0) * u / umax
        ay = -0.75
        axis = Line([px0, ay, 0], [px1, ay, 0]).set_stroke(MUTED, 2)
        ticks = VGroup()
        for u in (0, 1200, 2400, 3600):
            ticks.add(Line([ux(u), ay - 0.06, 0], [ux(u), ay + 0.06, 0]).set_stroke(MUTED, 2))
            ticks.add(L(f"{u}", 20, MUTED).next_to([ux(u), ay, 0], DOWN, buff=0.12))
        ax_lab = L("update", 20, FAINT).next_to(ticks, DOWN, buff=0.1)
        window = Rectangle(width=ux(2400) - ux(0), height=2.5).set_fill(SLOW_COLOR, 0.08).set_stroke(width=0)
        window.move_to([(ux(0) + ux(2400)) / 2, ay + 1.25, 0])
        win_edge = DashedLine([ux(2400), ay, 0], [ux(2400), ay + 2.5, 0], dash_length=0.08).set_stroke(SLOW_COLOR, 2)
        slow_lab = L("slow the memory", 22, SLOW_COLOR, weight="BOLD").move_to([px0, 1.45, 0], aligned_edge=LEFT)
        lo, hi = 0.75, 1.1
        lr_step = VMobject().set_points_as_corners([[ux(0), lo, 0], [ux(2400), lo, 0], [ux(2400), hi, 0],
                                                    [ux(3600), hi, 0]]).set_stroke(SLOW_COLOR, 4)
        lr_lab = L("memory at 1/10 the rate", 20, SLOW_COLOR).next_to([ux(1200), lo, 0], DOWN, buff=0.1)
        full_lab = L("full rate", 20, MUTED).next_to([ux(3000), hi, 0], UP, buff=0.1)
        hinge_lab = VGroup(L("push the gate off", 22, HINGE_COLOR, weight="BOLD"),
                           L("positional splits", 22, HINGE_COLOR, weight="BOLD")).arrange(DOWN, buff=0.06,
                                                                                          aligned_edge=LEFT)
        hinge_lab.move_to([ux(450), -0.02, 0], aligned_edge=LEFT)
        spikes = VGroup(*[Line([ux(u), ay, 0], [ux(u), ay + 0.55, 0]).set_stroke(HINGE_COLOR, 1.5, opacity=0.8)
                          for u in FIRINGS])
        fire_note = VGroup(L(f"hinge firings: {len(FIRINGS)} in {len(_WIN)} runs", 20, MUTED),
                           L("(Muon + WINDOW, machine X)", 20, FAINT)).arrange(DOWN, buff=0.06)
        fire_note.move_to([cx, -2.05, 0])
        both = L("both confined to updates 1–2400", 20, INK).move_to([cx, -2.75, 0])

        # --- C: the gate forgets the context
        cx = cxs[2]
        inp = token("CTX5", "ctx", 5).move_to([cx - 1.4, 0.8, 0])
        hbox = RoundedRectangle(width=1.0, height=0.62, corner_radius=0.1).set_stroke(GATE_COLOR, 2.5)
        hbox.move_to([cx - 0.05, 0.8, 0])
        stripes = VGroup(*[Rectangle(width=0.1, height=0.46).set_fill(STREAM_COLORS[j], 0.85).set_stroke(width=0)
                           for j in range(8)]).arrange(RIGHT, buff=0.0).move_to(hbox)
        h_lab = L("gate state", 20, MUTED).next_to(hbox, DOWN, buff=0.1)
        in_ar = Arrow(inp.get_right(), hbox.get_left(), buff=0.08, thickness=2.5).set_color(MEMORY_COLOR)
        loop = arc_arrow(hbox.get_top() + 0.25 * RIGHT, hbox.get_top() + 0.25 * LEFT, 1.3 * PI, GATE_COLOR,
                         width=3, tip=0.18)
        rho = chip("ρ > 1", BAD, 22).move_to([cx + 1.35, 1.0, 0])
        rho_lab = L("recurrent gain", 20, BAD).next_to(rho, DOWN, buff=0.1)
        dec_lab = L("stream readable from the state", 20, MUTED).move_to([cx, -0.25, 0])
        dax = Axes(x_range=[0, 2400, 1200], y_range=[0, 1, 0.5], width=3.0, height=1.25,
                   axis_config=dict(stroke_color=MUTED, stroke_width=1.5, include_tip=False, tick_size=0.04))
        dax.move_to([cx + 0.15, -1.3, 0])
        dticks = VGroup(*[L(f"{u}", 20, MUTED).next_to(dax.c2p(u, 0), DOWN, buff=0.08) for u in (0, 1200, 2400)])
        chance = DashedLine(dax.c2p(0, 0.125), dax.c2p(2400, 0.125), dash_length=0.06).set_stroke(FAINT, 1.5)
        ch_lab = L("chance", 20, FAINT).next_to(dax.c2p(2400, 0.125), UP, buff=0.06).align_to(dax, RIGHT)
        ylab = VGroup(L("1", 20, MUTED).next_to(dax.c2p(0, 1), LEFT, buff=0.1),
                      L("0", 20, MUTED).next_to(dax.c2p(0, 0), LEFT, buff=0.1))
        dec_curve = VMobject().set_points_as_corners([dax.c2p(u, d) for u, d in zip(S39_UPDATES, S39_DECOD)])
        dec_curve.set_stroke(BAD, 3.5)
        d0 = L(f"{S39_DECOD[0]:.2f}", 20, INK).next_to(dax.c2p(0, S39_DECOD[0]), RIGHT, buff=0.12)
        d1 = L(f"{S39_DECOD[-1]:.2f}", 20, BAD).next_to(dax.c2p(2400, S39_DECOD[-1]), UP, buff=0.32)
        lost = L("lost before routing forms", 20, INK).move_to([cx, -2.75, 0])

        src_s = source_note("Report §13 Table 8, §11 Table 7 (S39, Muon) and S42; data/hinge_firings.json")

        with self.voiceover(N4) as vo:
            self.play(LaggedStartMap(FadeIn, panels, shift=0.15 * UP, lag_ratio=0.2), run_time=1.0)
            # A
            vo.wait_until_sentence(1)
            self.play(FadeIn(heads[0]), panels[0].animate.set_stroke(MEMORY_COLOR, 2.5),
                      FadeIn(ctx0, shift=0.1 * DOWN), FadeIn(ctx1, shift=0.1 * DOWN), FadeIn(gate), FadeIn(chans),
                      run_time=1.0)
            vo.wait_until(at_phrase(vo, 1, "solves context"))
            self.play(GrowArrow(in0), GrowArrow(in1), Indicate(gate, color=GATE_COLOR, scale_factor=1.08), run_time=0.9)
            vo.wait_until(at_phrase(vo, 1, "by partitioning"))
            self.play(GrowArrow(out0), FadeIn(fills[0], lag_ratio=0.1), run_time=0.8)
            self.play(GrowArrow(out1), FadeIn(fills[1], lag_ratio=0.1), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "and a label-free"))
            self.play(Indicate(gate, color=GATE_COLOR, scale_factor=1.1), FadeIn(res_head), FadeIn(src_s), run_time=0.9)
            vo.wait_until(at_phrase(vo, 1, "at two streams"))
            self.play(FadeIn(fb1.name_mob), FadeIn(fb1.track), run_time=0.4)
            g1 = grow_bar(fb1, 0.8)
            self.add(fb1.fill)
            self.play(*g1, run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "and at four"))
            self.play(FadeIn(fb2.name_mob), FadeIn(fb2.track), run_time=0.4)
            g2 = grow_bar(fb2, 0.8)
            self.add(fb2.fill)
            self.play(*g2, FadeIn(fb_note), run_time=0.8)
            # B
            vo.wait_until_sentence(2)
            self.play(FadeIn(heads[1]), panels[0].animate.set_stroke(PANEL_EDGE, 1.5),
                      panels[1].animate.set_stroke(WARN, 2.5), ShowCreation(axis), FadeIn(ticks), FadeIn(ax_lab),
                      run_time=0.8)
            self.play(FadeIn(window), ShowCreation(win_edge), FadeIn(both, shift=0.1 * UP), run_time=0.8)
            vo.wait_until(at_phrase(vo, 2, "slow the memory"))
            self.play(FadeIn(slow_lab, shift=0.1 * RIGHT), ShowCreation(lr_step), FadeIn(lr_lab), FadeIn(full_lab),
                      run_time=1.0)
            vo.wait_until(at_phrase(vo, 2, "and push the gate"))
            self.play(FadeIn(hinge_lab, shift=0.1 * RIGHT), LaggedStartMap(ShowCreation, spikes, lag_ratio=0.03),
                      run_time=1.3)
            self.play(FadeIn(fire_note), run_time=0.5)
            self.play(Indicate(both, color=WARN, scale_factor=1.05), run_time=0.8)
            # C
            vo.wait_until_sentence(3)
            self.play(FadeIn(heads[2]), panels[1].animate.set_stroke(PANEL_EDGE, 1.5),
                      panels[2].animate.set_stroke(BAD, 2.5), FadeIn(inp), FadeIn(hbox), FadeIn(stripes),
                      FadeIn(h_lab), GrowArrow(in_ar), run_time=0.8)
            self.play(ShowCreation(loop), FadeIn(dax), FadeIn(dticks), FadeIn(ylab), FadeIn(dec_lab),
                      ShowCreation(chance), FadeIn(ch_lab), FadeIn(d0), run_time=0.8)
            self.play(ShowCreation(dec_curve), stripes.animate.set_fill(MUTED, 0.35),
                      FadeIn(rho, scale=0.8), FadeIn(rho_lab), run_time=2.6)
            self.play(FadeIn(d1), Indicate(rho, color=BAD), run_time=0.8)
            vo.wait_until(at_phrase(vo, 3, "before any routing") - 0.3)
            self.play(FadeIn(lost, shift=0.1 * UP), run_time=0.7)
            self.play(Indicate(lost, color=BAD, scale_factor=1.06), run_time=0.9)
        self.wait(0.6)
        self.play(FadeOut(VGroup(*[m for m in self.mobjects if m is not self.frame and m is not self.title])),
                  run_time=0.8)

    # ================================================================== N23.5 end card
    def part_end(self):
        url = L("github.com/Ynkling/clankers", 44, GATE_COLOR, weight="BOLD").move_to([0, 2.45, 0])
        rule = Line(LEFT, RIGHT).set_width(url.get_width() + 0.4).set_stroke(FAINT, 2).next_to(url, DOWN, buff=0.2)
        rows_data = [
            ("the model", "MultiBDH, in test_multilayer_binding.py"),
            ("every test", "a test_*.py file per test; its docstring holds the pre-registered design"),
            ("the screens", "branch claude/outside-ideas"),
            ("the results", "results/X/ and results/L/"),
        ]
        rows = VGroup()
        for i, (a, b) in enumerate(rows_data):
            y = 1.35 - i * 0.62
            ta = L(a, 24, INK, weight="BOLD").move_to([-5.6, y, 0], aligned_edge=LEFT)
            tb = L(b, 22, MUTED).move_to([-3.25, y, 0], aligned_edge=LEFT)
            rows.add(VGroup(ta, tb))
        # machine L's Phase V results files (report Table 9): only the early-recipe file is committed
        lfiles = VGroup()
        for name, done in [("stream recipe", False), ("stream curriculum", False), ("slow start", False),
                           ("recipe scope", False), ("early recipe", True)]:
            col = GOOD if done else WARN
            t = L(name, 20, col)
            box = RoundedRectangle(width=t.get_width() + 0.3, height=t.get_height() + 0.24, corner_radius=0.08)
            if done:
                box.set_fill(col, 0.14).set_stroke(col, 1.5)
            else:
                box = DashedVMobject(box, num_dashes=28).set_stroke(col, 1.5)
            t.move_to(box)
            lfiles.add(VGroup(box, t))
        lfiles.arrange(RIGHT, buff=0.22)
        lfiles.move_to([0.6, -1.55, 0])
        l_badge = machine_badge("L", 22).next_to(lfiles, LEFT, buff=0.35)
        l_lab_a = L("not yet committed", 20, WARN).next_to(VGroup(*lfiles[:4]), DOWN, buff=0.18)
        l_lab_b = L("committed", 20, GOOD).next_to(lfiles[4], DOWN, buff=0.18)
        l_head = L("machine L's Phase V results files", 20, MUTED).next_to(lfiles, UP, buff=0.22)
        l_head.align_to(l_badge, LEFT)
        lgroup = VGroup(l_head, l_badge, lfiles, l_lab_a, l_lab_b)
        src_e = source_note("Report §18, Table 9; repository files")

        rtitle = VGroup(T("Multi-Channel Hebbian Plasticity in Multilayer BDH", 34),
                        T("Solves Context-Conditional Binding by Partitioning Memory", 34)).arrange(DOWN, buff=0.15)
        meta = L("A technical report · Revision 7 (preliminary)", 22, MUTED)
        credit = L("Made with ManimGL (3b1b/manim) · narration: Kokoro TTS", 20, FAINT)
        endcard = VGroup(rtitle, meta, credit).arrange(DOWN, buff=0.4).move_to([0, -0.2, 0])
        thanks = T("Thanks for watching.", 36, INK).move_to([0, -2.75, 0])

        with self.voiceover(N5, spoken=N5_SPOKEN) as vo:
            self.play(FadeOut(self.title), Write(url), ShowCreation(rule), run_time=1.4)
            for i, phrase in enumerate(["the model", "every test", "the screens", "and the results"]):
                vo.wait_until(at_phrase(vo, 0, phrase) - 0.1)
                self.play(FadeIn(rows[i], shift=0.15 * RIGHT), *([FadeIn(src_e)] if i == 0 else []), run_time=0.6)
            vo.wait_until(at_phrase(vo, 0, "apart from") - 0.2)
            self.play(FadeIn(l_head), FadeIn(l_badge, scale=0.6), LaggedStartMap(FadeIn, lfiles, shift=0.1 * UP,
                                                                                 lag_ratio=0.15), run_time=1.0)
            self.play(FadeIn(l_lab_a), FadeIn(l_lab_b), Indicate(VGroup(*lfiles[:4]), color=WARN, scale_factor=1.04),
                      run_time=1.0)
            vo.wait_until_sentence(1)
            self.play(FadeOut(rows, shift=0.2 * UP), FadeOut(lgroup, shift=0.2 * UP), FadeOut(src_e),
                      FadeIn(endcard, shift=0.2 * UP),
                      FadeIn(thanks), run_time=1.0)
        self.wait(3.0)
        self.clear_all()
