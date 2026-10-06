"""Chapter 18 — breaking merges at four streams and four channels (report §10, with §4 for the merge counts).

Numbers: report §10 (S21, S24, S27, S33a, S36, S37). Every count was also checked against the screens' run
records on the growth repo's `claude/outside-ideas` branch (explore_out/{merge_kick, merge_split, zloss_merge,
split_plateau, split_target}_results.json; not in this checkout, so the values used are copied below).
Example merged gate: video/data/gate_dynamics.json, X run A4k4|258 at update 9600 (k = 4, map [0, 2, 0, 1]).
That run is one of the eight continued runs (stream_recipe seed 258 in S21/S24/S27's run list).
Logit values and W_g entries are schematic and labelled as such.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load(name):
    with open(os.path.join(DATA, name)) as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------------- real data
_RUN = _load("gate_dynamics.json")["multichannel_runs"]["X:A4k4|258"]
_I9600 = _RUN["steps"].index(9600)
MAP_9600 = _RUN["ch_map"][_I9600]                 # stream -> data channel at update 9600
GATE_9600 = _RUN["stream_gate"][_I9600]           # rows: streams, columns: data channels
assert MAP_9600 == [0, 2, 0, 1], MAP_9600         # streams 0 and 2 share channel 0; channel 3 idle
assert _RUN["outcome"].startswith("MERGED")
# screen order of channels: shared, the one holding stream 1, the one holding stream 3, idle
DISPLAY = [0, 2, 1, 3]
SLOT = [DISPLAY.index(c) for c in MAP_9600]       # stream -> slot on screen: [0, 1, 0, 2]
HOLDS = [[s for s in range(4) if SLOT[s] == j] for j in range(4)]   # [[0, 2], [1], [3], []]
GATE_SHOWN = [[GATE_9600[s][DISPLAY[j]] for j in range(4)] for s in range(4)]
assert round(GATE_SHOWN[0][0], 2) == 0.96 and round(GATE_SHOWN[2][0], 2) == 0.96

# S24's record of this same run (merge_split_results.json, SR_A4k4|258): c* = data channel 0 (shared),
# c0 = 3 (idle); one channel per stream by update 13200. Mean read gates there, data channels 0..3:
S24_SPLIT_STEP = 13200
S24_SPLIT_GATE = {0: [0.9865, 0.0, 0.0, 0.0135], 2: [0.0338, 0.0219, 0.0, 0.9443]}

# S36 / S37 seed 241 (Muon, k = 4, from scratch; split_plateau_results.json SPLIT4k4_M|241 and
# split_target_results.json SPLITK_M|241). The two runs are identical up to update 4800, where the plateau
# trigger fired in both (probe accuracy 0.78 at 2400, 0.75 at 4800). S36 aimed by all-position mass, S37 by
# key-position mass. Held-out accuracy (the runs' curves):
R241_STEPS, R241_ACC = [1200, 2400, 3600, 4800], [0.352, 0.740, 0.747, 0.764]
R241_S36 = ([4800, 6000, 7200, 8400, 9600], [0.764, 0.499, 0.495, 0.749, 0.739])   # ended MERGED
R241_S37 = ([4800, 6000, 7200, 8400], [0.764, 1.0, 0.993, 0.999])                   # BOUND ROUTED at 6000
R241_MAP = [2, 3, 0, 0]                              # stream -> data channel at 4800 (channel 1 holds none)
R241_ALL = [0.29908, 0.34253, 0.1788, 0.17959]       # read-gate mass per channel, all probe positions
R241_KEY = [0.34884, 0.15116, 0.24982, 0.25018]      # the same, at key positions
R241_DISPLAY = [0, 2, 3, 1]                          # on screen: shared, stream 0's, stream 1's, empty
R241_HOLDS = [[s for s in range(4) if R241_MAP[s] == c] for c in R241_DISPLAY]   # [[2, 3], [0], [1], []]
assert int(np.argmax(R241_ALL)) == 1 and int(np.argmin(R241_ALL)) == 2      # all-position aim: wrong both ways
assert int(np.argmax(R241_KEY)) == 0 and int(np.argmin(R241_KEY)) == 1      # key-position aim: shared -> empty

# report §10 / §4 (screens; no data in this checkout)
MERGED_K4, MERGED_K16, N_PLAIN = 32, 9, 80
N_CONT = 8
S21, S27, S24, S24_CTRL = 0, 2, 8, 1
S36_BOUND, S36_NONE, S36_N, S36_SPLITS = 5, 4, 10, 31
S37_ON, S37_OLD_ON, S37_SPLITS = 10, 5, 11
S37_MUON, S37_MUON_NONE, S37_ADAM, S37_ADAM_NONE = 6, 3, 6, 5
S37_STUCK = 4      # "the four Muon runs that stayed merged under S37, both splits were on target"

XS = [-5.4, -4.2, -3.0, -1.8]          # x of the four streams / channels / bars on the left
MIX02 = interpolate_color(STREAM_COLORS[0], STREAM_COLORS[2], 0.5)
SLOT_COLORS = [MIX02, STREAM_COLORS[1], STREAM_COLORS[3], FAINT]


# ------------------------------------------------------------------------------------- helpers
def at_phrase(vo, i, phrase):
    """Approximate time (s into the block) at which `phrase` is spoken in sentence i."""
    text, a, b = vo.synth.sentences[i]
    k = text.find(phrase)
    if k < 0:
        raise ValueError(phrase)
    return a + (b - a) * k / max(len(text), 1)


def dots(n_good, n, good=GOOD, bad=BAD, r=0.12, buff=0.1):
    g = VGroup()
    for i in range(n):
        c = Circle(radius=r)
        if i < n_good:
            c.set_fill(good, 0.95).set_stroke(good, 2)
        else:
            c.set_fill(bad, 0.0).set_stroke(bad, 2)
        g.add(c)
    return g.arrange(RIGHT, buff=buff)


def strip(label, n_good, n, color, word="split", good=GOOD, bad=BAD, size=22):
    """Two-line result strip: label above, n dots and the count below."""
    lab = L(label, size=size, color=INK)
    d = dots(n_good, n, good, bad)
    cnt = L(f"{n_good}/{n} {word}", size=24, color=color, weight="BOLD")
    d.next_to(lab, DOWN, buff=0.16, aligned_edge=LEFT).shift(0.08 * RIGHT)
    cnt.next_to(d, RIGHT, buff=0.3)
    g = VGroup(lab, d, cnt)
    g.lab, g.dots, g.cnt = lab, d, cnt
    return g


def stat_line(label, k, n, color, label_w=2.0, r=0.09, buff=0.07, size=20, bad=MUTED, word=""):
    """One-line result: label, n dots, k/n."""
    lab = L(label, size=size, color=INK)
    d = dots(k, n, color, bad, r=r, buff=buff)
    cnt = L(f"{k}/{n}" + (f" {word}" if word else ""), size=22, color=color, weight="BOLD")
    d.next_to(lab, RIGHT, buff=0.2)
    d.shift((lab.get_left()[0] + label_w - d.get_left()[0]) * RIGHT)
    cnt.next_to(d, RIGHT, buff=0.22)
    g = VGroup(lab, d, cnt)
    g.lab, g.dots, g.cnt = lab, d, cnt
    return g


def channel_box(x, y=-0.15, w=1.0, h=1.2):
    return RoundedRectangle(width=w, height=h, corner_radius=0.12).set_fill(PANEL, 1).set_stroke(MEMORY_COLOR, 2).move_to([x, y, 0])


def content(box, streams, opacity=0.6):
    """What a channel holds: one slab per stream, in the stream's color."""
    if not streams:
        return VGroup()
    w, h = box.get_width() - 0.26, box.get_height() - 0.26
    g = VGroup(*[Rectangle(width=w / len(streams), height=h).set_fill(STREAM_COLORS[s], opacity).set_stroke(width=0)
                 for s in streams])
    return g.arrange(RIGHT, buff=0).move_to(box)


def route_arrow(start, end, color):
    return Arrow(start, end, buff=0.05, thickness=3.5).set_color(color)


ZERO_Y, LSCALE = 0.35, 0.36


def logit_bars(z, width=0.72):
    g = VGroup()
    for zi, x in zip(z, XS):
        h = max(abs(zi) * LSCALE, 0.015)
        r = Rectangle(width=width, height=h).set_fill(GATE_COLOR, 0.7).set_stroke(GATE_COLOR, 1.5)
        r.move_to([x, ZERO_Y + (h / 2 if zi >= 0 else -h / 2), 0])
        g.add(r)
    return g


def softmax(z):
    e = np.exp(np.asarray(z, float) - np.max(z))
    return e / e.sum()


def prob_bars(p, color, base_x, base_y, max_h=1.25, w=0.4, step=0.6):
    xs = [base_x + (j - 1.5) * step for j in range(4)]
    g = VGroup()
    for pj, x in zip(p, xs):
        h = max(pj * max_h, 0.012)
        g.add(Rectangle(width=w, height=h).set_fill(color, 0.85).set_stroke(color, 1).move_to([x, base_y + h / 2, 0]))
    return g


def slot_tag(streams, size=0.17):
    if not streams:
        return L("none", size=20, color=MUTED)
    return VGroup(*[Square(size).set_fill(STREAM_COLORS[s], 0.9).set_stroke(width=0) for s in streams]).arrange(RIGHT, buff=0.05)


def wg_color(v):
    return interpolate_color("#2A2238", GATE_COLOR, float((np.tanh(0.9 * v) + 1) / 2))


def wg_row(vals, y, x0=-5.25, w=0.118, h=0.26, buff=0.027):
    """One row of W_g (32 weights: W_g is k x 32, H_GATE = 32)."""
    row = VGroup(*[Rectangle(width=w, height=h).set_fill(wg_color(v), 1).set_stroke(BG, 0.5) for v in vals])
    row.arrange(RIGHT, buff=buff)
    row.move_to([x0, y, 0], aligned_edge=LEFT)
    return row


class Merges(ClankersScene):
    def construct(self):
        card = self.chapter_card(18, "Breaking merges")
        self.wait(0.6)
        title = section_title("Breaking a merge")
        self.play(FadeOut(card, shift=0.3 * UP), FadeIn(title, shift=0.3 * UP))
        self.title_mob = title

        self.part_merge()
        self.part_failed()
        self.part_surgery()
        self.part_trigger()
        self.wait(0.6)
        self.clear_all()

    # ============================================================== N18.1  a merge, and the screen
    def part_merge(self):
        toks = VGroup(*[token(f"CTX{s}", "ctx", s, width=1.0, height=0.55, size=20) for s in range(4)])
        for t, x in zip(toks, XS):
            t.move_to([x, 1.85, 0])
        streams_lab = L("four streams", size=22, color=MUTED).next_to(toks, UP, buff=0.22)
        boxes = VGroup(*[channel_box(x) for x in XS])
        chans_lab = L("four channels", size=22, color=MUTED).move_to([np.mean(XS), -1.55, 0])
        fills = VGroup(*[content(b, HOLDS[j]) for j, b in enumerate(boxes)])
        ends = {0: boxes[0].get_top() + 0.2 * LEFT, 2: boxes[0].get_top() + 0.2 * RIGHT,
                1: boxes[1].get_top(), 3: boxes[2].get_top()}
        arrows = VGroup(*[route_arrow(toks[s].get_bottom(), ends[s] + 0.02 * UP, STREAM_COLORS[s]) for s in range(4)])
        shared_lab = L("shared", size=22, color=BAD, weight="BOLD").next_to(boxes[0], DOWN, buff=0.15)
        idle_lab = L("idle", size=22, color=MUTED, weight="BOLD").next_to(boxes[3], DOWN, buff=0.15)
        src = source_note("Report §10, §4 · pictured: X, A4k4 seed 258 at update 9600, one of the eight (gate_dynamics.json)")

        # right: merge counts
        head = L("plain runs that merged, over two tests", size=22, color=MUTED)
        fb4 = frac_bar("4 channels", MERGED_K4, N_PLAIN, BAD, width=2.6, label_width=1.75, size=22)
        fb16 = frac_bar("16 channels", MERGED_K16, N_PLAIN, BAD, width=2.6, label_width=1.75, size=22)
        stats = VGroup(head, fb4, fb16).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        stats.move_to([3.6, 1.0, 0])
        fix = T("spare channels: the main fix, not a cure", size=26, color=WARN).next_to(stats, DOWN, buff=0.5)

        q = VGroup(T("Can a merge, once formed,", size=36), T("be broken?", size=36)).arrange(DOWN, buff=0.2)
        q.move_to([3.6, 0.6, 0])
        cut = DashedLine(boxes[0].get_top() + 0.15 * UP, boxes[0].get_bottom() + 0.15 * DOWN, dash_length=0.08)
        cut.set_stroke(WARN, 3)

        # right: eight recorded runs, continued
        ax_x0, ax_x1, ax_y = 1.0, 6.2, -0.45

        def tx(u):
            return ax_x0 + (ax_x1 - ax_x0) * u / 19200

        axis = Line([ax_x0, ax_y, 0], [ax_x1, ax_y, 0]).set_stroke(MUTED, 2)
        ticks = VGroup()
        for u in (0, 9600, 19200):
            ticks.add(Line([tx(u), ax_y - 0.08, 0], [tx(u), ax_y + 0.08, 0]).set_stroke(MUTED, 2))
            ticks.add(L(f"{u}", size=20, color=MUTED).next_to([tx(u), ax_y, 0], DOWN, buff=0.15))
        ax_lab = L("update", size=20, color=MUTED).next_to(axis, DOWN, buff=0.5)
        run_ys = [1.55 - 0.24 * i for i in range(N_CONT)]
        rec = VGroup(*[Line([tx(0), y, 0], [tx(9600), y, 0]).set_stroke(MUTED, 3) for y in run_ys])
        mdots = VGroup(*[Dot([tx(9600), y, 0], radius=0.075).set_fill(BAD, 1) for y in run_ys])
        cont = VGroup(*[DashedLine([tx(9600), y, 0], [tx(19200), y, 0], dash_length=0.1).set_stroke(WARN, 3) for y in run_ys])
        end_q = VGroup(*[Circle(radius=0.1).set_fill(BAD, 0).set_stroke(MUTED, 2).move_to([tx(19200) + 0.18, y, 0])
                         for y in run_ys])
        runs_head = L(f"{N_CONT} recorded runs on X, four channels", size=22, color=INK).move_to([3.6, 2.45, 0])
        rec_lab = L("merged at 9600", size=20, color=BAD).move_to([tx(9600), 1.95, 0])
        cont_lab = L("continued", size=20, color=WARN).move_to([tx(15900), 1.95, 0])
        timeline = VGroup(axis, ticks, ax_lab, rec, mdots, cont, rec_lab, cont_lab)

        with self.voiceover(
            "At four streams with only four channels, the failures are merges, and spare channels are the main fix, "
            "though not a cure: over two tests, plain runs merged thirty-two times in eighty with four channels, and "
            "nine in eighty with sixteen. Can a merge, once formed, be broken? Screens took eight recorded "
            "four-channel runs that were merged at update ninety-six hundred and continued each one."
        ) as vo:
            self.play(LaggedStartMap(FadeIn, toks, shift=0.15 * DOWN, lag_ratio=0.15), FadeIn(streams_lab),
                      LaggedStartMap(FadeIn, boxes, shift=0.15 * UP, lag_ratio=0.15), FadeIn(chans_lab), run_time=1.4)
            vo.wait_until(at_phrase(vo, 0, "the failures are merges") - 0.3)
            self.play(LaggedStart(*[GrowArrow(arrows[s]) for s in (0, 1, 3)], lag_ratio=0.2),
                      FadeIn(fills[1]), FadeIn(fills[2]), FadeIn(fills[0][0]), run_time=1.0)
            self.play(GrowArrow(arrows[2]), FadeIn(fills[0][1]), FadeIn(src), run_time=0.9)
            self.play(boxes[0].animate.set_stroke(BAD, 3), FadeIn(shared_lab, shift=0.1 * UP),
                      boxes[3].animate.set_stroke(FAINT, 2), FadeIn(idle_lab, shift=0.1 * UP), run_time=0.8)
            vo.wait_until(at_phrase(vo, 0, "spare channels") - 0.2)
            self.play(FadeIn(head, shift=0.1 * DOWN), FadeIn(fb4.name_mob), FadeIn(fb4.track),
                      FadeIn(fb16.name_mob), FadeIn(fb16.track), run_time=0.8)
            self.play(Write(fix), run_time=1.2)
            vo.wait_until(at_phrase(vo, 0, "plain runs merged"))
            self.play(*grow_bar(fb4, 1.2))
            vo.wait_until(at_phrase(vo, 0, "nine in eighty"))
            self.play(*grow_bar(fb16, 1.0))
            self.play(Indicate(fix, color=WARN, scale_factor=1.04), run_time=0.8)

            vo.wait_until_sentence(1)
            self.play(FadeOut(VGroup(stats, fix), shift=0.2 * UP), FadeIn(q, shift=0.2 * UP),
                      ShowCreation(cut), run_time=1.0)
            self.play(WiggleOutThenIn(VGroup(boxes[0], fills[0], cut), scale_value=1.06), run_time=1.0)

            vo.wait_until_sentence(2)
            self.play(FadeOut(q, shift=0.2 * UP), FadeOut(cut), FadeIn(runs_head), ShowCreation(axis),
                      FadeIn(ticks), FadeIn(ax_lab), run_time=0.8)
            self.play(LaggedStartMap(ShowCreation, rec, lag_ratio=0.08), run_time=1.2)
            self.play(LaggedStartMap(GrowFromCenter, mdots, lag_ratio=0.06), FadeIn(rec_lab), run_time=0.8)
            vo.wait_until(at_phrase(vo, 2, "continued"))
            self.play(LaggedStartMap(ShowCreation, cont, lag_ratio=0.06), FadeIn(cont_lab), run_time=1.3)
            self.play(LaggedStartMap(FadeIn, end_q, lag_ratio=0.06), run_time=0.6)

        self.toks, self.boxes, self.fills, self.arrows = toks, boxes, fills, arrows
        self.streams_lab, self.chans_lab, self.shared_lab, self.idle_lab = streams_lab, chans_lab, shared_lab, idle_lab
        self.timeline, self.end_q, self.runs_head, self.src = timeline, end_q, runs_head, src

    # ============================================================== N18.2  pushes and a z-loss
    def part_failed(self):
        toks, boxes, fills, arrows = self.toks, self.boxes, self.fills, self.arrows
        s21 = strip("repeated large pushes (S21)", S21, N_CONT, BAD)
        s27 = strip("router z-loss (S27)", S27, N_CONT, WARN)
        s24 = strip("row copy + noise + reset (S24)", S24, N_CONT, GOOD)
        s24c = strip("noise + reset alone", S24_CTRL, N_CONT, BAD)
        strips = VGroup(s21, s27, s24, s24c)
        for st, y in zip(strips, [1.75, 0.7, -0.35, -1.4]):
            st.move_to([0.75, y, 0], aligned_edge=UL)
        runs_head2 = L(f"{N_CONT} merged runs, continued", size=22, color=MUTED).move_to([0.75, 2.45, 0], aligned_edge=LEFT)
        self.strips = strips

        # kicks on the gate: stream 2's route swings toward the idle channel and springs back
        a2 = arrows[2]
        pivot, tip = a2.get_start(), a2.get_end()
        along = (tip - pivot) / np.linalg.norm(tip - pivot)
        perp = np.array([-along[1], along[0], 0.0])          # the way a counterclockwise swing moves the arrow
        if perp[0] < 0:
            perp = -perp

        def make_kicks():
            ks = VGroup()
            for f in (0.5, 0.68, 0.86):
                hit = pivot + f * (tip - pivot)
                ks.add(Arrow(hit - 0.62 * perp, hit - 0.08 * perp, buff=0, thickness=4).set_color(HINGE_COLOR))
            return ks

        kick_lab = L("large pushes on the gate", size=22, color=HINGE_COLOR, weight="BOLD")
        kick_lab.move_to([np.mean(XS), -2.2, 0])

        # logits of a merged stream's gate, and the softmax that reads them
        # chosen so the softmax matches the real merged gate's shape (0.94 shared / 0.04 idle, cf. 0.96 / 0.04)
        z = np.array([2.6, -2.0, -2.0, -0.6])
        c = 2.2
        bars = logit_bars(z)
        bars_low = logit_bars(z - c)
        zero = DashedLine([-6.1, ZERO_Y, 0], [-1.1, ZERO_Y, 0], dash_length=0.08).set_stroke(MUTED, 1.5)
        zero_lab = M(R"0", 26, MUTED).next_to(zero, LEFT, buff=0.12)
        logit_head = L("gate logits for one merged stream (schematic)", size=22, color=MUTED).move_to([-3.6, 2.5, 0])
        zl = M(R"\text{z-loss} = \Big(\log \sum_c e^{z_c}\Big)^2", 30).move_to([-3.6, 1.95, 0])
        p = softmax(z)
        gb = gate_bar(p, colors=SLOT_COLORS, width=3.6, height=0.38).move_to([-3.1, -2.05, 0])
        gb_lab = M(R"\mathrm{softmax}(z)", 28).next_to(gb, LEFT, buff=0.25)
        same = L("unchanged", size=22, color=GOOD, weight="BOLD").next_to(gb, DOWN, buff=0.18)
        eq = M(R"\mathrm{softmax}(z - c\,\mathbf{1}) = \mathrm{softmax}(z)", 30).move_to([-3.6, -3.05, 0])
        down = Arrow([-0.95, 1.4, 0], [-0.95, 0.2, 0], buff=0, thickness=4).set_color(WARN)
        down_lab = M(R"-c", 30, WARN).next_to(down, RIGHT, buff=0.1)
        src2 = source_note("Report §10 (S21, S27) · logits schematic")

        with self.voiceover(
            "Repeated large pushes on the gate split none of eight. A router z-loss split only two. It did lower the "
            "logits' scale, but by lowering them all together, along the one direction the softmax ignores."
        ) as vo:
            self.play(FadeIn(kick_lab), run_time=0.3)
            for i in range(2):
                kicks = make_kicks()
                self.play(LaggedStartMap(GrowArrow, kicks, lag_ratio=0.2), run_time=0.3)
                self.play(Rotate(a2, 0.75, about_point=pivot, rate_func=there_and_back),
                          FadeOut(kicks, shift=0.35 * perp), run_time=0.6)
            self.play(FadeOut(kick_lab), FadeOut(self.timeline), FadeTransform(self.runs_head, runs_head2),
                      *[ReplacementTransform(self.end_q[i], s21.dots[i]) for i in range(N_CONT)],
                      FadeIn(s21.lab), run_time=0.8)
            self.play(FadeIn(s21.cnt, shift=0.1 * LEFT), run_time=0.4)

            vo.wait_until_sentence(1)
            self.play(FadeIn(s27.lab), LaggedStartMap(FadeIn, s27.dots, lag_ratio=0.06), run_time=0.8)
            self.play(FadeIn(s27.cnt, shift=0.1 * LEFT), Indicate(s27.dots[:2], color=GOOD), run_time=0.7)
            self.play(FadeOut(self.src), run_time=0.3)        # swap source notes without a cross-fade

            vo.wait_until_sentence(2)
            # channel labels sit below the deepest bar the logits will reach
            lab_y = ZERO_Y - abs((z - c).min()) * LSCALE - 0.3
            self.play(FadeOut(VGroup(toks, arrows, fills, self.streams_lab, self.chans_lab), shift=0.1 * UP),
                      *[ReplacementTransform(boxes[j], bars[j]) for j in range(4)],
                      self.shared_lab.animate.move_to([XS[0], lab_y, 0]),
                      self.idle_lab.animate.move_to([XS[3], lab_y, 0]),
                      FadeIn(zero), FadeIn(zero_lab), FadeIn(logit_head), FadeIn(src2),
                      FadeIn(gb), FadeIn(gb_lab), run_time=1.1)
            vo.wait_until(at_phrase(vo, 2, "scale") - 0.3)
            self.play(Write(zl), run_time=0.8)
            vo.wait_until(at_phrase(vo, 2, "lowering them all") - 0.2)
            ghosts = VGroup(*[DashedVMobject(b.copy().set_fill(opacity=0).set_stroke(GATE_COLOR, 1.5, 0.8), num_dashes=24)
                              for b in bars])
            self.add(ghosts)
            self.play(GrowArrow(down), FadeIn(down_lab), Transform(bars, bars_low), run_time=1.2)
            self.play(FadeIn(same, shift=0.1 * UP), Indicate(gb, color=GOOD, scale_factor=1.05), run_time=0.7)
            vo.wait_until(at_phrase(vo, 2, "the one direction") - 0.3)
            self.play(Write(eq), run_time=1.0)

        self.bars = bars
        self.logit_extra = VGroup(zero, zero_lab, logit_head, zl, gb, gb_lab, same, eq, down, down_lab, ghosts)
        self.src = src2

    # ============================================================== N18.3  the row copy
    def part_surgery(self):
        strips = self.strips
        s24, s24c = strips[2], strips[3]
        rng = np.random.default_rng(18)
        W = rng.normal(0, 1, (4, 32))
        W[3] = rng.normal(-0.6, 0.5, 32)                       # the idle channel's row: low logits
        noise0, noise3 = rng.normal(0, 0.15, 32), rng.normal(0, 0.15, 32)   # S24 adds noise to both rows
        row_ys = [1.95, 1.5, 1.05, 0.6]
        rows = VGroup(*[wg_row(W[j], y) for j, y in zip(range(4), row_ys)])
        frames = VGroup(*[SurroundingRectangle(r, buff=0.02).set_stroke(GATE_COLOR, 1, opacity=0.6) for r in rows])
        row_tags = VGroup(*[slot_tag(HOLDS[j]) for j in range(4)])
        for t, r in zip(row_tags, rows):
            t.next_to(r, LEFT, buff=0.2)
        head = VGroup(L("gate readout", size=22, color=MUTED), M(R"W_g", 32, GATE_COLOR),
                      L("(row c gives channel c's logit)", size=22, color=MUTED)).arrange(RIGHT, buff=0.14)
        head.move_to([-3.4, 2.55, 0])
        copy_row = wg_row(W[0], row_ys[3])
        noisy_row = wg_row(W[0] + noise3, row_ys[3])
        noisy_row0 = wg_row(W[0] + noise0, row_ys[0])
        eps = M(R"+\,\varepsilon", 32, WARN).next_to(rows[3], RIGHT, buff=0.18)
        eps0 = M(R"+\,\varepsilon", 32, WARN).next_to(rows[0], RIGHT, buff=0.18)
        busy_box = SurroundingRectangle(rows[0], buff=0.06).set_stroke(BAD, 3)
        idle_box = SurroundingRectangle(rows[3], buff=0.06).set_stroke(MUTED, 2.5)
        opt_lab = L("optimizer state of", size=20, color=MUTED)
        opt_w = M(R"W_g", 28, GATE_COLOR)
        opt_cells = VGroup(*[Square(0.2).set_fill(WARN, 0.25 + 0.5 * abs(np.sin(i * 1.7))).set_stroke(BG, 1)
                             for i in range(12)]).arrange(RIGHT, buff=0.03)
        opt = VGroup(VGroup(opt_lab, opt_w).arrange(RIGHT, buff=0.1), opt_cells).arrange(RIGHT, buff=0.25)
        opt.move_to([-3.4, -0.15, 0])
        reset_lab = L("reset", size=22, color=WARN, weight="BOLD").next_to(opt_cells, DOWN, buff=0.12)
        src3 = source_note(f"Report §10 (S24) · gates: X seed 258 at 9600 (gate_dynamics.json) and {S24_SPLIT_STEP} "
                           "(S24 record) · W_g schematic")

        # two merged streams' gates, before and after the copy (real: update 9600, and S24's split at 13200)
        base_y, gmax = -2.75, 1.1
        cx0, cx2 = -4.85, -1.75
        g_before = [GATE_SHOWN[0], GATE_SHOWN[2]]
        g_level = [[0.51, 0, 0, 0.49], [0.49, 0, 0, 0.51]]          # right after the copy (equal rows): schematic
        g_split = [[S24_SPLIT_GATE[s][c] for c in DISPLAY] for s in (0, 2)]

        def gbars(p, s, cx):
            return prob_bars(p, STREAM_COLORS[s], cx, base_y, max_h=gmax)

        def gvals(p, s, cx):
            """Values over the shared and the idle bar."""
            return VGroup(*[L(f"{p[j]:.2f}", size=20, color=STREAM_COLORS[s]).move_to(
                [cx + (j - 1.5) * 0.6, base_y + max(p[j] * gmax, 0.012) + 0.17, 0]) for j in (0, 3)])

        b0 = gbars(g_before[0], 0, cx0)
        b2 = gbars(g_before[1], 2, cx2)
        v_before = VGroup(gvals(g_before[0], 0, cx0), gvals(g_before[1], 2, cx2))
        v_split = VGroup(gvals(g_split[0], 0, cx0), gvals(g_split[1], 2, cx2))
        bases = VGroup(*[Line([cx - 1.2, base_y, 0], [cx + 1.2, base_y, 0]).set_stroke(MUTED, 1.5) for cx in (cx0, cx2)])
        slot_labs = VGroup()
        for cx in (cx0, cx2):
            for j, txt in ((0, "shared"), (3, "idle")):
                slot_labs.add(L(txt, size=20, color=BAD if j == 0 else MUTED).move_to([cx + (j - 1.5) * 0.6, base_y - 0.25, 0]))
        g_titles = VGroup(L("stream 0's gate", size=22, color=STREAM_COLORS[0]).move_to([cx0, -1.05, 0]),
                          L("stream 2's gate", size=22, color=STREAM_COLORS[2]).move_to([cx2, -1.05, 0]))
        phase = T("before the copy: nearly the same", size=28, color=INK).move_to([-3.3, -0.5, 0])
        phase2 = T("after the copy: level", size=28, color=WARN).move_to(phase)
        phase3 = T("small differences grow: split", size=28, color=GOOD).move_to(phase)
        new_tags = VGroup(slot_tag([0]).move_to(row_tags[0], aligned_edge=RIGHT),
                          slot_tag([2]).move_to(row_tags[3], aligned_edge=RIGHT))

        with self.voiceover(
            "What worked was surgery: copy the busiest channel's gate row onto the idlest one, add a little noise, "
            "and reset that weight's optimizer state. Eight of eight split, against one of eight for the noise and "
            "the reset alone. The merged streams' gate states were nearly identical, but once the idle channel was "
            "level with the shared one, small differences could grow."
        ) as vo:
            self.play(FadeOut(self.logit_extra), FadeOut(self.src), FadeIn(src3),
                      *[ReplacementTransform(self.bars[j], frames[j]) for j in range(4)],
                      FadeOut(self.shared_lab), FadeOut(self.idle_lab), FadeIn(head), run_time=1.1)
            self.play(LaggedStart(*[LaggedStartMap(FadeIn, r, lag_ratio=0.03) for r in rows], lag_ratio=0.2),
                      LaggedStartMap(FadeIn, row_tags, lag_ratio=0.2), run_time=1.2)
            vo.wait_until(at_phrase(vo, 0, "copy the busiest"))
            self.play(ShowCreation(busy_box), Indicate(row_tags[0], scale_factor=1.3), run_time=0.8)
            vo.wait_until(at_phrase(vo, 0, "onto the idlest"))
            self.play(ShowCreation(idle_box), run_time=0.5)
            self.play(TransformFromCopy(rows[0], copy_row, path_arc=-0.6), FadeOut(rows[3]), run_time=1.2)
            vo.wait_until(at_phrase(vo, 0, "add a little noise"))
            self.play(Transform(copy_row, noisy_row), Transform(rows[0], noisy_row0),
                      FadeIn(eps, shift=0.1 * LEFT), FadeIn(eps0, shift=0.1 * LEFT), run_time=0.9)
            vo.wait_until(at_phrase(vo, 0, "reset that"))
            self.play(FadeIn(opt, shift=0.1 * UP), run_time=0.6)
            self.play(opt_cells.animate.set_fill(PANEL, 1).set_stroke(FAINT, 1), FadeIn(reset_lab), run_time=0.8)

            vo.wait_until_sentence(1)
            self.play(FadeIn(s24.lab), LaggedStartMap(FadeIn, s24.dots, lag_ratio=0.05), run_time=0.8)
            self.play(FadeIn(s24.cnt, shift=0.1 * LEFT), run_time=0.4)
            vo.wait_until(at_phrase(vo, 1, "against one"))
            self.play(FadeIn(s24c.lab), LaggedStartMap(FadeIn, s24c.dots, lag_ratio=0.05), run_time=0.8)
            self.play(FadeIn(s24c.cnt, shift=0.1 * LEFT), run_time=0.4)

            vo.wait_until_sentence(2)
            self.play(FadeOut(VGroup(opt, reset_lab)), run_time=0.4)
            self.play(FadeIn(bases), FadeIn(slot_labs), FadeIn(g_titles),
                      FadeIn(b0), FadeIn(b2), FadeIn(v_before), FadeIn(phase), run_time=0.8)
            self.play(Indicate(VGroup(b0[0], b2[0]), color=WARN, scale_factor=1.1), run_time=0.8)
            vo.wait_until(at_phrase(vo, 2, "once the idle"))
            self.play(Transform(b0, gbars(g_level[0], 0, cx0)), Transform(b2, gbars(g_level[1], 2, cx2)),
                      FadeOut(v_before), FadeTransform(phase, phase2),
                      Indicate(VGroup(copy_row, eps), color=WARN, scale_factor=1.04), run_time=1.2)
            vo.wait_until(at_phrase(vo, 2, "small differences"))
            self.play(Transform(b0, gbars(g_split[0], 0, cx0)), Transform(b2, gbars(g_split[1], 2, cx2)),
                      FadeTransform(phase2, phase3), run_time=1.8)
            self.play(FadeIn(v_split, shift=0.1 * UP), run_time=0.4)
            self.play(Transform(row_tags[0], new_tags[0]), Transform(row_tags[3], new_tags[1]),
                      busy_box.animate.set_stroke(STREAM_COLORS[0], 3), idle_box.animate.set_stroke(STREAM_COLORS[2], 3),
                      Indicate(s24.cnt, color=GOOD), run_time=0.9)

    # ============================================================== N18.4  a label-free trigger
    def part_trigger(self):
        keep = [self.title_mob, self.frame]
        title2 = section_title("Splitting without labels")
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], FadeTransform(self.title_mob, title2),
                  run_time=0.9)
        self.title_mob = title2

        # left top: one Muon run from scratch (S36 and S37 seed 241, identical until the split at 4800)
        ax = line_chart([0, 9600, 2400], [0, 1, 0.25], width=5.0, height=1.75, x_label="",
                        x_ticks=[0, 2400, 4800, 7200, 9600], y_ticks=[0, 0.5, 1], y_label="accuracy")
        ax.move_to([-3.25, 1.55, 0])
        curve = polyline(ax, R241_STEPS, R241_ACC, INK, 3.5)
        thr = DashedLine(ax.c2p(0, 0.95), ax.c2p(9600, 0.95), dash_length=0.08).set_stroke(GOOD, 2)
        thr_lab = L("0.95", size=20, color=GOOD).next_to(thr, RIGHT, buff=0.1)
        stall = Brace(VGroup(Dot(ax.c2p(2550, 0.64)), Dot(ax.c2p(4650, 0.64))), DOWN, buff=0.03).set_color(WARN)
        stall_lab = L("stalls", size=20, color=WARN).next_to(stall, DOWN, buff=0.08)
        trig = DashedLine(ax.c2p(4800, 0.05), ax.c2p(4800, 1.0), dash_length=0.06).set_stroke(WARN, 2.5)
        trig_lab = L("copy", size=22, color=WARN, weight="BOLD").next_to(ax.c2p(4800, 0.12), LEFT, buff=0.12)
        caption = L("seed 241, Muon · the trigger reads accuracy, no stream labels", size=20, color=MUTED)
        caption.next_to(ax, DOWN, buff=0.12)
        upd = L("update", size=20, color=MUTED).next_to(ax.c2p(9600, 0), RIGHT, buff=0.15)
        c36 = polyline(ax, *R241_S36, BAD, 3)
        c36_lab = L("all-position aim", size=20, color=BAD).move_to(ax.c2p(7900, 0.3))
        c37 = polyline(ax, *R241_S37, GOOD, 3.5)
        c37_lab = L("key-position aim", size=20, color=GOOD).next_to(ax.c2p(7800, 1.0), UP, buff=0.12)

        # left bottom: where to aim — the same run's gate mass per channel at 4800, all positions vs key positions
        base_y, mscale = -2.55, 3.0
        mx = [-5.0, -4.0, -3.0, -2.0]
        m_all = [R241_ALL[c] for c in R241_DISPLAY]
        m_key = [R241_KEY[c] for c in R241_DISPLAY]

        def mass_bars(m, color=GATE_COLOR):
            g = VGroup()
            for mj, x in zip(m, mx):
                h = max(mj * mscale, 0.012)
                g.add(Rectangle(width=0.62, height=h).set_fill(color, 0.75).set_stroke(color, 1).move_to([x, base_y + h / 2, 0]))
            return g

        def mark(word, ok, bar):
            col = GOOD if ok else BAD
            g = VGroup(L(word, size=20, color=col, weight="BOLD"), L("✓" if ok else "✗", size=26, color=col, weight="BOLD"))
            return g.arrange(RIGHT, buff=0.12).next_to(bar, UP, buff=0.1)

        mbars = mass_bars(m_all)
        key_bars = mass_bars(m_key)
        mbase = Line([mx[0] - 0.6, base_y, 0], [mx[-1] + 0.6, base_y, 0]).set_stroke(MUTED, 1.5)
        tags = VGroup(*[slot_tag(R241_HOLDS[j]).move_to([mx[j], base_y - 0.28, 0]) for j in range(4)])
        tag_cap = L("held at key positions", size=20, color=MUTED).next_to(tags, DOWN, buff=0.14)
        mhead_all = L("gate mass per channel, all positions", size=22, color=INK).move_to([-3.5, -0.4, 0])
        mhead_key = L("gate mass per channel, key positions only", size=22, color=INK).move_to(mhead_all)
        busy_all = mark("busiest", False, mbars[3])
        idle_all = mark("idlest", False, mbars[1])
        wrong_arc = CurvedArrow(busy_all.get_top() + 0.06 * UP, idle_all.get_top() + 0.06 * UP, angle=PI / 2.2)
        wrong_arc.set_color(BAD)
        wrong_lab = L("copy", size=20, color=BAD, weight="BOLD").next_to(wrong_arc.point_from_proportion(0.5), UP, buff=0.08)
        pick_key = mark("busiest", True, key_bars[0])
        idle_key = mark("idlest", True, key_bars[3])
        copy_arc = CurvedArrow(pick_key.get_right() + 0.08 * RIGHT, idle_key.get_top() + 0.06 * UP, angle=-PI / 2.2)
        copy_arc.set_color(WARN)
        copy_lab = L("copy row", size=20, color=WARN, weight="BOLD").next_to(copy_arc.point_from_proportion(0.5), UP, buff=0.08)

        # right: S36 results, then S37
        rx = 0.6
        h36 = L("plateau trigger (S36) · Muon · from scratch", size=20, color=MUTED)
        l36a = stat_line("with the copy", S36_BOUND, S36_N, GOOD, word="bound")
        l36b = stat_line("without it", S36_NONE, S36_N, GOOD, word="bound")
        aim36 = VGroup(L("aimed wrong in about half", size=24, color=BAD, weight="BOLD"),
                       L(f"of its {S36_SPLITS} splits", size=24, color=BAD, weight="BOLD")).arrange(DOWN, buff=0.1,
                                                                                               aligned_edge=LEFT)
        g36 = VGroup(h36, l36a, l36b).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        g36.move_to([rx, 2.0, 0], aligned_edge=UL)
        aim36.next_to(g36, DOWN, buff=0.55, aligned_edge=LEFT)

        h37 = L("aimed by key-position mass (S37)", size=22, color=MUTED)
        on_h = L("splits on target, Muon", size=20, color=MUTED)
        l37a = stat_line("key positions", S37_ON, S37_SPLITS, GOOD, bad=BAD)
        l37b = stat_line("all positions", S37_OLD_ON, S37_SPLITS, WARN, bad=BAD)
        b_h = L("bound, one channel per stream", size=20, color=MUTED)
        l37c = stat_line("Muon, split", S37_MUON, S36_N, GOOD)
        l37d = stat_line("Muon, no split", S37_MUON_NONE, S36_N, MUTED)
        l37e = stat_line("Adam, split", S37_ADAM, S36_N, GOOD)
        l37f = stat_line("Adam, no split", S37_ADAM_NONE, S36_N, MUTED)
        g37 = VGroup(h37, on_h, l37a, l37b, b_h, l37c, l37d, l37e, l37f).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        for m in (on_h, b_h):
            m.shift(0.12 * DOWN)
        b_h.shift(0.1 * DOWN)
        VGroup(l37c, l37d, l37e, l37f).shift(0.2 * DOWN)
        l37e.shift(0.12 * DOWN)
        l37f.shift(0.12 * DOWN)
        g37.move_to([rx, 2.55, 0], aligned_edge=UL)

        open_q = VGroup(T("Why does a split on target", size=34), T("sometimes fail?", size=34),
                        verdict_badge("OPEN", size=24, color=WARN)).arrange(DOWN, buff=0.18)
        open_q.move_to([-3.4, 1.75, 0])
        open_note = L(f"{S37_STUCK} Muon runs stayed merged, both splits on target", size=20, color=MUTED)
        open_note.next_to(open_q, DOWN, buff=0.25)
        chart = VGroup(ax, upd, curve, thr, thr_lab, stall, stall_lab, trig, trig_lab, caption, c36, c36_lab, c37, c37_lab)
        stuck = VGroup(*[SurroundingRectangle(d, buff=0.04).set_stroke(WARN, 2) for d in l37c.dots[S37_MUON:]])
        src4 = source_note("Report §10 (S36, S37) · curves and masses: seed 241, update 4800 (S36/S37 run records)")

        with self.voiceover(
            "Triggering the copy without labels, when accuracy stalls, was then screened from scratch under Muon: it "
            "bound five of ten, against four of ten without it, and it aimed at the wrong channel in about half of its "
            "thirty-one splits. Aiming by gate mass at key positions put ten of eleven splits on target under Muon, and "
            "bound six of ten with one channel per stream, against three without splitting; under Adam, six against "
            "five. In these small screens, why a split sometimes fails even on target is still open."
        ) as vo:
            self.play(FadeIn(ax), FadeIn(upd), FadeIn(src4), run_time=0.6)
            self.play(ShowCreation(curve), ShowCreation(thr), FadeIn(thr_lab), run_time=1.2)
            vo.wait_until(at_phrase(vo, 0, "when accuracy stalls"))
            self.play(GrowFromCenter(stall), FadeIn(stall_lab), run_time=0.6)
            self.play(ShowCreation(trig), FadeIn(trig_lab, shift=0.1 * LEFT), FadeIn(caption), run_time=0.8)
            vo.wait_until(at_phrase(vo, 0, "it bound five"))
            self.play(FadeIn(h36), FadeIn(l36a.lab), LaggedStartMap(FadeIn, l36a.dots, lag_ratio=0.05),
                      FadeIn(l36a.cnt), run_time=0.9)
            vo.wait_until(at_phrase(vo, 0, "against four"))
            self.play(FadeIn(l36b.lab), LaggedStartMap(FadeIn, l36b.dots, lag_ratio=0.05), FadeIn(l36b.cnt), run_time=0.9)
            vo.wait_until(at_phrase(vo, 0, "aimed at the wrong") - 0.4)
            self.play(FadeIn(mhead_all), ShowCreation(mbase), FadeIn(tags), FadeIn(tag_cap),
                      LaggedStartMap(GrowFromEdge, mbars, edge=DOWN, lag_ratio=0.1), run_time=1.0)
            self.play(FadeIn(busy_all, shift=0.1 * DOWN), FadeIn(idle_all, shift=0.1 * DOWN),
                      FadeIn(aim36, shift=0.1 * UP), run_time=0.7)
            self.play(ShowCreation(wrong_arc), FadeIn(wrong_lab), run_time=0.7)
            self.play(ShowCreation(c36), FadeIn(c36_lab), run_time=1.0)

            vo.wait_until_sentence(1)
            self.play(FadeTransform(mhead_all, mhead_key), Transform(mbars, key_bars),
                      FadeOut(VGroup(busy_all, idle_all, wrong_arc, wrong_lab)), run_time=1.0)
            self.play(FadeIn(pick_key, shift=0.1 * DOWN), FadeIn(idle_key, shift=0.1 * DOWN), run_time=0.5)
            self.play(ShowCreation(copy_arc), FadeIn(copy_lab), run_time=0.6)
            self.play(FadeOut(VGroup(g36, aim36), shift=0.2 * UP), ShowCreation(c37), FadeIn(c37_lab), run_time=0.7)
            self.play(FadeIn(h37, shift=0.1 * DOWN), FadeIn(on_h, shift=0.1 * DOWN), run_time=0.4)
            self.play(FadeIn(l37a.lab), LaggedStartMap(FadeIn, l37a.dots, lag_ratio=0.04), FadeIn(l37a.cnt), run_time=0.8)
            self.play(FadeIn(l37b.lab), LaggedStartMap(FadeIn, l37b.dots, lag_ratio=0.04), FadeIn(l37b.cnt), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "and bound six"))
            self.play(FadeIn(b_h), FadeIn(l37c.lab), LaggedStartMap(FadeIn, l37c.dots, lag_ratio=0.04),
                      FadeIn(l37c.cnt), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "against three"))
            self.play(FadeIn(l37d.lab), LaggedStartMap(FadeIn, l37d.dots, lag_ratio=0.04), FadeIn(l37d.cnt), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "under Adam"))
            self.play(FadeIn(VGroup(l37e, l37f), shift=0.1 * UP), run_time=0.8)

            vo.wait_until_sentence(2)
            self.play(FadeOut(chart, shift=0.2 * UP), run_time=0.6)
            self.play(Write(open_q[:2]), FadeIn(open_q[2], scale=0.8), run_time=1.3)
            self.play(FadeIn(open_note, shift=0.1 * UP), run_time=0.6)
            self.play(LaggedStartMap(ShowCreation, stuck, lag_ratio=0.1), Indicate(pick_key, color=WARN), run_time=1.0)
