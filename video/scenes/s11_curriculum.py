"""Chapter 11 — Eight streams by a stream curriculum (report §5, Table 1; test_stream_curriculum.py;
data/eight_streams.json, data/report_tables.json)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load(name):
    with open(os.path.join(DATA, name)) as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------------- real data
_E8 = _load("eight_streams.json")              # machine X, results/X/stream_curriculum_results.json
STEP = _E8["eval_every"]                       # 1200: curve value i is at step 1200 * (i + 1)
SC8 = _E8["arms"]["SC8"]
D8 = _E8["arms"]["D8"]
T1, T2, TOTAL = 4800, 9600, 28800              # stage switches and budget (test_stream_curriculum.py)
SEQ_LEN = {2: 27, 4: 51, 8: 99}                # 3 (s_active * P + 1), P = 4 (CHECK 96)

# Report §5 counts (X, L).  SC8 X and D8 X are also re-derived from the data (check_vs_report).
COUNTS = {"SC8": ((6, 16), (4, 20)), "D8": ((0, 6), (0, 10))}
assert (SC8["bound"], SC8["n"]) == COUNTS["SC8"][0] and (D8["bound"], D8["n"]) == COUNTS["D8"][0]


def _seed_idx(arm, pred):
    return [i for i, o in enumerate(arm["outcome"]) if pred(o)]


BINDERS = _seed_idx(SC8, lambda o: o.startswith("BOUND"))                   # 6 runs
ROUTED = [i for i in BINDERS if SC8["outcome"][i] == "BOUND ROUTED"]       # 3 runs
SHARED = [i for i in BINDERS if SC8["outcome"][i] != "BOUND ROUTED"]       # 3 runs
COLLAPSES = _seed_idx(SC8, lambda o: "collapsed" in o)                     # 4 runs
MERGES = _seed_idx(SC8, lambda o: o.startswith("MERGED"))                  # 6 runs
assert len(BINDERS) == 6 and len(ROUTED) == 3 and len(COLLAPSES) == 4 and len(MERGES) == 6
assert sorted({SC8["transition"][i] for i in BINDERS}) == [10800, 12000]
assert all(a < 0.15 for a in D8["final_acc"])

CHECK_FILL = "#C9CED8"                       # neutral fill for the stage bars: opacity ~ streams


def stage_fill(n_streams):
    return {2: 0.16, 4: 0.28, 8: 0.40}[n_streams]


# ------------------------------------------------------------------------------------- helpers
def at_phrase(vo, i, phrase):
    """Wait until (roughly) the moment `phrase` is spoken in sentence i of the block."""
    text, a, b = vo.synth.sentences[i]
    k = text.find(phrase)
    frac = k / max(len(text), 1) if k >= 0 else 0.0
    vo.wait_until(a + frac * (b - a))


def stream_dots(active, r=0.075, gap=0.2):
    """Eight stream dots; the active streams are filled in their colour, the rest faint rings."""
    g = VGroup()
    for s in range(8):
        d = Dot(radius=r)
        if s in active:
            d.set_fill(STREAM_COLORS[s], 1).set_stroke(STREAM_COLORS[s], 1)
        else:
            d.set_fill(BG, 1).set_stroke(FAINT, 1.2)
        g.add(d)
    g.arrange(RIGHT, buff=gap - 2 * r)
    return g


def run_cells(num, den, color, cell=0.26, gap=0.06):
    """One square per run, the first `num` filled (bound)."""
    sq = VGroup(*[Square(cell).set_stroke(PANEL_EDGE, 1.2).set_fill(PANEL, 1) for _ in range(den)])
    sq.arrange(RIGHT, buff=gap)
    fills = VGroup(*[Square(cell).set_stroke(color, 1.2).set_fill(color, 0.9).move_to(sq[i])
                     for i in range(num)])
    g = VGroup(sq, fills)
    g.track, g.fills = sq, fills
    return g


def chan_map_row(ch_map, r=0.07, pitch=0.17):
    """Streams grouped by the channel their read gate picks: one box per used channel.
    `.shared` holds the outlines of boxes that hold more than one stream."""
    groups = {}
    for s, c in enumerate(ch_map):
        groups.setdefault(c, []).append(s)
    boxes = VGroup()
    shared = VGroup()
    for c in sorted(groups):
        dots = VGroup(*[Dot(radius=r).set_fill(STREAM_COLORS[s], 1).set_stroke(width=0)
                        for s in groups[c]]).arrange(RIGHT, buff=pitch - 2 * r)
        box = RoundedRectangle(width=dots.get_width() + 0.1, height=0.26, corner_radius=0.06)
        box.set_fill(PANEL, 1).set_stroke(MUTED, 1)
        box.move_to(dots)
        boxes.add(VGroup(box, dots))
        if len(groups[c]) > 1:
            shared.add(box)
    boxes.arrange(RIGHT, buff=0.04)
    boxes.shared = shared
    return boxes


def curve(axes, ys, color, width=2.6, opacity=0.95):
    xs = [STEP * (i + 1) for i in range(len(ys))]
    return polyline(axes, xs, ys, color=color, width=width).set_stroke(opacity=opacity)


def swatch(color, w=0.3):
    return Line(ORIGIN, w * RIGHT).set_stroke(color, 4)


class StreamCurriculum(ClankersScene):
    def construct(self):
        card = self.chapter_card(11, "Eight streams by curriculum")
        self.wait(0.6)
        title = section_title("A stream curriculum")
        self.play(FadeOut(card, shift=0.3 * UP), FadeIn(title, shift=0.3 * UP))

        # ============================================================ N11.1 the curriculum
        toks = VGroup(*[token(f"CTX{s}", "ctx", s, width=0.8, height=0.5) for s in range(8)])
        toks.arrange(RIGHT, buff=0.1).move_to(0.9 * RIGHT + 2.05 * UP)
        toks_lab = L("8 streams", size=24, weight="BOLD").next_to(toks, LEFT, buff=0.45)
        chans = VGroup(*[RoundedRectangle(width=0.34, height=0.34, corner_radius=0.05)
                         .set_fill(PANEL, 1).set_stroke(MEMORY_COLOR, 1.2) for _ in range(16)])
        chans.arrange(RIGHT, buff=0.1).move_to(0.9 * RIGHT + 1.1 * UP)
        chans_lab = L("16 channels", size=24, color=MEMORY_COLOR, weight="BOLD")
        chans_lab.next_to(chans, LEFT, buff=0.45).align_to(toks_lab, RIGHT)

        # the timeline: updates 0 .. 28800 over 11 units
        X0, W, Y_AX = -5.3, 11.0, -1.2

        def tx(step):
            return X0 + W * step / TOTAL

        axis = Line([tx(0), Y_AX, 0], [tx(TOTAL) + 0.1, Y_AX, 0]).set_stroke(MUTED, 2)
        ticks = VGroup()
        for st in (0, T1, T2, 19200, TOTAL):
            tk = Line([tx(st), Y_AX, 0], [tx(st), Y_AX - 0.1, 0]).set_stroke(MUTED, 2)
            lab = L(f"{st}", size=20, color=MUTED).next_to(tk, DOWN, buff=0.08)
            ticks.add(VGroup(tk, lab))
        ax_lab = L("training updates", size=22, color=MUTED).move_to([tx(TOTAL / 2), Y_AX - 0.8, 0])
        BAR_H = 0.8

        def bar(a, b, n_streams, stroke=INK):
            r = Rectangle(width=tx(b) - tx(a), height=BAR_H)
            r.set_fill(CHECK_FILL, stage_fill(n_streams)).set_stroke(stroke, 1.2, opacity=0.8)
            r.move_to([(tx(a) + tx(b)) / 2, Y_AX + BAR_H / 2 + 0.03, 0])
            return r

        scratch = bar(0, TOTAL, 8)
        scratch_lab = L("all 8 streams from update 1", size=24).move_to(scratch)
        fail = VGroup(L("✗", size=34, color=BAD, weight="BOLD"),
                      L("did not bind", size=26, color=BAD, weight="BOLD")).arrange(RIGHT, buff=0.15)
        fail.next_to(scratch, UP, buff=0.22).align_to(scratch, RIGHT)
        src_a = source_note("Report §5; test_stream_curriculum.py")

        stages = [(0, T1, 2), (T1, T2, 4), (T2, TOTAL, 8)]
        ghosts = VGroup(*[bar(a, b, 2).set_fill(opacity=0.0).set_stroke(FAINT, 1.2) for a, b, _ in stages])
        bars = VGroup(*[bar(a, b, n) for a, b, n in stages])
        bar_labs = VGroup()
        for (a, b, n), r in zip(stages, bars):
            t1 = L(f"{n} streams", size=22, weight="BOLD")
            t2 = L(f"{SEQ_LEN[n]} tokens", size=22, color=INK)
            bar_labs.add(VGroup(t1, t2).arrange(DOWN, buff=0.08).move_to(r))
        dot_rows = VGroup(stream_dots({2, 5}), stream_dots({1, 3, 4, 6}), stream_dots(set(range(8))))
        for r, d in zip(bars, dot_rows):
            d.next_to(r, UP, buff=0.28)
        per_seq = L("streams in one sequence", size=22, color=MUTED)
        per_seq.next_to(dot_rows[2], RIGHT, buff=0.3)

        # the validity arm: a perfect gate, stream s -> channel s
        pg_dots = VGroup(*[Dot(radius=0.075).set_fill(STREAM_COLORS[s], 1).set_stroke(width=0)
                           for s in range(8)]).arrange(RIGHT, buff=0.1)
        pg_ch = VGroup(*[Square(0.23).set_fill(STREAM_COLORS[s], 0.35).set_stroke(STREAM_COLORS[s], 1.5)
                         for s in range(8)]).arrange(RIGHT, buff=0.05)
        pg_ch.next_to(pg_dots, DOWN, buff=0.3)
        for d, c in zip(pg_dots, pg_ch):
            d.set_x(c.get_x())
        pg_lines = VGroup(*[Line(d.get_bottom(), c.get_top(), buff=0.03).set_stroke(STREAM_COLORS[s], 2)
                            for s, (d, c) in enumerate(zip(pg_dots, pg_ch))])
        pg_pic = VGroup(pg_dots, pg_lines, pg_ch)
        pg_txt = L("perfect gate: stream s → channel s", size=22)
        pg_x = VGroup(machine_badge("X"), L("3/3", size=24, color=GOOD, weight="BOLD")).arrange(RIGHT, buff=0.15)
        pg_l = VGroup(machine_badge("L"), L("3/3", size=24, color=GOOD, weight="BOLD")).arrange(RIGHT, buff=0.15)
        valid = verdict_badge("VALID", size=24)
        pg_row = VGroup(pg_pic, pg_txt, pg_x, pg_l, valid).arrange(RIGHT, buff=0.45)
        pg_row.move_to(2.8 * DOWN)

        with self.voiceover(
            "From scratch, eight streams with sixteen channels did not bind. So the stream curriculum test trained "
            "on two of the eight streams per sequence for the first forty-eight hundred updates, then four, then "
            "all eight. The perfect gate bound on both machines, so the setup was valid."
        ) as vo:
            # "From scratch": all eight streams from update 1, on the updates axis
            self.play(ShowCreation(axis), FadeIn(ticks), FadeIn(ax_lab),
                      GrowFromEdge(scratch, LEFT), FadeIn(src_a), run_time=0.9)
            self.play(FadeIn(scratch_lab), run_time=0.3)
            at_phrase(vo, 0, "eight streams")
            self.play(LaggedStartMap(FadeIn, toks, shift=0.15 * DOWN, lag_ratio=0.08), FadeIn(toks_lab),
                      run_time=0.7)
            at_phrase(vo, 0, "sixteen channels")
            self.play(LaggedStartMap(FadeIn, chans, lag_ratio=0.04), FadeIn(chans_lab), run_time=0.6)
            at_phrase(vo, 0, "did not bind")
            self.play(FadeIn(fail, shift=0.15 * DOWN), scratch.animate.set_stroke(BAD, 2.0), run_time=0.5)

            # sentence 1: the scratch bar breaks into three stages
            at_phrase(vo, 1, "curriculum test")
            self.play(FadeOut(VGroup(chans, chans_lab, fail, scratch_lab)),
                      VGroup(toks, toks_lab).animate.shift(0.6 * DOWN),
                      ReplacementTransform(scratch, ghosts), run_time=0.9)
            at_phrase(vo, 1, "two of the eight")
            self.play(FadeIn(bars[0]), FadeIn(bar_labs[0][0]),
                      TransformFromCopy(toks, dot_rows[0]), run_time=0.9)
            at_phrase(vo, 1, "per sequence")
            for act in ({0, 6}, {3, 4}, {2, 5}):          # a fresh pair of streams for every sequence
                self.play(Transform(dot_rows[0], stream_dots(act).move_to(dot_rows[0])), run_time=0.35)
            self.play(FadeIn(bar_labs[0][1]), run_time=0.4)
            at_phrase(vo, 1, "forty-eight hundred")
            self.play(Indicate(ticks[1], color=WARN, scale_factor=1.3),
                      bars[0].animate.set_stroke(WARN, 2.5), run_time=0.8)
            at_phrase(vo, 1, "then four")
            self.play(FadeIn(bars[1]), FadeIn(bar_labs[1]), bars[0].animate.set_stroke(INK, 1.2),
                      TransformFromCopy(toks, dot_rows[1]), run_time=0.6)
            at_phrase(vo, 1, "then all eight")
            self.play(FadeIn(bars[2]), FadeIn(bar_labs[2]), TransformFromCopy(toks, dot_rows[2]),
                      FadeIn(per_seq), FadeOut(ghosts), run_time=0.7)

            # sentence 2: the perfect gate binds, so the test is valid
            vo.wait_until_sentence(2)
            self.play(FadeIn(pg_pic, shift=0.15 * UP), FadeIn(pg_txt), run_time=0.8)
            at_phrase(vo, 2, "on both machines")
            self.play(FadeIn(pg_x, shift=0.1 * UP), FadeIn(pg_l, shift=0.1 * UP), run_time=0.6)
            at_phrase(vo, 2, "so the setup")
            self.play(FadeIn(valid, scale=0.8), run_time=0.6)

        # ============================================================ N11.2 counts and C1
        CELL = 0.26
        X_CELLS = -4.7
        icon_cur = VGroup(*[Rectangle(width=w, height=0.3).set_fill(CHECK_FILL, stage_fill(n))
                            .set_stroke(INK, 1, opacity=0.8) for w, n in ((0.3, 2), (0.3, 4), (1.2, 8))])
        icon_cur.arrange(RIGHT, buff=0)
        icon_d8 = Rectangle(width=1.8, height=0.3).set_fill(CHECK_FILL, stage_fill(8)).set_stroke(INK, 1, opacity=0.8)
        head_cur_txt = VGroup(L("curriculum, 2 → 4 → 8", size=24, weight="BOLD"),
                              L("(SC8)", size=22, color=MUTED)).arrange(RIGHT, buff=0.25)
        head_cur = VGroup(head_cur_txt, icon_cur).arrange(RIGHT, buff=0.25)
        head_d8 = VGroup(L("all eight from the start", size=24, weight="BOLD"),
                         L("(D8)", size=22, color=MUTED), icon_d8).arrange(RIGHT, buff=0.25)
        head_cur.move_to([-6.2, 2.25, 0], aligned_edge=LEFT)
        head_d8.move_to([-6.2, 0.4, 0], aligned_edge=LEFT)

        rows = {}
        ys = {("SC8", 0): 1.65, ("SC8", 1): 1.1, ("D8", 0): -0.2, ("D8", 1): -0.75}
        for (arm, m), y in ys.items():
            num, den = COUNTS[arm][m]
            cells = run_cells(num, den, GOOD, cell=CELL)
            cells.move_to([X_CELLS, y, 0], aligned_edge=LEFT)
            badge = machine_badge("XL"[m], size=20).move_to([X_CELLS - 0.45, y, 0])
            val = L(f"{num}/{den}", size=26, color=GOOD if num else BAD, weight="BOLD")
            val.next_to(cells, RIGHT, buff=0.3)
            rows[arm, m] = (badge, cells, val)
        legend = VGroup(VGroup(Square(0.22).set_fill(GOOD, 0.9).set_stroke(GOOD, 1.2), L("bound", size=22)).arrange(RIGHT, buff=0.12),
                        VGroup(Square(0.22).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1.2), L("did not bind", size=22)).arrange(RIGHT, buff=0.12))
        legend.arrange(RIGHT, buff=0.4)
        legend_head = L("one square per run", size=22, color=MUTED)
        legend = VGroup(legend_head, legend).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        legend.move_to([4.6, 1.4, 0])

        c1_head = VGroup(L("C1", size=26, color=MUTED, weight="BOLD"),
                         T("“the curriculum beats training from scratch”", size=30)).arrange(RIGHT, buff=0.3)
        c1_head.move_to([0, -1.55, 0])
        pv_x = VGroup(machine_badge("X", size=20), L("3 vs 0 paired seeds,  p = 0.125", size=24)).arrange(RIGHT, buff=0.2)
        pv_l = VGroup(machine_badge("L", size=20), L("2 vs 0 paired seeds,  p = 0.25", size=24)).arrange(RIGHT, buff=0.2)
        pvs = VGroup(pv_x, pv_l).arrange(RIGHT, buff=1.2).move_to([0, -2.3, 0])
        ns_x = verdict_badge("NOT SHOWN", size=20).next_to(pv_x, DOWN, buff=0.22)
        ns_l = verdict_badge("NOT SHOWN", size=20).next_to(pv_l, DOWN, buff=0.22)
        need = L("SHOWN needs p < 0.05", size=22, color=MUTED).move_to([0, -2.95, 0])
        src_c = source_note("Report §5")

        timeline_mobs = VGroup(axis, ticks, ax_lab, bar_labs, dot_rows, per_seq, toks, toks_lab, pg_row)
        with self.voiceover(
            "The curriculum bound six of sixteen runs on X and four of twenty on L; training on all eight from the "
            "start bound none, zero of six and zero of ten. With so few runs, neither difference was significant, "
            "so \"the curriculum beats training from scratch\" was not shown."
        ) as vo:
            self.play(FadeOut(timeline_mobs), FadeOut(src_a), run_time=0.5)
            self.play(ReplacementTransform(bars, icon_cur), FadeIn(head_cur_txt),
                      *[FadeIn(VGroup(rows["SC8", m][0], rows["SC8", m][1].track)) for m in (0, 1)],
                      FadeIn(legend), FadeIn(src_c), run_time=0.8)
            at_phrase(vo, 0, "six of sixteen")
            b, c, v = rows["SC8", 0]
            self.play(LaggedStartMap(FadeIn, c.fills, scale=0.6, lag_ratio=0.15), FadeIn(v), run_time=1.0)
            at_phrase(vo, 0, "four of twenty")
            b, c, v = rows["SC8", 1]
            self.play(LaggedStartMap(FadeIn, c.fills, scale=0.6, lag_ratio=0.15), FadeIn(v), run_time=0.9)
            at_phrase(vo, 0, "training on all eight")
            self.play(FadeIn(head_d8, shift=0.1 * DOWN),
                      *[FadeIn(VGroup(rows["D8", m][0], rows["D8", m][1].track)) for m in (0, 1)], run_time=0.8)
            at_phrase(vo, 0, "zero of six")
            self.play(FadeIn(rows["D8", 0][2], shift=0.1 * LEFT), run_time=0.5)
            at_phrase(vo, 0, "zero of ten")
            self.play(FadeIn(rows["D8", 1][2], shift=0.1 * LEFT), run_time=0.5)

            vo.wait_until_sentence(1)
            d8_cells = VGroup(rows["D8", 0][1], rows["D8", 1][1])
            self.play(*[FlashAround(c, color=WARN, stroke_width=3) for c in d8_cells],
                      Indicate(VGroup(rows["D8", 0][2], rows["D8", 1][2]), color=WARN), run_time=1.1)
            at_phrase(vo, 1, "neither difference")
            self.play(LaggedStartMap(FadeIn, pvs, shift=0.15 * UP, lag_ratio=0.4), FadeIn(need), run_time=1.0)
            at_phrase(vo, 1, "the curriculum beats")
            self.play(Write(c1_head), run_time=1.0)
            at_phrase(vo, 1, "was not shown")
            self.play(FadeIn(ns_x, scale=0.8), FadeIn(ns_l, scale=0.8), FadeOut(need), run_time=0.6)
        self.play(FlashAround(ns_x, color=MUTED), FlashAround(ns_l, color=MUTED), run_time=1.0)

        # ============================================================ N11.2 the curves (machine X)
        axes = line_chart([0, TOTAL, 4800], [0, 1, 0.25], width=7.2, height=4.3,
                          x_label="training updates", y_label="held-out accuracy, all 8 streams",
                          x_ticks=[0, T1, T2, 19200, TOTAL], y_ticks=[0, 0.5, 1])
        axes.shift(np.array([-5.1, -2.45, 0]) - axes.c2p(0, 0))
        band = VGroup()
        for a, b, n in stages:
            r = Rectangle(width=axes.c2p(b, 0)[0] - axes.c2p(a, 0)[0], height=0.42)
            r.set_fill(CHECK_FILL, stage_fill(n)).set_stroke(INK, 1, opacity=0.8)
            r.move_to([(axes.c2p(a, 0)[0] + axes.c2p(b, 0)[0]) / 2, 2.25, 0])
            band.add(r)
        band_labs = VGroup(*[L(f"{n} streams" if n == 8 else f"{n}", size=22, weight="BOLD").move_to(r)
                             for (a, b, n), r in zip(stages, band)])
        guides = VGroup(*[DashedLine(axes.c2p(s, 0), axes.c2p(s, 1.0), dash_length=0.08)
                          .set_stroke(FAINT, 1.5) for s in (T1, T2)])
        guides_up = VGroup(*[DashedLine(axes.c2p(s, 1.0), [axes.c2p(s, 0)[0], 2.04, 0], dash_length=0.08)
                             .set_stroke(FAINT, 1.5) for s in (T1, T2)])
        x_note = VGroup(L("machine X", size=22, color=MUTED), L("one line per run", size=22, color=MUTED))
        x_note.arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        x_note.move_to(axes.c2p(500, 1.0), aligned_edge=UL)
        src_d = source_note("results/X/stream_curriculum_results.json; Report §5")

        d8_curves = VGroup(*[curve(axes, ys_, MUTED, width=2.2) for ys_ in D8["curves"].values()])
        thr = DashedLine(axes.c2p(0, 0.15), axes.c2p(TOTAL, 0.15), dash_length=0.1).set_stroke(BAD, 2)
        thr_lab = L("0.15", size=22, color=BAD).next_to(axes.c2p(0, 0.15), LEFT, buff=0.15)
        seeds = SC8["seeds"]
        bind_curves = VGroup(*[curve(axes, SC8["curves"][str(seeds[i])], GOOD, width=2.8) for i in BINDERS])
        bind_dots = VGroup()
        for i in BINDERS:
            t = SC8["transition"][i]
            y = SC8["curves"][str(seeds[i])][t // STEP - 1]
            bind_dots.add(Dot(axes.c2p(t, y), radius=0.06).set_fill(GOOD, 1).set_stroke(BG, 1.5))
        bind_note = L("bound at 10,800 or 12,000", size=22, color=GOOD)
        bind_note.move_to(axes.c2p(13800, 0.86), aligned_edge=LEFT)
        col_curves = VGroup(*[curve(axes, SC8["curves"][str(seeds[i])], BAD, width=2.4) for i in COLLAPSES])
        mer_curves = VGroup(*[curve(axes, SC8["curves"][str(seeds[i])], WARN, width=2.4) for i in MERGES])

        # side panel, x in [2.75, 6.65]
        PX = 2.75

        def head(text, color):
            return L(text, size=22, color=color, weight="BOLD")

        p1 = VGroup(VGroup(swatch(MUTED), head("from scratch (D8)", INK)).arrange(RIGHT, buff=0.15),
                    L("X: all 6 ended below 0.15", size=22),
                    L("L: 8 of 10 below 0.15, 2 near 0.2", size=22)).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        p1.move_to([PX, 2.15, 0], aligned_edge=UL)
        p2_head = VGroup(swatch(GOOD), head("curriculum binders, X", INK)).arrange(RIGHT, buff=0.15)
        p2_sub = L("stream → channel at the end", size=22, color=MUTED)
        p2_top = VGroup(p2_head, p2_sub).arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        p2_top.next_to(p1, DOWN, buff=0.35, aligned_edge=LEFT)
        map_rows = VGroup()
        y0 = p2_top.get_bottom()[1] - 0.3
        for j, i in enumerate(ROUTED + SHARED):
            row = chan_map_row(SC8["ch_map_end"][i])
            tag = (L("1 per stream", size=22, color=GOOD) if i in ROUTED else L("shared", size=22, color=WARN))
            y = y0 - j * 0.34 - (0.08 if j >= 3 else 0.0)
            row.move_to([PX, y, 0], aligned_edge=LEFT)
            tag.move_to([PX + 2.4, y, 0], aligned_edge=LEFT)
            map_rows.add(VGroup(row, tag))
        p2 = VGroup(p2_top, map_rows)
        # failures: X's runs are the lines drawn; the pooled X + L counts are the report's
        CX1, CX2 = PX + 3.05, PX + 3.6
        p3_head = VGroup(head("curriculum failures", INK),
                         L("X", size=20, color=MUTED, weight="BOLD"),
                         L("X + L", size=20, color=MUTED, weight="BOLD"))
        p3_rows = VGroup()
        for color, text, nx, nxl in ((BAD, "collapses to one channel", len(COLLAPSES), 8),
                                     (WARN, "merges", len(MERGES), 18)):
            p3_rows.add(VGroup(VGroup(swatch(color), L(text, size=20)).arrange(RIGHT, buff=0.15),
                               L(f"{nx}", size=22, color=color, weight="BOLD"),
                               L(f"{nxl}", size=22, color=color, weight="BOLD")))
        p3 = VGroup(p3_head, *p3_rows)
        y3 = p2.get_bottom()[1] - 0.5
        for j, r in enumerate(p3):
            yy = y3 - 0.36 * j
            r[0].move_to([PX, yy, 0], aligned_edge=LEFT)
            r[1].move_to([CX1, yy, 0])
            r[2].move_to([CX2, yy, 0])

        bars_mobs = VGroup(head_d8, legend, c1_head, pvs, ns_x, ns_l,
                           *[VGroup(*rows[k]) for k in rows])
        self.play(FadeOut(bars_mobs), FadeOut(head_cur_txt), FadeOut(src_c), run_time=0.6)
        self.play(ReplacementTransform(icon_cur, band), ShowCreation(axes.x_axis), ShowCreation(axes.y_axis),
                  run_time=1.0)
        self.play(FadeIn(VGroup(axes.tick_labels, axes.x_label_mob, axes.y_label_mob)), FadeIn(band_labs),
                  FadeIn(guides), FadeIn(guides_up), FadeIn(x_note), FadeIn(src_d), run_time=0.6)

        with self.voiceover(
            "The scratch runs mostly collapsed below 0.15. On X, every curriculum binder bound soon after the switch "
            "to eight streams, but three of its six bound without one channel per stream. Over both machines, the "
            "curriculum's failures were collapses and merges."
        ) as vo:
            self.play(ShowCreation(d8_curves, lag_ratio=0.1), FadeIn(p1[0]), run_time=1.4)
            self.play(ShowCreation(thr), FadeIn(thr_lab), FadeIn(p1[1:]), run_time=0.8)

            vo.wait_until_sentence(1)
            self.play(d8_curves.animate.set_stroke(FAINT, opacity=0.7), FadeOut(thr), FadeOut(thr_lab),
                      ShowCreation(bind_curves, lag_ratio=0.12), FadeIn(p2_head), run_time=1.8)
            at_phrase(vo, 1, "soon after the switch")
            self.play(guides[1].animate.set_stroke(WARN, 3), guides_up[1].animate.set_stroke(WARN, 3),
                      Indicate(band[2], color=WARN, scale_factor=1.02),
                      LaggedStartMap(FadeIn, bind_dots, scale=0.5, lag_ratio=0.1),
                      FadeIn(bind_note, shift=0.1 * LEFT), run_time=1.0)
            at_phrase(vo, 1, "but three of its six")
            self.play(FadeIn(p2_sub), LaggedStartMap(FadeIn, map_rows[:3], shift=0.1 * RIGHT, lag_ratio=0.2),
                      run_time=0.8)
            self.play(LaggedStartMap(FadeIn, map_rows[3:], shift=0.1 * RIGHT, lag_ratio=0.25), run_time=0.9)
            self.play(*[r[0].shared.animate.set_stroke(WARN, 2.5) for r in map_rows[3:]],
                      *[Indicate(r[1], color=WARN, scale_factor=1.15) for r in map_rows[3:]], run_time=1.0)

            vo.wait_until_sentence(2)
            self.play(guides[1].animate.set_stroke(FAINT, 1.5), guides_up[1].animate.set_stroke(FAINT, 1.5),
                      bind_curves.animate.set_stroke(opacity=0.45), bind_dots.animate.set_opacity(0.45),
                      FadeOut(bind_note), FadeIn(p3_head[:2]), run_time=0.6)
            # X's failed runs as lines, with X's counts ...
            self.play(ShowCreation(col_curves, lag_ratio=0.05), FadeIn(p3_rows[0][:2]), run_time=1.0)
            self.play(ShowCreation(mer_curves, lag_ratio=0.05), FadeIn(p3_rows[1][:2]), run_time=1.0)
            # ... then the counts over both machines
            self.play(FadeIn(p3_head[2], shift=0.1 * DOWN), run_time=0.4)
            at_phrase(vo, 2, "collapses")
            self.play(FadeIn(p3_rows[0][2], scale=1.4), col_curves.animate.set_stroke(width=3.6), run_time=0.5)
            at_phrase(vo, 2, "merges")
            self.play(FadeIn(p3_rows[1][2], scale=1.4), mer_curves.animate.set_stroke(width=3.6), run_time=0.5)

        # ============================================================ recap: the test's verdicts
        self.wait(0.4)
        chart = VGroup(axes, band, band_labs, guides, guides_up, x_note, d8_curves, bind_curves, bind_dots,
                       col_curves, mer_curves, p1, p2, p3)
        self.play(FadeOut(chart), FadeOut(src_d), run_time=0.7)
        recap_head = VGroup(T("The stream curriculum test", size=36),
                            L("same verdicts on X and L", size=22, color=MUTED)).arrange(RIGHT, buff=0.4,
                                                                                       aligned_edge=DOWN)
        recap_rows = [
            ("VALID", "", "perfect gate bound 3/3 on each machine"),
            ("NOT SHOWN", "C1", "beats scratch: 3 vs 0 (p = 0.125) on X, 2 vs 0 (p = 0.25) on L"),
            ("MINORITY", "C2", "reliable with restarts? SC8_R bound 7/16 on X, 5/20 on L"),
            ("UNTESTABLE", "C3", "do runs routed at step 4800 bind more? no run was routed at 4800"),
        ]
        rec = VGroup()
        for verdict, name, text in recap_rows:
            nm = L(name, size=24, color=MUTED, weight="BOLD") if name else VectorizedPoint()
            vb = verdict_badge(verdict, size=22)
            tx_ = L(text, size=26)
            rec.add(VGroup(nm, vb, tx_))
        for j, r in enumerate(rec):
            y = 1.0 - 0.8 * j
            r[0].move_to([-5.7, y, 0], aligned_edge=LEFT)
            r[1].move_to([-4.9, y, 0], aligned_edge=LEFT)
            r[2].move_to([-2.75, y, 0], aligned_edge=LEFT)
        recap_head.move_to([-5.7, 2.0, 0], aligned_edge=LEFT)
        recap = VGroup(recap_head, rec)
        src_r = source_note("Report §5, Table 1")
        self.play(FadeIn(recap_head, shift=0.15 * UP), FadeIn(src_r), run_time=0.5)
        self.play(LaggedStartMap(FadeIn, rec, shift=0.15 * RIGHT, lag_ratio=0.35), run_time=1.8)
        self.play(Indicate(rec[2][1], scale_factor=1.08), Indicate(rec[3][1], scale_factor=1.08), run_time=0.9)
        self.wait(1.4)
        self.clear_all()
