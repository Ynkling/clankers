"""Chapter 12 — the race: what the screens found about the gate's first move (report §6, Table 3, Table 10).

Per-seed data (seeds 160-199, two streams, four keys, k = 2, Adam 1e-3, no convolution), re-derived from the
records and checked against the report's counts:
  A          results/X/short_conv_results.json, arm A ("A (X, recorded)"): discovered 12/40, ROUTED* at 1200 11/40
  SLOW       screens S5 (seeds 160-179) and S7 (180-199), explore_out/slow_mem{,_ext}_results.json on branch
             claude/outside-ideas: discovered 24/40, ROUTED* at 1200 23/40; failures 11 POSITION, 3 KEY, 2 OTHER
  HINGE      screen S13, explore_out/slow_hinge_results.json: discovered 34/40; firings per run (end hinge_counts[0])
  first firings: the update of S16's first kick (explore_out/kick_control_results.json, kick_log). S16 is bit-identical
             to slow memory, and so to S13, until the hinge would first fire, so its first kick lands on S13's first
             firing. Shown only for the nine seeds where S13's hinge fired exactly once and the run bound.
ROUTED* = margin >= 0.9 and eta^2 by stream > 0.9 at the step (report §3); classes by test_router_layout.fail_class.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403

SEEDS = list(range(160, 200))
A_DISC = "0001110001000010101001110000000010000100"        # 12/40 (7/20 on 160-179)
A_R1200 = "0000110001000010101001110000000010000100"       # 11/40
SLOW_CLASS = "DDPDDDPPDDDODPDPPPDDPDDDDDDKDKPDOKDDDPDP"    # D = discovered; P/K/O = POSITION/KEY/OTHER
SLOW_R1200 = "1101110011101010001101111110100100110010"    # 23/40
HINGE_CLASS = "DDDDDDDODDDODDDDDDDDDDDDDDDKDKDDDODDDODD"   # 34/40, no POSITION failure left
HINGE_FIRED = [0, 0, 3, 0, 0, 0, 1, 834, 1, 0, 0, 34, 1, 1, 0, 1, 1, 1, 0, 0,
               2, 0, 0, 0, 0, 0, 0, 2, 0, 11, 3, 0, 3, 179, 0, 1, 1, 1938, 0, 2]
HINGE_STOP = [4800, 6000, 6000, 6000, 6000, 7200, 4800, 24000, 6000, 4800, 6000, 24000, 4800, 4800, 6000, 6000,
              6000, 6000, 4800, 4800, 4800, 6000, 6000, 4800, 6000, 4800, 6000, 24000, 4800, 24000, 4800, 4800,
              4800, 24000, 4800, 4800, 4800, 24000, 4800, 4800]
# seeds where S13's hinge fired exactly once and the run bound: (seed, update of that firing)
ONE_FIRING = [(175, 49), (177, 50), (173, 126), (176, 133), (168, 138), (195, 151), (166, 155), (196, 164),
              (172, 3572)]

SLOW_DISC = "".join("1" if c == "D" else "0" for c in SLOW_CLASS)
HINGE_DISC = "".join("1" if c == "D" else "0" for c in HINGE_CLASS)
assert A_DISC.count("1") == 12 and SLOW_DISC.count("1") == 24 and HINGE_DISC.count("1") == 34
assert A_R1200.count("1") == 11 and SLOW_R1200.count("1") == 23 and SLOW_CLASS.count("P") == 11
assert sum(1 for a, s in zip(A_DISC, SLOW_DISC) if s > a) == 15 and sum(1 for a, s in zip(A_DISC, SLOW_DISC) if a > s) == 3
assert sum(1 for s, h in zip(SLOW_DISC, HINGE_DISC) if h > s) == 10 and sum(1 for s, h in zip(SLOW_DISC, HINGE_DISC) if s > h) == 0
assert sum(1 for f in HINGE_FIRED if f == 0) == 20

# streams at the eight key positions of one grouped sequence (a key's two triples sit together)
SEQ_STREAMS = [1, 0, 0, 1, 1, 0, 0, 1]

# strip geometry (blocks 3-5)
DX = 0.22
X0 = -3.6
COUNT_X = 5.3


def at(vo, i, phrase=None, frac=0.0):
    """Time (in the block) at which `phrase` is reached inside sentence i, by character position."""
    text, a, b = vo.synth.sentences[i]
    if phrase is not None:
        k = text.find(phrase)
        frac = max(k, 0) / max(len(text), 1)
    return a + frac * (b - a)


def split_cells(kind, cell=0.34, buff=0.05, ring=2.5):
    """Eight key positions of one sequence: ring = the token's stream, fill = the channel the gate chose."""
    cells = VGroup()
    for j, s in enumerate(SEQ_STREAMS):
        ch = s if kind == "stream" else int(j >= 4)
        box = RoundedRectangle(width=cell, height=cell, corner_radius=cell * 0.18)
        box.set_stroke(STREAM_COLORS[s], ring).set_fill(opacity=0)
        inner = RoundedRectangle(width=cell * 0.6, height=cell * 0.6, corner_radius=cell * 0.1)
        inner.set_fill(STREAM_COLORS[ch], 0.95).set_stroke(width=0)
        cells.add(VGroup(box, inner))
    cells.arrange(RIGHT, buff=buff)
    return cells


def node_box(text, color=MUTED, size=20, dashed=False, text_color=INK):
    t = L(text, size, text_color)
    box = RoundedRectangle(width=t.get_width() + 0.35, height=t.get_height() + 0.3, corner_radius=0.1)
    box.set_fill(PANEL, 1).set_stroke(color, 1.8)
    if dashed:
        box = DashedVMobject(box, num_dashes=28).set_stroke(color, 1.8)
        bg = RoundedRectangle(width=box.get_width(), height=box.get_height(), corner_radius=0.1)
        bg.set_fill(PANEL, 1).set_stroke(width=0)
        t.move_to(bg)
        box.move_to(bg)
        return VGroup(bg, box, t)
    t.move_to(box)
    return VGroup(box, t)


def strip_dot(kind, r=0.075):
    """kind: 'D' discovered, 'P' POSITION failure (highlighted later), other = not discovered."""
    if kind == "D":
        return Dot(radius=r).set_fill(GOOD, 1).set_stroke(GOOD, 1)
    return Circle(radius=r).set_fill(BG, 1).set_stroke(FAINT, 1.8)


def seed_strip(classes, y, x0=X0, dx=DX):
    dots = VGroup(*[strip_dot(c) for c in classes])
    for i, d in enumerate(dots):
        d.move_to([x0 + i * dx, y, 0])
    return dots


def row_label(name, sub, y, color=INK):
    a = L(name, 24, color, weight="BOLD")
    b = L(sub, 20, MUTED)
    g = VGroup(a, b).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
    g.move_to([-6.6, y, 0], aligned_edge=LEFT)
    return g


def conn_legend(*entries, y=-3.2):
    """Second legend row for the seed-by-seed connectors: (text, colour) per entry."""
    g = VGroup(*[VGroup(Line(0.13 * DOWN, 0.13 * UP).set_stroke(c, 2.5), L(t, 20, MUTED)).arrange(RIGHT, buff=0.12)
                 for t, c in entries]).arrange(RIGHT, buff=0.5)
    g.move_to([X0 - 0.1, y, 0], aligned_edge=LEFT)
    return g


def mini_bar(label, num, den, color, y, x_bar, width=2.4, label_size=20, value_color=None):
    track = Rectangle(width=width, height=0.2).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1)
    track.move_to([x_bar + width / 2, y, 0])
    fill = Rectangle(width=max(width * num / den, 1e-3), height=0.2).set_fill(color, 0.9).set_stroke(width=0)
    fill.align_to(track, LEFT).set_y(y)
    name = L(label, label_size, INK).next_to(track, LEFT, buff=0.2)
    val = L(f"{num}/{den}", 20, value_color or color, weight="BOLD").next_to(track, RIGHT, buff=0.15)
    g = VGroup(name, track, fill, val)
    g.fill, g.val = fill, val
    return g


class Screens(ClankersScene):
    def construct(self):
        card = self.chapter_card(12, "The race")
        self.wait(0.6)
        self.title = section_title("Screens")
        self.play(FadeOut(card, shift=0.3 * UP), FadeIn(self.title, shift=0.3 * UP))

        self.part_screens()
        self.part_observations()
        self.part_race()
        self.part_slow()
        self.part_hinge()
        self.part_rare()
        # let the summary table stand a little longer, with the carried-forward screen marked
        self.play(FlashAround(self.hinge_trow, color=HINGE_COLOR, buff=0.08), run_time=1.5)
        self.wait(1.8)
        self.clear_all()

    # ------------------------------------------------------------------ helpers
    def retitle(self, text):
        new = section_title(text)
        old = self.title
        self.title = new
        return FadeTransform(old, new)

    def clear_but_title(self, run_time=0.7, extra=()):
        mobs = [m for m in self.mobjects if m is not self.title and m is not self.frame]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], *extra, run_time=run_time)

    # ================================================================== N12.1 screens
    def part_screens(self):
        lane_y = 1.75
        ml_name = L("main line", 26, INK, weight="BOLD")
        ml_badges = VGroup(machine_badge("X", 20), machine_badge("L", 20)).arrange(RIGHT, buff=0.12)
        ml_sub = L("pre-registered tests", 20, MUTED)
        ml_head = VGroup(ml_name, VGroup(ml_badges, ml_sub).arrange(RIGHT, buff=0.18)).arrange(
            DOWN, buff=0.14, aligned_edge=LEFT)
        ml_head.move_to([-6.6, lane_y, 0], aligned_edge=LEFT)
        lane = Arrow([-3.2, lane_y, 0], [6.55, lane_y, 0], buff=0, thickness=0.03).set_fill(MUTED, 0.6)
        n1 = node_box("stream recipe").move_to([-1.75, lane_y, 0])
        n2 = node_box("stream curriculum").move_to([0.95, lane_y, 0])
        n_q = node_box("new recipe ?", color=WARN, dashed=True, text_color=WARN).move_to([4.55, lane_y, 0])

        e_badge = machine_badge("E", 26).move_to([-5.95, -1.75, 0])
        e_lab = VGroup(L("container E", 20), L("own branch", 20, MUTED)).arrange(DOWN, buff=0.06)
        e_lab.next_to(e_badge, DOWN, buff=0.15)
        names = ["S4", "S5", "S7", "S11", "S12", "S13", "S16", "S20", "S26"]
        cards = VGroup()
        for nm in names:
            box = RoundedRectangle(width=0.9, height=0.6, corner_radius=0.1).set_fill(PANEL, 1).set_stroke(MUTED, 1.5)
            t = L(nm, 22, INK, weight="BOLD").move_to(box)
            cards.add(VGroup(box, t))
        cards.arrange(RIGHT, buff=0.17).move_to([0.05, -1.75, 0])
        banner = L("exploratory, not a result", 20, WARN).move_to([cards.get_left()[0], -2.65, 0], aligned_edge=LEFT)
        setup = L("mostly 2 streams, 4 keys", 20, INK).move_to([cards.get_right()[0], -2.65, 0], aligned_edge=RIGHT)
        src = source_note("Report §6; Table 10 (exploratory batches, container E)")

        # pairing inset: screen S5 vs X's recorded arm A on seeds 160-179
        p_dx, p_x0 = 0.25, -2.3
        top = VGroup(*[Circle(radius=0.075).set_stroke(MEMORY_COLOR, 2).set_fill(BG, 1) for _ in range(20)])
        bot = VGroup(*[Circle(radius=0.075).set_stroke(MUTED, 2).set_fill(BG, 1) for _ in range(20)])
        for i in range(20):
            top[i].move_to([p_x0 + i * p_dx, 0.45, 0])
            bot[i].move_to([p_x0 + i * p_dx, -0.35, 0])
        links = VGroup(*[Line(top[i].get_bottom(), bot[i].get_top()).set_stroke(FAINT, 1.5) for i in range(20)])
        t_lab = L("screen S5", 20, MEMORY_COLOR).next_to(top, LEFT, buff=0.3)
        b_lab = L("arm A, recorded on X", 20, MUTED).next_to(bot, LEFT, buff=0.3)
        s_lab = L("seeds 160 … 179", 20, MUTED).next_to(top, UP, buff=0.18)
        same = VGroup(L("same seed:", 20, INK), L("same initial weights,", 20, MUTED),
                      L("same batches", 20, MUTED)).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
        same.next_to(VGroup(top, bot), RIGHT, buff=0.45)
        inset = VGroup(top, bot, links, t_lab, b_lab, s_lab, same)

        with self.voiceover(
            "Where did the new recipe come from? From screens: cheap exploratory runs, mostly at two streams and "
            "four keys, each paired by seed with a recorded run or with an earlier screen. A screen that looks "
            "promising goes back to the main line for a proper test."
        ) as vo:
            self.play(FadeIn(ml_head, shift=0.2 * RIGHT), GrowArrow(lane), run_time=0.9)
            self.play(LaggedStart(FadeIn(n1), FadeIn(n2), FadeIn(n_q, scale=1.15), lag_ratio=0.3), run_time=0.9)
            vo.wait_until_sentence(1)
            self.play(FadeIn(e_badge, scale=0.6), FadeIn(e_lab), run_time=0.7)
            for c in cards:
                c.save_state()
                c.move_to(e_badge).scale(0.3).set_opacity(0)
            self.play(LaggedStart(*[Restore(c) for c in cards], lag_ratio=0.12), FadeIn(src), run_time=1.6)
            self.play(FadeIn(banner, shift=0.1 * UP), FadeIn(setup, shift=0.1 * UP), run_time=0.7)
            vo.wait_until(at(vo, 1, "each paired"))
            self.play(Indicate(cards[1], color=MEMORY_COLOR, scale_factor=1.15), run_time=0.7)
            self.play(LaggedStartMap(FadeIn, top, lag_ratio=0.04), LaggedStartMap(FadeIn, bot, lag_ratio=0.04),
                      FadeIn(t_lab), FadeIn(b_lab), FadeIn(s_lab), run_time=1.0)
            self.play(LaggedStartMap(ShowCreation, links, lag_ratio=0.05), FadeIn(same, shift=0.1 * LEFT), run_time=1.0)
            vo.wait_until_sentence(2)
            s13 = cards[5]
            badge = verdict_badge("promising", 20).next_to(s13, UP, buff=0.12)
            self.play(FadeOut(inset), s13[0].animate.set_stroke(HINGE_COLOR, 2.5),
                      *[c.animate.set_opacity(0.35) for i, c in enumerate(cards) if i != 5],
                      FadeIn(badge, shift=0.1 * UP), run_time=0.8)
            target = node_box("slow start test", color=GOOD).move_to(n_q)
            mover = VGroup(s13.copy(), badge.copy())
            self.play(mover.animate(path_arc=-PI / 4).move_to(n_q).scale(0.9), FadeOut(n_q, scale=0.8),
                      run_time=1.1)
            self.play(FadeTransform(mover, target), run_time=0.7)
            self.play(Indicate(target, color=GOOD, scale_factor=1.08), run_time=0.7)
        self.wait(0.3)
        self.clear_but_title(extra=[self.retitle("What the screens saw")])

    # ================================================================== N12.2a observations
    def part_observations(self):
        # --- routing at 1200 predicts routing at the end
        ax_y = 2.0
        seg1 = Line([-5.6, ax_y, 0], [-0.95, ax_y, 0]).set_stroke(MUTED, 2)
        seg2 = Line([-0.55, ax_y, 0], [4.3, ax_y, 0]).set_stroke(MUTED, 2)
        brk = VGroup(*[Line(0.12 * DOWN + 0.05 * LEFT, 0.12 * UP + 0.05 * RIGHT).set_stroke(MUTED, 2).move_to(
            [x, ax_y, 0]) for x in (-0.9, -0.6)])
        x12, xend = -2.95, 1.65
        t0 = VGroup(Line(0.1 * DOWN, 0.1 * UP).set_stroke(MUTED, 2).move_to([-5.6, ax_y, 0]),
                    L("update 0", 20, MUTED).next_to([-5.6, ax_y, 0], UP, buff=0.18))
        t12 = VGroup(Line(0.12 * DOWN, 0.12 * UP).set_stroke(GATE_COLOR, 3).move_to([x12, ax_y, 0]),
                     L("update 1200", 22, GATE_COLOR, weight="BOLD").next_to([x12, ax_y, 0], UP, buff=0.18))
        tend = VGroup(Line(0.1 * DOWN, 0.1 * UP).set_stroke(MUTED, 2).move_to([xend, ax_y, 0]),
                      L("end of training", 20, MUTED).next_to([xend, ax_y, 0], UP, buff=0.18))
        axis = VGroup(seg1, seg2, brk, t0)

        ya, yb = 1.0, -0.15
        pa1 = split_cells("stream", cell=0.3, buff=0.045).move_to([x12, ya, 0])
        pb1 = split_cells("position", cell=0.3, buff=0.045).move_to([x12, yb, 0])
        pa2 = split_cells("stream", cell=0.3, buff=0.045).move_to([xend, ya, 0])
        pb2 = split_cells("position", cell=0.3, buff=0.045).move_to([xend, yb, 0])
        ra = L("routed", 22, GOOD, weight="BOLD").next_to(pa2, RIGHT, buff=0.3)
        rb = L("not routed", 22, BAD, weight="BOLD").next_to(pb2, RIGHT, buff=0.3)
        la = L("stream split", 20, MUTED).next_to(pa1, LEFT, buff=0.3)
        lb = L("position split", 20, MUTED).next_to(pb1, LEFT, buff=0.3)
        arr_a = Arrow(pa1.get_right(), pa2.get_left(), buff=0.25, thickness=0.03).set_fill(INK, 1)
        arr_b = Arrow(pb1.get_right(), pb2.get_left(), buff=0.25, thickness=0.03).set_fill(INK, 1)
        pred = L("predicts", 20, INK).next_to(arr_a, UP, buff=0.08)
        pred_b = L("predicts", 20, INK).next_to(arr_b, UP, buff=0.08)
        legend = L("ring: the token's stream    fill: the channel the gate chose (8 key positions)", 20, MUTED)
        legend.move_to([0, -3.05, 0])
        xb = 5.75
        bind_h = L("binds?", 22, INK, weight="BOLD").next_to([xb, ax_y, 0], UP, buff=0.18)
        bind_qs = VGroup(T("?", 40, MUTED).move_to([xb, ya, 0]), T("?", 40, MUTED).move_to([xb, yb, 0]))
        bind_n = L("not predicted", 20, MUTED).move_to([xb, yb - 0.62, 0])
        bind_sep = DashedLine([xb - 0.7, ax_y - 0.25, 0], [xb - 0.7, yb - 0.45, 0], dash_length=0.08).set_stroke(FAINT, 1.5)
        bind_q = VGroup(bind_h, bind_qs, bind_n)
        early = T("the gate's split is decided early", 30, INK).move_to([0, -1.75, 0])

        # --- sparse memory codes steer the gate
        grid = memory_grid(6, 6, cell=0.3, color=MEMORY_COLOR)
        grid.move_to([-5.15, 0.55, 0])
        dense = [i for i in range(36) if (i * 7) % 5 != 0]
        sparse = [3, 14, 22, 31]
        for i in dense:
            grid.cells[i].set_fill(MEMORY_COLOR, 0.55)
        g_lab = VGroup(L("memory", 24, MEMORY_COLOR, weight="BOLD"), L("sparse codes for", 20, MUTED),
                       L("the first 2400 updates", 20, MUTED)).arrange(DOWN, buff=0.06)
        g_lab.next_to(grid, DOWN, buff=0.25)
        gate = VGroup(RoundedRectangle(width=1.5, height=0.85, corner_radius=0.12).set_fill(GATE_COLOR, 0.15)
                      .set_stroke(GATE_COLOR, 2.5), L("gate", 26, GATE_COLOR, weight="BOLD"))
        gate[1].move_to(gate[0])
        gate.move_to([-1.95, 0.55, 0])
        gate_lab = L("untouched", 22, GATE_COLOR).next_to(gate, DOWN, buff=0.25)
        steer = Arrow(grid.get_right(), gate.get_left(), buff=0.2, thickness=0.035).set_fill(MEMORY_COLOR, 1)
        steer_lab = L("steers?", 20, MEMORY_COLOR).next_to(steer, UP, buff=0.08)

        bx = 3.0
        h1 = L("routed by stream at update 1200", 22, INK).move_to([bx + 0.6, 1.75, 0])
        b1 = mini_bar("sparse codes (S4)", 15, 20, GATE_COLOR, 1.15, bx, width=2.3)
        b2 = mini_bar("plain gate (A)", 6, 20, MUTED, 0.65, bx, width=2.3, value_color=INK)
        h2 = L("discovered", 22, INK).move_to([bx + 0.6, -0.15, 0])
        b3 = mini_bar("sparse codes (S4)", 7, 20, GOOD, -0.75, bx, width=2.3)
        b4 = mini_bar("plain gate (A)", 7, 20, MUTED, -1.25, bx, width=2.3, value_color=INK)
        seeds_note = L("seeds 160–179, paired", 20, MUTED).move_to([bx + 0.6, -1.85, 0])
        verdict = verdict_badge("route early, bind no more", 22, WARN).move_to([0.2, -2.75, 0])
        src = source_note("Report §6; Table 3 (screen S4, seeds 160–179)")
        src0 = source_note("Report §6")

        with self.voiceover(
            "Several observations shaped everything. First, routing at step twelve hundred predicts routing at the "
            "end, though not binding: the gate's split is decided early. And what the memory learns seems to steer "
            "it: sparse memory codes, with the gate untouched, made the gate route by stream at step twelve hundred "
            "in fifteen of twenty runs instead of six, without making it bind."
        ) as vo:
            self.play(ShowCreation(seg1), ShowCreation(seg2), FadeIn(brk), FadeIn(t0), run_time=1.0)
            self.play(FadeIn(t12, shift=0.1 * DOWN), FadeIn(tend, shift=0.1 * DOWN), run_time=0.7)
            vo.wait_until_sentence(1)
            self.play(LaggedStartMap(FadeIn, pa1, lag_ratio=0.08), LaggedStartMap(FadeIn, pb1, lag_ratio=0.08),
                      FadeIn(la), FadeIn(lb), FadeIn(legend), FadeIn(src0), run_time=1.2)
            vo.wait_until(at(vo, 1, "predicts"))
            self.play(GrowArrow(arr_a), GrowArrow(arr_b), FadeIn(pred), FadeIn(pred_b),
                      TransformFromCopy(pa1, pa2), TransformFromCopy(pb1, pb2), run_time=1.2)
            self.play(FadeIn(ra, shift=0.1 * LEFT), FadeIn(rb, shift=0.1 * LEFT), run_time=0.6)
            vo.wait_until(at(vo, 1, "though not"))
            self.play(ShowCreation(bind_sep), FadeIn(bind_h, shift=0.1 * DOWN), run_time=0.5)
            self.play(LaggedStartMap(FadeIn, bind_qs, scale=1.4, lag_ratio=0.3), FadeIn(bind_n), run_time=0.8)
            vo.wait_until(at(vo, 1, "the gate's split"))
            self.play(Write(early), Indicate(t12, color=GATE_COLOR, scale_factor=1.15), run_time=1.2)

            vo.wait_until_sentence(2)
            first = VGroup(axis, t12, tend, pa1, pb1, pa2, pb2, ra, rb, la, lb, arr_a, arr_b, pred, pred_b, legend,
                           bind_q, bind_sep, early)
            self.play(FadeOut(first, shift=0.2 * UP), run_time=0.7)
            self.play(FadeIn(grid), FadeIn(g_lab[0]), FadeIn(gate), run_time=0.8)
            self.play(GrowArrow(steer), FadeIn(steer_lab), run_time=0.7)
            vo.wait_until(at(vo, 2, "sparse memory"))
            self.play(*[grid.cells[i].animate.set_fill(MEMORY_COLOR, 0.04) for i in dense if i not in sparse],
                      *[grid.cells[i].animate.set_fill(MEMORY_COLOR, 0.95) for i in sparse],
                      FadeIn(g_lab[1:]), run_time=1.2)
            vo.wait_until(at(vo, 2, "with the gate"))
            self.play(FadeIn(gate_lab, shift=0.1 * UP), Indicate(gate, color=GATE_COLOR, scale_factor=1.06),
                      run_time=0.8)
            vo.wait_until(at(vo, 2, "made the gate"))
            self.play(FadeIn(h1), FadeIn(b1[:2]), FadeIn(b2[:2]), FadeTransform(src0, src), run_time=0.6)
            for b in (b1, b2):
                b.fill.save_state()
                b.fill.stretch(1e-3, 0, about_edge=LEFT)
            self.play(Restore(b1.fill), FadeIn(b1.val), run_time=0.9)
            self.play(Restore(b2.fill), FadeIn(b2.val), run_time=0.9)
            vo.wait_until(at(vo, 2, "without making"))
            for b in (b3, b4):
                b.fill.save_state()
                b.fill.stretch(1e-3, 0, about_edge=LEFT)
            self.play(FadeIn(h2), FadeIn(b3[:2]), FadeIn(b4[:2]), FadeIn(seeds_note), run_time=0.4)
            self.play(Restore(b3.fill), Restore(b4.fill), FadeIn(b3.val), FadeIn(b4.val), run_time=0.8)
            self.play(FadeIn(verdict, shift=0.1 * UP), run_time=0.5)
        self.wait(0.8)
        self.clear_but_title(extra=[self.retitle("The race")])

    # ================================================================== N12.2b the race
    def part_race(self):
        lx0, lx1 = -3.0, 4.2
        gy, my = 1.15, -0.85
        self.lanes = (lx0, lx1, gy, my)

        def pt(y, p):
            return np.array([lx0 + p * (lx1 - lx0), y, 0])

        beds = VGroup(*[Line(pt(y, 0), pt(y, 1)).set_stroke(PANEL_EDGE, 12) for y in (gy, my)])
        starts = VGroup(*[Line(pt(y, 0) + 0.25 * DOWN, pt(y, 0) + 0.25 * UP).set_stroke(MUTED, 2) for y in (gy, my)])
        start_lab = L("update 0", 20, MUTED).next_to(starts[0], UP, buff=0.15)
        finish = DashedLine([lx1, 1.95, 0], [lx1, -1.7, 0], dash_length=0.1).set_stroke(INK, 2)
        fin_lab = L("commitment", 22, INK, weight="BOLD").next_to(finish, UP, buff=0.12)

        g_lab = VGroup(L("gate", 28, GATE_COLOR, weight="BOLD"), L("searches for the", 20, INK),
                       L("stream split", 20, INK)).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
        g_lab.move_to([-6.6, gy, 0], aligned_edge=LEFT)
        m_lab = VGroup(L("memory", 28, MEMORY_COLOR, weight="BOLD"), L("exploits a", 20, INK),
                       L("position split", 20, INK)).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
        m_lab.move_to([-6.6, my, 0], aligned_edge=LEFT)

        pic_g = split_cells("stream", cell=0.2, buff=0.035, ring=1.6).move_to([5.65, gy, 0])
        pic_m = split_cells("position", cell=0.2, buff=0.035, ring=1.6).move_to([5.65, my, 0])
        cap_g = L("stream split", 20, GOOD).next_to(pic_g, UP, buff=0.15)
        cap_m = L("position split", 20, BAD).next_to(pic_m, UP, buff=0.15)
        early_late = VGroup(L("early", 20, MUTED).next_to(pic_m[:4], DOWN, buff=0.12),
                            L("late", 20, MUTED).next_to(pic_m[4:], DOWN, buff=0.12))
        relief = L("half the keys per channel: less key confusion", 20, MUTED).move_to([0.6, -1.7, 0])

        pg, pm = ValueTracker(0.0), ValueTracker(0.0)
        self.pg, self.pm = pg, pm

        def runner(color):
            return VGroup(Circle(radius=0.32).set_fill(color, 0.18).set_stroke(width=0),
                          Dot(radius=0.17).set_fill(color, 1).set_stroke(INK, 1.5))

        rg, rm = runner(GATE_COLOR), runner(MEMORY_COLOR)
        rg.add_updater(lambda m: m.move_to(pt(gy, pg.get_value())))
        rm.add_updater(lambda m: m.move_to(pt(my, pm.get_value())))
        trail_g = always_redraw(lambda: Line(pt(gy, 0), pt(gy, max(pg.get_value(), 1e-3))).set_stroke(GATE_COLOR, 8, 0.75))
        trail_m = always_redraw(lambda: Line(pt(my, 0), pt(my, max(pm.get_value(), 1e-3))).set_stroke(MEMORY_COLOR, 8, 0.75))
        self.runners = (rg, rm, trail_g, trail_m)
        self.race_static = VGroup(beds, starts, start_lab, finish, fin_lab, g_lab, m_lab, pic_g, pic_m, cap_g, cap_m)
        self.pic_g, self.pic_m, self.finish = pic_g, pic_m, finish

        unaided = L("unaided, the gate found the stream split in 12 of 40 runs (arm A, seeds 160–199)", 22, INK)
        unaided.move_to([0, -2.65, 0])
        src = source_note("Report §6, Table 3; arm A per seed: results/X/short_conv_results.json")

        with self.voiceover(
            "So the report reads early training as a race. The gate is searching for the stream split. Meanwhile "
            "the memory is learning too, and it can start exploiting some other split, early positions against "
            "late ones, which already relieves part of the key confusion. Whichever settles first tends to win."
        ) as vo:
            self.play(ShowCreation(beds), FadeIn(starts), FadeIn(start_lab), run_time=0.9)
            self.play(ShowCreation(finish), FadeIn(fin_lab, shift=0.1 * DOWN), run_time=0.8)
            vo.wait_until_sentence(1)
            self.add(trail_g)
            self.play(FadeIn(g_lab, shift=0.2 * RIGHT), FadeIn(rg, scale=0.5), FadeIn(pic_g), FadeIn(cap_g),
                      run_time=0.8)
            self.play(pg.animate.set_value(0.16), run_time=1.1, rate_func=linear)
            vo.wait_until_sentence(2)
            self.add(trail_m)
            self.play(FadeIn(m_lab, shift=0.2 * RIGHT), FadeIn(rm, scale=0.5), pg.animate.set_value(0.2),
                      run_time=0.8, rate_func=linear)
            self.play(pm.animate.set_value(0.3), pg.animate.set_value(0.27), run_time=2.0, rate_func=linear)
            self.play(FadeIn(pic_m), FadeIn(cap_m), pm.animate.set_value(0.45), pg.animate.set_value(0.31),
                      run_time=0.9, rate_func=linear)
            # rate_func goes on the trackers only: a play-wide rate_func=linear would override Indicate's
            # there_and_back and leave the cells scaled and recoloured.
            self.play(FadeIn(early_late, shift=0.1 * UP), FlashAround(pic_m[:4], color=STREAM_COLORS[0], buff=0.06),
                      pm.animate(rate_func=linear).set_value(0.58), pg.animate(rate_func=linear).set_value(0.35),
                      run_time=1.0)
            self.play(FlashAround(pic_m[4:], color=STREAM_COLORS[1], buff=0.06),
                      pm.animate(rate_func=linear).set_value(0.68), pg.animate(rate_func=linear).set_value(0.38),
                      run_time=1.0)
            vo.wait_until(at(vo, 2, "which already"))
            self.play(FadeIn(relief, shift=0.1 * UP), pm.animate.set_value(0.85), pg.animate.set_value(0.43),
                      run_time=2.2, rate_func=linear)
            vo.wait_until_sentence(3)
            self.play(pm.animate.set_value(1.0), pg.animate.set_value(0.47), run_time=0.7, rate_func=rush_into)
            self.play(Flash(pt(my, 1.0), color=MEMORY_COLOR, flash_radius=0.45), Indicate(pic_m, color=BAD),
                      cap_m.animate.set_color(BAD), run_time=0.8)
            self.play(FadeIn(unaided, shift=0.1 * UP), FadeIn(src), run_time=0.7)
        self.wait(0.8)
        self.race_extra = VGroup(early_late, relief, unaided, src)

    # ================================================================== N12.3 slow memory
    def part_slow(self):
        lx0, lx1, gy, my = self.lanes
        pg, pm = self.pg, self.pm
        rg, rm, trail_g, trail_m = self.runners

        tag_t = L("a tenth of the learning rate", 20, BG, weight="BOLD")   # "÷ 10" read as "+ 10" at 480p
        tag = VGroup(RoundedRectangle(width=tag_t.get_width() + 0.35, height=0.4, corner_radius=0.12)
                     .set_fill(MEMORY_COLOR, 0.95).set_stroke(width=0), tag_t)
        tag[1].move_to(tag[0])
        tag.add_updater(lambda m: m.next_to(rm, UP, buff=0.12))

        # learning-rate schedule (two Adam groups)
        ox, oy, w = -2.2, -3.05, 5.4
        y4, y3 = oy + 0.3, oy + 1.05

        def X(u):
            return ox + w * u / 6000

        axes = VGroup(Line([ox, oy, 0], [ox + w, oy, 0]).set_stroke(MUTED, 2),
                      Line([ox, oy, 0], [ox, y3 + 0.3, 0]).set_stroke(MUTED, 2))
        xt = VGroup(*[L(s, 20, MUTED).next_to([X(u), oy, 0], DOWN, buff=0.1) for u, s in ((0, "0"), (2400, "2400"),
                                                                                          (6000, "6000"))])
        upd = L("updates", 20, MUTED).next_to([ox + w, oy, 0], RIGHT, buff=0.15)
        yt = VGroup(M(R"10^{-3}", 24, MUTED).next_to([ox, y3, 0], LEFT, buff=0.12),
                    M(R"10^{-4}", 24, MUTED).next_to([ox, y4, 0], LEFT, buff=0.12))
        gate_line = Line([X(0), y3, 0], [X(6000), y3, 0]).set_stroke(GATE_COLOR, 7)
        rest_line = VMobject().set_points_as_corners([[X(0), y4, 0], [X(2400), y4, 0], [X(2400), y3, 0],
                                                      [X(6000), y3, 0]]).set_stroke(MEMORY_COLOR, 3.5)
        gl = L("gate", 20, GATE_COLOR).next_to([X(500), y3, 0], UP, buff=0.1)
        rl = L("everything else", 20, MEMORY_COLOR).next_to([X(1200), y4, 0], UP, buff=0.1)
        sched = VGroup(axes, xt, upd, yt, gate_line, rest_line, gl, rl)
        src_lr = source_note("Report §3, recipe SLOW")

        # per-seed strips
        ya, ys = 1.4, 0.0
        seed_labs = VGroup(L("seed", 20, MUTED), *[L(str(s), 20, MUTED) for s in (160, 180, 199)])
        seed_labs[0].move_to([X0 - 0.25, 1.95, 0], aligned_edge=RIGHT)
        for lab, i in zip(seed_labs[1:], (0, 20, 39)):
            lab.move_to([X0 + i * DX, 1.95, 0])
        a_lab = row_label("plain gate (A)", "X, recorded", ya)
        s_lab = row_label("slow memory", "screens S5, S7", ys, SLOW_COLOR)
        a_dots = seed_strip(["D" if c == "1" else "N" for c in A_DISC], ya)
        s_dots = seed_strip(SLOW_CLASS, ys)
        a_cnt = L("12/40", 26, GOOD, weight="BOLD").move_to([COUNT_X, ya, 0], aligned_edge=LEFT)
        s_cnt = L("24/40", 26, GOOD, weight="BOLD").move_to([COUNT_X, ys, 0], aligned_edge=LEFT)
        wins = [i for i in range(40) if SLOW_DISC[i] == "1" and A_DISC[i] == "0"]
        loss = [i for i in range(40) if A_DISC[i] == "1" and SLOW_DISC[i] == "0"]
        conn = VGroup(*[Line(a_dots[i].get_bottom(), s_dots[i].get_top()).set_stroke(GOOD, 2.5) for i in wins],
                      *[Line(a_dots[i].get_bottom(), s_dots[i].get_top()).set_stroke(BAD, 2.5) for i in loss])
        disc = VGroup(L("15", 24, GOOD, weight="BOLD"), L("vs", 20, MUTED), L("3", 24, BAD, weight="BOLD"))
        disc.arrange(RIGHT, buff=0.12).move_to([COUNT_X, (ya + ys) / 2, 0], aligned_edge=LEFT)

        def tri(dot):
            return Triangle().set_width(0.14).set_fill(GATE_COLOR, 1).set_stroke(width=0).next_to(dot, UP, buff=0.05)

        a_tri = VGroup(*[tri(a_dots[i]) for i in range(40) if A_R1200[i] == "1"])
        s_tri = VGroup(*[tri(s_dots[i]) for i in range(40) if SLOW_R1200[i] == "1"])

        lg_y = -2.75
        lg = VGroup(
            VGroup(Dot(radius=0.075).set_fill(GOOD, 1), L("discovered", 20, MUTED)).arrange(RIGHT, buff=0.12),
            VGroup(Circle(radius=0.075).set_stroke(FAINT, 1.8), L("not discovered", 20, MUTED)).arrange(RIGHT, buff=0.12),
        ).arrange(RIGHT, buff=0.5)
        lg.move_to([X0 - 0.1, lg_y, 0], aligned_edge=LEFT)
        lg_tri = VGroup(Triangle().set_width(0.14).set_fill(GATE_COLOR, 1).set_stroke(width=0),
                        L("routed by stream at update 1200:  A 11 → slow 23", 20, GATE_COLOR)).arrange(RIGHT, buff=0.12)
        lg_tri.next_to(lg, RIGHT, buff=0.5)
        lg_conn = conn_legend(("only slow memory discovered", GOOD), ("only A discovered", BAD))
        src = source_note("Report §6, Table 3; per seed: X short_conv (A), screens S5, S7")

        with self.voiceover(
            "So, slow the memory down. For the first twenty-four hundred updates, everything except the gate trains "
            "at a tenth of the learning rate. Pooled over two screens and forty seeds, discovery rose from twelve to "
            "twenty-four, and twenty-three runs routed at step twelve hundred instead of eleven."
        ) as vo:
            self.play(FadeOut(self.race_extra), self.retitle("Slow memory"), pg.animate.set_value(0.0),
                      pm.animate.set_value(0.0), run_time=0.9)
            self.play(FadeIn(tag, scale=0.6), run_time=0.5)
            vo.wait_until_sentence(1)
            self.play(pg.animate.set_value(1.0), pm.animate.set_value(0.3), FadeIn(axes), FadeIn(xt), FadeIn(upd),
                      FadeIn(yt), FadeIn(src_lr), run_time=2.4, rate_func=linear)
            self.play(Flash(np.array([lx1, gy, 0]), color=GATE_COLOR, flash_radius=0.45),
                      Indicate(self.pic_g, color=GOOD), ShowCreation(gate_line), ShowCreation(rest_line),
                      FadeIn(gl), FadeIn(rl), run_time=1.2)
            self.play(Indicate(rest_line, color=MEMORY_COLOR, scale_factor=1.0), run_time=0.8)
            vo.wait_until_sentence(2)
            tag.clear_updaters()
            rg.clear_updaters()
            rm.clear_updaters()
            trail_g.clear_updaters()
            trail_m.clear_updaters()
            self.play(FadeOut(VGroup(self.race_static, rg, rm, trail_g, trail_m, tag, sched, src_lr)), run_time=0.6)
            self.play(FadeIn(seed_labs), FadeIn(a_lab), LaggedStartMap(FadeIn, a_dots, lag_ratio=0.03),
                      FadeIn(lg), FadeIn(src), run_time=1.0)
            self.play(FadeIn(s_lab), LaggedStartMap(FadeIn, s_dots, lag_ratio=0.03), run_time=1.0)
            vo.wait_until(at(vo, 2, "discovery rose"))
            self.play(FadeIn(a_cnt, shift=0.1 * LEFT), run_time=0.5)
            self.play(FadeIn(s_cnt, shift=0.1 * LEFT), run_time=0.5)
            self.play(LaggedStartMap(ShowCreation, conn, lag_ratio=0.05), FadeIn(disc), FadeIn(lg_conn), run_time=1.3)
            vo.wait_until(at(vo, 2, "and twenty-three"))
            self.play(LaggedStartMap(FadeIn, a_tri, lag_ratio=0.05), FadeIn(lg_tri[0]), run_time=0.8)
            self.play(LaggedStartMap(FadeIn, s_tri, lag_ratio=0.03), FadeIn(lg_tri[1]), run_time=1.0)
        self.strips = dict(seed_labs=seed_labs, a_lab=a_lab, s_lab=s_lab, a_dots=a_dots, s_dots=s_dots, a_cnt=a_cnt,
                           s_cnt=s_cnt, conn=conn, disc=disc, a_tri=a_tri, s_tri=s_tri, lg=lg, lg_tri=lg_tri, src=src,
                           lg_conn=lg_conn)

    # ================================================================== N12.4 the hinge
    def part_hinge(self):
        st = self.strips
        a_dots, s_dots = st["a_dots"], st["s_dots"]
        yh = -1.7
        pos_idx = [i for i in range(40) if SLOW_CLASS[i] == "P"]
        fail_lab = VGroup(L("failures: 11 of 16", 20, BAD), L("POSITION splits", 20, BAD)).arrange(
            DOWN, buff=0.06, aligned_edge=LEFT)
        fail_lab.move_to([-6.6, -0.72, 0], aligned_edge=LEFT)
        lg_pos = VGroup(Dot(radius=0.075).set_fill(BAD, 1), L("POSITION failure", 20, MUTED)).arrange(RIGHT, buff=0.12)
        lg_pos.next_to(st["lg"], RIGHT, buff=0.5)

        chips = VGroup(*[node_box(t, color=c, size=22, text_color=c) for t, c in (
            ("gate reset (S11)", MUTED), ("position penalty (S12)", MUTED), ("hinge (S13)", HINGE_COLOR))])
        chips.arrange(RIGHT, buff=0.35).move_to([0.7, yh, 0])
        h_lab = row_label("hinge", "screen S13", yh, HINGE_COLOR)
        h_dots = seed_strip(HINGE_CLASS, yh)
        h_cnt = L("34/40", 26, GOOD, weight="BOLD").move_to([COUNT_X, yh, 0], aligned_edge=LEFT)
        rescued = [i for i in range(40) if HINGE_DISC[i] == "1" and SLOW_DISC[i] == "0"]
        h_conn = VGroup(*[Line(s_dots[i].get_bottom(), h_dots[i].get_top()).set_stroke(HINGE_COLOR, 2.5)
                          for i in rescued])
        disc2 = VGroup(L("10", 24, HINGE_COLOR, weight="BOLD"), L("vs", 20, MUTED), L("0", 24, BAD, weight="BOLD"))
        disc2.arrange(RIGHT, buff=0.12).move_to([COUNT_X, (0.0 + yh) / 2, 0], aligned_edge=LEFT)
        lg_conn2 = conn_legend(("only the hinge discovered", HINGE_COLOR), ("only slow memory discovered", BAD))
        src = source_note("Report §6, Table 3; per seed: screens S5, S7, S13")

        with self.voiceover(
            "With slow memory, the failures that remained were mostly position splits, eleven of sixteen. So later "
            "screens went after exactly those, and the one carried forward was a hinge penalty. On the same seeds "
            "as slow memory alone, it discovered on ten seeds where slow memory did not, and on none where slow "
            "memory won."
        ) as vo:
            self.play(FadeOut(st["conn"]), FadeOut(st["disc"]), FadeOut(st["a_tri"]), FadeOut(st["s_tri"]),
                      FadeOut(st["lg_tri"]), FadeOut(st["lg_conn"]), self.retitle("The hinge"),
                      VGroup(a_dots, st["a_lab"], st["a_cnt"]).animate.set_opacity(0.3), run_time=0.8)
            vo.wait_until(at(vo, 0, "mostly position"))
            self.play(*[s_dots[i].animate.set_fill(BAD, 1).set_stroke(BAD, 1.5) for i in pos_idx],
                      FadeIn(lg_pos), run_time=1.0)
            self.play(FadeIn(fail_lab, shift=0.1 * RIGHT),
                      LaggedStart(*[Indicate(s_dots[i], color=BAD, scale_factor=1.6) for i in pos_idx], lag_ratio=0.08),
                      run_time=1.4)
            vo.wait_until(at(vo, 1, "went after"))
            self.play(LaggedStartMap(FadeIn, chips, shift=0.15 * UP, lag_ratio=0.3), run_time=1.1)
            vo.wait_until(at(vo, 1, "the one carried"))
            self.play(FadeOut(chips[:2], shift=0.15 * DOWN), FadeTransform(chips[2], h_lab), run_time=1.0)
            self.play(LaggedStartMap(FadeIn, h_dots, lag_ratio=0.03), FadeIn(h_cnt, shift=0.1 * LEFT),
                      FadeTransform(st["src"], src), run_time=1.1)
            vo.wait_until_sentence(2)
            self.play(LaggedStartMap(ShowCreation, h_conn, lag_ratio=0.08), FadeIn(lg_conn2), run_time=1.8)
            self.play(LaggedStart(*[Indicate(h_dots[i], color=HINGE_COLOR, scale_factor=1.6) for i in rescued],
                                  lag_ratio=0.1), FadeIn(disc2[0]), FadeIn(disc2[1]), run_time=1.6)
            vo.wait_until(at(vo, 2, "and on none"))
            self.play(FadeIn(disc2[2], scale=1.5), run_time=0.6)
            self.play(FlashAround(disc2[2], color=BAD), run_time=1.0)
        self.hinge_row = dict(h_lab=h_lab, h_dots=h_dots, h_cnt=h_cnt, src=src)

    # ================================================================== N12.5 the hinge acts rarely
    def part_rare(self):
        hr = self.hinge_row
        h_lab, h_dots, h_cnt = hr["h_lab"], hr["h_dots"], hr["h_cnt"]
        keep = {h_lab, h_dots, self.title, hr["src"]}
        others = [m for m in self.mobjects if m not in keep and m is not self.frame]
        top_y = 2.0
        shift = (top_y - h_dots[0].get_y()) * UP

        never = [i for i in range(40) if HINGE_FIRED[i] == 0]
        fired = [i for i in range(40) if HINGE_FIRED[i] > 0]
        gap = 0.6
        new_x = {}
        for j, i in enumerate(never):
            new_x[i] = X0 + j * DX
        for j, i in enumerate(fired):
            new_x[i] = X0 + (20 + j) * DX + gap
        rings = VGroup(*[Circle(radius=0.13).set_stroke(HINGE_COLOR, 2).move_to([new_x[i], top_y, 0]) for i in fired])
        never_lab = L("never fired: 20 of 40, identical to slow memory", 20, MUTED).move_to([X0 + 9.5 * DX, 2.45, 0])
        fired_lab = L("fired: 20", 20, HINGE_COLOR).move_to([X0 + 29.5 * DX + gap, 2.45, 0])

        # timeline of the nine runs that fired exactly once and bound
        tx0, tx1 = -4.4, 2.4
        ax_y = -1.4
        row_y = [1.0 - 0.26 * k for k in range(len(ONE_FIRING))]

        def TX(u, umax):
            return tx0 + (tx1 - tx0) * u / umax

        def axis_for(umax, ticks):
            line = Line([tx0, ax_y, 0], [tx1, ax_y, 0]).set_stroke(MUTED, 2)
            tk = VGroup(*[Line([TX(u, umax), ax_y - 0.07, 0], [TX(u, umax), ax_y + 0.07, 0]).set_stroke(MUTED, 2)
                          for u in ticks])
            labs = VGroup(*[L(f"{u:,}", 20, MUTED).next_to([TX(u, umax), ax_y, 0], DOWN, buff=0.12) for u in ticks])
            return VGroup(line, tk, labs)

        ax_full = axis_for(6000, [0, 1200, 2400, 3600, 4800, 6000])
        ax_zoom = axis_for(250, [0, 50, 100, 150, 200, 250])
        ax_name = L("training batches (updates)", 20, MUTED).move_to([(tx0 + tx1) / 2, ax_y - 0.68, 0])
        ax_name_z = L("zoomed in: the first 250 updates", 20, HINGE_COLOR).move_to(ax_name)
        labels, lines, ticks, lines_z, ticks_z = VGroup(), VGroup(), VGroup(), VGroup(), VGroup()
        for k, (seed, u) in enumerate(ONE_FIRING):
            y = row_y[k]
            labels.add(L(f"seed {seed}", 20, MUTED).move_to([tx0 - 0.2, y, 0], aligned_edge=RIGHT))
            stop = HINGE_STOP[seed - 160]
            lines.add(Line([tx0, y, 0], [TX(stop, 6000), y, 0]).set_stroke(FAINT, 2.5))
            lines_z.add(Line([tx0, y, 0], [tx1, y, 0]).set_stroke(FAINT, 2.5))
            ticks.add(Line([TX(u, 6000), y - 0.1, 0], [TX(u, 6000), y + 0.1, 0]).set_stroke(HINGE_COLOR, 4))
            uz = min(u, 250)
            ticks_z.add(Line([TX(uz, 250), y - 0.1, 0], [TX(uz, 250), y + 0.1, 0]).set_stroke(HINGE_COLOR, 4))
        late_lab = L("→ 3,572", 20, HINGE_COLOR).next_to(lines_z[-1], RIGHT, buff=0.12)
        run_end = L("each run ends bound (≈ 5,000 batches)", 20, MUTED).move_to([tx0, 1.42, 0], aligned_edge=LEFT)

        summ = VGroup(L("runs that fired and bound: 14", 20, INK), L("fired once: 9 (shown)", 20, HINGE_COLOR),
                      L("twice: 2,  three times: 3", 20, HINGE_COLOR), L("the 6 that fired and failed:", 20, MUTED),
                      L("2 to 1,938 times", 20, MUTED)).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        summ[3].shift(0.15 * DOWN)
        summ[4].shift(0.15 * DOWN)
        summ.move_to([3.35, 0.15, 0], aligned_edge=LEFT)

        legend = VGroup(Line(0.1 * DOWN, 0.1 * UP).set_stroke(HINGE_COLOR, 4),
                        L("hinge firing", 20, HINGE_COLOR)).arrange(RIGHT, buff=0.12)
        legend.move_to([tx1, 1.42, 0], aligned_edge=RIGHT)
        legend_r = VGroup(Line(0.1 * DOWN, 0.1 * UP).set_stroke(KEY_COLOR, 4),
                          L("random push, same size, same batch", 20, KEY_COLOR)).arrange(RIGHT, buff=0.12)
        legend_r.move_to(legend, aligned_edge=RIGHT)
        legend_u = VGroup(Line(0.1 * DOWN, 0.1 * UP).set_stroke(WARN, 4),
                          L("one random push at update 120", 20, WARN)).arrange(RIGHT, buff=0.12)
        legend_u.move_to(legend, aligned_edge=RIGHT)
        v120 = DashedLine([TX(120, 250), row_y[0] + 0.2, 0], [TX(120, 250), row_y[-1] - 0.2, 0],
                          dash_length=0.06).set_stroke(WARN, 2)

        # comparison panels
        def panel(title, rows, x_left, chip=None):
            t = L(title, 20, INK).move_to([x_left, -2.42, 0], aligned_edge=LEFT)
            bars = VGroup(*[mini_bar(lab, n, d, c, y, x_left + 1.75, width=2.6) for (lab, n, d, c), y in
                            zip(rows, (-2.85, -3.2))])
            g = VGroup(t, bars)
            if chip is not None:
                chip.next_to(bars, RIGHT, buff=0.3)
                g.add(chip)
            return g

        pan_r = panel("random push on the hinge's batches (S16; the 20 seeds where it fired)",
                      [("random push", 12, 20, KEY_COLOR), ("hinge", 14, 20, HINGE_COLOR)], -6.5)
        chip_u = VGroup(L("0", 22, BAD, weight="BOLD"), L("vs", 20, MUTED), L("7", 22, HINGE_COLOR, weight="BOLD"))
        chip_u.arrange(RIGHT, buff=0.1)
        pan_u = panel("one push at update 120 (S20)",
                      [("push at 120", 27, 40, WARN), ("hinge", 34, 40, HINGE_COLOR)], 0.45, chip=chip_u)
        for pan in (pan_r, pan_u):
            for b in pan[1]:
                b.fill.save_state()
                b.fill.stretch(1e-3, 0, about_edge=LEFT)
        src = source_note("Report §6, Table 3; per seed: screens S13, S16, S20")

        with self.voiceover(
            "And the hinge acts rarely. On twenty of forty seeds it never fired. Where it did, it usually fired on "
            "one to three of the roughly five thousand batches before binding. A random push of the same size, on "
            "the same batches, did most of what the hinge did. A single random push at a fixed update did not. "
            "Timing appears to do most of the work; whether the hinge's direction adds anything was not resolved."
        ) as vo:
            self.play(*[FadeOut(m) for m in others], self.retitle("The hinge acts rarely"), run_time=0.6)
            self.play(VGroup(h_lab, h_dots).animate.shift(shift), run_time=0.7)
            vo.wait_until_sentence(1)
            self.play(*[h_dots[i].animate.move_to([new_x[i], top_y, 0]) for i in range(40)], run_time=1.0)
            self.play(FadeIn(never_lab, shift=0.1 * DOWN), FadeIn(fired_lab, shift=0.1 * DOWN),
                      LaggedStartMap(ShowCreation, rings, lag_ratio=0.03),
                      *[h_dots[i].animate.set_opacity(0.45) for i in never], run_time=1.0)
            vo.wait_until_sentence(2)
            sel = [SEEDS.index(s) for s, _ in ONE_FIRING]
            self.play(*[TransformFromCopy(h_dots[i], labels[k]) for k, i in enumerate(sel)], FadeIn(ax_full),
                      FadeIn(ax_name), FadeTransform(hr["src"], src), run_time=1.1)
            self.play(LaggedStartMap(ShowCreation, lines, lag_ratio=0.06), FadeIn(run_end), run_time=1.0)
            self.play(LaggedStartMap(GrowFromCenter, ticks, lag_ratio=0.08), FadeIn(legend), run_time=0.9)
            self.play(FadeIn(summ, shift=0.1 * LEFT), run_time=0.8)
            vo.wait_until(vo.time_of(3) - 0.9)
            self.play(Transform(ax_full, ax_zoom), Transform(lines, lines_z), Transform(ticks, ticks_z),
                      FadeTransform(ax_name, ax_name_z), FadeOut(run_end), FadeIn(late_lab), run_time=1.2)
            vo.wait_until(at(vo, 3, "same size"))
            self.play(*[t.animate.set_stroke(KEY_COLOR, 4) for t in ticks], late_lab.animate.set_color(KEY_COLOR),
                      FadeTransform(legend, legend_r), run_time=0.8)
            self.play(FadeIn(pan_r[0]), FadeIn(pan_r[1]), run_time=0.5)
            self.play(*[Restore(b.fill) for b in pan_r[1]], run_time=0.9)
            vo.wait_until_sentence(4)
            self.play(*[t.animate.move_to([TX(120, 250), t.get_y(), 0]).set_stroke(WARN, 4) for t in ticks[:-1]],
                      ticks[-1].animate.move_to([TX(120, 250), row_y[-1], 0]).set_stroke(WARN, 4),
                      FadeOut(late_lab), ShowCreation(v120), FadeTransform(legend_r, legend_u), run_time=1.0)
            self.play(FadeIn(pan_u[0]), FadeIn(pan_u[1]), *[Restore(b.fill) for b in pan_u[1]], FadeIn(chip_u),
                      run_time=0.9)
            vo.wait_until(vo.time_of(5) - 0.5)
            self.part_table(vo)

    # ------------------------------------------------------------------ Table 3 as bars
    def part_table(self, vo):
        rows = [
            # name, screen value, comparison label + value, pairs, verdict, color
            ("plain gate A (X, recorded)", (12, 40), None, "", "", ADAM_COLOR),
            ("kWTA sparse codes (S4)", (7, 20), ("A", 7, 20), "", "not", KEY_COLOR),
            ("slow memory (S5, S7)", (24, 40), ("A", 12, 40), "15 vs 3", "pooled, no verdict", SLOW_COLOR),
            ("gate reset (S11)", (10, 20), ("A", 7, 20), "3 vs 0", "inconclusive", KEY_COLOR),
            ("position penalty (S12)", (15, 20), ("A", 7, 20), "9 vs 1", "promising", KEY_COLOR),
            ("hinge (S13)", (34, 40), ("SLOW", 24, 40), "10 vs 0", "promising", HINGE_COLOR),
            ("random kick (S16)", (12, 20), ("HINGE", 14, 20), "1 vs 3", "“suffices”", KEY_COLOR),
            ("untimed kick (S20)", (27, 40), ("HINGE", 34, 40), "0 vs 7", "inconclusive", KEY_COLOR),
            ("gate noise (S26)", (27, 40), ("HINGE", 34, 40), "0 vs 7", "inconclusive", KEY_COLOR),
        ]
        x_name, x_bar, bw = -6.6, -3.15, 2.6
        x_val, x_pair, x_verd = -0.35, 3.0, 5.15
        y0, dy = 1.75, 0.5
        head = VGroup(L("screen", 20, MUTED, weight="BOLD").move_to([x_name, 2.35, 0], aligned_edge=LEFT),
                      L("discovered, beside its comparison", 20, MUTED, weight="BOLD").move_to([x_bar, 2.35, 0],
                                                                                              aligned_edge=LEFT),
                      L("pairs", 20, MUTED, weight="BOLD").move_to([x_pair, 2.35, 0]),
                      L("verdict", 20, MUTED, weight="BOLD").move_to([x_verd, 2.35, 0]))
        rule = Line([x_name, 2.08, 0], [6.6, 2.08, 0]).set_stroke(FAINT, 1.5)
        trows = VGroup()
        for k, (name, (n, d), comp, pairs, verdict, color) in enumerate(rows):
            y = y0 - k * dy
            nm = L(name, 20, HINGE_COLOR if name.startswith("hinge") else INK).move_to([x_name, y, 0], aligned_edge=LEFT)
            track = Rectangle(width=bw, height=0.2).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1)
            track.move_to([x_bar + bw / 2, y + 0.06, 0])
            fill = Rectangle(width=bw * n / d, height=0.2).set_fill(color, 0.9).set_stroke(width=0)
            fill.align_to(track, LEFT).set_y(y + 0.06)
            parts = [nm, track, fill]
            val_txt = f"{n}/{d}" + (f"  vs {comp[0]} {comp[1]}/{comp[2]}" if comp else "")
            val = L(val_txt, 20, INK).move_to([x_val, y, 0], aligned_edge=LEFT)
            parts.append(val)
            if comp:
                cbar = Rectangle(width=bw * comp[1] / comp[2], height=0.07).set_fill(MUTED, 0.9).set_stroke(width=0)
                cbar.align_to(track, LEFT).set_y(y - 0.12)
                parts.append(cbar)
            if pairs:
                parts.append(L(pairs, 20, INK).move_to([x_pair, y, 0]))
            if verdict:
                if verdict in ("promising", "inconclusive", "not"):
                    parts.append(verdict_badge(verdict, 20).move_to([x_verd, y, 0]))
                else:
                    parts.append(L(verdict, 20, MUTED).move_to([x_verd, y, 0]))
            g = VGroup(*parts)
            trows.add(g)
        foot = L("exploratory screens; 20-seed screens use seeds 160–179, S16 the 20 seeds where the hinge fired",
                 20, MUTED).move_to([x_name, y0 - 9 * dy, 0], aligned_edge=LEFT)
        hl_kick = SurroundingRectangle(trows[6], buff=0.06).set_stroke(KEY_COLOR, 1.5)
        hl_unt = SurroundingRectangle(trows[7], buff=0.06).set_stroke(WARN, 1.5)
        call_t = VGroup(L("timing:", 22, GOOD, weight="BOLD"), L("appears to do most of the work", 22, INK)).arrange(
            RIGHT, buff=0.15)
        call_d = VGroup(L("direction:", 22, MUTED, weight="BOLD"),
                        L("not resolved (1 vs 3, p = 0.63)", 22, MUTED)).arrange(RIGHT, buff=0.15)
        calls = VGroup(call_t, call_d).arrange(RIGHT, buff=0.7).move_to([0, -3.25, 0])
        calls.align_to(foot, LEFT)
        src = source_note("Report §6, Table 3")

        self.clear_but_title(run_time=0.6, extra=[self.retitle("All the screens")])
        self.play(FadeIn(head), ShowCreation(rule), FadeIn(src), run_time=0.5)
        self.play(LaggedStartMap(FadeIn, trows, shift=0.1 * RIGHT, lag_ratio=0.12), FadeIn(foot), run_time=1.6)
        vo.wait_until(at(vo, 5, "whether"))
        self.play(ShowCreation(hl_kick), ShowCreation(hl_unt), FadeIn(call_t, shift=0.1 * UP), run_time=0.8)
        self.play(FadeIn(call_d, shift=0.1 * UP), Indicate(trows[6][-2], color=INK), run_time=0.9)
        self.hinge_trow = trows[5]
