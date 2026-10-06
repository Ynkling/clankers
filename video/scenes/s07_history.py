"""Chapter 7 — Phases I to IV (report §2; README §2, §4, §5, §7, §8.3–8.4, §9 and Tables 2, 3, 5, 6, 7;
facts_repo §1.2 for the phase-to-revision map)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403

from manimlib.config import manim_config  # noqa: E402

# manimlib compiles every formula in one shared working.tex, so scenes rendered in parallel can swap each
# other's formulas. Compile this scene's TeX privately, and tag its strings so no shared entry is reused.
_TEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "latex_s07")
os.makedirs(_TEX_DIR, exist_ok=True)
manim_config.directories.latex_cache = _TEX_DIR


def M(tex: str, size: float = 40, color=INK, **kw) -> Tex:  # noqa: F811  (shadows common.style.M)
    return Tex(tex + "{}", font_size=size, fill_color=color, **kw)


B0, Y1 = STREAM_COLORS[0], STREAM_COLORS[1]

# ---------------------------------------------------------------- timeline (README §2; report §2, Revision 7 note)
PHASES = [("Phase I", "Rev 1–3"), ("Phase II", "Rev 4"), ("Phase III", "Rev 5"), ("Phase IV", "Rev 6"),
          ("Phase V", "Rev 7")]
TL_X = [-0.85, 0.9, 2.65, 4.4, 6.1]
TL_Y = 3.17

# ---------------------------------------------------------------- Phase IV: the ten tests (README Table 1)
TESTS = ["readout_path", "router_reliability", "router_layout", "short_conv", "conv_lr", "p_scaling",
         "load_curriculum", "curriculum_confirm", "scale_axes", "stream_channels"]

PX = -2.45          # left edge of the Phase IV evidence panel
BAR_LW, BAR_W = 3.2, 2.5


def build_timeline() -> VGroup:
    line = Line([TL_X[0] - 0.55, TL_Y, 0], [TL_X[-1] + 0.45, TL_Y, 0]).set_stroke(FAINT, 3)
    nodes, names, revs = VGroup(), VGroup(), VGroup()
    for x, (nm, rv) in zip(TL_X, PHASES):
        d = Circle(radius=0.11).move_to([x, TL_Y, 0]).set_fill(PANEL, 1).set_stroke(MUTED, 2)
        n = L(nm, 22, MUTED, weight="BOLD").next_to(d, DOWN, buff=0.13)
        r = L(rv, 20, FAINT).next_to(n, DOWN, buff=0.07)
        nodes.add(d)
        names.add(n)
        revs.add(r)
    tl = VGroup(line, nodes, names, revs)
    tl.line, tl.nodes, tl.names, tl.revs = line, nodes, names, revs
    return tl


def xmark(size=0.26, color=BAD, width=5) -> VGroup:
    return VGroup(Line(UL, DR), Line(DL, UR)).set_stroke(color, width).set_width(size)


def checkmark(size=0.32, color=GOOD, width=5) -> VMobject:
    m = VMobject().set_points_as_corners([[-0.5, 0.05, 0], [-0.12, -0.35, 0], [0.55, 0.45, 0]])
    return m.set_stroke(color, width).set_width(size)


def star(radius=0.24, color=WARN) -> Polygon:
    pts = []
    for i in range(10):
        r = radius if i % 2 == 0 else radius * 0.45
        a = PI / 2 + i * PI / 5
        pts.append([r * np.cos(a), r * np.sin(a), 0])
    return Polygon(*pts).set_fill(color, 1).set_stroke(color, 1)


def ellipse_exit(center, a, b, start, target):
    """Point where the ray start→target leaves the ellipse (center, semi-axes a, b)."""
    d = np.array(target) - np.array(start)
    ox, oy = start[0] - center[0], start[1] - center[1]
    qa = (d[0] / a) ** 2 + (d[1] / b) ** 2
    qb = 2 * (ox * d[0] / a ** 2 + oy * d[1] / b ** 2)
    qc = (ox / a) ** 2 + (oy / b) ** 2 - 1
    t = (-qb + np.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa)
    return np.array(start) + t * d


def curved_arrow(a, b, angle, color, tip=0.18, width=3) -> VMobject:
    """A curved arrow with a tip sized for short arcs (CurvedArrow's default tip is 0.35)."""
    arc = ArcBetweenPoints(np.array(a), np.array(b), angle=angle)
    arc.add_tip(width=tip, length=tip)
    arc.set_stroke(color, width)
    arc.tip.set_fill(color, 1)
    return arc


def machine_split(x: str, l: str, size=22) -> VGroup:
    g = VGroup(L("X", size, MACHINE_COLORS["X"], weight="BOLD"), L(x, size, MUTED),
               L("L", size, MACHINE_COLORS["L"], weight="BOLD"), L(l, size, MUTED))
    g.arrange(RIGHT, buff=0.08)
    g[2:].shift(0.22 * RIGHT)
    return g


def pbar(label, num, den, color, y, split=None, x_left=PX, label_width=BAR_LW, width=BAR_W) -> VGroup:
    """A frac_bar whose label starts at x_left and whose track is centred on y; optional X/L split below."""
    fb = frac_bar(label, num, den, color=color, width=width, label_width=label_width, size=22)
    if fb.name_mob.get_width() > label_width - 0.1:
        fb.name_mob.set_width(label_width - 0.1, about_edge=LEFT)
    fb.shift((x_left - fb.name_mob.get_left()[0]) * RIGHT + (y - fb.track.get_y()) * UP)
    fb.msplit = None
    if split:
        sp = machine_split(*split).next_to(fb.track, DOWN, buff=0.08, aligned_edge=LEFT)
        fb.add(sp)
        fb.msplit = sp
    return fb


def show_bar(fb, run_time=0.9) -> list:
    """Name and track fade in while the fill grows; the split fades in with the value."""
    anims = [FadeIn(fb.name_mob, run_time=0.5), FadeIn(fb.track, run_time=0.5), *grow_bar(fb, run_time=run_time)]
    if fb.msplit is not None:
        anims.append(FadeIn(fb.msplit, run_time=run_time))
    return anims


def grow_only(fb, run_time=0.9) -> list:
    """For a bar whose name and track are already on screen: grow the fill, show value and split."""
    anims = [*grow_bar(fb, run_time=run_time)]
    if fb.msplit is not None:
        anims.append(FadeIn(fb.msplit, run_time=run_time))
    return anims


def morph_bar(old, new) -> list:
    anims = [FadeTransform(old.name_mob, new.name_mob), ReplacementTransform(old.track, new.track),
             ReplacementTransform(old.fill, new.fill), FadeTransform(old.value, new.value)]
    if old.msplit is not None and new.msplit is not None:
        anims.append(FadeTransform(old.msplit, new.msplit))
    elif old.msplit is not None:
        anims.append(FadeOut(old.msplit))
    elif new.msplit is not None:
        anims.append(FadeIn(new.msplit))
    return anims


def bar_parts(fb) -> VGroup:
    parts = [fb.name_mob, fb.track, fb.fill, fb.value]
    if fb.msplit is not None:
        parts.append(fb.msplit)
    return VGroup(*parts)


def mini_grid(rows, cols, color, cell=0.14, fill_cells=(), fill_color=None, opacity=0.6) -> VGroup:
    g = memory_grid(rows, cols, cell=cell, color=color)
    for i in fill_cells:
        g.cells[i].set_fill(fill_color or color, opacity)
    return g


def panel_header(text, sub=None) -> VGroup:
    h = L(text, 24, INK, weight="BOLD")
    h.move_to([PX, 2.05, 0], aligned_edge=LEFT)
    g = VGroup(h)
    if sub:
        s = L(sub, 20, MUTED).next_to(h, DOWN, buff=0.12, aligned_edge=LEFT)
        g.add(s)
    return g


class History(ClankersScene):
    def at(self, vo, i, frac=0.0):
        """Wait until a fraction `frac` of the way through sentence i of the block."""
        _, a, b = vo.synth.sentences[i]
        vo.wait_until(a + frac * (b - a))

    def at_phrase(self, vo, i, phrase, offset=0.0):
        """Wait until (roughly) the moment `phrase` is spoken in sentence i (by character position)."""
        text, a, b = vo.synth.sentences[i]
        k = text.find(phrase)
        assert k >= 0, phrase
        vo.wait_until(a + (b - a) * k / len(text) + offset)

    def focus_phase(self, i, *extra, run_time=0.8):
        tl = self.tl
        anims = [self.ring.animate.move_to(tl.nodes[i])]
        for j in range(5):
            anims.append(tl.names[j].animate.set_fill(INK if j == i else MUTED))
            anims.append(tl.revs[j].animate.set_fill(MUTED if j == i else FAINT))
            fill = GATE_COLOR if j == i else (MUTED if j < i else PANEL)
            anims.append(tl.nodes[j].animate.set_fill(fill, 1).set_stroke(GATE_COLOR if j == i else MUTED, 2))
        self.play(*anims, *extra, run_time=run_time)

    def swap_source(self, text):
        new = source_note(text)
        anim = FadeTransform(self.src, new) if self.src is not None else FadeIn(new)
        self.src = new
        return anim

    def construct(self):
        self.src = None
        card = self.chapter_card(7, "Phases I to IV")
        self.wait(0.6)
        title = section_title("Four earlier phases")
        self.play(FadeOut(card, shift=0.3 * UP), FadeIn(title, shift=0.3 * UP))

        self.n71_timeline()
        self.n72_phase_one_two()
        self.n73_phase_three()
        self.n74_phase_four_a()
        self.n75_phase_four_b()
        self.wait(1.2)
        self.clear_all()

    # ------------------------------------------------------------------------------------------ N7.1
    def n71_timeline(self):
        tl = build_timeline()
        self.tl = tl
        tl.save_state()
        tl.scale(1.35).move_to(0.1 * DOWN)
        self.ring = Circle(radius=0.2).set_stroke(GATE_COLOR, 3).move_to(tl.saved_state[1][0])
        for n in tl.nodes:
            n.set_stroke(MUTED, 2)
        tag = L("this report", 28, GATE_COLOR, weight="BOLD").next_to(tl.revs[4], DOWN, buff=0.3)
        early = VGroup(tl.nodes[:4], tl.names[:4], tl.revs[:4])
        brace = Brace(VGroup(tl.revs[0], tl.revs[3]), DOWN, buff=0.25).set_fill(MUTED)
        brace_lab = L("four earlier phases", 28, MUTED).next_to(brace, DOWN, buff=0.15)

        with self.voiceover("This report is Revision Seven, and it stands on four earlier phases.") as vo:
            self.play(ShowCreation(tl.line), run_time=0.7)
            tl.nodes[4].set_fill(GATE_COLOR, 1).set_stroke(GATE_COLOR, 2)
            tl.names[4].set_fill(INK)
            self.play(FadeIn(VGroup(tl.nodes[4], tl.names[4], tl.revs[4]), shift=0.15 * DOWN), run_time=0.6)
            self.play(FadeIn(tag, shift=0.15 * UP), self.swap_source("Report §2; README §2"), run_time=0.5)
            self.at_phrase(vo, 0, "and it stands")
            self.play(LaggedStart(*[FadeIn(VGroup(tl.nodes[j], tl.names[j], tl.revs[j]), shift=0.15 * DOWN)
                                    for j in range(4)], lag_ratio=0.25), run_time=1.3)
            self.play(GrowFromCenter(brace), FadeIn(brace_lab, shift=0.1 * DOWN), run_time=0.6)
        self.remove(*tl.get_family())
        self.add(tl)
        self.n71_extra = VGroup(tag, brace, brace_lab)
        self.early = early

    # ------------------------------------------------------------------------------------------ N7.2
    def n72_phase_one_two(self):
        tl = self.tl
        # ---- left: the gate at a key position, in two streams
        cx = -3.2
        lab1 = L("Phase I gate: a function of the token alone", 24, GATE_COLOR).move_to([cx, 2.1, 0])
        f1 = M(r"g_t = \mathrm{softmax}(W_{\mathrm{tok}}\, v_t)", 38).move_to([cx, 1.6, 0])
        heads = VGroup(L("token", 20, MUTED).move_to([-5.35, 0.7, 0]),
                       L("gate gives", 20, MUTED).move_to([-2.75, 0.7, 0]),
                       L("routing needs", 20, MUTED).move_to([-0.55, 0.7, 0]))
        rows_y = [-0.05, -1.05]
        toks, gives, needs, arrows, marks = VGroup(), VGroup(), VGroup(), VGroup(), VGroup()
        for s, y in enumerate(rows_y):
            t = VGroup(token(f"CTX{s}", "ctx", s), token("K2")).arrange(RIGHT, buff=0.06).move_to([-5.35, y, 0])
            g = gate_bar([0.62, 0.38], width=1.5, height=0.32).move_to([-2.75, y, 0])
            nd = gate_bar([1.0, 0.0] if s == 0 else [0.0, 1.0], width=1.5, height=0.32).move_to([-0.55, y, 0])
            a = Arrow(t[1].get_right(), g.get_left(), buff=0.12).set_color(MUTED)
            mk = xmark().move_to([-1.65, y, 0])
            toks.add(t)
            gives.add(g)
            needs.add(nd)
            arrows.add(a)
            marks.add(mk)
        legend = VGroup()
        for c, col in enumerate((B0, Y1)):
            chip = Square(0.2).set_fill(col, 0.9).set_stroke(width=0)
            legend.add(VGroup(chip, L(f"channel {c}", 20, MUTED)).arrange(RIGHT, buff=0.12))
        legend.arrange(RIGHT, buff=0.5).move_to([-1.65, -1.62, 0])
        same = L("same token, same gate", 20, BAD).move_to([-2.75, -0.55, 0])
        question = T("Which stream is K2 in? Not a property of K2.", 26, WARN).move_to([cx, -2.2, 0])

        # ---- right: what each gate can express
        hdr = L("what each gate can express", 22, MUTED).move_to([3.85, 2.05, 0])
        sc, sa, sb = np.array([3.0, -0.15, 0]), 1.2, 0.8
        small = Ellipse(width=2 * sa, height=2 * sb).move_to(sc).set_fill(PANEL, 1).set_stroke(MUTED, 2)
        small_lab = L("token gates", 22, INK).move_to(sc + 0.47 * UP)
        dots = VGroup(*[Dot(sc + np.array(p), radius=0.05, fill_color=MUTED)
                        for p in [(-0.55, -0.25, 0), (0.15, -0.4, 0), (0.6, -0.1, 0), (-0.2, 0.05, 0)]])
        tgt_p = np.array([5.2, 0.5, 0])
        tgt = star(0.26, WARN).move_to(tgt_p)
        tgt_lab = VGroup(L("routing", 22, WARN), L("by stream", 22, WARN)).arrange(DOWN, buff=0.06)
        tgt_lab.next_to(tgt, DOWN, buff=0.15)
        start = sc + np.array([-0.55, -0.25, 0])
        edge = ellipse_exit(sc, sa, sb, start, tgt_p)
        train = Arrow(start, edge, buff=0.02).set_color(BAD)
        stall = xmark(0.24).move_to(edge + 0.04 * RIGHT)
        fail_lab = L("a failure to learn", 20, BAD).next_to(small, DOWN, buff=0.15)
        bc, ba, bb = np.array([3.85, 0.2, 0]), 2.65, 1.62
        big = Ellipse(width=2 * ba, height=2 * bb).move_to(bc).set_fill(GATE_COLOR, 0.07).set_stroke(GATE_COLOR, 2.5)
        big_lab = L("recurrent gates", 22, GATE_COLOR, weight="BOLD").move_to(bc + 1.22 * UP)
        reach = Arrow(start, tgt_p, buff=0.3).set_color(GOOD)

        lesson = VGroup(T("Lesson: check that a target is in the model's function class", 28),
                        T("before reading anything into a failure to learn it.", 28)).arrange(DOWN, buff=0.12)
        lesson_card = card(lesson, buff=0.22, edge=GATE_COLOR).move_to([0, -3.08, 0])

        # ---- Phase II versions
        lab2 = L("Phase II gate: a recurrent state carries the context", 24, GATE_COLOR).move_to([cx, 2.1, 0])
        f2 = M(r"g_t = \mathrm{softmax}(W_g\, h_t)", 38).move_to([cx, 1.6, 0])
        f2b = M(r"h_t = \tanh(W_{\mathrm{in}}\, v_t + W_h\, h_{t-1})", 30, MUTED).move_to([cx, 1.1, 0])
        heads2 = L("token + state", 20, MUTED).move_to(heads[0])
        carry = VGroup(*[curved_arrow(t[0].get_top() + 0.02 * UP, t[1].get_top() + 0.02 * UP, -PI / 2.5,
                                      STREAM_COLORS[s]) for s, t in enumerate(toks)])
        gives2 = VGroup(*[gate_bar(p, width=1.5, height=0.32).move_to(g)
                          for p, g in zip(([0.95, 0.05], [0.05, 0.95]), gives)])
        checks = VGroup(*[checkmark().move_to(m) for m in marks])
        result = T("Phase II: streams separated on 10 of 10 seeds", 26, GOOD).move_to(question)

        with self.voiceover(
            "In Phase One the gate was a function of token identity alone, and it could not express the routing, "
            "because which stream a key belongs to is not a property of the key. The lesson: check that a target is "
            "in the model's function class before reading anything into a failure to learn it. In Phase Two a "
            "recurrent gate could express it, and it separated two streams."
        ) as vo:
            self.play(Restore(tl), FadeOut(self.n71_extra), FadeIn(self.ring), run_time=1.0)
            self.focus_phase(0, FadeIn(lab1, shift=0.1 * DOWN), FadeIn(f1, shift=0.1 * DOWN), run_time=0.7)
            self.play(LaggedStartMap(FadeIn, toks, shift=0.15 * RIGHT, lag_ratio=0.3), FadeIn(heads[0]), run_time=0.8)
            self.at_phrase(vo, 0, "token identity alone")
            self.play(*[Indicate(t[1], color=KEY_COLOR, scale_factor=1.15) for t in toks],
                      *[GrowArrow(a) for a in arrows], run_time=0.8)
            self.play(*[FadeIn(g, shift=0.2 * RIGHT) for g in gives], FadeIn(heads[1]), FadeIn(same), run_time=0.8)
            self.at_phrase(vo, 0, "could not express the routing")
            self.play(*[FadeIn(n, shift=0.2 * RIGHT) for n in needs], FadeIn(heads[2]), FadeIn(legend), run_time=0.8)
            self.play(LaggedStartMap(ShowCreation, marks, lag_ratio=0.3), run_time=0.7)
            self.at_phrase(vo, 0, "because which stream")
            self.play(Write(question), run_time=1.3)
            self.at_phrase(vo, 0, "is not a property")
            self.play(*[FlashAround(t[1], color=WARN) for t in toks], run_time=1.0)
            # ---- sentence 1: the lesson, as a picture of function classes
            self.at(vo, 1)
            self.play(FadeIn(hdr), DrawBorderThenFill(small), FadeIn(small_lab), FadeIn(dots), run_time=0.9)
            self.play(GrowFromCenter(tgt), FadeIn(tgt_lab, shift=0.1 * UP), run_time=0.6)
            self.play(GrowArrow(train), run_time=0.7)
            self.play(ShowCreation(stall), FadeIn(fail_lab), run_time=0.5)
            self.play(FadeIn(lesson_card, shift=0.2 * UP), run_time=0.8)
            self.play(Indicate(tgt, color=WARN, scale_factor=1.3), run_time=0.8)
            # ---- sentence 2: Phase II, a recurrent gate
            self.at(vo, 2)
            self.focus_phase(1, FadeTransform(lab1, lab2), TransformMatchingTex(f1, f2), FadeIn(f2b, shift=0.2 * LEFT),
                             FadeTransform(heads[0], heads2), run_time=0.9)
            self.play(*[ShowCreation(c) for c in carry], run_time=0.6)
            self.play(*[Transform(g, g2) for g, g2 in zip(gives, gives2)],
                      *[ReplacementTransform(m, c) for m, c in zip(marks, checks)], FadeOut(same),
                      GrowFromCenter(big), FadeIn(big_lab), run_time=0.9)
            self.bring_to_back(big)
            self.play(FadeOut(VGroup(train, stall, fail_lab)), GrowArrow(reach), tgt.animate.set_fill(GOOD).set_stroke(GOOD),
                      FadeTransform(question, result), self.swap_source("Report §2; README §2 (Phases I–II)"),
                      run_time=0.8)
        self.n72_mobs = VGroup(lab2, f2, f2b, heads2, heads[1], heads[2], toks, gives, needs, arrows, checks, legend,
                               carry, result, hdr, big, big_lab, small, small_lab, dots, tgt, tgt_lab, reach,
                               lesson_card)

    # ------------------------------------------------------------------------------------------ N7.3
    def n73_phase_three(self):
        # the binding task, as a short strip (first training sequence of seed 280, as in Ch. 2)
        tw = dict(width=0.85)
        strip = VGroup(triple(1, 0, 13, **tw), triple(0, 0, 2, **tw), triple(0, 3, 11, **tw),
                       L("…", 30, MUTED),
                       VGroup(token("CTX0", "ctx", 0, **tw), token("K0", **tw), token("?", "query", **tw))
                       .arrange(RIGHT, buff=0.06))
        strip.arrange(RIGHT, buff=0.3).move_to(1.85 * UP)
        strip_lab = L("the binding task: 2 streams, 4 keys, 27 tokens", 22, MUTED).next_to(strip, DOWN, buff=0.18)

        lx, lw, bw = -5.35, 3.3, 3.6
        ys = [0.35, -0.55, -1.45, -2.3]

        def row(label, num, den, color, y):
            return pbar(label, num, den, color, y, x_left=lx, label_width=lw, width=bw)

        r_perfect = row("perfect gate, two channels", 10, 10, GOOD, ys[0])
        r_one = row("one channel, 80 distinct runs", 0, 80, BAD, ys[1])
        r_x = row("learned gate", 19, 40, GATE_COLOR, ys[2])
        r_l = row("learned gate", 21, 40, GATE_COLOR, ys[3])
        icon2 = VGroup(mini_grid(3, 3, B0, cell=0.12, fill_cells=(0, 4, 8)),
                       mini_grid(3, 3, Y1, cell=0.12, fill_cells=(1, 3, 7)))
        icon2.arrange(RIGHT, buff=0.07).move_to([-6.08, ys[0], 0])
        mix = interpolate_color(B0, Y1, 0.5)
        icon1 = mini_grid(3, 3, MEMORY_COLOR, cell=0.12, fill_cells=(0, 1, 3, 4, 7, 8), fill_color=mix,
                          opacity=0.75)
        icon1.move_to([-6.08, ys[1], 0])
        badge_x = machine_badge("X").move_to([-6.08, ys[2], 0])
        badge_l = machine_badge("L").move_to([-6.08, ys[3], 0])
        tick = checkmark(0.3).next_to(r_perfect.value, RIGHT, buff=0.3)
        cross = xmark(0.26).next_to(r_one.value, RIGHT, buff=0.3)
        half_x = r_x.track.get_left()[0] + bw / 2
        half = DashedLine([half_x, ys[2] + 0.38, 0], [half_x, ys[3] - 0.38, 0]).set_stroke(WARN, 2.5)
        half_lab = L("half", 20, WARN, weight="BOLD").next_to(half, DOWN, buff=0.08)
        refs = VGroup(*[L("vs one-channel refs 0/10, 0/10", 22, MUTED).next_to(r.value, RIGHT, buff=0.35)
                        for r in (r_x, r_l)])
        pvals = VGroup(*[L(p, 22, GOOD).next_to(rf, DOWN, buff=0.06, aligned_edge=LEFT)
                         for p, rf in zip(("p = 0.004", "p = 0.002"), refs)])

        with self.voiceover(
            "Phase Three built the binding task. Two channels with the perfect gate bound two streams where no "
            "single-channel model did, and the learned gate found the routing on about half its runs, beating every "
            "single-channel reference on both machines."
        ) as vo:
            self.play(FadeOut(self.n72_mobs), run_time=0.6)
            self.focus_phase(2, LaggedStartMap(FadeIn, strip, shift=0.15 * RIGHT, lag_ratio=0.15),
                             FadeIn(strip_lab), self.swap_source("Report §2, §3; README §2 (Phase III; C1 on X and L)"),
                             run_time=1.2)
            self.at(vo, 1)
            self.play(FadeIn(icon2), *show_bar(r_perfect))
            self.play(ShowCreation(tick), run_time=0.4)
            self.at_phrase(vo, 1, "where no single-channel")
            self.play(FadeIn(icon1), *show_bar(r_one))
            self.play(ShowCreation(cross), run_time=0.4)
            self.at_phrase(vo, 1, "and the learned gate")
            self.play(FadeIn(badge_x), FadeIn(badge_l), *show_bar(r_x, 1.1), *show_bar(r_l, 1.1))
            self.at_phrase(vo, 1, "on about half")
            self.play(ShowCreation(half), FadeIn(half_lab), run_time=0.7)
            self.at_phrase(vo, 1, "beating every")
            self.play(LaggedStartMap(FadeIn, refs, shift=0.2 * LEFT, lag_ratio=0.3), run_time=0.9)
            self.play(FadeIn(pvals), run_time=0.5)
            self.at_phrase(vo, 1, "on both machines")
            self.play(Indicate(badge_x, scale_factor=1.3), Indicate(badge_l, scale_factor=1.3), run_time=0.9)
        self.n73_mobs = VGroup(strip, strip_lab, icon2, icon1, badge_x, badge_l, tick, cross, half, half_lab, refs,
                               pvals, *[bar_parts(r) for r in (r_perfect, r_one, r_x, r_l)])

    # ------------------------------------------------------------------------------------------ Phase IV list
    def build_test_list(self):
        head = L("ten pre-registered tests", 22, INK, weight="BOLD").move_to([-6.5, 2.05, 0], aligned_edge=LEFT)
        sub = VGroup(L("each run on", 20, MUTED), machine_badge("X", 20), L("and", 20, MUTED),
                     machine_badge("L", 20)).arrange(RIGHT, buff=0.12)
        sub.next_to(head, DOWN, buff=0.15, aligned_edge=LEFT)
        items = VGroup()
        for i, name in enumerate(TESTS):
            box = Square(0.13).set_stroke(MUTED, 1.5).set_fill(GATE_COLOR, 0)
            txt = Text(name, font=MONO, font_size=22, fill_color=MUTED)
            it = VGroup(box, txt).arrange(RIGHT, buff=0.15)
            it.move_to([-6.5, 1.22 - 0.4 * i, 0], aligned_edge=LEFT)
            items.add(it)
        sep = Line([-2.8, 2.25, 0], [-2.8, -3.3, 0]).set_stroke(FAINT, 1.5)
        self.test_items = items
        self.test_hl = VGroup()
        return VGroup(head, sub, items, sep)

    def focus_tests(self, idx) -> list:
        anims = []
        new_hl = VGroup()
        for i, it in enumerate(self.test_items):
            on = i in idx
            anims.append(it[1].animate.set_fill(INK if on else MUTED))
            anims.append(it[0].animate.set_fill(GATE_COLOR, 1 if on else 0).set_stroke(GATE_COLOR if on else MUTED))
            if on:
                r = SurroundingRectangle(it, buff=0.07).set_fill(GATE_COLOR, 0.14).set_stroke(GATE_COLOR, 1.2)
                new_hl.add(r)
        if len(self.test_hl):
            anims.append(FadeOut(self.test_hl))
        if len(new_hl):
            anims.append(FadeIn(new_hl))
        self.test_hl = new_hl
        return anims

    # ------------------------------------------------------------------------------------------ N7.4
    def n74_phase_four_a(self):
        tests = self.build_test_list()

        # F1: the readout (README §4, Table 2; discovered of 30 per machine)
        h1 = panel_header("a readout from the gate's state into the residual stream", "two streams, four keys · discovered")
        ys = [0.95, 0.05, -0.85]
        f1 = [pbar("no readout", 41, 60, GATE_COLOR, ys[0], ("22/30", "19/30")),
              pbar("readout", 2, 60, BAD, ys[1], ("2/30", "0/30")),
              pbar("readout, gradient stopped", 44, 60, GOOD, ys[2], ("22/30", "22/30"))]
        harm = verdict_badge("HARMS", 20, color=BAD).next_to(f1[1].value, RIGHT, buff=0.3)
        grad = verdict_badge("GRADIENT", 20, color=GOOD).next_to(f1[2].value, RIGHT, buff=0.3)
        grad_note = L("stop the gradient into the gate, and the rate comes back", 20, GOOD)
        grad_note.move_to([PX, -1.75, 0], aligned_edge=LEFT)

        # F2: more channels and restarts (README §5, Table 3)
        h2 = panel_header("two streams: more channels, or restarts?", "discovered · restart: trials succeeded")
        f2 = [pbar("k = 2 channels", 43, 80, GATE_COLOR, ys[0], ("21/40", "22/40")),
              pbar("k = 4 channels", 47, 80, GATE_COLOR, ys[1], ("23/40", "24/40")),
              pbar("k = 8 channels", 53, 80, GATE_COLOR, ys[2], ("27/40", "26/40"))]
        brace = Brace(VGroup(*[f.value for f in f2]), RIGHT, buff=0.15).set_fill(MUTED)
        ns = verdict_badge("NOT SHOWN", 20).next_to(brace, RIGHT, buff=0.12)
        restart = pbar("restart rule, k = 2", 60, 60, GOOD, -1.95, ("30/30", "30/30"))
        rel = verdict_badge("RELIABLE", 20).next_to(restart.value, RIGHT, buff=0.3)
        rule = VGroup(L("at step 2400 it reads held-out accuracy, nothing else:", 22, MUTED),
                      L("≥ 0.6: continue  ·  else: restart from a fresh seed (up to 5 attempts)", 22, MUTED))
        rule.arrange(DOWN, buff=0.08, aligned_edge=LEFT).move_to([PX, -2.95, 0], aligned_edge=LEFT)

        # F3/F4: the short convolution (README §7, Table 5; bound)
        h3 = panel_header("the short convolution", "two streams, four keys · bound")
        lr1 = verdict_badge("lr 10⁻³, the working rate", 22, color=GATE_COLOR).move_to([PX, 1.3, 0], aligned_edge=LEFT)
        lr4 = verdict_badge("lr 4 × 10⁻³", 22, color=WARN).move_to(lr1, aligned_edge=LEFT)
        f3 = [pbar("one channel + conv.", 7, 80, BAD, 0.5, ("2/40", "5/40")),
              pbar("gate + conv.", 51, 80, GATE_COLOR, -0.4, ("23/40", "28/40"))]
        f4 = pbar("one channel + conv.", 34, 80, WARN, 0.5, ("18/40", "16/40"))
        gx = f3[0].track.get_left()[0] + BAR_W * 7 / 80
        ghost = DashedLine([gx, 0.27, 0], [gx, 0.75, 0], dash_length=0.06).set_stroke(INK, 2)
        ghost_lab = L("7/80 at 10⁻³", 20, MUTED).next_to(ghost, UP, buff=0.04)
        lag_tok = VGroup(token("CTX0", "ctx", 0), token("K2"), token("V5", "val", 0)).arrange(RIGHT, buff=0.06)
        lag_tok.move_to([PX, -1.75, 0], aligned_edge=LEFT)
        lag_arc = curved_arrow(lag_tok[0].get_top() + 0.03 * UP, lag_tok[2].get_top() + 0.03 * UP, -PI / 2.2, WARN,
                               tip=0.22)
        lag_lab = L("lag 2", 22, WARN, weight="BOLD").next_to(lag_arc, UP, buff=0.05)
        lag_txt = VGroup(L("the lag-2 weight writes the context", 22, INK),
                         L("into the value's position, so one channel", 22, INK),
                         L("binds without separating the streams", 22, INK)).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        lag_txt.next_to(lag_tok, RIGHT, buff=0.45).shift(0.1 * UP)
        lag_num = L("median lag-2 weight on X: one-channel binders 0.158, non-binders 0.082", 22, MUTED)
        lag_num.move_to([PX, -2.85, 0], aligned_edge=LEFT)

        with self.voiceover(
            "Phase Four ran ten pre-registered tests. A readout from the gate's state into the residual stream harmed "
            "the gate, through its gradient. At two streams, more channels did not raise the gate's rate, but a restart "
            "rule that reads only held-out accuracy at step twenty-four hundred succeeded in sixty of sixty trials. The "
            "short convolution did not replace the gate at the working learning rate: one channel bound seven of eighty "
            "runs, the gate fifty-one. But at four times that rate, one channel with the convolution bound thirty-four "
            "of eighty, through its lag-two weight."
        ) as vo:
            self.play(FadeOut(self.n73_mobs), run_time=0.5)
            self.focus_phase(3, FadeIn(tests[0]), FadeIn(tests[1]), ShowCreation(tests[3]), run_time=0.7)
            self.play(LaggedStartMap(FadeIn, tests[2], shift=0.15 * RIGHT, lag_ratio=0.12), run_time=1.2)
            # ---- the readout
            self.at(vo, 1)
            self.play(*self.focus_tests([0]), FadeIn(h1), self.swap_source("README §4, Table 2 (readout_path)"),
                      run_time=0.6)
            self.play(*show_bar(f1[0], 0.7), *show_bar(f1[1], 0.7), run_time=0.8)
            self.play(FadeIn(harm, shift=0.1 * LEFT), run_time=0.5)
            self.at_phrase(vo, 1, "through its gradient", -0.3)
            self.play(*show_bar(f1[2], 0.8), run_time=0.8)
            self.play(FadeIn(grad, shift=0.1 * LEFT), FadeIn(grad_note, shift=0.1 * UP), run_time=0.5)
            # ---- more channels, then restarts
            self.at(vo, 2)
            self.play(*self.focus_tests([1]), FadeTransform(h1, h2), FadeOut(harm), FadeOut(grad), FadeOut(grad_note),
                      *[a for o, n in zip(f1, f2) for a in morph_bar(o, n)],
                      self.swap_source("README §5, Table 3 (router_reliability)"), run_time=1.0)
            self.at_phrase(vo, 2, "did not raise")
            self.play(GrowFromCenter(brace), FadeIn(ns, shift=0.1 * LEFT), run_time=0.6)
            self.at_phrase(vo, 2, "but a restart rule")
            self.play(FadeIn(restart.name_mob), FadeIn(restart.track), run_time=0.5)
            self.at_phrase(vo, 2, "reads only held-out")
            self.play(FadeIn(rule[0], shift=0.1 * UP), run_time=0.6)
            self.play(FadeIn(rule[1], shift=0.1 * UP), run_time=0.6)
            self.at_phrase(vo, 2, "succeeded in sixty")
            self.play(*grow_bar(restart, 1.0), FadeIn(restart.msplit), run_time=1.0)
            self.play(FadeIn(rel, shift=0.1 * LEFT), Indicate(restart.value, color=GOOD), run_time=0.7)
            # ---- the short convolution at the working rate
            self.at(vo, 3)
            self.play(*self.focus_tests([3]), FadeTransform(h2, h3),
                      FadeOut(VGroup(*[bar_parts(f) for f in f2], brace, ns, bar_parts(restart), rel, rule)),
                      self.swap_source("README §7, Table 5 (short_conv, conv_lr)"), run_time=0.8)
            self.play(*[FadeIn(VGroup(f.name_mob, f.track), shift=0.1 * UP) for f in f3], run_time=0.6)
            self.at_phrase(vo, 3, "at the working learning rate")
            self.play(FadeIn(lr1, shift=0.1 * DOWN), run_time=0.6)
            self.at_phrase(vo, 3, "one channel bound")
            self.play(*grow_only(f3[0]))
            self.at_phrase(vo, 3, "the gate fifty-one")
            self.play(*grow_only(f3[1]))
            # ---- four times the rate
            self.at(vo, 4)
            self.play(*self.focus_tests([4]), FadeTransform(lr1, lr4), FadeOut(bar_parts(f3[1])), run_time=0.7)
            self.at_phrase(vo, 4, "one channel with the convolution")
            self.play(*morph_bar(f3[0], f4), FadeIn(ghost_lab, shift=0.1 * UP), run_time=1.1)
            self.play(ShowCreation(ghost), run_time=0.4)
            self.play(LaggedStartMap(FadeIn, lag_tok, shift=0.1 * UP, lag_ratio=0.2), run_time=0.7)
            self.at_phrase(vo, 4, "through its lag-two", -0.3)
            self.play(ShowCreation(lag_arc), FadeIn(lag_lab), FadeIn(lag_txt, shift=0.1 * LEFT), run_time=0.8)
            self.play(FadeIn(lag_num, shift=0.1 * UP), run_time=0.5)
        # let the lag-2 route sit for a moment before the next paragraph
        self.play(Indicate(lag_tok[2], color=STREAM_COLORS[0], scale_factor=1.15),
                  FlashAround(lag_arc, color=WARN), run_time=1.0)
        self.wait(0.8)
        self.tests = tests
        self.n74_mobs = VGroup(h3, lr4, bar_parts(f4), ghost, ghost_lab, lag_tok, lag_arc, lag_lab, lag_txt, lag_num)

    # ------------------------------------------------------------------------------------------ N7.5
    def n75_phase_four_b(self):
        # F5: eight keys (README §8.3–8.4, Table 6; report §2)
        h5 = panel_header("eight keys per stream, lr 10⁻³", "two streams · bound")
        g8 = pbar("gate, from scratch", 49, 70, GATE_COLOR, 0.95, ("25/35", "24/35"))
        o8 = pbar("one channel, same memory", 0, 24, BAD, -0.05, ("0/12", "0/12"))
        icon_g = VGroup(mini_grid(3, 3, B0, fill_cells=(0, 4, 5, 8)), mini_grid(3, 3, Y1, fill_cells=(1, 3, 6, 7)))
        icon_g.arrange(RIGHT, buff=0.08).move_to([5.75, g8.track.get_y(), 0])
        mix = interpolate_color(B0, Y1, 0.5)
        icon_o = mini_grid(3, 6, MEMORY_COLOR, fill_cells=(0, 1, 4, 7, 8, 9, 13, 16, 17), fill_color=mix,
                           opacity=0.75)
        icon_o.move_to([5.75, o8.track.get_y(), 0])
        icon_lab = L("memory", 20, MUTED).next_to(icon_g, UP, buff=0.15)
        mem_note = VGroup(L("same fast-weight state, 49,152 each;", 22, MUTED),
                          L("the single channel (N = 512) has 1.8 × the parameters", 22, MUTED))
        mem_note.arrange(DOWN, buff=0.08, aligned_edge=LEFT).move_to([PX, -1.0, 0], aligned_edge=LEFT)
        params = L("50,944 vs 28,480", 20, WARN, weight="BOLD").next_to(mem_note[1], RIGHT, buff=0.25)
        moral = T("The advantage is the partition, not the memory.", 32, INK, t2c={"partition": GOOD})
        moral.move_to([2.0, -2.25, 0])
        moral_line = Underline(moral, buff=0.08, stretch_factor=1.0).set_stroke(GOOD, 2)

        # F6: four streams (README §9, Table 7; seeds 220–239; 40 runs per row)
        h6 = panel_header("four streams, four keys, k channels", "40 runs per row, by outcome")
        legend = VGroup()
        for name, col in (("bound", GOOD), ("streams merged", BAD), ("non-stream split", MUTED)):
            chip = Square(0.22).set_fill(col, 0.9).set_stroke(width=0)
            legend.add(VGroup(chip, L(name, 20, MUTED)).arrange(RIGHT, buff=0.12))
        legend.arrange(RIGHT, buff=0.4).move_to([PX, 1.2, 0], aligned_edge=LEFT)
        col_head = L("bound per machine", 20, MUTED)
        sx0, sw, sh = -1.25, 5.3, 0.42
        col_head.move_to([sx0 + sw + 0.25, 1.2, 0], aligned_edge=LEFT)
        data = [(4, (14, 16, 10), ("9/20", "5/20"), 0.5), (8, (23, 4, 13), ("10/20", "13/20"), -0.3),
                (16, (23, 1, 16), ("10/20", "13/20"), -1.1)]
        rows = []
        for k, counts, split, y in data:
            lab = L(f"k = {k}", 22, INK).move_to([PX, y, 0], aligned_edge=LEFT)
            frame = Rectangle(width=sw, height=sh).set_stroke(PANEL_EDGE, 1).set_fill(PANEL, 1)
            frame.move_to([sx0 + sw / 2, y, 0])
            segs, nums = VGroup(), VGroup()
            x0 = sx0
            for n, col in zip(counts, (GOOD, BAD, MUTED)):
                w = sw * n / 40
                r = Rectangle(width=w, height=sh).set_fill(col, 0.9).set_stroke(BG, 1.5)
                r.move_to([x0 + w / 2, y, 0])
                segs.add(r)
                t = L(str(n), 20, BG, weight="BOLD")
                if w > 0.42:
                    t.move_to(r)
                else:
                    t.set_fill(col).next_to(r, DOWN, buff=0.1).shift(0.07 * LEFT)
                nums.add(t)
                x0 += w
            sp = machine_split(*split).next_to(frame, RIGHT, buff=0.25)
            g = VGroup(lab, frame, segs, nums, sp)
            g.lab, g.frame, g.segs, g.nums, g.msplit = lab, frame, segs, nums, sp
            rows.append(g)

        # the stream-to-channel picture: four streams, k channels
        def chan_map(targets, k, y=-2.1, x_left=PX + 0.15):
            dots = VGroup(*[Circle(radius=0.13).set_fill(STREAM_COLORS[s], 1).set_stroke(width=0)
                            for s in range(4)])
            boxes = VGroup(*[RoundedRectangle(width=0.36, height=0.33, corner_radius=0.05).set_stroke(MUTED, 1.5)
                             for _ in range(k)])
            boxes.arrange(RIGHT, buff=0.1).move_to([x_left, y - 0.34, 0], aligned_edge=LEFT)
            for s, d in enumerate(dots):
                d.move_to([boxes[s].get_x(), y + 0.4, 0])
            used = {}
            for s, c in enumerate(targets):
                used.setdefault(c, []).append(s)
            # a channel holding one stream is drawn in that stream's color; a shared channel is striped with the
            # colors of every stream in it (stream 3 is red, so red alone cannot also mean "shared")
            stripes = VGroup()
            for c, ss in used.items():
                b = boxes[c]
                if len(ss) == 1:
                    col = STREAM_COLORS[ss[0]]
                    b.set_fill(col, 0.35).set_stroke(col, 2)
                    continue
                w, h = (b.get_width() - 0.08) / len(ss), b.get_height() - 0.08
                for j, s in enumerate(ss):
                    r = Rectangle(width=w, height=h).set_fill(STREAM_COLORS[s], 0.75).set_stroke(width=0)
                    r.move_to(b.get_left() + (0.04 + (j + 0.5) * w) * RIGHT)
                    stripes.add(r)
                b.set_stroke(INK, 2.5)
            arrows = VGroup(*[Arrow(dots[s].get_bottom(), boxes[c].get_top(), buff=0.04, thickness=2.5)
                              .set_color(STREAM_COLORS[s]) for s, c in enumerate(targets)])
            return VGroup(dots, boxes, stripes, arrows)

        map_merge = chan_map([0, 1, 2, 2], 4)
        map_spare = chan_map([0, 1, 2, 5], 8)
        map_none = chan_map([0, 0, 0, 0], 8)

        def morph_map(a, b) -> list:
            """Dots, boxes and arrows move; the stripes of shared channels fade (they have no counterpart)."""
            anims = [ReplacementTransform(a[i], b[i]) for i in (0, 1, 3)]
            if len(a[2]):
                anims.append(FadeOut(a[2]))
            if len(b[2]):
                anims.append(FadeIn(b[2]))
            return anims

        def two_lines(a, b, color):
            return VGroup(L(a, 22, color), L(b, 22, color)).arrange(DOWN, buff=0.08, aligned_edge=LEFT)

        txt_merge = two_lines("a merge:", "two streams share one channel", BAD)
        txt_spare = two_lines("spare channels:", "room for every stream", GOOD)
        txt_none = two_lines("a non-stream split:", "every stream lands in one channel", INK)
        for t in (txt_merge, txt_spare, txt_none):
            t.move_to([map_spare.get_right()[0] + 0.45, -2.1, 0], aligned_edge=LEFT)
        final = T("What remained: an early commitment to a non-stream split", 30, WARN)
        final.move_to([2.0, -3.08, 0])
        if final.get_right()[0] > 6.6:
            final.shift((6.6 - final.get_right()[0]) * RIGHT)

        with self.voiceover(
            "With eight keys per stream, at the working rate, the gate bound from scratch in forty-nine of seventy "
            "runs, while one channel with the same fast-weight memory and nearly twice the parameters bound none of "
            "twenty-four. The advantage is the partition, not the memory. At four streams with four channels the gate "
            "bound only fourteen of forty, mostly failing by merging streams. Eight or sixteen channels removed most "
            "of those merges. What remained was an early commitment to a split by something other than the stream."
        ) as vo:
            self.play(FadeOut(self.n74_mobs), *self.focus_tests([7, 8]), FadeIn(h5),
                      self.swap_source("Report §2; README §8.3–8.4, Table 6"), run_time=0.8)
            self.play(FadeIn(VGroup(g8.name_mob, g8.track), shift=0.1 * UP), FadeIn(icon_g, shift=0.1 * LEFT),
                      FadeIn(icon_lab), run_time=0.6)
            self.at_phrase(vo, 0, "the gate bound from scratch", -0.2)
            self.play(*grow_only(g8), Indicate(icon_g, scale_factor=1.1))
            self.at_phrase(vo, 0, "while one channel")
            self.play(FadeIn(o8.name_mob), FadeIn(o8.track), FadeIn(icon_o, shift=0.1 * LEFT), run_time=0.6)
            self.play(FadeIn(mem_note[0], shift=0.1 * UP), run_time=0.5)
            self.at_phrase(vo, 0, "and nearly twice")
            self.play(FadeIn(mem_note[1], shift=0.1 * UP), FadeIn(params, shift=0.1 * LEFT), run_time=0.7)
            self.at_phrase(vo, 0, "bound none of")
            self.play(*grow_bar(o8, 0.6), FadeIn(o8.msplit), run_time=0.6)
            self.play(Indicate(o8.value, color=BAD, scale_factor=1.3), run_time=0.7)
            # ---- the moral
            self.at(vo, 1)
            self.play(Write(moral), run_time=1.2)
            self.play(ShowCreation(moral_line), Indicate(icon_g, scale_factor=1.15), run_time=0.8)
            # ---- four streams, four channels
            self.at(vo, 2)
            self.play(FadeOut(VGroup(bar_parts(g8), bar_parts(o8), icon_g, icon_o, icon_lab, mem_note, params, moral,
                                     moral_line)),
                      *self.focus_tests([8]), FadeTransform(h5, h6),
                      self.swap_source("Report §2; README §9, Table 7 (seeds 220–239)"), run_time=0.8)
            r4 = rows[0]
            self.play(FadeIn(legend), FadeIn(col_head), FadeIn(r4.lab), FadeIn(r4.frame), run_time=0.6)
            self.at_phrase(vo, 2, "the gate bound only", 0.3)
            self.play(GrowFromEdge(r4.segs[0], LEFT), FadeIn(r4.nums[0]), FadeIn(r4.msplit), run_time=0.8)
            self.at_phrase(vo, 2, "mostly failing", -0.2)
            self.play(GrowFromEdge(r4.segs[1], LEFT), FadeIn(r4.nums[1]), FadeIn(map_merge),
                      FadeIn(txt_merge, shift=0.1 * LEFT), run_time=0.8)
            self.play(GrowFromEdge(r4.segs[2], LEFT), FadeIn(r4.nums[2]), run_time=0.5)
            self.play(Indicate(r4.segs[1], color=BAD, scale_factor=1.08),
                      Indicate(VGroup(map_merge[1][2], map_merge[2]), color=INK, scale_factor=1.3), run_time=0.8)
            # ---- spare channels remove most merges
            self.at(vo, 3)
            self.play(*self.focus_tests([9]), self.swap_source("Report §2; README §9.2, Table 7 (stream_channels)"),
                      *[TransformFromCopy(r4, r) for r in rows[1:]], run_time=1.0)
            self.play(*morph_map(map_merge, map_spare), FadeTransform(txt_merge, txt_spare),
                      *[Indicate(r.nums[1], color=BAD, scale_factor=1.5) for r in rows], run_time=1.0)
            # ---- what remained
            self.at(vo, 4)
            boxes = VGroup(*[SurroundingRectangle(r.segs[2], buff=0.035).set_stroke(WARN, 3) for r in rows])
            self.play(LaggedStartMap(ShowCreation, boxes, lag_ratio=0.25),
                      *[r.segs[2].animate.set_fill(MUTED, 1) for r in rows], run_time=0.9)
            self.play(*morph_map(map_spare, map_none), FadeTransform(txt_spare, txt_none), run_time=0.8)
            self.play(Write(final), run_time=1.3)
            self.play(Indicate(final, color=WARN, scale_factor=1.04), run_time=0.9)
