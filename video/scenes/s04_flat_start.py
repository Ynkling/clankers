"""Chapter 4 — A flat start: the uniform gate is a stationary point (report §1, eq. 3; README §1;
bdh.py; data/gate_dynamics.json)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403
import tempfile  # noqa: E402

import manimlib.utils.cache as _mcache  # noqa: E402
from diskcache import Cache  # noqa: E402
from manimlib.config import manim_config  # noqa: E402

VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ManimGL compiles every formula in one shared latex_cache/working.tex and caches the SVG on disk under
# the formula's hash, so scenes rendered in parallel can swap each other's formulas (seen in the first
# render of this scene: "< 10^{-5}" came back as another chapter's formula). This scene therefore
# compiles in a private directory and keeps its own SVG cache; nothing outside this process changes.
_PRIVATE_TEX = os.path.join(tempfile.gettempdir(), "clankers_tex_s04")
_WORK = os.path.join(_PRIVATE_TEX, f"work_{os.getpid()}")
os.makedirs(_WORK, exist_ok=True)
manim_config.directories.latex_cache = _WORK
_mcache._cache = Cache(os.path.join(_PRIVATE_TEX, "svg"), size_limit=int(2e8))

# ---------------------------------------------------------------- geometry of the left column
C = np.array([-4.1, 0.37, 0.0])            # centroid of the k = 3 simplex (the uniform gate)
R = 3.6 / np.sqrt(3)                        # circumradius for side 3.6
VERTS = [C + R * UP,
         C + R * np.array([-np.cos(PI / 6), -np.sin(PI / 6), 0.0]),
         C + R * np.array([np.cos(PI / 6), -np.sin(PI / 6), 0.0])]
CH = STREAM_COLORS[:3]                      # channel c drawn in stream c's color
G_T = (0.62, 0.22, 0.16)                    # an example gate for token t
G_S = (0.17, 0.63, 0.20)                    # and for token s
BAR_SCALE = 1.6                             # probability 1 -> 1.6 units in the decomposition
BASE_Y = -2.45

# landscape (schematic loss along one routing direction), drawn under the simplex
LS_CX, LS_TOP, LS_HALF, LS_DROP = -4.1, -1.65, 2.15, 1.45

# a signed-matrix palette for the illustrative gradient
POS_C, NEG_C = "#7FB2D9", "#E07A5F"


def ls_shape(uu):
    """0 on a flat plateau at uu = 0, falling to 1 on flat floors at |uu| -> 1 (schematic loss)."""
    return np.tanh((uu / 0.62) ** 4)


def bary(w):
    return sum(wi * v for wi, v in zip(w, VERTS))


def at(vo, i, phrase=None, frac=0.0):
    """Time (in the block) at which `phrase` is reached inside sentence i, by character position."""
    text, a, b = vo.synth.sentences[i]
    if phrase is not None:
        k = text.find(phrase)
        frac = max(k, 0) / max(len(text), 1)
    return a + frac * (b - a)


def split_lhs(eq, key):
    """(the glyphs of `key`, every other glyph of eq) as two VGroups."""
    lhs = eq[key]
    pts = set(lhs.family_members_with_points())
    rest = VGroup(*[m for m in eq.family_members_with_points() if m not in pts])
    return lhs, rest


def prob_bars(vals, colors, width=0.26, gap=0.07, scale=BAR_SCALE):
    """Vertical signed bars on a common baseline at y = 0 (one per channel)."""
    bars = VGroup()
    for i, v in enumerate(vals):
        h = max(abs(v) * scale, 1e-3)
        r = Rectangle(width=width, height=h).set_fill(colors[i], 0.85).set_stroke(width=0)
        x = i * (width + gap)
        r.move_to([x, h / 2 if v >= 0 else -h / 2, 0])
        bars.add(r)
    total = (len(vals) - 1) * (width + gap)
    base = Line([-width / 2 - 0.06, 0, 0], [total + width / 2 + 0.06, 0, 0]).set_stroke(FAINT, 1.5)
    g = VGroup(base, bars)
    g.shift(-total / 2 * RIGHT)
    g.bars, g.base = bars, base
    return g


def vec_bars(vals, scale, color, width=0.17, gap=0.06):
    """A small signed vector drawn as bars around a baseline (for the LayerNorm picture)."""
    bars = VGroup()
    for i, v in enumerate(vals):
        h = max(abs(v) * scale, 1e-3)
        r = Rectangle(width=width, height=h).set_fill(color, 0.85).set_stroke(width=0)
        r.move_to([i * (width + gap), h / 2 if v >= 0 else -h / 2, 0])
        bars.add(r)
    total = (len(vals) - 1) * (width + gap)
    base = Line([-width / 2 - 0.05, 0, 0], [total + width / 2 + 0.05, 0, 0]).set_stroke(FAINT, 1.5)
    g = VGroup(base, bars)
    g.shift(-total / 2 * RIGHT)
    g.bars = bars
    return g


def ln_box(width=1.55, height=0.62):
    box = RoundedRectangle(width=width, height=height, corner_radius=0.1)
    box.set_fill(PANEL, 1).set_stroke(MUTED, 1.5)
    lab = L("LayerNorm", size=22, color=INK).move_to(box)
    return VGroup(box, lab)


class FlatStart(ClankersScene):
    def construct(self):
        card = self.chapter_card(4, "A flat start")
        self.wait(0.6)
        title = section_title("The uniform gate")
        self.play(FadeOut(card, shift=0.3 * UP), FadeIn(title, shift=0.3 * UP))

        # ============================================================ N4.1 the decomposition
        tri = Polygon(*VERTS).set_stroke(MUTED, 2).set_fill(GATE_COLOR, 0.06)
        corner_dots = VGroup(*[Dot(v, radius=0.09).set_fill(CH[i], 1) for i, v in enumerate(VERTS)])
        corner_labs = VGroup(
            L("ch0", size=22, color=CH[0], weight="BOLD").next_to(VERTS[0], RIGHT, buff=0.18),
            L("ch1", size=22, color=CH[1], weight="BOLD").next_to(VERTS[1], LEFT, buff=0.18),
            L("ch2", size=22, color=CH[2], weight="BOLD").next_to(VERTS[2], RIGHT, buff=0.18),
        )
        caption = L("k = 3: every gate is a point of the triangle", size=22, color=MUTED)
        caption.move_to([C[0], -1.0, 0])
        cdot = Dot(C, radius=0.08).set_fill(INK, 1)
        c_lab = VGroup(M(R"\tfrac{1}{k}\mathbf{1}", size=30), L("uniform", size=22, color=MUTED))
        c_lab.arrange(DOWN, buff=0.06).next_to(cdot, RIGHT, buff=0.15).shift(0.3 * DOWN)

        pt_t, pt_s = bary(G_T), bary(G_S)
        dot_t = Dot(pt_t, radius=0.075).set_fill(GATE_COLOR, 1)
        dot_s = Dot(pt_s, radius=0.075).set_fill(GATE_COLOR, 1)
        arr_t = Arrow(C, pt_t, buff=0.04, thickness=3.5).set_fill(GATE_COLOR)
        arr_s = Arrow(C, pt_s, buff=0.04, thickness=3.5).set_fill(GATE_COLOR)
        lab_gt = M(R"g_t", size=28, color=GATE_COLOR).next_to(dot_t, RIGHT, buff=0.1)
        lab_gs = M(R"g_s", size=28, color=GATE_COLOR).next_to(dot_s, LEFT, buff=0.1)
        lab_dt = M(R"\delta_t", size=28, color=GATE_COLOR).next_to(arr_t.get_center(), LEFT, buff=0.12)
        lab_ds = M(R"\delta_s", size=28, color=GATE_COLOR).next_to(arr_s.get_center(), DOWN + RIGHT * 0.3, buff=0.06)

        # decomposition g_t = uniform + delta_t, as bars per channel
        uni = [1 / 3] * 3
        dvals = [g - 1 / 3 for g in G_T]
        grp_g = prob_bars(G_T, CH).shift([-5.65, BASE_Y, 0])
        grp_u = prob_bars(uni, CH).shift([-4.25, BASE_Y, 0])
        grp_d = prob_bars(dvals, CH).shift([-2.85, BASE_Y, 0])
        eq_sign = M("=", size=34).move_to([-4.95, BASE_Y + 0.3, 0])
        plus_sign = M("+", size=34).move_to([-3.55, BASE_Y + 0.3, 0])
        dl_g = M(R"g_t", size=28).next_to(grp_g, DOWN, buff=0.22).set_y(BASE_Y - 0.55)
        dl_u = M(R"\tfrac{1}{k}\mathbf{1}", size=28).set_y(BASE_Y - 0.55).set_x(-4.25)
        dl_d = M(R"\delta_t", size=28, color=GATE_COLOR).set_y(BASE_Y - 0.55).set_x(-2.85)
        sum0 = M(R"\textstyle\sum = 0", size=26, color=GATE_COLOR).move_to([-1.75, BASE_Y + 0.05, 0])
        decomp = VGroup(grp_g, eq_sign, grp_u, plus_sign, grp_d, dl_g, dl_u, dl_d, sum0)

        # equations (right column, centred at x = 2.85)
        XR = 2.85
        e0 = M(R"g_t\cdot g_s", size=44).move_to([XR, 2.3, 0])
        e0_lab = L("the context-gate match, eq. (1)", size=22, color=MUTED).next_to(e0, DOWN, buff=0.22)
        e1 = M(R"g_t = \tfrac{1}{k}\mathbf{1} + \delta_t,\qquad \textstyle\sum_c \delta_t[c] = 0", size=38)
        e1.move_to([XR, 0.85, 0])
        e1_a = e1[R"g_t = \tfrac{1}{k}\mathbf{1}"]
        e1_b = e1[R"+ \delta_t,"]
        e1_c = e1[R"\textstyle\sum_c \delta_t[c] = 0"]
        e2a = M(R"g_t\cdot g_s = \left(\tfrac{1}{k}\mathbf{1}+\delta_t\right)\cdot\left(\tfrac{1}{k}\mathbf{1}+\delta_s\right)",
                size=36).move_to([XR, 2.3, 0])
        mid_s = R"\tfrac{1}{k}\textstyle\sum_c \delta_s[c]"
        mid_t = R"\tfrac{1}{k}\textstyle\sum_c \delta_t[c]"
        e_full = M(R"g_t\cdot g_s = \tfrac{1}{k} + " + mid_s + " + " + mid_t + R" + \delta_t\cdot\delta_s",
                   size=34, isolate=[mid_s, mid_t]).move_to([XR, 2.3, 0])
        if e_full.get_width() > 7.2:
            e_full.set_width(7.2)
        e3 = M(R"g_t\cdot g_s = \tfrac{1}{k} + \delta_t\cdot\delta_s", size=44).move_to([XR, 2.3, 0])
        e3_tag = L("(3)", size=22, color=MUTED).next_to(e3, RIGHT, buff=0.45)
        src = source_note("Report §1, eq. (3); README §1")

        with self.voiceover(
            "Learning that is harder than it looks, and a short calculation shows why. Write each gate as the "
            "uniform distribution plus a deviation, delta, whose entries sum to zero. The gate match becomes one "
            "over k, plus the dot product of the two deviations."
        ) as vo:
            self.play(ShowCreation(tri), LaggedStartMap(GrowFromCenter, corner_dots, lag_ratio=0.3),
                      LaggedStartMap(FadeIn, corner_labs, lag_ratio=0.3), run_time=1.4)
            self.play(FadeIn(caption, shift=0.1 * UP), FadeIn(e0, shift=0.2 * DOWN), FadeIn(e0_lab), FadeIn(src))
            self.play(Indicate(e0, color=GATE_COLOR, scale_factor=1.08), run_time=1.0)
            # sentence 2: uniform + deviation
            vo.wait_until_sentence(1)
            self.play(GrowFromCenter(cdot), FadeIn(c_lab, shift=0.1 * RIGHT), Write(e1_a), run_time=1.0)
            self.play(FadeIn(grp_u), FadeIn(dl_u), run_time=0.7)
            vo.wait_until(at(vo, 1, "a deviation") - 0.2)
            self.play(GrowArrow(arr_t), FadeIn(dot_t, scale=0.5), FadeIn(lab_gt), FadeIn(lab_dt), Write(e1_b),
                      run_time=1.0)
            self.play(FadeIn(grp_g), FadeIn(dl_g), FadeIn(eq_sign), FadeIn(plus_sign),
                      TransformFromCopy(grp_u, grp_d), FadeIn(dl_d), run_time=1.1)
            vo.wait_until(at(vo, 1, "whose entries") - 0.1)
            self.play(Write(e1_c), FadeIn(sum0, shift=0.1 * LEFT),
                      Indicate(grp_d.bars, color=WARN, scale_factor=1.15), run_time=1.1)
            # sentence 3: the gate match expands
            vo.wait_until_sentence(2)
            self.play(GrowArrow(arr_s), FadeIn(dot_s, scale=0.5), FadeIn(lab_gs), FadeIn(lab_ds),
                      FadeOut(e0_lab), TransformMatchingTex(e0, e2a), run_time=0.9)
            self.wait(0.3)
            # expand the product: the left side slides, the right side cross-fades (no symbol scramble)
            lhs_a, rest_a = split_lhs(e2a, R"g_t\cdot g_s =")
            lhs_f, rest_f = split_lhs(e_full, R"g_t\cdot g_s =")
            self.play(Transform(lhs_a, lhs_f), FadeTransform(rest_a, rest_f), run_time=0.9)
            self.remove(e2a, rest_f)
            self.add(e_full)
            strikes = VGroup(*[
                Line(e_full[m].get_corner(DL) + 0.04 * DL, e_full[m].get_corner(UR) + 0.04 * UR).set_stroke(BAD, 3)
                for m in (mid_s, mid_t)])
            zero_note = L("each sum is zero", size=22, color=BAD).next_to(e_full, DOWN, buff=0.18)
            zero_note.match_x(VGroup(e_full[mid_s], e_full[mid_t]))
            self.play(ShowCreation(strikes, lag_ratio=0.5), FadeIn(zero_note), run_time=0.7)
            # the struck terms leave with their strikes, then the rest closes up into eq. (3)
            # (opacity, not FadeOut: a FadeOut restores sub-parts of e_full when it finishes)
            pluses = e_full["+"]                # 1/k (+) mid_s (+) mid_t (+) dd: the last two go too
            self.play(e_full[mid_s].animate.set_opacity(0).shift(0.15 * DOWN),
                      e_full[mid_t].animate.set_opacity(0).shift(0.15 * DOWN),
                      pluses[1].animate.set_opacity(0), pluses[2].animate.set_opacity(0),
                      FadeOut(strikes, shift=0.15 * DOWN), FadeOut(zero_note, shift=0.15 * DOWN), run_time=0.4)
            self.play(TransformMatchingTex(e_full, e3, matched_keys=[R"\delta_t\cdot\delta_s"]), run_time=0.9)
            self.play(FadeIn(e3_tag, shift=0.1 * LEFT), run_time=0.4)

        # ============================================================ N4.2 LayerNorm, gradient, flat
        v = np.array([0.8, -0.3, 1.1, 0.2, -0.6, 0.5])
        u = (v - v.mean()) / v.std()
        k_disp = 3.0
        ya, yb = 0.55, -0.85
        xv, xln, xu = 0.85, 3.05, 5.25
        rowA_v = vec_bars(v, 0.5, MEMORY_COLOR).move_to([xv, ya, 0])
        rowB_v = vec_bars(v / k_disp, 0.5, MEMORY_COLOR)
        rowB_v.shift([xv - rowB_v[0].get_center()[0], yb - rowB_v[0].get_center()[1], 0])
        rowA_v_base_y = rowA_v[0].get_center()[1]
        rowA_v.shift((ya - rowA_v_base_y) * UP)
        rowA_u = vec_bars(u, 0.3, MEMORY_COLOR)
        rowA_u.shift([xu - rowA_u[0].get_center()[0], ya - rowA_u[0].get_center()[1], 0])
        rowB_u = vec_bars(u, 0.3, MEMORY_COLOR)
        rowB_u.shift([xu - rowB_u[0].get_center()[0], yb - rowB_u[0].get_center()[1], 0])
        lnA = ln_box().move_to([xln, ya, 0])
        lnB = ln_box().move_to([xln, yb, 0])
        arrsA = VGroup(Arrow([xv + 0.8, ya, 0], [xln - 0.85, ya, 0], buff=0.05, thickness=2.5),
                       Arrow([xln + 0.85, ya, 0], [xu - 0.8, ya, 0], buff=0.05, thickness=2.5)).set_fill(MUTED)
        arrsB = arrsA.copy().shift((yb - ya) * UP)
        tagA = M(R"\times 1", size=30, color=MUTED).move_to([-0.45, ya, 0])
        tagB = M(R"\times \tfrac{1}{k}", size=30, color=WARN).move_to([-0.45, yb, 0])
        head_v = L("attention output", size=22, color=MUTED).move_to([xv, ya + 0.75, 0])
        head_u = L("after LayerNorm", size=22, color=MUTED).move_to([xu, ya + 0.75, 0])
        same_box = SurroundingRectangle(VGroup(rowA_u, rowB_u), buff=0.12).set_stroke(GOOD, 2)
        same_lab = L("identical", size=22, color=GOOD).next_to(same_box, DOWN, buff=0.12)
        ln_group = VGroup(rowA_v, rowB_v, rowA_u, rowB_u, lnA, lnB, arrsA, arrsB, tagA, tagB, head_v, head_u,
                          same_box, same_lab)

        frac_part = e3[R"\tfrac{1}{k}"]
        dd_part = e3[R"\delta_t\cdot\delta_s"]
        strike_k = Line(frac_part.get_corner(DL) + 0.06 * DL, frac_part.get_corner(UR) + 0.06 * UR)
        strike_k.set_stroke(BAD, 4)
        strike_lab = L("LayerNorm cancels", size=22, color=BAD).next_to(frac_part, DOWN, buff=0.22)
        route_box = SurroundingRectangle(dd_part, buff=0.1).set_stroke(GATE_COLOR, 2.5)
        route_lab = L("routing", size=22, color=GATE_COLOR, weight="BOLD").next_to(route_box, UP, buff=0.1)

        e4 = M(R"\frac{\partial \mathcal{L}}{\partial \delta_t} = \sum_{s\neq t} (\,\cdots)\,\delta_s",
               size=40).move_to([XR, 0.5, 0])
        e4_ds = e4[R"\delta_s"][-1]
        ds_note = L("the other tokens' deviations", size=22, color=WARN).next_to(e4, DOWN, buff=0.2)
        ds_note.align_to(e4, RIGHT)
        e5 = M(R"\frac{\partial \mathcal{L}}{\partial \delta_t}\,\Big|_{\delta = 0} = 0", size=40)
        e5.move_to([XR, 0.5, 0])
        pull = T("no first-order pull toward any routing", size=30, color=WARN).move_to([XR, -1.0, 0])

        # landscape under the simplex
        def ls_point(uu, cx=LS_CX, top=LS_TOP, half=LS_HALF, drop=LS_DROP):
            return np.array([cx + half * uu, top - drop * ls_shape(uu), 0.0])

        curve = ParametricCurve(lambda uu: ls_point(uu), t_range=(-1, 1, 0.02)).set_stroke(INK, 3)
        base_y = LS_TOP - LS_DROP - 0.15
        hill = Polygon(*[ls_point(uu) for uu in np.linspace(-1, 1, 81)],
                       [LS_CX + LS_HALF, base_y, 0], [LS_CX - LS_HALF, base_y, 0])
        hill.set_fill(PANEL, 1).set_stroke(width=0)
        ls_axis = Line([LS_CX - LS_HALF - 0.1, base_y, 0], [LS_CX + LS_HALF + 0.1, base_y, 0]).set_stroke(MUTED, 2)
        ls_tick = Line([LS_CX, base_y - 0.08, 0], [LS_CX, base_y + 0.08, 0]).set_stroke(MUTED, 2)
        ls_uni = L("uniform", size=22, color=MUTED).next_to(ls_tick, DOWN, buff=0.08)
        ls_left = L("routed", size=22, color=MUTED).next_to([LS_CX - LS_HALF, base_y, 0], DOWN, buff=0.12)
        ls_right = L("routed", size=22, color=MUTED).next_to([LS_CX + LS_HALF, base_y, 0], DOWN, buff=0.12)
        ls_name = L("task loss (schematic)", size=22, color=MUTED).move_to([LS_CX, base_y + 0.45, 0])
        tangent = DashedLine([LS_CX - 0.95, LS_TOP + 0.02, 0], [LS_CX + 0.95, LS_TOP + 0.02, 0], dash_length=0.1)
        tangent.set_stroke(WARN, 3)
        slope_lab = L("slope 0", size=22, color=WARN).next_to(tangent, UP, buff=0.08)
        anchor_c = Dot(ls_point(0), radius=0.01).set_opacity(0)
        anchor_r = Dot(ls_point(1), radius=0.01).set_opacity(0)
        landscape = VGroup(hill, ls_axis, ls_tick, curve, ls_uni, ls_left, ls_right, ls_name, tangent, slope_lab,
                           anchor_c, anchor_r)

        with self.voiceover(
            "The constant one over k scales every score alike, and the layer norm after attention cancels it. "
            "So routing enters only through products of deviations. The gradient for one token's deviation is "
            "proportional to the other tokens' deviations, and at the uniform gate those are all zero. The task "
            "loss exerts no first-order pull toward any routing."
        ) as vo:
            self.play(FadeOut(e1), frac_part.animate.set_color(WARN), run_time=0.7)
            self.play(FadeIn(head_v), FadeIn(head_u), FadeIn(tagA), FadeIn(rowA_v), GrowArrow(arrsA[0]),
                      FadeIn(lnA), run_time=0.8)
            self.play(GrowArrow(arrsA[1]), FadeIn(rowA_u, shift=0.2 * RIGHT), run_time=0.6)
            vo.wait_until(at(vo, 0, "scales every score") - 0.1)
            self.play(TransformFromCopy(rowA_v, rowB_v), TransformFromCopy(tagA, tagB), run_time=1.0)
            vo.wait_until(at(vo, 0, "and the layer norm") - 0.1)
            self.play(GrowArrow(arrsB[0]), FadeIn(lnB), run_time=0.6)
            self.play(GrowArrow(arrsB[1]), FadeIn(rowB_u, shift=0.2 * RIGHT), run_time=0.6)
            self.play(ShowCreation(same_box), FadeIn(same_lab), ShowCreation(strike_k), FadeIn(strike_lab),
                      run_time=0.8)
            # sentence 2: only products of deviations
            vo.wait_until_sentence(1)
            self.play(FadeOut(ln_group), frac_part.animate.set_opacity(0.35), run_time=0.6)
            self.play(ShowCreation(route_box), FadeIn(route_lab, shift=0.1 * DOWN), run_time=0.8)
            self.play(Indicate(VGroup(arr_t, arr_s), color=WARN, scale_factor=1.1), run_time=1.0)
            # sentence 3: the gradient
            vo.wait_until_sentence(2)
            self.play(Write(e4), run_time=1.2)
            self.play(Indicate(lab_dt, color=WARN), Indicate(arr_t, color=WARN, scale_factor=1.05), run_time=0.9)
            vo.wait_until(at(vo, 2, "proportional") - 0.1)
            self.play(e4_ds.animate.set_color(WARN), FadeIn(ds_note, shift=0.1 * UP),
                      Indicate(lab_ds, color=WARN), run_time=0.9)
            vo.wait_until(at(vo, 2, "at the uniform gate") - 0.1)
            uni_g = prob_bars(uni, CH).shift([-5.65, BASE_Y, 0])
            zero_d = prob_bars([0, 0, 0], CH).shift([-2.85, BASE_Y, 0])
            self.play(dot_t.animate.move_to(C), dot_s.animate.move_to(C),
                      FadeOut(arr_t), FadeOut(arr_s), FadeOut(lab_gt), FadeOut(lab_gs), FadeOut(lab_dt),
                      FadeOut(lab_ds), Transform(grp_g, uni_g), Transform(grp_d, zero_d), run_time=1.3)
            vo.wait_until(at(vo, 2, "those are all zero") - 0.1)
            self.play(TransformMatchingTex(e4, e5), FadeOut(ds_note), run_time=1.0)
            # sentence 4: no first-order pull
            vo.wait_until_sentence(3)
            self.play(FadeOut(decomp, shift=0.2 * DOWN), FadeOut(caption), run_time=0.5)
            self.play(FadeIn(hill), ShowCreation(curve), ShowCreation(ls_axis),
                      FadeIn(ls_tick), FadeIn(ls_uni), FadeIn(ls_left), FadeIn(ls_right), run_time=1.1)
            self.play(ShowCreation(tangent), FadeIn(slope_lab), FadeIn(ls_name), Write(pull), run_time=1.1)
            self.add(anchor_c, anchor_r)

        # ============================================================ N4.3 the numbers
        gd = json.load(open(os.path.join(VIDEO_DIR, "data", "gate_dynamics.json")))
        run = gd["multichannel_runs"]["X:A4k4|258"]
        assert run["steps"][0] == 0
        G0 = np.array(run["stream_gate"][0])
        rows = VGroup()
        for s in range(G0.shape[0]):
            lab = L(f"stream {s}", size=22, color=STREAM_COLORS[s])
            bar = gate_bar(list(G0[s]), colors=STREAM_COLORS[:G0.shape[1]], width=3.2, height=0.24)
            rows.add(VGroup(lab, bar).arrange(RIGHT, buff=0.3))
        rows.arrange(DOWN, buff=0.1, aligned_edge=RIGHT)
        # name the columns: channel c is drawn in stream c's color (as on the triangle)
        ch_head = VGroup(*[L(f"ch{c}", size=20, color=STREAM_COLORS[c], weight="BOLD")
                           .next_to(rows[0][1][0][c], UP, buff=0.1) for c in range(G0.shape[1])])
        bars_block = VGroup(ch_head, rows)
        head0 = L(f"one real run at step 0 (k = {run['k']}): each stream's mean gate", size=22, color=MUTED)
        stat0 = L(f"every entry between {G0.min():.3f} and {G0.max():.3f}", size=24, color=INK)
        emb0 = L("embeddings initialized at std 0.02 (bdh.py)", size=22, color=MUTED)
        panel0 = VGroup(head0, bars_block, stat0, emb0).arrange(DOWN, buff=0.17)
        panel0.move_to([XR, 1.62, 0])
        rng = np.random.default_rng(4)
        cluster = VGroup(*[Dot(C + 0.035 * np.array([*rng.normal(size=2), 0]), radius=0.035).set_fill(GATE_COLOR, 0.9)
                           for _ in range(24)])
        here_lab = L("every gate starts here", size=22, color=GATE_COLOR).move_to([-2.35, 1.75, 0])
        here_arr = Arrow(here_lab.get_bottom() + 0.05 * DOWN + 0.4 * LEFT, C + 0.12 * (UP + RIGHT), buff=0.05,
                         thickness=2.5).set_fill(GATE_COLOR)
        src3 = source_note("Report §1; README §1; gate_dynamics.json (X, A4k4, seed 258)")

        # log-scale bars: gradient norm at step 1
        X0, DEC = 0.55, 0.75

        def xlog(val):
            return X0 + DEC * (np.log10(val) + 6)

        y_gate, y_rest, y_ax = -0.8, -1.45, -1.95
        ax = Line([X0, y_ax, 0], [xlog(10), y_ax, 0]).set_stroke(MUTED, 2)
        ticks = VGroup(*[Line([xlog(10.0 ** e), y_ax - 0.07, 0], [xlog(10.0 ** e), y_ax + 0.07, 0]).set_stroke(MUTED, 2)
                         for e in range(-6, 2)])
        tick_labs = VGroup(*[M(f"10^{{{e}}}", size=22, color=MUTED).next_to(ticks[i], DOWN, buff=0.1)
                             for i, e in enumerate(range(-6, 2))])
        head1 = L("gradient norm at step 1 (log scale)", size=22, color=MUTED).move_to([XR, -0.25, 0])
        bar_gate = Rectangle(width=xlog(8e-6) - X0, height=0.38).set_fill(GATE_COLOR, 0.9).set_stroke(width=0)
        bar_gate.move_to([X0, y_gate, 0], aligned_edge=LEFT)
        bar_gate_lo = Rectangle(width=xlog(5e-6) - X0, height=0.38).set_fill(GATE_COLOR, 1).set_stroke(width=0)
        bar_gate_lo.move_to([X0, y_gate, 0], aligned_edge=LEFT)
        bar_rest = Rectangle(width=xlog(2.6) - X0, height=0.38).set_fill(INK, 0.85).set_stroke(width=0)
        bar_rest.move_to([X0, y_rest, 0], aligned_edge=LEFT)
        name_gate = L("gate", size=22, color=GATE_COLOR, weight="BOLD").next_to(bar_gate, LEFT, buff=0.15)
        name_rest = L("rest of model", size=22, color=INK).next_to(bar_rest, LEFT, buff=0.15)
        name_gate.align_to(name_rest, RIGHT)
        val_gate = VGroup(M(R"< 10^{-5}", size=30, color=GATE_COLOR),
                          M(R"(\text{medians } 5\text{--}8 \times 10^{-6})", size=24, color=MUTED))
        val_gate.arrange(RIGHT, buff=0.2)
        val_gate.next_to(bar_gate, RIGHT, buff=0.2)
        val_rest = L("about 2.6", size=22, color=INK, weight="BOLD").next_to(bar_rest, RIGHT, buff=0.15)
        span = Line([xlog(1e-5), y_ax - 0.5, 0], [xlog(2.6), y_ax - 0.5, 0])
        brace = Brace(span, DOWN, buff=0.0).set_fill(WARN)
        brace_lab = L("more than five orders of magnitude", size=22, color=WARN).next_to(brace, DOWN, buff=0.08)
        guides = VGroup(DashedLine([xlog(1e-5), y_gate - 0.22, 0], [xlog(1e-5), y_rest + 0.22, 0], dash_length=0.06),
                        DashedLine([xlog(1e-5), y_rest - 0.22, 0], [xlog(1e-5), y_ax - 0.45, 0], dash_length=0.06),
                        DashedLine([xlog(2.6), y_rest, 0], [xlog(2.6), y_ax - 0.45, 0], dash_length=0.06))
        guides.set_stroke(WARN, 1.5, opacity=0.7)
        log_chart = VGroup(head1, ax, ticks, tick_labs, bar_gate, bar_gate_lo, bar_rest, name_gate, name_rest,
                           val_gate, val_rest, guides, brace, brace_lab)

        # softmax and the rank of the W_g gradient (illustration, k = 3)
        ch_arr1 = Arrow(LEFT, RIGHT, buff=0, thickness=2.5).set_width(1.1).set_fill(MUTED)
        ch_arr2 = Arrow(LEFT, RIGHT, buff=0, thickness=2.5).set_width(1.6).set_fill(MUTED)
        chain_row = VGroup(M(R"h_t", size=36), ch_arr1, M(R"z_t", size=36), ch_arr2,
                           M(R"g_t", size=36, color=GATE_COLOR)).arrange(RIGHT, buff=0.18)
        chain_row.move_to([XR, 2.45, 0])
        wg_lab = M(R"W_g", size=28).next_to(ch_arr1, UP, buff=0.06)
        soft_part = L("softmax", size=22, color=INK).next_to(ch_arr2, UP, buff=0.08)
        chain = VGroup(chain_row, wg_lab, soft_part)
        rng2 = np.random.default_rng(7)
        A = rng2.normal(size=(3, 10))
        A -= A.mean(axis=0, keepdims=True)
        A /= np.abs(A).max()
        cell = 0.3
        mat = VGroup()
        for r in range(3):
            row = VGroup()
            for c in range(10):
                val = A[r, c]
                sq = Square(cell).set_stroke(PANEL_EDGE, 1)
                sq.set_fill(POS_C if val >= 0 else NEG_C, 0.15 + 0.8 * abs(val))
                row.add(sq)
            row.arrange(RIGHT, buff=0.03)
            mat.add(row)
        mat.arrange(DOWN, buff=0.03)
        mat.move_to([XR - 0.35, 1.5, 0])
        row_dots = VGroup(*[Dot(radius=0.07).set_fill(CH[r], 1).next_to(mat[r], LEFT, buff=0.12) for r in range(3)])
        mat_lab = M(R"\frac{\partial \mathcal{L}}{\partial W_g}", size=32).next_to(row_dots, LEFT, buff=0.18)
        dims = VGroup(L("k rows", size=22, color=MUTED), L("(illustrative)", size=22, color=MUTED))
        dims.arrange(DOWN, buff=0.08, aligned_edge=LEFT).next_to(mat, RIGHT, buff=0.2)
        sum_row = VGroup(*[Square(cell).set_stroke(PANEL_EDGE, 1).set_fill(PANEL, 1) for _ in range(10)])
        sum_row.arrange(RIGHT, buff=0.03).next_to(mat, DOWN, buff=0.35)
        zeros = VGroup(*[L("0", size=22, color=INK).move_to(sq) for sq in sum_row])
        sum_lab = L("sum of rows", size=22, color=MUTED).next_to(sum_row, LEFT, buff=0.2)
        rank_lab = M(R"\text{rank} \le k - 1", size=30, color=WARN).next_to(sum_row, RIGHT, buff=0.3)
        rank_panel = VGroup(chain, mat, row_dots, mat_lab, dims, sum_row, zeros, sum_lab, rank_lab)

        with self.voiceover(
            "And the gate does start almost exactly uniform, because the embeddings are initialized small. At the "
            "first step, the gradient reaching the gate has a norm below ten to the minus five; the rest of the "
            "model's is about 2.6. One more detail: the gate ends in a softmax, so the k rows of the gradient into "
            "its output matrix, W_g, sum to zero, and that gradient has rank at most k minus one."
        ) as vo:
            right_old = VGroup(e3, e3_tag, strike_k, strike_lab, route_box, route_lab, e5, pull)
            self.play(FadeOut(right_old), FadeOut(src), run_time=0.6)
            self.play(FadeIn(head0), FadeIn(ch_head), LaggedStartMap(FadeIn, rows, shift=0.1 * RIGHT, lag_ratio=0.2),
                      LaggedStartMap(GrowFromCenter, cluster, lag_ratio=0.03), FadeIn(src3), run_time=1.6)
            self.play(FadeIn(here_lab), GrowArrow(here_arr), FadeIn(stat0, shift=0.1 * UP), run_time=0.9)
            vo.wait_until(at(vo, 0, "because the embeddings") - 0.1)
            self.play(FadeIn(emb0, shift=0.1 * UP), run_time=0.7)
            # sentence 2: the gradient norms
            vo.wait_until_sentence(1)
            self.play(FadeIn(head1), ShowCreation(ax), FadeIn(ticks), LaggedStartMap(FadeIn, tick_labs, lag_ratio=0.1),
                      run_time=1.1)
            vo.wait_until(at(vo, 1, "has a norm below") - 0.2)
            for b in (bar_gate, bar_gate_lo):
                b.save_state()
                b.stretch(1e-3, 0, about_edge=LEFT)
            self.play(FadeIn(name_gate), Restore(bar_gate), Restore(bar_gate_lo), run_time=1.0)
            self.play(FadeIn(val_gate, shift=0.1 * LEFT), run_time=0.7)
            vo.wait_until(at(vo, 1, "the rest of the") - 0.1)
            bar_rest.save_state()
            bar_rest.stretch(1e-3, 0, about_edge=LEFT)
            self.play(FadeIn(name_rest), Restore(bar_rest), run_time=1.4)
            self.play(FadeIn(val_rest), ShowCreation(guides), GrowFromCenter(brace), FadeIn(brace_lab),
                      run_time=1.0)
            # sentence 3: softmax, rank
            vo.wait_until_sentence(2)
            self.play(FadeOut(VGroup(head0, bars_block, stat0, emb0)), FadeOut(here_lab), FadeOut(here_arr), run_time=0.6)
            self.play(Write(chain), run_time=1.2)
            vo.wait_until(at(vo, 2, "softmax") - 0.1)
            self.play(Indicate(soft_part, color=WARN, scale_factor=1.15), run_time=0.9)
            vo.wait_until(at(vo, 2, "so the k rows") - 0.1)
            self.play(FadeIn(mat_lab), LaggedStartMap(FadeIn, mat, shift=0.1 * DOWN, lag_ratio=0.3),
                      LaggedStartMap(GrowFromCenter, row_dots, lag_ratio=0.3), FadeIn(dims), run_time=1.3)
            vo.wait_until(at(vo, 2, "sum to zero") - 0.6)
            self.play(FadeIn(sum_row), FadeIn(sum_lab), run_time=0.5)
            self.play(*[Transform(mat[r].copy(), sum_row.copy().set_fill(opacity=0), remover=True)
                        for r in range(3)], run_time=1.0)
            self.play(LaggedStartMap(FadeIn, zeros, scale=0.5, lag_ratio=0.06), run_time=0.8)
            vo.wait_until(at(vo, 2, "and that gradient") - 0.1)
            self.play(Write(rank_lab), run_time=1.0)
            self.play(FlashAround(rank_lab, color=WARN), run_time=1.0)

        # ============================================================ N4.4 the ball
        simplex_grp = VGroup(tri, corner_dots, corner_labs, cdot, c_lab, cluster, dot_t, dot_s)
        sc_s, sc_l = 1.1, 1.25
        with self.voiceover(
            "So the gate sits on a flat spot, and whatever pushes it off first decides where it goes."
        ) as vo:
            self.play(FadeOut(VGroup(log_chart, rank_panel)), FadeOut(src3), FadeOut(slope_lab), run_time=0.5)
            landscape.remove(slope_lab)
            self.play(simplex_grp.animate.scale(sc_s).move_to([-3.6, -0.15, 0]),
                      landscape.animate.scale(sc_l).move_to([3.0, -0.35, 0]), run_time=1.0)
            cx, cy = anchor_c.get_center()[:2]
            rx, ry = anchor_r.get_center()[:2]
            half, drop = rx - cx, cy - ry
            r_ball = 0.17

            def ball_pos(uu):
                p = np.array([cx + half * uu, cy - drop * ls_shape(uu), 0.0])
                slope = -drop * (ls_shape(uu + 1e-4) - ls_shape(uu - 1e-4)) / 2e-4 / half
                n = np.array([-slope, 1.0, 0.0])
                return p + r_ball * n / np.linalg.norm(n)

            ball = Circle(radius=r_ball).set_fill(GATE_COLOR, 1).set_stroke(INK, 1.5).move_to(ball_pos(0))
            gate_dot = Dot(cdot.get_center(), radius=0.1).set_fill(GATE_COLOR, 1).set_stroke(INK, 1.5)
            v0 = corner_dots[0].get_center()
            self.play(FadeIn(ball, shift=0.4 * DOWN), FadeIn(gate_dot, scale=0.5), run_time=0.7)
            vo.wait_until(at(vo, 0, "whatever pushes") - 0.1)
            push = Arrow(ball.get_left() + 1.0 * LEFT, ball.get_left() + 0.05 * LEFT, buff=0.0, thickness=3)
            push.set_fill(WARN)
            push_lab = L("first push", size=22, color=WARN).next_to(push, UP, buff=0.08)
            self.play(GrowArrow(push), FadeIn(push_lab), run_time=0.6)
            trail = TracedPath(gate_dot.get_center, stroke_color=GATE_COLOR, stroke_width=4)
            self.add(trail)
            u_end = 0.93

            def roll(mob, alpha):
                uu = 0.02 + (u_end - 0.02) * alpha
                ball.move_to(ball_pos(uu))
                gate_dot.move_to(interpolate(cdot.get_center(), v0, 0.92 * alpha))

            self.play(UpdateFromAlphaFunc(ball, roll, rate_func=rush_into), FadeOut(push), FadeOut(push_lab),
                      run_time=1.8)
            trail.clear_updaters()
            self.play(Flash(corner_dots[0], color=CH[0]), Indicate(corner_labs[0], color=CH[0]), run_time=0.8)

        # the other ways it could have gone
        c_now = cdot.get_center()
        alts = VGroup(*[DashedLine(c_now, interpolate(c_now, corner_dots[i].get_center(), 0.9), dash_length=0.1)
                        .set_stroke(GATE_COLOR, 2.5, opacity=0.6) for i in (1, 2)])
        ghost = Circle(radius=r_ball).set_fill(GATE_COLOR, 0.55).set_stroke(INK, 1.5, opacity=0.7)
        ghost.move_to(ball_pos(-u_end))
        alt_lab = L("another push, another channel", size=22, color=MUTED)
        alt_lab.next_to(simplex_grp, DOWN, buff=0.3)
        moral = T("The first push decides the routing.", size=32, color=WARN).move_to([0, -3.05, 0])
        self.play(FadeOut(c_lab), LaggedStartMap(ShowCreation, alts, lag_ratio=0.4), FadeIn(ghost, shift=0.3 * LEFT),
                  FadeIn(alt_lab), run_time=1.2)
        self.play(Write(moral), run_time=1.0)
        self.wait(1.0)
        self.clear_all()
