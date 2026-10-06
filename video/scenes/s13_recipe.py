"""Chapter 13 — the recipe: SLOW, HINGE, WINDOW, MUON (report §3 Recipes, §16; facts_model §3, §5.2).

Facts on screen:
  SLOW    one Adam, two groups; gate (W_in, W_h, W_g) 1e-3 throughout; every other parameter (embedding and
          convolution included) 1e-4 for updates 1-2400, 1e-3 after (report §3; test_slow_start.py:37-39, 239-241,
          418-423, 485-489).
  HINGE   SLOW + 1.0 * [relu(eta2_index - 0.2) + relu(eta2_half - 0.2)] on every training batch; eta2 = between-group
          / total sum of squares of the read gate at the key positions 3j+1, pooled over channels; gradient reaches
          the gate and the embedding only (report §3; test_slow_start.py:40-46, 242, 335-345).
          Example gates (facts_model §5.2, test_slow_start.eta2_hinge, S = 2, P = 4, B = 32, k = 2):
            position split (first-half keys -> channel 0): eta2 = 1.0 / 1.0 -> penalty 1.6
            random split per sequence (torch.Generator().manual_seed(0), rand > 0.5): eta2 = 0.031 / 0.003 -> 0
          Both are recomputed below with the same formula (asserted), and the morph between them is computed too.
  WINDOW  hinge weight 1 on updates 1-2400, 0 after (report §3; test_early_recipe.py:256).
  MUON    gate group Muon 0.005 throughout; other Muon weights (encoder, encoder_v, decoder, lm_head) 0.0005 for
          updates 1-2400, then 0.005; Adam groups (embedding, conv) 1e-4 then 1e-3 (report §3;
          test_early_recipe.py:254-259, 391-401).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403
from manimlib.config import manim_config  # noqa: E402

# ManimGL compiles every Tex through one shared working.tex; scenes rendered in parallel can swap each other's
# SVGs. Compile this scene's Tex in a private directory and salt the source so no racing cache entry is reused.
manim_config.directories.latex_cache = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "tex_work", "s13_recipe")
os.makedirs(manim_config.directories.latex_cache, exist_ok=True)


def M(tex: str, size: float = 40, color=INK, **kw) -> Tex:  # noqa: F811  (salted local version)
    return Tex(tex + "{}{}{}", font_size=size, fill_color=color, **kw)


# ---------------------------------------------------------------- the two example gates (facts_model §5.2)
RANDOM_ROWS = ['01000101', '01000001', '11001101', '01100001', '00000111', '10100011', '00111110', '10001101',
               '10111100', '11111010', '01101000', '10111100', '11001100', '00000101', '01100111', '11010000',
               '00010010', '10111110', '00100000', '01101000', '00000101', '00101101', '00101011', '11000111',
               '01000100', '11000101', '11001001', '00100111', '01000011', '11101001', '11011010', '10011000']
RAND = np.array([[float(c) for c in row] for row in RANDOM_ROWS])          # p(channel 0), 32 sequences x 8 keys
POSN = np.tile((np.arange(8) < 4).astype(float), (32, 1))                   # first-half keys -> channel 0
TAU = 0.2


def eta2(p):
    """eta^2 by triple index and by sequence half (test_slow_start.py:335-345). With k = 2 channels,
    p(channel 1) = 1 - p(channel 0) has the same sums of squares, so pooling over channels changes nothing."""
    b, n = p.shape
    mu = p.mean()
    tot = ((p - mu) ** 2).sum() + 1e-12
    between_idx = b * ((p.mean(0) - mu) ** 2).sum()
    h = np.arange(n) >= n // 2
    between_half = b * (h.sum() * (p[:, h].mean() - mu) ** 2 + (~h).sum() * (p[:, ~h].mean() - mu) ** 2)
    return between_idx / tot, between_half / tot


_ei, _eh = eta2(RAND)
assert abs(_ei - 0.0309) < 5e-4 and abs(_eh - 0.0030) < 5e-4, (_ei, _eh)
assert abs(eta2(POSN)[0] - 1) < 1e-9 and abs(eta2(POSN)[1] - 1) < 1e-9


def relu(x):
    return max(x, 0.0)


CH0, CH1 = STREAM_COLORS[0], STREAM_COLORS[1]
MID = "#3A4050"
HALF_COLOR = "#FFB27A"      # the eta2_half term (a lighter hinge orange)


def pcolor(p):
    """p(channel 0) -> colour: 1 = channel 0 (blue), 0 = channel 1 (yellow), 0.5 = undecided (grey)."""
    if p >= 0.5:
        return interpolate_color(MID, CH0, min(1.0, 2 * (p - 0.5)))
    return interpolate_color(MID, CH1, min(1.0, 2 * (0.5 - p)))


# ---------------------------------------------------------------- small helpers
def chip(name, color=INK, width=1.3, size=20):
    t = L(name, size, color, weight="BOLD")
    rect = RoundedRectangle(width=width, height=0.5, corner_radius=0.1).set_fill(PANEL, 1).set_stroke(color, 1.5)
    t.move_to(rect)
    g = VGroup(rect, t)
    g.name = name
    return g


def recolor_chip(c, color):
    c[0].set_stroke(color, 1.5)
    c[1].set_fill(color)
    return c


def pill_level(p, level):
    p[0].set_fill(opacity=0.14 * level).set_stroke(opacity=level)
    p[1].set_fill(opacity=level)
    return p


def hrect(x0, x1, y, h, color, opacity=1.0):
    w = max(x1 - x0, 1e-3)
    r = Rectangle(width=w, height=h).set_fill(color, opacity if x1 - x0 > 1e-3 else 0).set_stroke(width=0)
    return r.move_to([x0 + w / 2, y, 0])


SEQ = [(1, 0, 13), (0, 0, 2), (0, 3, 11), (1, 3, 2), (0, 2, 10), (1, 2, 6), (0, 1, 12), (1, 1, 6)]


def mini_token(kind, stream=None, s=0.38):
    sq = RoundedRectangle(width=s, height=s, corner_radius=0.07)
    if kind in ("ctx", "val"):
        c = STREAM_COLORS[stream]
        sq.set_fill(c, 0.9 if kind == "ctx" else 0.12).set_stroke(c, 1.5)
        return VGroup(sq)
    if kind == "key":
        sq.set_fill(PANEL, 1).set_stroke(KEY_COLOR, 1.5)
        t = L("K", 20, KEY_COLOR)
    else:
        sq.set_fill(QUERY_COLOR, 0.15).set_stroke(QUERY_COLOR, 2)
        t = L("?", 20, QUERY_COLOR, weight="BOLD")
    return VGroup(sq, t.move_to(sq))


def mini_strip():
    """The first training sequence of seed 280 (as in Ch. 2), drawn compactly: 8 triples and the query."""
    triples = VGroup()
    for s, _, _ in SEQ:
        triples.add(VGroup(mini_token("ctx", s), mini_token("key"), mini_token("val", s)).arrange(RIGHT, buff=0.05))
    triples.add(VGroup(mini_token("ctx", 0), mini_token("key"), mini_token("query")).arrange(RIGHT, buff=0.05))
    triples.arrange(RIGHT, buff=0.2)
    return triples


def make_lr_plot(x0, x1, y0, y1, umax=7200, lg0=-4.4, lg1=-1.9):
    """A learning-rate (log) vs update plot; returns (P, axes group). P(u, log10 lr) -> point."""
    def P(u, lg):
        return np.array([x0 + (x1 - x0) * u / umax, y0 + (y1 - y0) * (lg - lg0) / (lg1 - lg0), 0])

    x_axis = Line(P(0, lg0), P(umax, lg0) + 0.15 * RIGHT).set_stroke(MUTED, 2)
    y_axis = Line(P(0, lg0), P(0, lg1) + 0.1 * UP).set_stroke(MUTED, 2)
    ticks = VGroup()
    labels = VGroup()
    for u in range(0, umax + 1, 1200):
        ticks.add(Line(P(u, lg0), P(u, lg0) + 0.1 * DOWN).set_stroke(MUTED, 2))
        if u % 2400 == 0:
            labels.add(L(f"{u}", 22, MUTED).next_to(P(u, lg0), DOWN, buff=0.17))
    u_lab = L("update", 22, MUTED).next_to(labels[0], LEFT, buff=0.35)
    for lg, tex in [(-4, R"10^{-4}"), (-3, R"10^{-3}"), (-2, R"10^{-2}")]:
        ticks.add(Line(P(0, lg), P(0, lg) + 0.1 * LEFT).set_stroke(MUTED, 2))
        labels.add(M(tex, 26, MUTED).next_to(P(0, lg), LEFT, buff=0.17))
    title = L("learning rate (log scale)", 22, MUTED).next_to(y_axis.get_top(), RIGHT, buff=0.15).shift(0.05 * DOWN)
    axes = VGroup(x_axis, y_axis, ticks, labels, u_lab, title)
    return P, axes


def step_line(P, lo, hi, switch=2400, umax=7200, color=INK, width=4):
    vm = VMobject().set_points_as_corners([P(0, lo), P(switch, lo), P(switch, hi), P(umax, hi)])
    return vm.set_stroke(color, width)


LG5E3 = np.log10(0.005)
LG5E4 = np.log10(0.0005)


class Recipe(ClankersScene):
    def construct(self):
        card = self.chapter_card(13, "The recipe")
        self.wait(0.6)
        title = section_title("The recipe")
        self.play(FadeOut(card, shift=0.3 * UP), FadeIn(title, shift=0.3 * UP))

        self.pills = VGroup(*[verdict_badge(n, size=22, color=c) for n, c in [
            ("SLOW", SLOW_COLOR), ("HINGE", HINGE_COLOR), ("WINDOW", HINGE_COLOR), ("MUON", MUON_COLOR)]])
        self.pills.arrange(RIGHT, buff=0.2).to_corner(UR, buff=0.45)

        self.part_slow()
        self.part_hinge_stat()
        self.part_penalty()
        self.part_labels()
        self.part_variants()
        self.wait(1.0)
        self.clear_all()

    # ------------------------------------------------------------------ helpers
    def focus(self, *idx):
        anims = []
        for i, p in enumerate(self.pills):
            p.generate_target()
            pill_level(p.target, 1.0 if i in idx else 0.3)
            anims.append(MoveToTarget(p))
        return anims

    # ================================================================== N13.1 SLOW
    def part_slow(self):
        pills = self.pills
        names = ["embedding", "W_in", "encoder", "conv", "W_h", "encoder_v", "decoder", "W_g", "lm_head"]
        chips = {n: chip(n, INK, width=1.2) for n in names}
        pool = VGroup(*[chips[n] for n in names]).arrange_in_grid(3, 3, h_buff=0.2, v_buff=0.2)
        pool.move_to([0, 0.1, 0])
        pool_lab = L("the model's parameters", 22, MUTED).next_to(pool, UP, buff=0.3)

        # the optimizer panel (left)
        g1_names = ["W_in", "W_h", "W_g"]
        g2_names = ["embedding", "conv", "encoder", "encoder_v", "decoder", "lm_head"]
        g1_slots = VGroup(*[chips[n].copy() for n in g1_names]).arrange(RIGHT, buff=0.12)
        g2_slots = VGroup(*[chips[n].copy() for n in g2_names]).arrange_in_grid(2, 3, h_buff=0.12, v_buff=0.12)
        for s in g1_slots:
            recolor_chip(s, GATE_COLOR)
        for s in g2_slots:
            recolor_chip(s, SLOW_COLOR)

        def group_box(text, color, slots):
            lab = L(text, 22, color, weight="BOLD")
            box = RoundedRectangle(width=4.1, height=lab.get_height() + slots.get_height() + 0.5, corner_radius=0.12)
            box.set_fill(color, 0.06).set_stroke(color, 1.5)
            lab.next_to(box.get_corner(UL), DR, buff=0.15)
            slots.next_to(lab, DOWN, buff=0.15).set_x(box.get_x())
            return VGroup(box, lab)

        g1 = group_box("group 1: the gate", GATE_COLOR, g1_slots)
        g2 = group_box("group 2: everything else", SLOW_COLOR, g2_slots)
        panel_head = L("one Adam optimizer", 24, INK, weight="BOLD")
        stack = VGroup(panel_head, VGroup(g1, g1_slots), VGroup(g2, g2_slots)).arrange(DOWN, buff=0.3)
        stack.move_to([-4.35, -0.25, 0])
        panel = SurroundingRectangle(stack, buff=0.2).round_corners(0.18).set_fill(PANEL, 0.6)
        panel.set_stroke(ADAM_COLOR, 1.5)
        g1_box, g1_lab = g1
        g2_box, g2_lab = g2

        # the plot (right)
        P, axes = make_lr_plot(-1.0, 6.2, -2.6, 1.4)
        gate_line = Line(P(0, -3), P(7200, -3)).set_stroke(GATE_COLOR, 7)
        rest_lo = Line(P(0, -4), P(2400, -4)).set_stroke(SLOW_COLOR, 4)
        rest_hi = VMobject().set_points_as_corners([P(2400, -4), P(2400, -3), P(7200, -3)]).set_stroke(SLOW_COLOR, 3.5)
        gate_lab = L("gate: 10⁻³ throughout", 22, GATE_COLOR).next_to(P(4800, -3), UP, buff=0.15)
        rest_lab = L("everything else: 10⁻⁴", 22, SLOW_COLOR).next_to(P(1200, -4), DOWN, buff=0.15)
        rest_lab.align_to(P(150, -4), LEFT)
        jump_lab = L("× 10 after update 2400", 22, SLOW_COLOR).next_to(P(2400, -3.5), RIGHT, buff=0.15)
        shade = Rectangle(width=P(2400, 0)[0] - P(0, 0)[0], height=P(0, -1.9)[1] - P(0, -4.4)[1])
        shade.set_fill(SLOW_COLOR, 0.08).set_stroke(width=0).move_to(
            [(P(0, 0)[0] + P(2400, 0)[0]) / 2, (P(0, -1.9)[1] + P(0, -4.4)[1]) / 2, 0])
        shade_lab = VGroup(L("slow phase", 22, SLOW_COLOR, weight="BOLD"), L("updates 1–2400", 22, SLOW_COLOR))
        shade_lab.arrange(DOWN, buff=0.08).move_to(shade).align_to(shade, UP).shift(0.15 * DOWN)
        src = source_note("Report §3 (Recipes); test_slow_start.py:37-39, 239-241, 418-423")

        with self.voiceover(
            "Here is the recipe precisely. SLOW is one Adam optimizer with two groups. The gate's three weight "
            "matrices train at ten to the minus three throughout. Every other parameter, the embedding and the "
            "convolution included, trains at ten to the minus four for updates one to twenty-four hundred, and at "
            "ten to the minus three after that."
        ) as vo:
            self.play(LaggedStartMap(FadeIn, pills, shift=0.15 * DOWN, lag_ratio=0.2), run_time=1.2)
            self.play(*self.focus(0), run_time=0.5)
            vo.wait_until_sentence(1)
            self.play(FadeIn(pool_lab), LaggedStartMap(FadeIn, pool, shift=0.1 * UP, lag_ratio=0.06), run_time=1.0)
            self.play(FadeOut(pool_lab), FadeIn(panel), FadeIn(panel_head), FadeIn(g1_box), FadeIn(g1_lab),
                      FadeIn(g2_box), FadeIn(g2_lab),
                      *[ReplacementTransform(chips[n], s) for n, s in zip(g1_names, g1_slots)],
                      *[ReplacementTransform(chips[n], s) for n, s in zip(g2_names, g2_slots)],
                      run_time=1.6)
            vo.wait_until_sentence(2)
            self.play(FadeIn(axes), FadeIn(src), run_time=0.8)
            self.play(LaggedStart(*[FlashAround(c, color=GATE_COLOR) for c in g1_slots], lag_ratio=0.15),
                      ShowCreation(gate_line), run_time=1.2)
            self.play(FadeIn(gate_lab, shift=0.1 * UP), run_time=0.5)
            vo.wait_until_sentence(3)
            self.play(FlashAround(g2_box, color=SLOW_COLOR), Indicate(g2_lab, color=SLOW_COLOR, scale_factor=1.08),
                      run_time=1.0)
            vo.wait_until(vo.time_of(3) + 1.45)
            self.play(*[FlashAround(c, color=WARN) for c in g2_slots[:2]],
                      *[Indicate(c[1], color=WARN, scale_factor=1.15) for c in g2_slots[:2]], run_time=1.2)
            vo.wait_until(vo.time_of(3) + 3.9)
            self.play(FadeIn(shade), FadeIn(shade_lab), ShowCreation(rest_lo), run_time=1.4)
            self.play(FadeIn(rest_lab, shift=0.1 * DOWN), run_time=0.5)
            vo.wait_until(vo.time_of(3) + 8.0)
            self.play(ShowCreation(rest_hi), run_time=1.4)
            self.play(FadeIn(jump_lab, shift=0.1 * RIGHT), run_time=0.5)

        self.play(FadeOut(VGroup(panel, panel_head, g1_box, g1_lab, g2_box, g2_lab, g1_slots, g2_slots, axes, gate_line,
                                 rest_lo, rest_hi, gate_lab, rest_lab, jump_lab, shade, shade_lab, src)),
                  run_time=0.7)

    # ================================================================== N13.2 the eta^2 statistic
    def part_hinge_stat(self):
        hdr_a = M(R"\text{HINGE} = \text{SLOW} +", 34, t2c={R"\text{HINGE}": HINGE_COLOR, R"\text{SLOW}": SLOW_COLOR})
        hdr_b = M(R"\text{penalty}", 34, HINGE_COLOR)
        hdr = VGroup(hdr_a, hdr_b).arrange(RIGHT, buff=0.2).move_to([0, 2.55, 0])
        self.hdr_a, self.hdr_b = hdr_a, hdr_b

        # one sequence; the read gate at its 8 key positions (query key excluded)
        strip = mini_strip().move_to([0, 1.75, 0])
        keys = VGroup(*[strip[j][1] for j in range(8)])
        q_key = strip[8][1]

        # the batch matrix: 8 of 32 sequences x 8 key positions, coloured by p(channel 0)
        PITCH, CELL = 0.39, 0.34
        X0, Y0 = -5.75, 0.62
        cells = VGroup(*[VGroup(*[Square(CELL).set_stroke(width=0).move_to([X0 + c * PITCH, Y0 - r * PITCH, 0])
                                  for c in range(8)]) for r in range(8)])
        col_means = VGroup(*[Square(CELL).set_stroke(INK, 1).move_to([X0 + c * PITCH, -2.62, 0]) for c in range(8)])
        half_w = 4 * PITCH - (PITCH - CELL)
        half_means = VGroup(
            Rectangle(width=half_w, height=CELL).set_stroke(INK, 1).move_to([X0 + 1.5 * PITCH, -3.17, 0]),
            Rectangle(width=half_w, height=CELL).set_stroke(INK, 1).move_to([X0 + 5.5 * PITCH, -3.17, 0]))
        self.cells, self.col_means, self.half_means = cells, col_means, half_means
        self.set_cells(POSN)
        gate_cells = VGroup(*[cells[0][j].copy().next_to(keys[j], DOWN, buff=0.22) for j in range(8)])

        j_labels = VGroup(*[L(f"{c}", 22, MUTED).move_to([X0 + c * PITCH, 1.02, 0]) for c in range(8)])
        j_cap = L("j", 22, MUTED, weight="BOLD").move_to([X0 - 0.5, 1.02, 0])
        rows_lab = L("8 of the batch's 32 sequences", 22, MUTED).rotate(PI / 2).move_to([X0 - 0.5, Y0 - 3.5 * PITCH, 0])

        def swatch(color, text, tcolor):
            return VGroup(Square(0.24).set_fill(color, 1).set_stroke(width=0), L(text, 22, tcolor)).arrange(RIGHT, buff=0.1)

        legend = VGroup(swatch(CH0, "channel 0", CH0), swatch(CH1, "channel 1", CH1),
                        swatch(MID, "50/50", MUTED)).arrange(RIGHT, buff=0.3)
        # first shown under the strip, captioned, then moved above the batch matrix
        gate_cap = L("the read gate at each key:", 22, INK)
        intro_row = VGroup(gate_cap, legend.copy()).arrange(RIGHT, buff=0.35).move_to([0, 0.55, 0])
        legend.move_to([X0 - 0.17, 1.5, 0], aligned_edge=LEFT)
        # grouping annotations in INK (yellow is channel 1 here)
        col_outlines = VGroup(*[Rectangle(width=CELL + 0.06, height=8 * PITCH).set_stroke(INK, 2)
                                .move_to([X0 + c * PITCH, Y0 - 3.5 * PITCH, 0]) for c in range(8)])
        idx_lab = L("by index", 22, MUTED).next_to(col_means, RIGHT, buff=0.25)
        half_lab = L("by half", 22, MUTED).next_to(half_means, RIGHT, buff=0.25)
        divider = DashedLine([X0 + 3.5 * PITCH, 1.2, 0], [X0 + 3.5 * PITCH, -3.38, 0], dash_length=0.08)
        divider.set_stroke(INK, 2)
        half_heads = VGroup(L("first half", 22, INK).move_to([X0 + 1.5 * PITCH, 1.02, 0]),
                            L("second half", 22, INK).move_to([X0 + 5.5 * PITCH, 1.02, 0]))
        mean_note = L("means over all 32", 20, MUTED).next_to(half_lab, DOWN, buff=0.1).align_to(half_lab, LEFT)
        ex_cap = VGroup(L("example gate:", 22, MUTED), L("position split", 22, INK, weight="BOLD"))
        ex_cap.arrange(DOWN, buff=0.1).move_to([-1.75, Y0 - 3.5 * PITCH, 0])

        # right column: the definition and two meters
        frac = M(R"\eta^2 = \frac{\text{between-group variance}}{\text{total variance}}", 30)
        frac.move_to([1.85, 1.5, 0])
        pooled = VGroup(L("pooled over", 22, MUTED), L("channels", 22, MUTED)).arrange(DOWN, buff=0.06)
        pooled.next_to(frac, RIGHT, buff=0.35)
        MX0, MW = 0.6, 3.2
        self.MX0, self.MW = MX0, MW
        ys = [0.35, -0.3]
        tracks = VGroup(*[Rectangle(width=MW, height=0.3).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1)
                          .move_to([MX0 + MW / 2, y, 0]) for y in ys])
        m_labs = VGroup(M(R"\eta^2_{\text{index}}", 36), M(R"\eta^2_{\text{half}}", 36))
        for lab, y in zip(m_labs, ys):
            lab.move_to([0.3, y, 0], aligned_edge=RIGHT)
        fills = VGroup(*[hrect(MX0, MX0 + 1e-3, y, 0.3, KEY_COLOR) for y in ys])
        vals = VGroup(*[DecimalNumber(0, num_decimal_places=3, font_size=26, text_config=dict(font=SANS))
                        .set_color(INK).move_to([4.0, y, 0], aligned_edge=LEFT) for y in ys])
        scale = VGroup(L("0", 20, MUTED).move_to([MX0, 0.72, 0]), L("1", 20, MUTED).move_to([MX0 + MW, 0.72, 0]))
        self.ys, self.tracks, self.fills, self.vals = ys, tracks, fills, vals
        src = source_note("Report §3; test_slow_start.py:40-46, 335-345")

        with self.voiceover(
            "HINGE adds a penalty. On each training batch, take the read gate's channel probabilities at the key "
            "positions. Ask how much of their variance is explained by the triple's index, and how much by which "
            "half of the sequence the key is in. Those are two eta-squared values between zero and one."
        ) as vo:
            self.play(*self.focus(1), Write(hdr), run_time=1.0)
            vo.wait_until_sentence(1)
            self.play(LaggedStartMap(FadeIn, strip, lag_ratio=0.08), run_time=0.9)
            self.play(LaggedStart(*[Indicate(k[1], color=INK, scale_factor=1.6) for k in keys], lag_ratio=0.08),
                      *[k[0].animate.set_stroke(INK, 3) for k in keys],
                      q_key.animate.set_opacity(0.3), run_time=1.0)
            self.play(LaggedStart(*[FadeIn(g, shift=0.15 * DOWN) for g in gate_cells], lag_ratio=0.08),
                      FadeIn(intro_row, shift=0.1 * DOWN), run_time=0.8)
            self.play(*[ReplacementTransform(gate_cells[j], cells[0][j]) for j in range(8)],
                      ReplacementTransform(intro_row[1], legend), FadeOut(gate_cap),
                      FadeOut(strip), FadeIn(j_labels), FadeIn(j_cap), run_time=1.0)
            self.play(LaggedStart(*[FadeIn(cells[r], shift=0.1 * DOWN) for r in range(1, 8)], lag_ratio=0.12),
                      FadeIn(rows_lab), FadeIn(src), run_time=1.2)
            self.play(FadeIn(ex_cap, shift=0.1 * LEFT), run_time=0.5)
            vo.wait_until_sentence(2)
            self.play(FadeIn(frac, shift=0.1 * DOWN), FadeIn(pooled), run_time=0.8)
            vo.wait_until(vo.time_of(2) + 2.6)
            self.play(LaggedStartMap(ShowCreation, col_outlines, lag_ratio=0.08), j_labels.animate.set_color(INK),
                      run_time=0.8)
            self.play(*[TransformFromCopy(VGroup(*[cells[r][c] for r in range(8)]), col_means[c]) for c in range(8)],
                      FadeOut(col_outlines), FadeIn(idx_lab), run_time=1.0)
            vo.wait_until(vo.time_of(2) + 4.4)
            self.play(ShowCreation(divider), FadeTransform(j_labels[:4], half_heads[0]),
                      FadeTransform(j_labels[4:], half_heads[1]), FadeOut(j_cap), run_time=0.8)
            self.play(TransformFromCopy(VGroup(*[cells[r][c] for r in range(8) for c in range(4)]), half_means[0]),
                      TransformFromCopy(VGroup(*[cells[r][c] for r in range(8) for c in range(4, 8)]), half_means[1]),
                      FadeIn(half_lab), FadeIn(mean_note), run_time=1.0)
            vo.wait_until_sentence(3)
            self.play(FadeIn(tracks), FadeIn(m_labs), FadeIn(scale), FadeIn(vals), run_time=0.6)
            self.add(fills)
            self.play(UpdateFromAlphaFunc(fills, lambda m, a: self.set_meters(a, a, over=False)), run_time=1.2)

        self.stat = dict(rows_lab=rows_lab, legend=legend, idx_lab=idx_lab, half_lab=half_lab, divider=divider,
                         half_heads=half_heads, mean_note=mean_note, frac=frac, pooled=pooled, m_labs=m_labs,
                         scale=scale, src=src, ex_cap=ex_cap)

    def set_cells(self, p):
        for r in range(8):
            for c in range(8):
                self.cells[r][c].set_fill(pcolor(p[r, c]), 1)
        cm = p.mean(0)
        for c in range(8):
            self.col_means[c].set_fill(pcolor(cm[c]), 1)
        self.half_means[0].set_fill(pcolor(p[:, :4].mean()), 1)
        self.half_means[1].set_fill(pcolor(p[:, 4:].mean()), 1)

    def set_meters(self, ei, eh, over=True):
        MX0, MW = self.MX0, self.MW
        for k, (e, y) in enumerate(zip((ei, eh), self.ys)):
            self.fills[k].become(hrect(MX0, MX0 + e * MW, y, 0.3, KEY_COLOR))
            self.vals[k].set_value(e)
            if over:
                col = HINGE_COLOR if k == 0 else HALF_COLOR
                self.overs[k].become(hrect(MX0 + TAU * MW, MX0 + max(e, TAU) * MW, y, 0.3, col))

    # ================================================================== N13.3 the penalty
    def part_penalty(self):
        MX0, MW = self.MX0, self.MW
        st = self.stat
        full_b = M(R"1.0\times\big[\,\mathrm{relu}(\eta^2_{\text{index}}-0.2)+\mathrm{relu}(\eta^2_{\text{half}}-0.2)\,\big]",
                   34, HINGE_COLOR)
        self.full_b = full_b
        new_hdr = VGroup(self.hdr_a.copy(), full_b).arrange(RIGHT, buff=0.2).move_to([0, 2.55, 0])
        xt = MX0 + TAU * MW
        RY0, RSY = -1.85, 1.25
        thresh = DashedLine([xt, 0.6, 0], [xt, RY0, 0], dash_length=0.08).set_stroke(HINGE_COLOR, 2)
        t_lab = L("0.2", 20, HINGE_COLOR, weight="BOLD").move_to([xt, 0.72, 0])
        self.overs = VGroup(*[hrect(xt, xt + (1 - TAU) * MW, y, 0.3, HINGE_COLOR if k == 0 else HALF_COLOR)
                              for k, y in enumerate(self.ys)])

        # relu(x - 0.2) on the same horizontal scale as the meters
        r_axis = Line([MX0, RY0, 0], [MX0 + MW + 0.15, RY0, 0]).set_stroke(MUTED, 2)
        r_yaxis = Line([MX0, RY0, 0], [MX0, RY0 + 1.1, 0]).set_stroke(MUTED, 2)
        r_ticks = VGroup(L("0", 20, MUTED).move_to([MX0, RY0 - 0.25, 0]),
                         L("0.2", 20, HINGE_COLOR).move_to([xt, RY0 - 0.25, 0]),
                         L("1", 20, MUTED).move_to([MX0 + MW, RY0 - 0.25, 0]))
        curve = VMobject().set_points_as_corners([[MX0, RY0, 0], [xt, RY0, 0], [MX0 + MW, RY0 + (1 - TAU) * RSY, 0]])
        curve.set_stroke(HINGE_COLOR, 4)
        r_lab = M(R"\mathrm{relu}(\eta^2 - 0.2)", 26, HINGE_COLOR).next_to(curve.get_end(), RIGHT, buff=0.32)
        dot_i = Dot(radius=0.08).set_fill(HINGE_COLOR, 1).set_stroke(BG, 1.5)
        dot_h = Circle(radius=0.13).set_stroke(HALF_COLOR, 3)
        self.dots = VGroup(dot_i, dot_h)
        self.RY0, self.RSY = RY0, RSY

        # the penalty bar: the two orange parts laid end to end
        PY = -2.7
        p_lab = L("penalty", 22, HINGE_COLOR, weight="BOLD").move_to([0.3, PY, 0], aligned_edge=RIGHT)
        p_track = Rectangle(width=2 * (1 - TAU) * MW, height=0.3).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1)
        p_track.move_to([MX0 + (1 - TAU) * MW, PY, 0])
        self.pen_segs = VGroup(hrect(MX0, MX0 + (1 - TAU) * MW, PY, 0.3, HINGE_COLOR),
                               hrect(MX0 + (1 - TAU) * MW, MX0 + 2 * (1 - TAU) * MW, PY, 0.3, HALF_COLOR))
        self.pen_val = DecimalNumber(1.6, num_decimal_places=2, font_size=28, text_config=dict(font=SANS))
        self.pen_val.set_color(HINGE_COLOR).move_to([p_track.get_right()[0] + 0.15, PY, 0], aligned_edge=LEFT)
        self.badge = verdict_badge("fires", size=20, color=HINGE_COLOR).move_to([5.3, -2.2, 0])
        self.PY = PY
        src = source_note("Report §3; examples scored by eta2_hinge, test_slow_start.py:329-345")

        with self.voiceover(
            "The penalty is the part of each value above 0.2, added together. It is zero while the gate's choice "
            "has little to do with position, and it fires when the gate starts splitting by position. Its gradient "
            "reaches only the gate and the embedding.",
            spoken="The penalty is the part of each value above zero point two, added together. It is zero while the "
            "gate's choice has little to do with position, and it fires when the gate starts splitting by position. "
            "Its gradient reaches only the gate and the embedding.",
        ) as vo:
            self.play(self.hdr_a.animate.move_to(new_hdr[0]), FadeTransform(self.hdr_b, full_b),
                      ShowCreation(thresh), FadeIn(t_lab), run_time=1.0)
            self.play(*[GrowFromEdge(o, LEFT) for o in self.overs], run_time=0.6)
            self.set_relu(1.0, 1.0)
            self.play(FadeIn(r_axis), FadeIn(r_yaxis), FadeIn(r_ticks), ShowCreation(curve), FadeIn(r_lab),
                      FadeIn(self.dots), run_time=0.9)
            self.play(FadeIn(p_lab), FadeIn(p_track),
                      TransformFromCopy(self.overs[0], self.pen_segs[0]),
                      TransformFromCopy(self.overs[1], self.pen_segs[1]), run_time=1.0)
            self.play(FadeIn(self.pen_val), FadeIn(self.badge, scale=0.8), FadeOut(st["src"]), FadeIn(src),
                      run_time=0.5)
            # zero while the choice has little to do with position: morph to the random split
            cap = st["ex_cap"]
            cap_rand = L("random split", 20, INK, weight="BOLD").move_to(cap[1])
            cap_pos = L("→ position split", 20, INK, weight="BOLD").move_to(cap[1])
            vo.wait_until_sentence(1)
            self.play(UpdateFromAlphaFunc(self.cells, lambda m, a: self.apply_state(1 - a)),
                      FadeTransform(cap[1], cap_rand, time_span=(0, 0.6)), run_time=1.8)
            vo.wait_until(vo.time_of(1) + 0.55 * (vo.time_of(2) - vo.time_of(1)))   # "... and it fires when"
            # ... and fires when the gate starts splitting by position
            self.play(UpdateFromAlphaFunc(self.cells, lambda m, a: self.apply_state(a), rate_func=linear),
                      FadeTransform(cap_rand, cap_pos, time_span=(0, 0.6)), run_time=2.6)
            self.play(Indicate(self.pen_val, color=HINGE_COLOR, scale_factor=1.3), run_time=0.5)
            st["ex_cap"] = VGroup(cap[0], cap_pos)
            vo.wait_until_sentence(2)
            right_col = VGroup(st["frac"], st["pooled"], st["m_labs"], st["scale"], self.tracks, self.fills,
                               self.overs, self.vals, thresh, t_lab, r_axis, r_yaxis, r_ticks, curve, r_lab, self.dots,
                               p_lab, p_track, self.pen_segs, self.pen_val, self.badge)
            self.part_gradient(right_col, src)
        self.wait(0.6)

    def apply_state(self, a):
        """a = 1: position split; a = 0: random split; in between, the mixture (all values computed)."""
        p = a * POSN + (1 - a) * RAND
        self.set_cells(p)
        ei, eh = eta2(p)
        self.set_meters(ei, eh)
        self.set_relu(ei, eh)
        MX0, MW, PY = self.MX0, self.MW, self.PY
        hi, hh = relu(ei - TAU), relu(eh - TAU)
        self.pen_segs[0].become(hrect(MX0, MX0 + hi * MW, PY, 0.3, HINGE_COLOR))
        self.pen_segs[1].become(hrect(MX0 + hi * MW, MX0 + (hi + hh) * MW, PY, 0.3, HALF_COLOR))
        self.pen_val.set_value(hi + hh)
        pill_level(self.badge, 1.0 if hi + hh > 0 else 0.0)

    def set_relu(self, ei, eh):
        for d, e in zip(self.dots, (ei, eh)):
            d.move_to([self.MX0 + e * self.MW, self.RY0 + relu(e - TAU) * self.RSY, 0])

    def part_gradient(self, right_col, old_src):
        """N13.3, last sentence: the penalty's gradient reaches the gate and the embedding only."""
        row1 = VGroup(chip("embedding", INK), chip("W_in", GATE_COLOR), chip("W_h", GATE_COLOR),
                      chip("W_g", GATE_COLOR)).arrange(RIGHT, buff=0.18).move_to([2.7, 0.0, 0])
        row2 = VGroup(*[chip(n, MUTED) for n in ["conv", "encoder", "encoder_v", "decoder", "lm_head"]])
        row2.arrange(RIGHT, buff=0.18).move_to([2.7, -1.35, 0])
        gate_brace = Brace(row1[1:], DOWN, buff=0.08).set_color(GATE_COLOR)
        gate_txt = L("the gate", 22, GATE_COLOR).next_to(gate_brace, DOWN, buff=0.06)
        src_box = VGroup(L("penalty", 24, HINGE_COLOR, weight="BOLD"), L("1.60", 24, HINGE_COLOR))
        src_box.arrange(RIGHT, buff=0.25)
        src_box = card(src_box, buff=0.18, edge=HINGE_COLOR).move_to([2.7, 1.55, 0])
        arrows = VGroup(*[Arrow(src_box.get_bottom(), c.get_top(), buff=0.08, thickness=3).set_color(HINGE_COLOR)
                          for c in row1])
        no_grad = L("no gradient from the penalty", 22, MUTED).next_to(row2, DOWN, buff=0.2)
        src = source_note("Report §3; test_slow_start.py:40-46, 135 (CHECK 107)")
        self.play(FadeOut(right_col), FadeOut(old_src), FadeIn(src), FadeIn(src_box), FadeIn(row1), FadeIn(row2),
                  run_time=0.8)
        self.play(LaggedStartMap(GrowArrow, arrows, lag_ratio=0.15), run_time=0.8)
        self.play(row2.animate.set_opacity(0.35), FadeIn(no_grad), GrowFromCenter(gate_brace), FadeIn(gate_txt),
                  run_time=0.6)
        self.grad = VGroup(row1, row2, gate_brace, gate_txt, src_box, arrows, no_grad, src)

    # ================================================================== N13.4 what it uses
    def part_labels(self):
        st = self.stat
        left = VGroup(self.cells, self.col_means, self.half_means, st["rows_lab"], st["legend"], st["idx_lab"],
                      st["half_lab"], st["divider"], st["half_heads"], st["mean_note"], st["ex_cap"])
        strip = mini_strip().move_to([0, 1.45, 0])
        stream_tags = VGroup(*[L(f"stream {s}", 22, STREAM_COLORS[s]).next_to(strip[i], DOWN, buff=0.22)
                               for i, s in enumerate([t[0] for t in SEQ] + [0])])
        strike = Line(stream_tags.get_left() + 0.15 * LEFT, stream_tags.get_right() + 0.15 * RIGHT)
        strike.set_stroke(BAD, 3)
        never = L("stream labels: the hinge never reads them", 22, INK).next_to(stream_tags, DOWN, buff=0.2)
        keys = VGroup(*[strip[j][1] for j in range(8)])
        marks = VGroup(*[Triangle().scale(0.09).rotate(PI).set_fill(WARN, 1).set_stroke(width=0)
                         .next_to(k, UP, buff=0.1) for k in keys])

        # two cards
        ok_head = VGroup(L("✓", 30, GOOD, weight="BOLD"), L("no stream labels", 28, GOOD, weight="BOLD"))
        ok_head.arrange(RIGHT, buff=0.2)
        maps = VGroup(*[VGroup(token(f"CTX{s}", "ctx", s), Arrow(LEFT, RIGHT, buff=0).set_width(0.6),
                               L("channel ?", 22, MUTED)).arrange(RIGHT, buff=0.18) for s in (0, 1)])
        maps.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        ok_body = L("never says which channel a stream goes to", 22, INK)
        ok_in = VGroup(ok_head, maps, ok_body).arrange(DOWN, buff=0.22)
        ok_card = card(ok_in, buff=0.25, edge=GOOD)
        ok_card[0].set_width(5.9, stretch=True)
        ok_card.move_to([-3.25, -1.6, 0])

        need_head = VGroup(L("!", 30, WARN, weight="BOLD"), L("needs the key positions", 28, WARN, weight="BOLD"))
        need_head.arrange(RIGHT, buff=0.2)
        tri = VGroup(mini_token("ctx", 0), mini_token("key"), mini_token("val", 0)).arrange(RIGHT, buff=0.05)
        tri[1][0].set_stroke(WARN, 2.5)
        pos_txt = VGroup(L("keys at positions 3j + 1,", 22, INK), L("and their index j", 22, INK)).arrange(
            DOWN, buff=0.08, aligned_edge=LEFT)
        need_ex = VGroup(tri, pos_txt).arrange(RIGHT, buff=0.3)
        need_body = L("that is task structure", 22, WARN)
        need_in = VGroup(need_head, need_ex, need_body).arrange(DOWN, buff=0.3)
        need_card = card(need_in, buff=0.25, edge=WARN)
        need_card[0].set_width(5.9, stretch=True)
        card_h = max(ok_card[0].get_height(), need_card[0].get_height())
        ok_card[0].set_height(card_h, stretch=True)
        need_card[0].set_height(card_h, stretch=True)
        need_card.move_to([3.25, -1.6, 0])
        src = source_note("Report §3, §16")

        with self.voiceover(
            "Notice what it uses and what it doesn't. It uses no stream labels; it never says which channel any "
            "stream should go to. But it does need to know where the keys are, and that is task structure."
        ) as vo:
            self.play(FadeOut(left), FadeOut(self.grad), run_time=0.7)
            self.play(LaggedStartMap(FadeIn, strip, lag_ratio=0.06), FadeIn(src), run_time=1.0)
            vo.wait_until_sentence(1)
            self.play(LaggedStartMap(FadeIn, stream_tags, shift=0.1 * DOWN, lag_ratio=0.06), run_time=0.8)
            self.play(ShowCreation(strike), FadeIn(never), run_time=0.7)
            self.play(FadeIn(ok_card, shift=0.15 * UP), run_time=0.8)
            self.play(LaggedStart(*[Indicate(m[2], color=WARN) for m in maps], lag_ratio=0.3), run_time=1.0)
            vo.wait_until_sentence(2)
            self.play(LaggedStart(*[Indicate(k[1], color=WARN, scale_factor=1.6) for k in keys], lag_ratio=0.06),
                      *[k[0].animate.set_stroke(WARN, 2.5) for k in keys],
                      LaggedStartMap(FadeIn, marks, shift=0.1 * DOWN, lag_ratio=0.06), run_time=1.2)
            self.play(FadeIn(need_card, shift=0.15 * UP), run_time=0.8)
            self.play(Indicate(need_body, color=WARN, scale_factor=1.1), run_time=0.8)

        self.play(FadeOut(VGroup(strip, stream_tags, strike, never, marks, ok_card, need_card, src,
                                 self.hdr_a, self.full_b)), run_time=0.7)

    # ================================================================== N13.5 WINDOW and MUON
    def part_variants(self):
        P, axes = make_lr_plot(-1.0, 6.2, -2.9, 0.55)
        gate_line = Line(P(0, -3), P(7200, -3)).set_stroke(GATE_COLOR, 7)
        rest_line = step_line(P, -4, -3, color=SLOW_COLOR, width=3.5)
        shade = Rectangle(width=P(2400, 0)[0] - P(0, 0)[0], height=P(0, -1.9)[1] - P(0, -4.4)[1])
        shade.set_fill(SLOW_COLOR, 0.08).set_stroke(width=0).move_to(
            [(P(0, 0)[0] + P(2400, 0)[0]) / 2, (P(0, -1.9)[1] + P(0, -4.4)[1]) / 2, 0])

        def legend_row(color, head, sub):
            sw = Line(ORIGIN, 0.45 * RIGHT).set_stroke(color, 6)
            txt = VGroup(L(head, 22, color, weight="BOLD"), L(sub, 22, MUTED)).arrange(DOWN, buff=0.06,
                                                                                      aligned_edge=LEFT)
            return VGroup(sw, txt).arrange(RIGHT, buff=0.2, aligned_edge=UP)

        leg_slow = VGroup(legend_row(GATE_COLOR, "gate (Adam)", "10⁻³ throughout"),
                          legend_row(SLOW_COLOR, "everything else (Adam)", "10⁻⁴, then 10⁻³")).arrange(
            DOWN, buff=0.35, aligned_edge=LEFT)
        leg_slow.move_to([-6.45, -0.4, 0], aligned_edge=LEFT)
        leg_muon = VGroup(legend_row(GATE_COLOR, "gate (Muon)", "0.005 throughout"),
                          legend_row(MUON_COLOR, "other matrices (Muon)", "0.0005, then 0.005"),
                          legend_row(ADAM_COLOR, "embedding, conv (Adam)", "10⁻⁴, then 10⁻³")).arrange(
            DOWN, buff=0.3, aligned_edge=LEFT)
        leg_muon.move_to([-6.45, -0.8, 0], aligned_edge=LEFT)
        leg_title = L("SLOW, as before", 22, SLOW_COLOR, weight="BOLD").move_to([-6.45, 0.45, 0], aligned_edge=LEFT)
        leg_title_m = L("MUON", 22, MUON_COLOR, weight="BOLD").move_to([-6.45, 0.45, 0], aligned_edge=LEFT)

        # hinge weight (WINDOW), on the same update axis
        HY0, HY1 = 1.2, 2.1
        h_axis = Line([P(0, 0)[0], HY0, 0], [P(7200, 0)[0] + 0.15, HY0, 0]).set_stroke(MUTED, 2)
        h_yaxis = Line([P(0, 0)[0], HY0, 0], [P(0, 0)[0], HY1 + 0.2, 0]).set_stroke(MUTED, 2)
        h_ticks = VGroup(L("0", 22, MUTED).next_to([P(0, 0)[0], HY0, 0], LEFT, buff=0.17),
                         L("1", 22, MUTED).next_to([P(0, 0)[0], HY1, 0], LEFT, buff=0.17))
        h_name = L("hinge weight", 22, HINGE_COLOR, weight="BOLD").next_to(h_ticks, LEFT, buff=0.3)
        x = lambda u: P(u, 0)[0]  # noqa: E731
        h_line = VMobject().set_points_as_corners([[x(0), HY1, 0], [x(2400), HY1, 0], [x(2400), HY0, 0],
                                                   [x(7200), HY0, 0]]).set_stroke(HINGE_COLOR, 5)
        h_on = L("hinge on", 22, HINGE_COLOR).next_to([x(1200), HY1, 0], DOWN, buff=0.15)
        h_off = L("off after update 2400", 22, MUTED).next_to([x(4800), HY0, 0], UP, buff=0.15)
        win_shade = Rectangle(width=x(2400) - x(0), height=HY1 + 0.3 - HY0).set_fill(HINGE_COLOR, 0.08)
        win_shade.set_stroke(width=0).move_to([(x(0) + x(2400)) / 2, (HY1 + 0.3 + HY0) / 2, 0])

        # MUON lines
        gate_m = Line(P(0, LG5E3), P(7200, LG5E3)).set_stroke(GATE_COLOR, 7)
        rest_m = step_line(P, LG5E4, LG5E3, color=MUON_COLOR, width=3.5)
        adam_m = step_line(P, -4, -3, color=ADAM_COLOR, width=3.5)
        t_mid = (P(1200, LG5E3) + P(1200, LG5E4)) / 2
        tenth = VGroup(Arrow(t_mid, P(1200, LG5E3), buff=0.06, thickness=2.5),
                       Arrow(t_mid, P(1200, LG5E4), buff=0.06, thickness=2.5)).set_color(MUON_COLOR)
        tenth_lab = L("× 0.1", 22, MUON_COLOR).next_to(tenth, RIGHT, buff=0.1)
        lr_lab = L("0.005", 22, GATE_COLOR, weight="BOLD").next_to(P(6000, LG5E3), UP, buff=0.12)
        src = source_note("Report §3 (Recipes); test_early_recipe.py:254-259, 391-401")

        with self.voiceover(
            "Two variants. WINDOW applies the hinge only during updates one to twenty-four hundred. MUON runs the same "
            "two-speed schedule under Muon, at a learning rate of 0.005.",
            spoken="Two variants. Window applies the hinge only during updates one to twenty-four hundred. Muon runs "
            "the same two-speed schedule under Muon, at a learning rate of zero point zero zero five.",
        ) as vo:
            self.play(*self.focus(2, 3), FadeIn(axes), FadeIn(shade), FadeIn(gate_line), FadeIn(rest_line),
                      FadeIn(leg_slow), FadeIn(leg_title), FadeIn(src), run_time=1.0)
            vo.wait_until_sentence(1)
            self.play(*self.focus(2), FadeIn(h_axis), FadeIn(h_yaxis), FadeIn(h_ticks), FadeIn(h_name),
                      run_time=0.6)
            self.play(ShowCreation(h_line), FadeIn(win_shade), run_time=1.6)
            self.play(FadeIn(h_on), FadeIn(h_off), run_time=0.6)
            vo.wait_until_sentence(2)
            # MUON is the two-speed schedule under Muon; the window hinge is a separate variant: dim it
            hinge_lines = VGroup(h_axis, h_yaxis, h_line)          # open polylines: dim the stroke only
            hinge_texts = VGroup(h_ticks, h_name, h_on, h_off)
            self.play(*self.focus(3), hinge_lines.animate.set_stroke(opacity=0.25),
                      hinge_texts.animate.set_fill(opacity=0.25), win_shade.animate.set_fill(opacity=0.03),
                      run_time=0.5)
            self.play(Transform(gate_line, gate_m), TransformFromCopy(rest_line, rest_m),
                      Transform(rest_line, adam_m),
                      FadeOut(VGroup(leg_slow, leg_title), time_span=(0, 0.6)),
                      FadeIn(VGroup(leg_muon, leg_title_m), time_span=(0.8, 1.8)), run_time=1.8)
            self.play(GrowFromCenter(tenth), FadeIn(tenth_lab), run_time=0.7)
            vo.wait_until(vo.time_of(2) + 4.9)                                      # "... of 0.005"
            self.play(FadeIn(lr_lab, shift=0.1 * UP), Indicate(leg_muon[0], color=GATE_COLOR, scale_factor=1.05),
                      run_time=0.8)
