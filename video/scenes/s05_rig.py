"""Chapter 5 — two more parts of the setup: the short causal convolution and Muon
(report §1, §2, §3; README §7 and Table 5; test_multilayer_binding.py:225-319; test_early_recipe.py:28-39,
231, 254-259, 382-384; torch.optim._muon)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403

import atexit  # noqa: E402
import shutil  # noqa: E402

import manimlib.utils.cache as _mcache  # noqa: E402
from diskcache import Cache  # noqa: E402
from manimlib.config import manim_config  # noqa: E402

# manimlib compiles every formula in one shared working.tex and caches the SVG in one global disk cache
# keyed by the TeX string, so renders running in parallel (other scenes, or two renders of this one) can
# compile each other's formulas and cache them under the wrong key (an early render of this scene showed
# Chapter 4's formulas). So this process compiles in its own working directory and keeps its own SVG cache;
# nothing shared is read or written.
_TEX_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "latex_s05")
_TEX_DIR = os.path.join(_TEX_ROOT, f"work_{os.getpid()}")
os.makedirs(_TEX_DIR, exist_ok=True)
manim_config.directories.latex_cache = _TEX_DIR
_mcache._cache = Cache(os.path.join(_TEX_ROOT, "svg"), size_limit=int(2e8))
atexit.register(shutil.rmtree, _TEX_DIR, True)


def M(tex: str, size: float = 40, color=INK, **kw) -> Tex:  # noqa: F811  (shadows common.style.M)
    """common.style.M, but a formula that fails to compile raises instead of silently reusing the previous
    formula's working.dvi (manimlib only raises when no .dvi exists: "\\|M\\|" fails under this template's T3
    encoding, and an earlier render showed "U Σ' V^T" in its place)."""
    try:
        os.remove(os.path.join(_TEX_DIR, "working.dvi"))
    except FileNotFoundError:
        pass
    return Tex(tex, font_size=size, fill_color=color, **kw)


B0, Y1 = STREAM_COLORS[0], STREAM_COLORS[1]

# Muon's Newton–Schulz coefficients (test_early_recipe.py MUON_KW; torch.optim._muon)
NS_A, NS_B, NS_C = 3.4445, -4.775, 2.0315
# a momentum matrix whose singular values spread over 0.1 ... 20 (illustration)
SV0 = np.array([20, 12, 7, 4, 2, 1, 0.4, 0.1], dtype=float)
SV0_LABELS = ["20", "12", "7", "4", "2", "1", "0.4", "0.1"]


def ns_poly(s):
    return NS_A * s + NS_B * s ** 3 + NS_C * s ** 5


def ns_trajectory():
    """Singular values after normalising by the Frobenius norm, then after each of the five steps."""
    s = SV0 / np.sqrt((SV0 ** 2).sum())
    out = [s]
    for _ in range(5):
        s = ns_poly(s)
        out.append(s)
    return out


def ns_matrices():
    """An 8x8 matrix M with singular values SV0, and its five-step Newton–Schulz output (float64)."""
    rng = np.random.default_rng(7)
    U, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    V, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    Mm = U @ np.diag(SV0) @ V.T
    X = Mm / np.linalg.norm(Mm)
    for _ in range(5):
        A = X @ X.T
        X = NS_A * X + (NS_B * A + NS_C * A @ A) @ X
    return Mm, X


def heatmap(mat, cell=0.22, buff=0.025):
    m = np.max(np.abs(mat))
    cells = VGroup()
    for v in mat.flatten():
        t = abs(v) / m
        col = interpolate_color(PANEL, MUON_COLOR if v >= 0 else KEY_COLOR, 0.15 + 0.85 * t)
        cells.add(Square(cell).set_fill(col, 1).set_stroke(width=0))
    cells.arrange_in_grid(mat.shape[0], mat.shape[1], buff=buff)
    frame = SurroundingRectangle(cells, buff=0.06).set_stroke(PANEL_EDGE, 1.5)
    return VGroup(frame, cells)


def box_label(lines, width, height, color=INK, size=24, fill=PANEL):
    rect = RoundedRectangle(width=width, height=height, corner_radius=0.12)
    rect.set_fill(fill, 1).set_stroke(color, 2)
    if isinstance(lines, str):
        lines = [lines]
    txt = VGroup(*[L(s, size, color) for s in lines]).arrange(DOWN, buff=0.08)
    txt.move_to(rect)
    g = VGroup(rect, txt)
    g.rect, g.txt = rect, txt
    return g


def weight_chip(value, lag, w=0.8, h=0.5):
    box = RoundedRectangle(width=w, height=h, corner_radius=0.08).set_fill(PANEL, 1).set_stroke(INK, 1.5)
    num = L(value, 22, INK).move_to(box)
    lab = M(rf"w_{lag}", 28, MUTED).next_to(box, DOWN, buff=0.1)
    g = VGroup(box, num, lab)
    g.box, g.num, g.lab = box, num, lab
    return g


def ghost_token(width=0.95, height=0.55):
    r = RoundedRectangle(width=width, height=height, corner_radius=0.1).set_stroke(FAINT, 1.5)
    d = DashedVMobject(r, num_dashes=18)
    z = L("0", 20, FAINT).move_to(r)
    return VGroup(d, z)


def mini_ctx_key(stream):
    a = token(f"CTX{stream}", "ctx", stream, width=0.85, height=0.46)
    b = token("K0", width=0.62, height=0.46)
    plus = L("+", 22, MUTED)
    g = VGroup(a, plus, b).arrange(RIGHT, buff=0.08)
    frame = SurroundingRectangle(g, buff=0.08).round_corners(0.08).set_stroke(MUTED, 1.2).set_fill(PANEL, 0.6)
    return VGroup(frame, g)


def param_chip(name, shape, color=INK, width=1.75, dashed=False):
    n = L(name, 22, color, weight="BOLD")
    s = L(shape, 20, MUTED)
    txt = VGroup(n, s).arrange(DOWN, buff=0.08)
    rect = RoundedRectangle(width=width, height=txt.get_height() + 0.3, corner_radius=0.1)
    if dashed:
        rect.set_stroke(FAINT, 1.5)
        rect = DashedVMobject(rect, num_dashes=30)
    else:
        rect.set_fill(PANEL, 1).set_stroke(color, 1.5)
    txt.move_to(rect)
    if txt.get_width() > width - 0.15:
        txt.set_width(width - 0.15)
    return VGroup(rect, txt)


class ConvAndMuon(ClankersScene):
    def at(self, vo, i, frac=0.0):
        """Wait until a fraction `frac` of the way through sentence i of the block."""
        _, a, b = vo.synth.sentences[i]
        vo.wait_until(a + frac * (b - a))

    def construct(self):
        chap = self.chapter_card(5, "Two more parts")
        self.wait(0.6)
        title = section_title("Two more parts")
        self.play(FadeOut(chap, shift=0.3 * UP), FadeIn(title, shift=0.3 * UP))

        # ================================================================ N5.1 the convolution
        part1 = card_part("1", "a short causal convolution", INK)
        part2 = card_part("2", "the Muon optimizer", MUON_COLOR)
        parts = VGroup(part1, part2).arrange(RIGHT, buff=0.8).move_to(0.3 * UP)

        # block diagram: tokens -> short conv -> token mixer
        tok_lab = L("tokens", 24, MUTED).move_to([-5.4, 0.2, 0])
        conv_blk = box_label(["short causal", "convolution"], 2.8, 1.2, INK, 24)
        conv_blk.move_to([-2.2, 0.2, 0])
        mix_blk = box_label(["token mixer", "(attention)"], 2.8, 1.2, MEMORY_COLOR, 24)
        mix_blk.move_to([1.9, 0.2, 0])
        more_lab = L("…", 30, MUTED).move_to([5.2, 0.2, 0])
        blk_arrows = VGroup(
            Arrow(tok_lab.get_right(), conv_blk.get_left(), buff=0.15, thickness=3, fill_color=MUTED),
            Arrow(conv_blk.get_right(), mix_blk.get_left(), buff=0.15, thickness=3, fill_color=MUTED),
            Arrow(mix_blk.get_right(), more_lab.get_left(), buff=0.15, thickness=3, fill_color=MUTED),
        )
        blk_head = L("the usual layout in Mamba and most modern linear-attention models", 24, MUTED)
        blk_head.move_to([0, 1.65, 0])
        blk_tag = verdict_badge("in most arms since Phase IV", size=24, color=WARN).move_to([-2.2, -1.25, 0])
        src1 = source_note("Report §1; README eq. (4); test_multilayer_binding.py:225-319")

        # sliding window over a token row (first two triples of Ch. 2's sequence)
        toks = [token(f"CTX1", "ctx", 1), token("K0"), token("V13", "val", 1),
                token("CTX0", "ctx", 0), token("K0"), token("V2", "val", 0)]
        ghosts = [ghost_token() for _ in range(3)]
        in_row = VGroup(*ghosts, *toks).arrange(RIGHT, buff=0.12).move_to([0.3, 0.95, 0])
        in_lab = M(r"x", 40, INK).next_to(in_row, LEFT, buff=0.35)
        outs = VGroup(*[t.copy() for t in toks])
        for o, t in zip(outs, toks):
            o.move_to([t.get_x(), -1.3, 0])
        out_lab = M(r"\tilde x", 40, INK).next_to(outs, LEFT, buff=0.35)
        pad_lab = L("zero padding", 22, MUTED).next_to(VGroup(*ghosts), DOWN, buff=0.2)

        def window_at(t):
            """window over row items t..t+3 (item t+3 = output position)."""
            items = in_row[t:t + 4]
            return SurroundingRectangle(items, buff=0.08).round_corners(0.1).set_stroke(INK, 3)

        window = window_at(0)
        chips = VGroup(*[weight_chip("?", j) for j in (3, 2, 1, 0)])  # values land on "initialized to the identity"
        for c, item in zip(chips, in_row[0:4]):
            c.shift([item.get_x() - c.box.get_x(), -0.05 - c.box.get_y(), 0])

        eq = M(r"\tilde x_t = \sum_{j=0}^{3} w_j \odot x_{t-j}", 44, t2c={"w_j": WARN})
        eq.move_to([-2.4, 2.3, 0])
        eq_notes = VGroup(
            L("depthwise: one weight per feature and lag", 24, MUTED),
            L("one filter, used at every layer", 24, MUTED),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT).next_to(eq, RIGHT, buff=0.6)
        same = L("identity at init: the output equals the input", 24, GOOD).move_to([0.6, -2.2, 0])

        with self.voiceover(
            "Two more parts of the setup. Since Phase Four, most arms include a short causal convolution, the kind "
            "Mamba and most modern linear-attention models place before their token mixer. It is a depthwise filter "
            "four tokens wide, applied at every layer and initialized to the identity, so at first it changes "
            "nothing. It feeds the queries and the values; the residual stream and the gate's input are not convolved."
        ) as vo:
            self.play(LaggedStartMap(FadeIn, parts, shift=0.2 * UP, lag_ratio=0.4), run_time=1.2)
            # --- sentence 1: "Since Phase Four, most arms ..." then the usual layout
            self.at(vo, 1)
            title1 = section_title("A short causal convolution")
            self.play(FadeOut(part2, shift=0.3 * RIGHT), FadeOut(title, shift=0.2 * UP),
                      FadeIn(title1, shift=0.2 * UP), ReplacementTransform(part1, conv_blk), run_time=1.0)
            title = title1
            self.play(FadeIn(blk_tag, shift=0.1 * UP), FadeIn(src1), run_time=0.6)
            self.at(vo, 1, 0.4)
            self.play(FadeIn(tok_lab), GrowArrow(blk_arrows[0]), GrowArrow(blk_arrows[1]), FadeIn(mix_blk),
                      GrowArrow(blk_arrows[2]), FadeIn(more_lab), run_time=1.0)
            self.play(FadeIn(blk_head, shift=0.1 * DOWN), run_time=0.6)
            self.at(vo, 1, 0.8)
            self.play(FlashAround(mix_blk, color=MEMORY_COLOR), mix_blk.rect.animate.set_fill(MEMORY_COLOR, 0.18),
                      run_time=0.9)
            # --- sentence 2: the filter
            self.at(vo, 2)
            self.play(FadeOut(VGroup(tok_lab, blk_arrows, mix_blk, more_lab, blk_head, blk_tag)),
                      ReplacementTransform(conv_blk.rect, window), FadeOut(conv_blk.txt), FadeIn(in_row),
                      FadeIn(in_lab),
                      Write(eq), run_time=1.2)
            self.play(LaggedStartMap(FadeIn, chips, shift=0.1 * DOWN, lag_ratio=0.15), FadeIn(pad_lab),
                      FadeIn(eq_notes[0]), run_time=0.9)
            self.at(vo, 2, 0.34)
            self.play(FadeIn(eq_notes[1], shift=0.1 * DOWN), run_time=0.6)
            self.at(vo, 2, 0.54)
            init_vals = [L(v, 22, INK, weight="BOLD" if v == "1" else "NORMAL").move_to(c.num)
                         for v, c in zip(("0", "0", "0", "1"), chips)]
            self.play(*[Transform(c.num, v) for c, v in zip(chips, init_vals)],
                      chips[3].box.animate.set_fill(WARN, 0.25).set_stroke(WARN, 2),
                      FadeIn(out_lab), FadeIn(same, shift=0.1 * UP), run_time=0.6)
            win_group = VGroup(window, chips)
            for t in range(6):
                if t > 0:
                    self.play(win_group.animate.shift((in_row[t + 3].get_x() - in_row[t + 2].get_x()) * RIGHT),
                              run_time=0.22)
                self.play(TransformFromCopy(toks[t], outs[t]), run_time=0.22)
            self.play(Indicate(same, color=GOOD, scale_factor=1.05), run_time=0.6)
            # --- sentence 3: where it is inserted
            self.at(vo, 3)
            flow = self.build_flow()
            self.play(FadeOut(VGroup(in_row, in_lab, outs, out_lab, pad_lab, chips, eq, eq_notes, same)),
                      ReplacementTransform(window, flow.conv.rect), FadeIn(flow.conv.txt), FadeIn(flow.x_node),
                      GrowArrow(flow.a_xc), GrowArrow(flow.a_cx), FadeIn(flow.xt), run_time=0.8)
            self.bring_to_front(flow.conv.txt)
            self.play(GrowArrow(flow.a_q), GrowArrow(flow.a_v), FadeIn(flow.qk), FadeIn(flow.v), run_time=0.6)
            self.play(FadeIn(flow.attn), GrowArrow(flow.a_qa), GrowArrow(flow.a_va),
                      FlashAround(flow.qk, color=WARN), FlashAround(flow.v, color=WARN), run_time=0.8)
            self.at(vo, 3, 0.42)
            self.play(ShowCreation(flow.res_path), FadeIn(flow.plus), GrowArrow(flow.a_ap), FadeIn(flow.out),
                      FadeIn(flow.res_lab), run_time=0.8)
            self.at(vo, 3, 0.66)
            self.play(FadeIn(flow.emb), GrowArrow(flow.a_eg), FadeIn(flow.gate), GrowArrow(flow.a_ga),
                      FadeIn(flow.gg), FadeIn(flow.gate_lab), run_time=0.8)
        self.wait(0.6)

        # ================================================================ N5.2 lag two
        tri1 = triple(1, 0, 13)
        tri2 = triple(0, 0, 2)
        dots = L("…", 30, MUTED)
        qry = VGroup(token("CTX0", "ctx", 0), token("K0"), token("?", "query")).arrange(RIGHT, buff=0.06)
        row = VGroup(tri1, tri2, dots, qry).arrange(RIGHT, buff=0.45).move_to([0, 1.8, 0])
        ctx0, k0, v2 = tri2
        v13 = tri1[2]
        win2 = SurroundingRectangle(VGroup(v13, v2), buff=0.08).round_corners(0.1).set_stroke(INK, 3)
        chips2 = VGroup(*[weight_chip(v, j) for v, j in (("0", 3), ("0", 2), ("0", 1), ("1", 0))])
        for c, item in zip(chips2, (v13, ctx0, k0, v2)):
            c.shift([item.get_x() - c.box.get_x(), 1.0 - c.box.get_y(), 0])
        chips2[3].box.set_fill(WARN, 0.25).set_stroke(WARN, 2)
        lag_arc = CurvedArrow(ctx0.get_top() + 0.05 * UP, v2.get_top() + 0.05 * UP, angle=-PI / 2)
        lag_arc.set_color(B0).set_stroke(width=4)
        lag_lab = L("lag 2", 24, B0, weight="BOLD").next_to(lag_arc, UP, buff=0.08)
        w2_new = L("0.16", 22, B0, weight="BOLD").move_to(chips2[1].num)
        # what the filter writes at V2's position: one term per lag, in the order of the chips
        terms = VGroup()
        for lag, src in ((3, v13), (2, ctx0), (1, k0), (0, v2)):
            col = B0 if lag == 2 else MUTED
            op = M((r"" if lag == 3 else r"+\,") + rf"w_{lag}\odot", 32, col)
            terms.add(VGroup(op, src.copy()).arrange(RIGHT, buff=0.1))
        lhs = M(r"\tilde x_t =", 34)
        eqrow = VGroup(lhs, *terms).arrange(RIGHT, buff=0.18).move_to([0, -0.3, 0])
        eq_hl = SurroundingRectangle(terms[1], buff=0.08).round_corners(0.08).set_stroke(B0, 2.5)
        eq_lab = L("t = the value's position: the context now sits there", 24, B0).next_to(eqrow, DOWN, buff=0.3)
        # the measured lag-2 weight (README §7, pre-registered M1, machine X), as a card beside the chips
        lag_rows = VGroup(
            VGroup(L("runs that bound", 22, INK), L("0.158", 22, B0, weight="BOLD")),
            VGroup(L("runs that did not", 22, INK), L("0.082", 22, MUTED, weight="BOLD")),
        )
        for r_ in lag_rows:
            r_[1].move_to([2.4, 0, 0], aligned_edge=LEFT)
            r_[0].move_to([0, 0, 0], aligned_edge=LEFT)
        lag_rows.arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        lag_note = VGroup(
            L("lag-2 weight after training", 22, B0, weight="BOLD"),
            lag_rows,
            VGroup(L("median of mean |w|, one channel + conv.,", 22, MUTED),
                   L("lr 4×10⁻³, machine X (test M1)", 22, MUTED)).arrange(DOWN, buff=0.08, aligned_edge=LEFT),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        lag_card = card(lag_note, buff=0.2)
        lag_card.move_to([4.15, 1.3, 0])
        src_lag = source_note("README §7 (M1); test_conv_lr.py, results on X")

        sees = VGroup(mini_ctx_key(1), mini_ctx_key(0), mini_ctx_key(0))
        # values: the pair sits at the value's position; the query: read at its key, it asks with the same pair
        for s_, x_ in zip(sees, (v13.get_x(), v2.get_x(), qry[:2].get_x())):
            s_.move_to([x_, 0.95, 0])
        m_yes = CurvedArrow(sees[2].get_bottom() + 0.05 * DOWN, sees[1].get_bottom() + 0.05 * DOWN, angle=-PI / 3)
        m_yes.set_color(GOOD).set_stroke(width=3)
        yes_lab = L("match", 22, GOOD, weight="BOLD").next_to(m_yes, DOWN, buff=0.06)
        no_lab = L("no match", 22, BAD, weight="BOLD").next_to(sees[0], DOWN, buff=0.15)
        sees_lab = VGroup(
            L("with lag 2, each value position carries its own (context, key) pair;", 22, MUTED),
            L("the query is that same pair, so it matches only its own value", 22, MUTED),
        ).arrange(DOWN, buff=0.1)
        sees_lab.move_to([0, -0.55, 0])
        route_txt = T("a second route: one channel, the streams never separated", 26, INK)
        route_txt.move_to([0, -1.25, 0])
        one_ch = frac_bar("one channel + convolution, lr 4×10⁻³", 34, 80, color=WARN, width=3.6, label_width=4.9,
                          size=22)
        one_ch.move_to([0, -1.95, 0])
        gate_take = VGroup(gate_bar([0.5, 0.5], width=0.9, height=0.26),
                           L("a gated model can take it too", 24, GATE_COLOR)).arrange(RIGHT, buff=0.25)
        gate_take.move_to([0, -2.7, 0])
        src2 = source_note("Report §2; README §7 and Table 5 (S = 2, P = 4, machines X and L)")

        with self.voiceover(
            "In this task each context token sits two positions before its value, so the convolution's lag-two "
            "weight can carry the context right onto the value's position. That gives even a single-channel model a "
            "second route to binding, one that never separates the streams, and a gated model can take it too."
        ) as vo:
            title_r = section_title("A second route to binding")
            self.play(FadeOut(flow, shift=0.4 * DOWN), FadeOut(src1), FadeOut(title, shift=0.2 * UP),
                      FadeIn(title_r, shift=0.2 * UP), FadeIn(tri1, shift=0.2 * DOWN), FadeIn(tri2, shift=0.2 * DOWN),
                      run_time=0.9)
            title = title_r
            self.play(FadeIn(win2), LaggedStartMap(FadeIn, chips2, shift=0.1 * DOWN, lag_ratio=0.1),
                      tri1[:2].animate.set_opacity(0.35), run_time=0.9)
            self.at(vo, 0, 0.25)
            self.play(FlashAround(ctx0, color=B0), FlashAround(v2, color=B0), run_time=0.9)
            self.play(ShowCreation(lag_arc), FadeIn(lag_lab, shift=0.1 * DOWN), run_time=0.9)
            self.at(vo, 0, 0.45)
            # after training: the lag-2 weight has grown; the other lags are not the point here
            self.play(Transform(chips2[1].num, w2_new), chips2[1].box.animate.set_fill(B0, 0.2).set_stroke(B0, 2.5),
                      *[chips2[i].animate.set_opacity(0.35) for i in (0, 2, 3)], run_time=0.8)
            self.play(FadeIn(lag_card, shift=0.2 * LEFT), FadeIn(src_lag), run_time=0.7)
            self.play(FadeIn(lhs), *[FadeIn(t[0]) for t in terms],
                      *[TransformFromCopy(src, t[1]) for src, t in zip((v13, ctx0, k0, v2), terms)], run_time=1.2)
            self.play(ShowCreation(eq_hl), FadeIn(eq_lab, shift=0.1 * UP), run_time=0.7)
            # --- sentence 1: the second route
            self.at(vo, 1)
            self.play(FadeOut(VGroup(chips2, win2, lag_card, lag_arc, lag_lab, eqrow, eq_hl, eq_lab)),
                      tri1[:2].animate.set_opacity(1), run_time=0.5)
            self.play(FadeIn(dots), FadeIn(qry, shift=0.2 * LEFT), run_time=0.5)
            self.play(LaggedStart(*[FadeIn(s_, shift=0.2 * DOWN) for s_ in sees], lag_ratio=0.3),
                      FadeIn(sees_lab), run_time=1.2)
            self.play(ShowCreation(m_yes), FadeIn(yes_lab), sees[1][0].animate.set_stroke(GOOD, 2.5),
                      sees[2][0].animate.set_stroke(GOOD, 2.5), run_time=0.7)
            self.play(FadeIn(no_lab, shift=0.1 * DOWN), sees[0][0].animate.set_stroke(BAD, 2.5), run_time=0.6)
            ans = token("V2", "val", 0).move_to(qry[2])
            self.play(Transform(qry[2], ans), Write(route_txt), run_time=1.0)
            self.play(FadeIn(one_ch.name_mob), FadeIn(one_ch.track), FadeOut(src_lag), FadeIn(src2), run_time=0.4)
            self.play(*grow_bar(one_ch, run_time=0.8))
            self.at(vo, 1, 0.82)
            self.play(FadeIn(gate_take, shift=0.1 * UP), run_time=0.7)

        # ---------------------------------------------------------------- gate + conv: bound vs routed
        rows_data = [("gate + convolution, lr 10⁻³", 51, 40), ("gate + convolution, lr 4×10⁻³", 50, 21)]
        bar_w, den = 5.0, 80
        split_rows = VGroup()
        for lab, nb, nd in rows_data:
            name = L(lab, 22, INK)
            track = Rectangle(width=bar_w, height=0.4).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1)
            fb = Rectangle(width=bar_w * nb / den, height=0.4).set_fill(MUTED, 0.75).set_stroke(width=0)
            fd = Rectangle(width=bar_w * nd / den, height=0.4).set_fill(GOOD, 0.95).set_stroke(width=0)
            fb.align_to(track, LEFT)
            fd.align_to(track, LEFT)
            vb = L(f"{nb}/80 bound", 22, INK)
            vd = L(f"{nd} discovered", 22, GOOD, weight="BOLD")
            vals = VGroup(vb, vd).arrange(RIGHT, buff=0.3)
            bar = VGroup(track, fb, fd)
            name.next_to(bar, LEFT, buff=0.3)
            vals.next_to(bar, RIGHT, buff=0.25)
            r = VGroup(name, bar, vals)
            r.name_mob, r.track, r.fb, r.fd, r.vb, r.vd = name, track, fb, fd, vb, vd
            split_rows.add(r)
        split_rows.arrange(DOWN, buff=0.35)
        for r in split_rows:  # align the bars
            r.shift((split_rows[0].track.get_left()[0] - r.track.get_left()[0]) * RIGHT)
        split_rows.move_to([0, -1.75, 0])
        legend = VGroup(
            Square(0.22).set_fill(MUTED, 0.75).set_stroke(width=0), L("bound", 22, MUTED),
            Square(0.22).set_fill(GOOD, 0.95).set_stroke(width=0), L("bound and streams separated (discovered)", 22, MUTED),
        ).arrange(RIGHT, buff=0.15)
        legend[2].shift(0.3 * RIGHT)
        legend[3].shift(0.3 * RIGHT)
        legend.next_to(split_rows, UP, buff=0.3).align_to(split_rows[0].track, LEFT)
        verdict = VGroup(L("with the convolution, score on", 26, INK), verdict_badge("ROUTING", size=24, color=GOOD),
                         L("not on binding alone", 26, INK)).arrange(RIGHT, buff=0.25)
        verdict.move_to([0, -3.1, 0])

        with self.voiceover(
            "That is why results with the convolution are scored on routing, not on binding alone."
        ) as vo:
            self.play(FadeOut(VGroup(route_txt, one_ch, gate_take, sees_lab, m_yes, yes_lab, no_lab)),
                      VGroup(row, sees).animate.shift(0.35 * UP), run_time=0.6)
            for r in split_rows:
                r.fb.save_state()
                r.fb.stretch(1e-3, 0, about_edge=LEFT)
                r.fd.save_state()
                r.fd.stretch(1e-3, 0, about_edge=LEFT)
            self.play(FadeIn(legend), *[FadeIn(VGroup(r.name_mob, r.track)) for r in split_rows], run_time=0.5)
            self.play(*[Restore(r.fb) for r in split_rows], *[FadeIn(r.vb) for r in split_rows], run_time=0.9)
            self.play(*[Restore(r.fd) for r in split_rows], *[FadeIn(r.vd) for r in split_rows], run_time=0.9)
            self.play(FlashAround(split_rows[1].vd, color=GOOD), FadeIn(verdict, shift=0.1 * UP), run_time=1.0)
        self.wait(1.2)

        self.play(FadeOut(VGroup(row, sees, legend, split_rows, verdict, src2)), run_time=0.7)

        # ================================================================ N5.3 Muon
        title2 = section_title("The optimizer: Muon")
        phase_x = [-4.0, -2.0, 0.0, 2.0, 4.0]
        tl = Line([-5, 1.4, 0], [5, 1.4, 0]).set_stroke(FAINT, 2)
        ticks = VGroup(*[Line(0.12 * UP, 0.12 * DOWN).set_stroke(MUTED, 2).move_to([x, 1.4, 0]) for x in phase_x])
        plabs = VGroup(*[L(f"Phase {r}", 24, MUTED).next_to(ticks[i], UP, buff=0.18)
                         for i, r in enumerate(["I", "II", "III", "IV", "V"])])
        adam_bar = Rectangle(width=8.0, height=0.46).set_fill(ADAM_COLOR, 0.85).set_stroke(width=0)
        adam_bar.move_to([-1.0, 0.75, 0])
        adam_txt = L("Adam: every main-line test", 22, BG, weight="BOLD").move_to(adam_bar)
        adam_v = Rectangle(width=2.0, height=0.46).set_fill(ADAM_COLOR, 0.4).set_stroke(width=0).move_to([4, 0.75, 0])
        muon_bar = Rectangle(width=2.0, height=0.46).set_fill(MUON_COLOR, 0.95).set_stroke(width=0)
        muon_bar.move_to([4, 0.15, 0])
        muon_txt = L("+ Muon", 22, BG, weight="BOLD").move_to(muon_bar)
        adam_v_txt = L("Adam", 22, INK, weight="BOLD").move_to(adam_v)
        timeline = VGroup(tl, ticks, plabs, adam_bar, adam_txt, adam_v, adam_v_txt, muon_bar, muon_txt)

        # matrix, singular values, Newton–Schulz
        Mm, Om = ns_matrices()
        hm = heatmap(Mm).move_to([-5.3, -0.55, 0])
        hm_o = heatmap(Om).move_to(hm)
        hm_lab = L("momentum M", 24, MUON_COLOR)
        hm_sub = L("of one weight matrix", 20, MUTED)
        VGroup(hm_lab, hm_sub).arrange(DOWN, buff=0.08).next_to(hm, UP, buff=0.2)
        svd_tex = M(r"M = U\,\Sigma\,V^{\top}", 32).next_to(hm, DOWN, buff=0.3)
        svd_after = M(r"U\,\Sigma'\,V^{\top}", 32).move_to(svd_tex)
        orth_lab = L("≈ orthogonalized", 22, MUON_COLOR).next_to(svd_after, DOWN, buff=0.15)

        X0, Y0, H = -2.55, -2.3, 3.2
        bar_xs = [X0 + 0.45 + i * 0.52 for i in range(8)]
        y_axis = Line([X0, Y0, 0], [X0, Y0 + H + 0.25, 0]).set_stroke(MUTED, 2)
        x_axis = Line([X0, Y0, 0], [X0 + 4.4, Y0, 0]).set_stroke(MUTED, 2)

        def yticks(vals, vmax):
            g = VGroup()
            for v in vals:
                y = Y0 + v / vmax * H
                tk = Line([X0 - 0.08, y, 0], [X0, y, 0]).set_stroke(MUTED, 2)
                tl_ = L(f"{v:g}", 20, MUTED).next_to(tk, LEFT, buff=0.08)
                g.add(VGroup(tk, tl_))
            return g

        ticks_raw = yticks([0, 10, 20], 20)
        ticks_norm = yticks([0, 0.5, 1], 1.25)

        def bars(vals, vmax, color=MUON_COLOR):
            g = VGroup()
            for x, v in zip(bar_xs, vals):
                h = max(v / vmax * H, 0.012)
                r = Rectangle(width=0.34, height=h).set_fill(color, 0.9).set_stroke(width=0)
                r.move_to([x, Y0 + h / 2, 0])
                g.add(r)
            return g

        traj = ns_trajectory()
        bars_raw = bars(SV0, 20)
        bar_vals = VGroup(*[L(s, 20, INK).next_to(b, UP, buff=0.08) for s, b in zip(SV0_LABELS, bars_raw)])
        chart_head = L("singular values σ of M", 22, MUTED).move_to([X0 + 2.2, Y0 + H + 0.6, 0])
        spread_lab = L("spread: 0.1 to 20", 22, INK).move_to([X0 + 2.9, Y0 + H - 0.9, 0])
        norm_tex = M(r"M \to \frac{M}{\lVert M \rVert_F}", 32, INK).move_to([X0 + 3.1, Y0 + H - 0.45, 0])
        band_y0, band_y1 = Y0 + 0.7 / 1.25 * H, Y0 + 1.23 / 1.25 * H
        band = Rectangle(width=4.3, height=band_y1 - band_y0).set_fill(MUON_COLOR, 0.13).set_stroke(MUON_COLOR, 1, 0.5)
        band.move_to([X0 + 2.2, (band_y0 + band_y1) / 2, 0])
        band_lab = L("near 1", 22, MUON_COLOR).next_to(band, RIGHT, buff=0.12)
        step_labs = [L(f"Newton–Schulz step {k} of 5", 22, MUON_COLOR).move_to(chart_head) for k in range(1, 6)]
        svd_arrow = Arrow([-4.15, -0.55, 0], [-3.15, -0.55, 0], buff=0, thickness=3, fill_color=MUTED)
        svd_arrow_lab = L("SVD", 20, MUTED).next_to(svd_arrow, UP, buff=0.08)

        ax = Axes(x_range=[0, 1.25, 0.25], y_range=[0, 1.25, 0.25], width=2.5, height=2.5,
                  axis_config=dict(stroke_color=MUTED, stroke_width=2, include_tip=False))
        ax.move_to([5.05, -0.95, 0])
        ax_ticks = VGroup(
            L("0", 20, MUTED).next_to(ax.c2p(0, 0), DL, buff=0.08),
            L("1", 20, MUTED).next_to(ax.c2p(1, 0), DOWN, buff=0.1),
            L("1", 20, MUTED).next_to(ax.c2p(0, 1), LEFT, buff=0.1),
        )
        ax_xl = M(r"\sigma", 30, MUTED).next_to(ax.c2p(1.25, 0), DOWN, buff=0.12)
        p_band = Rectangle(width=ax.c2p(1.25, 0)[0] - ax.c2p(0, 0)[0], height=ax.c2p(0, 1.23)[1] - ax.c2p(0, 0.7)[1])
        p_band.set_fill(MUON_COLOR, 0.13).set_stroke(width=0)
        p_band.move_to([(ax.c2p(0, 0)[0] + ax.c2p(1.25, 0)[0]) / 2, (ax.c2p(0, 0.7)[1] + ax.c2p(0, 1.23)[1]) / 2, 0])
        diag = DashedLine(ax.c2p(0, 0), ax.c2p(1.25, 1.25)).set_stroke(FAINT, 1.5)
        curve = ax.get_graph(ns_poly, x_range=[0, 1.25, 0.01]).set_stroke(MUON_COLOR, 3)
        p_tex = M(r"\sigma \mapsto a\sigma + b\sigma^3 + c\sigma^5", 30).move_to([4.85, 1.25, 0])
        p_coef = L("a, b, c = 3.4445, −4.775, 2.0315", 22, MUTED).next_to(p_tex, DOWN, buff=0.15)
        poly = VGroup(p_band, ax, ax_ticks, ax_xl, diag, curve, p_tex, p_coef)
        src3 = source_note("torch.optim._muon via test_early_recipe.py:231, 254-259 (MUON_KW); Report §1")

        def curve_dots(vals):
            return VGroup(*[Dot(ax.c2p(v, ns_poly(v)), radius=0.06, fill_color=INK) for v in vals])

        with self.voiceover(
            "Second, the optimizer. Before Phase Five every main-line test used Adam. Phase Five adds Muon. Muon "
            "takes the momentum of each weight matrix it handles and replaces it by an approximately orthogonalized "
            "version, computed with five Newton–Schulz steps. As a result, the size of an update no longer depends "
            "on the size of the gradient."
        ) as vo:
            part2b = card_part("2", "the Muon optimizer", MUON_COLOR).move_to([0, -1.3, 0])
            self.play(FadeOut(title, shift=0.2 * UP), FadeIn(title2, shift=0.2 * UP), FadeIn(part2b, shift=0.3 * UP),
                      run_time=1.0)
            title = title2
            self.at(vo, 1)
            self.play(ShowCreation(tl), FadeIn(ticks), LaggedStartMap(FadeIn, plabs, lag_ratio=0.15), run_time=1.0)
            adam_bar.save_state()
            adam_bar.stretch(1e-3, 0, about_edge=LEFT)
            self.play(Restore(adam_bar), FadeIn(adam_txt), run_time=1.0)
            self.at(vo, 2)
            self.play(FadeIn(adam_v), FadeIn(adam_v_txt), ReplacementTransform(part2b[0], muon_bar), FadeOut(part2b[1]),
                      run_time=0.9)
            self.play(FadeIn(muon_txt), run_time=0.3)
            self.play(Indicate(plabs[4], color=MUON_COLOR, scale_factor=1.15), FlashAround(muon_bar, color=MUON_COLOR),
                      run_time=0.8)
            # --- sentence 3: orthogonalize
            self.at(vo, 3)
            self.play(FadeOut(timeline, shift=0.3 * UP), FadeIn(hm), FadeIn(hm_lab), FadeIn(hm_sub), run_time=0.8)
            self.play(FadeIn(svd_tex), GrowArrow(svd_arrow), FadeIn(svd_arrow_lab), ShowCreation(y_axis),
                      ShowCreation(x_axis), FadeIn(ticks_raw), FadeIn(chart_head), FadeIn(src3), run_time=0.7)
            self.play(TransformFromCopy(hm[1], bars_raw), run_time=0.9)
            self.play(LaggedStartMap(FadeIn, bar_vals, lag_ratio=0.08), FadeIn(spread_lab), run_time=0.6)
            # normalise
            bars_cur = bars(traj[0], 1.25)
            self.play(FadeIn(norm_tex, shift=0.1 * DOWN), FadeOut(spread_lab), run_time=0.4)
            self.play(Transform(bars_raw, bars_cur), FadeOut(bar_vals), ReplacementTransform(ticks_raw, ticks_norm),
                      FadeIn(poly), run_time=0.9)
            self.play(FadeOut(norm_tex), FadeIn(band), FadeIn(band_lab), run_time=0.4)
            # five steps
            head = chart_head
            for k in range(5):
                d = curve_dots(traj[k])
                self.play(FadeIn(d, scale=0.5), Transform(head, step_labs[k]), run_time=0.3)
                self.play(Transform(bars_raw, bars(traj[k + 1], 1.25)), FadeOut(d), run_time=0.55)
            self.play(Transform(hm[1], hm_o[1]), ReplacementTransform(svd_tex, svd_after), FadeIn(orth_lab),
                      Indicate(band_lab, color=MUON_COLOR), run_time=1.0)
            # --- sentence 4: scale independence
            self.at(vo, 4)
            meter = self.build_meter()
            self.play(FadeOut(poly), run_time=0.5)
            self.play(FadeIn(meter.head_g), FadeIn(meter.g_rows), FadeIn(meter.head_u), FadeIn(meter.u_rows),
                      run_time=0.6)
            self.play(meter.g_rows[1][1].animate.stretch(10, 0, about_edge=LEFT), FadeIn(meter.x10), run_time=0.9)
            self.play(FadeIn(meter.same, shift=0.1 * LEFT), Indicate(meter.u_rows, color=GOOD, scale_factor=1.04),
                      FadeIn(meter.why), run_time=0.9)

        # ================================================================ N5.4 who Muon handles
        self.play(FadeOut(VGroup(hm, hm_lab, hm_sub, svd_after, orth_lab, svd_arrow, svd_arrow_lab, y_axis, x_axis,
                                 ticks_norm, bars_raw, band, band_lab, head, meter)), FadeOut(src3), run_time=0.7)

        muon_chips = VGroup(
            param_chip("encoder", "1 × 32 × 256"), param_chip("encoder_v", "1 × 32 × 256"),
            param_chip("decoder", "256 × 32"), param_chip("lm_head", "32 × V"),
            param_chip("W_in", "32 × 32", GATE_COLOR), param_chip("W_h", "32 × 32", GATE_COLOR),
            param_chip("W_g", "k × 32", GATE_COLOR),
        )
        adam_chips = VGroup(param_chip("embed", "V × 32"), param_chip("short_conv", "32 × 4"),
                            param_chip("1-D params", "none here", MUTED, dashed=True))
        pool = VGroup(*muon_chips, *adam_chips).arrange_in_grid(2, 5, h_buff=0.22, v_buff=0.25)
        pool.move_to([0, 1.95, 0])
        pool_lab = L("the model's parameters", 22, MUTED).next_to(pool, UP, buff=0.15)

        muon_bin = RoundedRectangle(width=8.1, height=3.5, corner_radius=0.2)
        muon_bin.set_fill(MUON_COLOR, 0.05).set_stroke(MUON_COLOR, 2).move_to([-2.35, -1.25, 0])
        adam_bin = RoundedRectangle(width=4.3, height=3.5, corner_radius=0.2)
        adam_bin.set_fill(ADAM_COLOR, 0.05).set_stroke(ADAM_COLOR, 2).move_to([4.35, -1.25, 0])
        muon_head = L("Muon: weight matrices", 24, MUON_COLOR, weight="BOLD")
        muon_head.next_to(muon_bin.get_top(), DOWN, buff=0.2)
        adam_head = L("Adam", 24, ADAM_COLOR, weight="BOLD").next_to(adam_bin.get_top(), DOWN, buff=0.2)

        muon_slots = VGroup(*[c.copy() for c in muon_chips]).arrange_in_grid(2, 4, h_buff=0.15, v_buff=0.25)
        muon_slots.next_to(muon_head, DOWN, buff=0.25)
        adam_slots = VGroup(*[c.copy() for c in adam_chips]).arrange_in_grid(2, 2, h_buff=0.15, v_buff=0.25)
        adam_slots.next_to(adam_head, DOWN, buff=0.25)
        slice_note = L("a 3-D weight goes one head slice at a time", 20, MUTED)
        slice_note.next_to(muon_slots, DOWN, buff=0.22)
        # outline the two 3-D weights in the bin
        slice_marks = VGroup(*[SurroundingRectangle(muon_slots[i], buff=0.05).round_corners(0.1)
                               .set_stroke(MUON_COLOR, 2) for i in (0, 1)])
        src4 = source_note("test_early_recipe.py:28-39, 382-384 (is_adam); Report §1")

        with self.voiceover(
            "Muon handles the weight matrices, each head slice of a three-dimensional weight included. The embedding "
            "and the convolution stay with Adam, along with any one-dimensional parameters."
        ) as vo:
            self.play(LaggedStartMap(FadeIn, pool, shift=0.1 * DOWN, lag_ratio=0.06), FadeIn(pool_lab),
                      ShowCreation(muon_bin), ShowCreation(adam_bin), FadeIn(muon_head), FadeIn(adam_head),
                      FadeIn(src4), run_time=1.2)
            self.bring_to_front(*reversed(list(muon_chips)))  # earlier movers drawn on top
            self.play(LaggedStart(*[muon_chips[i].animate.move_to(muon_slots[i]) for i in range(7)], lag_ratio=0.1),
                      run_time=1.6)
            self.play(ShowCreation(slice_marks), FadeIn(slice_note, shift=0.1 * UP), run_time=0.8)
            self.at(vo, 1)
            self.play(LaggedStart(*[adam_chips[i].animate.move_to(adam_slots[i]) for i in range(2)], lag_ratio=0.3),
                      run_time=1.2)
            self.at(vo, 1, 0.62)
            self.play(adam_chips[2].animate.move_to(adam_slots[2]), FadeOut(pool_lab), run_time=0.9)
            bins = VGroup(muon_bin, adam_bin, muon_head, adam_head, muon_chips, adam_chips, slice_marks, slice_note)
            self.play(bins.animate.shift(0.9 * UP), run_time=0.8)
        self.wait(1.0)
        self.clear_all()

    # ---------------------------------------------------------------- helpers that build larger pictures
    def build_flow(self):
        """One layer's data flow: the convolution feeds queries (= keys) and values only."""
        y = 0.55
        f = VGroup()
        f.x_node = VGroup(Circle(radius=0.34).set_stroke(INK, 2).set_fill(PANEL, 1), M(r"x", 34))
        f.x_node.move_to([-5.7, y, 0])
        f.conv = box_label("conv", 1.5, 0.75, INK, 26)
        f.conv.move_to([-3.75, y, 0])
        f.xt = M(r"\tilde x", 36).move_to([-2.35, y, 0])
        f.qk = box_label("queries = keys", 2.55, 0.65, INK, 22).move_to([0.05, y + 0.85, 0])
        f.v = box_label("values", 2.55, 0.65, INK, 22).move_to([0.05, y - 0.85, 0])
        f.attn = box_label("attention", 1.9, 0.95, MEMORY_COLOR, 24).move_to([2.85, y, 0])
        f.plus = VGroup(Circle(radius=0.26).set_stroke(INK, 2).set_fill(PANEL, 1), M(r"+", 32))
        f.plus.move_to([4.75, y, 0])
        f.out = M(r"x + y", 32).next_to(f.plus, RIGHT, buff=0.25)

        def arr(a, b, color=INK):
            return Arrow(a, b, buff=0.08, thickness=3, fill_color=color)

        f.a_xc = arr(f.x_node.get_right(), f.conv.get_left())
        f.a_cx = arr(f.conv.get_right(), f.xt.get_left())
        f.a_q = arr(f.xt.get_right() + 0.05 * UP, f.qk.get_left())
        f.a_v = arr(f.xt.get_right() + 0.05 * DOWN, f.v.get_left())
        f.a_qa = arr(f.qk.get_right(), f.attn.get_left() + 0.2 * UP)
        f.a_va = arr(f.v.get_right(), f.attn.get_left() + 0.2 * DOWN)
        f.a_ap = arr(f.attn.get_right(), f.plus.get_left())
        top = 2.55
        f.res_path = VMobject().set_points_as_corners([
            f.x_node.get_top(), [f.x_node.get_x(), top, 0], [f.plus.get_x(), top, 0], f.plus.get_top() + 0.06 * UP,
        ]).set_stroke(MUTED, 3)
        f.res_lab = L("residual stream: not convolved", 22, MUTED).move_to([0.0, top + 0.27, 0])
        gy = -1.55
        f.emb = L("token embeddings", 22, MUTED).move_to([-5.2, gy, 0])
        f.gate = box_label("gate", 1.3, 0.6, GATE_COLOR, 24).move_to([2.85, gy, 0])
        f.a_eg = arr(f.emb.get_right(), f.gate.get_left(), MUTED)
        f.a_ga = arr(f.gate.get_top(), f.attn.get_bottom(), GATE_COLOR)
        f.gg = M(r"g_t\cdot g_s", 28, GATE_COLOR).next_to(f.a_ga, RIGHT, buff=0.15)
        f.gate_lab = L("the gate's input: not convolved", 22, MUTED).next_to(f.a_eg, DOWN, buff=0.15)
        f.add(f.x_node, f.conv, f.xt, f.qk, f.v, f.attn, f.plus, f.out, f.a_xc, f.a_cx, f.a_q, f.a_v, f.a_qa,
              f.a_va, f.a_ap, f.res_path, f.res_lab, f.emb, f.gate, f.a_eg, f.a_ga, f.gg, f.gate_lab)
        return f

    def build_meter(self):
        """Gradient ×1 vs ×10 against the size of Muon's update."""
        m = VGroup()
        x0 = 3.55
        m.head_g = L("size of the gradient", 22, MUTED).move_to([4.75, 0.95, 0])

        def row(lab, length, color, y):
            name = L(lab, 22, INK)
            bar = Rectangle(width=length, height=0.32).set_fill(color, 0.9).set_stroke(width=0)
            bar.move_to([x0 + length / 2, y, 0])
            name.next_to([x0, y, 0], LEFT, buff=0.2)
            return VGroup(name, bar)

        m.g_rows = VGroup(row("G", 0.22, KEY_COLOR, 0.45), row("10 G", 0.22, KEY_COLOR, -0.05))
        m.x10 = L("× 10", 22, WARN, weight="BOLD").next_to([x0 + 2.2, -0.05, 0], RIGHT, buff=0.15)
        m.head_u = L("size of Muon's update", 22, MUON_COLOR).move_to([4.75, -0.75, 0])
        m.u_rows = VGroup(row("G", 1.5, MUON_COLOR, -1.25), row("10 G", 1.5, MUON_COLOR, -1.75))
        m.same = L("the same", 24, GOOD, weight="BOLD").next_to(m.u_rows, RIGHT, buff=0.3)
        m.why = VGroup(L("dividing by ‖M‖ first", 22, MUTED), L("cancels the scale", 22, MUTED))
        m.why.arrange(DOWN, buff=0.08).move_to([4.75, -2.5, 0])
        m.add(m.head_g, m.g_rows, m.x10, m.head_u, m.u_rows, m.same, m.why)
        return m


def card_part(num, text, color):
    n = L(num, 30, MUTED, weight="BOLD")
    t = L(text, 30, color)
    g = VGroup(n, t).arrange(RIGHT, buff=0.3)
    return card(g, buff=0.35)
