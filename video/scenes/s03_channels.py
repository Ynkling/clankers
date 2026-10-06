"""Chapter 3 — channels and a gate (report §1, eqs. (1)-(2); §13, Table 8; facts_model §2)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403
from manimlib.config import manim_config  # noqa: E402

# ManimGL compiles every Tex through one shared working.tex, so scenes rendered in parallel can swap each
# other's SVGs (and the swap is then cached on disk). Compile this scene's Tex in a private directory, and
# salt the source with a trailing empty group so no entry cached by a racing process is reused.
manim_config.directories.latex_cache = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "tex_work", "s03_channels")
os.makedirs(manim_config.directories.latex_cache, exist_ok=True)


def M(tex: str, size: float = 40, color=INK, **kw) -> Tex:  # noqa: F811  (salted local version)
    return Tex(tex + "{}", font_size=size, fill_color=color, **kw)


B, Y = STREAM_COLORS[0], STREAM_COLORS[1]
IN_COLOR = MEMORY_COLOR      # the gate's input term (teal), reused in Ch. 19
REC_COLOR = GATE_COLOR       # the gate's recurrent term (purple), reused in Ch. 19

# First training sequence of seed 280, as in Ch. 2: (stream, key, value), grouped by key.
SEQ = [(1, 0, 13), (0, 0, 2), (0, 3, 11), (1, 3, 2), (0, 2, 10), (1, 2, 6), (0, 1, 12), (1, 1, 6)]
QUERY = (0, 0)
ANSWER = 2
# the K0 group plus the query's context and key: the tokens the gate-match matrix is drawn over
MAT_TOKENS = [("CTX1", "ctx", 1), ("K0", "key", 1), ("V13", "val", 1),
              ("CTX0", "ctx", 0), ("K0", "key", 0), ("V2", "val", 0),
              ("CTX0", "ctx", 0), ("K0", "key", 0)]


# ---------------------------------------------------------------- local helpers
def pos_token(sym: str, color=KEY_COLOR) -> VGroup:
    """A token box named by its position (s or t), to match the symbols of eq. (1)."""
    box = RoundedRectangle(width=0.9, height=0.62, corner_radius=0.1).set_fill(PANEL, 1).set_stroke(color, 2)
    lab = M(sym, 40, color=color).move_to(box)
    return VGroup(box, lab)


def gate_with_numbers(probs, width=1.8, height=0.32, size=22) -> VGroup:
    bar = gate_bar(probs, width=width, height=height)
    nums = VGroup(*[L(f"{p:g}", size=size, color=STREAM_COLORS[i]).next_to(bar[0][i], DOWN, buff=0.12)
                    for i, p in enumerate(probs)])
    g = VGroup(bar, nums)
    g.bar, g.nums = bar, nums
    return g


def grow_gate(bar: VGroup, run_time=0.9):
    """Animations that build a gate_bar: frame first, then segments growing left to right."""
    return [ShowCreation(bar[1], run_time=run_time),
            LaggedStart(*[GrowFromEdge(s, LEFT) for s in bar[0]], lag_ratio=0.5, run_time=run_time)]


def vec_cells(values, colors, cell=0.48, size=24) -> VGroup:
    """A small vector drawn as cells: a non-zero entry is filled in its channel's color."""
    g = VGroup()
    for v, c in zip(values, colors):
        sq = Square(cell)
        if v:
            sq.set_fill(c, 0.85).set_stroke(c, 1.5)
            t = L(f"{v:g}", size=size, color=BG, weight="BOLD")
        else:
            sq.set_fill(PANEL, 1).set_stroke(FAINT, 1.5)
            t = L("0", size=size, color=MUTED)
        t.move_to(sq)
        g.add(VGroup(sq, t))
    g.arrange(RIGHT, buff=0)
    return g


def matrix_cell(value: int, color, cell: float) -> VGroup:
    sq = Square(cell)
    if value:
        sq.set_fill(color, 0.75).set_stroke(color, 1)
        t = L("1", size=20, color=BG, weight="BOLD")
    else:
        sq.set_fill(PANEL, 1).set_stroke(FAINT, 1)
        t = L("0", size=20, color=MUTED)
    return VGroup(sq, t.move_to(sq))


def channel_panel(c: int, entries) -> VGroup:
    """One channel as a panel listing the key→value bindings it stores."""
    color = STREAM_COLORS[c]
    icon = memory_grid(3, 3, cell=0.12, color=color)
    head = VGroup(icon, L(f"channel {c}", 24, color, weight="BOLD")).arrange(RIGHT, buff=0.2)
    ents = VGroup()
    for k, v in entries:
        arrow = M(R"\rightarrow", 30, color=MUTED)
        ents.add(VGroup(token(f"K{k}", width=0.8, height=0.46), arrow,
                        token(f"V{v}", "val", c, width=0.8, height=0.46)).arrange(RIGHT, buff=0.1))
    ents.arrange_in_grid(2, 2, h_buff=0.45, v_buff=0.16)
    inner = VGroup(head, ents).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
    rect = SurroundingRectangle(inner, buff=0.22).round_corners(0.12)
    rect.set_fill(PANEL, 1).set_stroke(color, 2)
    g = VGroup(rect, head, ents)
    g.rect, g.head, g.ents = rect, head, ents
    return g


class ChannelsAndGate(ClankersScene):
    def construct(self):
        chap = self.chapter_card(3, "Channels and a gate")
        self.wait(0.6)
        title = section_title("Channels and a gate")
        self.play(FadeOut(chap, shift=0.3 * UP), FadeIn(title, shift=0.3 * UP))

        # ================================================================ N3.1 k channels, one gate per token
        one = memory_grid(5, 5, cell=0.3, color=MEMORY_COLOR).move_to(0.1 * DOWN)
        one_lab = L("one Hebbian memory per synapse", 22, MEMORY_COLOR).next_to(one, DOWN, buff=0.2)
        ch0 = memory_grid(5, 5, cell=0.3, color=B).move_to(1.25 * UP)
        ch1 = memory_grid(5, 5, cell=0.3, color=Y).move_to(1.35 * DOWN)
        lab0 = L("channel 0", 22, B, weight="BOLD").next_to(ch0, UP, buff=0.15)
        lab1 = L("channel 1", 22, Y, weight="BOLD").next_to(ch1, DOWN, buff=0.15)
        shared = VGroup(L("all weights shared", 20, MUTED, weight="BOLD"),
                        L("encoder · encoder_v · decoder · …", 20, MUTED)).arrange(DOWN, buff=0.08)
        shared.move_to((ch0.get_bottom() + ch1.get_top()) / 2)
        brace = Brace(VGroup(ch0, ch1), RIGHT, buff=0.3)
        # the shared-weights label is wider than the grids: keep the brace clear of it
        brace.shift(max(0.0, shared.get_right()[0] + 0.2 - brace.get_left()[0]) * RIGHT)
        brace_lab = VGroup(L("k separate", 24), L("Hebbian memories", 24),
                           L("(k = 2 here)", 22, MUTED)).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        brace_lab.next_to(brace, RIGHT, buff=0.2)

        # writer s (left) and reader t (right)
        tok_s = pos_token("s").move_to([-4.9, 0.6, 0])
        cap_s = L("earlier token", 22, MUTED).next_to(tok_s, UP, buff=0.15)
        g_s = gate_with_numbers([0.8, 0.2]).next_to(tok_s, DOWN, buff=0.3)
        g_s_lab = M("g_s", 36, GATE_COLOR).next_to(g_s.bar, LEFT, buff=0.2)
        g_s_sum = M(R"0.8 + 0.2 = 1", 28, MUTED).next_to(g_s, DOWN, buff=0.2)
        tok_t = pos_token("t").move_to([4.9, 0.6, 0])
        cap_t = L("later token", 22, MUTED).next_to(tok_t, UP, buff=0.15)
        g_t = gate_with_numbers([0.3, 0.7]).next_to(tok_t, DOWN, buff=0.3)
        g_t_lab = M("g_t", 36, GATE_COLOR).next_to(g_t.bar, RIGHT, buff=0.2)
        g_t_sum = M(R"0.3 + 0.7 = 1", 28, MUTED).next_to(g_t, DOWN, buff=0.2)

        w_start = g_s.bar.get_right() + 0.1 * RIGHT
        r_end = g_t.bar.get_left() + 0.1 * LEFT
        w0 = Arrow(w_start, ch0.get_left() + 0.1 * LEFT, buff=0.05, thickness=5).set_fill(B, 1)
        w1 = Arrow(w_start, ch1.get_left() + 0.1 * LEFT, buff=0.05, thickness=2).set_fill(Y, 0.6)
        r0 = Arrow(ch0.get_right() + 0.1 * RIGHT, r_end, buff=0.05, thickness=2.5).set_fill(B, 0.7)
        r1 = Arrow(ch1.get_right() + 0.1 * RIGHT, r_end, buff=0.05, thickness=4.5).set_fill(Y, 1)
        w0_lab = L("× 0.8", 22, B).next_to(w0.get_center(), UL, buff=0.08)
        w1_lab = L("× 0.2", 22, Y).next_to(w1.get_center(), DL, buff=0.08)
        r0_lab = L("× 0.3", 22, B).next_to(r0.get_center(), UR, buff=0.08)
        r1_lab = L("× 0.7", 22, Y).next_to(r1.get_center(), DR, buff=0.08)
        write_tag = L("write", 22, INK, weight="BOLD").next_to(w0_lab, UP, buff=0.12)
        read_tag = L("read", 22, INK, weight="BOLD").next_to(r0_lab, UP, buff=0.12)
        # the write's footprint (an outer product: rows x cols), stored at the gate's weight in each channel
        pattern = [r * 5 + c for r in (1, 2, 4) for c in (0, 1, 3)]
        fill0 = VGroup(*[ch0.cells[i].copy().set_fill(B, 0.8).set_stroke(width=0) for i in pattern])
        fill1 = VGroup(*[ch1.cells[i].copy().set_fill(Y, 0.2).set_stroke(width=0) for i in pattern])
        src1 = source_note("Report §1; gate values illustrative")

        with self.voiceover(
            "The extension gives every synapse k channels instead of one: k separate Hebbian memories that share "
            "all the network's weights. Every token also gets a gate, a probability distribution over the k "
            "channels. The gate spreads the token's write across the channels and blends its read from them."
        ) as vo:
            self.play(FadeIn(one), FadeIn(one_lab, shift=0.1 * UP), run_time=1.0)
            vo.wait_until(1.7)
            one_copy = one.copy()
            self.play(ReplacementTransform(one, ch0), ReplacementTransform(one_copy, ch1),
                      FadeOut(one_lab), run_time=1.3)
            self.play(FadeIn(lab0, shift=0.1 * DOWN), FadeIn(lab1, shift=0.1 * UP), run_time=0.6)
            vo.wait_until(3.5)
            self.play(GrowFromCenter(brace), FadeIn(brace_lab, shift=0.15 * RIGHT), run_time=0.9)
            vo.wait_until(5.4)
            self.play(FadeIn(shared, scale=0.9), run_time=0.8)
            self.play(Indicate(ch0.cells, color=B, scale_factor=1.04),
                      Indicate(ch1.cells, color=Y, scale_factor=1.04), run_time=1.0)
            vo.wait_until_sentence(1)
            self.play(FadeOut(VGroup(brace, brace_lab)), FadeIn(tok_s, shift=0.2 * RIGHT),
                      FadeIn(cap_s), FadeIn(src1), run_time=0.8)
            self.play(*grow_gate(g_s.bar), FadeIn(g_s_lab), run_time=1.0)
            vo.wait_until(vo.time_of(1) + 2.6)
            self.play(LaggedStartMap(FadeIn, g_s.nums, shift=0.1 * DOWN, lag_ratio=0.3),
                      FadeIn(g_s_sum, shift=0.1 * DOWN), run_time=1.0)
            vo.wait_until_sentence(2)
            self.play(GrowArrow(w0), GrowArrow(w1), FadeIn(write_tag), run_time=0.8)
            self.play(FadeIn(w0_lab), FadeIn(w1_lab), FadeIn(fill0), FadeIn(fill1), run_time=0.8)
            vo.wait_until(vo.time_of(2) + 2.0)
            self.play(FadeIn(tok_t, shift=0.2 * LEFT), FadeIn(cap_t), *grow_gate(g_t.bar), FadeIn(g_t_lab),
                      FadeIn(g_t.nums), FadeIn(g_t_sum), run_time=0.9)
            self.play(GrowArrow(r0), GrowArrow(r1), FadeIn(r0_lab), FadeIn(r1_lab), FadeIn(read_tag),
                      run_time=0.9)

        # ================================================================ N3.2 the score gains a factor
        piece_specs = [("0.8", B), (R"\times", INK), ("0.3", B), ("+", INK), ("0.2", Y), (R"\times", INK),
                       ("0.7", Y), ("=", INK), ("0.38", INK)]
        comp = VGroup(*[M(s, 40, c) for s, c in piece_specs]).arrange(RIGHT, buff=0.16)
        comp_gg = M(R"= g_t \cdot g_s", 40, GATE_COLOR)
        comp_cap = L("t's read of s's write, over both channels:", 22, MUTED)
        comp_line = VGroup(comp_cap, comp, comp_gg).arrange(RIGHT, buff=0.3)
        comp_gg.next_to(comp, RIGHT, buff=0.16)          # same spacing as inside the sum
        comp_line.move_to(3.1 * DOWN)
        comp_gg.shift((comp[7].get_center()[1] - comp_gg[0].get_center()[1]) * UP)

        eq1 = M(R"\text{score}(t,s) = (x_t \cdot x_s) \times (g_t \cdot g_s)", 56,
                isolate=[R"(x_t \cdot x_s)", R"(g_t \cdot g_s)"])   # \times left free: keeps its spacing
        eq1.move_to(0.35 * UP)
        content = eq1[R"(x_t \cdot x_s)"]
        gatef = eq1[R"(g_t \cdot g_s)"]
        times = eq1[R"\times"]
        _skip = {id(m) for m in [*gatef.get_family(), *times.get_family()]}
        eq1_left = VGroup(*[m for m in eq1 if id(m) not in _skip])
        br_c = Brace(content, DOWN, buff=0.12)
        br_c_lab = VGroup(L("content match", 26), L("as in plain BDH", 22, MUTED)).arrange(DOWN, buff=0.08)
        br_c_lab.next_to(br_c, DOWN, buff=0.1)
        br_g = Brace(gatef, DOWN, buff=0.12).set_color(GATE_COLOR)
        br_g_lab = VGroup(L("context-gate match", 26, GATE_COLOR),
                          L("one gate: writes and reads", 22, MUTED)).arrange(DOWN, buff=0.08)
        br_g_lab.next_to(br_g, DOWN, buff=0.1)
        code = code_block('G = torch.einsum("btk,bsk->bts", gr, gw)\nscores = scores * self.G',
                          title="test_multilayer_binding.py:331, :218 (condensed)", size=24)
        code.to_corner(DL, buff=0.35)
        src2 = source_note("Report §1, eq. (1)")
        diagram = VGroup(ch0, ch1, lab0, lab1, shared, fill0, fill1, tok_s, cap_s, g_s, g_s_lab, g_s_sum,
                         tok_t, cap_t, g_t, g_t_lab, g_t_sum, w0, w1, r0, r1, w0_lab, w1_lab, r0_lab, r1_lab,
                         write_tag, read_tag)

        with self.voiceover(
            "With one distribution doing both jobs, the attention score gains a second factor. The first is the "
            "content match from before. The second is the match between the two tokens' gates."
        ) as vo:
            self.play(FadeIn(comp_cap),
                      TransformFromCopy(g_s.nums[0], comp[0]), TransformFromCopy(g_t.nums[0], comp[2]),
                      TransformFromCopy(g_s.nums[1], comp[4]), TransformFromCopy(g_t.nums[1], comp[6]),
                      FadeIn(VGroup(comp[1], comp[3], comp[5])), run_time=1.2)
            self.play(FadeIn(VGroup(comp[7], comp[8]), shift=0.1 * RIGHT), run_time=0.5)
            self.play(Write(comp_gg), run_time=0.6)
            self.play(FadeOut(diagram, shift=0.2 * UP), FadeOut(comp_cap),
                      VGroup(comp, comp_gg).animate.scale(0.85).move_to(2.05 * UP), run_time=0.8)
            self.play(Write(eq1_left), FadeOut(src1), FadeIn(src2), run_time=0.8)
            self.play(Write(times), TransformFromCopy(comp_gg[1:], gatef), run_time=0.8)
            vo.wait_until_sentence(1)
            self.play(GrowFromCenter(br_c), FadeIn(br_c_lab, shift=0.1 * DOWN),
                      content.animate.set_color(KEY_COLOR), run_time=0.8)
            vo.wait_until_sentence(2)
            self.play(GrowFromCenter(br_g), FadeIn(br_g_lab, shift=0.1 * DOWN), run_time=0.8)
            self.play(FlashAround(gatef, color=GATE_COLOR), run_time=1.0)
            self.play(FadeIn(code, shift=0.2 * UP), run_time=0.8)

        # ================================================================ N3.3 one or zero; k = 1 is BDH
        eq_box = SurroundingRectangle(gatef, buff=0.08).set_stroke(GATE_COLOR, 2)

        def case_card(head_text, head_color, gs_vals, gs_colors, comp_tex, result, res_color):
            head = L(head_text, 26, head_color, weight="BOLD")
            vt = vec_cells([1, 0], [B, Y])
            vs = vec_cells(gs_vals, gs_colors)
            vecs = VGroup(M("g_t =", 36), vt, M("g_s =", 36), vs).arrange(RIGHT, buff=0.18)
            vecs[2].shift(0.35 * RIGHT)
            vecs[3].shift(0.35 * RIGHT)
            cmp_ = M(comp_tex, 36)
            res = L(result, 24, res_color)
            inner = VGroup(head, vecs, cmp_, res).arrange(DOWN, buff=0.3)
            c = card(inner, buff=0.3)
            c.head, c.vecs, c.cmp, c.res = head, vecs, cmp_, res
            return c

        case_a = case_card("same channel", INK, [1, 0], [B, Y],
                           R"g_t \cdot g_s = 1\cdot 1 + 0\cdot 0 = 1", "they see each other's memory", GOOD)
        case_b = case_card("different channels", INK, [0, 1], [B, Y],
                           R"g_t \cdot g_s = 1\cdot 0 + 0\cdot 1 = 0", "invisible to each other", BAD)
        cases = VGroup(case_a, case_b).arrange(RIGHT, buff=0.7).move_to(0.35 * DOWN)
        for c in (case_a, case_b):   # the two cards share one height
            c[0].stretch_to_fit_height(max(case_a[0].get_height(), case_b[0].get_height()))

        # gate-match matrix over the K0 group and the query's key: rows read (t), columns write (s)
        n, cell = len(MAT_TOKENS), 0.46
        streams = [s for _, _, s in MAT_TOKENS]
        grid_bg = VGroup()
        entries, entries_k1 = VGroup(), VGroup()
        for t in range(n):
            for s in range(n):
                pos = np.array([s * cell, -t * cell, 0])
                if s >= t:
                    grid_bg.add(Square(cell).set_stroke(FAINT, 1, opacity=0.35).set_fill(opacity=0).move_to(pos))
                else:
                    same = streams[s] == streams[t]
                    entries.add(matrix_cell(1 if same else 0, STREAM_COLORS[streams[t]], cell).move_to(pos))
                    entries_k1.add(matrix_cell(1, MUTED, cell).move_to(pos))
                    entries[-1].ts = entries_k1[-1].ts = (t, s)
        mat = VGroup(grid_bg, entries)
        row_labs = VGroup(*[token(lbl, kind, st, width=0.9, height=0.4, size=20) for lbl, kind, st in MAT_TOKENS])
        col_labs = VGroup(*[token(lbl, kind, st, width=0.9, height=0.4, size=20).rotate(PI / 2)
                            for lbl, kind, st in MAT_TOKENS])
        for t, rl in enumerate(row_labs):
            rl.move_to([-cell / 2 - 0.12 - 0.45, -t * cell, 0])
        for s, cl in enumerate(col_labs):
            cl.move_to([s * cell, cell / 2 + 0.12 + 0.45, 0])
        corner = M(R"g_t \cdot g_s", 30, GATE_COLOR).move_to([-cell / 2 - 0.57, cell / 2 + 0.57, 0])
        matrix = VGroup(mat, row_labs, col_labs, corner)
        entries_k1.move_to(entries)  # same coordinates before the group moves
        matrix_k1_holder = VGroup(entries_k1)
        whole = VGroup(matrix, matrix_k1_holder)
        whole.shift(np.array([-4.2, 0.95, 0]) - mat.get_corner(UL) + np.array([-0.2, 0, 0]))
        axis_note = L("row t reads, column s wrote (s before t)", 22, MUTED).next_to(mat, DOWN, buff=0.25)
        # blocks to point at
        zero_block = VGroup(*[e for e in entries if streams[e.ts[0]] != streams[e.ts[1]]])
        same_block = VGroup(*[e for e in entries if streams[e.ts[0]] == streams[e.ts[1]]])
        zero_rect = SurroundingRectangle(zero_block, buff=0.04).set_stroke(BAD, 2.5)

        leg_same = VGroup(VGroup(matrix_cell(1, B, cell), matrix_cell(1, Y, cell)).arrange(RIGHT, buff=0.08),
                          L("same context: shared memory", 24, GOOD)).arrange(RIGHT, buff=0.3)
        leg_diff = VGroup(VGroup(matrix_cell(0, B, cell)), L("different contexts: isolated", 24, BAD)
                          ).arrange(RIGHT, buff=0.3)
        legend = VGroup(leg_same, leg_diff).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        legend.move_to([3.4, 0.35, 0]).align_to(np.array([0.9, 0, 0]), LEFT)
        # what the matrix assumes: every token's gate is one-hot on its own context's channel
        assume = VGroup(L("if each context has its own channel", 24, GATE_COLOR),
                        L("(every gate one-hot: stream s → channel s)", 22, MUTED)
                        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        assume.next_to(legend, UP, buff=0.35, aligned_edge=LEFT)
        k1_line1 = VGroup(L("k = 1:", 26, INK, weight="BOLD"), M("g_t = g_s =", 34),
                          vec_cells([1], [MUTED])).arrange(RIGHT, buff=0.2)
        k1_line2 = M(R"g_t \cdot g_s = 1 \text{ for every pair}", 34)
        k1_line3 = L("the model is plain BDH", 26, INK, weight="BOLD")
        k1 = VGroup(k1_line1, k1_line2, k1_line3).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        k1.next_to(legend, DOWN, buff=0.7, aligned_edge=LEFT)
        src3 = source_note("Report §1, eq. (1); k = 1: bdh_recurrent.py:76-77")

        with self.voiceover(
            "If two tokens put their gates on the same channel, that factor is one and they see each other's "
            "memory. If they use different channels it is zero, and they are invisible to each other. So tokens "
            "that share a context share memory, and tokens from different contexts are isolated. With k equal "
            "to one, the factor is always one, and the model is plain BDH."
        ) as vo:
            self.play(FadeOut(VGroup(comp, comp_gg, br_c, br_c_lab, br_g, br_g_lab)),
                      eq1.animate.scale(0.72).move_to(2.5 * UP + 1.2 * RIGHT), run_time=0.9)
            # a selection's cached bounding box goes stale when its parent moves: measure the glyphs afresh
            eq_box.become(SurroundingRectangle(VGroup(*gatef.family_members_with_points()), buff=0.08)
                          .set_stroke(GATE_COLOR, 2))
            self.play(ShowCreation(eq_box), FadeIn(case_a[0]), FadeIn(case_a.head), FadeIn(case_a.vecs),
                      FadeOut(src2), FadeIn(src3), run_time=0.8)
            vo.wait_until(2.6)
            self.play(Write(case_a.cmp), run_time=1.2)
            vo.wait_until(4.0)
            self.play(FadeIn(case_a.res, shift=0.1 * UP), run_time=0.7)
            vo.wait_until_sentence(1)
            self.play(FadeIn(case_b[0]), FadeIn(case_b.head), FadeIn(case_b.vecs), run_time=0.8)
            self.play(Write(case_b.cmp), run_time=1.2)
            vo.wait_until(vo.time_of(1) + 2.9)
            self.play(FadeIn(case_b.res, shift=0.1 * UP), run_time=0.7)
            vo.wait_until_sentence(2)
            self.play(FadeOut(cases, shift=0.2 * DOWN), FadeOut(code), run_time=0.6)
            self.play(LaggedStartMap(FadeIn, row_labs, shift=0.1 * RIGHT, lag_ratio=0.08),
                      LaggedStartMap(FadeIn, col_labs, shift=0.1 * DOWN, lag_ratio=0.08),
                      FadeIn(grid_bg), FadeIn(corner), run_time=1.0)
            self.play(LaggedStartMap(FadeIn, entries, scale=0.6, lag_ratio=0.04), FadeIn(axis_note),
                      FadeIn(assume, shift=0.1 * LEFT), run_time=1.3)
            self.play(FadeIn(leg_same, shift=0.1 * LEFT), Indicate(same_block, scale_factor=1.0, color=WHITE),
                      run_time=0.9)
            self.play(FadeIn(leg_diff, shift=0.1 * LEFT), ShowCreation(zero_rect), run_time=0.9)
            vo.wait_until_sentence(3)
            self.play(FadeIn(k1_line1, shift=0.1 * UP), assume.animate.set_opacity(0.25),
                      legend.animate.set_opacity(0.25), run_time=0.7)
            self.play(FadeOut(zero_rect), LaggedStart(*[Transform(e, e1) for e, e1 in zip(entries, entries_k1)],
                                                      lag_ratio=0.02), run_time=1.4)
            self.play(Write(k1_line2), run_time=0.9)
            vo.wait_until(vo.time_of(3) + 3.6)
            self.play(FadeIn(k1_line3, shift=0.1 * UP), run_time=0.7)

        # ================================================================ N3.4 the perfect gate
        triples = VGroup(*[triple(s, k, v) for s, k, v in SEQ])
        query = VGroup(token("CTX0", "ctx", 0), token("K0"), token("?", "query")).arrange(RIGHT, buff=0.06)
        rows = VGroup(*[VGroup(triples[2 * r], triples[2 * r + 1]) for r in range(4)])
        for row in rows:
            row.arrange(RIGHT, buff=0.45)
        rows.arrange(DOWN, buff=0.26, aligned_edge=LEFT)
        query.next_to(rows, DOWN, buff=0.6, aligned_edge=LEFT)
        seq = VGroup(rows, query)
        seq.move_to([0, -0.2, 0]).to_edge(LEFT, buff=0.5)
        unders = VGroup(*[Rectangle(width=tr.get_width(), height=0.08).set_fill(STREAM_COLORS[s], 1)
                          .set_stroke(width=0).next_to(tr, DOWN, buff=0.06) for tr, (s, _, _) in zip(triples, SEQ)])
        q_under = Rectangle(width=query.get_width(), height=0.08).set_fill(B, 1).set_stroke(width=0)
        q_under.next_to(query, DOWN, buff=0.06)
        pg_lab = VGroup(L("hand-set perfect gate:", 24, GATE_COLOR, weight="BOLD"),
                        L("stream s → channel s", 24, INK)).arrange(RIGHT, buff=0.2)
        pg_lab.next_to(seq, UP, buff=0.4, aligned_edge=LEFT)
        bind0 = [(k, v) for s, k, v in SEQ if s == 0]
        bind1 = [(k, v) for s, k, v in SEQ if s == 1]
        p0, p1 = channel_panel(0, bind0), channel_panel(1, bind1)
        panels = VGroup(p0, p1).arrange(DOWN, buff=0.35).move_to([3.55, -0.05, 0])
        # which panel entry each triple lands in
        slot = {}
        c0 = c1 = 0
        for i, (s, k, v) in enumerate(SEQ):
            if s == 0:
                slot[i] = p0.ents[c0]
                c0 += 1
            else:
                slot[i] = p1.ents[c1]
                c1 += 1
        answer = token(f"V{ANSWER}", "val", 0).move_to(query[2])
        k0_hl = VGroup(SurroundingRectangle(p0.ents[0], buff=0.06).set_stroke(B, 2.5),
                       SurroundingRectangle(p1.ents[0], buff=0.06).set_stroke(Y, 2.5))
        fb = frac_bar("perfect gate", 10, 10, color=GOOD, width=2.4, label_width=0, size=24)
        fb_cap = VGroup(L("runs that bind, both machines", 22, MUTED),
                        L("S = 2, P = 4, no conv.", 22, MUTED)).arrange(DOWN, buff=0.08, aligned_edge=RIGHT)
        result = VGroup(fb_cap, fb).arrange(RIGHT, buff=0.4)
        result.to_edge(RIGHT, buff=0.5).set_y(-3.05)
        sep_lab = L("K0's two values now sit in different channels", 22, GOOD)
        sep_lab.next_to(panels, DOWN, buff=0.15)
        src4 = source_note("Report §13, Table 8; sequence: first training sequence of seed 280")

        # the matrix's row tokens are the K0 group and the query's first two tokens
        targets = [triples[0][0], triples[0][1], triples[0][2], triples[1][0], triples[1][1], triples[1][2],
                   query[0], query[1]]
        rest = VGroup(*[m for m in [*triples[2:], query[2]]])

        with self.voiceover(
            "A hand-set perfect gate sends every token of stream s to channel s. With it, the conflict "
            "disappears, and the model binds: ten of ten runs in the basic setting."
        ) as vo:
            self.play(FadeOut(VGroup(mat, col_labs, corner, axis_note, assume, legend, k1, eq1, eq_box, src3)),
                      run_time=0.5)
            self.play(*[ReplacementTransform(rl, tg) for rl, tg in zip(row_labs, targets)],
                      LaggedStartMap(FadeIn, rest, lag_ratio=0.05), run_time=1.0)
            self.play(FadeIn(pg_lab, shift=0.1 * DOWN), FadeIn(p0.rect), FadeIn(p0.head), FadeIn(p1.rect),
                      FadeIn(p1.head), LaggedStartMap(GrowFromEdge, unders, edge=LEFT, lag_ratio=0.1),
                      run_time=1.1)
            self.play(LaggedStart(*[AnimationGroup(TransformFromCopy(VGroup(tr[1], tr[2]),
                                                                     VGroup(slot[i][0], slot[i][2])),
                                                   FadeIn(slot[i][1]))
                                    for i, tr in enumerate(triples)], lag_ratio=0.12), run_time=2.2)
            vo.wait_until_sentence(1)
            self.play(ShowCreation(k0_hl), FadeIn(sep_lab, shift=0.1 * UP), run_time=0.7)
            self.play(GrowFromEdge(q_under, LEFT), p1.animate.set_opacity(0.3), FadeOut(k0_hl[1]),
                      Indicate(p0.ents[0], color=B, scale_factor=1.08), run_time=1.0)
            self.play(TransformFromCopy(p0.ents[0][2], answer, path_arc=-PI / 4), FadeOut(query[2]), run_time=1.2)
            self.play(FadeOut(sep_lab), FadeIn(fb_cap), FadeIn(fb.name_mob), FadeIn(fb.track), FadeIn(src4),
                      run_time=0.5)
            self.play(*grow_bar(fb), run_time=1.0)
            self.play(Indicate(fb.value, color=GOOD), run_time=0.8)

        # ================================================================ N3.5 the learned gate
        title2 = section_title("A learned gate")
        ask = T("Can a gate learn this?", 40).move_to(0.3 * UP)

        xs = [-4.1, -1.6, 0.9, 3.4]
        rnn_toks = VGroup(token("CTX1", "ctx", 1), token("K0"), token("V13", "val", 1), token("CTX0", "ctx", 0))
        for tk, x in zip(rnn_toks, xs):
            tk.move_to([x, -3.05, 0])
        rng = np.random.default_rng(3)
        hgrids = VGroup()
        for x in xs:
            g = VGroup(*[Square(0.13).set_stroke(GATE_COLOR, 0.8, opacity=0.6)
                         .set_fill(GATE_COLOR, 0.12 + 0.75 * rng.random()) for _ in range(32)])
            g.arrange_in_grid(4, 8, buff=0.02).move_to([x, -1.15, 0])
            hgrids.add(g)
        h_labs = VGroup(*[M(f"h_{i + 1}", 30).next_to(g, UP, buff=0.1).align_to(g, LEFT)
                          for i, g in enumerate(hgrids)])
        in_arrows = VGroup(*[Arrow(tk.get_top(), g.get_bottom(), buff=0.1, thickness=4).set_fill(IN_COLOR, 1)
                             for tk, g in zip(rnn_toks, hgrids)])
        v_labs = VGroup(*[M(f"v_{i + 1}", 30, IN_COLOR).next_to(a, RIGHT, buff=0.12)
                          for i, a in enumerate(in_arrows)])
        h0 = M("h_0 = 0", 30, MUTED).move_to([-6.05, -1.15, 0])
        rec_arrows = VGroup()
        prev = h0
        for g in hgrids:
            rec_arrows.add(Arrow(prev.get_right(), g.get_left(), buff=0.12, thickness=4).set_fill(REC_COLOR, 1))
            prev = g
        units = L("32 units", 22, GATE_COLOR).next_to(hgrids[-1], RIGHT, buff=0.3)
        probs = [[0.52, 0.48], [0.49, 0.51], [0.5, 0.5], [0.51, 0.49]]
        gbars = VGroup(*[gate_bar(p, width=1.3, height=0.28).move_to([x, 0.35, 0]) for p, x in zip(probs, xs)])
        g_labs = VGroup(*[M(f"g_{i + 1}", 30, GATE_COLOR).next_to(b, LEFT, buff=0.15)
                          for i, b in enumerate(gbars)])
        up_arrows = VGroup(*[Arrow(g.get_top(), b.get_bottom(), buff=0.1, thickness=3).set_fill(INK, 0.8)
                             for g, b in zip(hgrids, gbars)])
        arm_lab = L("arm A: the learned gate", 24, GATE_COLOR, weight="BOLD").move_to([-4.7, 1.25, 0])

        t_in, t_rec = R"W_{\text{in}}\, v_t", R"W_h\, h_{t-1}"
        eq_h = M(R"h_t = \tanh(" + t_in + " + " + t_rec + ")", 42, isolate=[t_in, t_rec])
        eq_g = M(R"g_t = \mathrm{softmax}(W_g\, h_t)", 42)
        eqs = VGroup(eq_h, eq_g).arrange(RIGHT, buff=1.2).move_to(2.35 * UP + 0.6 * RIGHT)
        term_in, term_rec = eq_h[t_in], eq_h[t_rec]
        br_in = Brace(term_in, DOWN, buff=0.08).set_color(IN_COLOR)
        br_rec = Brace(term_rec, DOWN, buff=0.08).set_color(REC_COLOR)
        lab_in = L("input term", 22, IN_COLOR).next_to(br_in, DOWN, buff=0.06)
        lab_rec = L("recurrent term", 22, REC_COLOR).next_to(br_rec, DOWN, buff=0.06)
        src5 = source_note("Report §1, eq. (2); H_GATE = 32 (test_instrument_v2.py:125); gate values illustrative")

        with self.voiceover(
            "The real question is whether a gate can learn this. The learned gate, arm A, is a small recurrent "
            "network over the token embeddings, with a thirty-two unit state h. At each step, h is the hyperbolic "
            "tangent of the sum of two terms: the input term, which looks at the current token, and the recurrent "
            "term, which carries the previous state forward. A linear map and a softmax turn h into the gate."
        ) as vo:
            self.play(FadeOut(VGroup(rows, query[0], query[1], unders, q_under, answer, pg_lab, panels, result,
                                     k0_hl[0], src4)),
                      Transform(title, title2), run_time=0.9)
            self.play(Write(ask), run_time=1.1)
            vo.wait_until_sentence(1)
            self.play(FadeOut(ask, shift=0.3 * UP), LaggedStartMap(FadeIn, rnn_toks, shift=0.2 * UP, lag_ratio=0.15),
                      FadeIn(arm_lab), FadeIn(src5), run_time=1.0)
            self.play(FadeIn(h0), run_time=0.4)
            self.play(LaggedStart(*[AnimationGroup(GrowArrow(a), FadeIn(vl), GrowArrow(ra), FadeIn(g), FadeIn(hl))
                                    for a, vl, ra, g, hl in zip(in_arrows, v_labs, rec_arrows, hgrids, h_labs)],
                                  lag_ratio=0.45), run_time=3.0)
            vo.wait_until(vo.time_of(1) + 5.3)
            self.play(FadeIn(units, shift=0.1 * LEFT),
                      LaggedStart(*[Indicate(g, color=GATE_COLOR, scale_factor=1.12) for g in hgrids],
                                  lag_ratio=0.2), run_time=1.4)
            vo.wait_until_sentence(2)
            self.play(Write(eq_h), run_time=1.5)
            vo.wait_until(vo.time_of(2) + 3.7)
            self.play(term_in.animate.set_color(IN_COLOR), GrowFromCenter(br_in), FadeIn(lab_in), run_time=0.7)
            self.play(LaggedStart(*[Indicate(a, color=IN_COLOR, scale_factor=1.2) for a in in_arrows],
                                  lag_ratio=0.2), run_time=1.3)
            vo.wait_until(vo.time_of(2) + 6.4)
            self.play(term_rec.animate.set_color(REC_COLOR), GrowFromCenter(br_rec), FadeIn(lab_rec), run_time=0.7)
            self.play(LaggedStart(*[Indicate(a, color=REC_COLOR, scale_factor=1.2) for a in rec_arrows],
                                  lag_ratio=0.2), run_time=1.5)
            vo.wait_until_sentence(3)
            self.play(Write(eq_g), run_time=0.9)
            self.play(LaggedStart(*[GrowArrow(a) for a in up_arrows], lag_ratio=0.15),
                      LaggedStart(*[AnimationGroup(*grow_gate(b, 0.6)) for b in gbars], lag_ratio=0.15),
                      LaggedStartMap(FadeIn, g_labs, lag_ratio=0.15), run_time=1.3)

        # ================================================================ N3.6 no labels; remember two terms
        tag_txt = L("stream labels", 26, MUTED)
        tag_box = DashedVMobject(SurroundingRectangle(tag_txt, buff=0.18), num_dashes=34).set_stroke(MUTED, 1.5)
        tag = VGroup(tag_box, tag_txt)
        tag.move_to([5.45, -2.6, 0])
        cross = Line(tag_txt.get_left() + 0.08 * LEFT, tag_txt.get_right() + 0.08 * RIGHT).set_stroke(BAD, 3.5)
        never = L("never given", 24, BAD, weight="BOLD").next_to(tag, UP, buff=0.15)
        loss = card(L("task loss", 24, INK, weight="BOLD"), buff=0.2).move_to([5.6, 0.35, 0])
        loss_arrow = CurvedArrow(loss.get_corner(UL) + 0.3 * RIGHT + 0.08 * UP, gbars[0].get_top() + 0.12 * UP,
                                 angle=0.35).set_stroke(INK, 3, opacity=0.85)
        loss_arrow.get_tip().set_fill(INK, 0.85).set_stroke(width=0)
        only = L("the only signal", 22, MUTED).next_to(loss, DOWN, buff=0.15)

        # the badge's terms are shrunken copies of the equation's own glyphs, so the move is a clean shrink
        glyphs_in = VGroup(*term_in.family_members_with_points())
        glyphs_rec = VGroup(*term_rec.family_members_with_points())
        badge_in = VGroup(glyphs_in.copy().scale(0.8), L("input term", 22, IN_COLOR)).arrange(RIGHT, buff=0.25)
        badge_rec = VGroup(glyphs_rec.copy().scale(0.8), L("recurrent term", 22, REC_COLOR)
                           ).arrange(RIGHT, buff=0.25)
        badge_body = VGroup(badge_in, badge_rec).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        badge = card(badge_body, buff=0.2)
        badge.to_edge(RIGHT, buff=0.4).set_y(3.3 - badge.get_height() / 2)

        with self.voiceover(
            "Nothing tells this gate which stream a token belongs to. It has to find that out from the task loss "
            "alone. Keep its two terms in mind; they come back at the end."
        ) as vo:
            self.play(FadeIn(tag), run_time=0.6)
            self.play(ShowCreation(cross), FadeIn(never, shift=0.1 * DOWN), run_time=0.8)
            vo.wait_until_sentence(1)
            self.play(FadeIn(loss, shift=0.2 * LEFT), run_time=0.6)
            self.play(ShowCreation(loss_arrow), FadeIn(only),
                      LaggedStart(*[Indicate(b, color=WHITE, scale_factor=1.1) for b in reversed(gbars)],
                                  lag_ratio=0.2), run_time=1.4)
            vo.wait_until_sentence(2)
            moving = {id(m) for m in [*term_in.get_family(), *term_rec.get_family()]}
            eq_rest = VGroup(*[m for m in eq_h.submobjects if id(m) not in moving], eq_g, br_in, br_rec)
            self.play(FadeIn(badge[0]), ReplacementTransform(glyphs_in, badge_in[0]),
                      ReplacementTransform(glyphs_rec, badge_rec[0]), ReplacementTransform(lab_in, badge_in[1]),
                      ReplacementTransform(lab_rec, badge_rec[1]), FadeOut(eq_rest), run_time=1.4)
            self.play(Indicate(badge_in[0], color=IN_COLOR), Indicate(badge_rec[0], color=REC_COLOR), run_time=1.0)
        self.wait(0.6)
        self.clear_all()
