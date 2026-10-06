"""Chapter 1 — the Dragon Hatchling (report §1, §3, §16; bdh.py; bdh_recurrent.py; facts_model §1, §2.11)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403

# ------------------------------------------------------------------ schematic network (N1.1)
N_DRAWN, D_DRAWN = 12, 6            # drawn neurons / value features (stand-ins for N = 256, d = 32)
# illustrative tokens: (label, active neurons, their activity, value-feature intensities)
TOKENS = [
    ("the", [1, 6, 10], [0.9, 0.6, 0.8], [0.8, 0.2, 0.0, 0.6, 0.3, 0.9]),
    ("door", [3, 7, 9], [1.0, 0.7, 0.8], [0.1, 0.9, 0.7, 0.0, 0.8, 0.3]),
    ("code", [0, 4, 7], [0.7, 0.9, 0.6], [0.6, 0.0, 0.4, 0.9, 0.2, 0.7]),
]
QUERY_ACT = ([3, 7, 9], [1.0, 0.7, 0.8])   # "door" again: same sparse code

# bdh.py:122-144, the layer loop (facts_model §1.4; the attn call condensed to one line)
LAYER_CODE = """for level in range(C.n_layer):
    x_latent = x @ self.encoder
    x_sparse = F.relu(x_latent)  # B, nh, T, N
    yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
    yKV = self.ln(yKV)
    y_latent = yKV @ self.encoder_v
    y_sparse = F.relu(y_latent)
    xy_sparse = x_sparse * y_sparse  # B, nh, T, N
    xy_sparse = self.drop(xy_sparse)
    yMLP = (
        xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
    )  # B, 1, T, D
    y = self.ln(yMLP)
    x = self.ln(x + y)"""

DECAY = 0.95
HALF_LIFE = np.log(0.5) / np.log(DECAY)    # 13.51 tokens


def syn_style(line, w):
    """Synapse look for accumulated strength w >= 0."""
    k = min(1.0, w)
    line.set_stroke(interpolate_color(FAINT, MEMORY_COLOR, min(1.0, 0.3 + 1.2 * k)),
                    width=1 + 5 * k, opacity=0.3 + 0.7 * k)
    return line


def code_line_groups(cb, src):
    """Split a code_block's glyphs into one VGroup per source line (glyphs follow text order)."""
    groups, idx = [], 0
    for line in src.split("\n"):
        n = len("".join(line.split()))
        groups.append(VGroup(*cb.code[idx: idx + n]))
        idx += n
    return groups


def code_sub(line_src, line_group, sub):
    """Glyphs of substring `sub` inside one code line."""
    start = line_src.index(sub)
    a = len("".join(line_src[:start].split()))
    n = len("".join(sub.split()))
    return VGroup(*line_group[a: a + n])


def flow_box(text, width=3.7, height=0.5, color=INK, size=22):
    t = L(text, size=size, color=color)
    if t.get_width() > width - 0.25:
        t.set_width(width - 0.25)
    box = RoundedRectangle(width=width, height=height, corner_radius=0.1)
    box.set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1.5)
    t.move_to(box)
    return VGroup(box, t)


class BDHModel(ClankersScene):
    def construct(self):
        ch_card = self.chapter_card(1, "The Dragon Hatchling")
        self.wait(0.6)
        title = section_title("BDH: memory in synapses")
        self.play(FadeOut(ch_card, shift=0.3 * UP), FadeIn(title, shift=0.3 * UP))

        def retitle(old, text):
            new = section_title(text)
            return [FadeOut(old, shift=0.2 * UP), FadeIn(new, shift=0.2 * UP)], new

        # ============================================================ N1.1  memory in synapses
        bdh = T("BDH", size=84, color=MEMORY_COLOR)
        name = T("the Dragon Hatchling", size=40)
        cite = VGroup(
            T("The Dragon Hatchling: The Missing Link between", size=26),
            T("the Transformer and Models of the Brain", size=26),
            L("A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz", size=22, color=MUTED),
            L("Pathway  ·  2025  ·  arXiv:2509.26507", size=22, color=INK),
        ).arrange(DOWN, buff=0.14)
        cite[2].shift(0.08 * DOWN)
        cite[3].shift(0.08 * DOWN)
        cite_card = card(cite, buff=0.3)
        intro = VGroup(bdh, name, cite_card).arrange(DOWN, buff=0.35).move_to(0.15 * DOWN)
        src_ref = source_note("Report ref. [1]; bdh.py:1 “Copyright 2025 Pathway Technology, Inc.”")

        # the network: neurons (sparse code x_t) on top, value features (v_t) below, synapses between
        neur_pos = [np.array([-3.85 + 0.7 * i, 0.85, 0]) for i in range(N_DRAWN)]
        feat_pos = [np.array([-2.5 + 1.0 * j, -1.75, 0]) for j in range(D_DRAWN)]
        neurons = VGroup(*[Circle(radius=0.16).set_fill(PANEL, 1).set_stroke(MUTED, 1.5).move_to(p)
                           for p in neur_pos])
        feats = VGroup(*[Square(0.3).set_fill(PANEL, 1).set_stroke(MUTED, 1.5).move_to(p) for p in feat_pos])
        syn = [[syn_style(Line(neur_pos[i], feat_pos[j], buff=0.18), 0.0) for j in range(D_DRAWN)]
               for i in range(N_DRAWN)]
        syn_all = VGroup(*[ln for row in syn for ln in row])
        n_lab = VGroup(L("neurons", size=24), L("N of them, few fire", size=22, color=MUTED)).arrange(
            DOWN, buff=0.08, aligned_edge=RIGHT)
        n_lab.next_to(neurons, LEFT, buff=0.35)
        f_lab = VGroup(L("value features", size=24), L("d of them", size=22, color=MUTED)).arrange(
            DOWN, buff=0.08, aligned_edge=RIGHT)
        f_lab.next_to(feats, LEFT, buff=0.4)
        s_lab = VGroup(L("synapses", size=26, color=MEMORY_COLOR, weight="BOLD"),
                       L("the working memory", size=22, color=MUTED)).arrange(DOWN, buff=0.08)
        s_lab.move_to([5.15, -0.45, 0])

        with self.voiceover(
            "The model underneath is BDH, the Dragon Hatchling, introduced by Pathway in 2025. "
            "BDH keeps its working memory in synapses."
        ) as vo:
            self.play(FadeIn(bdh, scale=0.85), run_time=1.0)
            vo.wait_until(2.4)
            self.play(Write(name), run_time=1.0)
            vo.wait_until(3.85)
            self.play(FadeIn(cite_card, shift=0.2 * UP), FadeIn(src_ref), run_time=1.0)
            vo.wait_until_sentence(1)
            self.play(FadeOut(intro, shift=0.4 * UP), FadeOut(src_ref), run_time=0.7)
            self.play(LaggedStartMap(FadeIn, neurons, scale=0.5, lag_ratio=0.05),
                      LaggedStartMap(FadeIn, feats, scale=0.5, lag_ratio=0.08),
                      FadeIn(n_lab), FadeIn(f_lab), run_time=1.0)
            self.play(LaggedStartMap(ShowCreation, syn_all, lag_ratio=0.01), FadeIn(s_lab, shift=0.2 * LEFT),
                      run_time=1.2)

        # ------------------------------------------------------------ N1.1b  Hebbian writes
        W = np.zeros((N_DRAWN, D_DRAWN))
        tok_mobs = VGroup(*[token(lab) for lab, *_ in TOKENS])
        for k, tm in enumerate(tok_mobs):
            tm.move_to([0.0 + 1.15 * k, 2.2, 0])
        wire = T("fire together, wire together", size=28, color=MEMORY_COLOR, slant="ITALIC")
        wire.move_to([0, -2.75, 0])

        def light(act, xs, vs):
            anims = [neurons[i].animate.set_fill(INK, 0.55 + 0.45 * x).set_stroke(INK, 2.5) for i, x in zip(act, xs)]
            anims += [feats[j].animate.set_fill(INK, 0.9 * v).set_stroke(INK if v > 0 else MUTED, 2 if v > 0 else 1.5)
                      for j, v in enumerate(vs)]
            return anims

        def unlight():
            return [n.animate.set_fill(PANEL, 1).set_stroke(MUTED, 1.5) for n in neurons] + \
                   [f.animate.set_fill(PANEL, 1).set_stroke(MUTED, 1.5) for f in feats]

        def flashes(tm, act):
            return [ShowPassingFlash(Line(tm.get_bottom(), neurons[i].get_top()).set_stroke(INK, 3), time_width=0.7)
                    for i in act]

        def wire_up(act, xs, vs):
            anims = []
            for i, x in zip(act, xs):
                for j, v in enumerate(vs):
                    if x * v > 0:
                        W[i, j] += x * v
                        anims.append(syn[i][j].animate.match_style(syn_style(syn[i][j].copy(), W[i, j])))
            return anims

        with self.voiceover(
            "Each token activates a sparse set of neurons, and a Hebbian rule strengthens the connections "
            "between neurons that are active together."
        ) as vo:
            lab, act, xs, vs = TOKENS[0]
            self.play(FadeIn(tok_mobs[0], shift=0.2 * DOWN), run_time=0.5)
            vo.wait_until(1.1)
            self.play(*flashes(tok_mobs[0], act), *light(act, xs, vs), run_time=1.0)
            self.play(*[Flash(neurons[i], color=INK, flash_radius=0.3, line_length=0.15) for i in act], run_time=0.7)
            vo.wait_until(3.3)
            self.play(*wire_up(act, xs, vs), run_time=1.0)
            self.play(Write(wire), run_time=0.9)
            for k in (1, 2):
                lab, act, xs, vs = TOKENS[k]
                self.play(*unlight(), FadeIn(tok_mobs[k], shift=0.2 * DOWN), run_time=0.35)
                self.play(*flashes(tok_mobs[k], act), *light(act, xs, vs), run_time=0.5)
                self.play(*wire_up(act, xs, vs), run_time=0.5)
            self.play(*unlight(), run_time=0.4)

        # ------------------------------------------------------------ the same picture as a matrix S
        pitch, cell = 0.34, 0.3
        g_center = np.array([-2.6, -0.55, 0])
        g_top = g_center[1] + N_DRAWN * pitch / 2
        g_left = g_center[0] - D_DRAWN * pitch / 2

        def cpos(i, j):
            return np.array([g_left + (j + 0.5) * pitch, g_top - (i + 0.5) * pitch, 0])

        wmax = W.max()
        cells = [[Square(cell).move_to(cpos(i, j)).set_stroke(MEMORY_COLOR, 1, opacity=0.45)
                  .set_fill(MEMORY_COLOR, 0.04 + 0.86 * W[i, j] / wmax) for j in range(D_DRAWN)]
                 for i in range(N_DRAWN)]
        cells_all = VGroup(*[c for row in cells for c in row])
        col_x = g_left - 0.32
        row_y = g_top + 0.32
        grid_neurons = [neurons[i].copy().scale(0.75).move_to([col_x, cpos(i, 0)[1], 0]) for i in range(N_DRAWN)]
        grid_feats = [feats[j].copy().scale(0.7).move_to([cpos(0, j)[0], row_y, 0]) for j in range(D_DRAWN)]
        n_lab2 = VGroup(L("neurons", size=24), L("N = 256", size=22, color=MUTED)).arrange(
            DOWN, buff=0.08, aligned_edge=RIGHT)
        n_lab2.move_to([col_x - 0.35, g_center[1], 0], aligned_edge=RIGHT)
        f_lab2 = VGroup(L("value features", size=24), L("d = 32", size=22, color=MUTED)).arrange(RIGHT, buff=0.25)
        f_lab2.next_to(cells_all, DOWN, buff=0.3)
        s_name = VGroup(M(R"S", 52, color=MEMORY_COLOR), L("synapse state", size=22, color=MEMORY_COLOR),
                        L("(ρ in the paper)", size=22, color=MUTED)).arrange(DOWN, buff=0.1)
        s_name.next_to(cells_all, RIGHT, buff=0.3)
        # each synapse line first slides into its own cell as a short dash (line-to-line: no shards)
        dashes = [[Line(cpos(i, j) + 0.12 * LEFT, cpos(i, j) + 0.12 * RIGHT).match_style(syn[i][j])
                   for j in range(D_DRAWN)] for i in range(N_DRAWN)]
        w_lab = L("write, at every token", size=22, color=MUTED)
        w_eq = M(R"S \ \leftarrow\ S + x_t \otimes v_t", 44)
        w_grp = VGroup(w_lab, w_eq).arrange(DOWN, buff=0.22).move_to([2.6, 0.85, 0])
        src_rec = source_note("bdh_recurrent.py:4-10  (the parallel and Hebbian forms agree)")

        self.play(
            *[Transform(syn[i][j], dashes[i][j]) for i in range(N_DRAWN) for j in range(D_DRAWN)],
            *[neurons[i].animate.scale(0.75).move_to(grid_neurons[i]) for i in range(N_DRAWN)],
            *[feats[j].animate.scale(0.7).move_to(grid_feats[j]) for j in range(D_DRAWN)],
            ReplacementTransform(n_lab, n_lab2), ReplacementTransform(f_lab, f_lab2),
            ReplacementTransform(s_lab, s_name), FadeOut(wire),
            run_time=1.5,
        )
        self.play(FadeOut(syn_all), *[FadeIn(c, scale=2.5) for c in cells_all], run_time=0.6)
        self.play(FadeIn(w_lab), Write(w_eq), FadeIn(src_rec), run_time=1.0)

        # ------------------------------------------------------------ N1.1c  reading
        q_tok = token("door", "query").move_to([3.6, 2.2, 0])
        q_now = L("now", size=22, color=QUERY_COLOR).next_to(q_tok, RIGHT, buff=0.15)
        q_act, q_x = QUERY_ACT
        row_boxes = VGroup(*[
            SurroundingRectangle(VGroup(neurons[i], *cells[i]), buff=0.04).set_stroke(QUERY_COLOR, 2.5)
            for i in q_act])
        read_val = sum(x * W[i] for i, x in zip(q_act, q_x))
        read_vec = VGroup(*[Square(cell).set_stroke(MEMORY_COLOR, 1.5).set_fill(MEMORY_COLOR, 0.08 + 0.85 * v / read_val.max())
                            for v in read_val]).arrange(RIGHT, buff=pitch - cell)
        read_vec.move_to([2.6, -1.75, 0])
        read_eq = M(R"x_t \cdot S", 44)
        read_lab = VGroup(L("read:", size=24, color=MUTED), read_eq).arrange(RIGHT, buff=0.25)
        read_lab.move_to([2.6, -0.85, 0])
        read_note = L("a weighted sum of the rows that are active now", size=22, color=MUTED)
        read_note.next_to(read_vec, DOWN, buff=0.3)

        with self.voiceover(
            "Reading the memory means asking which stored patterns overlap with what is active now."
        ) as vo:
            self.play(FadeIn(q_tok, shift=0.2 * DOWN), FadeIn(q_now), run_time=0.5)
            self.play(*[neurons[i].animate.set_fill(QUERY_COLOR, 0.4 + 0.6 * x).set_stroke(QUERY_COLOR, 2.5)
                        for i, x in zip(q_act, q_x)], run_time=0.6)
            vo.wait_until(1.15)
            self.play(LaggedStartMap(ShowCreation, row_boxes, lag_ratio=0.25), run_time=0.9)
            copies = VGroup(*[VGroup(*cells[i]).copy() for i in q_act])
            self.play(LaggedStart(*[c.animate.move_to(read_vec) for c in copies], lag_ratio=0.2),
                      FadeIn(read_lab), run_time=1.3)
            self.play(FadeOut(copies), FadeIn(read_vec), FadeIn(read_note), run_time=0.6)

        # ============================================================ N1.2  linear attention
        t_anims, title = retitle(title, "Linear attention")
        eq1 = M(R"x_t \cdot S \ =\ \displaystyle\sum_{s<t} (x_t \cdot x_s)\, v_s", 44).move_to([1.4, 2.2, 0])
        la_cap = L("linear attention: a weighted sum of earlier values, no softmax", size=22, color=MUTED)
        la_cap.next_to(eq1, DOWN, buff=0.3)

        # T x T score pattern, strictly causal (s < t)
        TT, mp, mc = 28, 0.125, 0.104
        m_center = np.array([-4.25, -0.95, 0])
        m_x0 = m_center[0] - (TT - 1) * mp / 2
        m_y0 = m_center[1] + (TT - 1) * mp / 2
        mcell = {}
        for t in range(TT):
            for s in range(t):
                mcell[(t, s)] = Square(mc).set_stroke(width=0).set_fill(MEMORY_COLOR, 0.6).move_to(
                    [m_x0 + s * mp, m_y0 - t * mp, 0])
        mat = VGroup(*mcell.values())
        m_frame = Square(TT * mp).move_to(m_center).set_stroke(FAINT, 1)
        m_top = L("earlier position s", size=22, color=MUTED).next_to(m_frame, UP, buff=0.15)
        m_left = L("current position t", size=22, color=MUTED).rotate(PI / 2).next_to(m_frame, LEFT, buff=0.15)
        m_cap = L("every pair s < t at once", size=22, color=MUTED).next_to(m_frame, DOWN, buff=0.2)

        # content match: two sparse, positive codes
        NB, bp = 16, 0.34
        code_t = {2: 0.8, 5: 0.55, 9: 0.9, 12: 0.65, 14: 0.4}
        code_s = {0: 0.5, 5: 0.75, 7: 0.6, 12: 0.9, 15: 0.5}

        def code_row(vals, color, base_y):
            g = VGroup()
            for k in range(NB):
                x = 0.55 + k * bp
                g.add(Line([x - 0.13, base_y, 0], [x + 0.13, base_y, 0]).set_stroke(FAINT, 2))
                if k in vals:
                    h = 0.85 * vals[k]
                    g.add(Rectangle(width=0.24, height=h).set_fill(color, 0.9).set_stroke(width=0)
                          .move_to([x, base_y + h / 2, 0]))
            return g

        row_t = code_row(code_t, QUERY_COLOR, 0.0)
        row_s = code_row(code_s, INK, -1.35)
        lab_t = M(R"x_t", 38, color=QUERY_COLOR).next_to(row_t, LEFT, buff=0.3).align_to(row_t, DOWN).shift(0.1 * UP)
        lab_s = M(R"x_s", 38, color=INK).next_to(row_s, LEFT, buff=0.3).align_to(row_s, DOWN).shift(0.1 * UP)
        lab_s.align_to(lab_t, LEFT)
        overlaps = VGroup(*[
            Rectangle(width=0.32, height=2.45).set_stroke(MEMORY_COLOR, 2).set_fill(MEMORY_COLOR, 0.1)
            .move_to([0.55 + k * bp, -0.45, 0]) for k in sorted(set(code_t) & set(code_s))])
        ov_lab = VGroup(L("content match: only neurons active in both count", size=22, color=MEMORY_COLOR),
                        L("codes are sparse and positive, so the score is never negative", size=22, color=MUTED))
        ov_lab.arrange(DOWN, buff=0.12).next_to(overlaps, DOWN, buff=0.18).set_x(row_t.get_center()[0])

        brace = Brace(eq1[9:16], DOWN, buff=0.1).set_fill(QUERY_COLOR)
        b_lab = M(R"\text{score}(t,s)", 34, color=QUERY_COLOR).next_to(brace, DOWN, buff=0.1)
        hl_t, hl_s = (17, 7)
        row_hl = Rectangle(width=TT * mp, height=mp).set_stroke(QUERY_COLOR, 2).move_to(
            [m_center[0], m_y0 - hl_t * mp, 0])
        col_hl = Rectangle(width=mp, height=TT * mp).set_stroke(INK, 2).move_to([m_x0 + hl_s * mp, m_center[1], 0])
        hit = mcell[(hl_t, hl_s)].copy().set_fill(QUERY_COLOR, 1)

        # decay
        eq2 = M(R"\text{score}(t,s) = (x_t \cdot x_s)\,\times\,0.95^{t-s}", 44).move_to(eq1)
        eq2.shift((eq1[9:16].get_y() - eq2[11:18].get_y()) * UP)
        eq2[18:].set_color(WARN)
        m_cap2 = VGroup(L("weighted by", size=22, color=MUTED), M(R"0.95^{t-s}", 30, color=WARN)).arrange(
            RIGHT, buff=0.15).move_to(m_cap)
        # the half-life boundary t - s = 13.5 drawn across the score pattern
        m_half = DashedLine([m_x0 - 0.5 * mp, m_y0 - 13.0 * mp, 0], [m_x0 + 14.0 * mp, m_y0 - 27.5 * mp, 0],
                            dash_length=0.08).set_stroke(WARN, 2.5)
        axes = line_chart([0, 40, 10], [0, 1, 0.5], width=5.6, height=2.8, x_label="distance t − s (tokens)",
                          y_label="weight", x_ticks=[0, 10, 20, 30, 40], y_ticks=[0, 0.5, 1], size=22)
        axes.move_to([3.15, -0.95, 0])
        curve = axes.get_graph(lambda d: DECAY ** d, x_range=[0, 40, 0.25]).set_stroke(WARN, 4)
        hp = axes.c2p(HALF_LIFE, 0.5)
        h_lines = VGroup(DashedLine(axes.c2p(0, 0.5), hp).set_stroke(MUTED, 2),
                         DashedLine(hp, axes.c2p(HALF_LIFE, 0)).set_stroke(MUTED, 2))
        h_dot = Dot(hp, radius=0.08, fill_color=WARN)
        h_lab = VGroup(L("half strength after 13.5 tokens", size=22, color=WARN),
                       M(R"0.95^{13.5} \approx 0.5", 30, color=MUTED)).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        h_lab.next_to(hp, UR, buff=0.2).shift(0.25 * UP)
        src_decay = source_note("decay 0.95: report §3, test_instrument_v2.py decay_mask · half-life: report §16")

        to_clear = VGroup(cells_all, neurons, feats, n_lab2, f_lab2, s_name, w_grp, tok_mobs, q_tok, q_now,
                          row_boxes, read_vec, read_note, read_lab[0])
        with self.voiceover(
            "On a GPU this becomes linear attention. The score between the current position t and an earlier "
            "position s is a content match: the dot product of their sparse, positive neuron codes. In this "
            "project a positional decay of 0.95 per token of distance weights it, so a memory fades to half its "
            "strength in about fourteen tokens.",
            # the voice reads a bare "0.95" as "zero. ninety five"
            spoken="On a G P U this becomes linear attention. The score between the current position t and an "
            "earlier position s is a content match: the dot product of their sparse, positive neuron codes. In "
            "this project a positional decay of zero point nine five per token of distance weights it, so a "
            "memory fades to half its strength in about fourteen tokens."
        ) as vo:
            self.play(*t_anims, FadeOut(to_clear), FadeOut(src_rec), ReplacementTransform(read_eq, eq1[0:4]),
                      FadeIn(eq1[4:], shift=0.2 * LEFT), run_time=1.1)
            self.play(FadeIn(la_cap, shift=0.1 * UP), ShowCreation(m_frame),
                      LaggedStartMap(FadeIn, mat, lag_ratio=0.004), FadeIn(m_top), FadeIn(m_left), FadeIn(m_cap),
                      run_time=1.4)
            vo.wait_until(2.95)
            self.play(FadeOut(la_cap), GrowFromCenter(brace), FadeIn(b_lab, shift=0.1 * DOWN), run_time=0.8)
            vo.wait_until(3.9)
            self.play(ShowCreation(row_hl), FadeIn(lab_t), FadeIn(row_t, shift=0.15 * UP), run_time=0.8)
            vo.wait_until(5.25)
            self.play(ShowCreation(col_hl), FadeIn(lab_s), FadeIn(row_s, shift=0.15 * UP), run_time=0.8)
            self.play(FadeIn(hit, scale=1.6), run_time=0.5)
            vo.wait_until(8.3)
            self.play(LaggedStartMap(FadeIn, overlaps, lag_ratio=0.3), run_time=0.9)
            vo.wait_until(10.25)
            self.play(FadeIn(ov_lab, shift=0.1 * UP), run_time=0.8)
            # --- decay
            vo.wait_until(12.0)
            self.play(FadeOut(VGroup(row_t, row_s, lab_t, lab_s, overlaps, ov_lab)),
                      FadeOut(VGroup(row_hl, col_hl, hit)),
                      ReplacementTransform(b_lab, eq2[0:10]), ReplacementTransform(eq1[9:16], eq2[11:18]),
                      FadeOut(VGroup(eq1[0:9], eq1[16:], brace)), FadeIn(eq2[10]), run_time=1.0)
            vo.wait_until(13.1)
            self.play(Write(eq2[18:]), run_time=0.9)
            self.play(*[c.animate.set_fill(MEMORY_COLOR, 0.03 + 0.92 * DECAY ** (t - s)) for (t, s), c in mcell.items()],
                      ReplacementTransform(m_cap, m_cap2), FadeIn(src_decay), run_time=1.4)
            self.play(FadeIn(axes), run_time=0.7)
            self.play(ShowCreation(curve), run_time=1.6)
            vo.wait_until(18.55)                       # "so a memory fades to half its strength"
            self.play(ShowCreation(h_lines), FadeIn(h_dot, scale=2), ShowCreation(m_half), run_time=0.9)
            self.play(FadeIn(h_lab, shift=0.1 * UP), run_time=0.7)
            vo.wait_until(20.35)                       # "in about fourteen tokens"
            self.play(Flash(h_dot, color=WARN, flash_radius=0.3), Indicate(h_lab[0], color=WARN, scale_factor=1.06),
                      run_time=0.9)

        # ============================================================ N1.3  bdh.py
        t_anims, title = retitle(title, "bdh.py: one layer")
        self.play(*t_anims, FadeOut(VGroup(eq2, mat, m_frame, m_top, m_left, m_cap2, m_half, axes, curve, h_lines,
                                           h_dot, h_lab, src_decay)), run_time=0.8)
        cb = code_block(LAYER_CODE, title="bdh.py:122-144  (Pathway, used unmodified)", size=20)
        cb.move_to([-2.35, -0.3, 0])
        src_lines = LAYER_CODE.split("\n")
        lines = code_line_groups(cb, LAYER_CODE)
        short = L("the whole file: 171 lines", size=22, color=MUTED).next_to(cb.panel, DOWN, buff=0.25).align_to(
            cb.panel, LEFT)
        src_code = source_note("bdh.py (Pathway, 2025): forward(), lines 109-151")

        # flow diagram mirroring the loop
        fx = 4.6
        top_lab = L("x: residual stream, width d", size=22, color=MUTED).move_to([fx, 2.85, 0])
        specs = [("encoder: d → N", 2.2), ("ReLU: keep the positive part", 1.45), ("attention: Q = K, no softmax", 0.7),
                 ("LayerNorm", -0.05), ("encoder_v, then ReLU", -0.8), ("× multiply the two codes", -1.55),
                 ("decoder: N → d, add to x", -2.3)]
        boxes = VGroup(*[flow_box(t, width=3.5).move_to([fx, y, 0]) for t, y in specs])
        arrows = VGroup(Arrow(top_lab.get_bottom(), boxes[0].get_top(), buff=0.06, thickness=2.5).set_fill(MUTED))
        for a, b in zip(boxes[:-1], boxes[1:]):
            arrows.add(Arrow(a.get_bottom(), b.get_top(), buff=0.04, thickness=2.5).set_fill(MUTED))
        skip = CurvedArrow(boxes[1].get_left() + 0.05 * LEFT, boxes[5].get_left() + 0.05 * LEFT, angle=PI / 4)
        skip.set_stroke(MEMORY_COLOR, 2.5).set_fill(MEMORY_COLOR)
        skip_lab = L("x_sparse", size=22, color=MEMORY_COLOR).rotate(PI / 2).next_to(skip, LEFT, buff=0.06)
        loop_pts = [boxes[6].get_right() + 0.05 * RIGHT, [6.55, boxes[6].get_y(), 0], [6.55, 2.85, 0],
                    top_lab.get_right() + 0.1 * RIGHT]
        loop = VMobject().set_points_as_corners([np.array(p, dtype=float) for p in loop_pts]).set_stroke(WARN, 3)
        loop_tip = Arrow(loop_pts[2], loop_pts[3], buff=0, thickness=3).set_fill(WARN)
        loop_lab = L("3 layers, the same weights each time", size=22, color=WARN).move_to([fx, -3.05, 0])

        hl_w = cb.panel.get_width() - 0.15

        def hl_boxes(idx):
            runs, cur = [], [idx[0]]
            for i in idx[1:]:
                if i == cur[-1] + 1:
                    cur.append(i)
                else:
                    runs.append(cur)
                    cur = [i]
            runs.append(cur)
            g = VGroup()
            for run in runs:
                grp = VGroup(*[lines[i] for i in run])
                r = Rectangle(width=hl_w, height=grp.get_height() + 0.12)
                g.add(r.set_fill(WARN, 0.13).set_stroke(WARN, 1.2).move_to(grp).set_x(cb.panel.get_x()))
            return g

        hl = hl_boxes([1])

        def focus(idx):
            """Highlight code lines `idx` (and dim the rest)."""
            anims = [Transform(hl, hl_boxes(idx))]
            for i, ln in enumerate(lines):
                anims.append(ln.animate.set_opacity(1.0 if i in idx else 0.35))
            return anims

        def show_box(k, with_arrow=True):
            anims = [FadeIn(boxes[k], shift=0.15 * DOWN), boxes[k][0].animate.set_stroke(WARN, 2.5)]
            if with_arrow:
                anims.append(GrowArrow(arrows[k]))
            if k > 0:
                anims.append(boxes[k - 1][0].animate.set_stroke(PANEL_EDGE, 1.5))
            return anims

        with self.voiceover(
            "The public implementation, bdh dot py, is short. Each layer encodes the residual stream into N neurons "
            "and keeps the positive part. It attends with queries tied to keys and no softmax, normalizes the "
            "result, passes it through a second sparse code, multiplies the two codes, and decodes back. The same "
            "weights serve every layer."
        ) as vo:
            self.play(FadeIn(cb.panel), FadeIn(cb.tab), LaggedStartMap(FadeIn, VGroup(*lines), lag_ratio=0.08),
                      FadeIn(src_code), run_time=1.6)
            vo.wait_until(2.3)
            self.play(FadeIn(short, shift=0.1 * UP), FadeIn(top_lab), run_time=0.7)
            vo.wait_until(3.7)
            self.play(FadeIn(hl), *focus([1])[1:], *show_box(0), run_time=0.8)
            vo.wait_until(7.15)
            self.play(*focus([2]), *show_box(1), run_time=0.7)
            vo.wait_until(8.85)
            self.play(*focus([3]), *show_box(2), run_time=0.7)
            self.play(Indicate(code_sub(src_lines[3], lines[3], "Q=x_sparse, K=x_sparse"), color=WARN), run_time=1.0)
            vo.wait_until(12.35)
            self.play(*focus([4]), *show_box(3), run_time=0.6)
            vo.wait_until(14.05)
            self.play(*focus([5, 6]), *show_box(4), run_time=0.7)
            vo.wait_until(16.4)
            self.play(*focus([7]), *show_box(5), ShowCreation(skip), FadeIn(skip_lab), run_time=0.8)
            vo.wait_until(18.0)                        # "and decodes back"
            self.play(*focus([9, 10, 11, 12, 13]), *show_box(6), run_time=0.8)
            vo.wait_until(19.2)
            w_names = VGroup(code_sub(src_lines[1], lines[1], "self.encoder"),
                             code_sub(src_lines[5], lines[5], "self.encoder_v"),
                             code_sub(src_lines[10], lines[10], "self.decoder"))
            self.play(*focus([0, 1, 5, 10]), boxes[6][0].animate.set_stroke(PANEL_EDGE, 1.5),
                      ShowCreation(loop), FadeIn(loop_tip), run_time=0.8)
            self.play(FadeIn(loop_lab, shift=0.1 * UP),
                      *[Indicate(w, color=WARN, scale_factor=1.15) for w in w_names], run_time=1.0)

        # ============================================================ N1.4  this project's size
        t_anims, title = retitle(title, "The size used here")
        self.play(*t_anims, FadeOut(VGroup(cb.panel, cb.tab, *lines, hl, short, src_code)), run_time=0.7)
        diagram = VGroup(top_lab, boxes, arrows, skip, skip_lab, loop, loop_tip, loop_lab)
        lx = -4.55
        layers = VGroup(*[flow_box(f"BDH layer {k}", width=3.0, height=0.6, size=24) for k in (3, 2, 1)])
        layers.arrange(DOWN, buff=0.35).move_to([lx, 1.15, 0])
        lay_arrows = VGroup(*[Arrow(layers[k + 1].get_top(), layers[k].get_bottom(), buff=0.04, thickness=2.5)
                              .set_fill(MUTED) for k in range(2)])
        one_layer = flow_box("one BDH layer", width=3.0, height=0.6, size=24).move_to(layers[1])
        lay_note = L("one set of weights, used by all three", size=22, color=MUTED).next_to(layers, DOWN, buff=0.25)

        rows = [["", "bdh.py default", "this project"],
                ["layers", "6", "3"],
                ["neurons N (per head)", "8,192", "256"],
                ["embedding width d", "256", "32"],
                ["heads", "4", "1"],
                ["positions", "RoPE", "decay 0.95"]]
        table = simple_table(rows, col_widths=[2.9, 2.3, 2.2], size=22, row_height=0.52)
        table.move_to([2.85, 1.2, 0])
        for r in range(1, 6):
            table.cells[r][1].set_color(MUTED)
            table.cells[r][2].set_color(WARN).set_opacity(1)
        table.cells[0][2].set_color(WARN)
        here_col = VGroup(*[table.cells[r][2] for r in range(6)])
        rest = VGroup(table.rules, *[VGroup(table.cells[r][0], table.cells[r][1]) for r in range(6)])
        src_size = source_note("bdh.py:11-18 · report §3 · parameter counts: MultiBDH (test_multilayer_binding.py) arms B, A, D8")

        # parameter bar
        unit = 10.6 / 28096
        segs_spec = [("encoder", 8192, "#3F6E8C"), ("encoder_v", 8192, "#4C82A3"), ("decoder", 8192, "#5A96BA"),
                     ("embedding + readout", 1408, "#6B7280"), ("gate", 2112, GATE_COLOR)]
        segs = VGroup()
        x = -5.75
        for nm, n, col in segs_spec:
            w = n * unit
            r = Rectangle(width=w, height=0.55).set_fill(col, 0.95).set_stroke(BG, 1.5)
            r.move_to([x + w / 2, -2.2, 0])
            segs.add(r)
            x += w
        seg_labs = VGroup()
        for (nm, n, col), r in zip(segs_spec[:3], segs[:3]):
            seg_labs.add(L(f"{nm}  {n:,}", size=22, color=INK).move_to(r))
        emb_lab = L("embedding + readout 1,408", size=22, color=MUTED)
        emb_lab.next_to(segs[3], DOWN, buff=0.18).align_to(segs[3], RIGHT)
        gate_lab = L("gate 2,112", size=22, color=GATE_COLOR, weight="BOLD")
        gate_lab.next_to(segs[4], RIGHT, buff=0.2).shift(0.1 * UP)
        core = VGroup(*segs[:4])
        br1 = Brace(core, UP, buff=0.08)
        br1_lab = L("25,984 parameters, about 26 thousand", size=24, color=INK).next_to(br1, UP, buff=0.08)
        br2 = Brace(segs, UP, buff=0.08)
        br2_lab = L("with the gate: 28,096 at 2 streams, 29,056 at 8 streams with 16 channels", size=22,
                    color=GATE_COLOR).next_to(br2, UP, buff=0.08)
        next_lab = L("next chapter", size=22, color=GATE_COLOR).next_to(gate_lab, DOWN, buff=0.08).align_to(gate_lab, LEFT)

        row_box = row_highlight(table, 1)

        with self.voiceover(
            "The repository uses that file unmodified, at a tiny size: three layers, two hundred fifty-six neurons, "
            "an embedding width of thirty-two, one head. About twenty-six thousand parameters, and twenty-eight to "
            "twenty-nine thousand once the gate we meet next is added."
        ) as vo:
            self.play(FadeOut(VGroup(arrows, skip, skip_lab, loop, loop_tip, loop_lab, top_lab)),
                      ReplacementTransform(boxes, one_layer), run_time=1.0)
            self.play(FadeIn(rest, shift=0.15 * UP), FadeIn(src_size), run_time=0.9)
            vo.wait_until(3.0)
            self.play(LaggedStartMap(FadeIn, here_col, shift=0.25 * LEFT, lag_ratio=0.1), run_time=1.0)
            vo.wait_until(4.65)
            self.play(FadeIn(row_box), ReplacementTransform(VGroup(one_layer, one_layer.copy(), one_layer.copy()), layers),
                      LaggedStartMap(GrowArrow, lay_arrows), FadeIn(lay_note), run_time=0.8)
            for r, t in ((2, 5.55), (3, 7.5), (4, 9.25), (5, 9.85)):
                vo.wait_until(t)
                self.play(Transform(row_box, row_highlight(table, r)), run_time=0.45)
            vo.wait_until(10.05)
            self.play(FadeOut(row_box), LaggedStart(*[GrowFromEdge(s, LEFT) for s in segs[:4]], lag_ratio=0.35),
                      run_time=1.3)
            self.play(LaggedStartMap(FadeIn, seg_labs, lag_ratio=0.2), FadeIn(emb_lab), GrowFromCenter(br1),
                      FadeIn(br1_lab, shift=0.1 * UP), run_time=0.8)
            vo.wait_until(12.4)
            self.play(GrowFromEdge(segs[4], LEFT), FadeIn(gate_lab), run_time=0.8)
            self.play(ReplacementTransform(br1, br2), FadeOut(br1_lab, shift=0.1 * UP), FadeIn(br2_lab, shift=0.1 * UP),
                      run_time=0.9)
            vo.wait_until(14.65)
            self.play(Indicate(segs[4], color=GATE_COLOR, scale_factor=1.12), FadeIn(next_lab), run_time=1.0)
        self.wait(1.0)
        self.clear_all()
