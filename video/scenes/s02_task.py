"""Chapter 2 — the context-conditional binding task (report §3; facts_task §1-2)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403

# First training sequence of seed 280 (slow-start Part 0's first seed), S=2, P=4, grouped layout.
# token ids [1,2,19, 0,2,8, 0,5,17, 1,5,8, 0,4,16, 1,4,12, 0,3,18, 1,3,12, 0,2,8]
SEQ = [(1, 0, 13), (0, 0, 2), (0, 3, 11), (1, 3, 2), (0, 2, 10), (1, 2, 6), (0, 1, 12), (1, 1, 6)]
QUERY = (0, 0)   # CTX0 KEY0 ?
ANSWER = 2


class BindingTask(ClankersScene):
    def construct(self):
        card = self.chapter_card(2, "Context-conditional binding")
        self.wait(0.6)
        title = section_title("The binding task")
        self.play(FadeOut(card, shift=0.3 * UP), FadeIn(title, shift=0.3 * UP))

        # ---------------------------------------------------------------- N2.1 the ingredients
        ctx_row = VGroup(token("CTX0", "ctx", 0), token("CTX1", "ctx", 1)).arrange(RIGHT, buff=0.15)
        key_row = VGroup(*[token(f"K{i}") for i in range(4)]).arrange(RIGHT, buff=0.15)
        val_row = VGroup(*[token(f"V{j}", "val", None, width=0.62, height=0.42, size=16)
                           for j in range(16)]).arrange_in_grid(2, 8, buff=0.08)
        groups = VGroup()
        for label, row, note in [("S streams", ctx_row, "one context token each"),
                                 ("P keys", key_row, "shared by every stream"),
                                 ("16 values", val_row, "drawn per key and stream")]:
            head = L(label, size=28, weight="BOLD")
            sub = L(note, size=20, color=MUTED)
            col = VGroup(head, row, sub).arrange(DOWN, buff=0.3)
            groups.add(col)
        groups.arrange(RIGHT, buff=1.0, aligned_edge=UP).move_to(0.6 * UP)

        binding = VGroup(
            L("K0 in stream 0", size=24, color=STREAM_COLORS[0]), Arrow(LEFT, RIGHT, buff=0).set_width(0.8),
            token("V2", "val", 0),
        ).arrange(RIGHT, buff=0.25)
        binding2 = VGroup(
            L("K0 in stream 1", size=24, color=STREAM_COLORS[1]), Arrow(LEFT, RIGHT, buff=0).set_width(0.8),
            token("V13", "val", 1),
        ).arrange(RIGHT, buff=0.25)
        bind_grp = VGroup(binding, binding2).arrange(DOWN, buff=0.25, aligned_edge=LEFT).next_to(groups, DOWN, buff=0.7)
        same_key = T("same key, a different value in every stream", size=26, color=WARN)
        same_key.next_to(bind_grp, DOWN, buff=0.3)

        with self.voiceover(
            "Here is the task the whole project is built on. There are S streams, each announced by its own "
            "context token, and P keys. For every key, each stream gets its own value, drawn from sixteen value "
            "tokens. So the same key means something different in every stream."
        ) as vo:
            self.play(FadeIn(groups[0], shift=0.2 * UP))
            vo.wait_until_sentence(1)
            self.play(FadeIn(groups[1], shift=0.2 * UP), run_time=0.8)
            vo.wait_until(vo.time_of(1) + 2.2)
            self.play(LaggedStartMap(FadeIn, val_row, lag_ratio=0.05), FadeIn(groups[2][0]), FadeIn(groups[2][2]))
            vo.wait_until_sentence(2)
            self.play(FadeIn(bind_grp, shift=0.2 * DOWN))
            self.play(Write(same_key), run_time=1.0)

        self.play(FadeOut(VGroup(groups, bind_grp, same_key)), run_time=0.7)

        # ---------------------------------------------------------------- N2.2 the sequence
        triples = VGroup(*[triple(s, k, v) for s, k, v in SEQ])
        query = VGroup(token("CTX0", "ctx", 0), token("K0"), token("?", "query")).arrange(RIGHT, buff=0.06)
        allt = VGroup(*triples, query)
        # first: one long strip (it IS a single sequence of 27 tokens)
        strip = allt.copy()
        strip.arrange(RIGHT, buff=0.12)
        strip.set_width(13.2).move_to(2.45 * UP)
        brace = Brace(strip, DOWN, buff=0.12)
        brace_lab = L("27 tokens: 8 triples and one query", size=22, color=MUTED).next_to(brace, DOWN, buff=0.1)

        # then: grouped by key, two triples per row, query below
        rows = VGroup(*[VGroup(triples[2 * r], triples[2 * r + 1]).copy() for r in range(4)])
        for row in rows:
            row.arrange(RIGHT, buff=0.45)
        rows.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        q_copy = query.copy()
        layout = VGroup(rows, q_copy)
        q_copy.next_to(rows, DOWN, buff=0.55, aligned_edge=LEFT)
        layout.move_to(0.75 * DOWN + 1.2 * LEFT)
        key_labels = VGroup(*[L(f"key K{SEQ[2 * r][1]}", size=20, color=MUTED).next_to(rows[r], RIGHT, buff=0.4)
                              for r in range(4)])
        q_label = L("query: which value did K0 have in stream 0?", size=20, color=QUERY_COLOR)
        q_label.next_to(q_copy, RIGHT, buff=0.4)
        src = source_note("first training sequence of seed 280 (test_binding_capacity.BindTask)")

        with self.voiceover(
            "The sequence is a list of triples: context, key, value. In the grouped layout, the triples for one "
            "key sit together; the order of the keys is random, and so is the order of the streams within each "
            "key's group. At the end comes a query: a context token, a key, and a blank to fill in."
        ) as vo:
            self.play(LaggedStartMap(FadeIn, strip, shift=0.15 * RIGHT, lag_ratio=0.04), run_time=2.0)
            self.play(GrowFromCenter(brace), FadeIn(brace_lab), FadeIn(src))
            vo.wait_until_sentence(1)
            self.play(*[TransformFromCopy(strip[i], rows[i // 2][i % 2]) for i in range(8)],
                      TransformFromCopy(strip[8], q_copy), FadeOut(VGroup(brace, brace_lab)),
                      strip.animate.set_opacity(0.35), run_time=1.8)
            self.play(LaggedStartMap(FadeIn, key_labels, lag_ratio=0.2), run_time=1.2)
            vo.wait_until_sentence(2)
            self.play(Indicate(q_copy, color=QUERY_COLOR, scale_factor=1.1), FadeIn(q_label, shift=0.1 * RIGHT))

        # ---------------------------------------------------------------- N2.3 the answer
        k0_a, k0_b = rows[0][0], rows[0][1]       # CTX1 K0 V13 , CTX0 K0 V2
        hl_a = SurroundingRectangle(k0_a, buff=0.08).set_stroke(STREAM_COLORS[1], 2)
        hl_b = SurroundingRectangle(k0_b, buff=0.08).set_stroke(STREAM_COLORS[0], 3)
        answer = token("V2", "val", 0).move_to(q_copy[2])
        link = CurvedArrow(q_copy[2].get_left() + 0.05 * UP, k0_b.get_bottom(), angle=-PI / 3)
        link.set_stroke(STREAM_COLORS[0], 3)

        with self.voiceover(
            "With two streams and four keys that is twenty-seven tokens. Here the query asks for key zero in "
            "stream zero. Key zero appears twice: value thirteen in stream one, and value two in stream zero. "
            "The answer is value two."
        ) as vo:
            vo.wait_until_sentence(1)
            self.play(Indicate(rows[0], color=WARN, scale_factor=1.04), run_time=0.8)
            vo.wait_until_sentence(2)
            self.play(ShowCreation(hl_a), ShowCreation(hl_b), key_labels[0].animate.set_color(WARN))
            vo.wait_until_sentence(3)
            self.play(ShowCreation(link), run_time=0.8)
            self.play(Transform(q_copy[2], answer), FadeOut(hl_a))

        self.play(FadeOut(VGroup(strip, rows, q_copy, key_labels, q_label, link, hl_b, src)), run_time=0.7)

        # ---------------------------------------------------------------- N2.4 three reference levels
        line = NumberLine(x_range=[0, 1, 0.25], width=11, include_tip=False).set_stroke(MUTED, 2)
        line.move_to(0.4 * DOWN)
        ends = VGroup(L("0", size=22, color=MUTED).next_to(line.n2p(0), DOWN, buff=0.2),
                      L("1", size=22, color=MUTED).next_to(line.n2p(1), DOWN, buff=0.2))
        axis_lab = L("accuracy on the query", size=22, color=MUTED).next_to(line, DOWN, buff=0.75)

        def marker(x, top, bottom, color):
            tick = Line(line.n2p(x) + 0.25 * DOWN, line.n2p(x) + 0.25 * UP).set_stroke(color, 4)
            t1 = L(top, size=26, color=color, weight="BOLD")
            t2 = L(bottom, size=20, color=MUTED)
            lab = VGroup(t1, t2).arrange(DOWN, buff=0.12).next_to(tick, UP, buff=0.2)
            return VGroup(tick, lab)

        m_guess = marker(1 / 16, "1/16", "guess a value", MUTED)
        m_blind = marker(0.5, "1/S = 1/2", "binds keys, blind to streams", BAD)
        m_solve = marker(1.0, "1", "resolves by stream", GOOD)
        m_guess[1].shift(0.35 * RIGHT)

        with self.voiceover(
            "There are three reference levels. Guessing among the values gives one in sixteen. A model that binds "
            "keys to values but cannot tell the streams apart has S candidates and scores one over S: here, one "
            "half. Only a model that resolves the conflict by stream reaches one."
        ) as vo:
            self.play(ShowCreation(line), FadeIn(ends), FadeIn(axis_lab))
            vo.wait_until_sentence(1)
            self.play(FadeIn(m_guess, shift=0.2 * DOWN))
            vo.wait_until_sentence(2)
            self.play(FadeIn(m_blind, shift=0.2 * DOWN))
            vo.wait_until_sentence(3)
            self.play(FadeIn(m_solve, shift=0.2 * DOWN))

        # ---------------------------------------------------------------- N2.5 the trap
        trap = SurroundingRectangle(m_blind, buff=0.15).set_stroke(BAD, 2)
        grid = memory_grid(5, 5, cell=0.3, color=MEMORY_COLOR, label="one memory")
        grid.to_edge(LEFT, buff=1.0).shift(0.2 * DOWN)
        cells = grid.cells
        for i in (6, 7, 11, 12, 13, 17):
            cells[i].set_fill(STREAM_COLORS[0], 0.55)
        for i in (7, 8, 12, 13, 18):
            cells[i].set_fill(STREAM_COLORS[1], 0.55)
        for i in (7, 12, 13):
            cells[i].set_fill(interpolate_color(STREAM_COLORS[0], STREAM_COLORS[1], 0.5), 0.8)
        writes = VGroup(
            VGroup(token("K0"), Arrow(LEFT, RIGHT, buff=0).set_width(0.5), token("V2", "val", 0)).arrange(RIGHT, buff=0.1),
            VGroup(token("K0"), Arrow(LEFT, RIGHT, buff=0).set_width(0.5), token("V13", "val", 1)).arrange(RIGHT, buff=0.1),
        ).arrange(DOWN, buff=0.2).next_to(grid, RIGHT, buff=0.6)
        read = VGroup(L("read K0 →", size=22), gate_bar([0.5, 0.5], width=1.4),
                      L("V2 or V13?", size=22, color=BAD)).arrange(RIGHT, buff=0.2)
        read.next_to(writes, RIGHT, buff=0.7)
        stat = VGroup(L("single-channel runs that bound (no convolution)", size=22, color=MUTED),
                      L("0 of more than 200", size=34, color=BAD, weight="BOLD"),
                      L("S = 2, P = 4, no convolution", size=18, color=FAINT)).arrange(DOWN, buff=0.12)
        stat.to_corner(DR, buff=0.7).shift(0.5 * UP)
        lengths = simple_table([["S, P", "tokens"], ["2, 4", "27"], ["2, 8", "51"], ["4, 4", "51"], ["8, 4", "99"]],
                               col_widths=[1.4, 1.2], size=20, row_height=0.42)
        lengths.next_to(stat, UP, buff=0.55).align_to(stat, RIGHT)
        len_lab = L("length 3(SP + 1)", size=18, color=MUTED).next_to(lengths, DOWN, buff=0.15)

        with self.voiceover(
            "That middle level is a trap. A single memory stores both bindings of a key in the same place, and in "
            "the basic setting, without the convolution we meet later, not one of more than two hundred "
            "single-channel runs escaped it."
        ) as vo:
            self.play(ShowCreation(trap))
            mob_line = VGroup(line, ends, axis_lab, m_guess, m_blind, m_solve, trap)
            self.play(mob_line.animate.scale(0.75).to_edge(UP, buff=1.2), run_time=1.0)
            self.play(FadeIn(grid), LaggedStartMap(FadeIn, writes, shift=0.2 * RIGHT, lag_ratio=0.4))
            self.play(FadeIn(read, shift=0.2 * RIGHT))
            vo.wait_until(vo.time_of(0) + vo.duration * 0.55)
            self.play(FadeIn(stat, shift=0.2 * UP), FadeIn(source_note("Report §13, Table 8")))
            self.play(FadeIn(lengths), FadeIn(len_lab))
        self.wait(0.5)
        self.clear_all()
