"""Chapter 20 — distant cues: the header layout (report §12).

Numbers: report §12 (screens S6, S10, S15, S17, S23 on seeds 160-169); the screens' records are on the branch
claude/outside-ideas (explore_out/README.md), not in this checkout.
  S6  far_ceil (perfect gate) bound 3/3; far_A (plain gate) discovered 0/10
  S10 far_nudge (5% labelled nudge) discovered 5/10;  S15 far_nudge_slow 10/10
  S17 FAR_SLOW_HINGE (slow memory + hinge) discovered 0/10, failing by key splits
  S23 one channel on the header layout: B_FAR bound 3/10, B_FAR_SLOW (slow memory) 7/10
Header layout (explore_far_cue.py on claude/outside-ideas): each stream's block is [CTX_s, K, V, K, V, ...],
blocks in random stream order, keys random within a block, then the query [CTX_q, K_q, ?]. At S = 2, P = 4
the sequence has 21 tokens, and the key of pair i sits 1 + 2i positions after its block's stream token.
The token sequence drawn here is an example built by those rules, not a recorded batch.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403

B, Y = STREAM_COLORS[0], STREAM_COLORS[1]

# example header-layout sequence, S = 2, P = 4: (key, value) pairs per block, blocks in order 0, 1
BLOCKS = [(0, [(2, 9), (1, 12), (3, 5), (0, 7)]),
          (1, [(3, 14), (0, 3), (2, 1), (1, 6)])]
QUERY = (0, 0)                    # CTX0 K0 ?  -> V7
DOOR_LINE = {0: 3, 1: 1}          # which line of each "conversation" holds the door code (pair index)
DOOR_VAL = {0: 7, 1: 3}

# report §12 counts
S6_PLAIN, S6_CEIL = (0, 10), (3, 3)
S17 = (0, 10)
S10, S15 = (5, 10), (10, 10)
S23, S23_SLOW = (3, 10), (7, 10)

ROW_Y = 1.05
CTX_W, KV_W, TOK_H, TOK_BUFF = 0.8, 0.5, 0.5, 0.05


# ------------------------------------------------------------------------------------- helpers
def at_phrase(vo, i, phrase):
    """Approximate time (s into the block) at which `phrase` is spoken in sentence i."""
    text, a, b = vo.synth.sentences[i]
    k = text.find(phrase)
    if k < 0:
        raise ValueError(phrase)
    return a + (b - a) * k / max(len(text), 1)


def tok(label, kind="key", stream=None):
    return token(label, kind, stream, width=CTX_W if kind == "ctx" else KV_W, height=TOK_H, size=20)


def dots(n_good, n, color, r=0.1, buff=0.08):
    g = VGroup()
    for i in range(n):
        c = Circle(radius=r)
        if i < n_good:
            c.set_fill(color, 0.95).set_stroke(color, 2)
        else:
            c.set_fill(BAD, 0.0).set_stroke(BAD, 2)
        g.add(c)
    return g.arrange(RIGHT, buff=buff)


def result_row(name, screen, k, n, word, color, y):
    """'name (Sxx)   o o o o o o o o o o   k/n word' on one line at height y."""
    lab = VGroup(L(name, 22, INK), L(f"({screen})", 20, MUTED)).arrange(RIGHT, buff=0.15)
    lab.move_to([-6.3, y, 0], aligned_edge=LEFT)
    d = dots(k, n, color)
    d.move_to([-1.35, y, 0], aligned_edge=LEFT)
    cnt = L(f"{k}/{n} {word}", 22, color if k else BAD, weight="BOLD")
    cnt.move_to([1.65, y, 0], aligned_edge=LEFT)
    g = VGroup(lab, d, cnt)
    g.lab, g.dots, g.cnt = lab, d, cnt
    return g


def chip(text, color, size=20):
    t = L(text, size, color, weight="BOLD")
    box = RoundedRectangle(width=t.get_width() + 0.3, height=t.get_height() + 0.2, corner_radius=0.1)
    box.set_fill(color, 0.12).set_stroke(color, 1.5)
    t.move_to(box)
    return VGroup(box, t)


def doc_card(stream, door_line, door_val, x):
    """A tiny 'conversation': a coloured header, then four lines, one of which holds the door code."""
    c = STREAM_COLORS[stream]
    head_box = RoundedRectangle(width=3.0, height=0.5, corner_radius=0.1).set_fill(c, 0.9).set_stroke(c, 2)
    head_txt = L(f"conversation {stream}", 22, BG, weight="BOLD").move_to(head_box)
    head = VGroup(head_box, head_txt)
    lines = VGroup()
    widths = [2.6, 2.2, 2.75, 2.4]
    for i in range(4):
        if i == door_line:
            ln = VGroup(L("door code:", 24, KEY_COLOR), L(str(door_val), 24, c, weight="BOLD")).arrange(RIGHT, buff=0.15)
        else:
            ln = VGroup(Rectangle(width=widths[i] * 0.55, height=0.12), Rectangle(width=widths[i] * 0.4, height=0.12))
            ln.arrange(RIGHT, buff=0.12)
            ln.set_fill(FAINT, 1).set_stroke(width=0)
        lines.add(ln)
    lines.arrange(DOWN, buff=0.32, aligned_edge=LEFT)
    inner = VGroup(head, lines).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
    lines.shift(0.15 * RIGHT)
    bg = SurroundingRectangle(inner, buff=0.25).round_corners(0.15).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1.5)
    g = VGroup(bg, head, lines)
    g.move_to([x, 0.45, 0])
    g.bg, g.head, g.lines = bg, head, lines
    return g


def reach_arrow(start, end, color, angle):
    """A curved arrow kept as VGroup(path, tip), so path and tip can be dimmed separately."""
    arc = ArcBetweenPoints(start, end, angle=angle)
    arc.set_stroke(color, 2.5)
    arc.add_tip(length=0.12, width=0.12)
    tip = arc.tip
    arc.remove(tip)
    tip.set_fill(color, 1).set_stroke(width=0)
    g = VGroup(arc, tip)
    g.path, g.tip = arc, tip
    return g


# ------------------------------------------------------------------------------------- scene
class DistantCues(ClankersScene):
    def construct(self):
        chap = self.chapter_card(20, "Distant cues")
        self.wait(0.6)
        title = section_title("Distant cues")
        self.play(FadeOut(chap, shift=0.3 * UP), FadeIn(title, shift=0.3 * UP))

        # ---------------------------------------------------------------- objects: the header row
        blocks = VGroup()
        for s, pairs in BLOCKS:
            ctx = tok(f"CTX{s}", "ctx", s)
            prs = VGroup(*[VGroup(tok(f"K{k}"), tok(f"V{v}", "val", s)).arrange(RIGHT, buff=TOK_BUFF)
                           for k, v in pairs])
            blk = VGroup(ctx, prs)
            VGroup(ctx, *prs).arrange(RIGHT, buff=TOK_BUFF)
            blk.ctx, blk.pairs, blk.stream = ctx, prs, s
            blocks.add(blk)
        q = VGroup(tok(f"CTX{QUERY[0]}", "ctx", QUERY[0]), tok(f"K{QUERY[1]}"), tok("?", "query"))
        q.arrange(RIGHT, buff=TOK_BUFF)
        row = VGroup(blocks[0], blocks[1], q)
        blocks.arrange(RIGHT, buff=0.28)
        q.next_to(blocks, RIGHT, buff=0.4)
        row.move_to([0, ROW_Y, 0])
        keys = [blk.pairs[i][0] for blk in blocks for i in range(4)]
        key_stream = [blk.stream for blk in blocks for i in range(4)]
        key_id = [BLOCKS[b][1][i][0] for b in range(2) for i in range(4)]

        layout_lab = L("header layout · S = 2, P = 4 · 21 tokens", 22, MUTED)
        layout_lab.to_corner(UR, buff=0.5).shift(0.08 * UP)

        # ---------------------------------------------------------------- N20.1 (sentence 0) language model
        docs = VGroup(doc_card(0, DOOR_LINE[0], DOOR_VAL[0], -3.1), doc_card(1, DOOR_LINE[1], DOOR_VAL[1], 3.1))
        doc_arrows = VGroup()
        for s_, (d, sign) in enumerate(((docs[0], -1), (docs[1], 1))):
            line = d.lines[DOOR_LINE[s_]]
            side = LEFT if sign < 0 else RIGHT
            start = line.get_corner(side + UP) * np.array([0, 1, 0]) + d.bg.get_edge_center(side) * np.array([1, 0, 0]) \
                + side * 0.12 + 0.12 * DOWN
            end = d.head.get_corner(side + DOWN) * np.array([0, 1, 0]) + d.bg.get_edge_center(side) * np.array([1, 0, 0]) \
                + side * 0.12 + 0.18 * UP
            doc_arrows.add(reach_arrow(start, end, STREAM_COLORS[s_], angle=sign * TAU / 4))
        cue_caption = T("the cue comes once, far before the tokens it governs", 26, INK)
        cue_caption.move_to([0, -2.05, 0])
        q_hint = VGroup(L("later:", 22, MUTED), L("conversation 0, door code?", 22, QUERY_COLOR)).arrange(RIGHT, buff=0.15)
        q_hint.move_to([0, -2.75, 0])

        # ---------------------------------------------------------------- N20.1 (sentence 1) header layout
        braces = VGroup()
        for mob, text in ((blocks[0], "stream 0's block"), (blocks[1], "stream 1's block"), (q, "query")):
            br = Brace(mob, DOWN, buff=0.1).set_color(MUTED)
            braces.add(VGroup(br, L(text, 20, MUTED).next_to(br, DOWN, buff=0.08)))

        arcs = VGroup()
        dist_nums = VGroup()
        for blk in blocks:
            c = STREAM_COLORS[blk.stream]
            for i in range(4):
                key = blk.pairs[i][0]
                start = key.get_top() + 0.04 * UP
                end = blk.ctx.get_top() + (0.3 - 0.2 * i) * RIGHT + 0.04 * UP
                arcs.add(reach_arrow(start, end, c, angle=TAU / 3))
                dist_nums.add(L(str(1 + 2 * i), 22, c, weight="BOLD").next_to(key, DOWN, buff=0.12))
        dist_cap = L("tokens back from each key to its stream token   (grouped layout: always 1)", 20, MUTED)
        dist_cap.move_to([0, 0.08, 0])

        # ---------------------------------------------------------------- the gate's choice at each key
        squares = VGroup(*[Square(0.3).set_fill(STREAM_COLORS[key_stream[i]], 0.92).set_stroke(INK, 1, opacity=0.5)
                           .move_to([k.get_x(), ROW_Y - 0.62, 0]) for i, k in enumerate(keys)])
        sq_lab = L("square under a key: the channel the gate sends it to", 20, MUTED)
        sq_lab.move_to([0, 0.08, 0])
        state_y = -0.4

        def recolor(rule, run_time=1.0):
            return [sq.animate.set_fill(rule(i), 0.92) for i, sq in enumerate(squares)]

        by_stream = lambda i: STREAM_COLORS[key_stream[i]]            # noqa: E731
        by_key = lambda i: STREAM_COLORS[0 if key_id[i] in (0, 2) else 1]  # noqa: E731

        rows_y = [-1.0, -1.52, -2.04, -2.56, -3.08]
        r_plain = result_row("plain gate", "S6", *S6_PLAIN, "discovered", GOOD, rows_y[0])
        r_ceil = result_row("perfect gate", "S6", *S6_CEIL, "bound", GOOD, rows_y[1])
        r_rec = result_row("recipe: slow memory + hinge", "S17", *S17, "discovered", GOOD, rows_y[2])
        r_nudge = result_row("5% labelled nudge", "S10", *S10, "discovered", WARN, rows_y[3])
        r_nslow = result_row("nudge + slow memory", "S15", *S15, "discovered", WARN, rows_y[4])
        src = source_note("Report §12 (screens S6, S10, S15, S17, S23; seeds 160–169)")
        src_layout = source_note("explore_far_cue.py (header layout); example sequence")

        with self.voiceover(
            "In a language model, the cues that set a context are usually far from the tokens they govern. "
            "The header layout tests that: each stream's context token appears once, at the start of its block. "
            "The plain gate discovered the routing on none of ten, while the perfect gate bound three of three."
        ) as vo:
            self.play(LaggedStartMap(FadeIn, docs, shift=0.2 * UP, lag_ratio=0.3), run_time=1.2)
            self.play(*[Indicate(d.head, color=WHITE, scale_factor=1.05) for d in docs], run_time=0.8)
            self.play(*[ShowCreation(a) for a in doc_arrows], run_time=1.0)
            self.play(*[Indicate(d.lines[DOOR_LINE[i]], color=STREAM_COLORS[i]) for i, d in enumerate(docs)],
                      FadeIn(cue_caption, shift=0.1 * UP), run_time=0.9)
            self.play(FadeIn(q_hint, shift=0.1 * UP), run_time=0.6)

            # the conversations become the header layout's blocks
            vo.wait_until_sentence(1)
            anims = [FadeOut(doc_arrows), FadeOut(cue_caption)]
            for d, blk in zip(docs, blocks):
                anims.append(FadeTransform(d.head, blk.ctx))
                order = [i for i in range(4)]
                for i in order:
                    anims.append(FadeTransform(d.lines[i], blk.pairs[i]))
                anims.append(FadeOut(d.bg))
            self.play(*anims, FadeIn(layout_lab), FadeIn(src_layout), run_time=1.2)
            self.play(FadeTransform(q_hint, q), LaggedStartMap(FadeIn, braces, lag_ratio=0.25), run_time=0.9)
            vo.wait_until(at_phrase(vo, 1, "appears once") - 0.4)
            self.play(*[FlashAround(blk.ctx, color=STREAM_COLORS[blk.stream]) for blk in blocks],
                      FadeOut(braces), run_time=0.8)
            self.play(LaggedStart(*[AnimationGroup(ShowCreation(a), FadeIn(nb, shift=0.1 * UP))
                                    for a, nb in zip(arcs, dist_nums)], lag_ratio=0.18),
                      FadeIn(dist_cap, shift=0.1 * UP, run_time=0.8), run_time=1.7)

            # results: plain gate, then the perfect gate (its squares replace the distances)
            vo.wait_until_sentence(2)
            self.play(FadeIn(r_plain.lab, shift=0.1 * RIGHT), FadeIn(r_plain.dots), FadeOut(src_layout), FadeIn(src),
                      run_time=0.8)
            vo.wait_until(at_phrase(vo, 2, "none of ten") - 0.2)
            self.play(FadeIn(r_plain.cnt), run_time=0.5)
            vo.wait_until(at_phrase(vo, 2, "the perfect gate") - 0.2)
            state = chip("perfect gate: channel = stream, given", GOOD).move_to([0, state_y, 0])
            self.play(*[a.path.animate.set_stroke(opacity=0.35) for a in arcs],
                      *[a.tip.animate.set_fill(opacity=0.35) for a in arcs],
                      *[ReplacementTransform(nb, sq) for nb, sq in zip(dist_nums, squares)],
                      FadeTransform(dist_cap, sq_lab), run_time=0.9)
            self.play(FadeIn(state, shift=0.1 * DOWN), FadeIn(r_ceil.lab, shift=0.1 * RIGHT),
                      FadeIn(r_ceil.dots, lag_ratio=0.3), run_time=0.8)
            self.play(FadeIn(r_ceil.cnt), run_time=0.5)

        # ---------------------------------------------------------------- N20.2 the recipe and the nudge
        kchip = chip("key splits", HINGE_COLOR).next_to(r_rec.cnt, RIGHT, buff=0.3)
        lbl_brace = Brace(VGroup(r_nudge.cnt, r_nslow.cnt), RIGHT, buff=0.15).set_color(WARN)
        lbl_chip = chip("uses labels", WARN).next_to(lbl_brace, RIGHT, buff=0.15)

        with self.voiceover(
            "The recipe did not carry over either: slow memory with the hinge discovered none of ten at a distance, "
            "failing by key splits. A five percent labelled nudge toward the stream split discovered ten of ten with "
            "slow memory, but that uses labels."
        ) as vo:
            self.play(FadeIn(r_rec.lab, shift=0.1 * RIGHT), FadeIn(r_rec.dots), run_time=0.8)
            vo.wait_until(at_phrase(vo, 0, "discovered none"))
            self.play(FadeIn(r_rec.cnt), run_time=0.6)
            vo.wait_until(at_phrase(vo, 0, "failing by key splits") - 0.3)
            new_state = chip("key split: the channel follows the key, not the stream", HINGE_COLOR).move_to([0, state_y, 0])
            self.play(*recolor(by_key), FadeTransform(state, new_state), FadeIn(kchip, shift=0.1 * LEFT), run_time=1.1)
            state = new_state
            self.play(LaggedStart(*[Indicate(VGroup(keys[i], squares[i]), color=WHITE, scale_factor=1.15)
                                    for i in range(8) if key_id[i] == 0], lag_ratio=0.3), run_time=1.0)

            vo.wait_until_sentence(1)
            new_state = chip("labelled nudge: pushed toward channel = stream", WARN).move_to([0, state_y, 0])
            self.play(FadeIn(r_nudge.lab, shift=0.1 * RIGHT), FadeIn(r_nudge.dots), FadeIn(r_nudge.cnt), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "toward the stream split"))
            self.play(*recolor(by_stream), FadeTransform(state, new_state), run_time=1.0)
            state = new_state
            vo.wait_until(at_phrase(vo, 1, "ten of ten") - 0.2)
            self.play(FadeIn(r_nslow.lab, shift=0.1 * RIGHT), FadeIn(r_nslow.dots, lag_ratio=0.1),
                      FadeIn(r_nslow.cnt), run_time=0.9)
            vo.wait_until(at_phrase(vo, 1, "uses labels") - 0.2)
            self.play(GrowFromCenter(lbl_brace), FadeIn(lbl_chip, shift=0.1 * LEFT), run_time=0.8)

        # ---------------------------------------------------------------- N20.3 one channel binds it too
        r_one = result_row("one channel, no gate", "S23", *S23, "bound", WARN, rows_y[0])
        r_one_slow = result_row("one channel + slow memory", "S23", *S23_SLOW, "bound", WARN, rows_y[1])
        one_color = MEMORY_COLOR
        stamp = VGroup(T("binding ≠ routing", 44, BAD))
        stamp_box = SurroundingRectangle(stamp, buff=0.2).set_stroke(BAD, 4)
        stamp = VGroup(stamp_box, stamp).rotate(6 * DEGREES).move_to([0.2, -2.55, 0])
        old_rows = VGroup(r_plain, r_ceil, r_rec, r_nudge, r_nslow, kchip, lbl_brace, lbl_chip)
        one_bar = Rectangle(width=row.get_width(), height=0.36)
        one_bar.set_fill(one_color, 0.85).set_stroke(INK, 1, opacity=0.5).move_to([row.get_x(), squares.get_y(), 0])
        one_lab = L("one channel: every token goes into the same memory", 20, BG, weight="BOLD").move_to(one_bar)

        with self.voiceover(
            "But a screen found a deeper problem. A single channel can bind the header layout too, three of ten, "
            "and seven of ten with slow memory. So binding there is no evidence that a gate routes. What is still "
            "needed is a distant-cue task that one channel provably cannot solve."
        ) as vo:
            self.play(FadeOut(old_rows, shift=0.3 * DOWN), FadeOut(state, shift=0.3 * DOWN), run_time=0.8)
            self.play(Indicate(VGroup(squares, sq_lab), color=BAD, scale_factor=1.05), run_time=0.9)

            vo.wait_until_sentence(1)
            new_title = section_title("One channel binds it too")
            new_state = chip("no gate: nothing to route", one_color).move_to([0, state_y, 0])
            self.play(FadeTransform(title, new_title), ReplacementTransform(squares, one_bar),
                      FadeOut(sq_lab), FadeIn(new_state, shift=0.1 * DOWN), run_time=1.1)
            self.play(FadeIn(one_lab), run_time=0.4)
            title, state = new_title, new_state
            vo.wait_until(at_phrase(vo, 1, "three of ten") - 0.3)
            self.play(FadeIn(r_one.lab, shift=0.1 * RIGHT), FadeIn(r_one.dots), FadeIn(r_one.cnt), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "seven of ten") - 0.3)
            self.play(FadeIn(r_one_slow.lab, shift=0.1 * RIGHT), FadeIn(r_one_slow.dots), FadeIn(r_one_slow.cnt),
                      run_time=0.8)

            vo.wait_until_sentence(2)
            self.play(FadeIn(stamp, scale=1.4), run_time=0.6)
            self.play(Indicate(VGroup(r_one.cnt, r_one_slow.cnt), color=WARN), run_time=0.9)

            vo.wait_until_sentence(3)
            self.play(FadeOut(VGroup(r_one, r_one_slow, stamp, state)), run_time=0.6)
            need = VGroup(
                L("still needed: a distant-cue task where", 24, INK),
                VGroup(L("one channel", 24, one_color, weight="BOLD"), L("→", 26, MUTED),
                       L("provably cannot solve it", 24, BAD, weight="BOLD")).arrange(RIGHT, buff=0.2),
                VGroup(L("so binding", 24, GOOD, weight="BOLD"), L("→", 26, MUTED),
                       L("shows that a gate routes", 24, GATE_COLOR, weight="BOLD")).arrange(RIGHT, buff=0.2),
            ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
            need.move_to([0, -1.75, 0])
            need_card = card(need, buff=0.35)
            open_chip = chip("OPEN", WARN, 22).next_to(need_card, RIGHT, buff=0.3)
            self.play(FadeIn(need_card[0]), FadeIn(need[0], shift=0.1 * UP), run_time=0.7)
            self.play(FadeIn(need[1], shift=0.1 * RIGHT), run_time=0.6)
            self.play(FadeIn(need[2], shift=0.1 * RIGHT), FadeIn(open_chip), run_time=0.6)
        self.wait(0.6)
        self.clear_all()
