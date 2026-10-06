"""Prologue — the problem in one picture, then the report's title (report title page)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403

B, Y = STREAM_COLORS[0], STREAM_COLORS[1]


def chat_card(lines, color, title):
    head = L(title, size=22, color=color, weight="BOLD")
    body = VGroup(*[L(t, size=24, color=INK) for t in lines]).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
    inner = VGroup(head, body).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
    box = SurroundingRectangle(inner, buff=0.3).round_corners(0.15)
    box.set_fill(PANEL, 1).set_stroke(color, 2)
    return VGroup(box, inner)


class Intro(ClankersScene):
    def construct(self):
        # ---------------------------------------------------------------- N0.1 two conversations
        conv0 = chat_card(["…", "the door code is 7", "…"], B, "conversation 0")
        conv1 = chat_card(["…", "the door code is 3", "…"], Y, "conversation 1")
        convs = VGroup(conv0, conv1).arrange(RIGHT, buff=1.6).move_to(1.3 * UP)

        t0 = VGroup(token("CTX0", "ctx", 0), token("door", width=1.1), token("7", "val", 0)).arrange(RIGHT, buff=0.06)
        t1 = VGroup(token("CTX1", "ctx", 1), token("door", width=1.1), token("3", "val", 1)).arrange(RIGHT, buff=0.06)
        filler = [VGroup(token("CTX0", "ctx", 0), token("…", width=1.1), token("…", "val", 0)).arrange(RIGHT, buff=0.06),
                  VGroup(token("CTX1", "ctx", 1), token("…", width=1.1), token("…", "val", 1)).arrange(RIGHT, buff=0.06)]
        q = VGroup(token("CTX0", "ctx", 0), token("door", width=1.1), token("?", "query")).arrange(RIGHT, buff=0.06)
        stream = VGroup(filler[1], t0, t1, filler[0], q).arrange(RIGHT, buff=0.3)
        stream.set_width(13).move_to(1.6 * DOWN)
        stream_lab = L("one interleaved sequence", size=22, color=MUTED).next_to(stream, DOWN, buff=0.25)
        q_lab = L("what was the code?", size=24, color=QUERY_COLOR).next_to(q, UP, buff=0.3)

        with self.voiceover(
            "Imagine reading two conversations at once, interleaved line by line. In one, the code for the door "
            "is seven. In the other, the code for the door is three. Later, someone from the first conversation "
            "asks: what was the code?"
        ) as vo:
            self.play(FadeIn(conv0, shift=0.3 * RIGHT), FadeIn(conv1, shift=0.3 * LEFT), run_time=1.2)
            vo.wait_until_sentence(1)
            self.play(Indicate(conv0[1][1][1], color=B), run_time=1.0)
            self.play(TransformFromCopy(conv0[1][1][1], t0), FadeIn(filler[1]), run_time=1.0)
            vo.wait_until_sentence(2)
            self.play(Indicate(conv1[1][1][1], color=Y), run_time=1.0)
            self.play(TransformFromCopy(conv1[1][1][1], t1), FadeIn(filler[0]), FadeIn(stream_lab), run_time=1.0)
            vo.wait_until_sentence(3)
            self.play(FadeIn(q, shift=0.2 * LEFT), Write(q_lab))

        # ---------------------------------------------------------------- N0.2 one memory
        self.play(FadeOut(convs), FadeOut(stream_lab), FadeOut(q_lab),
                  stream.animate.scale(0.85).to_edge(UP, buff=0.6), run_time=1.0)
        grid = memory_grid(6, 6, cell=0.4, color=MEMORY_COLOR, label="one memory")
        grid.move_to(0.6 * DOWN + 2.5 * LEFT)
        cells0 = [7, 8, 13, 14, 15, 20, 21, 26]
        cells1 = [8, 9, 14, 15, 16, 21, 22, 27]
        both = sorted(set(cells0) & set(cells1))
        mix = interpolate_color(B, Y, 0.5)
        w0 = VGroup(*[grid.cells[i].copy().set_fill(B, 0.7).set_stroke(width=0) for i in cells0])
        w1 = VGroup(*[grid.cells[i].copy().set_fill(Y, 0.7).set_stroke(width=0) for i in cells1])
        wmix = VGroup(*[grid.cells[i].copy().set_fill(mix, 0.85).set_stroke(width=0) for i in both])
        a0 = Arrow(stream[1].get_bottom(), grid.get_top() + 0.4 * LEFT, buff=0.15).set_stroke(B, 3)
        a1 = Arrow(stream[2].get_bottom(), grid.get_top() + 0.4 * RIGHT, buff=0.15).set_stroke(Y, 3)
        read = VGroup(L("read “door”", size=24), gate_bar([0.5, 0.5], width=2.2, height=0.36),
                      L("7 or 3?", size=28, color=BAD, weight="BOLD")).arrange(DOWN, buff=0.25)
        read.next_to(grid, RIGHT, buff=1.6)
        half = L("right half the time", size=22, color=MUTED).next_to(read, DOWN, buff=0.3)
        # the 50/50 bar's segments get their numbers
        read_nums = VGroup(L("7", size=20, color=BG, weight="BOLD").move_to(read[1][0][0]),
                           L("3", size=20, color=BG, weight="BOLD").move_to(read[1][0][1]))

        with self.voiceover(
            "A memory that simply links door to whatever followed it has stored both answers in the same place. "
            "The best it can do is guess, and it will be right half the time."
        ) as vo:
            self.play(FadeIn(grid))
            self.play(GrowArrow(a0), FadeIn(w0), run_time=1.0)
            self.play(GrowArrow(a1), FadeIn(w1), run_time=1.0)
            self.play(FadeIn(wmix), run_time=0.8)
            vo.wait_until_sentence(1)
            self.play(FadeIn(read, shift=0.2 * RIGHT), FadeIn(read_nums))
            self.play(FadeIn(half))

        # ---------------------------------------------------------------- N0.3 two compartments
        chans = channel_stack(2, rows=6, cols=6, cell=0.32, colors=[B, Y])
        chans.move_to(grid).shift(0.4 * RIGHT)
        c0, c1 = chans
        f0 = VGroup(*[c0.cells[i].copy().set_fill(B, 0.75).set_stroke(width=0) for i in cells0])
        f1 = VGroup(*[c1.cells[i].copy().set_fill(Y, 0.75).set_stroke(width=0) for i in cells1])
        b0 = Arrow(stream[1].get_bottom(), c0.get_top(), buff=0.15).set_stroke(B, 3)
        b1 = Arrow(stream[2].get_bottom(), c1.get_top(), buff=0.15).set_stroke(Y, 3)
        read2 = VGroup(L("read “door” in conversation 0", size=24), gate_bar([1.0, 0.0], width=2.2, height=0.36),
                       L("7", size=40, color=GOOD, weight="BOLD")).arrange(DOWN, buff=0.25)
        read2.next_to(chans, RIGHT, buff=1.2)
        qa = CurvedArrow(c0.get_bottom() + 0.45 * DOWN, read2.get_bottom() + 0.2 * DOWN, angle=PI / 3)
        qa.set_stroke(B, 3)

        with self.voiceover(
            "One fix is obvious. Give the memory separate compartments, and file each conversation in its own. "
            "Then the question reads only from the compartment it belongs to, and the conflict is gone."
        ) as vo:
            vo.wait_until_sentence(1)
            self.play(FadeOut(VGroup(w0, w1, wmix, a0, a1)), ReplacementTransform(grid, c0), FadeIn(c1, shift=0.6 * RIGHT), run_time=1.2)
            self.play(GrowArrow(b0), FadeIn(f0), run_time=0.8)
            self.play(GrowArrow(b1), FadeIn(f1), run_time=0.8)
            vo.wait_until_sentence(2)
            self.play(FadeOut(VGroup(read, read_nums, half)), FadeIn(read2, shift=0.2 * RIGHT), ShowCreation(qa))
            self.play(Indicate(stream[-1][2], color=GOOD))

        # ---------------------------------------------------------------- N0.4 who decides?
        gate = VGroup(RoundedRectangle(width=2.2, height=1.0, corner_radius=0.15).set_fill(GATE_COLOR, 0.15)
                      .set_stroke(GATE_COLOR, 2.5), L("gate", size=30, color=GATE_COLOR, weight="BOLD"))
        gate[1].move_to(gate[0])
        gate.move_to(VGroup(b0, b1).get_center() + 0.1 * UP)
        qm = T("?", size=96, color=GATE_COLOR).next_to(gate, RIGHT, buff=0.3)
        ask = T("Can a network learn to split its memory, by itself?", size=36)
        ask.to_edge(DOWN, buff=0.45)

        with self.voiceover(
            "The hard part is the word give. Can a network discover, from the task alone, that it should split its "
            "memory this way, and which tokens belong in which compartment?"
        ) as vo:
            self.play(FadeOut(VGroup(b0, b1)), FadeIn(gate, scale=0.8))
            self.play(Write(qm))
            vo.wait_until_sentence(1)
            self.play(Write(ask), run_time=1.5)

        self.play(FadeOut(VGroup(stream, chans, f0, f1, read2, qa, gate, qm, ask)), run_time=0.8)

        # ---------------------------------------------------------------- N0.5 title
        t_lines = VGroup(T("Multi-Channel Hebbian Plasticity in Multilayer BDH", size=44),
                         T("Solves Context-Conditional Binding", size=44),
                         T("by Partitioning Memory", size=44)).arrange(DOWN, buff=0.2)
        sub = T("An early-training recipe finds the partition without restarts in most runs for two streams",
                size=24, color=MUTED)
        sub2 = T("and for four with spare channels; at eight streams the gate forgets the context",
                 size=24, color=MUTED)
        meta = L("A technical report  ·  Revision 7 (preliminary)", size=24, color=INK)
        repo = L("github.com/Ynkling/clankers", size=28, color=GATE_COLOR, weight="BOLD")
        title = VGroup(t_lines, VGroup(sub, sub2).arrange(DOWN, buff=0.1), meta, repo).arrange(DOWN, buff=0.4)
        title.move_to(0.4 * UP)
        road = VGroup(*[verdict_badge(w, size=20, color=c) for w, c in [
            ("the model", MEMORY_COLOR), ("the task", KEY_COLOR), ("the code", MUTED), ("the experiments", WARN),
            ("what works", GOOD), ("what still fails", BAD), ("how it was checked", GATE_COLOR)]]).arrange(RIGHT, buff=0.2)
        road.set_width(min(road.get_width(), 13)).to_edge(DOWN, buff=0.45)

        with self.voiceover(
            "That is the question behind the repository Ynkling slash clankers, and its technical report: "
            "Multi-Channel Hebbian Plasticity in Multilayer BDH Solves Context-Conditional Binding by Partitioning "
            "Memory. Its subtitle gives the news: an early-training recipe now finds the partition without restarts "
            "in most runs, for two streams and for four with spare channels, while at eight streams the gate forgets "
            "the context. This video covers all of it: the model, the task, the code, the experiments, what works, "
            "what still fails, and how the work was checked.",
            spoken="That is the question behind the repository Inkling slash clankers, and its technical report: "
            "Multi-Channel Hebbian Plasticity in Multilayer B D H Solves Context-Conditional Binding by Partitioning "
            "Memory. Its subtitle gives the news: an early-training recipe now finds the partition without restarts "
            "in most runs, for two streams and for four with spare channels, while at eight streams the gate forgets "
            "the context. This video covers all of it: the model, the task, the code, the experiments, what works, "
            "what still fails, and how the work was checked.",
        ) as vo:
            self.play(FadeIn(repo, shift=0.2 * UP))
            vo.wait_until(2.5)
            self.play(LaggedStartMap(FadeIn, t_lines, shift=0.2 * UP, lag_ratio=0.3), run_time=2.0)
            self.play(FadeIn(meta))
            vo.wait_until_sentence(1)
            self.play(FadeIn(sub, shift=0.1 * UP), run_time=1.2)
            self.play(sub.select_part("finds the partition without restarts").animate.set_color(GOOD), run_time=1.0)
            self.play(sub.select_part("two streams").animate.set_color(GOOD), run_time=0.8)
            self.play(FadeIn(sub2, shift=0.1 * UP), run_time=1.0)
            self.play(sub2.select_part("for four with spare channels").animate.set_color(GOOD), run_time=0.8)
            self.play(sub2.select_part("at eight streams the gate forgets the context").animate.set_color(BAD), run_time=1.0)
            vo.wait_until_sentence(2)
            self.play(self.frame.animate.scale(0.97), run_time=2.0)
            self.play(LaggedStartMap(FadeIn, road, shift=0.2 * UP, lag_ratio=0.25), run_time=3.0)
        self.wait(0.8)
        self.clear_all()
        self.frame.to_default_state()
