"""Chapter 22 — keeping score on itself (report §14.1 Revision 6's claims, §14.2 errors made during Phase V,
§15 methodological findings; numbers from §4 Table 2, §7 Table 4 / data/slow_start.json, §13,
data/hinge_firings.json)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
B, Y = STREAM_COLORS[0], STREAM_COLORS[1]

# ---------------------------------------------------------------- §14.1: Revision 6's claims, graded
# (claim as quoted in §14.1, the report's own tag, Revision 7's verdict)
CLAIMS = [
    ("“…solves context-conditional binding by partitioning memory”", "(title)", "STANDS"),
    ("“…reliably with restarts for two streams; four streams are the open problem”", "(subtitle)", "SUPERSEDED"),
    ("“A simple early check at four streams picked binders with 38/38 precision”", "(post hoc)", "PARTLY CONFIRMED"),
    ("“Exploratory screens suggest it depends on what the memory learns first”", "", "SUPPORTED"),
    ("“Restarts are the working recipe”", "", "REPLACED"),
    ("Pooled post-hoc p-values across machines X and L", "", "CORRECTED"),
]
ROW_Y0, ROW_DY = 1.92, 0.55
TEXT_X, BADGE_X = -6.35, 3.95
DIV_Y = -1.2                # dashed rule between the ledger and the evidence strip
DETAIL_Y = -2.25            # centre of the evidence strip under the ledger
DETAIL_CUES = {1: "reliably with restarts"}   # claim 2: keep claim 1's evidence until the quote gets going
# N22.1, one voiceover block per graded claim (split at sentence boundaries), so that each claim's
# evidence can be read for a moment after its verdict is spoken
CLAIM_NARRATION = [
    "The report also grades its own earlier claims. The title claim stands.",
    "\"A learned gate finds the partition, reliably with restarts, at two streams; four are the open problem\": "
    "superseded, since the recipe now binds two streams, and four with sixteen channels, without restarts.",
    "The early check at four streams: partly confirmed, precise on L but not on X.",
    "\"It depends on what the memory learns first\": supported by a pre-registered test.",
    "Restarts as the working recipe: replaced, at two streams and at four with sixteen channels, by the "
    "early-training recipe.",
    "And Revision Six's pooled p-values across the machines: corrected, because both machines ran the same seeds.",
]
CLAIM_SENTENCE = [1, 0, 0, 0, 0, 0]          # index, within its block, of the sentence that grades the claim
CLAIM_HOLD = [0.4, 0.5, 1.3, 1.3, 0.9, 0.9]  # extra seconds on the finished evidence after each block

# ---------------------------------------------------------------- §14.2: errors made during Phase V
ERRORS = [
    ("Replicates treated as independent", "X and L ran the same seeds, yet were called independent"),
    ("“The hinge barely takes part”", "0–4 firings a run under Muon; then it rescued 8 of 40 runs"),
    ("Merge breaker over-claimed, mis-specified", "its trigger would fire on a correctly routed S = 8 gate"),
    ("An underpowered confirmation", "no-hinge baseline: 31/40 in screens; 36/40 fresh on X"),
    ("Machine differences misattributed", "blamed on bf16 kernels; X and L differ under Adam too"),
    ("Specifications lost between sessions", "the slow-start spec and screens S29–S31 never arrived"),
    ("A validity arm without the budget", "the eight-key header screen's perfect gate did not bind"),
    ("Process slips", "a shared optimizer state (S14); a results-file mix-up on L"),
]
# where each error starts in the narration (for timing)
ERROR_CUES = ["calling the two machines", "reading the hinge", "a merge breaker", "a confirmation planned",
              "machine differences", "specifications lost", "a validity arm", "and process slips"]

# ---------------------------------------------------------------- §15: five lessons
LESSONS = [
    ("Give each machine its own seeds", "shared seeds make the two machines dependent", 0),
    ("Record the kernel path", "Adam reproduced across X's two Xeon hosts, never between X and L", 4),
    ("Probe the mechanism before designing the fix", "a decoder on the gate's state found the S = 8 failure in two screens", None),
    ("Score routing, not binding", "S = 4 on four channels and the header layout bind without the partition", None),
    ("Power a confirmation for the baseline it will meet", "a screen's baseline can be low by chance", 3),
]


def cue(vo, i, fragment=None):
    """Seconds into the block at which `fragment` is spoken in sentence i (linear in characters)."""
    text, a, b = vo.synth.sentences[i]
    if fragment is None:
        return a
    k = text.find(fragment)
    if k < 0:
        raise ValueError(f"{fragment!r} not in sentence {i}: {text!r}")
    return a + (b - a) * k / max(len(text), 1)


def check_mark(color=GOOD, size=0.22):
    return VMobject().set_points_as_corners(
        [[-0.5, 0.0, 0], [-0.15, -0.38, 0], [0.55, 0.45, 0]]).set_stroke(color, 4).set_width(size)


def cross_mark(color=BAD, size=0.2):
    a = Line(UL, DR).set_stroke(color, 4)
    b = Line(DL, UR).set_stroke(color, 4)
    return VGroup(a, b).set_width(size)


def strike(mob, color=BAD, width=3):
    return Line(mob.get_left() + 0.05 * LEFT, mob.get_right() + 0.05 * RIGHT).set_stroke(color, width)


def mini_bar(num, den, color, width=2.2, height=0.26, fail_color=None):
    track = Rectangle(width=width, height=height).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1)
    fill = Rectangle(width=max(width * num / den, 1e-3), height=height).set_fill(color, 0.9).set_stroke(width=0)
    fill.align_to(track, LEFT)
    g = VGroup(track, fill)
    g.track, g.fill = track, fill
    if fail_color is not None and num < den:
        fail = Rectangle(width=width * (den - num) / den, height=height).set_fill(fail_color, 0.85).set_stroke(width=0)
        fail.align_to(track, RIGHT)
        g.add(fail)
        g.fail = fail
    return g


def grow(bar, run_time=0.8):
    bar.fill.save_state()
    bar.fill.stretch(1e-3, 0, about_edge=LEFT)
    return Restore(bar.fill, run_time=run_time)


def slot(width, height, number, color=FAINT):
    rect = RoundedRectangle(width=width, height=height, corner_radius=0.12)
    rect.set_fill(PANEL, 0.35).set_stroke(color, 1.2, opacity=0.6)
    num = L(str(number), size=20, color=FAINT).move_to(rect.get_left() + 0.38 * RIGHT)
    return VGroup(rect, num)


def numbered_card(number, title, gist, width, height, accent, title_size=24):
    rect = RoundedRectangle(width=width, height=height, corner_radius=0.12)
    rect.set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1.5)
    stripe = Rectangle(width=0.07, height=height - 0.2).set_fill(accent, 1).set_stroke(width=0)
    stripe.move_to(rect.get_left() + 0.12 * RIGHT)
    circ = Circle(radius=0.2).set_fill(accent, 0.15).set_stroke(accent, 1.8)
    circ.move_to(rect.get_left() + 0.48 * RIGHT)
    num = L(str(number), size=20, color=accent, weight="BOLD").move_to(circ)
    t = L(title, size=title_size, color=INK, weight="BOLD")
    g = L(gist, size=20, color=MUTED)
    text = VGroup(t, g).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
    max_w = width - 1.0
    for m in (t, g):
        if m.get_width() > max_w:
            print(f"[s22] WARNING text too wide ({m.get_width():.2f} > {max_w:.2f}): {getattr(m, 'text', '')}")
    text.move_to(rect.get_left() + 0.82 * RIGHT, aligned_edge=LEFT)
    c = VGroup(rect, stripe, circ, num, text)
    c.rect, c.stripe, c.circ, c.num, c.title, c.gist = rect, stripe, circ, num, t, g
    return c


class Retrospective(ClankersScene):
    def construct(self):
        card = self.chapter_card(22, "Keeping score on itself")
        self.wait(0.6)
        title = section_title("Revision 6, graded")
        self.play(FadeOut(card, shift=0.3 * UP), FadeIn(title, shift=0.3 * UP))

        # ============================================================ N22.1 the claims ledger
        head_l = L("Revision 6 claimed", size=22, color=MUTED, weight="BOLD")
        head_l.move_to([TEXT_X, 2.47, 0], aligned_edge=LEFT)
        head_r = L("Revision 7's verdict", size=22, color=MUTED, weight="BOLD")
        head_r.move_to([BADGE_X, 2.47, 0], aligned_edge=LEFT)
        head_rule = Line([-6.5, 2.22, 0], [6.5, 2.22, 0]).set_stroke(FAINT, 1.5)
        div = DashedLine([-6.5, DIV_Y, 0], [6.5, DIV_Y, 0], dash_length=0.08).set_stroke(FAINT, 1)

        rows, badges = VGroup(), VGroup()
        for i, (quote, tag, verdict) in enumerate(CLAIMS):
            y = ROW_Y0 - i * ROW_DY
            q = T(quote, size=24, color=INK)
            parts = [q]
            if tag:
                parts.append(L(tag, size=20, color=MUTED))
            row = VGroup(*parts).arrange(RIGHT, buff=0.18, aligned_edge=DOWN)
            row.move_to([TEXT_X, y, 0], aligned_edge=LEFT)
            rows.add(row)
            b = verdict_badge(verdict, size=20)
            b.move_to([BADGE_X, y, 0], aligned_edge=LEFT)
            badges.add(b)

        def hl_for(i):
            col = VERDICT_STYLES[CLAIMS[i][2]]
            r = RoundedRectangle(width=13.2, height=0.5, corner_radius=0.08)
            r.set_fill(col, 0.09).set_stroke(col, 1.2, opacity=0.55)
            r.move_to([0, ROW_Y0 - i * ROW_DY, 0])
            return r

        # ---- evidence strips, one per claim ---------------------------------------------------
        details = [self.detail_stands(), self.detail_superseded(), self.detail_partly(),
                   self.detail_supported(), self.detail_replaced(), self.detail_corrected()]
        notes = ["Report §13; §14.1", "Report §14.1; Table 4", "Report §4, Table 2 (R2); §14.1",
                 "Report §7, Table 4 (S0); data/slow_start.json",
                 "Report §14.1; hinge firings: X, Muon + WINDOW, seed 319 (data/hinge_firings.json)",
                 "Report §14.1"]

        hl = None
        note = None
        prev = None
        for i in range(6):
            s = CLAIM_SENTENCE[i]
            with self.voiceover(CLAIM_NARRATION[i]) as vo:
                if i == 0:
                    # sentence 0: the ledger's two columns
                    self.play(FadeIn(head_l, shift=0.15 * RIGHT), ShowCreation(head_rule), run_time=0.9)
                    self.play(FadeIn(head_r, shift=0.15 * LEFT), ShowCreation(div), run_time=0.8)
                    vo.wait_until(max(cue(vo, s) - 0.2, 0))
                new_hl = hl_for(i)
                anims = [FadeIn(rows[i], shift=0.2 * RIGHT)]
                anims.append(FadeIn(new_hl) if hl is None else Transform(hl, new_hl))
                if i:
                    anims.append(rows[:i].animate.set_opacity(0.55))
                new_note = source_note(notes[i])
                note_anim = FadeIn(new_note) if note is None else FadeTransform(note, new_note)
                # the old evidence leaves before the new one arrives (no cross-fade of two strips)
                out = [FadeOut(prev, shift=0.15 * DOWN)] if prev is not None else []
                if i in DETAIL_CUES:
                    # keep the previous evidence up a little longer; swap when the claim's words are spoken
                    self.play(*anims, run_time=0.8)
                    vo.wait_until(cue(vo, s, DETAIL_CUES[i]) - 0.6)
                    self.play(*out, note_anim, run_time=0.4)
                else:
                    self.play(*anims, *out, note_anim, run_time=0.5)
                self.play(FadeIn(details[i].intro, shift=0.15 * UP), run_time=0.5)
                if hl is None:
                    hl = new_hl
                note = new_note
                details[i].run(vo, s, badges[i])
                prev = details[i]
            if i < 5:
                self.wait(CLAIM_HOLD[i])
        # the whole ledger, graded
        self.play(FadeOut(hl), rows.animate.set_opacity(1.0), run_time=0.6)
        self.wait(CLAIM_HOLD[5])

        # ============================================================ N22.2 errors in Phase V
        title2 = section_title("Errors made during Phase V")
        ledger = VGroup(head_l, head_r, head_rule, rows, badges)
        cw, ch = 6.45, 1.18
        xs = [-3.3, 3.3]
        ys = [1.92, 0.6, -0.72, -2.04]
        slots, cards = VGroup(), VGroup()
        for i, (t, g) in enumerate(ERRORS):
            pos = [xs[i % 2], ys[i // 2], 0]
            slots.add(slot(cw, ch, i + 1).move_to(pos))
            cards.add(numbered_card(i + 1, t, g, cw, ch, BAD).move_to(pos))
        caught = verdict_badge("stopped before any training", size=20, color=GOOD)
        caught.next_to(cards[7].title, RIGHT, buff=0.3)
        note2 = source_note("Report §14.2")
        firing_ticks = self.firing_ticks(cards[1])

        with self.voiceover(
            "It lists its own errors in this phase too: calling the two machines independent replicates; reading "
            "the hinge's rare firings under Muon as a sign it might not be needed, when a few well-timed firings "
            "were the mechanism; a merge breaker claimed before its trigger existed, then specified wrongly; a "
            "confirmation planned from the screens' numbers that met a higher baseline; machine differences "
            "blamed on Muon's kernels alone; specifications lost between sessions; a validity arm without the "
            "budget to bind; and process slips, among them a results-file mix-up on machine L that the test's own "
            "pairing checks stopped before any training."
        ) as vo:
            # clear the ledger first, then lay out eight empty slots (no overlap of the two pictures)
            self.play(LaggedStart(
                AnimationGroup(FadeTransform(title, title2), FadeOut(ledger, shift=0.3 * UP),
                               FadeOut(prev, shift=0.15 * DOWN), FadeOut(div), FadeTransform(note, note2)),
                LaggedStartMap(FadeIn, slots, lag_ratio=0.08), lag_ratio=0.6), run_time=1.6)
            for i in range(8):
                vo.wait_until(max(cue(vo, 0, ERROR_CUES[i]) - 0.15, 0))
                self.play(FadeOut(slots[i]), FadeIn(cards[i], shift=0.12 * UP), run_time=0.6)
                if i == 1:
                    vo.wait_until(cue(vo, 0, "when a few well-timed"))
                    self.play(LaggedStartMap(GrowFromPoint, firing_ticks,
                                             point=firing_ticks.get_bottom(), lag_ratio=0.3),
                              cards[1].stripe.animate.set_fill(HINGE_COLOR), run_time=0.9)
                    self.play(Indicate(firing_ticks, color=HINGE_COLOR, scale_factor=1.15), run_time=0.8)
                if i == 2:
                    vo.wait_until(cue(vo, 0, "then specified wrongly"))
                    self.play(Indicate(cards[2].gist, color=INK, scale_factor=1.04), run_time=0.8)
            vo.wait_until(cue(vo, 0, "among them a results-file"))
            self.play(Indicate(cards[7].gist, color=INK, scale_factor=1.04), run_time=0.9)
            vo.wait_until(cue(vo, 0, "pairing checks stopped"))
            self.play(FadeIn(caught, scale=0.6), run_time=0.6)

        self.play(*[FadeOut(VGroup(c.rect, c.stripe, c.circ, c.gist)) for c in cards],
                  FadeOut(caught), FadeOut(firing_ticks), run_time=0.45)

        # ============================================================ N22.3 five lessons
        title3 = section_title("Five lessons")
        compact = VGroup()
        for i, (t, _) in enumerate(ERRORS):
            n = L(f"{i + 1}", size=20, color=BAD, weight="BOLD")
            tt = L(t, size=20, color=MUTED)
            row = VGroup(n, tt).arrange(RIGHT, buff=0.18)
            compact.add(row)
        compact.arrange(DOWN, buff=0.36, aligned_edge=LEFT)
        compact.move_to([-6.45, 0.05, 0], aligned_edge=LEFT)
        err_head = L("errors", size=22, color=BAD, weight="BOLD").next_to(compact, UP, buff=0.35, aligned_edge=LEFT)

        lw, lh = 7.75, 1.0
        lx = 6.5 - lw / 2
        lys = [1.95 - i * 1.17 for i in range(5)]
        lslots, lcards = VGroup(), VGroup()
        for i, (t, g, _) in enumerate(LESSONS):
            lslots.add(slot(lw, lh, i + 1).move_to([lx, lys[i], 0]))
            lcards.add(numbered_card(i + 1, t, g, lw, lh, GOOD).move_to([lx, lys[i], 0]))
        les_head = L("lessons", size=22, color=GOOD, weight="BOLD")
        les_head.next_to(lslots, UP, buff=0.2, aligned_edge=LEFT)
        note3 = source_note("Report §15")

        links = {}
        for i, (_, _, e) in enumerate(LESSONS):
            if e is None:
                continue
            a = compact[e].get_right() + 0.15 * RIGHT
            b = lcards[i].get_left() + 0.08 * LEFT
            # leave the error title horizontally, then bend into the lesson card (never crosses another title)
            curve = CubicBezier(a, a + 1.15 * RIGHT, b + 0.75 * LEFT, b).set_stroke(GOOD, 2.5, opacity=0.85)
            tip = ArrowTip(width=0.18, length=0.18).set_fill(GOOD, 0.9).set_stroke(width=0)
            tip.move_to(b + 0.09 * LEFT)
            links[i] = VGroup(curve, tip)

        with self.voiceover(
            "Five lessons come out of it. Give each machine its own seeds when you want independent confirmation. "
            "Bit-for-bit determinism depends on the processor and its kernels, so record the kernel path. Probe "
            "the mechanism before designing the fix. Score routing, not binding, wherever the memory has another "
            "route. And power a confirmation for the baseline it will actually meet."
        ) as vo:
            moves = []
            for i in range(8):
                c = cards[i]
                moves += [ReplacementTransform(c.title, compact[i][1]), ReplacementTransform(c.num, compact[i][0])]
            self.play(FadeTransform(title2, title3), *moves, FadeTransform(note2, note3), run_time=1.0)
            self.play(FadeIn(err_head), FadeIn(les_head), LaggedStartMap(FadeIn, lslots, lag_ratio=0.12), run_time=0.8)
            for i in range(5):
                vo.wait_until(max(cue(vo, i + 1) - 0.15, 0))
                anims = [FadeOut(lslots[i]), FadeIn(lcards[i], shift=0.12 * LEFT)]
                if i in links:
                    e = LESSONS[i][2]
                    anims += [ShowCreation(links[i][0]), FadeIn(links[i][1]), compact[e][1].animate.set_color(INK)]
                self.play(*anims, run_time=0.8)
                if i == 1:
                    vo.wait_until(cue(vo, 2, "so record the kernel path"))
                    self.play(Indicate(lcards[1].title, color=GOOD, scale_factor=1.06), run_time=0.8)
        self.wait(0.8)
        self.clear_all()

    # ================================================================ evidence strips
    # Each returns a VGroup with .intro (what fades in at the sentence's start) and .run(vo, s, badge),
    # which plays the rest of that sentence's visuals (including stamping the verdict badge).

    def stamp(self, badge):
        self.play(FadeIn(badge, scale=0.55), run_time=0.45)
        self.play(FlashAround(badge, color=badge[1].get_fill_color(), time_width=0.6), run_time=0.55)

    def detail_stands(self):
        two = channel_stack(2, rows=4, cols=4, cell=0.2, colors=[B, Y])
        for k, col in enumerate([B, Y]):
            for j in ([1, 5, 6, 10] if k == 0 else [2, 6, 9, 13]):
                two[k].cells[j].set_fill(col, 0.7)
        one = memory_grid(4, 4, cell=0.2, color=MEMORY_COLOR)
        mix = interpolate_color(B, Y, 0.5)
        for j in [1, 2, 5, 6, 9, 10, 13]:
            one.cells[j].set_fill(mix, 0.7)
        t_a = VGroup(L("given the partition (a perfect gate):", size=20, color=MUTED),
                     VGroup(check_mark(), L("every grouped-layout setting tried binds", size=22)).arrange(RIGHT, buff=0.15)
                     ).arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        t_b = VGroup(L("one channel, same memory:", size=20, color=MUTED),
                     VGroup(cross_mark(), L("does not", size=22)).arrange(RIGHT, buff=0.15)
                     ).arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        left = VGroup(two, t_a).arrange(RIGHT, buff=0.4)
        right = VGroup(one, t_b).arrange(RIGHT, buff=0.4)
        g = VGroup(left, right).arrange(RIGHT, buff=1.0).move_to([0, DETAIL_Y + 0.05, 0])
        g.intro = g

        def run(vo, s, badge):
            vo.wait_until(cue(vo, s, "stands"))
            self.stamp(badge)
        g.run = run
        return g

    def detail_superseded(self):
        lab6 = L("Revision 6", size=20, color=MUTED, weight="BOLD")
        r6 = T("reliably with restarts for two streams; four streams are the open problem", size=24, color=MUTED)
        lab7 = L("Revision 7", size=20, color=GOOD, weight="BOLD")
        r7a = T("without restarts for two streams, and for four with sixteen channels", size=24,
                t2c={"without restarts": GOOD})
        r7b = VGroup(L("(HINGE recipe: at least 85% of runs on each machine);", size=20, color=MUTED),
                     T("eight streams are the open problem", size=24, t2c={"eight streams": WARN})
                     ).arrange(RIGHT, buff=0.2)
        col_x = -6.35
        lab6.move_to([col_x, DETAIL_Y + 0.72, 0], aligned_edge=LEFT)
        r6.next_to(lab6, RIGHT, buff=0.35)
        lab7.move_to([col_x, DETAIL_Y - 0.08, 0], aligned_edge=LEFT)
        r7a.next_to(lab7, RIGHT, buff=0.35)
        r7b.next_to(r7a, DOWN, buff=0.22, aligned_edge=LEFT)
        s1 = strike(r6.select_part("with restarts"))
        s2 = strike(r6.select_part("four streams"))
        g = VGroup(lab6, r6, s1, s2, lab7, r7a, r7b)
        g.intro = VGroup(lab6, r6)

        def run(vo, s, badge):
            vo.wait_until(cue(vo, s, "superseded") - 0.1)
            self.stamp(badge)
            self.play(ShowCreation(s1), ShowCreation(s2), run_time=0.6)
            vo.wait_until(cue(vo, s, "since the recipe"))
            self.play(FadeIn(lab7), FadeTransform(r6.copy(), r7a), run_time=1.1)
            self.add(r7a)
            self.play(FadeIn(r7b, shift=0.1 * UP), run_time=0.8)
        g.run = run
        return g

    def detail_partly(self):
        head = L("the check on fresh seeds 240–259 (k = 16): of the attempts it passed, how many bound?",
                 size=22, color=MUTED)
        head.move_to([-6.35, DETAIL_Y + 0.72, 0], aligned_edge=LEFT)
        rows = VGroup()
        for m, num, den in [("L", 19, 19), ("X", 17, 20)]:
            badge = machine_badge(m, size=20)
            bar = mini_bar(num, den, GOOD, width=4.2, height=0.3, fail_color=BAD)
            val = L(f"{num}/{den}", size=22, color=GOOD if num == den else WARN, weight="BOLD")
            row = VGroup(badge, bar, val).arrange(RIGHT, buff=0.3)
            row.badge, row.bar, row.val = badge, bar, val
            rows.add(row)
        rows.arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        rows.move_to([-6.1, DETAIL_Y - 0.25, 0], aligned_edge=LEFT)
        v_l = verdict_badge("PRECISE", size=20).next_to(rows[0], RIGHT, buff=0.45)
        v_x = verdict_badge("NOT PRECISE", size=20).next_to(rows[1], RIGHT, buff=0.45)
        merges = L("the 3 that failed: merges", size=20, color=BAD).next_to(v_x, RIGHT, buff=0.35)
        g = VGroup(head, rows, v_l, v_x, merges)
        g.intro = VGroup(head, rows)
        for r in rows:
            r.bar.fill.save_state()
            r.bar.fill.stretch(1e-3, 0, about_edge=LEFT)
            if hasattr(r.bar, "fail"):
                r.bar.fail.set_opacity(0)

        def run(vo, s, badge):
            self.play(*[Restore(r.bar.fill) for r in rows], run_time=0.7)
            vo.wait_until(cue(vo, s, "partly confirmed") - 0.1)
            self.stamp(badge)
            vo.wait_until(cue(vo, s, "precise on L"))
            self.play(FadeIn(v_l, scale=0.7), Indicate(rows[0].val, color=GOOD), run_time=0.6)
            vo.wait_until(cue(vo, s, "but not on X"))
            self.play(FadeIn(v_x, scale=0.7), rows[1].bar.fail.animate.set_opacity(0.85),
                      FadeIn(merges, shift=0.1 * RIGHT), run_time=0.7)
        g.run = run
        return g

    def detail_supported(self):
        with open(os.path.join(DATA, "slow_start.json")) as fh:
            d = json.load(fh)
        head = L("pre-registered test S0, fresh seeds 280–299: slow memory alone against the plain gate "
                 "(discovered, of 20)", size=22, color=MUTED)
        head.move_to([-6.35, DETAIL_Y + 0.72, 0], aligned_edge=LEFT)
        arms = [("plain gate", "A0", ADAM_COLOR), ("slow memory", "SLOW0", SLOW_COLOR)]
        bar_w, name_w = 1.8, 1.55
        x_name = -6.2
        x_cols = [x_name + name_w + 0.3, x_name + name_w + 0.3 + bar_w + 0.95]   # left edge of each machine's bars
        ys = [DETAIL_Y - 0.33, DETAIL_Y - 0.78]
        y_head = DETAIL_Y + 0.12
        chart = VGroup()
        col_heads = VGroup(*[machine_badge(m, size=20).move_to([x + bar_w / 2, y_head, 0])
                             for m, x in zip(["X", "L"], x_cols)])
        chart.add(col_heads)
        bars = []
        for (lab, arm, col), y in zip(arms, ys):
            name = L(lab, size=20, color=col)
            name.move_to([x_name + name_w, y, 0], aligned_edge=RIGHT)
            chart.add(name)
            for m, x in zip(["X", "L"], x_cols):
                c = d["counts"][m][arm]
                bar = mini_bar(c["success"], c["n"], col, width=bar_w, height=0.26)
                bar.move_to([x, y, 0], aligned_edge=LEFT)
                val = L(f"{c['success']}/{c['n']}", size=20, color=col, weight="BOLD")
                val.next_to(bar, RIGHT, buff=0.12)
                chart.add(bar, val)
                bars.append(bar)
        stats, verdicts = VGroup(), VGroup()
        for m in ["X", "L"]:
            cl = d["claims"][m]["S0"]["report"]
            row = VGroup(machine_badge(m, size=20), L(f"{cl['b']} vs {cl['c']}, p = {cl['p']}", size=20))
            row.arrange(RIGHT, buff=0.2)
            stats.add(row)
            verdicts.add(verdict_badge(cl["verdict"], size=20))
        stats.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        stats.move_to([x_cols[1] + bar_w + 1.25, (ys[0] + ys[1]) / 2 + 0.2, 0], aligned_edge=LEFT)
        v_x0 = max(r.get_right()[0] for r in stats) + 0.3          # both verdicts in one column
        for v, r in zip(verdicts, stats):
            v.move_to([v_x0, r.get_y(), 0], aligned_edge=LEFT)
        test_lab = L("S0: slow memory beats the plain gate", size=20, color=MUTED)
        test_lab.next_to(stats, UP, buff=0.18, aligned_edge=LEFT)
        g = VGroup(head, chart, test_lab, stats, verdicts)
        g.intro = VGroup(head, chart)
        for bar in bars:
            bar.fill.save_state()
            bar.fill.stretch(1e-3, 0, about_edge=LEFT)

        def run(vo, s, badge):
            self.play(*[Restore(bar.fill) for bar in bars], FadeIn(test_lab), FadeIn(stats), run_time=0.8)
            vo.wait_until(cue(vo, s, "supported") - 0.1)
            self.stamp(badge)
            self.play(LaggedStartMap(FadeIn, verdicts, scale=0.7, lag_ratio=0.3),
                      *[Indicate(bar, color=SLOW_COLOR, scale_factor=1.06) for bar in bars[2:]], run_time=0.8)
        g.run = run
        return g

    def detail_replaced(self):
        with open(os.path.join(DATA, "hinge_firings.json")) as fh:
            fired = json.load(fh)["early_recipe"]["X"]["arms"]["WIN_M"]["runs"]["319"]["fired_at"]
        # left: restarts — attempts until one passes an early check
        left_head = L("restarts: retry until an early check passes", size=22, color=MUTED)
        att = VGroup()
        check_x = 1.6
        for k, ok in enumerate([False, False, True]):
            ln = Line(ORIGIN, (3.4 if ok else check_x) * RIGHT).set_stroke(INK if ok else FAINT, 3)
            if ok:
                mark = check_mark(GOOD).next_to(ln.get_start() + check_x * RIGHT, UP, buff=0.08)
            else:
                mark = cross_mark(BAD).move_to(ln.get_end() + 0.25 * RIGHT)
            lab = L(f"attempt {k + 1}", size=20, color=MUTED).next_to(ln, LEFT, buff=0.2)
            att.add(VGroup(lab, ln, mark))
        att.arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        lines_left = att[0][1].get_start()[0]
        chk = DashedLine(UP, DOWN, dash_length=0.06).set_stroke(WARN, 1.5)
        chk.set_height(att.get_height() + 0.1).move_to([lines_left + check_x, att.get_center()[1], 0])
        left = VGroup(left_head, VGroup(att, chk)).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        left.move_to([-6.35, DETAIL_Y - 0.1, 0], aligned_edge=LEFT)

        # right: the early-training recipe — one run; slow memory and hinge firings in updates 1-2400
        right_head = L("early-training recipe: one run, no restarts", size=22, color=GOOD)
        axis = NumberLine(x_range=[0, 3600, 1200], width=4.6, include_tip=False).set_stroke(MUTED, 2)
        ticks = VGroup(*[L(f"{v}", size=20, color=MUTED).next_to(axis.n2p(v), DOWN, buff=0.12)
                         for v in (0, 1200, 2400, 3600)])
        band = Rectangle(width=axis.n2p(2400)[0] - axis.n2p(0)[0], height=0.3)
        band.set_fill(SLOW_COLOR, 0.35).set_stroke(SLOW_COLOR, 1.2)
        band.move_to(axis.n2p(1200) + 0.3 * UP)
        band_lab = L("slow memory", size=20, color=SLOW_COLOR).move_to(band)
        kicks = VGroup(*[Line(axis.n2p(u), axis.n2p(u) + 0.75 * UP).set_stroke(HINGE_COLOR, 2.5) for u in fired])
        kick_lab = L("hinge firings", size=20, color=HINGE_COLOR).next_to(kicks, UP, buff=0.08)
        kick_lab.align_to(axis, LEFT)
        upd = L("updates", size=20, color=MUTED).next_to(axis, RIGHT, buff=0.15)
        plot = VGroup(axis, ticks, band, band_lab, kicks, kick_lab, upd)
        right = VGroup(right_head, plot).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        right.move_to([6.4, DETAIL_Y - 0.12, 0], aligned_edge=RIGHT)
        arrow = Arrow(left.get_right() + 0.1 * RIGHT, right.get_left() + 0.1 * LEFT, buff=0.15,
                      thickness=3).set_fill(WARN, 0.9).set_stroke(width=0)
        arrow.set_y(DETAIL_Y - 0.15)
        g = VGroup(left, arrow, right)
        g.intro = VGroup(left_head)

        def run(vo, s, badge):
            for a in att:
                self.play(ShowCreation(a[1]), FadeIn(a[0]), *([ShowCreation(chk)] if a is att[0] else []),
                          run_time=0.35)
                self.play(FadeIn(a[2], scale=0.6), run_time=0.2)
            vo.wait_until(cue(vo, s, "replaced") - 0.1)
            self.stamp(badge)
            self.play(left.animate.set_opacity(0.45), GrowArrow(arrow), FadeIn(right_head, shift=0.1 * RIGHT),
                      ShowCreation(axis), FadeIn(ticks), FadeIn(upd), run_time=0.8)
            vo.wait_until(cue(vo, s, "by the early-training recipe") - 0.6)
            self.play(GrowFromEdge(band, LEFT), FadeIn(band_lab), run_time=0.7)
            self.play(LaggedStartMap(GrowFromPoint, kicks, point=axis.n2p(0), lag_ratio=0.3),
                      FadeIn(kick_lab), run_time=0.8)
        g.run = run
        return g

    def detail_corrected(self):
        n = 10
        rows = VGroup()
        for m in ["X", "L"]:
            dots = VGroup(*[Dot(radius=0.09).set_fill(MACHINE_COLORS[m], 0.9) for _ in range(n)])
            dots.arrange(RIGHT, buff=0.22)
            rows.add(VGroup(machine_badge(m, size=20), dots).arrange(RIGHT, buff=0.35))
        rows.arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        rows.move_to([-6.3, DETAIL_Y - 0.05, 0], aligned_edge=LEFT)
        runs_lab = L("paired runs, one per seed", size=20, color=MUTED).next_to(rows, UP, buff=0.14).align_to(rows[0][1], LEFT)
        links = VGroup(*[DashedLine(a.get_bottom(), b.get_top(), dash_length=0.05).set_stroke(INK, 1.5)
                         for a, b in zip(rows[0][1], rows[1][1])])
        same = L("same seeds: same starting weights, same batches", size=20, color=INK)
        same.next_to(rows, DOWN, buff=0.16).align_to(rows[0][1], LEFT)
        brace = Brace(rows, RIGHT, buff=0.25).set_fill(MUTED)
        pool_head = L("Revision 6 pooled the pairs of X and L:", size=20, color=MUTED)
        p1 = L("p = 0.002 (layout)", size=22, color=INK)
        p2 = L("p = 0.049 (S = 4, k = 16)", size=22, color=INK)
        pv = VGroup(pool_head, p1, p2).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        pv.next_to(brace, RIGHT, buff=0.3)
        st1, st2 = strike(p1, width=2), strike(p2, width=2)
        desc = VGroup(Arrow(LEFT, RIGHT, buff=0, thickness=2.5).set_width(0.7).set_fill(WARN, 0.9).set_stroke(width=0),
                      VGroup(L("not independent:", size=20, color=MUTED),
                             L("descriptive only", size=24, color=WARN, weight="BOLD")).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
                      ).arrange(RIGHT, buff=0.25)
        desc.next_to(VGroup(p1, p2), RIGHT, buff=0.5)
        g = VGroup(rows, runs_lab, links, same, brace, pv, st1, st2, desc)
        g.intro = VGroup(rows, runs_lab)

        def run(vo, s, badge):
            vo.wait_until(cue(vo, s, "pooled"))
            self.play(GrowFromCenter(brace), FadeIn(pool_head, shift=0.1 * RIGHT), run_time=0.6)
            self.play(LaggedStartMap(FadeIn, VGroup(p1, p2), shift=0.1 * RIGHT, lag_ratio=0.4), run_time=0.8)
            vo.wait_until(cue(vo, s, "corrected") - 0.1)
            self.stamp(badge)
            vo.wait_until(cue(vo, s, "because both machines"))
            self.play(LaggedStartMap(ShowCreation, links, lag_ratio=0.08), FadeIn(same, shift=0.1 * UP), run_time=1.0)
            self.play(ShowCreation(st1), ShowCreation(st2), p1.animate.set_opacity(0.7),
                      p2.animate.set_opacity(0.7), FadeIn(desc, shift=0.1 * RIGHT), run_time=0.8)
        g.run = run
        return g

    def firing_ticks(self, card):
        """A short timeline with three orange ticks inside the hinge card: a few firings, early."""
        base = Line(ORIGIN, 1.2 * RIGHT).set_stroke(FAINT, 2)
        ticks = [Line(base.point_from_proportion(p), base.point_from_proportion(p) + 0.32 * UP).set_stroke(HINGE_COLOR, 3)
                 for p in (0.08, 0.16, 0.3)]
        g = VGroup(base, *ticks)
        g.move_to(card.rect.get_right() + 0.9 * LEFT, aligned_edge=DOWN).shift(0.08 * UP)
        return g
