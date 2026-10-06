"""Chapter 10 — Phase V: four streams (report Table 1, §4, Table 2; test_stream_recipe.py;
data/stream_recipe.json, data/report_tables.json)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load(name):
    with open(os.path.join(DATA, name)) as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------------- real data
_SR = _load("stream_recipe.json")
_T2 = _load("report_tables.json")["table2_stream_recipe"]
COUNTS = {r["arm"]: (r["X"], r["L"], r["n"]) for r in _T2["rows"]}   # Table 2

# A real restart-rule trial on X: A4k16_R, seed 247. Attempts 0-2 fail the check at step 4800,
# attempt 3 (seed 3247) passes at 0.596, sits near 0.75, and binds at step 15600.
_I247 = _SR["seeds"]["A4k16_R"].index(247)
TRIAL = _SR["per_seed_X"]["A4k16_R"]["attempts"][_I247]
TRIAL_CURVE = _SR["curves_X"]["A4k16_R|247"]
TRIAL_BIND = _SR["per_seed_X"]["A4k16_R"]["transition"][_I247]          # 15600

# Two k = 4 runs that passed the check (attempt 0 of A4k4_R) and then stalled as merges.
MERGE2 = _SR["curves_X"]["A4k4|258"]     # MERGED (2 share), 0.748 at 4800, ~0.75 to 28800
MERGE3 = _SR["curves_X"]["A4k4|244"]     # MERGED (3 share), 0.465 at 4800, ~0.5 to 28800

STEP = _SR["eval_every"]                 # 1200
T_CHECK, A_CHECK = _SR["check_rule"]["step"], _SR["check_rule"]["min_acc"]   # 4800, 0.4

X_COL, L_COL = MACHINE_COLORS["X"], MACHINE_COLORS["L"]
TWO_COL, THREE_COL = HINGE_COLOR, "#EC92AB"   # the 0.75 and the 0.5 plateau (not a stream 0-3 colour)

S_, NS_ = "SHOWN", "NOT SHOWN"
TABLE1 = [   # report Table 1, condensed: (test, question, §, X claims, L claims)
    ("stream recipe", "four streams: sixteen channels plus an early check and restarts", "§4",
     [("R1", "MAJORITY"), ("R2", "NOT PRECISE"), ("R3", S_), ("R4", NS_)],
     [("R1", "RELIABLE"), ("R2", "PRECISE"), ("R3", S_), ("R4", NS_)]),
    ("stream curriculum", "eight streams by a 2 → 4 → 8 stream curriculum", "§5",
     [("valid", "VALID"), ("C1", NS_), ("C2", "MINORITY"), ("C3", "UNTESTABLE")],
     [("valid", "VALID"), ("C1", NS_), ("C2", "MINORITY"), ("C3", "UNTESTABLE")]),
    ("slow start", "slow memory plus a hinge, no restarts", "§7",
     [("H0", S_), ("HA", S_), ("HB", NS_), ("S0", S_), ("H0S", NS_)],
     [("H0", S_), ("HA", S_), ("HB", S_), ("S0", S_), ("H0S", S_)]),
    ("recipe scope", "does the hinge need the slow phase; the recipe on the curriculum", "§8",
     [("Q1", NS_), ("Q2", S_), ("Q3", NS_), ("Q4", NS_), ("Q5", NS_)],
     [("Q1", NS_), ("Q2", S_), ("Q3", S_), ("Q4", NS_), ("Q5", NS_)]),
    ("early recipe", "the hinge in updates 1–2400 only; Muon", "§9.2",
     [("E1", NS_), ("E2", NS_)],
     [("E1", S_), ("E2", S_)]),
]


# ------------------------------------------------------------------------------------- helpers
def at_phrase(vo, i, phrase):
    """Wait until (roughly) the moment `phrase` is spoken in sentence i of the block."""
    text, a, b = vo.synth.sentences[i]
    k = text.find(phrase)
    frac = k / max(len(text), 1) if k >= 0 else 0.0
    vo.wait_until(a + frac * (b - a))


def claim_chip(name, verdict, size=20):
    color = VERDICT_STYLES.get(verdict, INK)
    t = L(name, size=size, color=color, weight="BOLD")
    w = max(t.get_width() + 0.24, 0.6)
    pill = RoundedRectangle(width=w, height=0.42, corner_radius=0.21)
    pill.set_fill(color, 0.16).set_stroke(color, 1.5)
    t.move_to(pill)
    return VGroup(pill, t)


def flow_box(lines, color, width=3.3, size=22, fill=0.10):
    txt = VGroup(*[L(s, size=size, color=c) for s, c in lines]).arrange(DOWN, buff=0.1)
    box = RoundedRectangle(width=width, height=txt.get_height() + 0.36, corner_radius=0.12)
    box.set_fill(color, fill).set_stroke(color, 2)
    txt.move_to(box)
    return VGroup(box, txt)


def cross_mark(point, color=BAD, r=0.13):
    return VGroup(Line(point + r * (UL), point + r * (DR)), Line(point + r * (UR), point + r * (DL))).set_stroke(color, 4)


def check_marker(axes, label=True):
    """The restart rule's check: a slit at step 4800, green above 0.4 (continue), red below."""
    low = Line(axes.c2p(T_CHECK, 0), axes.c2p(T_CHECK, A_CHECK)).set_stroke(BAD, 6, opacity=0.55)
    high = Line(axes.c2p(T_CHECK, A_CHECK), axes.c2p(T_CHECK, 1)).set_stroke(GOOD, 6, opacity=0.45)
    bar = Line(axes.c2p(T_CHECK, A_CHECK) + 0.18 * LEFT, axes.c2p(T_CHECK, A_CHECK) + 0.18 * RIGHT)
    bar.set_stroke(INK, 3)
    hline = DashedLine(axes.c2p(0, A_CHECK), axes.c2p(T_CHECK, A_CHECK), dash_length=0.08)
    hline.set_stroke(FAINT, 1.5)
    g = VGroup(hline, low, high, bar)
    if label:
        lab = L("check at step 4800", size=20, color=INK).next_to(high, UP, buff=0.12)
        g.add(lab)
        g.lab = lab
    g.low, g.high, g.bar = low, high, bar
    return g


def steps_for(n, start=1):
    return [STEP * (i + start) for i in range(n)]


def channel_icon(n, filled, cell=0.24):
    sq = VGroup(*[Square(cell).set_stroke(FAINT, 1.2).set_fill(PANEL, 1) for _ in range(n)])
    sq.arrange(RIGHT, buff=0.05)
    for i in range(filled):
        sq[i].set_fill(STREAM_COLORS[i], 0.85).set_stroke(STREAM_COLORS[i], 1.2)
    return sq


def stream_channel(colors, cell=0.2):
    """A small memory channel; a channel holding several streams is a checker of their colours."""
    g = memory_grid(3, 3, cell=cell, color=FAINT)
    for i, c in enumerate(g.cells):
        if colors:
            col = colors[(i + (i // 3)) % len(colors)] if len(colors) > 1 else colors[0]
            c.set_fill(col, 0.7).set_stroke(col, 1, opacity=0.8)
        else:
            c.set_fill(PANEL, 1).set_stroke(FAINT, 1)
    return g


class FourStreams(ClankersScene):
    def construct(self):
        chap = self.chapter_card(10, "Phase V: four streams")
        self.wait(0.6)
        title = section_title("Phase V: four streams")
        self.play(FadeOut(chap, shift=0.3 * UP), FadeIn(title, shift=0.3 * UP))

        # ============================================================ N10.1 Table 1, condensed
        LEFT_X, XC, LC = -6.5, 1.3, 4.85
        ys = [1.55 - 0.82 * i for i in range(5)]
        head = VGroup(
            L("Phase V test", size=20, color=MUTED, weight="BOLD").move_to([LEFT_X, 2.35, 0], aligned_edge=LEFT),
            VGroup(machine_badge("X", size=20), L("verdicts on X", size=20, color=MUTED)).arrange(RIGHT, buff=0.15)
            .move_to([XC, 2.35, 0]),
            VGroup(machine_badge("L", size=20), L("verdicts on L", size=20, color=MUTED)).arrange(RIGHT, buff=0.15)
            .move_to([LC, 2.35, 0]),
        )
        top_rule = Line([LEFT_X, 2.0, 0], [6.6, 2.0, 0]).set_stroke(INK, 1.5)
        rows = VGroup()
        for (name, q, sec, xc, lc), y in zip(TABLE1, ys):
            nm = L(name, size=24, weight="BOLD")
            sc = L(sec, size=20, color=FAINT)
            nm_row = VGroup(nm, sc).arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
            qq = L(q, size=20, color=MUTED)
            left = VGroup(nm_row, qq).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
            left.move_to([LEFT_X, y, 0], aligned_edge=LEFT)
            xchips = VGroup(*[claim_chip(a, b) for a, b in xc]).arrange(RIGHT, buff=0.08).move_to([XC, y, 0])
            lchips = VGroup(*[claim_chip(a, b) for a, b in lc]).arrange(RIGHT, buff=0.08).move_to([LC, y, 0])
            rule = Line([LEFT_X, y - 0.41, 0], [6.6, y - 0.41, 0]).set_stroke(PANEL_EDGE, 1)
            row = VGroup(left, xchips, lchips, rule)
            row.left, row.xchips, row.lchips = left, xchips, lchips
            rows.add(row)
        e_line = VGroup(machine_badge("E", size=20),
                        L("+ 13 batches of exploratory screens (batches 2–14, machine E)", size=20, color=INK))
        e_line.arrange(RIGHT, buff=0.2).move_to([LEFT_X, ys[-1] - 0.75, 0], aligned_edge=LEFT)
        legend = VGroup()
        for word, col in [("shown, reliable, precise", GOOD), ("valid", VERDICT_STYLES["VALID"]),
                          ("majority", WARN), ("minority", HINGE_COLOR), ("not shown, untestable", MUTED)]:
            legend.add(VGroup(Dot(radius=0.07, fill_color=col), L(word, size=20, color=col)).arrange(RIGHT, buff=0.1))
        legend.arrange(RIGHT, buff=0.35).move_to([LEFT_X, -3.35, 0], aligned_edge=LEFT)
        src1 = source_note("Report Table 1")

        with self.voiceover(
            "Phase Five adds five pre-registered tests, each run on both machines, and thirteen batches of "
            "screens. The first asked about four streams."
        ) as vo:
            self.play(FadeIn(head[0]), ShowCreation(top_rule), FadeIn(src1), run_time=0.6)
            self.play(LaggedStart(*[FadeIn(r.left, shift=0.15 * RIGHT) for r in rows], lag_ratio=0.18),
                      LaggedStart(*[ShowCreation(r[3]) for r in rows], lag_ratio=0.18), run_time=1.6)
            at_phrase(vo, 0, "each run on both")
            self.play(FadeIn(head[1], shift=0.1 * DOWN), FadeIn(head[2], shift=0.1 * DOWN), run_time=0.5)
            self.play(LaggedStart(*[FadeIn(VGroup(r.xchips, r.lchips), scale=0.9) for r in rows], lag_ratio=0.15),
                      FadeIn(legend), run_time=1.4)
            at_phrase(vo, 0, "thirteen batches")
            self.play(FadeIn(e_line, shift=0.15 * UP), run_time=0.7)
            vo.wait_until_sentence(1)
            hl = Rectangle(width=13.2, height=0.8).set_fill(WARN, 0.10).set_stroke(WARN, 1.5)
            hl.move_to([0.05, ys[0], 0])
            self.play(FadeIn(hl),
                      *[r.animate.set_opacity(0.3) for r in rows[1:]],
                      e_line.animate.set_opacity(0.3), legend.animate.set_opacity(0.3), run_time=0.8)
            self.play(Indicate(rows[0].left[0], color=WARN, scale_factor=1.06), run_time=0.9)

        # header for the rest of the chapter: the test's name
        test_tag = L("test_stream_recipe.py", size=22, color=MUTED)
        test_tag.to_edge(RIGHT, buff=0.45).match_y(title[0])
        self.play(FadeOut(VGroup(head, top_rule, rows[1:], e_line, legend, hl, rows[0].xchips, rows[0].lchips,
                                 rows[0][3], src1)),
                  ReplacementTransform(rows[0].left[0][0].copy(), test_tag),
                  FadeOut(rows[0].left), run_time=0.9)

        # ============================================================ N10.2 the restart rule
        axes = line_chart([0, 19200, 2400], [0, 1, 0.25], width=6.6, height=3.5,
                          x_label="training step", y_label="held-out accuracy",
                          x_ticks=[0, 4800, 9600, 14400], y_ticks=[0, 0.4, 1])
        axes.move_to([-2.45, -0.15, 0])
        chk = check_marker(axes)

        # Revision 6's post-hoc observation: 80 runs (one dot each), 38 passed the check, all 38 bound
        r6_title = VGroup(L("Revision 6, after the fact", size=22, color=MUTED),
                          L("80 four-stream runs with 8 or 16 channels", size=20, color=MUTED))
        r6_title.arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        dots = VGroup(*[Dot(radius=0.065).set_fill(FAINT, 1) for _ in range(80)]).arrange_in_grid(4, 20, buff=0.08)
        passers = dots[:38]
        r6_pass = L("passed the check: 38", size=24, color=WARN)
        r6_bound = L("of those, bound: 38 of 38", size=24, color=GOOD)
        r6_miss = L("(it missed 8 late binders)", size=20, color=MUTED)
        r6 = VGroup(r6_title, dots, r6_pass, r6_bound, r6_miss).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        r6_card = card(r6, buff=0.3)
        r6_card.move_to([4.25, 0.3, 0])
        src2a = source_note("Report §4; test_stream_recipe.py docstring")
        cont_lab = L("≥ 0.4: continue", size=20, color=GOOD).next_to(axes.c2p(T_CHECK, 0.88), RIGHT, buff=0.12)
        rest_lab = L("< 0.4: restart", size=20, color=BAD).next_to(axes.c2p(T_CHECK, 0.12), RIGHT, buff=0.75)

        FX = 3.55
        box_a = flow_box([("attempt j: train seed s + 1000·j", INK)], MUTED, width=3.6, size=20)
        box_b = flow_box([("at step 4800:", INK), ("held-out accuracy ≥ 0.4 ?", WARN)], WARN, width=3.6, size=20)
        box_c = flow_box([("yes: continue to the end", GOOD)], GOOD, width=3.6, size=20)
        flow = VGroup(box_a, box_b, box_c).arrange(DOWN, buff=0.55).move_to([FX, 0.55, 0])
        ar_ab = Arrow(box_a.get_bottom(), box_b.get_top(), buff=0.06, thickness=3).set_color(MUTED)
        ar_bc = Arrow(box_b.get_bottom(), box_c.get_top(), buff=0.06, thickness=3).set_color(GOOD)
        loop = CurvedArrow(box_b.get_right() + 0.02 * RIGHT, box_a.get_right() + 0.02 * RIGHT, angle=PI * 0.75)
        loop.set_stroke(BAD, 3).set_fill(BAD)
        no_grp = L("no: restart", size=20, color=BAD, weight="BOLD")
        no_grp.move_to([(ar_ab.get_right()[0] + box_a.get_right()[0]) / 2 + 0.1, ar_ab.get_center()[1], 0])
        rule_notes = VGroup(L("at most 5 attempts per trial", size=20, color=INK),
                            L("the 5th continues regardless", size=20, color=MUTED))
        rule_notes.arrange(DOWN, buff=0.1).next_to(box_c, DOWN, buff=0.35)

        setup = VGroup(*[verdict_badge(t, size=20, color=c) for t, c in [
            ("S = 4 streams", INK), ("P = 4 keys", KEY_COLOR), ("convolution", MEMORY_COLOR),
            ("fresh seeds 240–259", MUTED)]]).arrange(RIGHT, buff=0.18)
        setup.move_to([0, -3.3, 0]).to_edge(LEFT, buff=0.5)
        src2b = source_note("Report §4; X run A4k16_R seed 247 (data/stream_recipe.json)")

        # the trial's curves
        def attempt_curve(vals, color):
            return polyline(axes, steps_for(len(vals)), vals, color=color, width=3.5)

        att_curves = [attempt_curve(a["early"], INK) for a in TRIAL]
        att_labels = [L(f"attempt {a['j'] + 1}: seed {a['seed']}", size=20, color=INK) for a in TRIAL]
        for lab in att_labels:
            lab.move_to(axes.c2p(13200, 0.32))
        att_vals = [L(f"{a['acc_4800']:.2f}", size=20, color=GOOD if a["passed"] else BAD, weight="BOLD")
                    for a in TRIAL]
        for v, a in zip(att_vals, TRIAL):
            v.next_to(axes.c2p(T_CHECK, a["acc_4800"]), RIGHT, buff=0.18)
        n_tail = len(TRIAL_CURVE) - len(TRIAL[-1]["early"])
        tail = polyline(axes, steps_for(n_tail + 1, start=len(TRIAL[-1]["early"])),
                        TRIAL_CURVE[len(TRIAL[-1]["early"]) - 1:], color=GOOD, width=3.5)
        bind_dot = Dot(axes.c2p(TRIAL_BIND, 1.0), radius=0.08, fill_color=GOOD)
        bind_lab = L(f"binds at step {TRIAL_BIND:,}", size=20, color=GOOD).next_to(bind_dot, UP, buff=0.15)

        with self.voiceover(
            "Revision Six had noticed, after the fact, that at four streams with spare channels, a held-out "
            "accuracy of at least 0.4 at step forty-eight hundred had picked the eventual binders with "
            "thirty-eight of thirty-eight precision. The stream recipe test made that a restart rule on fresh "
            "seeds: four streams, four keys, the convolution; continue a run if it clears 0.4 at step "
            "forty-eight hundred, otherwise restart from a new seed, up to five attempts, the fifth continuing "
            "regardless."
        ) as vo:
            self.play(FadeIn(axes), run_time=0.8)
            at_phrase(vo, 0, "at four streams")
            self.play(FadeIn(r6_card[0]), FadeIn(r6_title), FadeIn(src2a), run_time=0.6)
            self.play(LaggedStartMap(FadeIn, dots, scale=0.5, lag_ratio=0.02), run_time=1.0)
            at_phrase(vo, 0, "a held-out")
            self.play(ShowCreation(chk[1:4]), ShowCreation(chk[0]), FadeIn(chk.lab, shift=0.1 * DOWN), run_time=1.0)
            self.play(Indicate(chk.bar, color=WARN), run_time=0.8)
            at_phrase(vo, 0, "had picked")
            self.play(LaggedStart(*[d.animate.set_fill(WARN) for d in passers], lag_ratio=0.03),
                      FadeIn(r6_pass, shift=0.1 * DOWN), run_time=1.0)
            at_phrase(vo, 0, "thirty-eight of")
            self.play(LaggedStart(*[d.animate.set_fill(GOOD) for d in passers], lag_ratio=0.03),
                      FadeIn(r6_bound, shift=0.1 * DOWN), run_time=1.0)
            self.play(FadeIn(r6_miss), run_time=0.5)

            vo.wait_until_sentence(1)
            self.play(FadeOut(r6_card), FadeOut(src2a), FadeIn(src2b), run_time=0.6)
            at_phrase(vo, 1, "made that")
            self.play(FadeIn(cont_lab, shift=0.1 * RIGHT), FadeIn(rest_lab, shift=0.1 * RIGHT),
                      chk.high.animate.set_stroke(GOOD, 8, opacity=0.8), chk.low.animate.set_stroke(BAD, 8, opacity=0.8),
                      run_time=0.7)
            at_phrase(vo, 1, "on fresh")
            self.play(FadeIn(setup[3], shift=0.1 * UP), run_time=0.5)
            at_phrase(vo, 1, "four streams, four")
            self.play(LaggedStartMap(FadeIn, setup[:3], shift=0.1 * UP, lag_ratio=0.3), run_time=1.0)
            at_phrase(vo, 1, "continue a run")
            # attempt 1 of the real trial heads for the check while the flowchart is built
            self.play(FadeIn(box_a, shift=0.1 * DOWN), FadeIn(att_labels[0]), run_time=0.5)
            self.play(GrowArrow(ar_ab), FadeIn(box_b, shift=0.1 * DOWN),
                      ShowCreation(att_curves[0], rate_func=linear), run_time=1.0)
            self.play(GrowArrow(ar_bc), FadeIn(box_c, shift=0.1 * DOWN), run_time=0.6)

            def fail(i, first=False):
                a = TRIAL[i]
                xm = cross_mark(axes.c2p(T_CHECK, a["acc_4800"]))
                extra = [ShowCreation(loop), FadeIn(no_grp)] if first else [Indicate(no_grp, color=BAD)]
                self.play(FadeIn(xm, scale=0.6), FadeIn(att_vals[i]), box_b[0].animate.set_stroke(BAD, 3), *extra,
                          run_time=0.7 if first else 0.45)
                self.play(att_curves[i].animate.set_stroke(FAINT, 2), FadeOut(xm), FadeOut(att_vals[i]),
                          box_b[0].animate.set_stroke(WARN, 2), run_time=0.3)

            at_phrase(vo, 1, "otherwise restart")
            fail(0, first=True)
            # attempts 2 and 3 (seeds 1247, 2247) fail too; attempt 4 (seed 3247) passes
            for i in (1, 2):
                extra = [FadeIn(rule_notes[0], shift=0.1 * UP)] if i == 1 else []
                self.play(ShowCreation(att_curves[i], rate_func=linear),
                          ReplacementTransform(att_labels[i - 1], att_labels[i]), *extra, run_time=0.6)
                fail(i)
            self.play(ShowCreation(att_curves[3], rate_func=linear),
                      ReplacementTransform(att_labels[2], att_labels[3]), run_time=0.7)
            self.play(att_curves[3].animate.set_stroke(GOOD, 3.5), FadeIn(att_vals[3]),
                      Flash(axes.c2p(T_CHECK, TRIAL[3]["acc_4800"]), color=GOOD),
                      box_b[0].animate.set_stroke(GOOD, 3), Indicate(box_c, color=GOOD), run_time=0.6)
            self.play(ShowCreation(tail, rate_func=linear), FadeIn(rule_notes[1], shift=0.1 * UP), run_time=1.5)
            self.play(FadeIn(bind_dot, scale=0.5), FadeIn(bind_lab), run_time=0.5)

        # ---------------------------------------------------------------- the arms
        k16 = channel_icon(16, 4, cell=0.22)
        k4 = channel_icon(4, 4, cell=0.22)
        k16_lab = VGroup(L("k = 16 channels", size=24, weight="BOLD"), L("twelve spare", size=20, color=MUTED))
        k4_lab = VGroup(L("k = 4 channels", size=24, weight="BOLD"), L("one per stream: the control", size=20,
                                                                         color=MUTED))
        for lab in (k16_lab, k4_lab):
            lab.arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        row16 = VGroup(k16_lab, k16).arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to([-6.3, 0.75, 0],
                                                                                         aligned_edge=LEFT)
        row4 = VGroup(k4_lab, k4).arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to([-6.3, -1.05, 0],
                                                                                      aligned_edge=LEFT)
        CX = [0.9, 4.2]
        col_heads = VGroup(L("plain", size=24, color=MUTED), L("restart rule", size=24, color=MUTED))
        for c, x in zip(col_heads, CX):
            c.move_to([x, 2.15, 0])
        arm_names = {}
        for arm, x, y in [("A4k16", CX[0], 0.75), ("A4k16_R", CX[1], 0.75), ("A4k4", CX[0], -1.05),
                          ("A4k4_R", CX[1], -1.05)]:
            arm_names[arm] = L(arm, size=26, color=INK, weight="BOLD").move_to([x, y, 0])
        cells = VGroup(*[SurroundingRectangle(arm_names[a], buff=0.0).set_width(2.6, stretch=True)
                         .set_height(1.2, stretch=True).move_to(arm_names[a]).set_stroke(PANEL_EDGE, 1.5)
                         .set_fill(PANEL, 1).round_corners(0.1) for a in arm_names])
        ceil_note = L("plus a perfect gate (k = 16, seeds 240–242) as a validity check", size=20, color=MUTED)
        ceil_note.move_to([0, -2.6, 0])
        src2c = source_note("Report §4; test_stream_recipe.py, Part 1")

        trial_mobs = VGroup(axes, chk, cont_lab, rest_lab, flow, ar_ab, ar_bc, loop, no_grp, rule_notes, setup, *att_curves,
                            att_labels[-1], att_vals[-1], tail, bind_dot, bind_lab, src2b)
        with self.voiceover("It ran with sixteen channels, and with four as a control.") as vo:
            self.play(FadeOut(trial_mobs), run_time=0.6)
            self.play(FadeIn(row16, shift=0.15 * RIGHT), FadeIn(src2c), run_time=0.7)
            at_phrase(vo, 0, "with four")
            self.play(FadeIn(row4, shift=0.15 * RIGHT), run_time=0.7)
            self.play(FadeIn(col_heads), LaggedStartMap(FadeIn, cells, lag_ratio=0.15),
                      LaggedStartMap(FadeIn, VGroup(*arm_names.values()), lag_ratio=0.15), run_time=1.0)
            self.play(FadeIn(ceil_note, shift=0.1 * UP), run_time=0.5)

        # ============================================================ N10.3 Table 2 as paired bars
        BASE, H, X0 = -1.75, 3.3, -5.75
        GC = [-4.4, -2.2, 0.0, 2.2, 4.95]
        ARMS = ["A4k16", "A4k16_R", "A4k4", "A4k4_R", "ceiling4k16"]
        SUBS = ["k = 16, plain", "k = 16, restarts", "k = 4, plain", "k = 4, restarts", "perfect gate"]
        y_axis = Line([X0, BASE, 0], [X0, BASE + H, 0]).set_stroke(MUTED, 2)
        x_axis = Line([X0, BASE, 0], [6.3, BASE, 0]).set_stroke(MUTED, 2)
        yt = VGroup()
        for v in (0, 0.5, 1):
            yt.add(L(f"{v:g}", size=20, color=MUTED).next_to([X0, BASE + H * v, 0], LEFT, buff=0.15))
            if v:
                yt.add(DashedLine([X0, BASE + H * v, 0], [6.3, BASE + H * v, 0], dash_length=0.06)
                       .set_stroke(FAINT, 1, opacity=0.6))
        y_lab = L("fraction of runs that bound", size=20, color=MUTED).rotate(PI / 2)
        y_lab.next_to(yt, LEFT, buff=0.15)
        sep = DashedLine([3.6, BASE, 0], [3.6, BASE + H, 0], dash_length=0.08).set_stroke(FAINT, 1.5)
        legend3 = VGroup(
            VGroup(machine_badge("X", size=20), L("machine X", size=20, color=X_COL)).arrange(RIGHT, buff=0.12),
            VGroup(machine_badge("L", size=20), L("machine L", size=20, color=L_COL)).arrange(RIGHT, buff=0.12))
        legend3.arrange(RIGHT, buff=0.4).move_to([4.6, 2.55, 0])

        bars, vals, glabels = {}, {}, {}
        BW = 0.62
        for arm, gx, sub in zip(ARMS, GC, SUBS):
            x_n, l_n, n = COUNTS[arm]
            pair = []
            vpair = []
            for num, col, dx in ((x_n, X_COL, -0.36), (l_n, L_COL, 0.36)):
                b = Rectangle(width=BW, height=max(H * num / n, 0.01)).set_fill(col, 0.85).set_stroke(width=0)
                b.move_to([gx + dx, BASE, 0], aligned_edge=DOWN)
                if arm == "ceiling4k16":
                    b.set_fill(col, 0.35).set_stroke(col, 1.5)
                v = L(f"{num}/{n}", size=20, color=col, weight="BOLD").next_to(b, UP, buff=0.08)
                pair.append(b)
                vpair.append(v)
            bars[arm] = VGroup(*pair)
            vals[arm] = VGroup(*vpair)
            nm = L(arm if arm != "ceiling4k16" else "perfect gate", size=22, weight="BOLD")
            sb = L(sub if arm != "ceiling4k16" else "k = 16, of 3", size=20, color=MUTED)
            glabels[arm] = VGroup(nm, sb).arrange(DOWN, buff=0.08).next_to([gx, BASE, 0], DOWN, buff=0.2)

        def gain_arc(a, b, text, color):
            p = vals[a].get_top() + 0.12 * UP
            q = vals[b].get_top() + 0.12 * UP
            arc = Arrow(p, q, buff=0.05, path_arc=-PI / 2.5, thickness=2.5).set_color(color)
            lab = L(text, size=20, color=color).next_to(arc, UP, buff=0.1)
            return VGroup(arc, lab)

        gain16 = gain_arc("A4k16", "A4k16_R", "restarts: +5 on X, +10 on L", GOOD)
        gain4 = gain_arc("A4k4", "A4k4_R", "restarts: +0 on X, +3 on L", MUTED)
        src3 = source_note("Report §4, Table 2")

        def grow_up(b):
            b.save_state()
            b.stretch(1e-3, 1, about_edge=DOWN)
            return Restore(b)

        def grow(arm):
            return [grow_up(bars[arm][0]), grow_up(bars[arm][1]), FadeIn(vals[arm], shift=0.1 * UP)]

        with self.voiceover(
            "With sixteen channels and restarts, seventeen of twenty bound on X and nineteen of twenty on L. "
            "Without restarts, twelve and nine. With four channels, restarts barely helped: ten and eight, "
            "against ten and five."
        ) as vo:
            self.play(FadeOut(VGroup(row16, row4, col_heads, cells, ceil_note, src2c)), run_time=0.5)
            self.play(*[ReplacementTransform(arm_names[a], glabels[a][0]) for a in ARMS[:4]],
                      *[FadeIn(glabels[a][1]) for a in ARMS[:4]],
                      ShowCreation(x_axis), ShowCreation(y_axis), FadeIn(yt), FadeIn(y_lab), FadeIn(legend3),
                      FadeIn(src3), run_time=1.2)
            at_phrase(vo, 0, "seventeen of")
            self.play(*grow("A4k16_R"), Indicate(glabels["A4k16_R"][0], color=GOOD), run_time=1.0)
            vo.wait_until_sentence(1)
            self.play(*grow("A4k16"), run_time=0.9)
            self.play(GrowArrow(gain16[0]), FadeIn(gain16[1], shift=0.1 * UP), run_time=0.8)
            vo.wait_until_sentence(2)
            self.play(*grow("A4k4_R"), run_time=0.9)
            at_phrase(vo, 2, "against ten")
            self.play(*grow("A4k4"), run_time=0.9)
            self.play(GrowArrow(gain4[0]), FadeIn(gain4[1], shift=0.1 * UP), run_time=0.8)
            self.play(ShowCreation(sep), FadeIn(glabels["ceiling4k16"]), *grow("ceiling4k16"), run_time=1.0)

        # ============================================================ N10.4 merges
        chart3 = VGroup(x_axis, y_axis, yt, y_lab, sep, legend3, src3, gain16, gain4,
                        *bars.values(), *vals.values(), *glabels.values())
        SX = [-5.0, -3.8, -2.6, -1.4]
        chips = VGroup(*[token(f"CTX{s}", "ctx", s) for s in range(4)])
        for c, x in zip(chips, SX):
            c.move_to([x, 1.45, 0])
        chip_lab = L("stream", size=20, color=MUTED).next_to(chips, LEFT, buff=0.2)

        def channel_row(assign):
            """assign[s] = channel of stream s; returns 4 small channels with their contents."""
            row = VGroup()
            for c in range(4):
                held = [STREAM_COLORS[s] for s in range(4) if assign[s] == c]
                ch = stream_channel(held)
                ch.move_to([SX[c], -0.05, 0])
                lab = L(f"ch{c}", size=20, color=MUTED).next_to(ch, DOWN, buff=0.12)
                row.add(VGroup(ch, lab))
            return row

        def route_arrows(assign, row):
            arr = VGroup()
            for s in range(4):
                tgt = row[assign[s]][0]
                off = 0.0
                sharers = [t for t in range(4) if assign[t] == assign[s]]
                if len(sharers) > 1:
                    off = (sharers.index(s) - (len(sharers) - 1) / 2) * 0.22
                a = Arrow(chips[s].get_bottom(), tgt.get_top() + off * RIGHT, buff=0.1, thickness=2.5)
                a.set_color(STREAM_COLORS[s])
                arr.add(a)
            return arr

        A2 = [0, 1, 2, 2]     # streams 2 and 3 share ch2
        A3 = [0, 1, 1, 1]     # streams 1, 2, 3 share ch1
        chans_a = channel_row(A2)
        arrows_a = route_arrows(A2, chans_a)
        chans_b = channel_row(A3)
        arrows_b = route_arrows(A3, chans_b)
        merge_box_a = SurroundingRectangle(chans_a[2][0], buff=0.1).set_stroke(TWO_COL, 2.5)
        merge_box_b = SurroundingRectangle(chans_b[1][0], buff=0.1).set_stroke(THREE_COL, 2.5)

        def acc_labels(texts, color):
            g = VGroup()
            for s, t in enumerate(texts):
                m = M(t, size=34, color=INK if t == "1" else color)
                m.next_to(chips[s], UP, buff=0.2)
                g.add(m)
            return g

        acc_a = acc_labels(["1", "1", R"\tfrac12", R"\tfrac12"], TWO_COL)
        acc_b = acc_labels(["1", R"\tfrac13", R"\tfrac13", R"\tfrac13"], THREE_COL)
        acc_cap = L("accuracy", size=20, color=MUTED).next_to(acc_a, LEFT, buff=0.2).align_to(chip_lab, RIGHT)

        form_a = M(R"\frac{1 + 1 + \tfrac12 + \tfrac12}{4} = 0.75", size=38)
        TERMS_A = [(0, 1), (2, 3), (4, 7), (8, 11)]          # glyph ranges of 1, 1, 1/2, 1/2
        for a, b in TERMS_A[2:]:
            form_a[a:b].set_color(TWO_COL)
        form_a[14:].set_color(TWO_COL)
        form_b = M(R"\frac{1 + \tfrac13 + \tfrac13 + \tfrac13}{4} = 0.5", size=38)
        TERMS_B = [(0, 1), (2, 5), (6, 9), (10, 13)]         # 1, 1/3, 1/3, 1/3
        for a, b in TERMS_B[1:]:
            form_b[a:b].set_color(THREE_COL)
        form_b[16:].set_color(THREE_COL)
        form_a.move_to([-3.2, -1.75, 0])
        form_b.move_to([-3.2, -3.0, 0])

        def build_formula(acc, form, terms):
            anims = [TransformFromCopy(acc[s], form[a:b]) for s, (a, b) in enumerate(terms)]
            used = {i for a, b in terms for i in range(a, b)}
            anims.append(FadeIn(VGroup(*[form[i] for i in range(len(form)) if i not in used])))
            return anims
        fa_lab = L("two share a channel", size=20, color=TWO_COL).next_to(form_a, UP, buff=0.12)
        fb_lab = L("three share a channel", size=20, color=THREE_COL).next_to(form_b, UP, buff=0.12)

        ax4 = line_chart([0, 28800, 4800], [0, 1, 0.25], width=5.2, height=3.3,
                         x_label="training step", y_label="held-out accuracy",
                         x_ticks=[0, 4800, 14400, 28800], y_ticks=[0, 0.4, 1])
        ax4.move_to([3.7, 0.35, 0])
        chk4 = check_marker(ax4)
        plat75 = DashedLine(ax4.c2p(0, 0.75), ax4.c2p(28800, 0.75), dash_length=0.08).set_stroke(TWO_COL, 2)
        plat50 = DashedLine(ax4.c2p(0, 0.5), ax4.c2p(28800, 0.5), dash_length=0.08).set_stroke(THREE_COL, 2)
        p75_lab = L("0.75", size=20, color=TWO_COL, weight="BOLD").next_to(ax4.c2p(28800, 0.75), UP, buff=0.08)
        p50_lab = L("0.5", size=20, color=THREE_COL, weight="BOLD").next_to(ax4.c2p(28800, 0.5), UP, buff=0.08)
        p75_lab.shift(0.25 * LEFT)
        p50_lab.shift(0.2 * LEFT)
        c2 = polyline(ax4, steps_for(len(MERGE2)), MERGE2, color=TWO_COL, width=3.5)
        c3 = polyline(ax4, steps_for(len(MERGE3)), MERGE3, color=THREE_COL, width=3.5)
        pass2 = Dot(ax4.c2p(T_CHECK, MERGE2[3]), radius=0.07, fill_color=GOOD)
        pass3 = Dot(ax4.c2p(T_CHECK, MERGE3[3]), radius=0.07, fill_color=GOOD)
        stall = L("pass, then stall", size=20, color=INK).move_to(ax4.c2p(17500, 0.25))

        verdict4 = VGroup(
            L("passed the check, then failed: all merges", size=22, color=INK),
            VGroup(machine_badge("X", size=20), L("9 of 9", size=26, color=X_COL, weight="BOLD"),
                   machine_badge("L", size=20), L("12 of 12", size=26, color=L_COL, weight="BOLD"))
            .arrange(RIGHT, buff=0.18),
        ).arrange(DOWN, buff=0.18)
        verdict4[1][2].shift(0.35 * RIGHT)
        verdict4[1][3].shift(0.35 * RIGHT)
        verdict4.move_to([3.7, -2.85, 0])
        src4 = source_note("Report §4; curves: X, k = 4, seeds 258 and 244 (data/stream_recipe.json)")

        with self.voiceover(
            "The reason is merges. If two of the four streams share a channel, those two are each right half "
            "the time, and the run's accuracy sits near three quarters. Three streams in one channel gives about "
            "one half. Both clear the 0.4 bar, and then stall. With four channels, every attempt that passed the "
            "check and then failed was a merge: nine of nine on X, twelve of twelve on L."
        ) as vo:
            self.play(FadeOut(chart3), run_time=0.6)
            self.play(LaggedStartMap(FadeIn, chips, shift=0.1 * DOWN, lag_ratio=0.15), FadeIn(chip_lab),
                      LaggedStartMap(FadeIn, VGroup(*[r[1] for r in chans_a]), lag_ratio=0.15),
                      *[FadeIn(r[0][0].copy().set_fill(PANEL, 1).set_stroke(FAINT, 1)) for r in chans_a],
                      FadeIn(src4), run_time=1.0)
            vo.wait_until_sentence(1)
            self.play(LaggedStartMap(GrowArrow, arrows_a, lag_ratio=0.2),
                      LaggedStart(*[FadeIn(r[0]) for r in chans_a], lag_ratio=0.2), run_time=1.3)
            self.play(ShowCreation(merge_box_a), run_time=0.5)
            at_phrase(vo, 1, "those two")
            self.play(FadeIn(acc_cap), LaggedStartMap(FadeIn, acc_a, shift=0.1 * DOWN, lag_ratio=0.15), run_time=1.0)
            at_phrase(vo, 1, "and the run")
            self.play(FadeIn(fa_lab), *build_formula(acc_a, form_a, TERMS_A), run_time=1.3)
            vo.wait_until_sentence(2)
            self.play(ReplacementTransform(arrows_a, arrows_b), ReplacementTransform(chans_a, chans_b),
                      ReplacementTransform(merge_box_a, merge_box_b),
                      ReplacementTransform(acc_a, acc_b), run_time=1.2)
            self.play(FadeIn(fb_lab), *build_formula(acc_b, form_b, TERMS_B), run_time=1.2)
            vo.wait_until_sentence(3)
            self.play(FadeIn(ax4), FadeIn(chk4), run_time=0.7)
            self.play(ShowCreation(c2, rate_func=linear), ShowCreation(c3, rate_func=linear), run_time=1.8)
            self.play(FadeIn(pass2, scale=0.5), FadeIn(pass3, scale=0.5),
                      Flash(ax4.c2p(T_CHECK, MERGE2[3]), color=GOOD), Flash(ax4.c2p(T_CHECK, MERGE3[3]), color=GOOD),
                      run_time=0.7)
            self.play(ShowCreation(plat75), ShowCreation(plat50), FadeIn(stall),
                      TransformFromCopy(form_a[14:], p75_lab), TransformFromCopy(form_b[16:], p50_lab), run_time=1.0)
            vo.wait_until_sentence(4)
            self.play(FadeIn(verdict4[0], shift=0.1 * UP), run_time=0.7)
            at_phrase(vo, 4, "nine of nine")
            self.play(FadeIn(verdict4[1][:2], scale=0.9), run_time=0.6)
            at_phrase(vo, 4, "twelve of twelve")
            self.play(FadeIn(verdict4[1][2:], scale=0.9), run_time=0.6)
        self.clear_all(exclude=(title, test_tag))

        # ============================================================ N10.5 verdicts
        CL, VX, VL = -6.4, 1.55, 4.75
        vhead = VGroup(VGroup(machine_badge("X", size=20), L("on X", size=20, color=MUTED)).arrange(RIGHT, buff=0.12)
                       .move_to([VX, 2.4, 0]),
                       VGroup(machine_badge("L", size=20), L("on L", size=20, color=MUTED)).arrange(RIGHT, buff=0.12)
                       .move_to([VL, 2.4, 0]))
        claims = [
            ("R1", "the recipe is reliable", "k = 16 with restarts", ("MAJORITY", "17/20"), ("RELIABLE", "19/20")),
            ("R2", "the check is precise", "≥ 0.9 of passing attempts bind", ("NOT PRECISE", "17/20 = 0.85"),
             ("PRECISE", "19/19")),
            ("R3", "spare channels make restarts work", "k = 16 vs k = 4, both with restarts",
             ("SHOWN", "p = 0.020"), ("SHOWN", "p = 0.0002")),
            ("R4", "spare channels alone", "k = 16 vs k = 4, no restarts", ("NOT SHOWN", "5 vs 3, p = 0.36"),
             ("NOT SHOWN", "8 vs 4, p = 0.19")),
        ]
        vrows = {}
        for (cid, desc, sub, xv, lv), y in zip(claims, [1.55, 0.5, -0.55, -1.6]):
            idm = L(cid, size=26, color=WARN, weight="BOLD")
            dm = VGroup(L(desc, size=24), L(sub, size=20, color=MUTED)).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
            idm.move_to([CL, y, 0], aligned_edge=LEFT)
            dm.move_to([CL + 0.75, y, 0], aligned_edge=LEFT)
            lhs = VGroup(idm, dm)
            cells_ = []
            for (vt, detail), cx in ((xv, VX), (lv, VL)):
                b = verdict_badge(vt, size=20)
                d = L(detail, size=20, color=MUTED)
                cells_.append(VGroup(b, d).arrange(DOWN, buff=0.1).move_to([cx, y, 0]))
            rl = Line([CL, y - 0.52, 0], [6.5, y - 0.52, 0]).set_stroke(PANEL_EDGE, 1)
            vrows[cid] = VGroup(lhs, *cells_, rl)
        vrule = Line([CL, 2.05, 0], [6.5, 2.05, 0]).set_stroke(INK, 1.5)

        e8_head = L("L also tried eight streams (S = 8, k = 16):", size=22, color=INK)
        e8_ceil = frac_bar("perfect gate", 2, 2, color=GOOD, width=1.6, label_width=0, size=22)
        e8_rest = frac_bar("restart arm", 0, 5, color=BAD, width=1.6, label_width=0, size=22)
        e8_rest.value.set_color(BAD)
        e8_bars = VGroup(e8_ceil, e8_rest).arrange(RIGHT, buff=0.8)
        e8_note = L("all 25 attempts at 0.06–0.09 at step 4800", size=20, color=MUTED)
        e8 = VGroup(e8_head, e8_bars, e8_note).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        e8.move_to([-0.3, -3.0, 0])
        src5 = source_note("Report §4, Table 2")

        with self.voiceover(
            "So the check works only with spare channels. The restart recipe was reliable on L and a majority on X, "
            "and spare channels making restarts work was shown on both machines. Without restarts, sixteen "
            "channels did not bind significantly more than four on either machine. L also tried eight streams: "
            "the perfect gate bound both its runs, and the restart arm none of five."
        ) as vo:
            self.play(FadeIn(vhead), ShowCreation(vrule), FadeIn(src5), run_time=0.6)
            self.play(FadeIn(vrows["R2"][0], shift=0.1 * RIGHT), ShowCreation(vrows["R2"][3]), run_time=0.6)
            self.play(FadeIn(vrows["R2"][1], scale=0.9), FadeIn(vrows["R2"][2], scale=0.9), run_time=0.7)
            vo.wait_until_sentence(1)
            self.play(FadeIn(vrows["R1"][0], shift=0.1 * RIGHT), ShowCreation(vrows["R1"][3]), run_time=0.6)
            at_phrase(vo, 1, "reliable on L")
            self.play(FadeIn(vrows["R1"][2], scale=0.9), run_time=0.5)
            at_phrase(vo, 1, "a majority")
            self.play(FadeIn(vrows["R1"][1], scale=0.9), run_time=0.5)
            at_phrase(vo, 1, "and spare channels")
            self.play(FadeIn(vrows["R3"][0], shift=0.1 * RIGHT), ShowCreation(vrows["R3"][3]), run_time=0.6)
            self.play(FadeIn(vrows["R3"][1], scale=0.9), FadeIn(vrows["R3"][2], scale=0.9), run_time=0.7)
            at_phrase(vo, 1, "shown on both")
            self.play(Indicate(vrows["R3"][1][0], color=GOOD), Indicate(vrows["R3"][2][0], color=GOOD), run_time=1.0)
            vo.wait_until_sentence(2)
            self.play(FadeIn(vrows["R4"][0], shift=0.1 * RIGHT), ShowCreation(vrows["R4"][3]), run_time=0.6)
            self.play(FadeIn(vrows["R4"][1], scale=0.9), FadeIn(vrows["R4"][2], scale=0.9), run_time=0.7)
            at_phrase(vo, 2, "significantly")
            r4_hl = Rectangle(width=13.15, height=0.95).set_fill(MUTED, 0.08).set_stroke(MUTED, 1.2, opacity=0.7)
            r4_hl.move_to([0.05, vrows["R4"][1].get_center()[1], 0])
            self.play(FadeIn(r4_hl), run_time=0.7)
            at_phrase(vo, 2, "on either")
            self.play(Indicate(vrows["R4"][1][1], color=INK), Indicate(vrows["R4"][2][1], color=INK), run_time=0.9)
            vo.wait_until_sentence(3)
            self.play(FadeIn(e8_head, shift=0.1 * UP), run_time=0.5)
            at_phrase(vo, 3, "the perfect gate")
            self.play(FadeIn(e8_ceil.name_mob), FadeIn(e8_ceil.track), *grow_bar(e8_ceil), run_time=0.8)
            at_phrase(vo, 3, "and the restart")
            self.play(FadeIn(e8_rest.name_mob), FadeIn(e8_rest.track), *grow_bar(e8_rest), FadeIn(e8_note),
                      run_time=0.8)
        self.wait(0.6)
        self.clear_all()
