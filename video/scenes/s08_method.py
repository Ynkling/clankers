"""Chapter 8 — how the work is checked (report §1 preamble, §3 "Statistics", §6, §15, Tables 1, 4, 6, 9, 10;
test_early_recipe.py docstring; facts_repo §2; data/slow_start.json, data/early_recipe.json)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
with open(os.path.join(DATA, "slow_start.json")) as fh:
    SLOW = json.load(fh)
with open(os.path.join(DATA, "early_recipe.json")) as fh:
    EARLY = json.load(fh)

# the 34 test files at the repository root (ls test_*.py)
TEST_FILES = [
    "test_arm2_readinit.py", "test_asymmetric_routing.py", "test_binding_capacity.py", "test_binding_onset.py",
    "test_binding_recipe.py", "test_bounded_capacity.py", "test_capacity_control.py", "test_channel_binding.py",
    "test_conv_lr.py", "test_cross_channel.py", "test_curriculum_confirm.py", "test_early_recipe.py",
    "test_grow_inference.py", "test_instrument_v2.py", "test_interference.py", "test_interference_auxloss.py",
    "test_interference_discrete.py", "test_load_curriculum.py", "test_multilayer_binding.py", "test_p_scaling.py",
    "test_phase2_seeds.py", "test_readout_path.py", "test_recipe_scope.py", "test_router_confirm.py",
    "test_router_curriculum.py", "test_router_discovery.py", "test_router_layout.py", "test_router_reliability.py",
    "test_scale_axes.py", "test_short_conv.py", "test_slow_start.py", "test_stream_channels.py",
    "test_stream_curriculum.py", "test_stream_recipe.py",
]

# test_early_recipe.py docstring: (line, section heading, a short gloss of what the section holds)
DOC_ROWS = [
    ("14", "BACKGROUND", "screens S34, S35, S32"),
    ("25", "OPTIMIZERS", "ADAM, MUON, WINDOW"),
    ("48", "ARMS", "seeds 300–339, 340–359"),
    ("61", "CLAIMS", "E1, E2, bands, readings"),
    ("71", "DIAGNOSTICS", "not part of the verdict"),
    ("80", "CHECKS", "120–128, after the chain"),
    ("115", "RUNTIME", "projection, cut rule"),
    ("123", "OUTPUT", "early_recipe_results.json"),
]

# test_early_recipe.py lines 61-69, wrapped to fit (text unchanged)
CLAIMS_TEXT = """CLAIMS (per machine; exact McNemar one-sided, first arm
higher; SHOWN if p < 0.05)
- E1: WIN_M beats SLOW_M (DISCOVERED).
- E2: WIN16_M is bound by update 4800 more often than WIN16_A
  (transition <= 4800).
- Bands (RELIABLE >= 90%, MAJORITY >= 50%, MINORITY >= 1 run,
  NEVER 0) for WIN_M, WIN16_A and WIN16_M.
- Readings:
  - E1 SHOWN: "the early hinge window works under Muon";
  - E2 SHOWN: "Muon binds four streams sooner".
- Printed, not claims: WIN16_M vs WIN16_A BOUND (McNemar);
  median transitions."""

# test_early_recipe.py line 115 and the first sentences of line 119
RUNTIME_TEXT = """RUNTIME
- If over 10 h, Part B is cut to seeds 340-351 (WIN16_A and
  WIN16_M). Part A is never cut."""

# what report() printed on X (test_early_recipe.py:959-966 with X's records), wrapped
TERMINAL_TEXT = """E1  WIN_M beats SLOW_M (DISCOVERED): 40/40 vs 36/40;
    WIN_M only 4, SLOW_M only 0; p = 0.0625
     *** E1: NOT SHOWN ***
E2  WIN16_M bound by update 4800 more often than
    WIN16_A: 10/12 vs 6/12; WIN16_M only 5,
    WIN16_A only 1; p = 0.1094
     *** E2: NOT SHOWN ***"""

# the verify() chain a run of test_early_recipe executes: (owner test, first CHECK, last CHECK, phase)
CHAIN = [
    ("multilayer_binding", 1, 6, 3), ("binding_onset", 7, 10, 3), ("channel_binding", 15, 18, 3),
    ("router_discovery", 19, 25, 3), ("router_curriculum", 26, 31, 3), ("router_confirm", 32, 36, 3),
    ("readout_path", 37, 41, 4), ("router_reliability", 42, 47, 4), ("router_layout", 48, 52, 4),
    ("short_conv", 53, 57, 4), ("conv_lr", 58, 63, 4), ("p_scaling", 64, 68, 4),
    ("load_curriculum", 69, 74, 4), ("curriculum_confirm", 75, 79, 4), ("scale_axes", 80, 84, 4),
    ("stream_channels", 85, 89, 4), ("stream_recipe", 90, 94, 5), ("stream_curriculum", 95, 100, 5),
    ("slow_start", 101, 112, 5), ("recipe_scope", 113, 119, 5), ("early_recipe", 120, 128, 5),
]


def at(vo, i, phrase):
    """Approximate time (seconds into the block) at which `phrase` is spoken in sentence i."""
    text, a, b = vo.synth.sentences[i]
    k = text.find(phrase)
    if k < 0:
        return a
    return a + (b - a) * k / max(len(text), 1)


def mono(s, size=20, color=INK):
    return Text(s, font=MONO, font_size=size, fill_color=color)


class MonoBlock(VGroup):
    """Plain monospaced text on a code panel. Lines keep their indentation, and `span(i, sub)` returns the
    glyphs of `sub` in line i (Code's markup escaping makes its own substring lookup unreliable)."""

    def __init__(self, text, title=None, size=20, color=INK):
        super().__init__()
        self.src = text.split("\n")
        one = Text("M", font=MONO, font_size=size)
        many = Text("M" * 40, font=MONO, font_size=size)
        adv = (many.get_width() - one.get_width()) / 39
        lh = adv * 2.45
        self.lines = VGroup()
        for i, ln in enumerate(self.src):
            t = Text("\u2502" + ln, font=MONO, font_size=size, fill_color=color)
            t.shift(np.array([-adv / 2, -i * lh, 0]) - t[0].get_center())
            t.remove(t[0])
            self.lines.add(t)
        self.panel = RoundedRectangle(width=self.lines.get_width() + 0.5, height=len(self.src) * lh + 0.3,
                                      corner_radius=0.12).set_fill("#15181F", 1).set_stroke(PANEL_EDGE, 1.5)
        self.panel.move_to(self.lines)
        self.panel.shift((self.lines.get_left()[0] - 0.25 - self.panel.get_left()[0]) * RIGHT)
        self.add(self.panel, self.lines)
        if title:
            self.tab = L(title, size=20, color=MUTED)
            self.tab.next_to(self.panel, UP, buff=0.1, aligned_edge=LEFT).shift(0.1 * RIGHT)
            self.add(self.tab)

    def span(self, i, sub, nth=0):
        ln = self.src[i]
        k = -1
        for _ in range(nth + 1):
            k = ln.index(sub, k + 1)
        a = len("".join(ln[:k].split()))
        b = a + len("".join(sub.split()))
        return VGroup(*self.lines[i][a:b])


def check_mark(ok, size=26):
    return L("✓" if ok else "✗", size=size, color=GOOD if ok else BAD, weight="BOLD")


class DocCard(VGroup):
    """The docstring of test_early_recipe.py as a card: line numbers, section headings, glosses."""
    RH = 0.44

    def __init__(self):
        super().__init__()
        heads = [L(h, size=22, weight="BOLD") for _, h, _ in DOC_ROWS]
        self.gloss_x = 0.75 + max(h.get_width() for h in heads) + 0.3
        self.rows = VGroup()
        self.rows.add(self._row("2", mono('"""', 22, MUTED), None))
        self.rows.add(self._row("5", T("Everything below is fixed before any run.", size=22, color=WARN), None))
        for (num, _, gloss), head in zip(DOC_ROWS, heads):
            self.rows.add(self._row(num, head, L(gloss, size=20, color=MUTED)))
        self.close = self._row("0", mono('"""', 22, MUTED), None)
        self.close.num.set_opacity(0)
        self.n = len(self.rows)
        self._place(self.close, self.n)
        self.width_ = 0
        self.rect = self._rect(self.n + 1)
        self.add(self.rect, self.rows, self.close)

    def _row(self, num, head, gloss):
        n = mono(num, 20, FAINT)
        n.move_to([0.55, 0, 0], aligned_edge=RIGHT)
        head.move_to([0.75, 0, 0], aligned_edge=LEFT)
        g = VGroup(n, head)
        if gloss is not None:
            gloss.move_to([self.gloss_x, 0, 0], aligned_edge=LEFT)
            g.add(gloss)
        g.num, g.head, g.gloss = n, head, gloss
        return g

    def _place(self, row, i):
        row.shift((-i * self.RH - row.num.get_center()[1]) * UP)

    def build_rows(self):
        for i, r in enumerate(self.rows):
            self._place(r, i)

    def _rect(self, n_rows, width=None):
        w = width or 5.55
        h = n_rows * self.RH + 0.2
        rect = Rectangle(width=w, height=h).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1.5)
        rect.round_corners(0.12)
        rect.move_to([w / 2 - 0.15, -(n_rows - 1) * self.RH / 2, 0])
        return rect

    def row_box(self, i, color=WARN):
        r = self.rows[i]
        box = Rectangle(width=self.rect.get_width() - 0.16, height=self.RH * 0.9)
        box.set_fill(color, 0.13).set_stroke(color, 1.2, opacity=0.7)
        box.move_to([self.rect.get_center()[0], r.num.get_center()[1], 0])
        return box


class Method(ClankersScene):
    def construct(self):
        card0 = self.chapter_card(8, "How the work is checked")
        self.wait(0.6)
        title = section_title("One experiment, one file")
        self.play(FadeOut(card0, shift=0.3 * UP), FadeIn(title, shift=0.3 * UP))
        self.title = title
        self.src = None

        doc = self.part_one()
        self.part_two(doc)
        self.part_three()
        self.part_four()
        self.wait(0.8)
        self.clear_all()

    # ------------------------------------------------------------------ helpers
    def retitle(self, text):
        new = section_title(text)
        anims = [LaggedStart(FadeOut(self.title, shift=0.2 * UP), FadeIn(new, shift=0.2 * UP), lag_ratio=0.75)]
        self.title = new
        return anims

    def source(self, text):
        new = source_note(text)
        if self.src is None:
            anims = [FadeIn(new)]
        else:
            anims = [LaggedStart(FadeOut(self.src), FadeIn(new), lag_ratio=0.75)]
        self.src = new
        return anims

    # ------------------------------------------------------------------ N8.1 one file per experiment
    def part_one(self):
        names = VGroup(*[L(n, size=20, t2c={"test_": MUTED, ".py": MUTED}) for n in TEST_FILES])
        cols = [names[0:12], names[12:23], names[23:34]]
        for c, col in enumerate(cols):
            for r, m in enumerate(col):
                m.move_to([-6.2 + c * 4.35, 2.3 - r * 0.42, 0], aligned_edge=LEFT)
        count = L("34 test files, 27,952 lines", size=22, color=MUTED)
        count.move_to([2.6, 3.05, 0])
        template = VGroup(mono("test_", 36, GATE_COLOR), mono("<name>", 36, WARN), mono(".py", 36, GATE_COLOR))
        template.arrange(RIGHT, buff=0.04).move_to(3.05 * DOWN)
        for part in template:
            part.align_to(template[0], DOWN)
        template[1].shift(0.02 * UP)
        er = names[TEST_FILES.index("test_early_recipe.py")]

        doc = DocCard()
        doc.build_rows()
        doc.move_to(ORIGIN)
        doc.shift(np.array([-6.6, 2.35, 0]) - np.array([doc.rect.get_left()[0], doc.rect.get_top()[1], 0]))
        tab = L("test_early_recipe.py", size=20, color=INK)
        tab.next_to(doc.rect, UP, buff=0.1, aligned_edge=LEFT).shift(0.12 * RIGHT)
        doc.tab = tab

        claims = MonoBlock(CLAIMS_TEXT, "test_early_recipe.py, lines 61–69")
        claims.span(0, "CLAIMS").set_color(GATE_COLOR)
        claims.move_to([2.85, 0, 0]).align_to(doc.rect, UP).shift(0.05 * DOWN)
        runtime = MonoBlock(RUNTIME_TEXT, "lines 115 and 119 (excerpt)")
        runtime.span(0, "RUNTIME").set_color(GATE_COLOR)
        runtime.next_to(claims, DOWN, buff=0.42, aligned_edge=LEFT)
        thr_box = SurroundingRectangle(claims.span(1, "SHOWN if p < 0.05"), buff=0.06).set_stroke(WARN, 2.5)

        with self.voiceover(
            "Before the new results, a word on how they were produced, because the repository is organized "
            "around it. Every confirmatory experiment is one file: test underscore name dot py. Its docstring is "
            "written before any run: the background, the arms, the seeds, the claims and their thresholds, the "
            "checks, and a runtime projection with rules for what to cut if time runs short."
        ) as vo:
            self.play(LaggedStartMap(FadeIn, names, shift=0.1 * DOWN, lag_ratio=0.04), run_time=2.4)
            self.play(FadeIn(count), *self.source("repository root: ls test_*.py; wc -l"))
            # sentence 1: one file per experiment
            vo.wait_until_sentence(1)
            self.play(Write(template), run_time=1.0)
            self.play(*[n["test_"].animate.set_color(GATE_COLOR) for n in names],
                      *[n[".py"].animate.set_color(GATE_COLOR) for n in names], run_time=0.8)
            sel = SurroundingRectangle(er, buff=0.08).set_stroke(WARN, 2.5)
            self.play(ShowCreation(sel), *[n.animate.set_opacity(0.25) for n in names if n is not er],
                      run_time=0.7)
            vo.wait_until(vo.time_of(2) - 1.3)
            others = VGroup(*[n for n in names if n is not er])
            self.play(FadeOut(others), FadeOut(count), FadeOut(template), FadeOut(sel),
                      ReplacementTransform(er, tab), FadeIn(doc.rect),
                      *self.source("test_early_recipe.py, docstring (lines 2–125)"), run_time=1.0)
            self.play(LaggedStartMap(FadeIn, VGroup(*doc.rows, doc.close), shift=0.15 * RIGHT, lag_ratio=0.08),
                      run_time=1.2)
            # sentence 2: the sections, named one by one
            hl = doc.row_box(1)
            self.play(FadeIn(hl), Indicate(doc.rows[1].head, color=WARN, scale_factor=1.05), run_time=0.8)
            vo.wait_until(at(vo, 2, "the background"))
            self.play(Transform(hl, doc.row_box(2)), run_time=0.45)
            vo.wait_until(at(vo, 2, "the arms"))
            self.play(Transform(hl, doc.row_box(4)), run_time=0.45)
            vo.wait_until(at(vo, 2, "the seeds"))
            self.play(Indicate(doc.rows[4].gloss, color=WARN, scale_factor=1.12), run_time=0.7)
            vo.wait_until(at(vo, 2, "the claims") - 0.2)
            self.play(Transform(hl, doc.row_box(5)), run_time=0.4)
            self.play(TransformFromCopy(VGroup(doc.rows[5].head, doc.rows[5].gloss), claims), run_time=1.0)
            vo.wait_until(at(vo, 2, "thresholds"))
            self.play(ShowCreation(thr_box), run_time=0.6)
            vo.wait_until(at(vo, 2, "the checks"))
            self.play(Transform(hl, doc.row_box(7)), run_time=0.45)
            vo.wait_until(at(vo, 2, "a runtime projection"))
            self.play(Transform(hl, doc.row_box(8)), run_time=0.45)
            self.play(TransformFromCopy(doc.rows[8].head, runtime), run_time=0.9)
            vo.wait_until(at(vo, 2, "what to cut"))
            self.play(Indicate(runtime.span(1, "Part B is cut to seeds 340-351"), color=WARN, scale_factor=1.04),
                      run_time=1.0)
        # keep the claims on screen a moment longer for reading, with the bands line marked
        bands_box = SurroundingRectangle(VGroup(claims.lines[5], claims.lines[6]), buff=0.06).set_stroke(GOOD, 2)
        self.play(ShowCreation(bands_box), run_time=0.7)
        self.wait(0.6)
        doc.hl = hl
        doc.extra = VGroup(claims, runtime, thr_box, bands_box)
        return doc

    # ------------------------------------------------------------------ N8.2 checks, then a verdict
    def part_two(self, doc):
        x0, x1 = -0.95, 6.55
        names = ["CHECKs", "projection", "training", "verdict"]
        nodes = VGroup()
        for nm in names:
            box = RoundedRectangle(width=1.6, height=0.62, corner_radius=0.12)
            box.set_fill(PANEL, 1).set_stroke(MUTED, 1.5)
            lab = L(nm, size=22)
            nodes.add(VGroup(box, lab.move_to(box)))
        nodes.arrange(RIGHT, buff=0.36)
        nodes.move_to([(x0 + x1) / 2, 2.25, 0])
        arrows = VGroup(*[Arrow(nodes[i].get_right(), nodes[i + 1].get_left(), buff=0.04, thickness=2.5)
                          .set_color(MUTED) for i in range(3)])
        nodes[3][0].set_stroke(WARN, 2)
        nodes[3][1].set_color(WARN)

        term = MonoBlock(TERMINAL_TEXT, "printed by the test on X")
        term.move_to([(x0 + x1) / 2, 0, 0]).align_to(nodes, LEFT)
        term.shift((1.45 - term.panel.get_top()[1]) * UP)

        # the RESULT row, appended to the docstring
        res_row = doc._row("127", L("RESULT", size=22, weight="BOLD", color=GOOD),
                           L("X: E1, E2 NOT SHOWN", size=20, color=GOOD))
        res_row.shift(doc.close.get_center()[1] * UP - res_row.num.get_center()[1] * UP)
        res_row.shift((doc.rows[0].num.get_right()[0] - res_row.num.get_right()[0]) * RIGHT)
        new_rect = Rectangle(width=doc.rect.get_width(), height=doc.rect.get_height() + doc.RH)
        new_rect.set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1.5).round_corners(0.12)
        new_rect.move_to(doc.rect.get_center() + doc.RH / 2 * DOWN)

        # the CHECK chain as a bar, one cell per CHECK, grouped by the test that owns it
        n_checks = sum(b - a + 1 for _, a, b, _ in CHAIN)
        bar_l, bar_r, bar_y = -0.55, 6.05, 0.45
        u = (bar_r - bar_l - 0.07 * (len(CHAIN) - 1)) / n_checks
        cells, outlines, checks, owner_of, seg_of = VGroup(), VGroup(), [], {}, {}
        x = bar_l
        for k, (owner, a, b, ph) in enumerate(CHAIN):
            seg = VGroup()
            for c in range(a, b + 1):
                r = Rectangle(width=u * 1.01, height=0.36).set_fill(GOOD, 0.9).set_stroke(width=0)
                r.move_to([x + u * (c - a) + u / 2, bar_y, 0])
                r.set_opacity(0)
                seg.add(r)
                checks.append(c)
                owner_of[c] = owner
                seg_of[c] = r
            w = u * (b - a + 1)
            out = Rectangle(width=w, height=0.46).set_fill(PANEL, 1).set_stroke(FAINT, 1.2)
            out.move_to([x + w / 2, bar_y, 0])
            outlines.add(out)
            cells.add(seg)
            x += w + 0.07
        phase_braces = VGroup()
        for ph in (3, 4, 5):
            idx = [k for k, c in enumerate(CHAIN) if c[3] == ph]
            grp = VGroup(*[outlines[k] for k in idx])
            br = Brace(grp, DOWN, buff=0.08).set_color(FAINT)
            lab = L(f"Phase {'I' * ph if ph < 4 else ('IV' if ph == 4 else 'V')}", size=20, color=MUTED)
            lab.next_to(br, DOWN, buff=0.06)
            phase_braces.add(VGroup(br, lab))

        cache = {}

        def counter_mob(c):
            if c not in cache:
                g = VGroup(L("CHECK", size=22, color=MUTED), L(str(c), size=28, weight="BOLD"),
                           L(f"test_{owner_of[c]}.py", size=20, color=MUTED))
                g.arrange(RIGHT, buff=0.18)
                g[2].align_to(g[0], DOWN)
                g.move_to([bar_l, bar_y + 0.68, 0], aligned_edge=LEFT)
                cache[c] = g
            return cache[c]

        counter = counter_mob(1).copy()

        def tick(mob, alpha):
            k = max(1, min(n_checks, int(round(alpha * n_checks))))
            for j, c in enumerate(checks):
                seg_of[c].set_opacity(1 if j < k else 0)
            counter.become(counter_mob(checks[k - 1]))

        passed = L("ALL VERIFICATION CHECKS PASSED", size=22, color=GOOD, weight="BOLD")
        passed.move_to([(bar_l + bar_r) / 2, -0.75, 0])
        inh = VGroup(*outlines[:20])
        own = outlines[20]
        br_inh = Brace(inh, UP, buff=0.08).set_color(MUTED)
        lab_inh = L("inherited from 20 earlier tests: CHECKs 1–10, 15–119", size=20, color=INK)
        lab_inh.next_to(br_inh, UP, buff=0.06)
        br_own = Brace(own, UP, buff=0.08).set_color(GATE_COLOR)
        lab_own = L("own: 120–128", size=20, color=GATE_COLOR)
        lab_own.next_to(br_own, UP, buff=0.06).align_to(br_own, RIGHT).shift(0.15 * RIGHT)

        # CHECK 121: a fresh run reproduces the recorded HINGE0 run of seed 280 bit for bit (through 2400)
        rec = SLOW["curves_X"]["HINGE0|280"][:2]
        c121 = seg_of[121]
        c121_box = SurroundingRectangle(c121, buff=0.06).set_stroke(WARN, 2.5)
        head121 = VGroup(L("CHECK 121", size=24, weight="BOLD", color=WARN),
                         L("a fresh run vs. the recorded HINGE0 run, seed 280, on X", size=20, color=MUTED))
        head121.arrange(RIGHT, buff=0.25)
        grid = VGroup()
        rows121 = [("held-out accuracy at", "step 1200", "step 2400"),
                   ("recorded (slow_start)", f"{rec[0]:.4f}", f"{rec[1]:.4f}"),
                   ("re-run (CHECK 121)", f"{rec[0]:.4f}", f"{rec[1]:.4f}")]
        for r, (a, b, c) in enumerate(rows121):
            col = MUTED if r == 0 else INK
            ra = L(a, size=20, color=col)
            rb = (L(b, size=20, color=col) if r == 0 else mono(b, 22))
            rc = (L(c, size=20, color=col) if r == 0 else mono(c, 22))
            ra.move_to([-0.55, -1.95 - r * 0.42, 0], aligned_edge=LEFT)
            rb.move_to([2.65, -1.95 - r * 0.42, 0])
            rc.move_to([4.05, -1.95 - r * 0.42, 0])
            grid.add(VGroup(ra, rb, rc))
        head121.move_to([-0.55, -1.42, 0], aligned_edge=LEFT)
        same = VGroup(check_mark(True, 30), L("bit for bit", size=22, color=GOOD, weight="BOLD"))
        same.arrange(RIGHT, buff=0.12).next_to(grid[2], RIGHT, buff=0.35)

        with self.voiceover(
            "The test prints its verdicts mechanically, and after the run the result is appended to the same "
            "docstring. Before training, each test runs a chain of numbered checks inherited from earlier tests, "
            "more than a hundred by now, including reproducing a recorded run bit for bit."
        ) as vo:
            self.play(*self.retitle("Checks, then a verdict"), FadeOut(doc.extra), FadeOut(doc.hl),
                      LaggedStartMap(FadeIn, nodes, shift=0.15 * RIGHT, lag_ratio=0.2),
                      LaggedStartMap(GrowArrow, arrows, lag_ratio=0.3), run_time=1.1)
            self.play(LaggedStart(*[FlashAround(n, color=WARN, time_width=0.8, buff=0.06) for n in nodes],
                                  lag_ratio=0.35),
                      FadeIn(term, shift=0.2 * DOWN),
                      *self.source("test_early_recipe.py:959-966 (report); RESULT (X), lines 127-158"), run_time=1.1)
            nots = VGroup(term.span(2, "NOT SHOWN"), term.span(6, "NOT SHOWN"))
            self.play(*[FlashAround(m, color=WARN, time_width=0.6) for m in nots], run_time=0.8)
            vo.wait_until(at(vo, 0, "after the run"))
            self.play(Transform(doc.rect, new_rect), doc.close.animate.shift(doc.RH * DOWN), run_time=0.6)
            self.play(FadeIn(VGroup(res_row.num, res_row.head)),
                      TransformFromCopy(nots, res_row.gloss), run_time=1.2)
            self.play(Indicate(res_row, color=GOOD, scale_factor=1.04), run_time=0.7)
            # sentence 1: the checks before training
            vo.wait_until_sentence(1)
            self.play(nodes[0][0].animate.set_stroke(GOOD, 2.5), nodes[0][1].animate.set_color(GOOD),
                      FadeOut(term), FadeIn(outlines), FadeIn(counter), FadeIn(cells),
                      *self.source("test_early_recipe.py:660-663: verify() runs the earlier tests' verify() chain"),
                      run_time=0.9)
            self.play(UpdateFromAlphaFunc(VGroup(cells, counter), tick), FadeIn(phase_braces, lag_ratio=0.3),
                      run_time=4.4, rate_func=linear)
            self.play(FadeIn(passed, shift=0.1 * UP), FadeOut(counter, shift=0.1 * UP), run_time=0.5)
            self.play(GrowFromCenter(br_inh), FadeIn(lab_inh), GrowFromCenter(br_own),
                      FadeIn(lab_own), Transform(doc.hl, doc.row_box(7)), run_time=0.9)
            vo.wait_until(at(vo, 1, "including reproducing"))
            self.play(ShowCreation(c121_box), FadeIn(head121, shift=0.1 * DOWN),
                      *self.source("test_early_recipe.py:83-85 (CHECK 121); results/X/slow_start_results.json (HINGE0|280)"), run_time=0.9)
            self.play(FadeIn(grid[0]), FadeIn(grid[1]), run_time=0.6)
            self.play(TransformFromCopy(grid[1], grid[2]), run_time=0.8)
            vo.wait_until(at(vo, 1, "bit for bit") - 0.3)
            self.play(FadeIn(same, shift=0.15 * RIGHT), Indicate(VGroup(grid[1][1:], grid[2][1:]), color=GOOD,
                                                                      scale_factor=1.06), run_time=0.9)
        self.wait(0.4)
        self.play(FadeOut(VGroup(doc, doc.tab, doc.hl, res_row, nodes, arrows, outlines, cells, phase_braces,
                                 passed, br_inh, lab_inh, br_own, lab_own, c121_box, head121, grid, same)),
                  run_time=0.8)

    # ------------------------------------------------------------------ N8.3 the machines
    def part_three(self):
        def machine(name, line1, line2):
            b = machine_badge(name, size=30)
            t = VGroup(L(line1, size=22), L(line2, size=20, color=MUTED)).arrange(DOWN, buff=0.1,
                                                                                 aligned_edge=LEFT)
            t.next_to(b, RIGHT, buff=0.25)
            g = VGroup(b, t)
            g.badge = b
            return g

        mx = machine("X", "a cloud container", "Intel Xeon, 2.10 / 2.80 GHz")
        ml = machine("L", "an Intel laptop processor", "i7-12650H")
        mx.move_to([-6.5, 1.95, 0], aligned_edge=LEFT)
        ml.move_to([-2.65, 1.95, 0], aligned_edge=LEFT)
        chip_txt = VGroup(L("every Phase IV and Phase V test", size=22),
                          L("10 + 5 pre-registered test files", size=20, color=MUTED)).arrange(DOWN, buff=0.08)
        chip = card(chip_txt, buff=0.18)
        chip.move_to([-2.65, 0.55, 0])
        a_x = Arrow(chip.get_top(), mx.get_bottom() + 0.6 * RIGHT, buff=0.1, thickness=2.5).set_color(MUTED)
        a_l = Arrow(chip.get_top(), ml.get_bottom() + 0.2 * LEFT, buff=0.1, thickness=2.5).set_color(MUTED)

        def verdict(machine_name, word, stat):
            row = VGroup(L("E1", size=22, weight="BOLD"), verdict_badge(word, size=20)).arrange(RIGHT, buff=0.2)
            g = VGroup(row, L(stat, size=20, color=MUTED)).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
            return g

        vx = verdict("X", "NOT SHOWN", "4 vs 0, p = 0.0625")
        vl = verdict("L", "SHOWN", "6 vs 0, p = 0.016")
        vx.next_to(mx, DOWN, buff=0.35).align_to(mx[1], LEFT)
        vl.next_to(ml, DOWN, buff=0.35).align_to(ml[1], LEFT)
        vtag = L("E1 = claim 1 of test_early_recipe.py, judged per machine", size=20, color=MUTED)
        vtag.next_to(VGroup(vx, vl), DOWN, buff=0.2)

        # the same seeds on both machines: SLOW_M outcomes per seed (data/early_recipe.json)
        seeds = EARLY["per_seed"]["X"]["SLOW_M"]["seeds"]
        out_x = [o == "DISCOVERED" for o in EARLY["per_seed"]["X"]["SLOW_M"]["outcome"]]
        out_l = [o == "DISCOVERED" for o in EARLY["per_seed"]["L"]["SLOW_M"]["outcome"]]
        x_left, x_right = -4.55, 1.35
        xs = [x_left + i * (x_right - x_left) / (len(seeds) - 1) for i in range(len(seeds))]
        y_seed, y_x, y_l = -0.45, -1.0, -1.45

        def dots(y, colors, fill=0.95):
            return VGroup(*[Dot(radius=0.065).move_to([x, y, 0]).set_fill(c, fill) for x, c in zip(xs, colors)])

        seed_dots = VGroup(*[Circle(radius=0.065).move_to([x, y_seed, 0]).set_stroke(INK, 1.2) for x in xs])
        dx = dots(y_x, [GOOD if ok else BAD for ok in out_x])
        dl = dots(y_l, [GOOD if ok else BAD for ok in out_l])
        lab_seed = L("seeds 300–339", size=20, color=INK).next_to(seed_dots, LEFT, buff=0.25)
        lab_x = L("SLOW_M on X", size=20, color=MACHINE_COLORS["X"]).next_to(dx, LEFT, buff=0.25)
        lab_l = L("SLOW_M on L", size=20, color=MACHINE_COLORS["L"]).next_to(dl, LEFT, buff=0.25)
        for lab in (lab_seed, lab_x, lab_l):
            lab.align_to(lab_seed, RIGHT)
        diff = [i for i in range(len(seeds)) if out_x[i] != out_l[i]]
        same = [i for i in range(len(seeds)) if not out_x[i] and not out_l[i]]
        same_cols = VGroup(*[RoundedRectangle(width=0.16, height=0.72, corner_radius=0.06)
                             .move_to([xs[i], (y_x + y_l) / 2, 0]).set_stroke(BAD, 1.5) for i in same])
        diff_cols = VGroup(*[RoundedRectangle(width=0.16, height=0.72, corner_radius=0.06)
                             .move_to([xs[i], (y_x + y_l) / 2, 0]).set_stroke(WARN, 1.5) for i in diff])
        assert len(diff) == 2 and len(same) == 4

        def key(color, text):
            box = RoundedRectangle(width=0.14, height=0.36, corner_radius=0.05).set_stroke(color, 1.5)
            return VGroup(box, L(text, size=20, color=color)).arrange(RIGHT, buff=0.1)

        legend = VGroup(key(BAD, f"failed on both: {len(same)}"), key(WARN, f"differ: {len(diff)}"))
        legend.arrange(RIGHT, buff=0.35)
        agree = L(f"same outcome on {len(seeds) - len(diff)} of {len(seeds)} seeds: not independent",
                  size=20, color=INK)
        row = VGroup(legend, agree).arrange(RIGHT, buff=0.5)
        row.move_to([0, -1.95, 0]).align_to(lab_seed, LEFT)
        pooled = VGroup(L("pooled, X + L:", size=22, color=MUTED),
                        L("WIN_M 80/80 vs SLOW_M 70/80", size=22),
                        verdict_badge("descriptive, no p-value", size=20, color=WARN)).arrange(RIGHT, buff=0.22)
        pooled.move_to([-2.55, -2.7, 0])
        if pooled.get_left()[0] < -6.5:
            pooled.shift((-6.5 - pooled.get_left()[0]) * RIGHT)

        # E: the exploratory container
        divider = DashedLine([1.95, 2.6, 0], [1.95, -3.3, 0], dash_length=0.12).set_stroke(FAINT, 1.5)
        me = machine("E", "exploratory screens", "Intel Xeon")
        me.move_to([2.25, 1.95, 0], aligned_edge=LEFT)
        branch = L("branch claude/outside-ideas", size=20, color=GATE_COLOR)
        branch.next_to(me, DOWN, buff=0.18).align_to(me[1], LEFT)
        ex_x = 4.35
        scr = card(VGroup(L("screen S34: an early hinge window", size=20),
                          L("Muon 38/40, Adam 34/40", size=20, color=MUTED)).arrange(DOWN, buff=0.08), buff=0.16)
        scr.move_to([ex_x, 0.35, 0])
        stamp = L("“exploratory, not a result”", size=20, color=BAD).next_to(scr, DOWN, buff=0.14)
        pre = card(VGroup(L("pre-registered test", size=22, color=GOOD),
                          L("test_early_recipe.py on X and L", size=20, color=MUTED)).arrange(DOWN, buff=0.08),
                   buff=0.16)
        pre.move_to([ex_x, -1.65, 0])
        a1 = Arrow(stamp.get_bottom(), pre.get_top(), buff=0.08, thickness=2.5).set_color(GOOD)
        res = VGroup(L("verdict:", size=22, color=GOOD), L("E1 SHOWN on L, NOT SHOWN on X", size=20))
        res.arrange(RIGHT, buff=0.15).set_max_width(4.4)
        res.move_to([ex_x, -2.75, 0])
        a2 = Arrow(pre.get_bottom(), res.get_top(), buff=0.08, thickness=2.5).set_color(GOOD)

        with self.voiceover(
            "Two machines run every Phase Four and Phase Five test: X, a cloud container, and L, an Intel laptop "
            "processor. Every verdict is given per machine. They run the same seeds, so pooled counts are "
            "descriptive, not independent evidence. A third container, E, runs quick exploratory screens on a "
            "separate branch, and nothing there counts as a result until a pre-registered test confirms it."
        ) as vo:
            self.play(*self.retitle("Machines X, L and E"),
                      *self.source("Report §1 and Table 9; Table 1 (Phase V); README Table 1 (Phase IV)"),
                      run_time=0.9)
            vo.wait_until(at(vo, 0, "every Phase") - 0.2)
            self.play(FadeIn(chip, shift=0.2 * UP), run_time=0.8)
            vo.wait_until(at(vo, 0, "X, a cloud"))
            self.play(FadeIn(mx, shift=0.2 * RIGHT), GrowArrow(a_x), run_time=0.8)
            vo.wait_until(at(vo, 0, "L, an Intel"))
            self.play(FadeIn(ml, shift=0.2 * RIGHT), GrowArrow(a_l), run_time=0.8)
            # sentence 1: verdicts per machine
            vo.wait_until_sentence(1)
            self.play(FadeOut(VGroup(a_x, a_l)), ReplacementTransform(chip, vx), TransformFromCopy(chip, vl),
                      FadeIn(vtag), *self.source("Report Table 6; test_early_recipe.py RESULT (X, L)"),
                      run_time=1.0)
            # sentence 2: the same seeds
            vo.wait_until_sentence(2)
            self.play(LaggedStartMap(FadeIn, seed_dots, lag_ratio=0.02), FadeIn(lab_seed), run_time=0.7)
            self.play(TransformFromCopy(seed_dots, dx), TransformFromCopy(seed_dots, dl), FadeIn(lab_x),
                      FadeIn(lab_l), *self.source("data/early_recipe.json (per seed); test_early_recipe.py:190-210"),
                      run_time=1.0)
            vo.wait_until(at(vo, 2, "pooled") - 0.1)
            self.play(FadeIn(pooled[:2], shift=0.15 * UP), run_time=0.7)
            self.play(FadeIn(pooled[2], scale=1.2), run_time=0.6)
            self.play(LaggedStartMap(ShowCreation, same_cols, lag_ratio=0.15), FadeIn(legend[0]), run_time=0.9)
            self.play(FadeIn(diff_cols), FadeIn(legend[1]), run_time=0.5)
            self.play(FadeIn(agree, shift=0.1 * UP), run_time=0.6)
            # sentence 3: E
            vo.wait_until_sentence(3)
            self.play(ShowCreation(divider), FadeIn(me, shift=0.2 * LEFT),
                      *self.source("Report §6 and Table 10; test_early_recipe.py BACKGROUND"), run_time=0.9)
            vo.wait_until(at(vo, 3, "quick exploratory") - 0.2)
            self.play(FadeIn(scr, shift=0.15 * DOWN), run_time=0.7)
            self.play(Write(stamp), run_time=0.7)
            vo.wait_until(at(vo, 3, "on a separate branch"))
            self.play(FadeIn(branch, shift=0.1 * DOWN), run_time=0.6)
            vo.wait_until(at(vo, 3, "until a pre-registered") - 0.4)
            self.play(GrowArrow(a1), FadeIn(pre, shift=0.15 * DOWN), run_time=0.9)
            vo.wait_until(at(vo, 3, "confirms it") - 0.3)
            self.play(GrowArrow(a2), FadeIn(res, shift=0.1 * DOWN), run_time=0.8)
        self.wait(0.6)
        self.play(FadeOut(VGroup(mx, ml, vx, vl, vtag, seed_dots, dx, dl, lab_seed, lab_x, lab_l, same_cols,
                                 diff_cols, legend, agree, pooled, divider, me, branch, scr, stamp, pre, a1, res, a2)),
                  run_time=0.8)

    # ------------------------------------------------------------------ N8.4 paired seeds, McNemar, bands
    def part_four(self):
        seeds = SLOW["seeds"]["A0"]
        ok_h = SLOW["per_seed"]["X"]["HINGE0"]["success"]
        ok_a = SLOW["per_seed"]["X"]["A0"]["success"]
        assert SLOW["seeds"]["HINGE0"] == seeds

        # one seed fixes the initial weights and the batches
        chip = card(L("seed 280", size=26, weight="BOLD"), buff=0.2)
        chip.move_to([-5.55, 0.45, 0])
        rng = np.random.default_rng(280)
        weights = memory_grid(5, 5, cell=0.17, color=MEMORY_COLOR)
        for c in weights.cells:
            c.set_fill(MEMORY_COLOR, float(rng.uniform(0.05, 0.85)))
        cols = [STREAM_COLORS[0], KEY_COLOR, STREAM_COLORS[0], STREAM_COLORS[1], KEY_COLOR, STREAM_COLORS[1]]
        batches = VGroup(*[
            VGroup(*[Rectangle(width=0.13, height=0.18).set_fill(cols[(j + 3 * r) % 6], 0.85).set_stroke(width=0)
                     for j in range(9)]).arrange(RIGHT, buff=0.03)
            for r in range(3)]).arrange(DOWN, buff=0.05)
        weights.move_to([-3.25, 1.35, 0])
        batches.move_to([-3.25, -0.35, 0])
        w_lab = L("initial weights", size=20, color=MUTED).next_to(weights, DOWN, buff=0.12)
        b_lab = L("training batches", size=20, color=MUTED).next_to(batches, DOWN, buff=0.12)
        a_w = Arrow(chip.get_right(), weights.get_left(), buff=0.12, thickness=2.5).set_color(MUTED)
        a_b = Arrow(chip.get_right(), batches.get_left(), buff=0.12, thickness=2.5).set_color(MUTED)

        def arm(name, extra, color, y):
            nm = L(name, size=22, weight="BOLD").move_to([-0.3, y, 0], aligned_edge=RIGHT)
            w = weights.copy().scale(0.75).move_to([0.35, y, 0])
            b = batches.copy().scale(0.75).move_to([1.55, y, 0])
            box = card(L(extra, size=20, color=color), buff=0.12, edge=color)
            box.move_to([2.4, y, 0], aligned_edge=LEFT)
            g = VGroup(nm, w, b, box)
            g.nm, g.w, g.b, g.box = nm, w, b, box
            return g

        arm_h = arm("HINGE0", "+ slow memory, + hinge", HINGE_COLOR, 1.35)
        arm_a = arm("A0", "plain gate", MUTED, -0.55)
        diff_lab = L("the only difference", size=20, color=HINGE_COLOR).next_to(arm_h.box, UP, buff=0.12)
        out_h = check_mark(ok_h[0], 30).next_to(arm_h, RIGHT, buff=0.3)
        out_a = check_mark(ok_a[0], 30).next_to(arm_a, RIGHT, buff=0.3)
        out_a.set_x(out_h.get_x())

        # the paired strip: one column per seed, HINGE0 on top, A0 below
        x0, x1 = -3.25, 4.6
        xs = [x0 + i * (x1 - x0) / 19 for i in range(20)]
        y_h, y_a = 2.25, 1.8
        cols_ = VGroup()
        for i in range(20):
            th = Dot(radius=0.1).set_fill(GOOD if ok_h[i] else BAD, 1).move_to([xs[i], y_h, 0])
            ta = Dot(radius=0.1).set_fill(GOOD if ok_a[i] else BAD, 1).move_to([xs[i], y_a, 0])
            link = Line(th.get_center(), ta.get_center()).set_stroke(FAINT, 1.5)
            cols_.add(VGroup(link, th, ta))
        lab_h = L("HINGE0 (recipe)", size=20).move_to([x0 - 0.35, y_h, 0], aligned_edge=RIGHT)
        lab_a = L("A0 (plain gate)", size=20).move_to([x0 - 0.35, y_a, 0], aligned_edge=RIGHT)
        tot_h = L(f"{sum(ok_h)}/20", size=22, color=GOOD, weight="BOLD").move_to([x1 + 0.45, y_h, 0],
                                                                                  aligned_edge=LEFT)
        tot_a = L(f"{sum(ok_a)}/20", size=22, color=HINGE_COLOR, weight="BOLD").move_to([x1 + 0.45, y_a, 0],
                                                                                         aligned_edge=LEFT)
        s_lab = VGroup(L("seed 280", size=20, color=MUTED).next_to(cols_[0], DOWN, buff=0.1),
                       L("299", size=20, color=MUTED).next_to(cols_[-1], DOWN, buff=0.1))

        # the 2x2 McNemar table: rows HINGE0 succeeded / failed, columns A0 succeeded / failed
        cw, chh = 2.6, 1.3
        cx = [-1.85, 0.75]
        cy = [-0.5, -1.8]
        cells = [[Rectangle(width=cw, height=chh).set_stroke(PANEL_EDGE, 1.5).set_fill(PANEL, 1)
                  .move_to([cx[c], cy[r], 0]) for c in range(2)] for r in range(2)]
        cell_g = VGroup(*[cells[r][c] for r in range(2) for c in range(2)])

        def head(txt, ok):
            return VGroup(L(txt, size=20), check_mark(ok, 22)).arrange(RIGHT, buff=0.1)

        col_h = VGroup(head("A0", True).next_to(cells[0][0], UP, buff=0.12),
                       head("A0", False).next_to(cells[0][1], UP, buff=0.12))
        row_h = VGroup(head("HINGE0", True).next_to(cells[0][0], LEFT, buff=0.2),
                       head("HINGE0", False).next_to(cells[1][0], LEFT, buff=0.2))
        groups = {(r, c): [] for r in range(2) for c in range(2)}
        for i in range(20):
            groups[(0 if ok_h[i] else 1, 0 if ok_a[i] else 1)].append(i)
        targets = {}
        for (r, c), idx in groups.items():
            for j, i in enumerate(idx):
                col, rr = j % 6, j // 6
                p = cells[r][c].get_center() + np.array([-1.0 + col * 0.27, 0.25 - rr * 0.5, 0])
                targets[i] = p
        counts = {}
        for (r, c), idx in groups.items():
            n = L(str(len(idx)), size=30, weight="BOLD",
                  color=WARN if r != c else INK)
            n.move_to(cells[r][c].get_corner(DR) + np.array([-0.32, 0.3, 0]))
            counts[(r, c)] = n
        b, c_ = len(groups[(0, 1)]), len(groups[(1, 0)])
        glow = VGroup(cells[0][1].copy().set_fill(WARN, 0.12).set_stroke(WARN, 3),
                      cells[1][0].copy().set_fill(WARN, 0.12).set_stroke(WARN, 3))
        d_lab = VGroup(L("HINGE0 only", size=20, color=WARN).next_to(cells[0][1], RIGHT, buff=0.15),
                       L("A0 only", size=20, color=WARN).next_to(cells[1][0], DOWN, buff=0.12))
        d_lab[0].align_to(cells[0][1], UP).shift(0.1 * DOWN)
        disc = L("discordant seeds", size=22, color=WARN, weight="BOLD")
        disc.next_to(d_lab[0], DOWN, buff=0.12).align_to(d_lab[0], LEFT)

        big = VGroup(T(f"{b}", size=60, color=WARN), T("vs", size=36, color=MUTED), T(f"{c_}", size=60, color=WARN))
        big.arrange(RIGHT, buff=0.25).move_to([4.75, -1.15, 0])
        pv = L("exact McNemar, one-sided: p = 0.0005", size=20, color=MUTED).next_to(big, DOWN, buff=0.25)
        sh = VGroup(L("H0 on X", size=20), verdict_badge("SHOWN", size=20)).arrange(RIGHT, buff=0.15)
        sh.next_to(pv, DOWN, buff=0.18)
        assert (b, c_) == (11, 0)

        # the bands ruler
        rl, rr_, ry = -5.4, 5.4, -1.0

        def X(f):
            return rl + (rr_ - rl) * f

        ruler = Line([rl, ry, 0], [rr_, ry, 0]).set_stroke(MUTED, 2)
        ticks = VGroup()
        for f, t in [(0, "0%"), (0.5, "50%"), (0.9, "90%"), (1.0, "100%")]:
            tk = Line([X(f), ry - 0.1, 0], [X(f), ry + 0.1, 0]).set_stroke(MUTED, 2)
            lb = L(t, size=20, color=MUTED).next_to(tk, DOWN, buff=0.1)
            ticks.add(VGroup(tk, lb))

        def zone(f0, f1, color, text):
            bar = Rectangle(width=X(f1) - X(f0), height=0.28).set_fill(color, 0.75).set_stroke(width=0)
            bar.move_to([(X(f0) + X(f1)) / 2, ry, 0])
            lab = L(text, size=22, color=color, weight="BOLD").next_to(bar, UP, buff=0.28)
            return VGroup(bar, lab)

        z_rel = zone(0.9, 1.0, GOOD, "RELIABLE ≥ 90%")
        z_rel[1].align_to(z_rel[0], RIGHT).shift(0.25 * RIGHT)
        z_maj = zone(0.5, 0.9, WARN, "MAJORITY ≥ 50%")
        z_min = zone(0.025, 0.5, HINGE_COLOR, "MINORITY ≥ 1 run")
        z_nev = VGroup(Dot(radius=0.12).set_fill(BAD, 1).move_to([X(0), ry, 0]),
                       L("NEVER 0", size=22, color=BAD, weight="BOLD"))
        z_nev[1].next_to(z_nev[0], UP, buff=0.3)
        z_nev[1].align_to(z_nev[0], LEFT).shift(0.15 * LEFT)

        def marker(frac, text, band, color):
            tri = Triangle().scale(0.12).rotate(PI).set_fill(color, 1).set_stroke(width=0)
            tri.move_to([X(frac), ry - 0.62, 0])
            lab = VGroup(L(text, size=20, color=color), verdict_badge(band, size=20)).arrange(DOWN, buff=0.1)
            lab.next_to(tri, DOWN, buff=0.12)
            if lab.get_right()[0] > 6.5:
                lab.shift((6.5 - lab.get_right()[0]) * RIGHT)
            return VGroup(tri, lab)

        mk_h = marker(sum(ok_h) / 20, f"HINGE0 {sum(ok_h)}/20", "RELIABLE", GOOD)
        mk_a = marker(sum(ok_a) / 20, f"A0 {sum(ok_a)}/20", "MINORITY", HINGE_COLOR)

        with self.voiceover(
            "Most comparisons are paired. A seed fixes the initial weights and the stream of training batches, so "
            "two arms on the same seed differ only in the thing under test. The statistic is an exact McNemar test "
            "on the discordant seeds, where one arm succeeded and the other failed. When you hear \"eleven versus "
            "zero\", that is what it counts. Results are also put in bands: reliable at ninety percent or more, "
            "majority at half, minority for at least one success."
        ) as vo:
            self.play(*self.retitle("Paired seeds and bands"), FadeIn(chip, scale=0.8),
                      *self.source("Report §3 (Statistics); data/slow_start.json (X, seeds 280–299)"), run_time=0.9)
            # sentence 1: what a seed fixes
            vo.wait_until(at(vo, 1, "the initial weights") - 0.2)
            self.play(GrowArrow(a_w), FadeIn(weights), FadeIn(w_lab), run_time=0.8)
            vo.wait_until(at(vo, 1, "the stream of training"))
            self.play(GrowArrow(a_b), LaggedStartMap(FadeIn, batches, shift=0.1 * RIGHT, lag_ratio=0.2),
                      FadeIn(b_lab), run_time=0.9)
            vo.wait_until(at(vo, 1, "two arms") - 0.2)
            self.play(FadeIn(arm_h.nm), FadeIn(arm_a.nm), TransformFromCopy(weights, arm_h.w),
                      TransformFromCopy(weights, arm_a.w), TransformFromCopy(batches, arm_h.b),
                      TransformFromCopy(batches, arm_a.b), run_time=1.2)
            self.play(FadeIn(arm_a.box), FadeIn(arm_h.box), run_time=0.6)
            vo.wait_until(at(vo, 1, "differ only"))
            self.play(Write(diff_lab), Indicate(arm_h.box, color=HINGE_COLOR, scale_factor=1.08), run_time=0.9)
            vo.wait_until(at(vo, 1, "thing under test"))
            self.play(FadeIn(out_h, scale=1.3), FadeIn(out_a, scale=1.3), run_time=0.6)
            # sentence 2: the paired strip, then the 2x2 table
            vo.wait_until_sentence(2)
            first = cols_[0]
            self.play(FadeOut(VGroup(chip, weights, batches, w_lab, b_lab, a_w, a_b, arm_h, arm_a, diff_lab)),
                      ReplacementTransform(out_h, first[1]), ReplacementTransform(out_a, first[2]),
                      FadeIn(first[0]), FadeIn(lab_h), FadeIn(lab_a), FadeIn(s_lab[0]), run_time=0.9)
            self.play(LaggedStartMap(FadeIn, cols_[1:], shift=0.1 * RIGHT, lag_ratio=0.08), FadeIn(s_lab[1]),
                      run_time=1.3)
            self.play(FadeIn(tot_h), FadeIn(tot_a), FadeIn(cell_g), FadeIn(col_h), FadeIn(row_h), run_time=0.7)
            ghosts = cols_.copy().set_opacity(0.18)
            self.add(ghosts)
            self.bring_to_front(cols_)
            movers = []
            for i in range(20):
                tgt = cols_[i].copy()
                tgt.scale(0.62).move_to(targets[i])
                movers.append(Transform(cols_[i], tgt))
            self.play(LaggedStart(*movers, lag_ratio=0.05), run_time=1.7)
            self.play(*[FadeIn(n) for n in counts.values()], run_time=0.5)
            vo.wait_until(at(vo, 2, "where one arm") - 0.3)
            self.play(FadeIn(glow), FadeIn(d_lab), run_time=0.7)
            self.bring_to_front(cols_)
            self.play(Write(disc), run_time=0.7)
            # sentence 3: eleven versus zero
            vo.wait_until_sentence(3)
            self.play(TransformFromCopy(counts[(0, 1)], big[0]), FadeIn(big[1]),
                      TransformFromCopy(counts[(1, 0)], big[2]),
                      *self.source("Report Table 4 (H0 on X); data/slow_start.json"), run_time=1.0)
            self.play(FadeIn(pv, shift=0.1 * UP), FadeIn(sh, shift=0.1 * UP), run_time=0.7)
            # sentence 4: bands
            vo.wait_until_sentence(4)
            moved = VGroup(*cols_)
            self.play(FadeOut(VGroup(cell_g, col_h, row_h, moved, glow, d_lab, disc, big, pv, sh,
                                     *counts.values())),
                      ghosts.animate.set_opacity(1), run_time=0.8)
            self.play(ShowCreation(ruler), FadeIn(ticks), *self.source(
                "bands: test_early_recipe.py:64; test_slow_start.py:99-100 (of 20: 18, 10, 1); Report Table 4"), run_time=0.7)
            vo.wait_until(at(vo, 4, "reliable"))
            self.play(GrowFromEdge(z_rel[0], LEFT), FadeIn(z_rel[1], shift=0.1 * DOWN), run_time=0.6)
            self.play(TransformFromCopy(tot_h, mk_h[1][0]), FadeIn(mk_h[0]), FadeIn(mk_h[1][1], scale=1.2),
                      run_time=0.9)
            vo.wait_until(at(vo, 4, "majority"))
            self.play(GrowFromEdge(z_maj[0], LEFT), FadeIn(z_maj[1], shift=0.1 * DOWN), run_time=0.6)
            vo.wait_until(at(vo, 4, "minority"))
            self.play(GrowFromEdge(z_min[0], LEFT), FadeIn(z_min[1], shift=0.1 * DOWN), run_time=0.6)
            self.play(TransformFromCopy(tot_a, mk_a[1][0]), FadeIn(mk_a[0]), FadeIn(mk_a[1][1], scale=1.2),
                      run_time=0.9)
            self.play(FadeIn(z_nev, scale=1.2), run_time=0.6)
        self.play(Indicate(VGroup(mk_h[1][1], mk_a[1][1]), scale_factor=1.1), run_time=0.9)
        self.wait(0.6)
