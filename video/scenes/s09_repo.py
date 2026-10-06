"""Chapter 9 — Inside the repository (README "Repository layout and how to run the tests"; report App. A,
Table 9; facts_repo §1-4; results/README.md; results/X, results/L; git log)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------------------------- data
# The 34 test files in first-commit order (git log --diff-filter=A), grouped by research phase.
PHASES = [("I", 8), ("II", 2), ("III", 9), ("IV", 10), ("V", 5)]
TESTS = [
    "cross_channel", "interference", "interference_auxloss", "interference_discrete", "grow_inference",
    "asymmetric_routing", "bounded_capacity", "arm2_readinit",
    "instrument_v2", "phase2_seeds",
    "capacity_control", "binding_capacity", "multilayer_binding", "binding_onset", "binding_recipe",
    "channel_binding", "router_discovery", "router_curriculum", "router_confirm",
    "readout_path", "router_reliability", "router_layout", "short_conv", "conv_lr", "p_scaling",
    "load_curriculum", "curriculum_confirm", "scale_axes", "stream_channels",
    "stream_recipe", "stream_curriculum", "slow_start", "recipe_scope", "early_recipe",
]
# `import test_*` / `from test_* import` lines of each file (grep over the checkout).
IMPORTS = {
    "asymmetric_routing": "interference", "bounded_capacity": "interference", "arm2_readinit": "interference",
    "instrument_v2": "interference", "phase2_seeds": "instrument_v2", "capacity_control": "instrument_v2",
    "binding_capacity": "instrument_v2 interference",
    "multilayer_binding": "binding_capacity instrument_v2",
    "binding_onset": "instrument_v2 multilayer_binding",
    "binding_recipe": "binding_onset instrument_v2 multilayer_binding",
    "channel_binding": "binding_onset binding_recipe multilayer_binding",
    "router_discovery": "binding_onset binding_recipe channel_binding instrument_v2 multilayer_binding",
    "router_curriculum": "binding_onset channel_binding router_discovery",
    "router_confirm": "binding_onset channel_binding multilayer_binding router_curriculum",
    "readout_path": "binding_onset channel_binding router_confirm router_curriculum",
    "router_reliability": "binding_onset channel_binding instrument_v2 multilayer_binding readout_path "
                          "router_confirm router_curriculum router_discovery",
    "router_layout": "binding_capacity binding_onset binding_recipe channel_binding multilayer_binding readout_path "
                     "router_confirm router_curriculum router_discovery router_reliability",
    "short_conv": "binding_onset channel_binding multilayer_binding readout_path router_confirm router_curriculum "
                  "router_discovery router_layout router_reliability",
    "conv_lr": "binding_onset channel_binding multilayer_binding readout_path router_confirm router_curriculum "
               "router_discovery router_layout router_reliability short_conv",
    "p_scaling": "binding_onset channel_binding conv_lr multilayer_binding readout_path router_confirm "
                 "router_curriculum router_discovery router_layout router_reliability short_conv",
    "load_curriculum": "binding_capacity binding_onset binding_recipe channel_binding conv_lr multilayer_binding "
                       "p_scaling readout_path router_confirm router_curriculum router_discovery router_layout "
                       "router_reliability short_conv",
    "curriculum_confirm": "binding_capacity binding_onset channel_binding conv_lr load_curriculum multilayer_binding "
                          "p_scaling readout_path router_confirm router_curriculum router_discovery router_layout "
                          "router_reliability short_conv",
    "scale_axes": "binding_onset channel_binding conv_lr curriculum_confirm multilayer_binding p_scaling readout_path "
                  "router_confirm router_curriculum router_discovery router_layout router_reliability short_conv",
    "stream_channels": "binding_onset channel_binding conv_lr multilayer_binding readout_path router_confirm "
                       "router_curriculum router_discovery router_layout router_reliability scale_axes short_conv",
    "stream_recipe": "binding_onset channel_binding conv_lr curriculum_confirm load_curriculum multilayer_binding "
                     "router_confirm router_curriculum router_discovery router_layout router_reliability scale_axes "
                     "short_conv stream_channels",
    "stream_curriculum": "binding_capacity binding_onset binding_recipe channel_binding conv_lr curriculum_confirm "
                         "load_curriculum multilayer_binding readout_path router_confirm router_curriculum "
                         "router_discovery router_layout router_reliability scale_axes short_conv stream_channels "
                         "stream_recipe",
    "slow_start": "binding_onset channel_binding conv_lr multilayer_binding router_curriculum router_discovery "
                  "router_layout router_reliability scale_axes short_conv stream_channels stream_curriculum "
                  "stream_recipe",
    "recipe_scope": "binding_onset channel_binding conv_lr multilayer_binding router_curriculum router_discovery "
                    "router_reliability slow_start stream_channels stream_curriculum stream_recipe",
    "early_recipe": "binding_onset conv_lr multilayer_binding recipe_scope router_curriculum router_discovery "
                    "router_reliability short_conv slow_start stream_channels stream_curriculum stream_recipe",
}
# Highest CHECK number each test owns (facts_repo §2.4; each test's verify()).
CHECK_TOP = {
    "multilayer_binding": 6, "binding_onset": 10, "binding_recipe": 14, "channel_binding": 18,
    "router_discovery": 25, "router_curriculum": 31, "router_confirm": 36, "readout_path": 41,
    "router_reliability": 47, "router_layout": 52, "short_conv": 57, "conv_lr": 63, "p_scaling": 68,
    "load_curriculum": 74, "curriculum_confirm": 79, "scale_axes": 84, "stream_channels": 89,
    "stream_recipe": 94, "stream_curriculum": 100, "slow_start": 112, "recipe_scope": 119, "early_recipe": 128,
}
X_FIRST = TESTS.index("channel_binding")      # results/X holds channel_binding .. early_recipe (19 files)

# Pull-request merges into the default branch (#2 to #20), git log --merges, in days after 2026-09-21 00:00 UTC.
def _days(mm_dd_hh_mm, utc_offset_h=0):
    md, hm = mm_dd_hh_mm.split()
    m, d = map(int, md.split("-"))
    h, mi = map(int, hm.split(":"))
    day = d - 21 if m == 9 else d + 9
    return day + (h - utc_offset_h + mi / 60) / 24


# the merges carry Ynkling's local time (-0600); converted to UTC like every other time in the diagram
MERGES = [_days(s, -6) for s in [
    "09-21 09:34", "09-24 08:30", "09-24 19:56", "09-25 09:42", "09-25 12:39", "09-25 22:05", "09-26 12:16",
    "09-26 17:42", "09-26 22:30", "09-27 11:44", "09-27 16:42", "09-27 23:34", "09-27 23:45", "09-30 09:44",
    "09-30 14:38", "10-01 19:57", "10-03 11:38", "10-04 10:33", "10-05 14:10"]]
# commits per day on claude/outside-ideas (facts_repo §3.3): 954 in total
OUTSIDE_PER_DAY = {"09-28": 4, "09-30": 210, "10-01": 144, "10-02": 196, "10-03": 224, "10-04": 132, "10-05": 44}


def run_seconds():
    """Wall seconds of every run in X's stream_recipe results (for the worker lanes)."""
    try:
        with open(os.path.join(REPO, "results", "X", "stream_recipe_results.json")) as fh:
            runs = json.load(fh)["runs"]
        secs = [float(r.get("secs") or (r.get("trial") or {}).get("secs") or 0) for r in runs.values()]
        return [s for s in secs if s > 0]
    except Exception:  # pragma: no cover - fallback keeps the scene renderable
        rng = np.random.default_rng(0)
        return list(rng.uniform(400, 1550, 83))


# ---------------------------------------------------------------------------------------------- helpers
CHAR_W = 0.1098   # advance of one DejaVu Sans Mono glyph at font size 20


def mono(s, size=20, color=INK, **kw):
    return Text(s, font=MONO, font_size=size, fill_color=color, **kw)


def mline(s, color=INK, t2c=None, size=20):
    """A mono line with a constant vertical extent: an invisible '|' glyph leads it."""
    t = Text("|" + s, font=MONO, font_size=size, fill_color=color, t2c=t2c or {})
    t[0].set_opacity(0)
    return t


def place_line(t, x, y):
    """Put a mline so that its first character cell starts at x, vertically centred on y."""
    t.shift(np.array([x - CHAR_W / 2, y, 0]) - t[0].get_center())
    return t


def at(vo, i, phrase):
    """Approximate time (in the block) at which `phrase` is spoken inside sentence i."""
    text, a, b = vo.synth.sentences[i]
    k = text.find(phrase)
    if k < 0:
        return a
    return a + (b - a) * k / max(1, len(text))


def box(text, width, height=0.5, color=MUTED, size=20, fill=PANEL, text_color=INK):
    r = RoundedRectangle(width=width, height=height, corner_radius=0.1)
    r.set_fill(fill, 1).set_stroke(color, 1.6)
    t = L(text, size=size, color=text_color) if isinstance(text, str) else text
    if t.get_width() > width - 0.2:
        t.set_width(width - 0.2)
    t.move_to(r)
    g = VGroup(r, t)
    g.rect, g.label = r, t
    return g


def file_icon(color, w=0.42, h=0.52, glyph="{ }", dashed=False):
    body = RoundedRectangle(width=w, height=h, corner_radius=0.06)
    if dashed:
        body.set_stroke(color, 1.5).set_fill(opacity=0)
        body = DashedVMobject(body, num_dashes=14)
        return VGroup(body)
    body.set_fill(PANEL, 1).set_stroke(color, 1.6)
    g = VGroup(body)
    if glyph:
        t = mono(glyph, 20, color)
        t.set_width(min(t.get_width(), w * 0.75)).move_to(body)
        g.add(t)
    return g


class Term:
    """A terminal panel that prints mono lines and scrolls."""

    def __init__(self, scene, center, width, height, title="~/clankers"):
        self.scene = scene
        panel = RoundedRectangle(width=width, height=height, corner_radius=0.15)
        panel.set_fill("#090B0F", 1).set_stroke(PANEL_EDGE, 1.5).move_to(center)
        bar_y = panel.get_top()[1] - 0.25
        dots = VGroup(*[Dot(radius=0.06, fill_color=c) for c in (BAD, WARN, GOOD)]).arrange(RIGHT, buff=0.12)
        dots.move_to([panel.get_left()[0] + 0.42, bar_y, 0])
        ttl = mono(title, 20, MUTED).move_to([panel.get_center()[0], bar_y, 0])
        sep = Line(panel.get_left() * RIGHT + (bar_y - 0.24) * UP, panel.get_right() * RIGHT + (bar_y - 0.24) * UP)
        sep.set_stroke(PANEL_EDGE, 1.2)
        self.chrome = VGroup(panel, dots, ttl, sep)
        self.x = panel.get_left()[0] + 0.28
        self.top = bar_y - 0.55
        self.bottom = panel.get_bottom()[1] + 0.25
        self.lh = 0.37
        self.lines = []
        self.next_y = self.top

    def make(self, s, color=INK, t2c=None):
        return place_line(mline(s, color, t2c), self.x, self.next_y)

    def push(self, s, color=INK, t2c=None, typed=False, rt=None, extra=()):
        anims = list(extra)
        if self.next_y < self.bottom - 1e-6:
            shift = self.lh
            for ln in self.lines:
                if ln.get_center()[1] + shift > self.top + 0.05:
                    anims.append(FadeOut(ln, shift=shift * UP))
                else:
                    anims.append(ln.animate.shift(shift * UP))
            self.lines = [ln for ln in self.lines if ln.get_center()[1] + shift <= self.top + 0.05]
            self.next_y += shift
        t = self.make(s, color, t2c)
        self.lines.append(t)
        self.next_y -= self.lh
        if typed:
            run = rt or min(1.6, max(0.4, 0.03 * len(s)))
            scroll = anims[len(extra):]
            if scroll:                    # scroll first, so the typed line never overlaps the previous one
                self.scene.play(*scroll, run_time=0.2)
                anims = list(extra)
            self.scene.play(ShowIncreasingSubsets(t), *anims, run_time=run)
        else:
            self.scene.play(FadeIn(t, shift=0.05 * UP), *anims, run_time=rt or 0.3)
        return t

    def clear(self):
        anims = [FadeOut(ln) for ln in self.lines]
        self.lines = []
        self.next_y = self.top
        return anims


# ---------------------------------------------------------------------------------------------- scene
class RepoTour(ClankersScene):
    def construct(self):
        card = self.chapter_card(9, "Inside the repository")
        self.wait(0.6)
        self.title = section_title("The repository")
        self.play(FadeOut(card, shift=0.3 * UP), FadeIn(self.title, shift=0.3 * UP))

        self.part_model()
        self.part_tests()
        self.part_results()
        self.part_branches()
        self.part_run()
        self.wait(0.5)
        self.clear_all()

    # ------------------------------------------------------------------------------------------ titles
    def retitle(self, text):
        new = section_title(text)
        old = self.title
        self.title = new
        return [FadeOut(old, shift=0.2 * UP), FadeIn(new, shift=0.2 * UP)]

    # ------------------------------------------------------------------------------------------ N9.1
    def part_model(self):
        # ---- the file tree (left)
        rows_spec = [
            ("clankers/", 0, MUTED), ("bdh.py", 1, INK), ("bdh_mc.py", 1, MUTED), ("bdh_multichannel.py", 1, MUTED),
            ("bdh_recurrent.py", 1, MUTED), ("train.py  train_two_task.py", 1, MUTED),
            ("test_*.py   (34 files)", 1, INK), ("test_instrument_v2.py", 2, INK),
            ("test_multilayer_binding.py", 2, INK), ("… 32 more", 2, MUTED),
            ("results/   X/  L/  README.md", 1, INK), ("README.md  requirements.txt", 1, MUTED),
            ("multichannel_hebbian_report*.pdf", 1, MUTED), ("figs/  LICENSE.md", 1, MUTED),
        ]
        x0, y0, dy, ind = -6.45, 2.45, 0.4, 0.42
        rows = VGroup()
        for i, (s, depth, col) in enumerate(rows_spec):
            rows.add(place_line(mline(s, col), x0 + depth * ind, y0 - i * dy))
        rows[6][1:10].set_color(INK)
        guides = VGroup()
        gx1 = x0 + 0.12
        guides.add(Line([gx1, y0 - 0.2, 0], [gx1, y0 - 13 * dy, 0]))
        for i, (_, depth, _) in enumerate(rows_spec):
            if depth == 1:
                guides.add(Line([gx1, y0 - i * dy, 0], [x0 + ind - 0.06, y0 - i * dy, 0]))
        gx2 = x0 + ind + 0.12
        guides.add(Line([gx2, y0 - 6 * dy - 0.2, 0], [gx2, y0 - 9 * dy, 0]))
        for i in (7, 8, 9):
            guides.add(Line([gx2, y0 - i * dy, 0], [x0 + 2 * ind - 0.06, y0 - i * dy, 0]))
        guides.set_stroke(FAINT, 1.5)
        tree = VGroup(guides, rows)

        # ---- the model diagram (right)
        DX, BW = 2.6, 3.9
        loop_y_a = [1.2, 0.45, -0.3, -1.05]
        loop_y_b = [1.2, 0.45, -0.3, -1.05, -1.8]
        hdr_a = VGroup(mono("bdh.py", 22, MEMORY_COLOR), L("class BDH:  Pathway's model, unmodified", 22, MUTED))
        hdr_a.arrange(RIGHT, buff=0.3).move_to([DX, 2.92, 0])
        hdr_b = VGroup(mono("test_multilayer_binding.py", 22, GATE_COLOR), L("class MultiBDH(BDH)", 22, INK))
        hdr_b.arrange(RIGHT, buff=0.3).move_to([DX, 2.92, 0])
        embed = box("tokens → embed, LayerNorm", 3.2, 0.46).move_to([DX, 2.22, 0])
        enc = box("encoder, ReLU → sparse code x", BW)
        attn = box("attention: Q = K = x, no softmax", BW)
        encv = box("encoder_v, ReLU → sparse code y", BW)
        dec = box("x · y → decoder → residual", BW)
        loop_boxes = [enc, attn, encv, dec]
        for b, y in zip(loop_boxes, loop_y_a):
            b.move_to([DX, y, 0])

        def container(ys):
            top, bot = ys[0] + 0.45, ys[-1] - 0.45
            r = RoundedRectangle(width=BW + 0.4, height=top - bot, corner_radius=0.15)
            r.move_to([DX, (top + bot) / 2, 0])
            return DashedVMobject(r.set_stroke(MUTED, 1.5), num_dashes=60)

        cont = container(loop_y_a)
        lm_a = box("lm_head → logits", 2.6, 0.46).move_to([DX, -2.1, 0])

        def down_arrow(y1, y2):
            return Arrow([DX, y1, 0], [DX, y2, 0], buff=0.02, thickness=2.0).set_color(MUTED)

        arr_in = down_arrow(1.98, 1.66)
        arr_out_a = down_arrow(-1.5, -1.86)

        def loop_arrow(y_first, y_last):
            xr = DX + BW / 2
            xl = xr + 0.38
            path = VMobject().set_points_as_corners([[xr + 0.04, y_last, 0], [xl, y_last, 0], [xl, y_first, 0],
                                                     [xr + 0.2, y_first, 0]])
            path.set_stroke(INK, 2.5)
            tip = Arrow([xr + 0.3, y_first, 0], [xr + 0.02, y_first, 0], buff=0, thickness=2.5,
                        max_tip_length_to_length_ratio=0.9).set_color(INK)
            return VGroup(path, tip)

        loop = loop_arrow(loop_y_a[0], loop_y_a[-1])
        loop_lab = VGroup(L("every layer,", 20, INK), L("same weights", 20, MUTED)).arrange(DOWN, buff=0.1)
        loop_lab.next_to(loop, RIGHT, buff=0.15)
        diagram = VGroup(embed, arr_in, cont, *loop_boxes, arr_out_a, lm_a, loop, loop_lab)

        def hl(mob, color, buff=0.06):
            return SurroundingRectangle(mob[1:], buff=buff).set_stroke(color, 2).round_corners(0.05)

        with self.voiceover(
            "Here is the repository itself. bdh dot py is Pathway's model, unmodified. The multi-channel model, "
            "MultiBDH, lives in test multilayer binding dot py. It subclasses BDH and replaces only the attention "
            "module, multiplying every score by the gate match at every layer. With the convolution on, it runs a "
            "copy of BDH's layer loop with the convolution inserted. The gate's own code is imported from the "
            "Phase Two instrument."
        ) as vo:
            self.play(ShowCreation(guides, lag_ratio=0.1), LaggedStartMap(FadeIn, rows, shift=0.1 * RIGHT,
                                                                          lag_ratio=0.08), run_time=2.0)
            vo.wait_until(at(vo, 0, "bdh dot py"))
            hl_bdh = hl(rows[1], MEMORY_COLOR)
            link0 = CurvedArrow(rows[1].get_right() + 0.15 * RIGHT, hdr_a[0].get_left() + 0.12 * LEFT,
                                angle=-PI / 6).set_color(MEMORY_COLOR)
            self.play(ShowCreation(hl_bdh), FadeIn(hdr_a, shift=0.15 * DOWN), ShowCreation(link0), run_time=0.9)
            self.play(FadeOut(link0), LaggedStart(FadeIn(embed), GrowArrow(arr_in), ShowCreation(cont),
                                  *[FadeIn(b, shift=0.1 * DOWN) for b in loop_boxes],
                                  GrowArrow(arr_out_a), FadeIn(lm_a), lag_ratio=0.18), run_time=2.2)
            self.play(ShowCreation(loop), FadeIn(loop_lab), run_time=0.9)

            # -- MultiBDH lives in test_multilayer_binding.py
            vo.wait_until_sentence(1)
            hl_mlb = hl(rows[8], GATE_COLOR)
            link = CurvedArrow(rows[8].get_right() + 0.1 * RIGHT, hdr_b[0].get_left() + 0.1 * LEFT + 0.05 * DOWN,
                               angle=-PI / 3)
            link.set_color(GATE_COLOR)
            self.play(hl_bdh.animate.set_stroke(opacity=0.35), ShowCreation(hl_mlb), run_time=0.7)
            self.play(ShowCreation(link), FadeOut(hdr_a, shift=0.15 * UP), FadeIn(hdr_b, shift=0.15 * UP),
                      run_time=1.2)
            vo.wait_until(at(vo, 1, "binding dot py"))
            self.play(Indicate(hdr_b[0], color=GATE_COLOR, scale_factor=1.06),
                      Indicate(rows[8][1:], color=GATE_COLOR, scale_factor=1.04), run_time=1.0)

            # -- replaces only the attention module, gate match at every layer
            vo.wait_until_sentence(2)
            self.play(FadeOut(link), Indicate(hdr_b[1], color=GATE_COLOR, scale_factor=1.06), run_time=0.8)
            vo.wait_until(at(vo, 2, "replaces only"))
            self.play(Indicate(attn, color=GATE_COLOR, scale_factor=1.05), run_time=0.9)
            gated_lab = VGroup(L("GatedAttention:", 20, GATE_COLOR),
                               M(R"\text{score}\times(g_t\cdot g_s)", 30)).arrange(RIGHT, buff=0.2)
            gattn = box(gated_lab, BW, color=GATE_COLOR).move_to(attn)
            gattn.rect.set_stroke(GATE_COLOR, 2.4)
            old_attn = attn.copy()
            gate = VGroup(RoundedRectangle(width=1.6, height=0.95, corner_radius=0.12)
                          .set_fill(GATE_COLOR, 0.12).set_stroke(GATE_COLOR, 2.2))
            gate_lab = L("gate", 22, GATE_COLOR, weight="BOLD")
            gate_bar_m = gate_bar([0.72, 0.28], width=1.1, height=0.2)
            VGroup(gate_lab, gate_bar_m).arrange(DOWN, buff=0.14).move_to(gate[0])
            gate.add(gate_lab, gate_bar_m)
            gate.move_to([-0.95, loop_y_a[1], 0])
            g_arrow = Arrow(gate.get_right(), attn.get_left(), buff=0.06, thickness=2.5).set_color(GATE_COLOR)
            g_lab = M("G", 30, color=GATE_COLOR).next_to(g_arrow, UP, buff=0.06)
            self.play(old_attn.animate.shift(0.3 * RIGHT).set_opacity(0), FadeTransform(attn, gattn),
                      FadeIn(gate, shift=0.2 * RIGHT), run_time=1.2)
            self.remove(old_attn)
            self.play(GrowArrow(g_arrow), FadeIn(g_lab), run_time=0.6)
            vo.wait_until(at(vo, 2, "multiplying every"))
            self.play(Indicate(gated_lab[1], color=GATE_COLOR, scale_factor=1.12), run_time=0.9)
            vo.wait_until(at(vo, 2, "at every layer"))
            flash = loop[0].copy().set_stroke(GATE_COLOR, 5)
            self.play(ShowPassingFlash(flash, time_width=0.6), Indicate(loop_lab[0], color=GATE_COLOR),
                      Indicate(gattn, color=GATE_COLOR, scale_factor=1.03), run_time=1.2)

            # -- the convolution: a copy of BDH's layer loop with the convolution inserted
            vo.wait_until_sentence(3)
            conv = box(VGroup(L("CausalConv", 20, WARN, weight="BOLD"), L("4 tokens wide, causal", 20, MUTED))
                       .arrange(RIGHT, buff=0.2), BW, color=WARN)
            conv.rect.set_stroke(WARN, 2.2)
            conv.move_to([DX, loop_y_b[0], 0])
            cont_b = container(loop_y_b)
            cont_b.set_stroke(WARN, 1.5)
            lm_b = lm_a.copy().move_to([DX, -2.85, 0])
            arr_out_b = down_arrow(-2.25, -2.61)
            loop_b = loop_arrow(loop_y_b[0], loop_y_b[-1])
            loop_lab_b = loop_lab.copy().next_to(loop_b, RIGHT, buff=0.15)
            tag = VGroup(L("a copy of BDH's loop,", 20, WARN), L("convolution inserted", 20, WARN))
            tag.arrange(DOWN, buff=0.08, aligned_edge=RIGHT)
            tag.move_to([cont_b.get_left()[0] - 0.15, loop_y_b[0], 0], aligned_edge=RIGHT)
            new_boxes = [enc, gattn, encv, dec]
            self.play(*[b.animate.move_to([DX, y, 0]) for b, y in zip(new_boxes, loop_y_b[1:])],
                      Transform(cont, cont_b), Transform(lm_a, lm_b), Transform(arr_out_a, arr_out_b),
                      Transform(loop, loop_b), Transform(loop_lab, loop_lab_b),
                      gate.animate.move_to([-0.95, loop_y_b[2], 0]),
                      g_arrow.animate.shift((loop_y_b[2] - loop_y_a[1]) * UP),
                      g_lab.animate.shift((loop_y_b[2] - loop_y_a[1]) * UP), run_time=1.2)
            self.play(FadeIn(conv, shift=0.25 * DOWN), FadeIn(tag, shift=0.2 * RIGHT), run_time=0.9)
            vo.wait_until(at(vo, 3, "inserted"))
            self.play(FlashAround(conv, color=WARN), Indicate(conv.label, color=WARN, scale_factor=1.05),
                      run_time=1.0)

            # -- the gate's code comes from the Phase II instrument
            vo.wait_until_sentence(4)
            hl_iv2 = hl(rows[7], GATE_COLOR)
            imp = Arrow(rows[7].get_right() + 0.1 * RIGHT, gate.get_left(), buff=0.08, thickness=2.5)
            imp.set_color(GATE_COLOR)
            imp_lab = VGroup(mono("Instrument.gates", 20, GATE_COLOR),
                             L("decay_mask, perfect gate", 20, MUTED)).arrange(DOWN, buff=0.1)
            imp_lab.next_to(gate, DOWN, buff=0.2)
            self.play(hl_mlb.animate.set_stroke(opacity=0.35), ShowCreation(hl_iv2), run_time=0.6)
            self.play(GrowArrow(imp), FadeIn(imp_lab, shift=0.1 * DOWN), run_time=0.9)
            src = source_note("bdh.py; test_multilayer_binding.py:187-340; test_instrument_v2.py:349")
            self.play(FadeIn(src), Indicate(gate, color=GATE_COLOR, scale_factor=1.06), run_time=1.0)

        self.p1 = dict(tree=tree, rows=rows, rest=VGroup(hl_bdh, hl_mlb, hl_iv2, imp, imp_lab, hdr_b, diagram, cont,
                                                         gattn, conv, tag, gate, g_arrow, g_lab, src))

    # ------------------------------------------------------------------------------------------ N9.2
    def part_tests(self):
        rows = self.p1["rows"]
        tile_w, tile_h, buff, gap = 0.28, 0.42, 0.06, 0.36
        xs, x = [], 0.0
        bands = []
        for _, n in PHASES:
            start = len(xs)
            for _ in range(n):
                xs.append(x + tile_w / 2)
                x += tile_w + buff
            bands.append((start, len(xs)))
            x += gap - buff
        width = x - (gap - buff) + 0.0
        xs = [xi - width / 2 for xi in xs]
        TY = -1.35
        tiles = VGroup(*[RoundedRectangle(width=tile_w, height=tile_h, corner_radius=0.05)
                         .set_fill(PANEL, 1).set_stroke(MUTED, 1.4).move_to([xi, TY, 0]) for xi in xs])
        phase_cols = ["#7C8494", "#9AA3B5", KEY_COLOR, INK, WARN]
        top_y = TY + tile_h / 2

        # time axis
        t_arrow = Arrow([-4.9, -3.02, 0], [4.9, -3.02, 0], buff=0, thickness=2).set_color(FAINT)
        t_left = L("2026-06-30", 20, MUTED).next_to(t_arrow, LEFT, buff=0.15)
        t_right = L("2026-10-04", 20, MUTED).next_to(t_arrow, RIGHT, buff=0.15)
        t_mid = L("order of each file's first commit", 20, MUTED).next_to(t_arrow, DOWN, buff=0.12)
        count = L("34 test files", 26, INK, weight="BOLD").move_to([-4.6, 1.4, 0])

        band_labels = VGroup()
        band_lines = VGroup()
        for (name, n), (a, b), col in zip(PHASES, bands, phase_cols):
            xa, xb = xs[a] - tile_w / 2, xs[b - 1] + tile_w / 2
            ln = Line([xa, TY - 0.33, 0], [xb, TY - 0.33, 0]).set_stroke(col, 3)
            lab = L(f"Phase {name}", 22, col, weight="BOLD").move_to([(xa + xb) / 2, TY - 0.6, 0])
            cnt = L(f"{n} tests", 20, MUTED).move_to([(xa + xb) / 2, TY - 0.93, 0])
            band_lines.add(ln)
            band_labels.add(VGroup(lab, cnt))
        desc_specs = [((0, 1), "instrument tests"), ((2, 2), "binding tests"), ((3, 3), "Revision 6"),
                      ((4, 4), "this report")]
        descs = VGroup()
        for (pa, pb), s in desc_specs:
            xa, xb = xs[bands[pa][0]] - tile_w / 2, xs[bands[pb][1] - 1] + tile_w / 2
            descs.add(L(s, 20, phase_cols[pb] if pb == 4 else MUTED).move_to([(xa + xb) / 2, TY - 1.25, 0]))

        # Phase V names with elbow connectors
        v_names = ["stream_recipe", "stream_curriculum", "slow_start", "recipe_scope", "early_recipe"]
        names, elbows = VGroup(), VGroup()
        for j, nm in enumerate(v_names):
            i = TESTS.index(nm)
            y = -0.55 + 0.42 * j
            t = mono(f"test_{nm}.py", 20, WARN)
            t.move_to([4.3, y, 0], aligned_edge=RIGHT)
            el = VMobject().set_points_as_corners([[xs[i], top_y + 0.04, 0], [xs[i], y, 0], [4.42, y, 0]])
            el.set_stroke(WARN, 1.5, opacity=0.8)
            names.add(t)
            elbows.add(el)

        def phase_anims(p):
            a, b = bands[p]
            col = phase_cols[p]
            anims = [tiles[i].animate.set_stroke(col, 2 if p == 4 else 1.4).set_fill(
                col if p == 4 else PANEL, 0.25 if p == 4 else 1) for i in range(a, b)]
            anims += [ShowCreation(band_lines[p]), FadeIn(band_labels[p], shift=0.1 * DOWN)]
            return anims

        with self.voiceover(
            "Then come the experiments, thirty-four test files in the order the project ran them: the Phase One and "
            "Two instrument tests, the Phase Three binding tests, the ten Phase Four tests, and the five Phase Five "
            "tests this report adds. Each later test imports its predecessors, so the chain of checks grows with the "
            "project."
        ) as vo:
            entry = rows[6]
            self.play(*self.retitle("The experiments"), FadeOut(self.p1["rest"]),
                      FadeOut(VGroup(self.p1["tree"][0], *[r for k, r in enumerate(rows) if k != 6])),
                      run_time=0.8)
            self.play(FadeTransform(entry, count),
                      LaggedStart(*[GrowFromPoint(t, entry.get_center()) for t in tiles], lag_ratio=0.03),
                      run_time=1.5)
            self.play(GrowArrow(t_arrow), FadeIn(t_left), FadeIn(t_right), FadeIn(t_mid), run_time=1.0)
            vo.wait_until(at(vo, 0, "the Phase One"))
            self.play(*phase_anims(0), *phase_anims(1), FadeIn(descs[0]), run_time=1.0)
            vo.wait_until(at(vo, 0, "the Phase Three"))
            self.play(*phase_anims(2), FadeIn(descs[1]), run_time=0.9)
            vo.wait_until(at(vo, 0, "the ten Phase Four"))
            self.play(*phase_anims(3), FadeIn(descs[2]), run_time=0.9)
            vo.wait_until(at(vo, 0, "the five Phase Five"))
            self.play(*phase_anims(4), FadeIn(descs[3]), run_time=0.8)
            self.play(LaggedStart(*[AnimationGroup(ShowCreation(e), FadeIn(n, shift=0.1 * LEFT))
                                    for e, n in zip(elbows, names)], lag_ratio=0.25), run_time=1.6)

            # -- imports: a real arc diagram of `import test_*` lines
            vo.wait_until_sentence(1)
            arcs, hot = VGroup(), VGroup()
            for j, nm in enumerate(TESTS):
                for dep in IMPORTS.get(nm, "").split():
                    i = TESTS.index(dep)
                    c = abs(xs[j] - xs[i])
                    h = min(0.12 + 0.17 * c, 1.45)
                    p0, p3 = np.array([xs[j], top_y + 0.03, 0]), np.array([xs[i], top_y + 0.03, 0])
                    a = VMobject()
                    a.start_new_path(p0)
                    a.add_cubic_bezier_curve_to(p0 + UP * h * 4 / 3, p3 + UP * h * 4 / 3, p3)
                    if nm == "early_recipe":
                        a.set_stroke(WARN, 2.2, opacity=0.95)
                        hot.add(a)
                    else:
                        a.set_stroke(MUTED, 1.1, opacity=0.45)
                        arcs.add(a)
            imp_lab = L("arcs: the earlier tests each file imports", 20, MUTED).move_to([-4.3, 0.75, 0])
            n_hot = len(IMPORTS["early_recipe"].split())
            hot_lab = L(f"yellow: test_early_recipe.py imports {n_hot}", 20, WARN)
            hot_lab.next_to(imp_lab, DOWN, buff=0.14).align_to(imp_lab, LEFT)
            self.play(FadeOut(names), FadeOut(elbows), LaggedStartMap(ShowCreation, arcs, lag_ratio=0.01),
                      FadeIn(imp_lab), run_time=1.6)
            self.play(LaggedStartMap(ShowCreation, hot, lag_ratio=0.05), FadeIn(hot_lab),
                      tiles[-1].animate.set_fill(WARN, 0.8), run_time=0.8)

            # -- the chain of numbered CHECKs grows with the project
            vo.wait_until(at(vo, 1, "so the chain"))
            bars = VGroup()
            for nm, top in CHECK_TOP.items():
                i = TESTS.index(nm)
                h = 3.55 * top / 128
                r = Rectangle(width=tile_w - 0.06, height=h).set_fill(GOOD, 0.8).set_stroke(width=0)
                r.move_to([xs[i], top_y + 0.06 + h / 2, 0])
                bars.add(r)
            i0, i1 = TESTS.index("multilayer_binding"), TESTS.index("early_recipe")
            lab0 = L("CHECK 1-6", 20, GOOD).move_to([xs[i0] - 0.2, top_y + 0.85, 0])
            lead0 = Line(lab0.get_bottom() + 0.04 * DOWN, bars[0].get_top() + 0.04 * UP).set_stroke(GOOD, 1.2)
            lab1 = L("CHECK 120-128", 20, GOOD, weight="BOLD")
            lab1.next_to(bars[-1], UP, buff=0.12).align_to(bars[-1], RIGHT).shift(0.08 * RIGHT)
            cap = VGroup(L("bar height: the test's highest CHECK", 22, GOOD),
                         L("one numbered chain, from 1 to 128", 22, GOOD)).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
            cap.move_to([-4.3, 1.55, 0])
            bars_anim = [GrowFromEdge(b, DOWN) for b in bars]
            self.play(arcs.animate.set_stroke(opacity=0.12), hot.animate.set_stroke(opacity=0.3),
                      FadeOut(count), imp_lab.animate.set_opacity(0.5), hot_lab.animate.set_opacity(0.5),
                      LaggedStart(*bars_anim, lag_ratio=0.12), FadeIn(cap), run_time=1.7)
            src = source_note("git log (first commits); import lines of test_*.py; CHECK numbers: each verify()")
            self.play(FadeIn(lab0), ShowCreation(lead0), FadeIn(lab1, shift=0.1 * UP), FadeIn(src), run_time=0.6)

        self.p2 = dict(tiles=tiles, rest=VGroup(t_arrow, t_left, t_right, t_mid, band_lines, band_labels, descs,
                                                arcs, hot, imp_lab, hot_lab, bars, lab0, lead0, lab1, cap, src))

    # ------------------------------------------------------------------------------------------ N9.3 (results)
    def part_results(self):
        tiles = self.p2["tiles"]
        XC, LC = MACHINE_COLORS["X"], MACHINE_COLORS["L"]
        # results/X: 19 JSON files
        icons = VGroup(*[file_icon(XC) for _ in range(19)])
        icons.arrange_in_grid(3, 7, buff=0.1)
        icons.move_to([-4.63, 1.12, 0])
        x_hdr = VGroup(machine_badge("X", 22), mono("results/X/", 22, INK), L("19 files", 20, MUTED))
        x_hdr.arrange(RIGHT, buff=0.22).next_to(icons, UP, buff=0.28).align_to(icons, LEFT)
        x_cap = L("one JSON file per recorded test", 20, MUTED).next_to(icons, DOWN, buff=0.2).align_to(icons, LEFT)

        table = simple_table([
            ["file", "machine (CPU)", "commit", "started"],
            ["X/channel_binding", "Xeon @ 2.10GHz", "93b227c", "2026-09-25 14:08"],
            ["X/stream_recipe", "Xeon @ 2.10GHz", "092937b", "2026-09-30 08:55"],
            ["X/early_recipe", "Xeon @ 2.80GHz", "9f25dc8", "2026-10-04 06:26"],
        ], col_widths=[1.95, 1.85, 2.6, 1.9], size=20, row_height=0.42)
        table.move_to([2.4, 1.72, 0])
        tab = mono("results/README.md", 20, MUTED).next_to(table, UP, buff=0.1).align_to(table, LEFT)

        def json_lines(git, cpu, started, workers, n_runs):
            ks = {'"meta"': KEY_COLOR, '"runs"': KEY_COLOR, '"git"': KEY_COLOR, '"cpu"': KEY_COLOR,
                  '"started"': KEY_COLOR, '"workers"': KEY_COLOR}
            specs = [
                ('{"meta": {"git": "%s",' % git, None),
                ('          "cpu": "%s",' % cpu, None),
                ('          "started": "%s",' % started, None),
                ('          "workers": %d, …},' % workers, None),
                (' "runs": {"CEIL_A|300": {…}, …}}', None),
            ]
            g = VGroup()
            for k, (s, _) in enumerate(specs):
                t2c = dict(ks)
                for val in (git, cpu, started):
                    t2c['"%s"' % val] = "#E6DB74"
                g.add(place_line(mline(s, INK, t2c=t2c), 0, -k * 0.37))
            note = L(f"{n_runs} runs", 20, MUTED).next_to(g[-1], RIGHT, buff=0.35)
            g.add(note)
            return g

        jx = json_lines("9f25dc8", "Intel(R) Xeon(R) Processor @ 2.80GHz", "2026-10-04 06:26:22", 4, 108)
        jl = json_lines("9437e01", "12th Gen Intel(R) Core(TM) i7-12650H", "2026-10-04 18:38:18", 6, 124)
        jpanel = RoundedRectangle(width=6.75, height=2.3, corner_radius=0.12).set_fill("#15181F", 1)
        jpanel.set_stroke(PANEL_EDGE, 1.5).move_to([2.0, -1.32, 0])
        for j in (jx, jl):
            j.move_to(jpanel).align_to(jpanel, LEFT).shift(0.25 * RIGHT)
        jtab_x = mono("X/early_recipe_results.json", 20, XC).next_to(jpanel, UP, buff=0.1).align_to(jpanel, LEFT)
        jtab_l = mono("L/early_recipe_results.json", 20, LC).next_to(jpanel, UP, buff=0.1).align_to(jpanel, LEFT)

        with self.voiceover(
            "Machine X's results live in results slash X: a JSON file for each recorded test, and a table of the "
            "commit, processor and start time behind each one. Of machine L's Phase Five results, only the "
            "early-recipe file is committed so far; the report regenerated L's other verdicts from L's own files."
        ) as vo:
            xt = tiles[X_FIRST:]
            self.play(*self.retitle("Results"), FadeOut(self.p2["rest"]), FadeOut(tiles[:X_FIRST]), run_time=0.7)
            self.play(ReplacementTransform(xt, VGroup(*[ic[0] for ic in icons])), run_time=1.4)
            self.play(FadeIn(VGroup(*[ic[1] for ic in icons])), FadeIn(x_hdr, shift=0.1 * RIGHT), FadeIn(x_cap),
                      run_time=0.7)
            vo.wait_until(at(vo, 0, "and a table"))
            self.play(FadeIn(tab), ShowCreation(table.rules), LaggedStartMap(FadeIn, table.cells, lag_ratio=0.2),
                      run_time=1.3)
            self.play(FadeIn(jpanel), FadeIn(jtab_x), LaggedStartMap(FadeIn, jx, shift=0.05 * RIGHT, lag_ratio=0.12),
                      run_time=1.2)
            rh = row_highlight(table, 3, color=XC, opacity=0.12)
            j_hl = SurroundingRectangle(jx[0:3], buff=0.06).set_stroke(XC, 1.6)
            src_icon = icons[-1]
            git_val = jx[0][17:24]
            c_box = SurroundingRectangle(table.cells[3][2], buff=0.05).set_stroke(XC, 2)
            link1 = Arrow(c_box.get_bottom(), git_val.get_top() + 0.06 * UP, buff=0.05, thickness=2.2).set_color(XC)
            self.play(FadeIn(rh), ShowCreation(j_hl), Indicate(src_icon, color=XC), run_time=0.9)
            self.play(ShowCreation(c_box), GrowArrow(link1), run_time=0.7)

            # -- machine L
            vo.wait_until_sentence(1)
            l_icon = file_icon(LC)
            l_icon.move_to([-6.15, -1.4, 0])
            l_hdr = VGroup(machine_badge("L", 22), mono("results/L/", 22, INK))
            l_hdr.arrange(RIGHT, buff=0.22).next_to(l_icon, UP, buff=0.3).align_to(l_icon, LEFT)
            l_name = mono("early_recipe_results.json", 20, LC).next_to(l_icon, RIGHT, buff=0.2)
            ghosts = VGroup(*[file_icon(MUTED, dashed=True) for _ in range(4)]).arrange(RIGHT, buff=0.1)
            ghosts.next_to(l_icon, DOWN, buff=0.3).align_to(l_icon, LEFT)
            g_lab = VGroup(L("other 4 Phase V tests:", 20, MUTED), L("L's own files not committed", 20, MUTED))
            g_lab.arrange(DOWN, buff=0.06, aligned_edge=LEFT).next_to(ghosts, RIGHT, buff=0.22)
            regen = VGroup(L("report: their verdicts regenerated", 20, INK),
                           L("from L's own files (Table 9)", 20, INK)).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
            regen.next_to(ghosts, DOWN, buff=0.25).align_to(ghosts, LEFT)
            # the README cell reads "9f25dc8 (run at 9437e01)": test code 9f25dc8, meta.git 9437e01
            l_row = VGroup(L("L/early_recipe", 20), L("i7-12650H", 20),
                           L("9f25dc8 (run at 9437e01)", 20, t2c={"9437e01": LC}), L("2026-10-04 18:38", 20))
            for c, m in enumerate(l_row):
                ref = table.cells[3][c]
                m.move_to(ref).shift(table.row_height * DOWN)
                if c == 0:
                    m.align_to(ref, LEFT)
            new_bottom = table.rules[2].copy().shift(table.row_height * DOWN)
            self.play(FadeOut(link1), FadeOut(c_box), FadeOut(rh), FadeOut(j_hl), FadeIn(l_hdr, shift=0.1 * RIGHT),
                      FadeIn(l_icon, scale=0.8), FadeIn(l_name, shift=0.1 * RIGHT), run_time=0.9)
            self.play(Transform(table.rules[2], new_bottom), FadeIn(l_row, shift=0.1 * DOWN),
                      FadeOut(jtab_x), FadeIn(jtab_l), FadeOut(jx), FadeIn(jl), jpanel.animate.set_stroke(LC, 1.5),
                      run_time=1.2)
            jl_hl = SurroundingRectangle(jl[0:3], buff=0.06).set_stroke(LC, 1.6)
            l_box = SurroundingRectangle(l_row[2][13:20], buff=0.03).set_stroke(LC, 2)
            link2 = Arrow(l_box.get_bottom(), jl[0][17:24].get_top() + 0.06 * UP, buff=0.05, thickness=2.2)
            link2.set_color(LC)
            self.play(ShowCreation(jl_hl), Indicate(l_icon, color=LC), run_time=0.8)
            self.play(ShowCreation(l_box), GrowArrow(link2), run_time=0.7)
            vo.wait_until(at(vo, 1, "committed so far"))
            self.play(LaggedStartMap(FadeIn, ghosts, lag_ratio=0.2), FadeIn(g_lab), run_time=1.0)
            vo.wait_until(at(vo, 1, "the report regenerated"))
            foot = L("18 other results/L files are byte-identical copies of X's (meta.cpu: Xeon).", 20, MUTED)
            foot.next_to(jpanel, DOWN, buff=0.25).align_to(jpanel, LEFT)
            self.play(FadeIn(regen, shift=0.1 * UP), run_time=0.8)
            src = source_note("results/README.md; results/X, results/L meta; report Table 9")
            self.play(FadeIn(foot), FadeIn(src), run_time=0.8)

        self.p3 = VGroup(icons, x_hdr, x_cap, table, tab, l_row, jpanel, jtab_l, jl, jl_hl, l_icon, l_hdr, l_name,
                         ghosts, g_lab, regen, foot, src, l_box, link2)

    # ------------------------------------------------------------------------------------------ N9.3 (branches)
    def part_branches(self):
        def bx(day):
            return -6.2 + 0.385 * day

        # default branch on top, the research branch below it (PRs merge up into the default branch), and the
        # screens branch at the bottom: it forks from c5ec279, a research-branch commit (merged by PR #15 on 09-30)
        YD, YR, YS = 1.6, 0.0, -1.6
        RC, DC, SC = GOOD, INK, MACHINE_COLORS["E"]
        head_r, head_d, head_s = _days("10-05 21:08"), _days("10-05 14:10", -6), _days("10-05 21:11")
        fork_s = _days("09-28 14:56")

        pre = DashedLine([-6.7, YD, 0], [bx(0), YD, 0], dash_length=0.08).set_stroke(DC, 2, opacity=0.5)
        main = Line([bx(0), YD, 0], [bx(head_d), YD, 0]).set_stroke(DC, 3)
        research = Line([bx(0.35), YR, 0], [bx(head_r), YR, 0]).set_stroke(RC, 3)
        fork_r = Line([bx(0.0), YD, 0], [bx(0.35), YR, 0]).set_stroke(RC, 2.5)
        merges = VGroup()
        for d in MERGES:
            m = Line([max(bx(d) - 0.16, bx(0.35)), YR, 0], [bx(d), YD, 0]).set_stroke(RC, 1.6, opacity=0.75)
            merges.add(VGroup(m, Dot([bx(d), YD, 0], radius=0.045, fill_color=DC)))
        fork = VMobject()
        p0, p3 = np.array([bx(fork_s), YR, 0]), np.array([bx(fork_s) + 0.6, YS, 0])
        fork.start_new_path(p0)
        fork.add_cubic_bezier_curve_to(p0 + 0.9 * DOWN, p3 + 0.4 * LEFT, p3)
        fork.set_stroke(SC, 2.5)
        screens = Line(p3, [bx(head_s), YS, 0]).set_stroke(SC, 3)
        rng = np.random.default_rng(9)
        sdots = VGroup()
        for md, n in OUTSIDE_PER_DAY.items():
            d0 = _days(md + " 00:00")
            for _ in range(max(1, round(n / 6))):
                d = d0 + rng.uniform(0.05, 0.95)
                if md == "09-28":
                    d = fork_s + rng.uniform(0.2, 0.6)
                if md == "10-05":
                    d = d0 + rng.uniform(0.05, 0.85)
                xx = max(bx(d), p3[0] + 0.05)
                sdots.add(Dot([xx, YS + rng.uniform(-0.06, 0.06), 0], radius=0.024, fill_color=SC).set_opacity(0.8))
        heads = VGroup(Dot([bx(head_r), YR, 0], radius=0.1, fill_color=RC),
                       Dot([bx(head_d), YD, 0], radius=0.1, fill_color=DC),
                       Dot([bx(head_s), YS, 0], radius=0.1, fill_color=SC))
        n954 = L("954 commits", 20, SC).next_to(screens, DOWN, buff=0.2).align_to(screens, RIGHT)
        ahead = L("+1 commit: 6dd4248", 20, RC).next_to(heads[0], DOWN, buff=0.15).align_to(heads[0], RIGHT)
        prs = L("pull requests #2 to #20", 20, RC).move_to([bx(5.5), (YR + YD) / 2 + 0.05, 0])
        prs.set_backstroke(BG, 6)
        ticks = VGroup()
        for md, lab in [("09-21 00:00", "09-21"), ("09-28 00:00", "09-28"), ("10-05 00:00", "10-05")]:
            xx = bx(_days(md))
            ticks.add(Line([xx, -2.35, 0], [xx, -2.25, 0]).set_stroke(FAINT, 1.5),
                      L(lab, 20, MUTED).move_to([xx, -2.6, 0]))
        axis = Line([bx(0) - 0.3, -2.3, 0], [bx(15), -2.3, 0]).set_stroke(FAINT, 1.5)
        asof = L("the branches as of 2026-10-05 (UTC)", 20, MUTED).move_to([-6.4, 2.45, 0], aligned_edge=LEFT)

        def bcard(role, name, status, color):
            g = VGroup(L(role, 24, color, weight="BOLD"), mono(name, 20, MUTED), L(status, 22, INK))
            g.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
            return card(g, buff=0.2)

        c_r = bcard("research branch", "claude/bdh-growth-hebbian-inference-w90069", "Revision 7 report and its README", RC)
        c_d = bcard("default branch", "claude/bdh-repo-curl-obgdqk", "README still follows Revision 6", DC)
        c_s = bcard("screens branch", "claude/outside-ideas", "exploratory screens on E, not results", SC)
        for c, y in ((c_r, YR), (c_d, YD), (c_s, YS)):
            c.move_to([0.25, y, 0], aligned_edge=LEFT)
        conns = VGroup(*[Line(h.get_right(), c.get_left()).set_stroke(col, 1.5, opacity=0.7)
                         for h, c, col in ((heads[0], c_r, RC), (heads[1], c_d, DC), (heads[2], c_s, SC))])
        c_d[0].set_stroke(DC, 1.5)
        c_r[0].set_stroke(RC, 1.5)
        c_s[0].set_stroke(SC, 1.5)

        with self.voiceover(
            "And mind the branches: at the time of this video, the default branch's README still follows Revision "
            "Six, Revision Seven and its README sit on the research branch, and the screens live on a branch of "
            "their own."
        ) as vo:
            self.play(*self.retitle("Branches"), FadeOut(self.p3), run_time=0.7)
            self.play(ShowCreation(pre), ShowCreation(main), ShowCreation(axis), FadeIn(ticks), FadeIn(asof),
                      run_time=1.0)
            vo.wait_until(at(vo, 0, "the default branch"))
            self.play(FadeIn(heads[1], scale=0.5), ShowCreation(conns[1]), FadeIn(c_d, shift=0.15 * RIGHT),
                      run_time=0.9)
            self.play(Indicate(c_d[1][2], color=WARN, scale_factor=1.06), run_time=0.9)
            vo.wait_until(at(vo, 0, "Revision Seven"))
            self.play(ShowCreation(fork_r), ShowCreation(research), LaggedStartMap(ShowCreation, merges, lag_ratio=0.05),
                      FadeIn(prs), run_time=1.3)
            self.play(FadeIn(heads[0], scale=0.5), FadeIn(ahead), ShowCreation(conns[0]),
                      FadeIn(c_r, shift=0.15 * RIGHT), run_time=0.9)
            vo.wait_until(at(vo, 0, "and the screens"))
            self.play(ShowCreation(fork), run_time=0.5)
            self.play(ShowCreation(screens), LaggedStartMap(FadeIn, sdots, lag_ratio=0.01), FadeIn(n954),
                      FadeIn(heads[2], scale=0.5), ShowCreation(conns[2]), FadeIn(c_s, shift=0.15 * RIGHT),
                      run_time=1.3)
            src = source_note("git ls-remote origin; git log --merges; facts_repo §3.2-3.3")
            self.play(FadeIn(src), run_time=0.4)

        self.p3b = VGroup(pre, main, research, fork_r, merges, fork, screens, sdots, heads, n954, ahead, prs, ticks,
                          axis, asof, c_r, c_d, c_s, conns, src)

    # ------------------------------------------------------------------------------------------ N9.4
    def part_run(self):
        term = Term(self, [-2.95, -0.3, 0], 7.3, 6.3)
        prompt = {"$": GOOD}
        XC = MACHINE_COLORS["X"]

        # right column: requirements, pipeline pills, worker lanes
        req = VGroup(mono("requirements.txt", 20, MUTED), L("torch · numpy · requests", 22, INK))
        req.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        req = card(req, buff=0.2).move_to([3.85, 1.2, 0])
        pills = VGroup(verdict_badge("CHECKs", 22, GOOD), verdict_badge("projection", 22, WARN),
                       verdict_badge("training", 22, XC))
        pills.arrange(RIGHT, buff=0.55).move_to([3.85, 2.5, 0])
        p_arrows = VGroup(*[Arrow(pills[i].get_right(), pills[i + 1].get_left(), buff=0.08, thickness=2)
                            .set_color(MUTED) for i in range(2)])

        secs = run_seconds()
        lanes_n, x_a, x_b = 4, 2.3, 6.55
        lane_h, lane_gap, y_top = 0.36, 0.12, 1.65
        ends = [0.0] * lanes_n
        jobs = []
        for s in sorted(secs, reverse=True):
            w = int(np.argmin(ends))
            jobs.append((w, ends[w], s))
            ends[w] += s
        scale = (x_b - x_a) / max(ends)
        lane_y = [y_top - k * (lane_h + lane_gap) for k in range(lanes_n)]
        tracks = VGroup(*[Rectangle(width=x_b - x_a, height=lane_h).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1)
                          .move_to([(x_a + x_b) / 2, y, 0]) for y in lane_y])
        lane_labs = VGroup(*[L(f"worker {k + 1}", 20, MUTED).move_to([x_a - 0.12, y, 0], aligned_edge=RIGHT)
                             for k, y in enumerate(lane_y)])
        lanes_cap = L("4 workers × 1 thread each", 22, INK).next_to(tracks, DOWN, buff=0.2)
        cut = 0.42 * max(ends)
        done, partial, later = VGroup(), VGroup(), VGroup()
        jobs.sort(key=lambda j: j[1])
        for w, st, s in jobs:
            x_start = x_a + st * scale
            wd = s * scale
            r = Rectangle(width=max(wd - 0.02, 0.01), height=lane_h - 0.08).set_fill(XC, 0.75).set_stroke(width=0)
            r.move_to([x_start + wd / 2, lane_y[w], 0])
            if st + s <= cut:
                done.add(r)
            else:
                later.add(r)
                if st < cut:
                    pw = (cut - st) * scale
                    pr = Rectangle(width=max(pw - 0.02, 0.01), height=lane_h - 0.08).set_fill(XC, 0.35)
                    pr.set_stroke(width=0).move_to([x_start + pw / 2, lane_y[w], 0])
                    partial.add(pr)
        x_cut = x_a + cut * scale
        cut_line = DashedLine([x_cut, y_top + 0.3, 0], [x_cut, lane_y[-1] - 0.3, 0], dash_length=0.06)
        cut_line.set_stroke(BAD, 2.5)
        cut_lab = mono("^C", 20, BAD).next_to(cut_line, UP, buff=0.05)

        src_run = source_note("prints of test_stream_recipe.py, shortened; hours: results/X meta")

        # ------------------------------------------------ block 1: command, checks, projection, workers, cache
        with self.voiceover(
            "Running a test is one command. It runs its checks, prints the projection, then trains on four workers "
            "with one thread each. Finished runs are cached, so an interrupted test resumes."
        ) as vo:
            self.play(*self.retitle("Running a test"), FadeOut(self.p3b), FadeIn(term.chrome), run_time=0.7)
            term.push("$ pip install -r requirements.txt", t2c=prompt, typed=True, rt=0.7,
                      extra=[FadeIn(req, shift=0.1 * LEFT)])
            term.push("$ python3 test_stream_recipe.py --workers 4", t2c=prompt, typed=True, rt=0.8)

            vo.wait_until_sentence(1)
            term.push("CHECK 1  k=1, no gate, RoPE, weights copied from a bdh.BDH …", MUTED, rt=0.3,
                      extra=[FadeOut(req), FadeIn(pills[0], scale=0.8), FadeIn(src_run)])
            term.push("         -> REPRODUCES bdh.BDH", GOOD, rt=0.2)
            ticker = term.push("CHECK 2", MUTED, rt=0.15)
            chain = [n for n in range(2, 95) if not 11 <= n <= 14]
            ticker_y = ticker[0].get_center()[1]

            def tick(mob, alpha):
                n = chain[min(len(chain) - 1, int(alpha * len(chain)))]
                mob.become(place_line(mline(f"CHECK {n}", MUTED), term.x, ticker_y))

            self.play(UpdateFromAlphaFunc(ticker, tick), run_time=1.0, rate_func=linear)
            term.push("ALL VERIFICATION CHECKS PASSED", GOOD, rt=0.25)
            vo.wait_until(at(vo, 1, "prints the projection"))
            term.push("PROJECTION — worst case … on 4 workers;  total 14.04 h", INK, rt=0.35,
                      extra=[GrowArrow(p_arrows[0]), FadeIn(pills[1], scale=0.8)])
            term.push("*** exceeds 12 h: PART 2 IS DROPPED. Part 1 alone: 11.28 h", WARN, rt=0.3)
            vo.wait_until(at(vo, 1, "then trains"))
            term.push("RUNS — 4 worker processes x 1 thread", INK, rt=0.5,
                      extra=[GrowArrow(p_arrows[1]), FadeIn(pills[2], scale=0.8), FadeIn(tracks), FadeIn(lane_labs),
                             FadeIn(lanes_cap)])
            fill = [GrowFromEdge(r, LEFT) for r in done] + [GrowFromEdge(r, LEFT) for r in partial]
            self.play(LaggedStart(*fill, lag_ratio=0.03), run_time=1.9)

            # -- interrupted, then resumed from the cache
            vo.wait_until_sentence(2)
            term.push("^C", BAD, rt=0.4, extra=[ShowCreation(cut_line), FadeIn(cut_lab), FadeOut(partial)])
            term.push("$ python3 test_stream_recipe.py --workers 4", t2c=prompt, typed=True, rt=0.5)
            term.push("CHECK 1 … 94   ALL VERIFICATION CHECKS PASSED", MUTED, rt=0.2)
            term.push("   A4k16_R     seed 240  cached", GOOD, rt=0.3, extra=[done.animate.set_fill(GOOD, 0.75)])
            term.push("   A4k16_R     seed 241  cached", GOOD, rt=0.2)
            self.play(LaggedStart(*[GrowFromEdge(r, LEFT) for r in later], lag_ratio=0.03), run_time=1.5)

        # ------------------------------------------------ block 2: determinism, pooling, pairing
        pair_hdr = VGroup(L("same processor, same thread count:", 22, INK), L("same bits", 22, GOOD, weight="BOLD"))
        pair_hdr.arrange(RIGHT, buff=0.15)
        pair_sub = L("CHECK 110 on L: A4k16 seed 240, update 2400 (accuracy, loss)", 20, MUTED)
        rec = mline("recorded  0.15576171875  2.5252773761749268", INK)
        rer = mline("re-run    0.15576171875  2.5252773761749268", INK)
        pairs = mline("-> PAIRS", GOOD)
        pair_rows = VGroup(rec, rer, pairs).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        pair = VGroup(pair_hdr, pair_sub, pair_rows).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        pair_card = card(pair, buff=0.2)
        pair_card.move_to([3.85, -1.95, 0])
        same = VGroup(*[SurroundingRectangle(VGroup(*rec[9:]), buff=0.05),
                        SurroundingRectangle(VGroup(*rer[7:]), buff=0.05)]).set_stroke(GOOD, 1.4)

        with self.voiceover(
            "Training is deterministic for a given processor and thread count, which is what makes bit-for-bit "
            "pairing possible. A flag can pool another machine's results, and tests that pair with earlier runs "
            "refuse to pair unless a recorded run reproduces exactly."
        ) as vo:
            src = source_note("results/L/recipe_scope_L_final.log (CHECK 110); README: running a test")
            self.play(FadeIn(pair_card, shift=0.15 * UP), FadeOut(src_run), FadeIn(src), run_time=0.8)
            self.play(ShowCreation(same), run_time=0.8)
            self.play(Indicate(pair_hdr[1], color=GOOD), run_time=0.8)
            vo.wait_until(at(vo, 0, "which is what makes"))
            self.play(FlashAround(pairs, color=GOOD), Indicate(pairs, color=GOOD, scale_factor=1.15), run_time=1.0)

            vo.wait_until_sentence(1)
            right_old = VGroup(pills, p_arrows, tracks, lane_labs, lanes_cap, done, later, cut_line, cut_lab)
            pool = VGroup(machine_badge("X", 26), L("+", 30, MUTED), machine_badge("L", 26))
            pool.arrange(RIGHT, buff=0.3)
            pool_lab = VGroup(L("--also: the other machine's counts,", 22, INK),
                              L("printed alongside, descriptive only", 22, MUTED)).arrange(DOWN, buff=0.08,
                                                                                            aligned_edge=LEFT)
            pool_g = VGroup(pool, pool_lab).arrange(RIGHT, buff=0.35).move_to([3.85, 2.3, 0])
            self.play(FadeOut(right_old), VGroup(pair_card, same).animate.shift(
                np.array([3.85, -0.45, 0]) - pair_card.get_center()), run_time=0.8)
            term.push("$ python3 test_stream_recipe.py --workers 4 \\", t2c=prompt, typed=True, rt=0.8)
            term.push("    --also results/X/stream_recipe_results.json", typed=True, rt=0.9,
                      extra=[FadeIn(pool_g, shift=0.1 * DOWN)])
            term.push("THE OTHER MACHINE (descriptive)", XC, rt=0.3)

            vo.wait_until(at(vo, 1, "and tests that pair"))
            q = L("does a recorded run reproduce bit for bit?", 22, INK)
            yes = VGroup(L("yes:", 22, GOOD, weight="BOLD"), L("pair the runs", 22, INK)).arrange(RIGHT, buff=0.15)
            no = VGroup(L("no:", 22, BAD, weight="BOLD"), L("DOES NOT PAIR, its claims UNTESTED", 22, INK))
            no.arrange(RIGHT, buff=0.15)
            dec = VGroup(q, VGroup(yes, no).arrange(DOWN, buff=0.15, aligned_edge=LEFT)).arrange(DOWN, buff=0.25,
                                                                                             aligned_edge=LEFT)
            dec.move_to([3.85, -2.75, 0])
            d_arrow = Arrow(pair_card.get_bottom(), q.get_top(), buff=0.06, thickness=2).set_color(MUTED)
            self.play(GrowArrow(d_arrow), FadeIn(q, shift=0.1 * DOWN), run_time=0.7)
            self.play(FadeIn(yes, shift=0.1 * RIGHT), run_time=0.5)
            self.play(FadeIn(no, shift=0.1 * RIGHT), run_time=0.5)
            self.play(Indicate(no[1], color=BAD, scale_factor=1.04), run_time=0.9)

        # ------------------------------------------------ block 3: git log and git history
        doc_rows = ["BACKGROUND", "THE RECIPE", "ARMS", "PAIRING", "CLAIMS", "DIAGNOSTICS", "CHECKS", "RUNTIME",
                    "OUTPUT"]
        DSG, RES = WARN, GOOD
        doc = VGroup(*[mono(t, 22, KEY_COLOR) for t in doc_rows]).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        res_row = mono("RESULT", 22, RES)
        res_row.next_to(doc, DOWN, buff=0.3, aligned_edge=LEFT)
        doc_all = VGroup(doc, res_row)
        doc_panel = RoundedRectangle(width=2.3, height=doc_all.get_height() + 0.6, corner_radius=0.12)
        doc_panel.set_fill("#15181F", 1).set_stroke(PANEL_EDGE, 1.5).move_to([2.25, -0.35, 0])
        doc_all.move_to(doc_panel).align_to(doc_panel, LEFT).shift(0.25 * RIGHT)
        doc_tab = mono("test_slow_start.py  docstring", 20, INK).next_to(doc_panel, UP, buff=0.12)
        doc_tab.align_to(doc_panel, LEFT)
        br1 = Brace(doc, RIGHT, buff=0.1).set_color(DSG)
        br1.next_to(doc_panel, RIGHT, buff=0.12).match_y(doc)
        br1_lab = VGroup(L("design, before any run", 22, DSG),
                         mono("9c5939e  2026-10-01", 20, MUTED, t2c={"9c5939e": DSG}))
        br1_lab.arrange(DOWN, buff=0.08, aligned_edge=LEFT).next_to(br1, RIGHT, buff=0.12)
        br2_arrow = Arrow(ORIGIN, 0.6 * LEFT, buff=0, thickness=2).set_color(RES)
        br2_lab = VGroup(L("result, appended after", 22, RES),
                         mono("b8c6007  2026-10-02", 20, MUTED, t2c={"b8c6007": RES}))
        br2_lab.arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        br2_lab.match_y(res_row).align_to(br1_lab, LEFT)
        br2_arrow = Arrow(np.array([br2_lab.get_left()[0] - 0.12, res_row.get_y(), 0]),
                          np.array([doc_panel.get_right()[0] + 0.05, res_row.get_y(), 0]), buff=0,
                          thickness=2).set_color(RES)
        br2 = VGroup(br2_arrow, br2_lab)

        legacy = VGroup(L("CHECK 15 loads an older version:", 22, INK),
                        mono("git show 2282d72:test_multilayer_binding.py", 20, WARN))
        legacy.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        legacy_card = card(legacy, buff=0.2).move_to([3.85, 1.55, 0])
        ok_row = VGroup(L("git clone", 24, INK, weight="BOLD"), L("history included", 22, MUTED),
                        Checkmark().set_color(GOOD).set_height(0.3)).arrange(RIGHT, buff=0.25)
        bad_row = VGroup(L("downloaded archive", 24, INK, weight="BOLD"), L("no history", 22, MUTED),
                         Exmark().set_color(BAD).set_height(0.3)).arrange(RIGHT, buff=0.25)
        choice = VGroup(ok_row, bad_row).arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to([3.85, -1.15, 0])
        need = L("so the tests need the repository's history", 22, MUTED)
        need.move_to([3.85, 0.2, 0])
        need_arrow = Arrow(legacy_card.get_bottom(), need.get_top(), buff=0.08, thickness=2).set_color(MUTED)

        with self.voiceover(
            "Two practical notes. The git log lets you audit the pre-registration, since each test's design is "
            "committed before its result. And the tests load older modules straight from git history, so run them "
            "from a git clone, not a downloaded archive."
        ) as vo:
            right_old2 = VGroup(pool_g, pair_card, same, d_arrow, dec)
            src_doc = source_note("git log -- test_slow_start.py; its docstring (sections, RESULT)")
            self.play(*self.retitle("Two practical notes"), FadeOut(right_old2), *term.clear(), FadeOut(src),
                      FadeIn(src_doc), run_time=0.7)
            term.push("$ git log --oneline -- test_slow_start.py", t2c=prompt, typed=True, rt=0.8)
            vo.wait_until_sentence(1)
            # newest first; the two later commits only added inert knobs used by later tests
            term.push("9f25dc8 Add test_early_recipe: the hinge on updates …", FAINT, rt=0.2)
            term.push("a429af9 Add test_recipe_scope: the hinge without the …", FAINT, rt=0.2)
            l1 = term.push("b8c6007 Record slow start result: H0 SHOWN, HA SHOWN, …", INK, rt=0.25,
                           t2c={"b8c6007": RES})
            l2 = term.push("9c5939e Add test_slow_start: slow memory plus a hinge …", INK, rt=0.25,
                           t2c={"9c5939e": DSG}, extra=[FadeIn(doc_panel), FadeIn(doc_tab)])
            self.play(Indicate(l2, color=DSG, scale_factor=1.03), LaggedStartMap(FadeIn, doc, lag_ratio=0.08),
                      GrowFromCenter(br1), FadeIn(br1_lab), run_time=1.4)
            vo.wait_until(at(vo, 1, "is committed before"))
            self.play(Indicate(l1, color=RES, scale_factor=1.03), FadeIn(res_row, shift=0.15 * UP),
                      FadeIn(br2, shift=0.1 * LEFT), run_time=1.0)

            vo.wait_until(at(vo, 2, "load older"))
            src2 = source_note("test_channel_binding.py:218-224, 255; README: running a test")
            self.play(FadeOut(VGroup(doc_panel, doc_all, doc_tab, br1, br1_lab, br2)),
                      FadeIn(legacy_card, shift=0.1 * DOWN), FadeOut(src_doc), FadeIn(src2), run_time=0.9)
            term.push("$ git clone https://github.com/Ynkling/clankers", t2c=prompt, typed=True, rt=1.1,
                      extra=[GrowArrow(need_arrow), FadeIn(need, shift=0.1 * DOWN)])
            term.push("Cloning into 'clankers'...", MUTED, rt=0.3)
            vo.wait_until(at(vo, 2, "so run them"))
            self.play(FadeIn(ok_row, shift=0.1 * RIGHT), run_time=0.6)
            vo.wait_until(at(vo, 2, "not a downloaded"))
            self.play(FadeIn(bad_row, shift=0.1 * RIGHT), run_time=0.6)
            self.play(Indicate(ok_row[0], color=GOOD), run_time=0.8)
