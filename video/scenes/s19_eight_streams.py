"""Chapter 19 — eight streams: the gate forgets the context (report §11, Table 7, Table 8; screens S38–S42).

Numbers come from video/data/report_tables.json (Table 7, Table 8 and the §11 text), from
video/data/critic/s39_gate_memory.json (screen S39, all eleven measurement points: input term at stream tokens,
saturation, gate entropy) and from video/data/critic/s42_spectrum.json (screen S42: rho(W_h) and the stream's
decodability every 50 updates for each of its ten runs). The phase-plane picture of N19.5 is a 2-D toy computed
below (h <- tanh(u + rho R h)); it is labelled on screen as a sketch, not data.
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403
from common.narration import split_sentences, spoken_form  # noqa: E402
from manimlib.config import manim_config  # noqa: E402

VIDEO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Compile this scene's Tex in a private directory (parallel renders share ManimGL's working.tex otherwise).
manim_config.directories.latex_cache = os.path.join(VIDEO, "build", "tex_work", "s19_eight_streams")
os.makedirs(manim_config.directories.latex_cache, exist_ok=True)


def M(tex: str, size: float = 40, color=INK, **kw) -> Tex:  # noqa: F811  (salted local version)
    return Tex(tex + "{}", font_size=size, fill_color=color, **kw)


def _load(rel):
    with open(os.path.join(VIDEO, "data", rel)) as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------------------ real numbers
_RT = _load("report_tables.json")
T7 = _RT["table7_gate_memory_S39"]
UPD7 = T7["updates"]                                   # 0 200 400 600 1200 2400
_B8M, _B8A, _B4M = T7["blocks"]
DEC_M8, REC_M8, IN_M8 = _B8M["decodability"], _B8M["recurrent_term"], _B8M["input_term"]
DEC_A8, REC_A8 = _B8A["decodability"], _B8A["recurrent_term"]
DEC_M4 = _B4M["decodability"]
assert UPD7 == [0, 200, 400, 600, 1200, 2400]
assert DEC_M8[0] == DEC_A8[0] == 0.91 and DEC_M8[-1] == 0.13 and DEC_A8[-1] == 0.14
assert REC_M8[2] == 0.2 and IN_M8[2] == 0.1 and REC_M8[-1] == 6.93          # "by update 400 (0.20 against 0.10)"
assert min(DEC_M4) == 0.61 and DEC_M4[-1] == 0.83 and _B4M["chance"] == 0.25
_T8 = {r["setting"]: r for r in _RT["table8_where_it_stands"]["rows"]}
PG8 = _T8["S=8, P=4, k=16, conv."]["perfect_gate"].split(";")[0].strip()    # "2/2" (from scratch)
assert PG8 == "2/2"
_TXT = T7["text"]
assert _TXT["S38_decodability_4800"].startswith("0.13 (Muon), 0.14 (Adam) at S=8; 0.84 (k=4) and 1.00 (k=16)")
assert _TXT["gate_entropy_8_streams_muon_2400"] == "2.70 of a possible 2.77"

_S39 = _load("critic/s39_gate_memory.json")["median"]
U39 = _S39["D8_HINGE_M"]["updates"]
WIN_S4 = _S39["A_HINGE_M"]["win_stream"]               # input term at stream tokens, 4 streams (Muon)
WIN_S8 = _S39["D8_HINGE_M"]["win_stream"]              # same, 8 streams (Muon)
assert round(WIN_S4[0], 2) == 0.06 and round(WIN_S4[-1], 2) == 0.55       # §11: "from 0.06 to 0.55"
assert max(_S39[c]["sat_key"][-1] for c in ("D8_HINGE_M", "D8_SLOW_M", "HINGE_D8_A")) <= 0.155   # "at most 15%"
ENTROPY = _S39["D8_HINGE_M"]["entropy_key"][-1]
assert round(ENTROPY, 2) == 2.70 and abs(np.log(16) - 2.77) < 0.01

_S42 = _load("critic/s42_spectrum.json")["runs"]
U42 = _S42["D8_HINGE_M|260"]["updates"]                # 0, 50, ..., 1600
assert U42 == list(range(0, 1601, 50))
EXAMPLE = "D8_HINGE_M|260"


def _first_cross(r):
    return next(u for u, x in zip(U42, r["rho"]) if x > 1)


CROSSED = [k for k, r in _S42.items() if max(r["rho"]) > 1.1]                # "clearly above one"
NEAR_ONE = [k for k, r in _S42.items() if max(r["rho"]) <= 1.1]
assert len(CROSSED) == 7 and len(NEAR_ONE) == 3
assert round(min(max(_S42[k]["rho"]) for k in CROSSED), 2) == 1.20
assert round(max(max(_S42[k]["rho"]) for k in CROSSED), 2) == 2.13
assert min(_first_cross(_S42[k]) for k in CROSSED) == 350 and max(_first_cross(_S42[k]) for k in CROSSED) == 1400
# decodability at the first logged update with rho > 1 (within 50 updates of the crossing): 0.13-0.26
_DEC_AT_CROSS = [round(_S42[k]["dec"][U42.index(_first_cross(_S42[k]))], 2) for k in CROSSED]
assert min(_DEC_AT_CROSS) == 0.13 and max(_DEC_AT_CROSS) == 0.26, _DEC_AT_CROSS
assert round(min(max(_S42[k]["rho"]) for k in NEAR_ONE), 2) == 0.88
assert round(max(max(_S42[k]["rho"]) for k in NEAR_ONE), 2) == 1.03
assert sorted(round(_S42[k]["dec"][-1], 2) for k in NEAR_ONE) == [0.31, 0.55, 0.72]
assert round(min(r["rho"][0] for r in _S42.values()), 2) == 0.52
assert round(max(r["rho"][0] for r in _S42.values()), 2) == 0.63
# the two runs whose rho fell back below one, and the update from which it stays below one
FELL_BACK = {"D8_HINGE_M|261": 1400, "D8_HINGE_M|264": 1150}
for _k, _u in FELL_BACK.items():
    assert all(x < 1 for x in _S42[_k]["rho"][U42.index(_u):])
assert round(_S42["D8_HINGE_M|261"]["dec"][U42.index(1400)], 2) == 0.50
assert round(_S42["D8_HINGE_M|264"]["dec"][U42.index(1300)], 2) == 0.69
assert _first_cross(_S42[EXAMPLE]) == 500 and round(_S42[EXAMPLE]["dec"][U42.index(500)], 2) == 0.21

IN_COLOR = MEMORY_COLOR      # the gate's input term (teal), as in Ch. 3
REC_COLOR = GATE_COLOR       # the gate's recurrent term (purple), as in Ch. 3
S4_COLOR = GOOD              # four streams, where the memory survives


# ------------------------------------------------------------------------------------------ 2-D toy (N19.5)
TOY_A, TOY_B, TOY_C, TOY_TH = 0.8, 0.3, 0.3, 1.1     # stream / key / value input sizes, rotation of W_h
RHO_IN, RHO_EX = 0.6, 1.5


def toy_sequence(n_trip, seed):
    """Triples [CTX_s, K2, V_v] of one key's group: only the stream token differs in its input direction."""
    rng = np.random.default_rng(seed)
    seq = []
    for _ in range(n_trip):
        s, v = int(rng.integers(8)), int(rng.integers(16))
        seq += [("ctx", s, s), ("key", 2, s), ("val", v, s)]
    return seq


def toy_input(kind, j):
    if kind == "ctx":
        ang, m = 2 * PI * j / 8, TOY_A
    elif kind == "key":
        ang, m = 0.5, TOY_B
    else:
        ang, m = 2 * PI * j / 16 + 0.3, TOY_C
    return m * np.array([np.cos(ang), np.sin(ang)])


def toy_run(rho, seq, h0):
    c, s = np.cos(TOY_TH), np.sin(TOY_TH)
    W = rho * np.array([[c, -s], [s, c]])
    h = np.array(h0, dtype=float)
    H, IN, REC = [h.copy()], [], []
    for kind, j, _ in seq:
        u, r = toy_input(kind, j), W @ h
        IN.append(float(np.linalg.norm(u)))
        REC.append(float(np.linalg.norm(r)))
        h = np.tanh(u + r)
        H.append(h.copy())
    return np.array(H), np.array(IN), np.array(REC)


def entropy_probs(target, n=16, seed=4):
    """A distribution over n channels with the given entropy (nats), for the near-uniform gate picture."""
    z = np.random.default_rng(seed).normal(size=n)
    lo, hi = 0.0, 5.0
    for _ in range(60):
        b = (lo + hi) / 2
        p = np.exp(b * z) / np.exp(b * z).sum()
        if -(p * np.log(p)).sum() > target:
            lo = b
        else:
            hi = b
    return p


# ------------------------------------------------------------------------------------------ helpers
class Cues:
    """Sub-sentence timing: the time at which a phrase starts, by its share of the spoken sentence."""

    def __init__(self, vo, text, spoken=None):
        self.vo = vo
        self.sp = split_sentences(spoken) if spoken else [spoken_form(s) for s in split_sentences(text)]
        assert len(self.sp) == len(vo.synth.sentences), (len(self.sp), len(vo.synth.sentences))

    def at(self, i, phrase):
        s = self.sp[i]
        k = s.find(phrase)
        assert k >= 0, (phrase, s)
        _, a, b = self.vo.synth.sentences[i]
        return a + (b - a) * k / len(s)

    def wait_for(self, i, phrase, offset=0.0):
        self.vo.wait_until(self.at(i, phrase) + offset)


def cross_mark(size=0.2, color=BAD, width=3):
    return VGroup(Line(UL, DR), Line(UR, DL)).set_width(size).set_stroke(color, width)


def check_mark(size=0.8, color=GOOD, width=7):
    m = VMobject().set_points_as_corners([np.array([-0.5, 0.05, 0]), np.array([-0.15, -0.35, 0]),
                                          np.array([0.55, 0.5, 0])])
    return m.set_width(size).set_stroke(color, width)


def h_grid(seed, cell=0.085, lo=0.12, hi=0.85):
    rng = np.random.default_rng(seed)
    g = VGroup(*[Square(cell).set_stroke(GATE_COLOR, 0.6, opacity=0.7).set_fill(GATE_COLOR, lo + (hi - lo) * rng.random())
                 for _ in range(32)])
    return g.arrange_in_grid(4, 8, buff=0.015)


def axes_at(origin, w, h, x_range, y_range, x_ticks=True):
    ax = Axes(x_range=x_range, y_range=y_range, width=w, height=h,
              axis_config=dict(stroke_color=MUTED, stroke_width=2, include_tip=False, tick_size=0.05),
              x_axis_config=dict(include_ticks=x_ticks))
    ax.shift(np.array(origin) - ax.c2p(x_range[0], y_range[0]))
    return ax


def x_labels(ax, xs, y0, size=20):
    return VGroup(*[L(f"{x:g}", size, MUTED).next_to(ax.c2p(x, y0), DOWN, buff=0.12) for x in xs])


def y_labels(ax, ys, x0, texts=None, size=20):
    texts = texts or [f"{y:g}" for y in ys]
    return VGroup(*[L(t, size, MUTED).next_to(ax.c2p(x0, y), LEFT, buff=0.12) for y, t in zip(ys, texts)])


def chart_line(ax, xs, ys, color, width=3.5, step=50, log=False):
    """Polyline through (x, y) with intermediate points every `step` in x, so ShowCreation runs evenly in x."""
    yv = [np.log10(y) for y in ys] if log else list(ys)
    pts = []
    for (x0, y0), (x1, y1) in zip(zip(xs, yv), zip(xs[1:], yv[1:])):
        n = max(1, int(round((x1 - x0) / step)))
        for k in range(n):
            f = k / n
            pts.append(ax.c2p(x0 + f * (x1 - x0), y0 + f * (y1 - y0)))
    pts.append(ax.c2p(xs[-1], yv[-1]))
    return VMobject().set_points_as_corners(pts).set_stroke(color, width)


def chart_dots(ax, xs, ys, color, r=0.055, log=False):
    return VGroup(*[Dot(ax.c2p(x, np.log10(y) if log else y), radius=r).set_fill(color, 1).set_stroke(BG, 1)
                    for x, y in zip(xs, ys)])


def seg(ax, xs, ys, i0, i1, color, width=3.0, opacity=1.0):
    pts = [ax.c2p(x, y) for x, y in zip(xs[i0:i1 + 1], ys[i0:i1 + 1])]
    return VMobject().set_points_as_corners(pts).set_stroke(color, width, opacity=opacity)


def swatch(color, dashed=False, w=0.38):
    ln = Line(LEFT * w / 2, RIGHT * w / 2).set_stroke(color, 4)
    return DashedLine(LEFT * w / 2, RIGHT * w / 2, dash_length=0.07).set_stroke(color, 3) if dashed else ln


def legend_row(color, text, dashed=False, size=20):
    return VGroup(swatch(color, dashed), L(text, size, color)).arrange(RIGHT, buff=0.12)


RHO_MAX = 2.0


def rho_angle(rho):
    return PI * (1 - min(max(rho, 0), RHO_MAX) / RHO_MAX)


def make_dial(center, r=1.35, size=22):
    """A semicircular gauge for the spectral radius: 0 left, 1 at the top, 2 right."""
    c = np.array(center)
    lo = Arc(start_angle=PI / 2, angle=PI / 2, radius=r, arc_center=c).set_stroke(GOOD, 9)
    hi = Arc(start_angle=0, angle=PI / 2, radius=r, arc_center=c).set_stroke(BAD, 9)
    ticks = VGroup()
    for v in (0, 0.5, 1, 1.5, 2):
        a = rho_angle(v)
        e = np.array([np.cos(a), np.sin(a), 0])
        ticks.add(Line(c + (r - 0.16) * e, c + (r + 0.16) * e).set_stroke(INK, 2 if v != 1 else 3))
    nums = VGroup(*[L(t, size - 2, MUTED).move_to(c + (r + 0.42) * np.array([np.cos(rho_angle(v)),
                                                                              np.sin(rho_angle(v)), 0]))
                    for v, t in ((0, "0"), (1, "1"), (2, "2"))])
    crit = DashedLine(c, c + (r + 0.05) * UP, dash_length=0.08).set_stroke(INK, 2, opacity=0.7)
    lab_lo = L("contracting", size - 2, GOOD).move_to(c + (-r, -0.32, 0))
    lab_hi = L("expanding", size - 2, BAD).move_to(c + (r, -0.32, 0))
    hub = Dot(c, radius=0.07).set_fill(INK, 1)
    g = VGroup(lo, hi, ticks, nums, crit, lab_lo, lab_hi, hub)
    g.c, g.r, g.lo, g.hi, g.crit, g.lab_lo, g.lab_hi, g.hub = c, r, lo, hi, crit, lab_lo, lab_hi, hub
    return g


def dial_needle(dial, rho_tracker, color=INK):
    def make():
        a = rho_angle(rho_tracker.get_value())
        e = np.array([np.cos(a), np.sin(a), 0])
        return Line(dial.c, dial.c + 0.86 * dial.r * e).set_stroke(color, 5)
    return always_redraw(make)


class LiveLabel(VMobject):
    """A sans label re-rendered only when its text changes."""

    def __init__(self, fn, size=24, color=INK, anchor=ORIGIN, weight="BOLD"):
        super().__init__()
        self.fn, self.size, self.color_, self.anchor, self.weight = fn, size, color, np.array(anchor), weight
        self.last = None
        self.refresh()
        self.add_updater(lambda m: m.refresh())

    def refresh(self):
        s = self.fn()
        if s != self.last:
            self.last = s
            new = L(s, self.size, self.color_, weight=self.weight).move_to(self.anchor)
            self.set_submobjects([new])
        return self


# ------------------------------------------------------------------------------------------ the scene
class EightStreams(ClankersScene):
    def construct(self):
        card_ = self.chapter_card(19, "The gate forgets the context")
        self.wait(0.6)
        self.title = section_title("Eight streams")
        self.play(FadeOut(card_, shift=0.3 * UP), FadeIn(self.title, shift=0.3 * UP))

        eq = self.part_failures()
        self.part_decoder(eq)
        left = self.part_when_lost()
        self.part_why_not(left)
        dial = self.part_phase_plane()
        self.part_gain_crossing(dial)
        self.part_next()
        self.wait(0.6)
        self.clear_all()

    # ---------------------------------------------------------------------------------- helpers
    def retitle(self, text):
        new = section_title(text)
        return Transform(self.title, new)

    # ---------------------------------------------------------------------------------- N19.1
    def part_failures(self):
        ctx_row = VGroup(*[token(f"CTX{s}", "ctx", s, width=0.82) for s in range(8)]).arrange(RIGHT, buff=0.1)
        ctx_row.move_to([2.75, 3.0, 0])

        cells = VGroup()
        for _ in range(100):
            sq = Square(0.32).set_fill(BAD, 0.08).set_stroke(BAD, 1.2, opacity=0.55)
            cells.add(VGroup(sq, cross_mark(0.17, BAD, 2.5).move_to(sq)))
        cells.arrange_in_grid(5, 20, buff=0.07).move_to([-2.35, -0.35, 0])
        blanks = VGroup(*[Square(0.32).set_fill(PANEL, 1).set_stroke(FAINT, 1).move_to(c) for c in cells])
        more = L("…", 30, BAD).next_to(cells, RIGHT, buff=0.12).align_to(cells, DOWN)
        order = np.random.default_rng(19).permutation(100)

        chip_txt = ["Adam", "Muon", "with or without hinge", "splits on plateaus", "restarts"]
        chips = VGroup(*[verdict_badge(t, size=20, color=INK) for t in chip_txt]).arrange(RIGHT, buff=0.15)
        chips.next_to(cells, UP, buff=0.4)
        chip_phrases = ["under Adam", "and Muon", "with and without", "with splits", "with restarts"]

        runs_lab = L("learned-gate runs on all eight streams from step 1", 22, MUTED)
        runs_lab.next_to(cells, DOWN, buff=0.35)
        zero = L("0 of more than 100 bound", 34, BAD, weight="BOLD").next_to(runs_lab, DOWN, buff=0.22)

        pg_head = L("perfect gate", 26, GOOD, weight="BOLD")
        pg_sub = L("stream s → channel s", 20, MUTED)
        pg_check = check_mark(0.75)
        pg_val = L(f"{PG8} bound", 30, GOOD, weight="BOLD")
        pg = card(VGroup(pg_head, pg_sub, pg_check, pg_val).arrange(DOWN, buff=0.22), buff=0.3)
        pg.move_to([4.75, -0.35, 0])
        src = source_note("Report §11; Table 8 (S = 8, P = 4, k = 16, convolution)")

        text = ("Eight streams. Every learned-gate arm trained on all eight from the start has failed: under Adam and "
                "Muon, with and without the hinge, with splits on plateaus, with restarts. None of more than a hundred "
                "runs bound, while the perfect gate binds. So instead of another fix, the screens probed the mechanism.")
        t_in, t_rec = R"W_{\text{in}}\, v_t", R"W_h\, h_{t-1}"
        eq = M(R"h_t = \tanh(" + t_in + " + " + t_rec + ")", 48, isolate=[t_in, t_rec, "h_t"])
        eq[t_in].set_color(IN_COLOR)
        eq[t_rec].set_color(REC_COLOR)
        eq.move_to(0.4 * DOWN)
        eq.t_in, eq.t_rec = t_in, t_rec
        h_glyph = eq["h_t"][0]
        lens = Circle(radius=0.42).set_stroke(WARN, 4).move_to(h_glyph)
        handle = Line(ORIGIN, 0.6 * (DR / np.sqrt(2))).set_stroke(WARN, 6)
        handle.shift(lens.get_center() + 0.42 * DR / np.sqrt(2) - handle.get_start())
        probe = VGroup(lens, handle)
        probe_lab = L("the gate's state h", 24, WARN).next_to(lens, DOWN, buff=0.3)
        gate_lab = L("the gate (Ch. 3)", 22, MUTED).next_to(eq, UP, buff=0.5)

        with self.voiceover(text) as vo:
            cue = Cues(vo, text)
            self.play(LaggedStartMap(FadeIn, ctx_row, shift=0.2 * DOWN, lag_ratio=0.08), run_time=1.0)
            vo.wait_until_sentence(1)
            self.play(FadeIn(src), LaggedStartMap(FadeIn, blanks, scale=0.7, lag_ratio=0.01), run_time=1.6)
            k = 0
            chunk = [20, 20, 25, 20, 15]
            for i, ph in enumerate(chip_phrases):
                cue.wait_for(1, ph)
                nxt = cue.at(1, chip_phrases[i + 1]) if i + 1 < len(chip_phrases) else vo.time_of(2)
                rt = max(0.6, min(1.6, nxt - vo.elapsed() - 0.05))
                idx = order[k:k + chunk[i]]
                k += chunk[i]
                self.play(FadeIn(chips[i], shift=0.15 * DOWN),
                          LaggedStart(*[FadeIn(cells[j], scale=0.6) for j in idx], lag_ratio=0.04),
                          *([FadeIn(more)] if i == len(chip_phrases) - 1 else []), run_time=rt)
            vo.wait_until_sentence(2)
            self.play(FadeIn(runs_lab, shift=0.1 * UP), Write(zero), run_time=1.2)
            cue.wait_for(2, "while the perfect gate")
            self.play(FadeIn(pg[0]), FadeIn(pg_head), FadeIn(pg_sub), run_time=0.5)
            self.play(ShowCreation(pg_check), FadeIn(pg_val, shift=0.1 * UP), run_time=0.7)
            self.play(Indicate(pg_val, color=GOOD, scale_factor=1.15), run_time=0.8)
            vo.wait_until_sentence(3)
            self.play(LaggedStart(*[Indicate(c, color=WARN, scale_factor=1.1) for c in chips[2:]], lag_ratio=0.25),
                      run_time=1.0)
            cue.wait_for(3, "the screens probed", -0.3)
            self.play(FadeOut(VGroup(blanks, cells, more, chips, runs_lab, zero, pg), shift=0.4 * DOWN),
                      FadeOut(src), run_time=0.7)
            self.play(FadeIn(gate_lab), Write(eq), run_time=0.9)
            self.play(ShowCreation(lens), ShowCreation(handle), FadeIn(probe_lab, shift=0.1 * UP), run_time=0.7)
        self.ctx_row = ctx_row
        self.play(FadeOut(VGroup(probe, probe_lab, gate_lab)), run_time=0.4)
        return eq

    # ---------------------------------------------------------------------------------- N19.2
    def part_decoder(self, eq):
        seqb = [(5, 9), (1, 4), (6, 13)]
        trips = VGroup(*[triple(s, 2, v) for s, v in seqb]).arrange(RIGHT, buff=0.32)
        ell_l, ell_r = L("…", 30, MUTED), L("…", 30, MUTED)
        row = VGroup(ell_l, trips, ell_r).arrange(RIGHT, buff=0.22).move_to([-0.75, 1.95, 0])
        toks = [t for tr in trips for t in tr]
        grids = VGroup(*[h_grid(40 + i).move_to([t.get_x(), 1.12, 0]) for i, t in enumerate(toks)])
        h_lab = M("h", 36, GATE_COLOR).next_to(grids[0], LEFT, buff=0.32)
        g_lab = L("gate state, 32 units, at every token", 20, GATE_COLOR)
        g_lab.next_to(grids, DOWN, buff=0.16).align_to(grids, LEFT)
        key_idx = [1, 4, 7]
        key_hl = VGroup(*[SurroundingRectangle(VGroup(toks[i], grids[i]), buff=0.07).set_stroke(WARN, 2)
                          for i in key_idx])

        dec_box = RoundedRectangle(width=2.25, height=1.05, corner_radius=0.12).set_fill(PANEL, 1).set_stroke(WARN, 2)
        dec_box.move_to([5.55, 0.75, 0])
        dec_t = L("decoder", 24, WARN, weight="BOLD").move_to(dec_box).shift(0.17 * UP)
        dec_s = L("nearest class mean", 20, MUTED).next_to(dec_t, DOWN, buff=0.08)
        dec = VGroup(dec_box, dec_t, dec_s)
        dec_q = L("which stream?", 22, INK).next_to(dec_box, DOWN, buff=0.45)
        dec_arrow = Arrow(dec_box.get_bottom(), dec_q.get_top(), buff=0.06, thickness=3).set_color(INK)

        # bar chart
        ax = axes_at([-5.2, -2.85, 0], 7.6, 2.5, [0, 8, 1], [0, 1, 0.5], x_ticks=False)
        ylab = y_labels(ax, [0, 0.5, 1], 0)
        yname = L("stream read from h", 20, MUTED).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.62)
        bw = 0.72

        def bar(x, v, color):
            p0, p1 = ax.c2p(x, 0), ax.c2p(x, v)
            r = Rectangle(width=bw, height=max(p1[1] - p0[1], 1e-3)).set_fill(color, 0.85).set_stroke(width=0)
            r.move_to(p0, aligned_edge=DOWN)
            val = L(f"{v:.2f}", 24, color, weight="BOLD").next_to(r, UP, buff=0.1)
            return VGroup(r, val)

        b_m8, b_a8 = bar(1.0, 0.13, MUON_COLOR), bar(2.2, 0.14, ADAM_COLOR)
        b_k4, b_k16 = bar(5.1, 0.84, S4_COLOR), bar(6.3, 1.00, S4_COLOR)
        b_k4[0].set_fill(S4_COLOR, 0.55)
        subl = VGroup(L("Muon", 20, MUON_COLOR).next_to(ax.c2p(1.0, 0), DOWN, buff=0.12),
                      L("Adam", 20, ADAM_COLOR).next_to(ax.c2p(2.2, 0), DOWN, buff=0.12),
                      L("k = 4", 20, S4_COLOR).next_to(ax.c2p(5.1, 0), DOWN, buff=0.12),
                      L("k = 16", 20, S4_COLOR).next_to(ax.c2p(6.3, 0), DOWN, buff=0.12))
        grp8 = L("8 streams, k = 16, update 4800", 20, INK).next_to(subl[:2], DOWN, buff=0.1)
        grp4 = L("4 streams", 20, INK).next_to(subl[2:], DOWN, buff=0.1)
        ch8 = DashedLine(ax.c2p(0.3, 0.125), ax.c2p(2.9, 0.125), dash_length=0.08).set_stroke(INK, 2)
        ch8_l = L("chance 1/8", 20, INK).next_to(ch8, RIGHT, buff=0.15)
        ch4 = DashedLine(ax.c2p(4.4, 0.25), ax.c2p(7.0, 0.25), dash_length=0.08).set_stroke(INK, 2)
        ch4_l = L("chance 1/4", 20, INK).next_to(ch4, RIGHT, buff=0.15)
        src = source_note("Report §11 (screen S38): 8 streams, medians at update 4800")

        mem_arrow = CurvedArrow(toks[7].get_top() + 0.05 * UP, toks[6].get_top() + 0.05 * UP, angle=PI * 0.75)
        mem_arrow.set_stroke(WARN, 3)
        mem_lab = L("one position back", 22, WARN).next_to(mem_arrow, UP, buff=0.12)
        need = L("needs 1 token of memory", 24, WARN, weight="BOLD")
        need.move_to([5.0, 2.95, 0])
        lost = cross_mark(0.45, BAD, 6).move_to(mem_arrow.get_center() + 0.1 * UP)
        lost8 = SurroundingRectangle(VGroup(b_m8, b_a8, subl[:2]), buff=0.12).set_stroke(BAD, 2.5)

        text = ("Does the gate's state even carry the stream? A simple decoder, reading the gate's hidden state at "
                "key positions after forty-eight hundred updates, read the stream at only 0.13 under Muon and 0.14 "
                "under Adam, against a chance level of 0.125. At four streams the same decoder read 0.84 with four "
                "channels, and 1.00 with sixteen. That is striking, because each key's stream token sits just one "
                "position earlier. The gate needs one token of memory, and it doesn't have it.")
        spoken = ("Does the gate's state even carry the stream? A simple decoder, reading the gate's hidden state at "
                  "key positions after forty-eight hundred updates, read the stream at only zero point one three under "
                  "Muon and zero point one four under Adam, against a chance level of zero point one two five. At four "
                  "streams the same decoder read zero point eight four with four channels, and one point zero zero "
                  "with sixteen. That is striking, because each key's stream token sits just one position earlier. "
                  "The gate needs one token of memory, and it doesn't have it.")
        with self.voiceover(text, spoken=spoken) as vo:
            cue = Cues(vo, text, spoken)
            h_glyph = eq["h_t"][0]
            moving = {id(m) for m in h_glyph.get_family()}
            rest = VGroup(*[m for m in eq.submobjects if id(m) not in moving])
            ctx_targets = [toks[0], toks[3], toks[6]]
            self.play(self.retitle("Does the state carry the stream?"),
                      FadeOut(rest, shift=0.3 * DOWN), ReplacementTransform(h_glyph, h_lab),
                      *[ReplacementTransform(self.ctx_row[s].copy(), ctx_targets[i]) for i, (s, _) in enumerate(seqb)],
                      FadeOut(self.ctx_row), run_time=1.2)
            self.play(LaggedStart(*[FadeIn(t, shift=0.15 * DOWN) for i, t in enumerate(toks) if i not in (0, 3, 6)],
                                  lag_ratio=0.1), FadeIn(ell_l), FadeIn(ell_r),
                      LaggedStartMap(FadeIn, grids, lag_ratio=0.08), FadeIn(g_lab), run_time=1.4)
            vo.wait_until_sentence(1)
            self.play(FadeIn(dec, shift=0.2 * LEFT), ShowCreation(key_hl), run_time=0.8)
            copies = VGroup(*[grids[i].copy() for i in key_idx])
            self.play(*[c.animate.scale(0.45).move_to(dec_box.get_center() + (j - 1) * 0.42 * RIGHT + 0.0 * UP)
                        .set_opacity(0) for j, c in enumerate(copies)], run_time=1.0)
            self.remove(copies)
            self.play(FlashAround(dec_box, color=WARN), Indicate(dec_t, color=WARN), GrowArrow(dec_arrow),
                      FadeIn(dec_q), run_time=0.9)
            cue.wait_for(1, "read the stream at only", -0.6)
            self.play(FadeIn(ax), FadeIn(ylab), FadeIn(yname), FadeIn(src), FadeIn(grp8), run_time=0.6)
            cue.wait_for(1, "zero point one three")
            self.play(GrowFromEdge(b_m8[0], DOWN), FadeIn(b_m8[1]), FadeIn(subl[0]), run_time=0.7)
            cue.wait_for(1, "zero point one four")
            self.play(GrowFromEdge(b_a8[0], DOWN), FadeIn(b_a8[1]), FadeIn(subl[1]), run_time=0.7)
            cue.wait_for(1, "against a chance")
            self.play(ShowCreation(ch8), FadeIn(ch8_l), run_time=0.7)
            self.play(Indicate(VGroup(b_m8[1], b_a8[1]), color=BAD, scale_factor=1.15), run_time=1.0)
            vo.wait_until_sentence(2)
            self.play(FadeIn(grp4, shift=0.1 * UP), ShowCreation(ch4), FadeIn(ch4_l), run_time=0.9)
            cue.wait_for(2, "zero point eight four")
            self.play(GrowFromEdge(b_k4[0], DOWN), FadeIn(b_k4[1]), FadeIn(subl[2]), run_time=0.7)
            cue.wait_for(2, "one point zero zero")
            self.play(GrowFromEdge(b_k16[0], DOWN), FadeIn(b_k16[1]), FadeIn(subl[3]), run_time=0.8)
            vo.wait_until_sentence(3)
            self.play(FadeOut(key_hl), Indicate(toks[7], color=WARN), run_time=0.7)
            self.play(ShowCreation(mem_arrow), FadeIn(mem_lab, shift=0.1 * DOWN), run_time=0.9)
            self.play(Indicate(toks[6], color=STREAM_COLORS[6], scale_factor=1.15), run_time=0.8)
            vo.wait_until_sentence(4)
            self.play(Write(need), run_time=0.9)
            cue.wait_for(4, "and it doesn't")
            self.play(ShowCreation(lost), ShowCreation(lost8), run_time=0.8)
        self.wait(0.3)
        self.play(FadeOut(VGroup(row, grids, h_lab, g_lab, dec, dec_arrow, dec_q, ax, ylab, yname, b_m8, b_a8,
                                 b_k4, b_k16, subl, grp8, grp4, ch8, ch8_l, ch4, ch4_l, mem_arrow, mem_lab, need,
                                 lost, lost8, src)), run_time=0.7)

    # ---------------------------------------------------------------------------------- N19.3
    def part_when_lost(self):
        axL = axes_at([-5.45, -2.55, 0], 5.2, 3.9, [0, 2400, 200], [0, 1, 0.25])
        axR = axes_at([1.3, -2.55, 0], 5.2, 3.9, [0, 2400, 200], [-1.5, 1.0, 0.5])
        xticks = [0, 600, 1200, 2400]
        xlL, xlR = x_labels(axL, xticks, 0), x_labels(axR, xticks, -1.5)
        ylL = y_labels(axL, [0, 0.5, 1], 0)
        ylR = y_labels(axR, [-1, 0, 1], 0, texts=["0.1", "1", "10"])
        xnL = L("updates", 20, MUTED).next_to(xlL, DOWN, buff=0.1)
        xnR = L("updates", 20, MUTED).next_to(xlR, DOWN, buff=0.1)
        hdL = L("stream read from h at keys", 22, INK).next_to(axL.c2p(1200, 1), UP, buff=0.3)
        hdR = VGroup(L("size of the two terms at keys", 22, INK), L("(log scale)", 20, MUTED)).arrange(RIGHT, buff=0.15)
        hdR.next_to(axR.c2p(1200, 1.0), UP, buff=0.3)

        l_m8 = chart_line(axL, UPD7, DEC_M8, MUON_COLOR)
        l_a8 = chart_line(axL, UPD7, DEC_A8, ADAM_COLOR)
        d_m8, d_a8 = chart_dots(axL, UPD7, DEC_M8, MUON_COLOR), chart_dots(axL, UPD7, DEC_A8, ADAM_COLOR)
        chance = DashedLine(axL.c2p(0, 0.125), axL.c2p(2400, 0.125), dash_length=0.08).set_stroke(INK, 2, opacity=0.8)
        chance_l = L("chance 1/8", 20, INK).move_to(axL.c2p(1900, 0.06))
        leg = VGroup(legend_row(MUON_COLOR, "Muon, 8 streams"), legend_row(ADAM_COLOR, "Adam, 8 streams"))
        leg.arrange(DOWN, buff=0.12, aligned_edge=LEFT).move_to(axL.c2p(1850, 0.70), aligned_edge=UP)
        init_dot = Dot(axL.c2p(0, 0.91), radius=0.09).set_fill(INK, 1)
        init_lab = L("0.91 at initialization", 22, INK, weight="BOLD").next_to(axL.c2p(0, 0.91), RIGHT, buff=0.25)
        init_lab.shift(0.22 * UP)
        band = Rectangle(width=axL.c2p(1200, 0)[0] - axL.c2p(600, 0)[0], height=axL.c2p(0, 1)[1] - axL.c2p(0, 0)[1])
        band.set_fill(WARN, 0.10).set_stroke(width=0).move_to(axL.c2p(900, 0.5))
        band_l = L("600–1200", 20, WARN).move_to(axL.c2p(900, 0.05))
        src = source_note("Report §11, Table 7 (screen S39: medians of five reruns)")

        lg = np.log10
        r_m8 = chart_line(axR, UPD7, REC_M8, REC_COLOR, log=True)
        i_m8 = chart_line(axR, UPD7, IN_M8, IN_COLOR, log=True)
        r_a8 = DashedVMobject(chart_line(axR, UPD7, REC_A8, REC_COLOR, width=2.5, log=True), num_dashes=45)
        d_r = chart_dots(axR, UPD7, REC_M8, REC_COLOR, log=True)
        d_i = chart_dots(axR, UPD7, IN_M8, IN_COLOR, log=True)
        lab_r = VGroup(L("recurrent term", 20, REC_COLOR), M(R"|W_h h_{t-1}|", 26, REC_COLOR)).arrange(DOWN, buff=0.08)
        lab_r.move_to(axR.c2p(430, 0.72))
        lab_i = VGroup(L("input term", 20, IN_COLOR), M(R"|W_{\text{in}} v_t|", 26, IN_COLOR)).arrange(RIGHT, buff=0.1)
        lab_i.move_to(axR.c2p(1800, -0.62))
        lab_a = L("Adam", 20, REC_COLOR).move_to(axR.c2p(2150, 0.2))
        lab_m = L("Muon", 20, REC_COLOR).move_to(axR.c2p(1500, 0.86))

        g400 = VGroup(DashedLine(axL.c2p(400, 0), axL.c2p(400, 0.82), dash_length=0.07),
                      DashedLine(axR.c2p(400, -1.5), axR.c2p(400, -0.25), dash_length=0.07)).set_stroke(WARN, 2)
        hl400 = VGroup(Dot(axR.c2p(400, lg(0.20)), radius=0.1).set_fill(REC_COLOR, 1),
                       Dot(axR.c2p(400, lg(0.10)), radius=0.1).set_fill(IN_COLOR, 1),
                       Dot(axL.c2p(400, 0.39), radius=0.1).set_fill(MUON_COLOR, 1))
        lab400 = VGroup(L("update 400:", 20, WARN), L("0.20", 20, REC_COLOR, weight="BOLD"),
                        L("vs", 20, WARN), L("0.10", 20, IN_COLOR, weight="BOLD")).arrange(RIGHT, buff=0.1)
        lab400.next_to(axR.c2p(400, -1.5), UP, buff=0.12).shift(1.05 * RIGHT)
        end_r = L("6.93", 24, REC_COLOR, weight="BOLD").move_to(axR.c2p(2280, 0.98))
        end_i = L("0.06–0.15", 22, IN_COLOR, weight="BOLD").move_to(axR.c2p(2050, -1.12))

        text = ("When is it lost? At initialization the decoder reads 0.91: with small weights, the state still carries "
                "the previous token. Training removes this memory within the first six hundred to twelve hundred "
                "updates. The loss coincides with the recurrent term overtaking the input term. Under Muon the "
                "recurrent term passes the input term by update four hundred, and by update twenty-four hundred it is "
                "near seven, while the input term stays around 0.1.")
        spoken = ("When is it lost? At initialization the decoder reads zero point nine one: with small weights, the "
                  "state still carries the previous token. Training removes this memory within the first six hundred "
                  "to twelve hundred updates. The loss coincides with the recurrent term overtaking the input term. "
                  "Under Muon the recurrent term passes the input term by update four hundred, and by update "
                  "twenty-four hundred it is near seven, while the input term stays around zero point one.")
        with self.voiceover(text, spoken=spoken) as vo:
            cue = Cues(vo, text, spoken)
            self.play(self.retitle("When is the stream lost?"), FadeIn(VGroup(axL, xlL, ylL, xnL, hdL)), FadeIn(src),
                      run_time=1.0)
            vo.wait_until_sentence(1)
            self.play(FadeIn(init_dot, scale=0.5), ShowCreation(chance), FadeIn(chance_l), run_time=0.8)
            self.play(Write(init_lab), Flash(init_dot.get_center(), color=INK, flash_radius=0.3), run_time=1.0)
            cue.wait_for(1, "the state still carries")
            self.play(Indicate(init_lab, color=WARN, scale_factor=1.05), run_time=1.0)
            vo.wait_until_sentence(2)
            self.play(FadeIn(leg), ShowCreation(l_m8), ShowCreation(l_a8), LaggedStartMap(FadeIn, d_m8, lag_ratio=0.3),
                      LaggedStartMap(FadeIn, d_a8, lag_ratio=0.3), init_lab.animate.set_opacity(0.5),
                      run_time=2.6, rate_func=linear)
            self.play(FadeIn(band), FadeIn(band_l), run_time=0.7)
            vo.wait_until_sentence(3)
            self.play(FadeIn(VGroup(axR, xlR, ylR, xnR, hdR)), run_time=0.6)
            self.play(ShowCreation(r_m8), ShowCreation(i_m8), LaggedStartMap(FadeIn, d_r, lag_ratio=0.3),
                      LaggedStartMap(FadeIn, d_i, lag_ratio=0.3), run_time=2.2, rate_func=linear)
            self.play(FadeIn(lab_r), FadeIn(lab_i), run_time=0.6)
            vo.wait_until_sentence(4)
            self.play(ShowCreation(r_a8), FadeIn(lab_a), FadeIn(lab_m), run_time=1.2)
            cue.wait_for(4, "by update four hundred")
            self.play(ShowCreation(g400), FadeIn(hl400, scale=0.5), FadeIn(lab400, shift=0.1 * UP), run_time=0.9)
            cue.wait_for(4, "twenty-four hundred it is")
            self.play(Indicate(d_r[-1], color=REC_COLOR, scale_factor=2.0), FadeIn(end_r, shift=0.1 * DOWN), run_time=0.9)
            cue.wait_for(4, "while the input")
            self.play(Indicate(d_i[-1], color=IN_COLOR, scale_factor=2.0), FadeIn(end_i, shift=0.1 * UP), run_time=0.9)

        right = VGroup(axR, xlR, ylR, xnR, hdR, r_m8, i_m8, r_a8, d_r, d_i, lab_r, lab_i, lab_a, lab_m, g400[1],
                       hl400[:2], lab400, end_r, end_i)
        left = SimpleNamespace(ax=axL, leg=leg, src=src, extra=VGroup(g400[0], hl400[2], band, band_l),
                                    group=VGroup(axL, xlL, ylL, xnL, hdL, l_m8, l_a8, d_m8, d_a8, chance, chance_l,
                                                 leg, init_dot, init_lab))
        left.right = right
        return left

    # ---------------------------------------------------------------------------------- N19.4
    def part_why_not(self, left):
        axL = left.ax
        lg = np.log10
        # row 1: saturation
        sat_head = L("units saturated (|h| > 0.95) at update 2400", 20, INK)
        sat_track = Rectangle(width=4.2, height=0.3).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1)
        sat_fill = Rectangle(width=4.2 * 0.15, height=0.3).set_fill(REC_COLOR, 0.9).set_stroke(width=0)
        sat_fill.align_to(sat_track, LEFT)
        sat_val = L("≤ 15%", 26, REC_COLOR, weight="BOLD").next_to(sat_track, RIGHT, buff=0.2)
        sat_ends = VGroup(L("0", 20, MUTED).next_to(sat_track, DOWN, buff=0.08).align_to(sat_track, LEFT),
                          L("100%", 20, MUTED).next_to(sat_track, DOWN, buff=0.08).align_to(sat_track, RIGHT))
        sat_bar = VGroup(sat_track, sat_fill)
        sat = VGroup(sat_head, VGroup(sat_bar, sat_val, sat_ends)).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        sat.move_to([3.75, 2.05, 0])
        sat_note = L("median over runs; mostly not saturation", 20, MUTED).next_to(sat, DOWN, buff=0.12)
        sat_note.align_to(sat, LEFT)

        # row 2: input term at stream tokens, 4 vs 8 streams
        axI = axes_at([1.75, -1.4, 0], 3.9, 1.6, [0, 2400, 600], [0, 0.6, 0.2])
        in_head = L("input term at stream tokens (Muon)", 20, INK).next_to(axI.c2p(1200, 0.6), UP, buff=0.2)
        in_y = y_labels(axI, [0, 0.6], 0, texts=["0", "0.6"])
        in_x = x_labels(axI, [0, 2400], 0)
        l4 = chart_line(axI, U39, WIN_S4, S4_COLOR, width=3.5)
        l8 = chart_line(axI, U39, WIN_S8, MUON_COLOR, width=3.5)
        e4 = L(f"{WIN_S4[-1]:.2f}", 22, S4_COLOR, weight="BOLD").next_to(axI.c2p(2400, WIN_S4[-1]), RIGHT, buff=0.12)
        e8 = L(f"{WIN_S8[-1]:.2f}", 22, MUON_COLOR, weight="BOLD").next_to(axI.c2p(2400, WIN_S8[-1]), RIGHT, buff=0.12)
        n4 = L("4 streams", 20, S4_COLOR).move_to(axI.c2p(950, 0.45))
        n8 = L("8 streams", 20, MUON_COLOR).move_to(axI.c2p(1500, 0.05))

        # row 3: gate entropy
        probs = entropy_probs(ENTROPY)
        ent_head = L("gate output, 8 streams, update 2400 (Muon)", 20, INK)
        ent_bar = gate_bar(list(probs), width=4.2, height=0.32)
        ent_val = VGroup(L("entropy 2.70", 22, GATE_COLOR, weight="BOLD"),
                         L("of max 2.77 = uniform over 16", 20, MUTED)).arrange(RIGHT, buff=0.15)
        ent = VGroup(ent_head, ent_bar, ent_val).arrange(DOWN, buff=0.16, aligned_edge=LEFT).move_to([3.95, -2.6, 0])

        # four streams on the left chart
        l_m4 = chart_line(axL, UPD7, DEC_M4, S4_COLOR)
        d_m4 = chart_dots(axL, UPD7, DEC_M4, S4_COLOR)
        leg4 = legend_row(S4_COLOR, "Muon, 4 streams").next_to(left.leg, DOWN, buff=0.12, aligned_edge=LEFT)
        ch4 = DashedLine(axL.c2p(0, 0.25), axL.c2p(2400, 0.25), dash_length=0.08).set_stroke(S4_COLOR, 2, opacity=0.8)
        ch4_l = L("chance 1/4", 20, S4_COLOR).move_to(axL.c2p(1900, 0.305))
        dip = L("0.61", 22, S4_COLOR, weight="BOLD").move_to(axL.c2p(700, 0.56))
        rec = L("0.83", 22, S4_COLOR, weight="BOLD").next_to(axL.c2p(2400, 0.83), UP, buff=0.15).shift(0.1 * LEFT)
        src = source_note("Report §11; S39 series: data/critic/s39_gate_memory.json")

        text = ("It is mostly not saturation of the tanh: at update twenty-four hundred, the median share of saturated "
                "units is at most fifteen percent. At four streams, by contrast, the input term grows and the memory "
                "survives. At eight streams the gate's output stays nearly uniform, so the task gives its input "
                "weights little reason to grow.")
        spoken = ("It is mostly not saturation of the tan H: at update twenty-four hundred, the median share of "
                  "saturated units is at most fifteen percent. At four streams, by contrast, the input term grows and "
                  "the memory survives. At eight streams the gate's output stays nearly uniform, so the task gives its "
                  "input weights little reason to grow.")
        with self.voiceover(text, spoken=spoken) as vo:
            cue = Cues(vo, text, spoken)
            self.play(FadeOut(left.right, shift=0.3 * RIGHT), FadeOut(left.extra), FadeOut(left.src),
                      self.retitle("Why the memory goes"), run_time=0.9)
            self.play(FadeIn(sat_head), FadeIn(sat_track), FadeIn(sat_ends), FadeIn(src), run_time=0.6)
            cue.wait_for(0, "at update twenty-four")
            self.play(GrowFromEdge(sat_fill, LEFT), run_time=1.0)
            cue.wait_for(0, "at most fifteen")
            self.play(FadeIn(sat_val, shift=0.1 * LEFT), run_time=0.6)
            self.play(FadeIn(sat_note), run_time=0.6)
            vo.wait_until_sentence(1)
            self.play(FadeIn(leg4), ShowCreation(ch4), FadeIn(ch4_l), run_time=0.6)
            self.play(ShowCreation(l_m4), LaggedStartMap(FadeIn, d_m4, lag_ratio=0.3), run_time=1.8, rate_func=linear)
            self.play(FadeIn(dip), FadeIn(rec), FadeIn(VGroup(axI, in_head, in_y, in_x)), run_time=0.6)
            self.play(ShowCreation(l4), ShowCreation(l8), FadeIn(n4), FadeIn(n8), run_time=1.4, rate_func=linear)
            self.play(FadeIn(e4), FadeIn(e8), Indicate(rec, color=S4_COLOR), run_time=0.7)
            vo.wait_until_sentence(2)
            self.play(FadeIn(ent_head), LaggedStart(*[GrowFromEdge(s, LEFT) for s in ent_bar[0]], lag_ratio=0.1),
                      ShowCreation(ent_bar[1]), run_time=1.2)
            self.play(FadeIn(ent_val, shift=0.1 * UP), run_time=0.6)
            cue.wait_for(2, "so the task")
            self.play(Indicate(e8, color=MUON_COLOR, scale_factor=1.3), Indicate(n8, color=MUON_COLOR), run_time=1.2)
        self.wait(0.3)
        self.play(FadeOut(VGroup(left.group, sat, sat_note, axI, in_head, in_y, in_x, l4, l8, e4, e8, n4, n8, ent,
                                 l_m4, d_m4, leg4, ch4, ch4_l, dip, rec, src)), run_time=0.8)

    # ---------------------------------------------------------------------------------- N19.5
    def part_phase_plane(self):
        half = 2.3
        pc = np.array([-3.6, -0.6, 0])

        def P(h):
            return pc + half * np.array([h[0], h[1], 0])

        box = Square(2 * half).move_to(pc).set_stroke(FAINT, 1.5)
        axes2 = VGroup(Line(P((-1, 0)), P((1, 0))), Line(P((0, -1)), P((0, 1)))).set_stroke(FAINT, 1.2)
        h1 = M("h_1", 30, MUTED).next_to(P((1, 0)), RIGHT, buff=0.1)
        h2 = M("h_2", 30, MUTED).next_to(P((0, 1)), UP, buff=0.06)
        plane_lab = L("gate state h, drawn in 2-D (a sketch, not data)", 20, MUTED).next_to(box, UP, buff=0.42)
        plane = VGroup(box, axes2, h1, h2)

        dial = make_dial([3.65, 1.05, 0], r=1.3)
        rho = ValueTracker(RHO_IN)
        needle = dial_needle(dial, rho)
        rho_lab = LiveLabel(lambda: f"ρ = {rho.get_value():.2f}", 26, INK, anchor=dial.c + 0.42 * DOWN)
        dial_head = L("spectral radius of the recurrent weights", 20, MUTED).move_to(dial.c + 2.05 * UP)

        # tape of input tokens and the two terms
        cx, ty, pitch = 3.65, -1.25, 0.98
        frame_ = RoundedRectangle(width=1.1, height=0.72, corner_radius=0.12).set_stroke(WARN, 2.5).move_to([cx, ty, 0])
        tape_lab = L("input token", 20, WARN).next_to(frame_, UP, buff=0.1)
        masks = VGroup()          # soft edges of the tape window; tokens farther out are not drawn at all
        for side in (-1, 1):
            for k in range(6):
                w = 0.22
                m = Rectangle(width=w, height=0.8).set_fill(BG, (k + 1) / 6).set_stroke(width=0)
                m.move_to([cx + side * (2.25 + k * w), ty, 0])
                masks.add(m)
            far = Rectangle(width=0.75, height=0.8).set_fill(BG, 1).set_stroke(width=0)
            far.move_to([cx + side * (2.25 + 6 * 0.22 + 0.375 - 0.11), ty, 0])
            masks.add(far)

        bar_x0, bar_scale = 2.55, 1.85
        in_name = L("input term", 20, IN_COLOR).move_to([bar_x0 - 0.15, -2.45, 0], aligned_edge=RIGHT)
        rec_name = L("recurrent term", 20, REC_COLOR).move_to([bar_x0 - 0.15, -2.95, 0], aligned_edge=RIGHT)
        base = Line([bar_x0, -2.2, 0], [bar_x0, -3.2, 0]).set_stroke(MUTED, 2)

        def build_run(rho_v, seq, h0, n_anim, seed_tokens):
            H, IN, REC = toy_run(rho_v, seq, h0)
            toks = VGroup()
            for kind, j, s in seq[:n_anim]:
                if kind == "ctx":
                    t = token(f"CTX{s}", "ctx", s, width=0.9, height=0.55)
                elif kind == "key":
                    t = token("K2", "key", width=0.9, height=0.55)
                else:
                    t = token(f"V{j}", "val", s, width=0.9, height=0.55)
                toks.add(t)
            toks.arrange(RIGHT, buff=pitch - 0.9).set_y(ty)
            keys = [i for i, (k, _, _) in enumerate(seq) if k == "key"]
            kdots = VGroup(*[Dot(P(H[i + 1]), radius=0.065).set_fill(STREAM_COLORS[seq[i][2]], 1).set_stroke(BG, 0.5)
                             for i in keys])
            win = VGroup()        # the tokens currently inside the tape window (toks itself is never drawn)
            return SimpleNamespace(H=H, IN=IN, REC=REC, toks=toks, win=win, keys=keys, kdots=kdots, n_anim=n_anim)

        seq_in = toy_sequence(64, 7)
        run_in = build_run(RHO_IN, seq_in, (0, 0), len(seq_in), 0)
        seq_ex = toy_sequence(64, 8)
        run_ex = build_run(RHO_EX, seq_ex, run_in.H[-1], len(seq_ex), 1)

        t = ValueTracker(0.0)
        state = {"run": run_in}

        def cur_pos():
            r = state["run"]
            v = t.get_value()
            i = int(np.clip(np.floor(v), 0, len(r.H) - 2))
            f = smooth(np.clip(v - i, 0, 1))
            return P(r.H[i] + f * (r.H[i + 1] - r.H[i]))

        dot = Dot(P((0, 0)), radius=0.1).set_fill(INK, 1).set_stroke(BG, 1.5)
        dot.add_updater(lambda m: m.move_to(cur_pos()))

        def trail_pts():
            r = state["run"]
            v = t.get_value()
            i = int(np.clip(np.floor(v), 0, len(r.H) - 2))
            pts = [P(r.H[j]) for j in range(max(0, i - 2), i + 1)] + [cur_pos()]
            return pts

        trail = VMobject().set_stroke(INK, 2, opacity=0.45)
        trail.set_points_as_corners(trail_pts())
        trail.add_updater(lambda m: m.set_points_as_corners(trail_pts()))

        def show_kdots(m):
            r = state["run"]
            v = t.get_value()
            for d, i in zip(m, r.keys):
                vis = v >= i + 1 - 1e-6
                if getattr(d, "_vis", None) != vis:
                    d._vis = vis
                    d.set_opacity(1.0 if vis else 0.0)

        def place_tape(m):
            r = state["run"]
            v = t.get_value()
            r.toks.shift((cx + (0.5 - v) * pitch - r.toks[0].get_x()) * RIGHT)
            m.set_submobjects([tk for tk in r.toks if abs(tk.get_x() - cx) < 3.7])

        def bar_of(kind):
            def make():
                r = state["run"]
                v = t.get_value()
                i = int(np.clip(np.floor(v), 0, len(r.IN) - 1))
                val = (r.IN if kind == "in" else r.REC)[i]
                w = max(val * bar_scale, 0.01)
                y = -2.45 if kind == "in" else -2.95
                col = IN_COLOR if kind == "in" else REC_COLOR
                return Rectangle(width=w, height=0.26).set_fill(col, 0.9).set_stroke(width=0).move_to(
                    [bar_x0 + w / 2, y, 0])
            return always_redraw(make)

        in_bar, rec_bar = bar_of("in"), bar_of("rec")

        def attach(run):
            state["run"] = run
            run.kdots.set_opacity(0)
            for d in run.kdots:
                d._vis = False
            run.kdots.add_updater(show_kdots)
            run.win.add_updater(place_tape)
            place_tape(run.win)

        def detach(run):
            run.kdots.clear_updaters()
            run.win.clear_updaters()

        verdict_in = VGroup(L("colors separate:", 22, GOOD, weight="BOLD"),
                            L("the stream can be read at the key", 22, GOOD)).arrange(DOWN, buff=0.08)
        verdict_in.next_to(box, DOWN, buff=0.15)
        init_note = L("weak recurrence, as at initialization: decoder 0.91", 20, INK)
        init_note.next_to(box, UP, buff=0.42)
        verdict_ex = VGroup(L("colors mixed:", 22, BAD, weight="BOLD"),
                            L("the state ignores its input", 22, BAD)).arrange(DOWN, buff=0.08)
        verdict_ex.move_to(verdict_in)
        src = source_note("2-D sketch: h ← tanh(u + ρ·R·h), not data; decoder 0.91: Report Table 7")

        text = ("Here is the report's reading. The gate's state is a small dynamical system driven by its input. If the "
                "recurrent weights contract, old history fades and the state follows the last few tokens, so every "
                "stream token leaves a mark that is still there at the next key. That is why the decoder reads 0.91 "
                "at initialization, when the recurrence is weak. If the weights expand, the state sustains itself and "
                "ignores its input.")
        spoken = ("Here is the report's reading. The gate's state is a small dynamical system driven by its input. If "
                  "the recurrent weights contract, old history fades and the state follows the last few tokens, so "
                  "every stream token leaves a mark that is still there at the next key. That is why the decoder reads "
                  "zero point nine one at initialization, when the recurrence is weak. If the weights expand, the state "
                  "sustains itself and ignores its input.")
        with self.voiceover(text, spoken=spoken) as vo:
            cue = Cues(vo, text, spoken)
            attach(run_in)
            self.add(run_in.kdots)
            self.play(self.retitle("The report's reading"), FadeIn(plane), FadeIn(plane_lab), FadeIn(src),
                      run_time=1.0)
            vo.wait_until_sentence(1)
            self.play(FadeIn(run_in.win), FadeIn(masks), FadeIn(frame_), FadeIn(dot), FadeIn(tape_lab),
                      FadeIn(VGroup(in_name, rec_name, base)), run_time=0.6)
            self.add(trail, in_bar, rec_bar)
            # first triple, slowly, while the sentence names the system
            self.play(t.animate.set_value(3), run_time=2.4, rate_func=linear)
            vo.wait_until_sentence(2)
            self.play(FadeIn(dial), FadeIn(dial_head), FadeIn(needle), FadeIn(rho_lab), run_time=0.8)
            self.play(Indicate(dial.lab_lo, color=GOOD, scale_factor=1.2), run_time=0.8)
            self.play(t.animate.set_value(15), run_time=4.5, rate_func=linear)
            self.play(t.animate.set_value(60), run_time=2.2, rate_func=linear)
            self.play(t.animate.set_value(len(seq_in)), run_time=1.0, rate_func=linear)   # fast-forward
            vo.wait_until_sentence(3)
            self.play(FadeIn(verdict_in, shift=0.1 * UP), FadeOut(plane_lab), FadeIn(init_note), run_time=0.8)
            self.play(LaggedStart(*[Indicate(d, scale_factor=1.6) for d in run_in.kdots[::3]], lag_ratio=0.05),
                      run_time=1.6)
            vo.wait_until_sentence(4)
            detach(run_in)
            self.play(run_in.kdots.animate.set_opacity(0.0), FadeOut(verdict_in), FadeOut(init_note),
                      FadeOut(run_in.win), rho.animate.set_value(RHO_EX), run_time=1.0)
            self.remove(run_in.kdots)
            t.set_value(0)
            attach(run_ex)
            self.add(run_ex.kdots)
            self.remove(masks, frame_)
            self.add(run_ex.win, masks, frame_)
            self.play(Indicate(dial.lab_hi, color=BAD, scale_factor=1.2), t.animate.set_value(6), run_time=2.0,
                      rate_func=linear)
            self.play(t.animate.set_value(36), run_time=2.0, rate_func=linear)
            self.play(t.animate.set_value(len(seq_ex)), run_time=1.0, rate_func=linear)   # fast-forward
        self.play(FadeIn(verdict_ex, shift=0.1 * UP), run_time=0.7)
        self.wait(0.4)
        detach(run_ex)
        for m in (dot, trail):
            m.clear_updaters()
        self.play(FadeOut(VGroup(plane, dot, trail, run_ex.kdots, run_ex.win, masks, frame_, tape_lab, in_name,
                                 rec_name, base, in_bar, rec_bar, verdict_ex, src, needle, rho_lab, dial_head,
                                 dial.lo, dial.hi, dial[2], dial[3], dial.lab_lo, dial.lab_hi, dial.hub)),
                  run_time=0.8)
        return dial

    # ---------------------------------------------------------------------------------- N19.6
    def part_gain_crossing(self, dial):
        axR = axes_at([-5.5, 0.3, 0], 6.4, 2.2, [0, 1600, 100], [0, 2.25, 0.5])
        axD = axes_at([-5.5, -2.75, 0], 6.4, 2.2, [0, 1600, 100], [0, 1, 0.25])
        ylR = y_labels(axR, [0, 1, 2], 0)
        ylD = y_labels(axD, [0, 0.5, 1], 0)
        xlD = x_labels(axD, [0, 400, 800, 1200, 1600], 0)
        xnD = L("updates", 20, MUTED).next_to(xlD, DOWN, buff=0.08)
        nR = VGroup(L("spectral radius", 20, MUTED), M(R"\rho(W_h)", 30, REC_COLOR)).arrange(RIGHT, buff=0.15)
        nR.next_to(axR.c2p(0, 2.25), RIGHT, buff=0.25).shift(0.05 * DOWN)
        nD = L("stream read from h at keys", 20, MUTED).move_to(axD.c2p(0, 1) + 0.42 * UP, aligned_edge=LEFT)
        one = DashedLine(axR.c2p(0, 1), axR.c2p(1600, 1), dash_length=0.09).set_stroke(INK, 2)
        one_l = L("ρ = 1", 20, INK).next_to(axR.c2p(1600, 1), RIGHT, buff=0.1)
        chance = DashedLine(axD.c2p(0, 0.125), axD.c2p(1600, 0.125), dash_length=0.08).set_stroke(INK, 1.5, opacity=0.7)
        chance_l = L("chance", 20, MUTED).next_to(axD.c2p(1600, 0.125), RIGHT, buff=0.1)
        src = source_note("Report §11 (screen S42); series: data/critic/s42_spectrum.json")
        frame_charts = VGroup(axR, axD, ylR, ylD, xlD, xnD, nR, nD)

        runs = _S42
        init_pts = VGroup(*[Dot(axR.c2p(0, r["rho"][0]), radius=0.07).set_fill(REC_COLOR, 1) for r in runs.values()])
        init_l = L("0.52–0.63 at the start", 20, REC_COLOR).move_to(axR.c2p(330, 0.3))

        ex = runs[EXAMPLE]
        ex_parts = [(0, 9), (9, 10), (10, 32)]
        ex_r = [seg(axR, U42, ex["rho"], a, b, INK, 4) for a, b in ex_parts]
        ex_d = [seg(axD, U42, ex["dec"], a, b, INK, 4) for a, b in ex_parts]
        cross_x = 500
        vline = DashedLine(axR.c2p(cross_x, 2.25), axD.c2p(cross_x, 0), dash_length=0.08).set_stroke(WARN, 2)
        cdot_r = Dot(axR.c2p(cross_x, ex["rho"][10]), radius=0.09).set_fill(WARN, 1)
        cdot_d = Dot(axD.c2p(500, ex["dec"][10]), radius=0.09).set_fill(WARN, 1)
        c_lab = L("crosses 1 at update 500", 20, WARN).move_to(axR.c2p(cross_x + 40, 0.42), aligned_edge=LEFT)
        d_lab = L(f"{ex['dec'][10]:.2f} at 500", 20, WARN)
        d_lab.next_to(axD.c2p(500, ex["dec"][10]), RIGHT, buff=0.15).shift(0.3 * UP)

        others = [k for k in CROSSED if k != EXAMPLE]
        o_r = VGroup(*[seg(axR, U42, runs[k]["rho"], 0, 32, BAD, 2, 0.75) for k in others])
        o_d = VGroup(*[seg(axD, U42, runs[k]["dec"], 0, 32, BAD, 2, 0.75) for k in others])
        n_r = VGroup(*[seg(axR, U42, runs[k]["rho"], 0, 32, WARN, 2.5, 0.9) for k in NEAR_ONE])
        n_d = VGroup(*[seg(axD, U42, runs[k]["dec"], 0, 32, WARN, 2.5, 0.9) for k in NEAR_ONE])
        f_r = VGroup(*[seg(axR, U42, runs[k]["rho"], U42.index(u) - 1, 32, GOOD, 3.5) for k, u in FELL_BACK.items()])
        f_d = VGroup(*[seg(axD, U42, runs[k]["dec"], U42.index(u) - 1, 32, GOOD, 3.5) for k, u in FELL_BACK.items()])
        rec_dots = VGroup(Dot(axD.c2p(1400, runs["D8_HINGE_M|261"]["dec"][U42.index(1400)]), radius=0.08),
                          Dot(axD.c2p(1300, runs["D8_HINGE_M|264"]["dec"][U42.index(1300)]), radius=0.08)
                          ).set_fill(GOOD, 1)

        drop_band = Rectangle(width=axD.c2p(1600, 0)[0] - axD.c2p(0, 0)[0],
                              height=axD.c2p(0, 0.26)[1] - axD.c2p(0, 0.13)[1])
        drop_band.set_fill(BAD, 0.18).set_stroke(BAD, 1, opacity=0.5).move_to(axD.c2p(800, 0.195))
        drop_l = L("0.13–0.26", 20, BAD, weight="BOLD").next_to(drop_band, RIGHT, buff=0.1).shift(0.14 * UP)

        # tally (right column)
        def row(color, head, sub, sub2=None):
            d = Dot(radius=0.08).set_fill(color, 1)
            h = L(head, 22, color, weight="BOLD")
            lines = [h, L(sub, 20, INK)] + ([L(sub2, 20, INK)] if sub2 else [])
            body = VGroup(*lines).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
            d.next_to(h, LEFT, buff=0.18)
            return VGroup(d, body)

        x_col = 2.2
        t_head = L("screen S42: 10 runs (5 Muon, 5 Adam)", 20, MUTED)
        t1 = row(BAD, "7 of 10 rose clearly above 1", "peaks 1.20–2.13, crossing at 350–1400",
                 "stream fell to 0.13–0.26 within 50 updates")
        t2 = row(WARN, "3 of 10 peaked at 0.88–1.03", "stream only partly lost: 0.31–0.72")
        t3 = row(GOOD, "2 fell back below 1", "stream partly recovered: 0.50, 0.69")
        rule_h = L("pre-set rule", 22, INK, weight="BOLD")
        rule_m = VGroup(L("Muon:", 20, MUON_COLOR, weight="BOLD"), L("“gain crossing” (4 of 5)", 20, INK)
                        ).arrange(RIGHT, buff=0.12)
        rule_a = VGroup(L("Adam:", 20, ADAM_COLOR, weight="BOLD"), L("“not” (3 of 5)", 20, INK)).arrange(RIGHT, buff=0.12)
        t4 = VGroup(rule_h, rule_m, rule_a).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        tally = VGroup(t_head, t1, t2, t3, t4).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        tally.move_to([x_col, 0.15, 0], aligned_edge=LEFT)
        stamp = verdict_badge("ten runs: a correlation, not a proof", size=22, color=WARN)
        stamp.next_to(tally, DOWN, buff=0.3).align_to(tally, LEFT)

        text = ("A later screen measured exactly this: the spectral radius of the recurrent weights, every fifty "
                "updates. It started between 0.52 and 0.63. In seven of ten runs it rose clearly above one, and in "
                "each of those the stream's decodability fell to between 0.13 and 0.26 within fifty updates of the "
                "crossing. Where it peaked near one, the stream was only partly lost; where it fell back below one, "
                "decodability partly recovered. By the screen's own pre-set rule, the Muon configuration showed the "
                "pattern and the Adam configuration did not. Ten runs give a correlation, not a proof.")
        spoken = ("A later screen measured exactly this: the spectral radius of the recurrent weights, every fifty "
                  "updates. It started between zero point five two and zero point six three. In seven of ten runs it "
                  "rose clearly above one, and in each of those the stream's decodability fell to between zero point "
                  "one three and zero point two six within fifty updates of the crossing. Where it peaked near one, "
                  "the stream was only partly lost; where it fell back below one, decodability partly recovered. By "
                  "the screen's own pre-set rule, the Muon configuration showed the pattern and the Adam configuration "
                  "did not. Ten runs give a correlation, not a proof.")
        with self.voiceover(text, spoken=spoken) as vo:
            cue = Cues(vo, text, spoken)
            self.play(self.retitle("The gain crossing"), FadeIn(frame_charts), FadeIn(src),
                      ReplacementTransform(dial.crit, one), FadeIn(one_l), run_time=1.2)
            self.play(FadeIn(t_head), ShowCreation(chance), FadeIn(chance_l), run_time=0.7)
            cue.wait_for(0, "the spectral radius")
            self.play(FlashAround(nR, color=REC_COLOR), Indicate(nR[1], color=REC_COLOR), run_time=1.0)
            cue.wait_for(0, "every fifty")
            self.play(LaggedStart(*[Indicate(tk, color=REC_COLOR, scale_factor=2.0)
                                    for tk in axR.x_axis.ticks[::1]], lag_ratio=0.03), run_time=1.4)
            vo.wait_until_sentence(1)
            self.play(LaggedStartMap(FadeIn, init_pts, scale=0.5, lag_ratio=0.1), FadeIn(init_l), run_time=1.0)
            cue.wait_for(1, "and zero point six")
            self.play(FlashAround(VGroup(init_pts, init_l), color=REC_COLOR), run_time=1.0)
            vo.wait_until_sentence(2)
            self.play(ShowCreation(ex_r[0]), ShowCreation(ex_d[0]), FadeOut(init_l), run_time=1.6, rate_func=linear)
            self.play(ShowCreation(ex_r[1]), ShowCreation(ex_d[1]), run_time=0.5, rate_func=linear)
            self.play(ShowCreation(vline), FadeIn(cdot_r, scale=0.5), FadeIn(c_lab), run_time=0.6)
            self.play(ShowCreation(ex_r[2]), ShowCreation(ex_d[2]), FadeIn(cdot_d, scale=0.5), FadeIn(d_lab),
                      run_time=1.8, rate_func=linear)
            cue.wait_for(2, "and in each of those")
            self.play(LaggedStartMap(ShowCreation, o_r, lag_ratio=0.15), LaggedStartMap(ShowCreation, o_d, lag_ratio=0.15),
                      FadeIn(t1, shift=0.1 * LEFT), FadeOut(VGroup(c_lab, d_lab)), run_time=2.2)
            cue.wait_for(2, "between zero point")
            self.play(FadeIn(drop_band), FadeIn(drop_l, shift=0.1 * LEFT), run_time=0.7)
            cue.wait_for(2, "within fifty")
            self.play(Indicate(t1[1][2], color=BAD), Indicate(drop_l, color=BAD), run_time=1.0)
            vo.wait_until_sentence(3)
            self.play(LaggedStartMap(ShowCreation, n_r, lag_ratio=0.2), LaggedStartMap(ShowCreation, n_d, lag_ratio=0.2),
                      FadeIn(t2, shift=0.1 * LEFT), run_time=1.6)
            self.play(FadeOut(drop_band), FadeOut(drop_l), run_time=0.4)
            cue.wait_for(3, "where it fell back")
            self.play(ShowCreation(f_r), ShowCreation(f_d), FadeIn(t3, shift=0.1 * LEFT), run_time=1.2)
            self.play(FadeIn(rec_dots, scale=0.5), Indicate(t3[1][1], color=GOOD), run_time=0.8)
            vo.wait_until_sentence(4)
            self.play(FadeIn(rule_h, shift=0.1 * LEFT), run_time=0.4)
            cue.wait_for(4, "the Muon configuration")
            self.play(FadeIn(rule_m, shift=0.1 * LEFT), run_time=0.6)
            cue.wait_for(4, "and the Adam")
            self.play(FadeIn(rule_a, shift=0.1 * LEFT), run_time=0.6)
            vo.wait_until_sentence(5)
            self.play(FadeIn(stamp, scale=1.2), run_time=0.7)
            self.play(Indicate(stamp, color=WARN, scale_factor=1.05), run_time=1.0)
        self.wait(0.3)
        self.play(FadeOut(VGroup(frame_charts, one, one_l, chance, chance_l, init_pts, *ex_r, *ex_d, vline, cdot_r,
                                 cdot_d, o_r, o_d, n_r, n_d, f_r, f_d, rec_dots, tally, stamp, src)), run_time=0.8)

    # ---------------------------------------------------------------------------------- N19.7
    def part_next(self):
        t_in, t_prev, t_rec = R"W_{\text{in}}\, v_t", R"W_{\text{prev}}\, v_{t-1}", R"W_h\, h_{t-1}"
        eq = M(R"h_t = \tanh(" + t_in + " + " + t_prev + " + " + t_rec + ")", 44, isolate=[t_in, t_prev, t_rec])
        eq[t_in].set_color(IN_COLOR)
        eq[t_prev].set_color(WARN)
        eq[t_rec].set_color(REC_COLOR)
        eq.move_to([0, 2.05, 0])
        br = Brace(eq[t_prev], DOWN, buff=0.08).set_color(WARN)
        br_l = L("S40: the previous token, fed in directly", 20, WARN).next_to(br, DOWN, buff=0.08)

        def res(color, a, b):
            return VGroup(L(a, 22, color, weight="BOLD"), L(b, 20, INK)).arrange(DOWN, buff=0.06, aligned_edge=LEFT)

        r1 = res(GOOD, "input decodes the stream: 0.97–1.00", "under Muon: almost perfect")
        r2 = res(WARN, "state h decodes it: 0.53–0.64", "diluted by the recurrent term")
        r3 = res(BAD, "bound: 0/10", "under each optimizer")
        s40 = VGroup(r1, r2, r3).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        s40_card = card(s40, buff=0.3)
        s40_card.move_to([-3.4, -1.1, 0])

        dial = make_dial([3.55, -1.75, 0], r=1.0, size=22)
        cap_zone = AnnularSector(inner_radius=0, outer_radius=1.0, angle=PI / 4, start_angle=3 * PI / 4,
                                 arc_center=dial.c).set_fill(GOOD, 0.25).set_stroke(width=0)
        a5 = rho_angle(0.5)
        cap_line = Line(dial.c, dial.c + 1.2 * np.array([np.cos(a5), np.sin(a5), 0])).set_stroke(WARN, 4)
        cap_lab = L("cap", 20, WARN).next_to(dial.c + 1.2 * np.array([np.cos(a5), np.sin(a5), 0]), UL, buff=0.05)
        rho = ValueTracker(1.6)
        needle = dial_needle(dial, rho)
        s41_h = L("S41: cap the recurrence", 24, INK, weight="BOLD").move_to([3.15, 0.75, 0])
        cap_eq = M(R"\sigma_{\max}(W_h) \le 0.5 \;\Rightarrow\; \rho \le 0.5 < 1", 32, INK)
        cap_eq.next_to(s41_h, DOWN, buff=0.25)
        running = verdict_badge("running", size=22, color=WARN).next_to(s41_h, RIGHT, buff=0.35)
        src = source_note("Report §11 (screens S40, S41)")

        text = ("Feeding the gate the previous token directly did not solve it either: its input then decodes the stream "
                "almost perfectly, but the recurrent term dilutes it, and no run bound. The intervention now running is "
                "a cap on the recurrent weights' spectral norm.")
        with self.voiceover(text) as vo:
            cue = Cues(vo, text)
            self.play(self.retitle("What is being tried"), Write(eq), FadeIn(src), run_time=1.4)
            self.play(GrowFromCenter(br), FadeIn(br_l, shift=0.1 * DOWN), run_time=0.7)
            cue.wait_for(0, "its input then")
            self.play(FadeIn(s40_card[0]), FadeIn(r1, shift=0.1 * RIGHT), Indicate(eq[t_prev], color=GOOD), run_time=0.8)
            cue.wait_for(0, "but the recurrent")
            self.play(FadeIn(r2, shift=0.1 * RIGHT), Indicate(eq[t_rec], color=REC_COLOR), run_time=0.8)
            cue.wait_for(0, "and no run")
            self.play(FadeIn(r3, shift=0.1 * RIGHT), run_time=0.6)
            vo.wait_until_sentence(1)
            self.play(FadeIn(s41_h), FadeIn(dial), FadeIn(needle), run_time=0.8)
            self.play(Write(cap_eq), FadeIn(cap_zone), ShowCreation(cap_line), FadeIn(cap_lab), run_time=0.9)
            self.play(rho.animate.set_value(0.45), run_time=1.2)
            self.play(FadeIn(running, scale=1.2), run_time=0.5)
        needle.clear_updaters()
        self.wait(0.8)

