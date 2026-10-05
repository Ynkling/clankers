"""Shared look for every scene: palette, fonts and reusable mobjects.

Import in a scene with
    from common.style import *
which also brings in `from manimlib import *`.
"""
from __future__ import annotations

from manimlib import *  # noqa: F401,F403

from common.narration import NarrationMixin

# ---------------------------------------------------------------- palette
BG = "#0E1016"
PANEL = "#1A1E28"          # card / table background
PANEL_EDGE = "#2C3240"
INK = "#ECECEC"            # main text
MUTED = "#9AA0AC"          # secondary text
FAINT = "#5A6170"          # tertiary / guides

# one color per stream; channel c drawn in stream c's color when routing is perfect
STREAM_COLORS = ["#58C4DD", "#FFC857", "#83C167", "#FC6255",
                 "#B189C6", "#FF8A3D", "#5CD0B3", "#EC92AB"]
KEY_COLOR = "#C9CED8"
VAL_COLOR = "#ECECEC"
QUERY_COLOR = "#FFE45E"

GATE_COLOR = "#B189C6"     # anything about the gate
MEMORY_COLOR = "#5CD0B3"   # Hebbian memory / synapses
HINGE_COLOR = "#FF8A3D"    # the hinge penalty
SLOW_COLOR = "#58C4DD"     # slow-memory phase
MUON_COLOR = "#F0AC5F"
ADAM_COLOR = "#9AA0AC"
GOOD = "#83C167"
BAD = "#FC6255"
WARN = "#FFC857"

MACHINE_COLORS = {"X": "#58C4DD", "L": "#FFC857", "E": "#B189C6"}

# ---------------------------------------------------------------- fonts
SERIF = "CMU Serif"
SANS = "Inter"
MONO = "DejaVu Sans Mono"


def T(text: str, size: float = 36, color=INK, font: str = SERIF, **kw) -> Text:
    """Serif text (titles, narration-like statements)."""
    return Text(text, font=font, font_size=size, fill_color=color, **kw)


def L(text: str, size: float = 26, color=INK, weight: str = "NORMAL", **kw) -> Text:
    """Sans label (diagram labels, table cells)."""
    return Text(text, font=SANS, font_size=size, fill_color=color, weight=weight, **kw)


def M(tex: str, size: float = 40, color=INK, **kw) -> Tex:
    """Math."""
    return Tex(tex, font_size=size, fill_color=color, **kw)


# ---------------------------------------------------------------- base scene
class ClankersScene(NarrationMixin, Scene):
    """Base for every chapter: narration + subtitles + common framing helpers."""

    def tear_down(self) -> None:
        self.write_subtitles()
        super().tear_down()

    # convenience -----------------------------------------------------------
    def clear_all(self, run_time: float = 0.8, exclude=()) -> None:
        mobs = [m for m in self.mobjects if m not in exclude and m is not self.frame]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    def chapter_card(self, number, title: str, subtitle: str | None = None, hold: float = 0.0):
        card = ChapterCard(number, title, subtitle)
        self.play(FadeIn(card.number, shift=0.3 * DOWN), Write(card.title, run_time=1.2),
                  *([FadeIn(card.subtitle)] if card.subtitle else []), ShowCreation(card.rule))
        if hold:
            self.wait(hold)
        return card


class ChapterCard(VGroup):
    def __init__(self, number, title: str, subtitle: str | None = None):
        self.number = L(f"{number}" if isinstance(number, str) else f"Chapter {number}",
                        size=28, color=MUTED)
        self.title = T(title, size=60)
        self.rule = Line(LEFT, RIGHT, stroke_color=FAINT, stroke_width=2).set_width(
            max(self.title.get_width() + 0.6, 4))
        parts = [self.number, self.title, self.rule]
        self.subtitle = None
        if subtitle:
            self.subtitle = T(subtitle, size=30, color=MUTED)
            parts.append(self.subtitle)
        super().__init__(*parts)
        self.arrange(DOWN, buff=0.35)


def section_title(text: str, size: float = 40) -> VGroup:
    """A title pinned to the top-left with a short accent rule."""
    t = T(text, size=size)
    t.to_corner(UL, buff=0.5)
    rule = Line(ORIGIN, RIGHT, stroke_color=GATE_COLOR, stroke_width=3).set_width(t.get_width())
    rule.next_to(t, DOWN, buff=0.12, aligned_edge=LEFT)
    return VGroup(t, rule)


def source_note(text: str) -> Text:
    """Small provenance tag, bottom right: e.g. 'Report §7, Table 4'."""
    n = L(text, size=18, color=FAINT)
    n.to_corner(DR, buff=0.3)
    return n


# ---------------------------------------------------------------- tokens
def token(label: str, kind: str = "key", stream: int | None = None, width: float = 0.95,
          height: float = 0.55, size: float = 20) -> VGroup:
    """A sequence token. kind in {'ctx','key','val','query'}; ctx/val take a stream color."""
    color = STREAM_COLORS[stream % len(STREAM_COLORS)] if stream is not None else KEY_COLOR
    box = RoundedRectangle(width=width, height=height, corner_radius=0.1)
    if kind == "ctx":
        box.set_fill(color, 0.85).set_stroke(color, 2)
        text = L(label, size=size, color=BG, weight="BOLD")
    elif kind == "val":
        box.set_fill(color, 0.12).set_stroke(color, 2)
        text = L(label, size=size, color=color)
    elif kind == "query":
        box.set_fill(QUERY_COLOR, 0.15).set_stroke(QUERY_COLOR, 2.5)
        text = L(label, size=size + 4, color=QUERY_COLOR, weight="BOLD")
    else:  # key
        box.set_fill(PANEL, 1).set_stroke(KEY_COLOR, 1.5)
        text = L(label, size=size, color=KEY_COLOR)
    if text.get_width() > width * 0.88:
        text.set_width(width * 0.88)
    text.move_to(box)
    g = VGroup(box, text)
    g.kind, g.stream, g.label = kind, stream, label
    return g


def triple(stream: int, key: int, val: int, **kw) -> VGroup:
    """[CTX_s, KEY_i, VAL_j] as three adjacent tokens."""
    return VGroup(
        token(f"CTX{stream}", "ctx", stream, **kw),
        token(f"K{key}", "key", **kw),
        token(f"V{val}", "val", stream, **kw),
    ).arrange(RIGHT, buff=0.06)


def token_row(tokens, buff: float = 0.06) -> VGroup:
    return VGroup(*tokens).arrange(RIGHT, buff=buff)


# ---------------------------------------------------------------- memory / channels
def memory_grid(rows: int = 6, cols: int = 6, cell: float = 0.28, color=MEMORY_COLOR,
                label: str | None = None) -> VGroup:
    """A synaptic matrix drawn as a grid of cells (fill opacity encodes strength)."""
    cells = VGroup(*[
        Square(cell).set_stroke(color, 1, opacity=0.5).set_fill(color, 0.04)
        for _ in range(rows * cols)
    ]).arrange_in_grid(rows, cols, buff=0.03)
    g = VGroup(cells)
    g.cells = cells
    g.rows, g.cols = rows, cols
    if label:
        lab = L(label, size=20, color=color).next_to(cells, DOWN, buff=0.15)
        g.add(lab)
        g.label = lab
    return g


def channel_stack(k: int, rows: int = 5, cols: int = 5, cell: float = 0.22,
                  colors=None, label_prefix: str = "ch") -> VGroup:
    """k memory channels side by side, each a small grid, labelled ch0..ch(k-1)."""
    colors = colors or [STREAM_COLORS[i % len(STREAM_COLORS)] for i in range(k)]
    chans = VGroup(*[memory_grid(rows, cols, cell, colors[i], f"{label_prefix}{i}") for i in range(k)])
    chans.arrange(RIGHT, buff=0.35)
    return chans


def gate_bar(probs, colors=None, width: float = 1.6, height: float = 0.28) -> VGroup:
    """A gate g_t drawn as a stacked horizontal bar of channel probabilities."""
    colors = colors or [STREAM_COLORS[i % len(STREAM_COLORS)] for i in range(len(probs))]
    segs = VGroup()
    x = 0.0
    for p, c in zip(probs, colors):
        w = max(width * p, 1e-3)
        r = Rectangle(width=w, height=height).set_fill(c, 0.9).set_stroke(width=0)
        r.move_to(RIGHT * (x + w / 2))
        segs.add(r)
        x += w
    frame = Rectangle(width=width, height=height).set_stroke(INK, 1).set_fill(opacity=0)
    frame.move_to(RIGHT * width / 2)
    g = VGroup(segs, frame)
    g.center()
    return g


# ---------------------------------------------------------------- results
def frac_bar(label: str, num: int, den: int, color=GOOD, width: float = 5.0, height: float = 0.38,
             label_width: float = 3.2, size: float = 24) -> VGroup:
    """'label  [#######-----]  19/20' — one row of a comparison."""
    name = L(label, size=size, color=INK)
    track = Rectangle(width=width, height=height).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1)
    fill = Rectangle(width=max(width * num / den, 1e-3), height=height).set_fill(color, 0.9).set_stroke(width=0)
    fill.align_to(track, LEFT)
    value = L(f"{num}/{den}", size=size, color=color, weight="BOLD")
    bar = VGroup(track, fill)
    name.next_to(bar, LEFT, buff=0.3)
    if label_width:
        name.shift((name.get_left()[0] - (bar.get_left()[0] - 0.3 - label_width)) * LEFT)
    value.next_to(bar, RIGHT, buff=0.25)
    g = VGroup(name, bar, value)
    g.name_mob, g.track, g.fill, g.value = name, track, fill, value
    g.num, g.den = num, den
    return g


def grow_bar(fb: VGroup, run_time: float = 1.0) -> list:
    """Animations that grow a frac_bar's fill from zero and fade in its value."""
    fb.fill.save_state()
    fb.fill.stretch(1e-3, 0, about_edge=LEFT)
    return [Restore(fb.fill, run_time=run_time), FadeIn(fb.value, run_time=run_time)]


VERDICT_STYLES = {
    "SHOWN": GOOD, "SUPPORTED": GOOD, "RELIABLE": GOOD, "CONFIRMED": GOOD, "VALID": "#5CD0B3",
    "PRECISE": GOOD, "STANDS": GOOD,
    "MAJORITY": WARN, "PARTLY CONFIRMED": WARN, "PROMISING": WARN, "SUPERSEDED": WARN, "REPLACED": WARN,
    "MINORITY": HINGE_COLOR, "INCONCLUSIVE": MUTED,
    "NOT SHOWN": MUTED, "NOT PRECISE": MUTED, "UNTESTABLE": MUTED, "NOT": MUTED,
    "NEVER": BAD, "CORRECTED": BAD, "FAILED": BAD,
}


def verdict_badge(text: str, size: float = 20, color=None) -> VGroup:
    color = color or VERDICT_STYLES.get(text.upper(), INK)
    t = L(text, size=size, color=color, weight="BOLD")
    pill = RoundedRectangle(width=t.get_width() + 0.32, height=t.get_height() + 0.22,
                            corner_radius=(t.get_height() + 0.22) / 2)
    pill.set_fill(color, 0.12).set_stroke(color, 1.5)
    t.move_to(pill)
    return VGroup(pill, t)


def machine_badge(name: str, size: float = 22) -> VGroup:
    c = MACHINE_COLORS.get(name, INK)
    t = L(name, size=size, color=BG, weight="BOLD")
    circ = Circle(radius=max(t.get_height(), t.get_width()) * 0.85).set_fill(c, 1).set_stroke(width=0)
    t.move_to(circ)
    return VGroup(circ, t)


def simple_table(rows, col_widths=None, size: float = 22, header_color=MUTED, row_height: float = 0.5,
                 align=None) -> VGroup:
    """rows: list of lists of str (first row = header). Returns VGroup with .cells[r][c] and .rules."""
    n_cols = len(rows[0])
    col_widths = col_widths or [2.4] * n_cols
    align = align or (["left"] + ["center"] * (n_cols - 1))
    cells = VGroup()
    y = 0.0
    for r, row in enumerate(rows):
        row_g = VGroup()
        x = 0.0
        for c, txt in enumerate(row):
            color = header_color if r == 0 else INK
            m = L(str(txt), size=size, color=color, weight="BOLD" if r == 0 else "NORMAL")
            if m.get_width() > col_widths[c] - 0.15:
                m.set_width(col_widths[c] - 0.15)
            if align[c] == "left":
                m.move_to([x + 0.08, y, 0], aligned_edge=LEFT)
            elif align[c] == "right":
                m.move_to([x + col_widths[c] - 0.08, y, 0], aligned_edge=RIGHT)
            else:
                m.move_to([x + col_widths[c] / 2, y, 0])
            row_g.add(m)
            x += col_widths[c]
        cells.add(row_g)
        y -= row_height
    total_w = sum(col_widths)
    top = Line([0, row_height / 2, 0], [total_w, row_height / 2, 0], stroke_color=INK, stroke_width=2)
    mid = Line([0, -row_height / 2, 0], [total_w, -row_height / 2, 0], stroke_color=FAINT, stroke_width=1.5)
    bottom = Line([0, y + row_height / 2, 0], [total_w, y + row_height / 2, 0], stroke_color=INK, stroke_width=2)
    rules = VGroup(top, mid, bottom)
    g = VGroup(rules, cells)
    g.cells, g.rules, g.row_height, g.col_widths = cells, rules, row_height, col_widths
    g.center()
    return g


def row_highlight(table: VGroup, r: int, color=WARN, opacity: float = 0.14) -> Rectangle:
    row = table.cells[r]
    rect = Rectangle(width=table.rules.get_width() + 0.1, height=table.row_height * 0.92)
    rect.set_fill(color, opacity).set_stroke(color, 1.2, opacity=0.6)
    rect.move_to([table.rules.get_center()[0], row.get_center()[1], 0])
    return rect


def code_block(code: str, title: str | None = None, size: float = 20, width: float | None = None,
               language: str = "python") -> VGroup:
    """Syntax-highlighted code on a panel, with an optional file-name tab."""
    code_mob = Code(code, font=MONO, font_size=size, language=language, code_style="monokai")
    if width and code_mob.get_width() > width:
        code_mob.set_width(width)
    panel = RoundedRectangle(width=code_mob.get_width() + 0.5, height=code_mob.get_height() + 0.45,
                             corner_radius=0.12).set_fill("#15181F", 1).set_stroke(PANEL_EDGE, 1.5)
    code_mob.move_to(panel)
    g = VGroup(panel, code_mob)
    if title:
        tab = L(title, size=18, color=MUTED)
        tab.next_to(panel, UP, buff=0.08, aligned_edge=LEFT).shift(0.1 * RIGHT)
        g.add(tab)
        g.tab = tab
    g.panel, g.code = panel, code_mob
    return g


def card(content: Mobject, buff: float = 0.3, color=PANEL, edge=PANEL_EDGE) -> VGroup:
    rect = SurroundingRectangle(content, buff=buff).set_fill(color, 1).set_stroke(edge, 1.5)
    rect.round_corners(0.12)
    return VGroup(rect, content)


def bullet_list(items, size: float = 28, buff: float = 0.28, color=INK, dot_color=GATE_COLOR,
                width: float | None = None) -> VGroup:
    rows = VGroup()
    for it in items:
        dot = Dot(radius=0.06, fill_color=dot_color)
        txt = T(it, size=size, color=color) if isinstance(it, str) else it
        if width and txt.get_width() > width:
            txt.set_width(width)
        dot.next_to(txt, LEFT, buff=0.25).align_to(txt, UP).shift(0.12 * DOWN)
        rows.add(VGroup(dot, txt))
    rows.arrange(DOWN, buff=buff, aligned_edge=LEFT)
    return rows


def line_chart(x_range, y_range, width=7.0, height=4.0, x_label: str = "", y_label: str = "",
               x_ticks=None, y_ticks=None, size: float = 20) -> Axes:
    """Axes with readable sans tick labels; use axes.c2p / axes.get_graph to plot."""
    axes = Axes(x_range=x_range, y_range=y_range, width=width, height=height,
                axis_config=dict(stroke_color=MUTED, stroke_width=2, include_tip=False))
    labels = VGroup()
    for x in (x_ticks or []):
        t = L(f"{x:g}" if isinstance(x, (int, float)) else str(x), size=size, color=MUTED)
        t.next_to(axes.c2p(x if isinstance(x, (int, float)) else 0, y_range[0]), DOWN, buff=0.15)
        labels.add(t)
    for y in (y_ticks or []):
        t = L(f"{y:g}", size=size, color=MUTED)
        t.next_to(axes.c2p(x_range[0], y), LEFT, buff=0.15)
        labels.add(t)
    axes.add(labels)
    axes.tick_labels = labels
    if x_label:
        xl = L(x_label, size=size, color=MUTED).next_to(axes.x_axis, DOWN, buff=0.55)
        axes.add(xl)
        axes.x_label_mob = xl
    if y_label:
        yl = L(y_label, size=size, color=MUTED).rotate(PI / 2).next_to(axes.y_axis, LEFT, buff=0.6)
        axes.add(yl)
        axes.y_label_mob = yl
    return axes


def polyline(axes: Axes, xs, ys, color=INK, width: float = 3) -> VMobject:
    pts = [axes.c2p(x, y) for x, y in zip(xs, ys)]
    vm = VMobject().set_points_as_corners(pts).set_stroke(color, width)
    return vm
