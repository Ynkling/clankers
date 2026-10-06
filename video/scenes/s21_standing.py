"""Chapter 21 — where the mechanism stands (report §13, Table 8; §7, §9–11, Table 7; data/hinge_firings.json)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

IN_COLOR = MEMORY_COLOR      # the gate's input term (teal), as in Ch. 3 and Ch. 19
REC_COLOR = GATE_COLOR       # the gate's recurrent term (purple)
CLAIM_COLORS = [GOOD, HINGE_COLOR, BAD, WARN]

# ---------------------------------------------------------------- Table 8 (report §13, p. 12), verbatim
# "^x" marks a footnote: d discovered, c with a curriculum, m same fast-weight memory, s not run on the main line
COLS = [("setting", 2.75, "l"), ("perfect gate", 1.6, "c"), ("plain gate", 1.6, "c"),
        ("recipe, no restarts", 3.95, "l"), ("restarts", 1.65, "c"), ("one channel", 1.75, "c")]
ROWS = [
    ["S=2, P=4, no conv.", "10/10", "223/400^d", ["HINGE 39/40^d", "Muon + WINDOW 80/80^d"], "60/60", "0 of > 200"],
    ["S=2, P=8, conv.", "6/6", "49/70", ["HINGE 37/40"], "40/40^c", "0/24^m"],
    ["S=4, P=4, k=4, conv.", "10/10", "29/80", ["—^s"], "18/40", "0/20"],
    ["S=4, P=4, k=16, conv.", "6/6", "44/80", ["HINGE 35/40; hinge only 34/40", "Muon + WINDOW 30/32"], "36/40", "—"],
    ["S=8, P=4, k=16, conv.", "2/2; 6/6^c", "0/16", ["HINGE 0/20;", "with the curriculum 15/36"], "0/5; 12/36^c", "—"],
]
CELL_SIZE = 22
LINE_GAP = 0.34

# ---------------------------------------------------------------- Table 7 (report §11, S39): Muon, S = 8, k = 16, HINGE
T7_UPD = [0, 200, 400, 600, 1200, 2400]
T7_DEC = [0.91, 0.70, 0.39, 0.18, 0.15, 0.13]
T7_REC = [0.04, 0.08, 0.20, 1.91, 4.02, 6.93]
T7_INP = [0.06, 0.07, 0.10, 0.11, 0.15, 0.13]


def interp(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


def load_firings():
    """Every hinge firing of Muon + WINDOW (WIN_M) on machine X: (update, |hinge grad| / |task grad|)."""
    with open(os.path.join(DATA, "hinge_firings.json")) as fh:
        runs = json.load(fh)["early_recipe"]["X"]["arms"]["WIN_M"]["runs"]
    out = []
    for r in runs.values():
        out += list(zip(r["fired_at"], r["ratio"]))
    return sorted(out), len(runs)


# ---------------------------------------------------------------- small builders
def cell_line(text, size=CELL_SIZE, color=INK):
    """One line of a cell; 'a^d' puts a small footnote letter after a."""
    main_s, sup_s = (text.split("^") + [None])[:2]
    main = L(main_s, size=size, color=color)
    g = VGroup(main)
    if sup_s:
        sup = L(sup_s, size=20, color=MUTED)
        sup.next_to(main, RIGHT, buff=0.04).align_to(main, UP).shift(0.07 * UP)
        g.add(sup)
    g.main = main
    return g


def build_table(top_y=2.62):
    widths = [c[1] for c in COLS]
    total_w = sum(widths)
    x_left = -total_w / 2
    lefts = [x_left + sum(widths[:i]) for i in range(len(widths))]
    head_h = 0.55
    header = VGroup()
    yc = top_y - head_h / 2
    for (name, w, al), x0 in zip(COLS, lefts):
        t = L(name, size=CELL_SIZE, color=MUTED, weight="BOLD")
        if al == "l":
            t.move_to([x0 + 0.1, yc, 0], aligned_edge=LEFT)
        else:
            t.move_to([x0 + w / 2, yc, 0])
        header.add(t)
    rows = []
    y = top_y - head_h
    row_spans = []
    for row in ROWS:
        n = max(len(c) if isinstance(c, list) else 1 for c in row)
        h = 0.52 if n == 1 else 0.52 + LINE_GAP * (n - 1)
        yc = y - h / 2
        cells = VGroup()
        for c_i, (content, (name, w, al), x0) in enumerate(zip(row, COLS, lefts)):
            lines = content if isinstance(content, list) else [content]
            cell = VGroup()
            for j, txt in enumerate(lines):
                ln = cell_line(txt)
                ly = yc + (len(lines) - 1) / 2 * LINE_GAP - j * LINE_GAP
                if al == "l":
                    ln.move_to([x0 + 0.1, ly, 0], aligned_edge=LEFT)
                else:
                    ln.move_to([x0 + w / 2, ly, 0])
                cell.add(ln)
            cells.add(cell)
        rows.append(cells)
        row_spans.append((y, y - h))
        y -= h
    bottom = y
    top_rule = Line([x_left, top_y, 0], [x_left + total_w, top_y, 0]).set_stroke(INK, 2)
    mid_rule = Line([x_left, top_y - head_h, 0], [x_left + total_w, top_y - head_h, 0]).set_stroke(FAINT, 1.5)
    bot_rule = Line([x_left, bottom, 0], [x_left + total_w, bottom, 0]).set_stroke(INK, 2)
    tab = VGroup(VGroup(top_rule, mid_rule, bot_rule), header, *rows)
    tab.header, tab.rows, tab.rules = header, rows, VGroup(top_rule, mid_rule, bot_rule)
    tab.lefts, tab.widths, tab.top, tab.bottom = lefts, widths, top_y, bottom
    tab.row_spans = row_spans
    return tab


def column(tab, c):
    return VGroup(tab.header[c], *[r[c] for r in tab.rows])


def column_box(tab, c, color):
    w = tab.widths[c] - 0.08
    rect = Rectangle(width=w, height=tab.top - tab.bottom + 0.08)
    rect.move_to([tab.lefts[c] + tab.widths[c] / 2, (tab.top + tab.bottom) / 2, 0])
    rect.set_fill(color, 0.10).set_stroke(color, 2.5)
    return rect


def mains(cells):
    """The main text of every line in a group of cells (footnote letters keep their colour)."""
    return VGroup(*[ln.main for cell in cells for ln in cell])


def claim_line(n, text, size=30):
    color = CLAIM_COLORS[n - 1]
    num = L(str(n), size=22, color=BG, weight="BOLD")
    dot = Circle(radius=0.22).set_fill(color, 1).set_stroke(width=0)
    num.move_to(dot)
    t = T(text, size=size, color=color)
    g = VGroup(VGroup(dot, num), t).arrange(RIGHT, buff=0.22)
    g.text = t
    return g


def fbar(label, num, den, color=GOOD, width=2.0, label_width=2.55, size=22, sup=None):
    fb = frac_bar(label, num, den, color=color, width=width, label_width=label_width, size=size)
    if sup:
        s = L(sup, size=20, color=MUTED).next_to(fb.value, RIGHT, buff=0.04).align_to(fb.value, UP).shift(0.07 * UP)
        fb.value.add(s)
    return fb


def shrink_fills(bars):
    """Call after the bars are laid out: Restore(b.fill) then grows each fill from its left edge."""
    for b in bars:
        b.fill.save_state()
        b.fill.stretch(1e-3, 0, about_edge=LEFT)


def chip(text, color):
    t = L(text, size=22, color=color, weight="BOLD")
    box = RoundedRectangle(width=t.get_width() + 0.4, height=0.48, corner_radius=0.12)
    box.set_fill(color, 0.12).set_stroke(color, 2)
    t.move_to(box)
    return VGroup(box, t)


def channel_box(color=None, w=0.62, h=0.8):
    r = Rectangle(width=w, height=h)
    if color is None:
        r.set_fill(PANEL, 1).set_stroke(FAINT, 1.5)
    else:
        r.set_fill(color, 0.55).set_stroke(color, 2)
    return r


def split_box(c1, c2, w=0.62, h=0.8):
    a = Rectangle(width=w / 2, height=h).set_fill(c1, 0.6).set_stroke(width=0)
    b = Rectangle(width=w / 2, height=h).set_fill(c2, 0.6).set_stroke(width=0)
    VGroup(a, b).arrange(RIGHT, buff=0)
    frame = Rectangle(width=w, height=h).set_stroke(INK, 2).move_to(VGroup(a, b))
    return VGroup(a, b, frame)


def route_arrow(start_mob, end_mob, color, end_shift=ORIGIN):
    a = Arrow(start_mob.get_bottom(), end_mob.get_top() + end_shift, buff=0.08, thickness=3.0)
    a.set_fill(color, 1).set_stroke(width=0)
    return a


class WhereItStands(ClankersScene):
    def construct(self):
        card = self.chapter_card(21, "Where the mechanism stands")
        self.wait(0.6)
        title = section_title("Where it stands")
        self.play(FadeOut(card, shift=0.3 * UP), FadeIn(title, shift=0.3 * UP))

        # ================================================================ N21.1 Table 8, row by row
        tab = build_table()
        legend = VGroup()
        for letter, txt in [("d", "discovered (else bound)"), ("c", "with a curriculum"),
                            ("m", "same fast-weight memory"), ("s", "not run on the main line")]:
            legend.add(VGroup(L(letter, size=20, color=MUTED, weight="BOLD"), L(txt, size=20, color=MUTED))
                       .arrange(RIGHT, buff=0.12))
        legend.arrange(RIGHT, buff=0.45).next_to(tab.rules[2], DOWN, buff=0.22)
        both = VGroup(machine_badge("X"), L("+", size=24, color=MUTED), machine_badge("L"),
                      L("counted over both machines: same seeds, so the counts are descriptive",
                        size=22, color=INK)).arrange(RIGHT, buff=0.15)
        both[3].shift(0.1 * RIGHT)
        both.next_to(legend, DOWN, buff=0.28)
        claim1 = claim_line(1, "The partition does what one channel cannot.")
        claim1.next_to(both, DOWN, buff=0.32)
        src1 = source_note("Report §13, Table 8")

        with self.voiceover(
            "Here is where things stand, counted over both machines. Given the partition, every grouped-layout "
            "configuration tried binds, and one channel with the same memory does not. The partition does what "
            "one channel cannot."
        ) as vo:
            self.play(ShowCreation(tab.rules[0]), ShowCreation(tab.rules[1]),
                      LaggedStartMap(FadeIn, tab.header, shift=0.1 * DOWN, lag_ratio=0.12), run_time=0.7)
            self.play(LaggedStart(*[LaggedStartMap(FadeIn, r, shift=0.15 * RIGHT, lag_ratio=0.12) for r in tab.rows],
                                  lag_ratio=0.45),
                      ShowCreation(tab.rules[2]), FadeIn(src1), run_time=2.0)
            self.play(FadeIn(both, shift=0.1 * UP), FadeIn(legend), run_time=0.6)
            # "Given the partition, every ... configuration tried binds"
            vo.wait_until_sentence(1)
            box_pg = column_box(tab, 1, GOOD)
            self.play(ShowCreation(box_pg), mains(column(tab, 1)[1:]).animate.set_color(GOOD),
                      tab.header[1].animate.set_color(GOOD), run_time=0.9)
            self.play(LaggedStart(*[Indicate(r[1], color=GOOD, scale_factor=1.12) for r in tab.rows],
                                  lag_ratio=0.25), run_time=1.6)
            # "... and one channel with the same memory does not."
            vo.wait_until(vo.time_of(1) + 0.56 * (vo.time_of(2) - vo.time_of(1)))
            box_oc = column_box(tab, 5, BAD)
            oc_vals = VGroup(*[tab.rows[i][5] for i in range(3)])
            self.play(ShowCreation(box_oc), mains(oc_vals).animate.set_color(BAD),
                      tab.header[5].animate.set_color(BAD),
                      mains(VGroup(tab.rows[3][5], tab.rows[4][5])).animate.set_color(FAINT), run_time=0.9)
            vo.wait_until_sentence(2)
            others = VGroup(*[column(tab, c) for c in (2, 3, 4)])
            self.play(FadeIn(claim1, shift=0.15 * UP), others.animate.set_opacity(0.35), run_time=0.9)

        # ================================================================ N21.2 discovery is early and steerable
        firings, n_runs = load_firings()
        rec = [r[3] for r in tab.rows]
        claim2 = claim_line(2, "Discovery is decided early, and it can be steered.")
        claim2.to_edge(LEFT, buff=0.5).set_y(2.55)

        ax_l, ax_r, ax_y = -5.6, 5.6, 0.85
        def X(u):
            return ax_l + (ax_r - ax_l) * u / 4800
        axis = Line([ax_l, ax_y, 0], [ax_r, ax_y, 0]).set_stroke(MUTED, 2)
        ticks = VGroup(*[Line([X(u), ax_y - 0.07, 0], [X(u), ax_y + 0.07, 0]).set_stroke(MUTED, 2)
                         for u in (0, 1200, 2400, 3600, 4800)])
        band = Rectangle(width=X(2400) - X(0), height=0.36).set_fill(SLOW_COLOR, 0.22).set_stroke(SLOW_COLOR, 1.5)
        band.move_to([(X(0) + X(2400)) / 2, ax_y - 0.3, 0])
        tick_labs = VGroup(*[L(f"{u}", size=20, color=MUTED).move_to([X(u), ax_y - 0.72, 0])
                             for u in (0, 1200, 2400, 3600, 4800)])
        upd_lab = L("update", size=20, color=MUTED).next_to(tick_labs[-1], LEFT, buff=0.35)
        band_lab = L("slow memory: a tenth of the gate's learning rate", size=20, color=SLOW_COLOR).move_to(band)
        race_lab = L("tilts the race toward the stream split", size=20, color=SLOW_COLOR)
        race_lab.next_to(band, RIGHT, buff=0.3)
        early = Rectangle(width=X(600) - X(0), height=1.25).set_fill(WARN, 0.10).set_stroke(WARN, 1.5)
        early.move_to([(X(0) + X(600)) / 2, ax_y + 0.625, 0])
        early_lab = L("the first few hundred updates", size=22, color=WARN).next_to(early, RIGHT, buff=0.25)
        early_lab.align_to(early, UP)

        spikes = VGroup()
        for u, r in firings:
            h = 0.26 * np.log10(max(r, 1.0))
            s = Line([X(u), ax_y, 0], [X(u), ax_y + h, 0]).set_stroke(HINGE_COLOR, 2.5)
            spikes.add(s)
        last_fire = max(u for u, _ in firings)
        med_ratio = float(np.median([r for _, r in firings]))
        spike_lab1 = L(f"hinge firings, Muon + WINDOW on X: {len(firings)} in {n_runs} runs, none after {last_fire}",
                       size=20, color=HINGE_COLOR)
        spike_lab2 = L(f"height: push ÷ task gradient, log; median ≈ {round(med_ratio, -2):,.0f}×",
                       size=20, color=MUTED)
        spike_labs = VGroup(spike_lab1, spike_lab2).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        spike_labs.next_to(early, RIGHT, buff=0.25).align_to(early, UP)

        cut = DashedLine([X(2400), ax_y - 0.5, 0], [X(2400), ax_y + 0.62, 0], dash_length=0.08).set_stroke(WARN, 2.5)
        cut_lab = VGroup(L("update 2400: the slow phase ends,", size=20, color=WARN),
                         L("and WINDOW switches the hinge off", size=20, color=WARN)
                         ).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        cut_lab.next_to(cut.get_top(), RIGHT, buff=0.15).align_to(cut, UP)

        head_l = L("two streams", size=22, color=MUTED, weight="BOLD")
        head_r = L("four streams, sixteen channels", size=22, color=MUTED, weight="BOLD")
        bars_l = VGroup(fbar("HINGE, P = 4", 39, 40, sup="d"), fbar("Muon + WINDOW, P = 4", 80, 80, sup="d"),
                        fbar("HINGE, P = 8", 37, 40))
        bars_r = VGroup(fbar("HINGE", 35, 40), fbar("hinge only", 34, 40), fbar("Muon + WINDOW", 30, 32))
        for bars in (bars_l, bars_r):
            bars.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        col_l = VGroup(head_l, bars_l).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        col_r = VGroup(head_r, bars_r).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        cols = VGroup(col_l, col_r).arrange(RIGHT, buff=0.8, aligned_edge=UP)
        cols.move_to([0, -1.75, 0])
        shrink_fills([*bars_l, *bars_r])
        bar_note = L("d  discovered, the others bound  ·  X and L together, descriptive", size=20, color=MUTED)
        bar_note.next_to(cols, DOWN, buff=0.3).align_to(cols, LEFT)
        src2 = source_note("Report §13, Table 8; firings: data/hinge_firings.json (X, WIN_M)")

        targets = [(rec[0][0], VGroup(bars_l[0].name_mob, bars_l[0].value)),
                   (rec[0][1], VGroup(bars_l[1].name_mob, bars_l[1].value)),
                   (rec[1][0], VGroup(bars_l[2].name_mob, bars_l[2].value)),
                   (rec[3][0], VGroup(bars_r[0].name_mob, bars_r[0].value, bars_r[1].name_mob, bars_r[1].value)),
                   (rec[3][1], VGroup(bars_r[2].name_mob, bars_r[2].value))]

        with self.voiceover(
            "Discovery is decided in the first few hundred updates, and it can be steered. Slowing the memory shifts "
            "the race toward the stream split, and a hinge that fires when the gate's choice turns positional "
            "delivers a few large, well-timed pushes. Together they bind without restarts at two streams, and at "
            "four streams with sixteen channels. The recipe's work is done by update twenty-four hundred."
        ) as vo:
            box_rc = column_box(tab, 3, HINGE_COLOR)
            lift = []
            for cell in rec:
                for ln in cell:
                    lift.append(ln.main.animate.set_fill(HINGE_COLOR, opacity=1))
                    if len(ln) > 1:
                        lift.append(ln[1].animate.set_fill(opacity=1))
            self.play(FadeOut(claim1), FadeOut(box_pg), FadeOut(box_oc), ShowCreation(box_rc), *lift,
                      tab.header[3].animate.set_fill(HINGE_COLOR, opacity=1),
                      VGroup(*[column(tab, c) for c in (0, 1, 2, 4, 5)]).animate.set_opacity(0.3), run_time=0.7)
            rest = VGroup(tab.rules, *[column(tab, c) for c in (0, 1, 2, 4, 5)], tab.header[3],
                          rec[2], rec[4], box_rc, legend, both)
            late = squish_rate_func(smooth, 0.45, 1.0)
            self.play(*[TransformFromCopy(a, b) for a, b in targets],
                      *[FadeOut(a) for a, _ in targets],
                      FadeOut(rest), FadeIn(VGroup(head_l, head_r)),
                      *[FadeIn(b.track) for b in [*bars_l, *bars_r]],
                      FadeOut(src1),
                      FadeIn(claim2, shift=0.15 * DOWN, rate_func=late), ShowCreation(axis, rate_func=late),
                      FadeIn(ticks, rate_func=late), FadeIn(tick_labs, rate_func=late),
                      FadeIn(upd_lab, rate_func=late), FadeIn(src2, rate_func=late),
                      # "... in the first few hundred updates"
                      FadeIn(early, rate_func=squish_rate_func(smooth, 0.6, 1.0)),
                      FadeIn(early_lab, shift=0.1 * RIGHT, rate_func=squish_rate_func(smooth, 0.6, 1.0)), run_time=1.5)
            self.play(Indicate(early_lab, color=WARN, scale_factor=1.08), run_time=0.9)
            # "Slowing the memory shifts the race toward the stream split"
            vo.wait_until_sentence(1)
            self.play(early.animate.set_fill(opacity=0.06).set_stroke(opacity=0.6), GrowFromEdge(band, LEFT),
                      run_time=1.0)
            self.play(FadeIn(band_lab), run_time=0.5)
            self.play(FadeIn(race_lab, shift=0.15 * RIGHT), run_time=0.7)
            # "... and a hinge that fires when the gate's choice turns positional ..."
            vo.wait_until(vo.time_of(1) + 0.38 * (vo.time_of(2) - vo.time_of(1)))
            self.play(FadeOut(early_lab), LaggedStart(*[GrowFromEdge(s, DOWN) for s in spikes], lag_ratio=0.12),
                      run_time=2.6)
            self.play(FadeIn(spike_labs[0], shift=0.1 * RIGHT), run_time=0.6)
            self.play(FadeIn(spike_labs[1], shift=0.1 * RIGHT), run_time=0.6)
            # "Together they bind without restarts at two streams ..."
            vo.wait_until_sentence(2)
            self.play(LaggedStart(*[Restore(b.fill) for b in bars_l], lag_ratio=0.3), run_time=1.4)
            vo.wait_until(vo.time_of(2) + 0.55 * (vo.time_of(3) - vo.time_of(2)))
            self.play(LaggedStart(*[Restore(b.fill) for b in bars_r], lag_ratio=0.3), FadeIn(bar_note), run_time=1.4)
            # "The recipe's work is done by update twenty-four hundred."
            vo.wait_until_sentence(3)
            self.play(ShowCreation(cut), FadeIn(cut_lab, shift=0.1 * RIGHT), run_time=0.9)
            self.play(FlashAround(bars_l[1], color=WARN), FlashAround(bars_r[2], color=WARN), run_time=1.2)

        # ================================================================ N21.3 eight streams; merges
        claim3 = claim_line(3, "At eight streams the obstacle moves upstream.")
        claim3.to_edge(LEFT, buff=0.5).set_y(2.55)
        s8_a = fbar("HINGE from scratch", 0, 20, color=BAD, width=3.0, label_width=2.95)
        s8_b = fbar("HINGE + stream curriculum", 15, 36, color=WARN, width=3.0, label_width=2.95)
        s8 = VGroup(s8_a, s8_b).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        s8.move_to([0, 1.65, 0]).to_edge(LEFT, buff=0.45)
        shrink_fills([s8_a, s8_b])
        s8_a.fill.set_opacity(0)   # 0/20: nothing to draw
        # 4 of the 15 binders hold one channel per stream (green), 11 do not (stay amber)
        per_stream = Rectangle(width=3.0 * 4 / 36, height=s8_b.track.get_height()).set_fill(GOOD, 0.95).set_stroke(width=0)
        per_stream.align_to(s8_b.track, LEFT).align_to(s8_b.track, DOWN)
        s8_note = VGroup(L("4 bound with one channel per stream", size=20, color=GOOD),
                         L("11 bound without", size=20, color=WARN)).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        s8_note.next_to(s8_b.value, RIGHT, buff=0.35)
        s8_tag = L("S = 8, k = 16, counted over X and L", size=20, color=MUTED)
        s8_tag.next_to(s8_a.value, RIGHT, buff=0.35)

        # the gate as a pipeline: token -> state h_t (recurrent loop) -> readout W_g -> g_t
        py = -1.0
        tok = token("CTX7", "ctx", 7).move_to([-5.75, py, 0])
        hbox = RoundedRectangle(width=1.3, height=1.0, corner_radius=0.12).set_fill(PANEL, 0).set_stroke(GATE_COLOR, 2.5)
        hbox.move_to([-3.05, py, 0])
        h_bg = hbox.copy().set_fill(PANEL, 1).set_stroke(width=0)
        echo = RoundedRectangle(width=1.3, height=1.0, corner_radius=0.12).set_fill(STREAM_COLORS[7], 0.6)
        echo.set_stroke(width=0).move_to(hbox)
        h_lab = M(R"h_t", 40).move_to(hbox)
        state_lab = L("gate state", size=20, color=GATE_COLOR).next_to(hbox, DOWN, buff=0.2)
        in_line = Line(tok.get_right() + 0.08 * RIGHT, hbox.get_left() + 0.1 * LEFT).set_stroke(IN_COLOR, 3)
        in_tip = ArrowTip(width=0.22, length=0.2).set_fill(IN_COLOR, 1).set_stroke(width=0)
        in_tip.move_to(hbox.get_left() + 0.05 * LEFT, aligned_edge=RIGHT)
        in_lab = L("input term", size=20, color=IN_COLOR).next_to(in_line, UP, buff=0.12)
        in_val = DecimalNumber(0.06, num_decimal_places=2, font_size=24, color=IN_COLOR)
        in_val.next_to(in_line, DOWN, buff=0.12)
        loop = ArcBetweenPoints(hbox.get_corner(UR) + 0.3 * LEFT, hbox.get_corner(UL) + 0.3 * RIGHT, angle=1.15 * PI)
        loop.set_stroke(REC_COLOR, 3)
        loop_tip = ArrowTip(width=0.22, length=0.2).set_fill(REC_COLOR, 1).set_stroke(width=0)
        loop_tip.rotate(-PI / 2).move_to(hbox.get_corner(UL) + 0.3 * RIGHT + 0.09 * UP)
        rec_lab = L("recurrent term", size=20, color=REC_COLOR)
        rec_val = DecimalNumber(0.04, num_decimal_places=2, font_size=24, color=REC_COLOR)
        rec_grp = VGroup(rec_lab, rec_val).arrange(RIGHT, buff=0.15).next_to(loop, UP, buff=0.1)
        g_bar = gate_bar([1 / 16] * 16, colors=[GATE_COLOR if i % 2 == 0 else "#7E6290" for i in range(16)],
                         width=1.9, height=0.34)
        g_bar.move_to([0.2, py, 0])
        g_lab = M(R"g_t", 36).next_to(g_bar, UP, buff=0.15)
        g_sub = L("which channel: near uniform", size=20, color=MUTED).next_to(g_bar, DOWN, buff=0.15)
        ro = Arrow(hbox.get_right(), g_bar.get_left(), buff=0.1, thickness=3.5).set_fill(GATE_COLOR, 1).set_stroke(width=0)
        ro_lab = VGroup(L("readout", size=20, color=GATE_COLOR), M(R"W_g", 30, color=GATE_COLOR)).arrange(RIGHT, buff=0.12)
        ro_lab.next_to(ro, UP, buff=0.12)
        pipe = VGroup(tok, in_line, in_tip, in_lab, in_val, h_bg, echo, hbox, h_lab, state_lab, loop, loop_tip, rec_grp,
                      ro, ro_lab, g_bar, g_lab, g_sub)

        # decodability of the stream from h_t at keys (Table 7, Muon, S = 8)
        dax = line_chart([0, 2400, 1200], [0, 1, 0.5], width=3.0, height=2.1, x_label="update",
                         x_ticks=[0, 1200, 2400], y_ticks=[0, 0.5, 1], size=20)
        dax.move_to([4.15, -1.25, 0])
        d_title = L("stream read from the gate state at keys", size=20, color=INK).next_to(dax, UP, buff=0.22)
        d_title.shift((dax.x_axis.get_center()[0] - d_title.get_center()[0]) * RIGHT)
        chance = DashedLine(dax.c2p(0, 0.125), dax.c2p(2400, 0.125), dash_length=0.06).set_stroke(MUTED, 1.5)
        chance_lab = L("chance", size=20, color=MUTED).next_to(chance, RIGHT, buff=0.2).shift(0.06 * DOWN)
        upd = ValueTracker(0)

        def dec_curve():
            u = upd.get_value()
            xs = [x for x in T7_UPD if x < u] + [u]
            pts = [dax.c2p(x, interp(T7_UPD, T7_DEC, x)) for x in xs]
            if len(pts) < 2:
                pts = [pts[0], pts[0] + 1e-3 * RIGHT]
            return VMobject().set_points_as_corners(pts).set_stroke(GATE_COLOR, 3.5)

        curve = always_redraw(dec_curve)
        dot = always_redraw(lambda: Dot(dax.c2p(upd.get_value(), interp(T7_UPD, T7_DEC, upd.get_value())),
                                        radius=0.07, fill_color=INK))
        d_val = DecimalNumber(0.91, num_decimal_places=2, font_size=26, color=INK)
        d_val.add_updater(lambda m: m.set_value(interp(T7_UPD, T7_DEC, upd.get_value())))
        d_val.add_updater(lambda m: m.next_to(dot, UR, buff=0.06).shift(0.06 * UP))
        dec_grp = VGroup(dax, d_title, chance, chance_lab)

        chips = VGroup(chip("hinge", HINGE_COLOR), chip("split", KEY_COLOR), chip("optimizer", MUON_COLOR))
        chips.arrange(RIGHT, buff=0.25).move_to([-0.55, -2.75, 0])
        chip_arrows = VGroup(*[Arrow(c.get_top(), ro.get_center() + 0.12 * DOWN, buff=0.08, thickness=2.0)
                               .set_fill(c[0].get_stroke_color(), 0.9).set_stroke(width=0) for c in chips])
        lost = L("the stream is not here", size=22, color=BAD, weight="BOLD")
        lost.next_to(state_lab, DOWN, buff=0.15).align_to(tok, LEFT)
        h_bad = SurroundingRectangle(hbox, buff=0.08).set_stroke(BAD, 3)
        src3 = source_note("Report §8 (4 and 11 of 15); §11, Table 7 (S39, Muon); §13, Table 8")

        with self.voiceover(
            "At eight streams the obstacle moves upstream. Before any routing can form, the gate's recurrent state "
            "stops reflecting its input. No change to the gate's readout, whether a hinge, a split, or an optimizer, "
            "can route by a stream its state does not carry."
        ) as vo:
            n2 = VGroup(early, spikes, spike_labs, band, band_lab, race_lab, axis, ticks, tick_labs, upd_lab, cut,
                        cut_lab, cols, bar_note)
            self.play(FadeOut(n2), FadeOut(src2), ReplacementTransform(claim2, claim3), run_time=0.9)
            self.play(FadeIn(VGroup(s8_a.name_mob, s8_a.track, s8_b.name_mob, s8_b.track)),
                      FadeIn(src3), run_time=0.5)
            self.play(Restore(s8_b.fill), FadeIn(s8_a.value), FadeIn(s8_b.value), FadeIn(s8_tag), run_time=0.8)
            self.play(FadeIn(per_stream), FadeIn(s8_note, shift=0.1 * RIGHT), run_time=0.6)
            # "Before any routing can form, the gate's recurrent state stops reflecting its input."
            vo.wait_until_sentence(1)
            self.play(FadeIn(tok, shift=0.2 * RIGHT), ShowCreation(in_line), FadeIn(in_tip), FadeIn(in_lab),
                      FadeIn(in_val), FadeIn(h_bg), FadeIn(echo), FadeIn(hbox), Write(h_lab), FadeIn(state_lab), run_time=0.9)
            self.play(ShowCreation(loop), FadeIn(loop_tip), FadeIn(rec_grp), GrowArrow(ro), FadeIn(ro_lab),
                      FadeIn(g_bar), FadeIn(g_lab), FadeIn(g_sub), FadeIn(dec_grp), run_time=0.9)
            self.add(curve, dot, d_val)
            in_val.add_updater(lambda m: m.set_value(interp(T7_UPD, T7_INP, upd.get_value())))
            rec_val.add_updater(lambda m: m.set_value(interp(T7_UPD, T7_REC, upd.get_value())))
            loop.add_updater(lambda m: m.set_stroke(width=3 + 1.1 * interp(T7_UPD, T7_REC, upd.get_value())))
            echo.add_updater(lambda m: m.set_fill(opacity=0.6 * max(
                0.0, (interp(T7_UPD, T7_DEC, upd.get_value()) - 0.125) / (0.91 - 0.125))))
            self.play(upd.animate.set_value(2400), run_time=3.2, rate_func=linear)
            for m in (in_val, rec_val, loop, echo, d_val):
                m.clear_updaters()
            curve.clear_updaters()
            dot.clear_updaters()
            # "No change to the gate's readout, ..."
            vo.wait_until_sentence(2)
            self.play(Indicate(VGroup(ro, ro_lab), color=GATE_COLOR, scale_factor=1.08), run_time=0.9)
            span = vo.time_of(2), vo.duration - vo.time_of(2)
            for i, frac in enumerate((0.32, 0.43, 0.53)):
                vo.wait_until(span[0] + frac * span[1])
                self.play(FadeIn(chips[i], shift=0.15 * UP), GrowArrow(chip_arrows[i]), run_time=0.5)
            # "... can route by a stream its state does not carry."
            vo.wait_until(span[0] + 0.63 * span[1])
            crosses = VGroup(*[Line(c.get_left() + 0.08 * LEFT, c.get_right() + 0.08 * RIGHT).set_stroke(BAD, 4)
                               for c in chips])
            self.play(ShowCreation(h_bad), FadeIn(lost, shift=0.1 * UP), run_time=0.8)
            self.play(LaggedStartMap(ShowCreation, crosses, lag_ratio=0.3),
                      *[c[0].animate.set_stroke(opacity=0.5) for c in chips],
                      *[c[1].animate.set_fill(opacity=0.6) for c in chips],
                      chip_arrows.animate.set_fill(opacity=0.3), run_time=0.9)

        # ---------------------------------------------------------------- merges
        claim4 = claim_line(4, "Merges remain at four channels.")
        claim4.to_edge(LEFT, buff=0.5).set_y(2.55)

        def panel(kind):
            toks = VGroup(*[token(f"CTX{s}", "ctx", s, width=0.82, height=0.48) for s in range(4)])
            toks.arrange(RIGHT, buff=0.16)
            if kind == "spare":
                chans = VGroup(*[channel_box(None, w=0.19, h=0.8) for _ in range(16)]).arrange(RIGHT, buff=0.045)
                targets = [1, 6, 9, 14]
            else:
                chans = VGroup(*[channel_box(None) for _ in range(4)]).arrange(RIGHT, buff=0.3)
                targets = [0, 1, 2, 2]
            chans.next_to(toks, DOWN, buff=1.25)
            g = VGroup(toks, chans)
            return g, targets

        pA, tA = panel("four")
        pB, tB = panel("spare")
        pC, tC = panel("four")
        panels = VGroup(pA, pB, pC).arrange(RIGHT, buff=0.75).move_to([0, 0.35, 0])
        heads = VGroup(L("four channels", size=22, color=MUTED, weight="BOLD"),
                       L("sixteen channels", size=22, color=MUTED, weight="BOLD"),
                       L("a targeted copy, in screens", size=22, color=MUTED, weight="BOLD"))
        for h, p in zip(heads, panels):
            h.next_to(p, UP, buff=0.3)

        def fill_routes(p, targets, merged_color=None):
            toks, chans = p
            arrows = VGroup(*[route_arrow(toks[s], chans[c], STREAM_COLORS[s],
                                          end_shift=(0.12 * (LEFT if s == 2 else RIGHT) if targets.count(c) > 1 else ORIGIN))
                              for s, c in enumerate(targets)])
            fills = VGroup()
            for c in sorted(set(targets)):
                streams = [s for s, cc in enumerate(targets) if cc == c]
                if len(streams) == 1:
                    fills.add(channel_box(STREAM_COLORS[streams[0]], w=chans[c].get_width(), h=0.8).move_to(chans[c]))
                else:
                    fills.add(split_box(STREAM_COLORS[streams[0]], STREAM_COLORS[streams[1]]).move_to(chans[c]))
            return arrows, fills

        arA, fA = fill_routes(pA, tA)
        arB, fB = fill_routes(pB, tB)
        arC, fC = fill_routes(pC, tC)
        merge_tagA = L("merged", size=20, color=BAD, weight="BOLD").next_to(pA[1][2], DOWN, buff=0.15)
        idle_tagA = L("idle", size=20, color=MUTED).next_to(pA[1][3], DOWN, buff=0.15)
        statA = VGroup(L("plain runs merged", size=20, color=MUTED), L("32/80", size=30, color=BAD, weight="BOLD"))
        statB = VGroup(L("plain runs merged", size=20, color=MUTED), L("9/80", size=30, color=GOOD, weight="BOLD"))
        statC = VGroup(L("Muon, bound with one channel per stream", size=20, color=MUTED),
                       L("6/10 against 3/10", size=30, color=GOOD, weight="BOLD"))
        for st, p in zip((statA, statB, statC), panels):
            st.arrange(DOWN, buff=0.12).next_to(p, DOWN, buff=1.05)
        statC_tag = L("with the copy vs without (screen S37)", size=20, color=MUTED).next_to(statC, DOWN, buff=0.1)
        # the copy: the shared channel's gate row onto the idle one, then stream 3 splits off
        copy_arrow = CurvedArrow(pC[1][2].get_bottom() + 0.05 * DOWN, pC[1][3].get_bottom() + 0.05 * DOWN, angle=PI * 0.8)
        copy_arrow.set_stroke(WARN, 3)
        copy_lab = L("copy the gate row, + noise", size=20, color=WARN).next_to(copy_arrow, DOWN, buff=0.08)
        if copy_lab.get_right()[0] > 6.55:
            copy_lab.shift((copy_lab.get_right()[0] - 6.55) * LEFT)
        new_arrow = route_arrow(pC[0][3], pC[1][3], STREAM_COLORS[3])
        new_c2 = channel_box(STREAM_COLORS[2]).move_to(pC[1][2])
        new_c3 = channel_box(STREAM_COLORS[3]).move_to(pC[1][3])
        src4 = source_note("Report §10, §13 (k = 4 vs 16: plain runs over both tests; S37 screen)")

        with self.voiceover(
            "And merges remain at four channels: spare channels prevent most of them, and in screens a targeted "
            "copy breaks some of the rest."
        ) as vo:
            n3 = VGroup(s8, per_stream, s8_note, s8_tag, pipe, dec_grp, curve, dot, d_val, chips, chip_arrows,
                        crosses, lost, h_bad)
            self.play(FadeOut(n3), FadeOut(src3), ReplacementTransform(claim3, claim4), run_time=0.6)
            self.play(FadeIn(heads[0]), FadeIn(pA), FadeIn(src4), run_time=0.5)
            self.play(LaggedStartMap(GrowArrow, arA, lag_ratio=0.2), FadeIn(fA, rate_func=squish_rate_func(smooth, 0.5, 1)),
                      run_time=0.9)
            self.play(FadeIn(merge_tagA, shift=0.1 * UP), FadeIn(idle_tagA), FadeIn(statA, shift=0.1 * UP), run_time=0.5)
            # "spare channels prevent most of them"
            vo.wait_until(0.25 * vo.duration)
            self.play(FadeIn(heads[1]), FadeIn(pB), LaggedStartMap(GrowArrow, arB, lag_ratio=0.2), FadeIn(fB),
                      run_time=1.0)
            self.play(FadeIn(statB, shift=0.1 * UP), run_time=0.5)
            # "and in screens a targeted copy breaks some of the rest"
            vo.wait_until(0.5 * vo.duration)
            self.play(FadeIn(heads[2]), FadeIn(pC), FadeIn(arC), FadeIn(fC), run_time=0.6)
            self.play(ShowCreation(copy_arrow), FadeIn(copy_lab), run_time=0.7)
            self.play(Transform(fC[2], VGroup(new_c2, new_c2.copy())), FadeIn(new_c3),
                      Transform(arC[3], new_arrow), run_time=0.8)
            self.play(FadeIn(statC, shift=0.1 * UP), FadeIn(statC_tag), run_time=0.6)
        self.wait(0.3)

        # ================================================================ recap: four claim cards
        self.play(FadeOut(VGroup(panels, heads, arA, fA, arB, fB, arC, fC, merge_tagA, idle_tagA, statA, statB,
                                 statC, statC_tag, copy_arrow, copy_lab, new_c3, src4)),
                  FadeOut(claim4), run_time=0.6)
        recap = VGroup()
        subs = ["perfect gate: every grouped setting binds; one channel: none",
                "slow memory + hinge, no restarts: S = 2, and S = 4 with k = 16",
                "the gate's state stops carrying the stream",
                "spare channels prevent most; in screens, a copy breaks some"]
        heads4 = ["The partition does what one channel cannot.",
                  "Discovery is decided early, and can be steered.",
                  "At eight streams the obstacle moves upstream.",
                  "Merges remain at four channels."]
        for i, (hd, sb) in enumerate(zip(heads4, subs)):
            cl = claim_line(i + 1, hd, size=26)
            sub = L(sb, size=20, color=MUTED)
            inner = VGroup(cl, sub).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
            sub.shift(0.66 * RIGHT)
            if inner.get_width() > 6.1:
                print("recap card too wide:", i, inner.get_width())
                inner.set_width(6.1)
            box = Rectangle(width=6.5, height=1.55).set_fill(PANEL, 1).set_stroke(CLAIM_COLORS[i], 2)
            box.round_corners(0.12)
            inner.move_to(box).align_to(box, LEFT).shift(0.22 * RIGHT)
            recap.add(VGroup(box, inner))
        recap.arrange_in_grid(2, 2, h_buff=0.3, v_buff=0.4).move_to(0.1 * DOWN)
        src5 = source_note("Report §13")
        self.play(LaggedStart(*[FadeIn(c, shift=0.2 * UP) for c in recap], lag_ratio=0.3), FadeIn(src5), run_time=1.8)
        self.play(LaggedStart(*[c[0].animate.set_stroke(width=4) for c in recap], lag_ratio=0.3), run_time=1.2)
        self.wait(0.6)
        self.clear_all()
