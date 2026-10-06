"""Chapter 15 — which part of the recipe matters (report §8, Table 5; test_recipe_scope.py;
video/data/recipe_scope.json; per-seed classes from results/X/{recipe_scope,slow_start}_results.json
and results/L/recipe_scope_L_final.log)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load(name):
    with open(os.path.join(DATA, name)) as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------------- real data
_RS = _load("recipe_scope.json")
COUNTS, CLAIMS, CLASSES = _RS["counts"], _RS["claims"], _RS["outcome_classes"]

# Part 0 (S = 2, P = 4, k = 2), seeds 280..299, one letter per seed:
# D discovered, P position split, K key split, O any other failure (STREAM-PARTIAL, OTHER, bound with VAL cos >= 0.5).
# X: test_router_layout.fail_class on results/X/recipe_scope_results.json (HONLY0) and slow_start_results.json
# (A0, HINGE0); L: "Part 0 per seed" in results/L/recipe_scope_L_final.log:1296-1316.
PER_SEED = {
    "X": {"A0": "DDKDPDPPDPOPKPPDPPDD", "HONLY0": "DDDKKDKDDKODODDKDODD", "HINGE0": "DDDDDDDODDDDDDDDDDDD"},
    "L": {"A0": "DDDDPDPPDPKPOPKDPPDO", "HONLY0": "DDDDDDOKDKDDKDDKDKDO", "HINGE0": "DDDDDDDDDDDDDDDDDDDD"},
}
for _m, _arms in PER_SEED.items():
    for _arm, _s in _arms.items():
        assert len(_s) == 20
        assert _s.count("D") == COUNTS[_m][_arm]["success"], (_m, _arm)
        assert _s.count("K") == CLASSES[_m][_arm].get("KEY", 0), (_m, _arm)
        assert _s.count("P") == CLASSES[_m][_arm].get("POSITION", 0), (_m, _arm)
    _b = sum(1 for h, o in zip(_arms["HINGE0"], _arms["HONLY0"]) if h == "D" and o != "D")
    _c = sum(1 for h, o in zip(_arms["HINGE0"], _arms["HONLY0"]) if h != "D" and o == "D")
    assert (_b, _c) == (CLAIMS[_m]["Q2"]["b"], CLAIMS[_m]["Q2"]["c"]), _m


Q2_TEXT = {m: (f"{CLAIMS[m]['Q2']['b']} vs {CLAIMS[m]['Q2']['c']}", CLAIMS[m]["Q2"]["report"].split(", ")[1])
           for m in ("X", "L")}            # ("8 vs 1", "p = 0.020"), ("7 vs 0", "p = 0.008")
Q4_TEXT = {m: CLAIMS[m]["Q4"]["report"] for m in ("X", "L")}                     # "4 vs 3", "2 vs 2"
Q5_TEXT = {m: CLAIMS[m]["Q5"]["report"].split(", ") for m in ("X", "L")}        # ["6 vs 4", "p = 0.38"]


def _composition(m, arm):
    """Outcome classes of a k = 16 arm, grouped for the stacked strip (test_stream_recipe.outcome)."""
    out = dict(routed=0, shared=0, merged=0, other=0, collapsed=0)
    for name, n in CLASSES[m][arm].items():
        if "collapsed" in name:
            out["collapsed"] += n
        elif name == "BOUND ROUTED":
            out["routed"] += n
        elif name == "BOUND NOT routed":
            out["shared"] += n
        elif name.startswith("MERGED"):
            out["merged"] += n
        else:
            out["other"] += n
    assert out["routed"] + out["shared"] == COUNTS[m][arm]["success"]
    assert out["collapsed"] == COUNTS[m][arm]["collapsed"]
    return out


COMP = {m: {a: _composition(m, a) for a in ("SC8", "SC8_H")} for m in ("X", "L")}

# X, SC8_H, read gate's channel per stream at step 4800 (results/X/recipe_scope_results.json,
# runs["SC8_H|<seed>"].stats[step=4800].ch_map). Seeds 260..275; the nine with two channels of four:
TWO_OF_FOUR_X = [261, 263, 264, 266, 267, 269, 271, 272, 274]
assert len(TWO_OF_FOUR_X) == _RS["SC8_H_two_channels_of_four_at_4800_X"]["count"]
EXAMPLE_SEED = 263
EXAMPLE_MAP = [6, 6, 8, 8, 8, 6, 8, 6]     # stream s -> channel at update 4800
assert sorted(EXAMPLE_MAP.count(c) for c in set(EXAMPLE_MAP)) == [4, 4]

KEY_SPLIT = "#EC92AB"
OUT_COLORS = {"D": GOOD, "P": BAD, "K": KEY_SPLIT, "O": MUTED}
OUT_OPAC = {"D": 0.9, "P": 0.9, "K": 0.9, "O": 0.45}
B, Y = STREAM_COLORS[0], STREAM_COLORS[1]


# ------------------------------------------------------------------------------------- helpers
def at_phrase(vo, i, phrase):
    """Approximate time (s into the block) at which `phrase` is spoken in sentence i."""
    text, a, b = vo.synth.sentences[i]
    k = text.find(phrase)
    if k < 0:
        raise ValueError(phrase)
    return a + (b - a) * k / max(len(text), 1)


def chip(text, color, size=26, padx=0.28, pady=0.16, fill=0.15, weight="BOLD"):
    t = L(text, size=size, color=color, weight=weight)
    box = RoundedRectangle(width=t.get_width() + 2 * padx, height=t.get_height() + 2 * pady, corner_radius=0.12)
    box.set_fill(color, fill).set_stroke(color, 2)
    t.move_to(box)
    return VGroup(box, t)


def legend_item(color, text, opacity=0.9, outline=False, size=22):
    sq = Square(0.24).set_fill(color, 0.25 if outline else opacity).set_stroke(color, 2 if outline else 0)
    return VGroup(sq, L(text, size=size, color=MUTED)).arrange(RIGHT, buff=0.12)


def grouped_sequence(rng, S=2, P=4):
    """Key positions of one grouped-layout body: [(stream, key)], keys random, streams random per key."""
    out = []
    for k in rng.permutation(P):
        for s in rng.permutation(S):
            out.append((int(s), int(k)))
    return out


RULES = {
    "position": lambda j, s, k: int(j >= 4),
    "key": lambda j, s, k: int(k >= 2),
    "stream": lambda j, s, k: s,
}


class RecipeScope(ClankersScene):
    def construct(self):
        card = self.chapter_card(15, "Which part matters")
        self.wait(0.6)
        title = section_title("Taking the recipe apart")
        self.play(FadeOut(card, shift=0.3 * UP), FadeIn(title, shift=0.3 * UP))
        self.title_mob = title

        self.part_two_streams()
        self.part_sixteen_channels()
        self.part_curriculum()
        self.wait(0.8)
        self.clear_all()
        self.wait(0.3)                      # hold an empty frame at the very end

    def fade_all_but_title(self, run_time=0.7):
        mobs = [m for m in self.mobjects if m is not self.title_mob and m is not self.frame]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    # ============================================================== N15.1 + N15.2  two streams
    def part_two_streams(self):
        # ---------------- the recipe as two parts, and its learning-rate schedule
        slow_chip = chip("slow memory", SLOW_COLOR, 30)
        hinge_chip = chip("hinge", HINGE_COLOR, 30)
        plus, eq = T("+", 40), T("=", 40)
        res = T("the full recipe", 32)
        eqn = VGroup(slow_chip, plus, hinge_chip, eq, res).arrange(RIGHT, buff=0.32).move_to([0, 1.85, 0])
        hinge_sub = L("penalizes position splits", 22, MUTED).next_to(hinge_chip, DOWN, buff=0.18)

        ax = line_chart([0, 4800, 1200], [0, 1.25, 1.25], width=6.4, height=2.3, x_label="update",
                        x_ticks=[0, 2400, 4800])
        ax.move_to([-0.3, -1.25, 0])
        lo_y, hi_y, gate_y = 0.25, 1.0, 1.05
        y_lo = M(R"10^{-4}", 30, MUTED).next_to(ax.c2p(0, lo_y), LEFT, buff=0.2)
        y_hi = M(R"10^{-3}", 30, MUTED).next_to(ax.c2p(0, (hi_y + gate_y) / 2), LEFT, buff=0.2)
        lr_lab = L("learning rate", 22, MUTED).next_to(ax.c2p(0, 1.25), UP, buff=0.12).shift(0.3 * RIGHT)
        shade = Rectangle(width=ax.c2p(2400, 0)[0] - ax.c2p(0, 0)[0], height=ax.c2p(0, 1.25)[1] - ax.c2p(0, 0)[1])
        shade.set_fill(SLOW_COLOR, 0.12).set_stroke(width=0).move_to(ax.c2p(1200, 0.625))
        shade_lab = L("slow phase", 22, SLOW_COLOR).move_to(ax.c2p(1200, 0.62))
        gate_line = polyline(ax, [0, 4800], [gate_y, gate_y], GATE_COLOR, 5)
        rest_line = polyline(ax, [0, 2400, 2400, 4800], [lo_y, lo_y, hi_y, hi_y], MEMORY_COLOR, 5)
        rest_flat = polyline(ax, [0, 2400, 2400, 4800], [hi_y, hi_y, hi_y, hi_y], MEMORY_COLOR, 5)
        gate_lab = L("the gate", 22, GATE_COLOR).next_to(ax.c2p(4800, gate_y), RIGHT, buff=0.15).shift(0.12 * UP)
        rest_lab = L("everything else", 22, MEMORY_COLOR).next_to(ax.c2p(1200, lo_y), DOWN, buff=0.12)
        rest_lab2 = L("everything else, from the start", 22, MEMORY_COLOR)
        rest_lab2.next_to(ax.c2p(1200, hi_y), DOWN, buff=0.15).align_to(ax.c2p(0, 0), LEFT).shift(0.15 * RIGHT)
        cross = VGroup(Line(shade.get_corner(UL), shade.get_corner(DR)), Line(shade.get_corner(DL), shade.get_corner(UR)))
        cross.set_stroke(BAD, 3, opacity=0.7)
        lr_chip = chip("one learning rate", MUTED, 30)
        lr_chip.move_to(slow_chip, aligned_edge=RIGHT)
        res2 = T("the hinge alone", 32, color=HINGE_COLOR).move_to(res, aligned_edge=LEFT)
        src_a = source_note("Report §8; test_recipe_scope.py (HONLY: one optimizer group at 1e-3)")

        with self.voiceover(
            "Which part of the recipe does the work? The recipe scope test removed the slow phase and kept the "
            "hinge, at one learning rate throughout."
        ) as vo:
            self.play(LaggedStartMap(FadeIn, eqn, shift=0.15 * UP, lag_ratio=0.15), run_time=1.2)
            self.play(FadeIn(ax), FadeIn(VGroup(y_lo, y_hi, lr_lab)), FadeIn(shade), FadeIn(shade_lab),
                      ShowCreation(gate_line), ShowCreation(rest_line), FadeIn(gate_lab), FadeIn(rest_lab),
                      FadeIn(hinge_sub), run_time=1.2)
            vo.wait_until(at_phrase(vo, 1, "removed the slow"))
            self.play(ShowCreation(cross), Indicate(slow_chip, color=BAD, scale_factor=1.06), FadeIn(src_a),
                      run_time=0.8)
            self.play(Transform(rest_line, rest_flat), FadeOut(VGroup(shade, shade_lab, cross)),
                      FadeTransform(rest_lab, rest_lab2), y_lo.animate.set_opacity(0.35),
                      ReplacementTransform(slow_chip, lr_chip), run_time=1.3)
            vo.wait_until(at_phrase(vo, 1, "kept the"))
            self.play(Indicate(hinge_chip, color=HINGE_COLOR, scale_factor=1.12), run_time=0.8)
            vo.wait_until(at_phrase(vo, 1, "one learning rate"))
            self.play(FadeTransform(res, res2), Indicate(lr_chip, color=INK, scale_factor=1.06), run_time=0.9)

        # ---------------- set up: gate map (top left), reminder (top right), seed strips (bottom)
        r_lr = chip("one learning rate", MUTED, 22)
        r_hinge = chip("hinge", HINGE_COLOR, 22)
        r_plus, r_eq = T("+", 30), T("=", 30)
        r_res = L("hinge alone", 22, HINGE_COLOR, weight="BOLD")
        reminder = VGroup(r_lr, r_plus, r_hinge, r_eq, r_res).arrange(RIGHT, buff=0.18)
        reminder.to_edge(RIGHT, buff=0.45).set_y(2.75)

        rng = np.random.default_rng(15)
        seqs = [grouped_sequence(rng) for _ in range(4)]
        size, pitch = 0.52, 0.58
        gcells = VGroup()
        for r, seq in enumerate(seqs):
            for j, (s, k) in enumerate(seq):
                ring = RoundedRectangle(width=size, height=size, corner_radius=0.08)
                ring.set_stroke(STREAM_COLORS[s], 3).set_fill(opacity=0)
                inner = RoundedRectangle(width=size - 0.14, height=size - 0.14, corner_radius=0.05)
                inner.set_fill(PANEL, 1).set_stroke(width=0)
                lab = L(f"K{k}", 20, INK)
                if lab.get_width() > size - 0.16:
                    lab.set_width(size - 0.16)
                c = VGroup(ring, inner, lab)
                c.move_to([j * pitch, -r * pitch, 0])
                c.s, c.k, c.j = s, k, j
                gcells.add(c)
        gcells.move_to([-4.1, 1.2, 0])
        g_head = VGroup(L("ring: the token's stream", 22, MUTED),
                        L("fill: the channel the gate picks", 22, MUTED)).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        g_head.next_to(gcells, RIGHT, buff=0.3).align_to(gcells, UP)

        def fill_anims(rule):
            out = []
            for c in gcells:
                ch = RULES[rule](c.j, c.s, c.k)
                out.append(c[1].animate.set_fill(STREAM_COLORS[ch], 0.92))
                out.append(c[2].animate.set_color(BG))
            return out

        def state_badge(kind):
            txt, col, line = {"position": ("position split", BAD, "channel follows the key's place"),
                              "key": ("key split", KEY_SPLIT, "channel follows the key itself"),
                              "stream": ("stream split", GOOD, "channel follows the stream")}[kind]
            b = verdict_badge(txt, 24, col)
            expl = L(line, 22, INK)
            g = VGroup(b, expl).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            g.next_to(gcells, RIGHT, buff=0.3).align_to(gcells, DOWN).shift(0.05 * UP)
            return g

        # seed strips
        CELL, PITCH, X0 = 0.3, 0.36, -4.05
        COUNT_X, NOTE_X = 3.55, 4.05
        ROW_Y = {"X": [-0.86, -1.28, -1.70], "L": [-2.30, -2.72, -3.14]}
        ARMS = ["A0", "HONLY0", "HINGE0"]
        ROW_NAMES = {"A0": "plain gate", "HONLY0": "hinge alone", "HINGE0": "full recipe"}
        strips, labels, counts, badges = {}, {}, {}, VGroup()
        for m in ("X", "L"):
            mb = machine_badge(m, 22).move_to([-6.35, ROW_Y[m][1], 0])
            badges.add(mb)
            for a, y in zip(ARMS, ROW_Y[m]):
                codes = PER_SEED[m][a]
                row = VGroup(*[Square(CELL).set_fill(OUT_COLORS[ch], OUT_OPAC[ch]).set_stroke(width=0)
                               .move_to([X0 + i * PITCH, y, 0]) for i, ch in enumerate(codes)])
                row.codes = codes
                strips[m, a] = row
                lab_col = {"A0": MUTED, "HONLY0": HINGE_COLOR, "HINGE0": SLOW_COLOR}[a]
                labels[m, a] = L(ROW_NAMES[a], 22, lab_col).move_to([-6.0, y, 0], aligned_edge=LEFT)
                n = COUNTS[m][a]
                counts[m, a] = L(f"{n['success']}/{n['n']}", 22, GOOD if a == "HINGE0" else INK,
                                 weight="BOLD").move_to([COUNT_X, y, 0])
        empties = VGroup(*[VGroup(*[Square(CELL).set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1).move_to(c)
                                    for c in strips[k]]) for k in strips])
        legend = VGroup(L("one square per seed (280 to 299):", 22, MUTED),
                        legend_item(GOOD, "discovered"), legend_item(BAD, "position split"),
                        legend_item(KEY_SPLIT, "key split"), legend_item(MUTED, "other failure", opacity=0.45))
        legend.arrange(RIGHT, buff=0.3).move_to([0, -0.38, 0]).align_to([-6.0, 0, 0], LEFT)

        def note(m, a, text, color=MUTED):
            return L(text, 22, color).move_to([NOTE_X, ROW_Y[m][ARMS.index(a)], 0], aligned_edge=LEFT)

        a0_notes = VGroup(note("X", "A0", "9 of 12: position", BAD), note("L", "A0", "8 of 12: position", BAD))
        key_notes = {"X": note("X", "HONLY0", "5 of 8: key", KEY_SPLIT), "L": note("L", "HONLY0", "5 of 7: key", KEY_SPLIT)}
        hinge_notes = VGroup(note("X", "HINGE0", "no key splits", SLOW_COLOR),
                             note("L", "HINGE0", "no key splits", SLOW_COLOR))
        src_b = source_note("Report §7, §8, Table 5; results/X/*_results.json; results/L/recipe_scope_L_final.log")

        # transition (between sentences 2 and 3 of the paragraph)
        self.play(FadeOut(VGroup(ax, y_lo, y_hi, lr_lab, gate_line, rest_line, gate_lab, rest_lab2, hinge_sub),
                          shift=0.3 * DOWN),
                  ReplacementTransform(lr_chip, r_lr), ReplacementTransform(plus, r_plus),
                  ReplacementTransform(hinge_chip, r_hinge), ReplacementTransform(eq, r_eq),
                  FadeTransform(res2, r_res), FadeOut(src_a), run_time=1.1)
        self.play(LaggedStartMap(FadeIn, gcells, lag_ratio=0.01), FadeIn(g_head),
                  FadeIn(badges), *[FadeIn(labels[k]) for k in labels], FadeIn(empties), FadeIn(legend),
                  FadeIn(src_b), run_time=1.0)

        with self.voiceover(
            "At two streams the position splits still disappeared, but key splits took their place: five of eight "
            "failures on X and five of seven on L were key splits. Slowing the memory prevents those."
        ) as vo:
            # the plain gate's usual failure
            badge = state_badge("position")
            self.play(*fill_anims("position"), FadeIn(badge, shift=0.1 * LEFT),
                      *[FadeIn(strips[m, "A0"], lag_ratio=0.05) for m in ("X", "L")],
                      *[FadeIn(counts[m, "A0"]) for m in ("X", "L")], FadeIn(a0_notes), run_time=1.1)
            vo.wait_until(at_phrase(vo, 0, "still disappeared"))
            hinge_box = SurroundingRectangle(gcells, buff=0.1).set_stroke(HINGE_COLOR, 4)
            hinge_tag = L("hinge", 22, HINGE_COLOR, weight="BOLD").next_to(hinge_box, UP, buff=0.06).align_to(hinge_box, RIGHT)
            self.play(ShowCreation(hinge_box), FadeIn(hinge_tag),
                      *[c[1].animate.set_fill(PANEL, 1) for c in gcells], *[c[2].animate.set_color(INK) for c in gcells],
                      FadeOut(badge), run_time=0.8)
            self.play(*[TransformFromCopy(strips[m, "A0"], strips[m, "HONLY0"]) for m in ("X", "L")],
                      *[FadeIn(counts[m, "HONLY0"]) for m in ("X", "L")], run_time=1.0)
            vo.wait_until(at_phrase(vo, 0, "but key splits"))
            badge = state_badge("key")
            self.play(*fill_anims("key"), FadeIn(badge, shift=0.1 * LEFT), FadeOut(hinge_box), FadeOut(hinge_tag),
                      run_time=1.0)
            k_cells = VGroup(*[strips[m, "HONLY0"][i] for m in ("X", "L")
                               for i, ch in enumerate(PER_SEED[m]["HONLY0"]) if ch == "K"])
            self.play(LaggedStartMap(Indicate, k_cells, color=KEY_SPLIT, scale_factor=1.5, lag_ratio=0.08),
                      run_time=1.0)
            vo.wait_until(at_phrase(vo, 0, "five of eight"))
            self.play(FadeIn(key_notes["X"], shift=0.1 * LEFT), run_time=0.6)
            vo.wait_until(at_phrase(vo, 0, "five of seven"))
            self.play(FadeIn(key_notes["L"], shift=0.1 * LEFT), run_time=0.6)
            vo.wait_until_sentence(1)
            r_slow = chip("slow memory", SLOW_COLOR, 22)
            r_full = L("full recipe", 22, SLOW_COLOR, weight="BOLD")
            new_rem = VGroup(r_slow, r_plus.copy(), r_hinge.copy(), r_eq.copy(), r_full).arrange(RIGHT, buff=0.18)
            new_rem.to_edge(RIGHT, buff=0.45).set_y(2.75)
            self.play(FadeTransform(r_lr, r_slow), r_plus.animate.move_to(new_rem[1]),
                      r_hinge.animate.move_to(new_rem[2]), r_eq.animate.move_to(new_rem[3]),
                      FadeTransform(r_res, r_full),
                      *[TransformFromCopy(strips[m, "HONLY0"], strips[m, "HINGE0"]) for m in ("X", "L")],
                      *[FadeIn(counts[m, "HINGE0"]) for m in ("X", "L")], run_time=1.1)
            badge2 = state_badge("stream")
            self.play(*fill_anims("stream"), FadeTransform(badge, badge2), FadeIn(hinge_notes), run_time=0.9)
            badge = badge2
        reminder = VGroup(r_slow, r_plus, r_hinge, r_eq, r_full)

        # ---------------- N15.2 paired comparison, the reading
        outlines = {}
        for m in ("X", "L"):
            yh, yf = ROW_Y[m][1], ROW_Y[m][2]
            grp = VGroup()
            for i, (o, h) in enumerate(zip(PER_SEED[m]["HONLY0"], PER_SEED[m]["HINGE0"])):
                if (o == "D") == (h == "D"):
                    continue
                col = GOOD if h == "D" else BAD
                r = Rectangle(width=PITCH - 0.02, height=(yh - yf) + CELL + 0.1)
                r.set_stroke(col, 2.5).set_fill(opacity=0).move_to([X0 + i * PITCH, (yh + yf) / 2, 0])
                grp.add(r)
            outlines[m] = grp
        q2 = {}
        for m in ("X", "L"):
            yh, yf = ROW_Y[m][1], ROW_Y[m][2]
            line1 = VGroup(L(Q2_TEXT[m][0], 24, INK, weight="BOLD"), verdict_badge("SHOWN", 20)).arrange(RIGHT, buff=0.2)
            line2 = L(Q2_TEXT[m][1], 22, MUTED)
            g = VGroup(line1, line2).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
            g.move_to([NOTE_X, (yh + yf) / 2, 0], aligned_edge=LEFT)
            q2[m] = g
        # what the outlines mean (the discordant seeds that the McNemar test counts)
        out_legend = VGroup(legend_item(GOOD, "only the full recipe discovered", outline=True),
                            legend_item(BAD, "only the hinge alone discovered", outline=True))
        for it in out_legend:
            it[0].set_fill(opacity=0).set_stroke(width=2.5)
        out_legend.arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to([1.7, 0.5, 0], aligned_edge=LEFT)
        brace = Brace(VGroup(r_slow, r_plus, r_hinge), DOWN, buff=0.12).set_color(INK)
        quote = T("“both parts are needed”", 32).next_to(brace, DOWN, buff=0.15)
        quote_sub = L("the pre-registered reading, on X and on L", 22, MUTED).next_to(quote, DOWN, buff=0.12)
        src_c = source_note("Report §8, Table 5: Q2, exact McNemar test (full recipe beats hinge alone)")

        with self.voiceover(
            "The full recipe beat the hinge alone on both machines, eight to one and seven to zero. The "
            "pre-registered reading: both parts are needed."
        ) as vo:
            self.play(*[strips[m, "A0"].animate.set_opacity(0.25) for m in ("X", "L")],
                      *[labels[m, "A0"].animate.set_opacity(0.4) for m in ("X", "L")],
                      *[counts[m, "A0"].animate.set_opacity(0.4) for m in ("X", "L")],
                      FadeOut(a0_notes), FadeOut(VGroup(*key_notes.values())), FadeOut(hinge_notes),
                      FadeTransform(src_b, src_c), run_time=0.8)
            vo.wait_until(at_phrase(vo, 0, "eight to one"))
            self.play(LaggedStartMap(ShowCreation, outlines["X"], lag_ratio=0.1), FadeIn(q2["X"], shift=0.1 * LEFT),
                      FadeIn(out_legend, shift=0.1 * LEFT), run_time=1.0)
            vo.wait_until(at_phrase(vo, 0, "seven to zero"))
            self.play(LaggedStartMap(ShowCreation, outlines["L"], lag_ratio=0.1), FadeIn(q2["L"], shift=0.1 * LEFT),
                      run_time=1.0)
            vo.wait_until_sentence(1)
            self.play(FadeOut(g_head), GrowFromCenter(brace), Write(quote), run_time=1.0)
            self.play(FadeIn(quote_sub, shift=0.1 * UP),
                      Indicate(r_slow, color=SLOW_COLOR, scale_factor=1.08),
                      Indicate(r_hinge, color=HINGE_COLOR, scale_factor=1.08), run_time=0.9)
        self.wait(0.4)
        self.fade_all_but_title()

    # ============================================================== N15.3 a  sixteen channels
    def part_sixteen_channels(self):
        head = L("four streams, sixteen channels", 24, MUTED).move_to([-6.4, 2.3, 0], aligned_edge=LEFT)
        q_head = L("full recipe better? (Q4)", 22, MUTED).move_to([3.5, 2.3, 0])
        ys = {"X": (1.62, 1.1), "L": (0.22, -0.3)}
        bars, verdicts, mbs = {}, {}, VGroup()
        for m in ("X", "L"):
            mbs.add(machine_badge(m, 22).move_to([-6.3, sum(ys[m]) / 2, 0]))
            for a, y, col, name in (("HONLY4k16", ys[m][0], HINGE_COLOR, "hinge alone"),
                                    ("HINGE4k16", ys[m][1], SLOW_COLOR, "full recipe")):
                n = COUNTS[m][a] if a in COUNTS[m] else None
                num, den = n["success"], n["n"]
                fb = frac_bar(name, num, den, color=col, width=4.0, height=0.34, label_width=1.9, size=22)
                fb.shift([-3.7 - fb.track.get_left()[0], y - fb.track.get_y(), 0])
                bars[m, a] = fb
            v = VGroup(verdict_badge("NOT SHOWN", 20), L(Q4_TEXT[m], 22, INK)).arrange(DOWN, buff=0.12)
            v.move_to([3.5, sum(ys[m]) / 2, 0])
            verdicts[m] = v
        src = source_note("Report §7, §8, Table 5 (HONLY4k16; HINGE4k16 recorded on seeds 240-259)")

        stmt = T("The slow phase may matter only when channels are few", 28).move_to([0, -1.2, 0])

        def pict(k, cols, side, color_fn):
            sq = VGroup(*[Square(side).set_fill(color_fn(i), 0.85).set_stroke(width=0) for i in range(k)])
            sq.arrange_in_grid(1 if k <= 8 else 2, min(k, 8), buff=0.06)
            return sq

        p2 = pict(2, 2, 0.42, lambda i: STREAM_COLORS[i])
        p16 = pict(16, 8, 0.26, lambda i: STREAM_COLORS[i] if i < 4 else FAINT)
        c2 = VGroup(L("two streams, 2 channels", 22, INK), L("slow phase adds: Q2 shown", 22, GOOD))
        c2.arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        c16 = VGroup(L("four streams, 16 channels", 22, INK), L("no gain shown: Q4", 22, MUTED))
        c16.arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        g2 = VGroup(p2, c2).arrange(RIGHT, buff=0.25)
        g16 = VGroup(p16, c16).arrange(RIGHT, buff=0.25)
        obs = verdict_badge("an observation, not a tested claim", 22, WARN)
        row = VGroup(g2, g16).arrange(RIGHT, buff=0.9)
        low = VGroup(row, obs).arrange(DOWN, buff=0.35).move_to([0, -2.55, 0])

        with self.voiceover(
            "With sixteen channels at four streams, the hinge alone did about as well as the full recipe. The slow "
            "phase may matter only when channels are few; that is an observation, not a tested claim."
        ) as vo:
            self.play(FadeIn(head), FadeIn(mbs), FadeIn(src),
                      *[FadeIn(VGroup(bars[k].name_mob, bars[k].track)) for k in bars], run_time=0.8)
            self.play(*[a for k in bars for a in grow_bar(bars[k], run_time=1.2)])
            vo.wait_until(at_phrase(vo, 0, "about as well"))
            self.play(FadeIn(q_head), LaggedStartMap(FadeIn, VGroup(*verdicts.values()), shift=0.1 * LEFT,
                                                     lag_ratio=0.3), run_time=1.0)
            vo.wait_until_sentence(1)
            self.play(Write(stmt), run_time=1.2)
            self.play(FadeIn(g2, shift=0.15 * UP), run_time=0.6)
            self.play(FadeIn(g16, shift=0.15 * UP), run_time=0.6)
            vo.wait_until(at_phrase(vo, 1, "an observation"))
            self.play(FadeIn(obs, scale=0.9), run_time=0.7)
        self.fade_all_but_title()

    # ============================================================== N15.3 b  eight-stream curriculum
    def part_curriculum(self):
        head = L("eight streams by curriculum, sixteen channels", 24, MUTED).move_to([-6.4, 2.3, 0], aligned_edge=LEFT)
        CELL, PITCH, X0 = 0.32, 0.38, -3.6
        COUNT_X, NOTE_X = 4.2, 4.85
        ys = {"X": (1.45, 0.85), "L": (-0.5, -1.1)}
        order = ["routed", "shared", "merged", "other", "collapsed"]
        style = {"routed": (GOOD, 0.9, 0), "shared": (GOOD, 0.22, 2), "merged": (WARN, 0.85, 0),
                 "other": (MUTED, 0.45, 0), "collapsed": (BAD, 0.9, 0)}
        rows, cells_by, labels, counts, mbs = {}, {}, VGroup(), {}, VGroup()
        bound_head = L("bound", 22, MUTED).move_to([COUNT_X, 2.3, 0])
        q5_head = L("hinge better? (Q5)", 22, MUTED).move_to([NOTE_X, 2.3, 0], aligned_edge=LEFT)
        for m in ("X", "L"):
            mbs.add(machine_badge(m, 22).move_to([-6.35, sum(ys[m]) / 2, 0]))
            for a, y, name, col in (("SC8", ys[m][0], "curriculum alone", MUTED),
                                    ("SC8_H", ys[m][1], "curriculum + hinge", HINGE_COLOR)):
                comp = COMP[m][a]
                row = VGroup()
                by = {}
                i = 0
                for cat in order:
                    by[cat] = VGroup()
                    c, op, sw = style[cat]
                    for _ in range(comp[cat]):
                        sq = Square(CELL).set_fill(c, op).set_stroke(c, sw).move_to([X0 + i * PITCH, y, 0])
                        row.add(sq)
                        by[cat].add(sq)
                        i += 1
                rows[m, a] = row
                cells_by[m, a] = by
                labels.add(L(name, 22, col).move_to([-6.05, y, 0], aligned_edge=LEFT))
                n = COUNTS[m][a]
                counts[m, a] = L(f"{n['success']}/{n['n']}", 22, INK, weight="BOLD").move_to([COUNT_X, y, 0])
        legend = VGroup(
            VGroup(legend_item(GOOD, "bound, one channel per stream"),
                   legend_item(GOOD, "bound without one channel per stream", outline=True)).arrange(RIGHT, buff=0.45),
            VGroup(legend_item(WARN, "merged", 0.85), legend_item(BAD, "collapsed"),
                   legend_item(MUTED, "other failure", 0.45),
                   L("one square per run", 22, MUTED)).arrange(RIGHT, buff=0.45),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT).move_to([-6.05, -2.0, 0], aligned_edge=UL)

        def block_note(m, lines):
            g = VGroup(*lines).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
            g.move_to([NOTE_X, sum(ys[m]) / 2, 0], aligned_edge=LEFT)
            return g

        q5 = {m: block_note(m, [verdict_badge("NOT SHOWN", 20), L(", ".join(Q5_TEXT[m]), 22, INK)]) for m in ("X", "L")}
        coll = {m: L(f"collapsed: {COMP[m]['SC8']['collapsed']} → {COMP[m]['SC8_H']['collapsed']}", 22, INK)
                for m in ("X", "L")}
        shared = {m: VGroup(Square(0.22).set_fill(GOOD, 0.22).set_stroke(GOOD, 2),
                            L(f"{COMP[m]['SC8_H']['shared']} of {COUNTS[m]['SC8_H']['success']} binders", 22, GOOD))
                  .arrange(RIGHT, buff=0.12) for m in ("X", "L")}
        notes2 = {m: block_note(m, [coll[m], shared[m]]) for m in ("X", "L")}
        src = source_note("Report §8, Table 5 (Q5); outcome classes from recipe_scope.json (X; L from its log)")

        with self.voiceover(
            "On the eight-stream curriculum, the hinge bound eight of sixteen on X and seven of twenty on L, not "
            "significantly more than the curriculum alone. It removed the collapses on X and all but one on L, but "
            "merges took their place, and most binders did not give each stream its own channel. On X, at update "
            "forty-eight hundred, nine of sixteen gates held all eight streams in just two channels, four streams "
            "to each."
        ) as vo:
            self.play(FadeIn(head), FadeIn(mbs), FadeIn(labels), FadeIn(legend), FadeIn(bound_head), FadeIn(src),
                      run_time=0.8)
            self.play(*[LaggedStartMap(FadeIn, rows[m, "SC8"], lag_ratio=0.03) for m in ("X", "L")],
                      *[FadeIn(counts[m, "SC8"]) for m in ("X", "L")], run_time=1.0)
            vo.wait_until(at_phrase(vo, 0, "eight of sixteen"))
            self.play(LaggedStartMap(FadeIn, rows["X", "SC8_H"], lag_ratio=0.03), FadeIn(counts["X", "SC8_H"]),
                      run_time=1.0)
            vo.wait_until(at_phrase(vo, 0, "seven of twenty"))
            self.play(LaggedStartMap(FadeIn, rows["L", "SC8_H"], lag_ratio=0.03), FadeIn(counts["L", "SC8_H"]),
                      run_time=1.0)
            vo.wait_until(at_phrase(vo, 0, "not significantly"))
            self.play(FadeIn(q5_head), LaggedStartMap(FadeIn, VGroup(q5["X"], q5["L"]), shift=0.1 * LEFT, lag_ratio=0.3),
                      run_time=0.9)

            vo.wait_until_sentence(1)
            gone = VGroup(*[sq for m in ("X", "L") for sq in cells_by[m, "SC8"]["collapsed"]])
            self.play(LaggedStartMap(Indicate, gone, color=BAD, scale_factor=1.5, lag_ratio=0.1),
                      FadeOut(q5["X"]), FadeOut(q5["L"]), FadeOut(q5_head), FadeIn(coll["X"]), FadeIn(coll["L"]),
                      run_time=1.2)
            left = VGroup(*[sq for sq in cells_by["L", "SC8_H"]["collapsed"]])
            self.play(Indicate(left, color=BAD, scale_factor=1.6), run_time=0.7)
            vo.wait_until(at_phrase(vo, 1, "merges took"))
            merged = VGroup(*[sq for m in ("X", "L") for sq in cells_by[m, "SC8_H"]["merged"]])
            self.play(LaggedStartMap(Indicate, merged, color=WARN, scale_factor=1.4, lag_ratio=0.05), run_time=1.2)
            vo.wait_until(at_phrase(vo, 1, "most binders"))
            sh_boxes = VGroup(*[SurroundingRectangle(cells_by[m, "SC8_H"]["shared"], buff=0.05).set_stroke(GOOD, 2)
                                for m in ("X", "L")])
            self.play(ShowCreation(sh_boxes), FadeIn(shared["X"], shift=0.1 * LEFT), FadeIn(shared["L"], shift=0.1 * LEFT),
                      run_time=1.0)

            # ---------------- two channels of four
            vo.wait_until_sentence(2)
            self.play(*[FadeOut(m) for m in self.mobjects if m is not self.title_mob and m is not self.frame],
                      run_time=0.6)
            top = L("X, curriculum + hinge, update 4800", 24, MUTED).move_to([-6.4, 2.3, 0], aligned_edge=LEFT)
            runs = VGroup(*[Square(0.26).set_fill(WARN if s in TWO_OF_FOUR_X else PANEL, 0.85 if s in TWO_OF_FOUR_X else 1)
                            .set_stroke(PANEL_EDGE if s not in TWO_OF_FOUR_X else WARN, 1) for s in range(260, 276)])
            runs.arrange(RIGHT, buff=0.06)
            runs_lab = L(f"{len(TWO_OF_FOUR_X)} of 16 gates", 22, WARN, weight="BOLD")
            runs_g = VGroup(runs, runs_lab).arrange(RIGHT, buff=0.25).move_to([6.5, 2.3, 0], aligned_edge=RIGHT)
            ex_i = EXAMPLE_SEED - 260
            ex_box = SurroundingRectangle(runs[ex_i], buff=0.05).set_stroke(INK, 2)
            ex_lab = L(f"seed {EXAMPLE_SEED}, shown below", 22, MUTED).next_to(runs, DOWN, buff=0.15).align_to(runs, LEFT)

            chips = VGroup(*[token(f"s{s}", "ctx", s, width=0.6, height=0.42, size=22) for s in range(8)])
            chips.arrange(RIGHT, buff=0.25).move_to([0.3, 1.15, 0])
            chips_lab = L("eight streams", 22, MUTED).next_to(chips, LEFT, buff=0.4)
            slot_w, slot_h, slot_pitch = 0.66, 2.0, 0.78
            slots = VGroup(*[RoundedRectangle(width=slot_w, height=slot_h, corner_radius=0.08)
                             .set_fill(PANEL, 1).set_stroke(PANEL_EDGE, 1.5) for _ in range(16)])
            slots.arrange(RIGHT, buff=slot_pitch - slot_w).move_to([0, -1.2, 0])
            slot_nums = VGroup(*[L(str(c), 20, MUTED).next_to(slots[c], DOWN, buff=0.12) for c in range(16)])
            slot_lab = L("the gate's sixteen channels", 22, MUTED).next_to(slot_nums, DOWN, buff=0.15)
            targets = []
            fill_n = {c: 0 for c in set(EXAMPLE_MAP)}
            for s, c in enumerate(EXAMPLE_MAP):
                pos = slots[c].get_top() + DOWN * (0.3 + 0.47 * fill_n[c])
                fill_n[c] += 1
                targets.append(pos)
            used = sorted(set(EXAMPLE_MAP))
            four_labs = VGroup(*[L("4 streams", 20, WARN, weight="BOLD").next_to(slots[c], UP, buff=0.12) for c in used])
            src2 = source_note("Report §8; results/X/recipe_scope_results.json (SC8_H, ch_map at step 4800)")

            self.play(FadeIn(top), FadeIn(slots), FadeIn(slot_nums), FadeIn(slot_lab),
                      LaggedStartMap(FadeIn, chips, shift=0.1 * DOWN, lag_ratio=0.08), FadeIn(chips_lab),
                      FadeIn(src2), run_time=1.0)
            vo.wait_until(at_phrase(vo, 2, "nine of sixteen"))
            self.play(LaggedStartMap(FadeIn, runs, lag_ratio=0.04), FadeIn(runs_lab), run_time=0.9)
            self.play(ShowCreation(ex_box), FadeIn(ex_lab), run_time=0.5)
            vo.wait_until(at_phrase(vo, 2, "held all eight"))
            self.play(LaggedStart(*[chips[s].animate.move_to(targets[s]) for s in range(8)], lag_ratio=0.12),
                      FadeOut(chips_lab), *[slots[c].animate.set_stroke(WARN, 2.5) for c in used], run_time=1.8)
            vo.wait_until(at_phrase(vo, 2, "four streams"))
            self.play(LaggedStartMap(FadeIn, four_labs, shift=0.1 * DOWN, lag_ratio=0.3), run_time=0.7)
            empty = VGroup(*[slots[c] for c in range(16) if c not in used])
            self.play(empty.animate.set_opacity(0.35), run_time=0.6)
