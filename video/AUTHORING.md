# Writing a scene

Each chapter of the video is one file `scenes/sNN_name.py` holding one `ClankersScene` subclass, listed in
`scenes/manifest.py`. The script (narration) and visual plan for every chapter are in `STORYBOARD.md`.
`scenes/s02_task.py` is the reference implementation: copy its structure.

## Skeleton

```python
"""Chapter N — <title> (report §x; <sources>)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.style import *  # noqa: E402,F403   (also brings in `from manimlib import *`)


class ClassName(ClankersScene):
    def construct(self):
        card = self.chapter_card(N, "Chapter title")
        self.wait(0.6)
        title = section_title("Short title")
        self.play(FadeOut(card, shift=0.3 * UP), FadeIn(title, shift=0.3 * UP))

        with self.voiceover("Narration paragraph, verbatim from the storyboard.") as vo:
            self.play(...)                    # runs while the voice speaks
            vo.wait_until_sentence(1)         # land the next visual on sentence 2 (0-based index)
            self.play(...)
        # leaving the block waits until the narration has finished (+0.35 s)
        ...
        self.clear_all()                      # end every scene on an empty frame
```

## Narration rules

- Use the storyboard's narration **verbatim**; split a paragraph into several `voiceover` blocks only at
  sentence boundaries. If a sentence is factually wrong, fix it and say so in your final report.
- Kokoro speaks the text. `common/narration.py` rewrites a few things for the voice only (BDH → "B D H",
  "19/20" → "19 out of 20", η² → "eta squared", ...). Write numbers the way they should be spoken. If the
  voice needs a different form than the subtitle, pass `spoken="..."` with the **same number of sentences**.
- `vo.duration` is the block's audio length; `vo.time_of(i)` the start of sentence i; `vo.wait_until(t)`
  waits until t seconds into the block; `vo.remaining()` what is left. Animations inside a block may exceed
  the narration (the block then simply runs longer), but avoid long silent stretches.

## Visual rules

- Frame is 14.2 × 8 units. Keep content in x ∈ [−6.7, 6.7], y ∈ [−3.7, 3.3] (the section title occupies the
  top-left corner). Nothing may overlap or leave the frame. Text size ≥ 20 for labels, ≥ 24 for statements.
- Use the palette and helpers in `common/style.py`: `T` (serif statement), `L` (sans label), `M` (TeX),
  `token`, `triple`, `token_row`, `memory_grid`, `channel_stack`, `gate_bar`, `frac_bar` + `grow_bar`,
  `verdict_badge`, `machine_badge`, `simple_table` + `row_highlight`, `code_block`, `card`, `bullet_list`,
  `line_chart` + `polyline`, `section_title`, `source_note`. Streams always use `STREAM_COLORS[s]`; concepts use
  `GATE_COLOR`, `MEMORY_COLOR`, `HINGE_COLOR`, `SLOW_COLOR`, `MUON_COLOR`, `ADAM_COLOR`, `GOOD`, `BAD`, `WARN`.
- Every number on screen gets a `source_note(...)` naming its report section/table or file.
- **Do not edit `common/` or any other scene.** If you need a helper, define it in your own file.
- Something should move at least every ~4 s; never hold a static frame longer than ~6 s.
- Prefer real data: `data/*.json` (see `data/README` or each file's `_source`), `data/report_tables.json`.
- Aim for the clarity of a 3Blue1Brown video: build ideas up piece by piece, transform one picture into the
  next instead of cutting, highlight what the narration names at the moment it names it.

## ManimGL notes (this is 3b1b's manimgl 1.7, not Manim Community)

- `Text(s, font=..., font_size=...)`, `Tex(R"...")` (no `MathTex`), `ShowCreation` (not `Create`),
  `FadeIn(m, shift=UP)`, `FadeOut`, `Write`, `GrowFromCenter`, `GrowArrow`, `TransformFromCopy`,
  `ReplacementTransform`, `TransformMatchingTex`, `TransformMatchingStrings`, `LaggedStartMap(FadeIn, group, lag_ratio=)`,
  `LaggedStart(*anims)`, `Indicate`, `FlashAround`, `CountInFrom`, `ChangeDecimalToValue`, `Restore`, `mob.animate.…`.
- `Arrow(start, end, buff=, thickness=)`, `CurvedArrow(a, b, angle=)`, `Brace(mob, DOWN)`, `NumberLine(x_range=[a,b,step], width=)`,
  `Axes(x_range, y_range, width, height)` with `axes.c2p(x, y)`, `DecimalNumber`, `SurroundingRectangle`,
  `VGroup.arrange(DIRECTION, buff=, aligned_edge=)`, `arrange_in_grid(rows, cols, buff=)`, `DashedLine`, `Dot`, `Circle`.
- Tex: use raw strings; `\text{...}` works; colour parts with `t2c={"x_t": BLUE}` or `tex[...]` slicing.
- The camera frame is `self.frame` (e.g. `self.play(self.frame.animate.scale(0.8).move_to(p))`); reset it before the scene ends.

## Render, look, fix

```bash
cd /home/user/clankers/video
python3 build.py render --quality low --only sNN_name     # 480p; log in build/logs/<Class>.low.log
grep -n "Traceback\|Error" -A5 build/logs/<Class>.low.log  # the log is mostly progress bars
python3 build.py frames <Class> --every 3                 # build/frames/<Class>/sheet.png + f_###.png
```

Read `sheet.png` (and single `f_###.png` frames for detail) with the Read tool. Check: overlaps, text off
frame, unreadable sizes, empty or static stretches, visuals that do not match what is being said, wrong
numbers. Fix and re-render until clean. The first render of new narration also synthesizes it (cached
afterwards in `.tts_cache/`). The render ends with `ok`/`FAIL` and the mp4 at `build/videos/low/<Class>.mp4`.
