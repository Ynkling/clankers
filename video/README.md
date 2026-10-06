# Partitioning memory: a video explainer of this repository

A 40-minute narrated explainer of this repository and its technical report, *Multi-Channel Hebbian
Plasticity in Multilayer BDH Solves Context-Conditional Binding by Partitioning Memory* (Revision 7,
preliminary). It is made with [ManimGL](https://github.com/3b1b/manim), 3Blue1Brown's animation engine.

- **The video itself is not stored in this repository.** `python3 build.py render --quality high &&
  python3 build.py assemble --quality high` rebuilds the 1080p master (with chapter markers and an English
  subtitle track) in `build/final/high/`; see "Rebuilding" below.
- **`clankers_explained.srt`**: the subtitles, which double as a full transcript of the narration.

## Chapters

| time | chapter | source |
|---|---|---|
| 00:00 | Prologue | `scenes/s00_intro.py` |
| 01:22 | 1. The Dragon Hatchling | `scenes/s01_bdh.py` |
| 02:56 | 2. Context-conditional binding | `scenes/s02_task.py` |
| 04:20 | 3. Channels and a gate | `scenes/s03_channels.py` |
| 05:59 | 4. A flat start | `scenes/s04_flat_start.py` |
| 07:15 | 5. Two more parts | `scenes/s05_rig.py` |
| 08:44 | 6. How a run is scored | `scenes/s06_scoring.py` |
| 10:38 | 7. Phases I to IV | `scenes/s07_history.py` |
| 12:29 | 8. How the work is checked | `scenes/s08_method.py` |
| 14:12 | 9. Inside the repository | `scenes/s09_repo.py` |
| 16:20 | 10. Phase V: four streams | `scenes/s10_four_streams.py` |
| 18:12 | 11. Eight streams by curriculum | `scenes/s11_curriculum.py` |
| 19:14 | 12. The race | `scenes/s12_screens.py` |
| 21:21 | 13. The recipe | `scenes/s13_recipe.py` |
| 22:42 | 14. Slow memory and a hinge, confirmed | `scenes/s14_slow_start.py` |
| 24:33 | 15. Which part matters | `scenes/s15_recipe_scope.py` |
| 25:47 | 16. The hinge is an early kick | `scenes/s16_kick.py` |
| 27:34 | 17. The early-recipe test | `scenes/s17_early_recipe.py` |
| 29:07 | 18. Breaking merges | `scenes/s18_merges.py` |
| 30:42 | 19. The gate forgets the context | `scenes/s19_eight_streams.py` |
| 33:57 | 20. Distant cues | `scenes/s20_distant_cues.py` |
| 34:48 | 21. Where the mechanism stands | `scenes/s21_standing.py` |
| 35:56 | 22. Keeping score on itself | `scenes/s22_retrospective.py` |
| 37:52 | 23. Limits and next steps | `scenes/s23_outlook.py` |

## How it was made, and how far to trust it

- **`STORYBOARD.md`** is the script: the narration, spoken verbatim, and a visual plan for every chapter.
  Before any scene was written it was fact-checked claim by claim against the report and the code by
  four adversarial checkers and a completeness critic, and 43 blocks were corrected.
- **Every chapter was written by one agent, then re-rendered, fact-checked and fixed by an independent
  reviewer.** Every number on screen carries a small source tag naming the report section or table, or the
  file it comes from.
- **Real data where it exists.** `data/` holds chart data extracted from `results/` (each file records
  its source in a `_source` key; `data/_build/` rebuilds them). Every count that could be checked against
  the report matches it. Screens (S1–S42) live on the branch `claude/outside-ideas`; numbers from them come
  from the report, or from that branch's records where `data/critic/` says so.
- **Illustrative pictures are labelled as such** (for example "schematic" or "gate values illustrative").
- Two caveats about the repository that the video states: most of `results/L/` is byte-identical to
  `results/X/` (only `early_recipe_results.json` is L's own, as the report's Table 9 says), and the
  default branch's README follows Revision 6 while Revision 7 sits on the research branch.

## Rebuilding

```bash
# ManimGL from source, plus what it needs
git clone https://github.com/3b1b/manim && pip install -e manim av
apt-get install texlive-latex-extra texlive-fonts-extra texlive-science dvisvgm cm-super ffmpeg xvfb
# narration: Kokoro (82M-parameter TTS) through ONNX
pip install kokoro-onnx soundfile
mkdir -p video/tts_models && cd video/tts_models
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
cd ..

python3 build.py render --quality low --only s02_task   # one chapter at 480p, to build/videos/low/
python3 build.py frames BindingTask --every 3          # contact sheet for review
python3 build.py render --quality high --jobs 2        # every chapter at 1080p
python3 build.py assemble --quality high               # build/final/high/: mp4 + srt + chapters
```

Rendering is headless (`build.py` wraps `manimgl` in `xvfb-run` when there is no display). Narration is
synthesized on first render and cached in `.tts_cache/`; subtitle timings come from the same synthesis.

## Layout

- `common/narration.py`: `voiceover()` for scenes (Kokoro TTS, audio placement, subtitle timing, spoken
  forms for acronyms and decimals).
- `common/style.py`: palette, fonts and reusable pieces (tokens, memory grids, gate bars, result bars,
  tables, verdict badges, code blocks). It also gives each render process its own TeX work directory,
  because ManimGL's shared one lets parallel renders cache one equation under another's key.
- `scenes/`: one file per chapter; `scenes/manifest.py` sets the order.
- `AUTHORING.md`: the conventions every scene follows.
- `build.py`: render, review frames, and assemble.
