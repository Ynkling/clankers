# Registry B: notes

Source: /home/user/clankers, branch claude/review-V (HEAD 0fdb363). Read-only. 92 rows: 46 X, 46 L.
Module docstrings read with `ast.get_docstring`. Commits found with `git log --all --full-history -S'<rule text>'`. Without `--full-history`, the pickaxe and `git log --all -- <file>` give 6c90a19 as the oldest commit, so neither is reliable on its own here.

## Things to know before using the TSV

- **No L RESULT block anywhere.** None of the five docstrings has an L RESULT block, at HEAD or on any ref (I grepped every ref under `git for-each-ref`). `results/L/` also has no JSON for these five tests; only X's files are committed.
  - L did run all five. docs/revision6.md:72 says "It also ran every Phase IV test". Its provenance table (docs/revision6.md:786-790) gives L's commits: 403605d, 99992ec, 4610878, ce62073 and 3b5e041, all merge commits.
  - L's outcomes survive only second-hand: in the BACKGROUND of the next test's docstring, and in Revision 6's README text (first at 631fd62, now docs/revision6.md).
- **What the L rows contain.**
  - Where such a source states L's verdict, verdict_printed starts with `SECONDARY:` and quotes the source. docstring_ref gives the citing file:line. recorded_commit is the commit where that citation first appears.
  - Where no source states the verdict (L bands for router_layout, short_conv and conv_lr; L VALID for short_conv, conv_lr and p_scaling; all L readings), verdict_printed is `NOT RECORDED` and recorded_commit/date are UNKNOWN. A parenthetical says what the counts imply.
  - **Filter on the `SECONDARY:` / `NOT RECORDED` prefixes if you want primary records only.** Every X row comes from the test's own RESULT block at HEAD.
- **Short conv X result has no surviving original commit.** recorded_commit = 6c90a19 is the earliest reachable commit holding it.
  - 6c90a19 is a parentless root commit (an import of the whole tree). Its tree equals 4610878 plus test_short_conv.py's RESULT and POST-HOC text, results/X/short_conv_results.json and a results/README line.
  - docs/revision6.md:788 lists the short_conv result commits as f7f8cd1, 9bb19f8 and 6c90a19. f7f8cd1 and 9bb19f8 are not in this repository.
  - The run itself started 2026-09-27 05:14 at f1cc9ab (results/README.md, JSON meta).
- **Readings have no id in the docstrings.** I named each branch `READING[<condition>]`, one row per branch per machine. A branch whose condition was not met has verdict_printed `(not printed: condition not met)` on X and points at its rule line.
- **Bands have no id either.** I named them `BAND-<arm>`. The validity gates are named `VALID` or `VALID-<layout>` / `DISCRIMINATIVE-<layout>`.
- **Excluded:** everything under "Diagnostics (not part of the verdict)", "Paired, post hoc (not claims)", descriptive arms (B_conv_in; A4, B4 and Part 2 of conv_lr) and short_conv's POST-HOC block.
- **No validity gate for router_reliability.** Its docstring says "No validity gate" (test_router_reliability.py:86), so it has no gate row.

## Per test: claim ids and machines

| test | fixed (rules first appear) | X result recorded | claim ids | X | L |
|---|---|---|---|---|---|
| test_router_reliability.py | 001c63e 2026-09-26T18:13:14Z | e5291da 2026-09-26T20:53:11Z | K8, K4, BAND-A, BAND-A_k4, BAND-A_k8, RESTART | RESULT block | second-hand: K4/K8/RESTART from test_router_layout.py:15-16 (017625c); counts, L p-values (K8 0.25, K4 0.41) and "every arm was MAJORITY" from docs/revision6.md:263-268 (631fd62) |
| test_router_layout.py | 017625c 2026-09-26T23:05:31Z | 37f9939 2026-09-27T02:11:55Z | VALID-/DISCRIMINATIVE-blocked, VALID-/DISCRIMINATIVE-shuffled, L1, L2, BAND-A_grouped/blocked/shuffled | RESULT block | second-hand: "valid and discriminative on both machines" and "L1, L2 NOT SHOWN on both" (docs/revision6.md:292-298, 631fd62); B 0/30 per machine in test_short_conv.py:16 (f1cc9ab); L p-values never printed; bands not recorded |
| test_short_conv.py | f1cc9ab 2026-09-27T04:27:52Z | 6c90a19 2026-09-27T16:36:28Z (see above) | VALID, K1, K2, K3, BAND-B_conv/A_conv/A, READING[K1+K3-], READING[K1+K3+], READING[K1-] | RESULT block | second-hand: K1-K3 from test_conv_lr.py:13-19 (3c69afd); L p-values (0.12, 0.03, 1e-7) from docs/revision6.md:333-335; VALID, bands and reading not recorded |
| test_conv_lr.py | 3c69afd 2026-09-27T16:49:04Z | b701eb6 2026-09-27T21:54:26Z | VALID, LR1-LR4, M1, BAND-B_conv4/A_conv4, 4 READING branches | RESULT block | second-hand: test_p_scaling.py:13-22 (745564f) and docs/revision6.md:338-342; VALID, bands and readings not recorded |
| test_p_scaling.py | 745564f 2026-09-27T22:39:04Z | 57479b8 2026-09-28T05:05:08Z | VALID, P1, P2, P3, M1-P8, 4 READING branches (no bands) | RESULT block | second-hand: test_load_curriculum.py:13-21 (b7c18f0) and docs/revision6.md:393-395; VALID and readings not recorded |

Every claim in a test was fixed in that test's adding commit. Each X run started after that commit and from it, per results/README.md and each JSON's `meta.git` / `meta.started`:
- router_reliability: 18:42 at 001c63e
- router_layout: 23:40 at 24987ec, a label-alignment commit after 017625c that touches only print widths, not the docstring
- short_conv: 05:14 at f1cc9ab
- conv_lr: 17:43 at 3c69afd
- p_scaling: 23:46 at 745564f

The stored `verdict` dicts in results/X/*.json match every printed X verdict and p-value.

## Did any rule change after its first run?

**No.** I diffed the docstring at every commit that touched each file (first commit → result commit → later commits → HEAD):
- In each file the only docstring change is one line, `-(Results are recorded at the bottom of this docstring after the run.)`, replaced by the RESULT block. For short_conv that change also adds the POST-HOC block.
- After the result commits, the docstrings are byte-identical up to HEAD.
- Code changes after the runs only add knobs that the CHECKs show are inert:
  - router_reliability: the `task`/`stats_fn`/`grad_fn`/`lr`/`builder` knobs on run_one
  - short_conv: `run_job(sp, recipe=None)` in 9c5939e
- ALPHA = 0.05, the band constants and the verdict expressions are the same as at first commit.

**6c90a19 ("Correct short conv docstring: POST-HOC ...") did not touch any claim.** Against f1cc9ab, its whole diff to test_short_conv.py is the RESULT block plus a POST-HOC block appended after it (test_short_conv.py:164-174). The POST-HOC block corrects only the BACKGROUND's scratch figures: they used lr 4e-3, not SUB_LR 1e-3. It says it "does not change the verdict or the reading above". The CLAIMS section (lines 69-90) is unchanged. No code changed.

## Ambiguities and small points

- **router_layout L1 on X prints `p = 1`.** The stored value is 0.99952 (rounded on print).
- **router_reliability's rule names the positive verdict HELPS**, while the printed headline is "K8 MORE CHANNELS HELP: NOT SHOWN". This is consistent with the rule.
- **router_layout L1/L2 on L: no p-value is printed anywhere.** For reference only (not in the TSV), my own one-sided Fisher from the revision6 counts gives L1 21 vs 28 → p = 0.967 and L2 29 vs 28 → p = 0.50. Both are NOT SHOWN, consistent with the cited verdict.
- **Other secondary L p-values I recomputed from the cited counts all match the cited values:**
  - K8 0.247 and K4 0.411
  - K1 0.12, K2 two-sided 0.034 and K3 1.2e-7
  - LR3 0.0065
  - P1 0.0128
  - P3 on L (not cited): 0.597
- **L's M1-P8 counts (5 bound vs 41 unbound) are my derivation** from B_conv8 1/30 + B_wide_conv8 4/16. No source prints them.
- **M1-P8 on X used exactly the minimum testable group size** (3 bound runs; UNTESTABLE needs < 3). The RESULT notes this.
- **K2 (short_conv) is the only two-sided claim.** Its verdict rides on Fisher; McNemar is printed alongside. LR3, LR4, P1 and P3 likewise decide on Fisher and only print McNemar.
- **Phase:** README Revision 8.1 (README.md:219) calls conv_lr "Phase III's". docs/revision6.md:84-92 lists all five tests as Phase IV, and docs/review_2026-10/revision_8_1_check.md already flags the README label. I used IV.
- **L code identity:** L's commits are merges whose test code is stated (not verified by me) to be identical to X's test commit (test_p_scaling.py:14, test_load_curriculum.py:14, docs/revision6.md:779). I checked 6c90a19 vs 4610878 directly: they differ only in the short_conv result files.
