# Revision 10: the checks of its draft

Two passes over the Revision 10 draft by a checking agent run from the author's own session on 19 October 2026 (git shows
the same session for the draft and the passes; the author states the agent had no view of the drafting; its independence cannot be shown).
They follow referee V's method where they recomputed (`test_delta_recipe`, the GPU run, H's screens) and read the other
screens' reports. V's own pass over the merged revision, `referee_pass.md`, and its check of Revision 10.1,
`revision_10_1_check.md`, with `check_rev10_*` and `audit_passes.md`, are the referee's files in this directory:

- `first_pass.md`: the first version (`ea3afef`), 123 rows, 92 match, 17 partial, 11 disagree (seven distinct errors,
  all statements, none a number of a claim), 3 not record-checkable by its own first verdicts (its summary says 91/17/11/4); fourteen items to change.
- `second_pass.md`: the second version (`6e75f69`), the 19 changed lines only, 31 rows, 27 match, 3 partial, 0 disagree;
  three items left, taken in the third version.
- `recompute_delta.py` / `_out.txt`: `test_delta_recipe` X, L and pooled from the raw records reading the recorded outcomes, with its own
  McNemar, sign test, Wilson and Newcombe; `recompute_gpu.py` / `_out.txt`: session I's GPU run;
  `recompute_H.py` / `_out.txt`: H's S87a and S87c, and the one-to-one guard count on the main line's own records.

Paths in the pass files were rewritten from the checker's working copies to repository paths; the checker's text is
otherwise unchanged.
