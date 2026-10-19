# Revision 10: the checks of its draft

Two passes over the Revision 10 draft by a session that did not write it, on 19 October 2026 (the referee session V had
closed on 13 October; these passes follow its method, with every number recomputed from the records):

- `first_pass.md`: the first version (`ea3afef`), 123 rows, 91 match, 17 partial, 11 disagree (seven distinct errors,
  all statements, none a number of a claim), 4 not record-checkable; fourteen items to change.
- `second_pass.md`: the second version (`6e75f69`), the 19 changed lines only, 31 rows, 27 match, 3 partial, 0 disagree;
  three items left, taken in the third version.
- `recompute_delta.py` / `_out.txt`: `test_delta_recipe` X, L and pooled from the raw records with an independent
  outcome function, McNemar, sign test, Wilson and Newcombe; `recompute_gpu.py` / `_out.txt`: session I's GPU run;
  `recompute_H.py` / `_out.txt`: H's S87a and S87c, and the one-to-one guard count on the main line's own records.

Paths in the pass files were rewritten from the checker's working copies to repository paths; the checker's text is
otherwise unchanged.
