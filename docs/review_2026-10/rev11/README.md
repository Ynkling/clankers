# Revision 11 (final): the checking agent's passes

Revision 11 was written after the referee session's last file (`check_G_report11.md`, 23 October 2026) and was not
checked by the referee. It was checked instead by a separate checking agent run from the author's own session, as
Revision 10's draft was: git cannot show the agent to be independent of the author (the commits carry the same session
trailer), and the report says so (the revision paragraph, Section 15 and Section 17 of the report). The passes are the agent's reports, unedited:

- `first_pass.md` — the first draft (`f11085a`): every line the revision adds or changes since Revision 10, checked
  against the sessions' records on `claude/explore-F/G/H/I`, the referee's files on `claude/review-V` and the main
  line's git history. 86 statement groups: 50 MATCH, 23 PARTIAL, 7 DISAGREE, 6 NOT CHECKABLE; every count, pair,
  p-value, transition, cap, seed block, run count, commit hash and date reproduced; the DISAGREE and PARTIAL rows are
  process statements and wording that outran the records. All were taken in the next commit, one of them with a new error
  of the pass's own suggestion (the operating characteristics of S89c's rule, inverted), and one inconsistency the pass
  listed (the conclusion's first paragraph) was missed; both are taken after the second pass.
- `second_pass.md` — the corrections, checked the same way: 26 of the 30 rows RESOLVED, 3 PARTLY, 1 with the new error
  above; eight residues, all taken in the commit after it (the two above; "none a count behind a reading"; one
  descriptive range, not two; the handed type map's 2/2 at sixteen channels only; S89c's review before its first record;
  three lines of STATUS.md; the spec draft's header).

The agent's working notes (copies of the branches' report and record files it read, and its arithmetic) are not
committed; the files it read are on the branches at the commits the passes cite.
