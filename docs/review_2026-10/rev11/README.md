# Revision 11 (final): the checking agent's passes

Revision 11 was written after the referee session's last file (`check_G_report11.md`, 23 October 2026) and was not
checked by the referee. It was checked instead by a separate checking agent run from the author's own session, as
Revision 10's draft was: git cannot show the agent to be independent of the author (the commits carry the same session
trailer), and the report says so (Sections 1, 15 and 17 of the report). The passes are the agent's reports, unedited:

- `first_pass.md` — the first draft (`f11085a`): every line the revision adds or changes since Revision 10, checked
  against the sessions' records on `claude/explore-F/G/H/I`, the referee's files on `claude/review-V` and the main
  line's git history. 86 statement groups: 50 MATCH, 23 PARTIAL, 7 DISAGREE, 6 NOT CHECKABLE; every count, pair,
  p-value, transition, cap, seed block, run count, commit hash and date reproduced; the DISAGREE and PARTIAL rows are
  process statements and wording that outran the records. All were taken in the next commit.
- `second_pass.md` — the corrections, checked the same way.

The agent's working notes (copies of the branches' report and record files it read, and its arithmetic) are not
committed; the files it read are on the branches at the commits the passes cite.
