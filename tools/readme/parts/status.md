## Status, machines and the Phase VI tests

**Revision 8 (7–10 October 2026).** Revision 8 added Phase VI: two pre-registered tests run on two machines with disjoint seeds, an audit of the project's records (`docs/audit_2026-10.md`), five exploratory batches on the side branch, three exploratory sessions run in parallel on their own branches, and a reading of twelve papers on context routing and memory (`docs/reading/`). Its first draft was reviewed by an independent referee session, which found statements that were wrong when committed; Revision 8.1 corrected them and nothing else (the list is in Section [15](#15-the-referees-report-and-the-claim-registry); the README's text as of Revision 8.1 is kept in `docs/revision8.md`).

REVISION_PARAGRAPHS

**Machines and sessions.** Two machines ran every main-line test: **X**, a cloud container on Intel Xeon hosts at 2.10 and 2.80 GHz, and **L**, an Intel i7-12650H laptop. From the Muon-recipe test on, X and L run disjoint seeds, and the pooled claim is primary. The exploratory sessions are **E** (branch `claude/outside-ideas`, batches 1–21), **F**, **G** and **H** (branches `claude/explore-F/G/H`: delta-rule channels, distant cues, and the split's devices and constants), **I** (branch `claude/explore-I`: a dense-loss task and a GPU package, run on the user's GTX 1660 SUPER, which left the project on 19 October, and RTX 3070 Laptop GPU), **R** (branch `claude/replicate-R`: the clean-room replication) and **V** (branch `claude/review-V`, merged here under `docs/review_2026-10/`: the referee); F, G, H and R ran on 2.10 GHz Xeons. Every verdict was fixed before its run and printed mechanically.


TABLE1
