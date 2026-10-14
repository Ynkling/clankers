"""
explore_gpu/s75b_mutants.py — the mutation pass on explore_gpu/s75b.py's failure rules (EXPLORATORY tooling; not imported by
the driver, so not in its code SHA).

M maps a mutant's name to the text replacements that break one rule (each replaced text occurs exactly once in s75b.py).
The pass ran each mutant in a copy of the repository (no .git, no explore_out), with s75b_checks.py unchanged:

    python -m explore_gpu.s75b_checks --device cpu --only B4,B5

A mutant is caught if the run fails (rc 1). The first 18 are the third review's mutants that passed every CHECK at f4cf7a8;
the other 17 break the rules added after it. Results: s75b_mutants_cpu_output.txt.
"""

M = {
 "sync_or_exit_noop": [('''    if require_commit:
        ok, why = sync_state()''', '''    if False:
        ok, why = sync_state()''')],
 "sync_stage1_removed": [('''    sync_or_exit("stage 1", require_commit)''', '''    pass''')],
 "sync_stage2_removed": [('''    sync_or_exit("stage 2", require_commit)          # first: a checkout that is behind is told to pull''', '''    pass''')],
 "sync_untracked_only": [('''top).splitlines() if l.strip().endswith(".json")]''', '''top).splitlines() if l.strip().endswith(".json") and l.startswith("??")]''')],
 "stage2_runs_refused": [('''            for arm in allowed:
                reached.append(arm)''', '''            for arm in arms:
                reached.append(arm)''')],
 "stage1_wrong_arms": [('''card_setup(dev, hw, sha, require_commit, "stage 1",
                                                                             ("ORACLE_PILOT",))''', '''card_setup(dev, hw, sha, require_commit, "stage 1",
                                                                             ("SINGLE",))''')],
 "abort_missing_all": [('''    missing = [s for s, v in d["statuses"].items() if v != "ok"]
    cards = {s: r.get("hw") for s, r in pilot.items()}''', '''    missing = list(PILOT_SEEDS) if not d["complete"] else [s for s, v in d["statuses"].items() if v != "ok"]
    cards = {s: r.get("hw") for s, r in pilot.items()}''')],
 "report_no_reading_pilot": [('''    elif not d["proceed"] and any(R[a] or DIV[a] for a in STAGE2):''', '''    elif False:''')],
 "report_nobody": [('''            if nobody:''', '''            if False:''')],
 "report_no_s2inv": [('''    if s2inv:''', '''    if False:''')],
 "note_not_recorded": [('''        note = dict(path=os.path.relpath(os.path.abspath(code_change_note), HERE), commit=nc,''', '''        note = None and dict(path=os.path.relpath(os.path.abspath(code_change_note), HERE), commit=nc,''')],
 "record_one_invocation": [('''if i.get("sha") == sha), key=lambda i: str(i.get("invocation")))''', '''if i.get("sha") == sha), key=lambda i: str(i.get("invocation")))[-1:]''')],
 "reason_records": [('''dict(stage=2, machine=machine, reason=reason, workers=workers''', '''dict(stage=2, machine=machine, reason=None, workers=workers''')],
 "records_no_bench": [('''workers=workers, bench_commit=bench_commit,
                                 bench_written=bench_written, invocation=INVOCATION)''', '''workers=workers, invocation=INVOCATION)''')],
 "stage2_retry_noop": [('''    if retry_halted:
        for arm in allowed:''', '''    if False:
        for arm in allowed:''')],
 "bench_archive_early": [('''    dev = dv.setup(device)
    mem0 = meminfo_gb()''', '''    dev = dv.setup(device)
    archive_bench(path) if path else None
    mem0 = meminfo_gb()''')],
 "bench_archive_fixed_name": [('''    while os.path.exists(dst):''', '''    while False:''')],
 "stage1_records_no_bench": [('''dict(stage=1, workers=workers, bench_commit=bench_commit,
                                                                      bench_written=bench_written)''', '''dict(stage=1, workers=workers)''')],
 "no_record_at_start": [('''    write_stage1_record(*_stage1_record(sha, note, running=INVOCATION))''', '''    pass''')],
 "note_only_record": [('''    if others and not own:''', '''    if prev and not own:''')],
 "superseded_archive_only": [('''    return cur != sha and sha in shas_with(st or load_all(), ("ORACLE_PILOT",))''', '''    return False''')],
 "lock_noop": [('''        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)''', '''        pass''')],
 "others_noop": [('''    if busy and not other_card_idle:''', '''    if False:''')],
 "idle_ignored": [('''    if busy and not other_card_idle:''', '''    if busy:''')],
 "code_before_noop": [('''            if now != sha:
                raise CodeChanged(f"the code on disk is {now}''', '''            if False:
                raise CodeChanged(f"the code on disk is {now}''')],
 "code_after_noop": [('''            changed = None if now == sha else dict(''', '''            changed = None if True else dict(''')],
 "code_changed_refusal_noop": [('''    cc = code_changed_runs(st, sha)
    if cc:
        raise SystemExit''', '''    cc = code_changed_runs(st, sha)
    if False:
        raise SystemExit''')],
 "report_cc_noop": [('''    if cc and not code_note:''', '''    if False:''')],
 "report_superseded_noop": [('''    elif s2 and superseded(sha, st):''', '''    elif False:''')],
 "running_ignored": [('''    if rec1.get("running"):''', '''    if False:''')],
 "utc_local": [('''    return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())''', '''    return time.strftime("%Y-%m-%d %H:%M:%S")''')],
 "abort_summary_old": [('''    if aborted:
        stt = load_all()
        for arm in allowed:''', '''    if False:
        stt = load_all()
        for arm in allowed:''')],
 "arms_ignored": [('''tuple(a for a in MACHINE_ARMS[machine] if arms is None or a in arms)''', '''tuple(MACHINE_ARMS[machine])''')],
 "ended_not_marked": [('''        st["meta"]["last_invocation"].update(ended=True, at=meta2["ended"], aborted=aborted)''', '''        pass''')],
 "sync_after_record": [('''    sync_or_exit("stage 2", require_commit)          # first: a checkout that is behind is told to pull
''', ''''''), ('''    with open(jp) as f:
        rec1 = json.load(f)
    if rec1.get("running"):''', '''    with open(jp) as f:
        rec1 = json.load(f)
    sync_or_exit("stage 2", require_commit)
    if rec1.get("running"):''')],
}
