#!/usr/bin/env python
"""
explore_gpu/calibrate_judge.py — the lock-step criterion (explore_gpu.checks.Judge) on the cases where two correct float32
paths round differently, at the lengths the CUDA CHECK uses. EXPLORATORY tooling; cpu, 1 thread per case.

    python -m explore_gpu.calibrate_judge        -> explore_gpu/calibrate_judge_output.txt (about 1.5 h)

CASES (each a correct pair of paths that are not bit for bit equal on CPU):
  reg3_scan   the S75 REG3 model (L = 768), the batched register "scan" (what CUDA runs) against the sequential loop,
              seeds 830-831, 300 updates: the length and seeds of K3b's REG3 case on CUDA.
  delta_gdn   one delta channel, learned beta, GDN decay, RandHeaderTask-lite, seeds 160-162, 300 updates (K3b's case).
  twin        arm A 160 against its neuron-permuted twin, 2400 updates (K3c's positive control).
Each line: updates, FAST failures arbitrated in float64, TEST-WORSE (A), REF-WORSE (B), the sign test's p (the case
passes at p >= 1e-4), the worst arbitration score.
"""

import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def run_case(name):
    import torch
    torch.set_num_threads(1)
    from explore_gpu import checks as ck
    from explore_gpu import run as gr
    import explore_i_task as it
    import explore_i_common as ic
    import explore_delta_mem as dm
    import test_short_conv as tsc
    import test_router_reliability as trr
    cpu = torch.device("cpu")
    t0 = time.time()
    if name == "reg3_scan":
        task = it.ICMCTask(**ck.S75_CFG)
        spec = gr.arm_spec("REG3", task.S)
        slow = ic.gate_names(gr.make_model(spec, task, 1))
        r = ck.lockstep(task, lambda s: gr.make_model(spec, task, s), (830, 831), 300, cpu, cpu, ck.TOL_SAME, loss="icmc",
                        slow_gate=slow, register="scan")
    elif name == "delta_gdn":
        r = ck.lockstep(ck.RHL, lambda s: dm.to_delta(ck.build(ck.RHL, tsc.ARM["B"], ck.ARCH, s), beta="learned",
                                                       decay="gdn", seed=s), (160, 161, 162), 300, cpu, cpu, ck.TOL_SAME)
    elif name == "twin":
        to_test, back = ck._perm_twin()
        r = ck.lockstep_pair(tsc.TASK, lambda s: trr.make_fn(ck.ARM_A, s)(), 160, 2400, ck.TOL_SAME, to_test=to_test,
                             grads_back=back)
    else:
        raise ValueError(name)
    r["secs"] = round(time.time() - t0)
    return name, r


def main():
    names = sys.argv[1:] or ["reg3_scan", "delta_gdn", "twin"]
    out = os.path.join(HERE, "explore_gpu", "calibrate_judge_output.txt")
    lines = ["explore_gpu.calibrate_judge (cpu, 1 thread per case); Judge: tolerances same device "
             "(logits 1e-5, grad 1e-4), arbitration factor 4, sign test at 1e-4", ""]
    with mp.get_context("spawn").Pool(len(names)) as pool:
        for name, r in pool.imap_unordered(run_case, names):
            line = (f"{name}: ok {r['ok']}; {r['updates']} run-updates, {r['arbitrated']} arbitrated, A (test worse) "
                    f"{r['test_worse']}, B (reference worse) {r['ref_worse']}, sign p {r['sign_p']:.3g}, non-finite "
                    f"{r['nonfinite']}, worst score {r['worst_arbitration']:.3g}; raw max |dlogits| {r['logits']:.2e}, "
                    f"raw grad {r['grad']:.2e} ({r['grad_param']}); {r['secs']} s; first A: "
                    + json.dumps(r["first_failures"][:3], default=str))
            print(line, flush=True)
            lines.append(line)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
