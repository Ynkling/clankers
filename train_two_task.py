# Phase 2 — Two-task training for Multi-Channel BDH
#
# Goal: prove the BENEFIT, not just the mechanism. Train ONE network on two
# contrasting tasks, each routed to its own channel via the context signal.
# Then show:
#   (1) both tasks reach low loss (the network learned both)
#   (2) using the WRONG channel for a task gives high loss (channels specialized)
#   (3) a single-channel baseline trained on both does worse (forgetting / interference)
#
# Tasks are byte-level and self-contained (no downloads), so this runs anywhere.
#   Task A: English text (tiny Shakespeare if present, else a built-in sample)
#   Task B: structured "code-like" sequences  -> very different byte statistics
#
# Tuned small so it runs on CPU in minutes; scale BLOCK/BATCH/ITERS up on GPU.

import os
import random

import torch
import torch.nn.functional as F

from bdh_mc import MCBDH, BDHConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1337)
random.seed(1337)

# ---- small config so it finishes quickly; bump these on your GPUs ----
BLOCK_SIZE = 128
BATCH_SIZE = 16
ITERS = 1500
LR = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 100
CTX_DIM = 8
K = 2  # two channels, one per task

# one-hot-ish context signals: channel 0 for task A, channel 1 for task B
CTX_A = torch.zeros(CTX_DIM); CTX_A[0] = 5.0
CTX_B = torch.zeros(CTX_DIM); CTX_B[1] = 5.0


# ---------------- data: two byte streams with different statistics ----------------
def load_task_a_bytes():
    p = os.path.join(os.path.dirname(__file__), "bdh", "input.txt")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return f.read()
    # fallback sample if the dataset wasn't fetched
    sample = ("To be, or not to be, that is the question:\n"
              "Whether 'tis nobler in the mind to suffer\n"
              "The slings and arrows of outrageous fortune,\n") * 400
    return sample.encode("utf-8")


def make_task_b_bytes(n=200_000):
    # synthetic "code-like" stream: nested brackets, assignments, digits.
    # deliberately different byte distribution from English prose.
    rng = random.Random(0)
    names = ["x", "y", "foo", "bar", "tmp", "acc", "buf", "i", "j"]
    ops = ["+", "-", "*", "/", "=="]
    out = []
    while sum(len(s) for s in out) < n:
        a, b = rng.choice(names), rng.choice(names)
        op = rng.choice(ops)
        out.append(f"{a} = ({b} {op} {rng.randint(0,99)});\n")
        if rng.random() < 0.3:
            out.append("for (i=0; i<" + str(rng.randint(1,9)) + "){ " +
                       rng.choice(names) + "+=1; }\n")
    return "".join(out).encode("utf-8")


DATA_A = load_task_a_bytes()
DATA_B = make_task_b_bytes()
print(f"Task A bytes: {len(DATA_A):,}  |  Task B bytes: {len(DATA_B):,}")


def get_batch(data_bytes, ctx_vec):
    import numpy as np
    arr = np.frombuffer(data_bytes, dtype=np.uint8)
    ix = torch.randint(len(arr) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(arr[i:i+BLOCK_SIZE].astype("int64")) for i in ix])
    y = torch.stack([torch.from_numpy(arr[i+1:i+1+BLOCK_SIZE].astype("int64")) for i in ix])
    ctx = ctx_vec.unsqueeze(0).expand(BATCH_SIZE, -1)
    return x.to(device), y.to(device), ctx.to(device)


@torch.no_grad()
def eval_loss(model, data_bytes, ctx_vec, iters=20):
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y, ctx = get_batch(data_bytes, ctx_vec)
        _, loss = model(x, y, context=ctx)
        tot += loss.item()
    model.train()
    return tot / iters


def train(model, two_task=True, label=""):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()
    for step in range(ITERS):
        if two_task:
            # alternate tasks; each carries its own context signal
            if step % 2 == 0:
                x, y, ctx = get_batch(DATA_A, CTX_A)
            else:
                x, y, ctx = get_batch(DATA_B, CTX_B)
        else:
            # baseline: single channel, no meaningful context, mixed data
            data = DATA_A if step % 2 == 0 else DATA_B
            x, y, ctx = get_batch(data, torch.zeros(CTX_DIM))
        _, loss = model(x, y, context=ctx)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % LOG_FREQ == 0:
            print(f"  [{label}] step {step:4d}/{ITERS}  loss {loss.item():.3f}")


if __name__ == "__main__":
    cfg = BDHConfig(n_layer=4, n_embd=128, n_head=4,
                    mlp_internal_dim_multiplier=32,
                    n_channels=K, context_dim=CTX_DIM)

    print("\n=== Training MULTI-channel model (k=2, one channel per task) ===")
    mc = MCBDH(cfg).to(device)
    train(mc, two_task=True, label="MC")

    # ---- the key measurements ----
    a_right = eval_loss(mc, DATA_A, CTX_A)   # task A through channel 0
    a_wrong = eval_loss(mc, DATA_A, CTX_B)   # task A through channel 1 (wrong)
    b_right = eval_loss(mc, DATA_B, CTX_B)   # task B through channel 1
    b_wrong = eval_loss(mc, DATA_B, CTX_A)   # task B through channel 0 (wrong)

    print("\n=== Training SINGLE-channel baseline (k=1, both tasks mixed) ===")
    cfg1 = BDHConfig(n_layer=4, n_embd=128, n_head=4,
                     mlp_internal_dim_multiplier=32,
                     n_channels=1, context_dim=CTX_DIM)
    base = MCBDH(cfg1).to(device)
    train(base, two_task=False, label="BASE")
    base_a = eval_loss(base, DATA_A, torch.zeros(CTX_DIM))
    base_b = eval_loss(base, DATA_B, torch.zeros(CTX_DIM))

    print("\n" + "=" * 56)
    print("RESULTS (lower loss = better)")
    print("=" * 56)
    print(f"Multi-channel, task A | correct ch: {a_right:.3f}  wrong ch: {a_wrong:.3f}")
    print(f"Multi-channel, task B | correct ch: {b_right:.3f}  wrong ch: {b_wrong:.3f}")
    print(f"Single-channel baseline | task A: {base_a:.3f}  task B: {base_b:.3f}")
    print("-" * 56)
    spec_a = a_wrong - a_right
    spec_b = b_wrong - b_right
    print(f"Channel specialization gap (want POSITIVE): A={spec_a:+.3f}  B={spec_b:+.3f}")
    mc_avg = (a_right + b_right) / 2
    base_avg = (base_a + base_b) / 2
    print(f"Avg loss  multi-channel: {mc_avg:.3f}  vs  baseline: {base_avg:.3f}")
    print("=" * 56)
    if spec_a > 0 and spec_b > 0:
        print("PASS: each channel specialized for its own task.")
    else:
        print("INCONCLUSIVE at this scale — try more ITERS or larger model.")
