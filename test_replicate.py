"""Unit tests for replicate.py. Independent replication, EXPLORATORY, not a result."""

import os

import torch
import torch.nn.functional as F

import replicate as R

torch.set_num_threads(1)


# ----------------------------------------------------------------------------- task


def test_grouped_layout_and_target():
    t = R.Task(S=8, P=4)
    b = t.sample(64, torch.Generator().manual_seed(1))
    idx, st, role = b["idx"], b["stream"], b["role"]
    assert idx.shape == (64, 99)
    for i in range(64):
        body = idx[i, :96].view(32, 3)
        # every triple is [CTX_s, KEY, VAL], grouped by key: 4 groups of 8 with one key each
        assert all(int(c) == t.ctx_tok(int(st[i, 3 * j])) for j, c in enumerate(body[:, 0]))
        keys = body[:, 1].view(4, 8)
        assert (keys == keys[:, :1]).all() and len(set(keys[:, 0].tolist())) == 4
        # each group has all 8 streams once, and 8 distinct values
        for g in range(4):
            grp = body[8 * g:8 * g + 8]
            assert sorted((grp[:, 0] - t.ctx_tok(0)).tolist()) == list(range(8))
            assert len(set(grp[:, 2].tolist())) == 8 and (grp[:, 2] < 16).all()
        # query [CTX_q, KEY_q, ?], target is the body's value for (key_q, stream_q)
        q = idx[i, 96:]
        assert int(q[2]) == t.qmark and int(q[0]) == t.ctx_tok(int(b["qstream"][i]))
        hit = (body[:, 0] == q[0]) & (body[:, 1] == q[1])
        assert hit.sum() == 1 and int(body[hit, 2]) == int(b["target"][i])
    assert (role[0, :6] == torch.tensor([0, 1, 2, 0, 1, 2])).all()
    assert role[0, t.query_key_pos] == R.ROLE_KEY


# ----------------------------------------------------------------------------- model


def test_window_gate_cannot_see_t_minus_3():
    torch.manual_seed(0)
    gate = R.WindowGate(D=32, k=16)
    v = torch.randn(1, 20, 32, requires_grad=True)
    g = gate(v)
    t = 12
    grad = torch.autograd.grad(g[0, t].pow(2).sum(), v)[0][0]  # d g_t / d v_s for all s
    norms = grad.norm(dim=-1)
    assert norms[t] > 0 and norms[t - 1] > 0 and norms[t - 2] > 0
    assert (norms[: t - 2] == 0).all() and (norms[t + 1:] == 0).all()
    # and by perturbation through the model's tokens
    task = R.Task()
    m = R.MCBDH(task.vocab)
    idx = task.sample(1, torch.Generator().manual_seed(3))["idx"]
    _, g0 = m(idx, return_gate=True)
    idx2 = idx.clone()
    idx2[0, t - 3] = (idx[0, t - 3] + 1) % task.vocab
    _, g1 = m(idx2, return_gate=True)
    assert torch.equal(g0[0, t], g1[0, t])
    assert not torch.equal(g0[0, t - 1], g1[0, t - 1])


def test_model_is_causal_and_scores_strict():
    task = R.Task()
    torch.manual_seed(0)
    m = R.MCBDH(task.vocab)
    b = task.sample(2, torch.Generator().manual_seed(0))
    idx = b["idx"]
    l0 = m(idx)
    idx2 = idx.clone()
    idx2[:, 50:] = (idx2[:, 50:] + 3) % task.vocab
    l1 = m(idx2)
    assert torch.allclose(l0[:, :50], l1[:, :50], atol=0, rtol=0)
    assert not torch.allclose(l0[:, 50], l1[:, 50])


def test_conv_is_identity_at_init():
    c = R.CausalConv(32)
    x = torch.randn(2, 10, 32)
    assert torch.allclose(c(x), x)


def test_perfect_gate_is_one_hot_on_stream():
    task = R.Task()
    m = R.MCBDH(task.vocab, gate="perfect")
    b = task.sample(4, torch.Generator().manual_seed(0))
    _, g = m(b["idx"], b["stream"], return_gate=True)
    assert torch.equal(g, F.one_hot(b["stream"], 16).float())
    assert R.stream_channel_map(g, b["stream"], b["role"], 8) == list(range(8))


# ----------------------------------------------------------------------------- split


def test_split_probe_schedule():
    s = R.Split()
    assert [u for u in range(1, 20000) if s.is_probe_step(u)] == [2400, 4800, 7200, 9600, 12000, 14400, 16800, 19200]


def test_split_fires_on_plateau_only():
    s = R.Split()
    assert not s.decide(2400, 0.30)  # reference read, never fires before 4800
    assert s.decide(4800, 0.31)  # rose 0.01 < 0.02, below 0.95 -> fire
    s = R.Split()
    s.decide(2400, 0.30)
    assert not s.decide(4800, 0.33)  # rose 0.03 -> no
    s = R.Split()
    s.decide(2400, 0.95)
    assert not s.decide(4800, 0.95)  # at threshold -> no
    s = R.Split()
    s.decide(2400, 0.80)
    assert s.decide(4800, 0.70)  # fell -> fire


def _simulate(s, accs):
    fired = []
    for u, a in accs:
        if s.decide(u, a):
            s.splits.append(dict(update=u))
            fired.append(u)
    return fired


def test_split_spacing_and_cap():
    s = R.Split()
    flat = [(u, 0.5) for u in range(2400, 43201, 2400)]
    assert _simulate(s, flat) == [4800, 9600, 14400]  # >= 4800 apart, at most three
    s = R.Split()
    seq = [(2400, .5), (4800, .5), (7200, .5), (9600, .9), (12000, .9)]
    assert _simulate(s, seq) == [4800, 12000]  # 7200 too close; 9600 rose


def test_split_targets_busiest_and_idlest():
    km = torch.tensor([0.1, 0.4, 0.05, 0.3, 0.15])
    assert R.Split.targets(km) == (1, 2)


def test_split_copy_noise_and_adam_reset():
    torch.manual_seed(0)
    k, H = 16, 20000
    W = torch.nn.Parameter(torch.randn(k, H) * 0.1)
    other = torch.nn.Parameter(torch.randn(5))
    opt = torch.optim.Adam([W, other], lr=1e-3)
    (W.sum() ** 2 + other.sum() ** 2).backward()
    opt.step()
    assert opt.state[W]["exp_avg"].abs().sum() > 0
    other_state = opt.state[other]["exp_avg"].clone()
    before = W.detach().clone()
    s = R.Split()
    s.apply(W, opt, c_star=3, c0=7, gen=torch.Generator().manual_seed(0))
    sd = before[3].std()
    d0 = W.detach()[7] - before[3]
    d1 = W.detach()[3] - before[3]
    assert abs(float(d0.std() / sd) - 0.1) < 0.005 and abs(float(d1.std() / sd) - 0.1) < 0.005
    assert not torch.equal(d0, d1)  # independent noise on both rows
    keep = [i for i in range(k) if i not in (3, 7)]
    assert torch.equal(W.detach()[keep], before[keep])
    assert all((v == 0).all() for v in opt.state[W].values() if torch.is_tensor(v))
    assert torch.equal(opt.state[other]["exp_avg"], other_state)


def test_split_in_training_loop_and_pairing():
    kw = dict(S=2, P=2, k=4, budget=60, eval_every=10, n_eval=64, slow_until=20)
    sk = dict(first=20, every=10, min_gap=20)
    a = R.run("WIN3_SPLIT_D8", 5, split_kw=sk, **kw)
    b = R.run("WIN3_SLOW_D8", 5, **kw)
    ups = [x["update"] for x in a["splits"]]
    assert ups and all(u2 - u1 >= 20 for u1, u2 in zip(ups, ups[1:])) and len(ups) <= 3
    assert all(x["c_star"] != x["c0"] for x in a["splits"])
    # identical training up to the first split (paired arms)
    pre = [e for e in a["evals"] if e[0] <= ups[0]]
    assert pre == b["evals"][: len(pre)]


def test_resume_is_bit_for_bit(tmp_path):
    kw = dict(S=2, P=2, k=4, budget=40, eval_every=10, n_eval=64, slow_until=20,
              split_kw=dict(first=20, every=10, min_gap=20))
    full = R.run("WIN3_SPLIT_D8", 7, **kw)
    ck = str(tmp_path / "c.pt")
    R.run("WIN3_SPLIT_D8", 7, ckpt=ck, stop_after=20, **kw)
    assert os.path.exists(ck)
    resumed = R.run("WIN3_SPLIT_D8", 7, ckpt=ck, **kw)
    for key in ("evals", "probes", "splits", "map", "final_acc", "per_stream_acc"):
        assert full[key] == resumed[key], key


# ----------------------------------------------------------------------------- outcomes


def test_bound_and_transition():
    B = 43200
    assert R.bound_and_transition([(1200, .5), (2400, .96), (3600, .97), (4800, .98)], B) == (True, 2400)
    assert R.bound_and_transition([(1200, .96), (2400, .5), (3600, .96), (4800, .97), (6000, .99)], B) == (True, 3600)
    ev = [(u, .5) for u in range(1200, B, 1200)] + [(B, .96)]
    assert R.bound_and_transition(ev, B) == (True, B)  # single final evaluation at the budget
    assert R.bound_and_transition([(1200, .5), (2400, .96)], B) == (False, None)
    assert R.bound_and_transition([(1200, .96), (2400, .97), (3600, .5)], B) == (False, None)


def _hand_gate(b, chan_of_stream, key_chan=None, k=16):
    """Gate one-hot on chan_of_stream[stream] at value positions; at key positions all on key_chan."""
    st, role = b["stream"], b["role"]
    ch = torch.tensor(chan_of_stream)[st]
    if key_chan is not None:
        ch = torch.where(role == R.ROLE_KEY, torch.full_like(ch, key_chan), ch)
    return F.one_hot(ch, k).float() * 0.9 + 0.1 / k


def test_bound_routed_on_hand_made_gates():
    task = R.Task()
    b = task.sample(256, torch.Generator().manual_seed(0))
    acc = [0.97] * 8
    # one channel per stream at value positions (key positions all on channel 0): routed
    g = _hand_gate(b, [2, 4, 6, 8, 10, 12, 14, 15], key_chan=0)
    m = R.stream_channel_map(g, b["stream"], b["role"], 8)
    assert m == [2, 4, 6, 8, 10, 12, 14, 15]
    assert R.bound_routed(True, m, acc)
    assert not R.bound_routed(False, m, acc)  # not bound
    assert not R.bound_routed(True, m, [0.97] * 7 + [0.89])  # one stream below 0.9
    # two streams merged on one channel: not routed
    g = _hand_gate(b, [2, 2, 6, 8, 10, 12, 14, 15])
    m = R.stream_channel_map(g, b["stream"], b["role"], 8)
    assert not R.bound_routed(True, m, acc)
    assert R.failure_class(True, m, 0.9, acc) == "merged(2 share)"
    # routed at key positions but merged at value positions: measured at values -> not routed
    gk = _hand_gate(b, [0] * 8)
    sep = _hand_gate(b, list(range(8)))
    g = torch.where((b["role"] == R.ROLE_KEY).unsqueeze(-1), sep, gk)
    assert not R.bound_routed(True, R.stream_channel_map(g, b["stream"], b["role"], 8), acc)
    assert R.failure_class(False, [0] * 8, 0.1, acc) == "collapsed"
