"""Checks on the random four-cluster graph: loading it, hybrid over gateways, and
isolated clients. Run from the repo root:

    python tests/test_four_cluster.py

Needs no dataset, no GPU and no swanlab account.

The graph below is hand-built so every expectation can be written down:

    cluster A = {0,1,2,3}   cluster B = {4,5,6,7}      (0-indexed)
    intra:  0-1  1-2  |  4-5
    inter:  3-4  2-5  2-6
    client 7 has no links at all.

    gateways: 2 (peers 5, 6), 3 (peer 4), 4 (peer 3), 5 (peer 2), 6 (peer 2)
    so client 2 is a gateway to *two* peers, and 7 is isolated.
"""
import json, os, random, sys, tempfile, types
sys.path.insert(0, ".")
sys.modules.setdefault("swanlab", types.SimpleNamespace(log=lambda *a, **k: None,
                                                        init=lambda *a, **k: None))
import numpy as np, torch

from Algorithm.Training_DecFus import (standard_aggregate, hybrid_aggregate,
                                       build_graph_file_topology)
from Algorithm.comm_log import CommunicationRecorder, EXCHANGE, KEPT_OWN

N = 8
EDGES = [(0, 1), (1, 2), (4, 5), (3, 4), (2, 5), (2, 6)]
CLUSTER_OF = ["A"] * 4 + ["B"] * 4
KEYS = ["conv.weight", "bn.weight", "bn.bias", "bn.num_batches_tracked", "fc.weight", "fc.bias"]
SHAPES = {"conv.weight": (4, 3, 3, 3), "fc.weight": (10, 8)}
k2l = {k: k.rsplit(".", 1)[0] for k in KEYS}


def write_graph():
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump({"num_clients": N, "adjacency_edges": [list(e) for e in EDGES],
                   "cluster_of": CLUSTER_OF, "seed": 123}, f)
    return path


def models(seed=0):
    torch.manual_seed(seed)
    out = []
    for _ in range(N):
        m = {}
        for k in KEYS:
            m[k] = torch.tensor(3) if "num_batches" in k else torch.randn(*SHAPES.get(k, (4,)))
        out.append(m)
    return out


def recorder():
    return CommunicationRecorder("/tmp/_unused", N, KEYS, k2l, {})


def args(p=0.4):
    return types.SimpleNamespace(num_users=N, exchange_prob=p)


path = write_graph()
topo, meta = build_graph_file_topology(path, N)

# ---- loading ---------------------------------------------------------------
assert topo[1] == [2], topo[1]                       # client 0 <-> 1 (1-indexed 1 <-> 2)
assert topo[3] == [2, 6, 7], topo[3]                 # client 2: 1, 5, 6
assert topo[8] == [], topo[8]                        # client 7 is isolated
assert all((b + 1) in topo[a + 1] and (a + 1) in topo[b + 1] for a, b in EDGES)  # symmetric
assert meta["clusters"][1] == "A" and meta["clusters"][5] == "B"
assert meta["num_isolated"] == 1 and meta["seed"] == 123
try:
    build_graph_file_topology(path, N + 4)
    raise SystemExit("FAIL: a client-count mismatch must be rejected")
except ValueError:
    pass
print("PASS  graph loads: 1-indexed, symmetric, clusters and isolation recorded")

# ---- isolated client under the standard pass -------------------------------
ms = models()
rec = recorder()
random.seed(0)
out = standard_aggregate(args(), ms, topo, KEYS, [0.5] * N, [100] * N, None, {}, recorder=rec)
for k in KEYS:
    assert torch.equal(out[7][k], ms[7][k]), ("isolated client changed", k)
t = len([k for k in KEYS if "num_batches" not in k])
assert rec.counts[7, 7, :, KEPT_OWN].sum() == t, "isolated client not logged as kept_own"
assert rec.counts[:, 7, :, :].sum() == t and rec.counts[7, :, :, :].sum() == t, \
    "isolated client must send and receive nothing"
print("PASS  isolated client keeps its model under the standard pass, and logs kept_own only")

# ---- hybrid: stage 1 never crosses a cluster edge --------------------------
# threshold 1.0 + exchange_prob 0 => stage 2 keeps stage 1 for every tensor of
# every gateway whose fallback pool is just itself; client 2 may still draw a
# peer, so compare only clients whose result is fully determined by stage 1.
ms = models(1)
random.seed(0)
out = hybrid_aggregate(args(0.0), ms, topo, meta, KEYS, [1.0] * N, None, {}, recorder=None)
def intra_mean(i, key):
    members = [i] + [n - 1 for n in topo[i + 1] if CLUSTER_OF[n - 1] == CLUSTER_OF[i]]
    return sum(ms[m][key].float() for m in members) / len(members)
for i in (0, 1, 7):  # no inter-cluster link: pure intra-cluster mean (7: itself)
    for k in KEYS:
        if "num_batches" in k:
            continue
        assert torch.allclose(out[i][k], intra_mean(i, k), atol=1e-6), (i, k)
print("PASS  clients without an inter-cluster link get exactly their intra-cluster mean")

# ---- hybrid: a single-peer gateway exchanges with that peer ----------------
random.seed(0)
rec = recorder()
out = hybrid_aggregate(args(1.0), ms, topo, meta, KEYS, [0.0] * N, None, {}, recorder=rec)
# threshold 0 => the least-similar tensor is below cutoff and is exchanged with
# the peer; that peer's stage-1 value must arrive intact.
peer_of_3 = 4
stage1_4 = intra_mean(peer_of_3, "fc.weight")
got = out[3]["fc.weight"]
averaged_with_peer = (intra_mean(3, "fc.weight") + stage1_4) / 2
assert torch.allclose(got, stage1_4, atol=1e-6) or torch.allclose(got, averaged_with_peer, atol=1e-6), \
    "client 3's result must come from itself and its only inter-cluster peer (client 4)"
print("PASS  single-peer gateway mixes only with its one inter-cluster peer")

# ---- hybrid: multi-peer gateway averages over itself AND both peers --------
random.seed(0)
rec = recorder()
out = hybrid_aggregate(args(0.0), ms, topo, meta, KEYS, [0.0] * N, None, {}, recorder=rec)
s1 = {i: {k: intra_mean(i, k) for k in KEYS if "num_batches" not in k} for i in range(N)}
avg_expected = lambda k: (s1[2][k] + s1[5][k] + s1[6][k]) / 3.0
hits = [k for k in KEYS if "num_batches" not in k
        and torch.allclose(out[2][k], avg_expected(k), atol=1e-5)]
assert hits, "client 2 averaged no tensor over {itself, 5, 6} -- multi-peer averaging is broken"
# and anything it recorded as an average names exactly itself + both peers
avg_donors = {d for d in range(N) for t_ in range(rec.counts.shape[2])
              if rec.counts[d, 2, t_, 0] > 0}
assert {2, 5, 6} <= avg_donors, avg_donors
assert not ({4, 7} & avg_donors), "client 2 must never receive from 4 or 7 directly"
print("PASS  multi-peer gateway averages over itself and both inter-cluster peers (%d tensors)" % len(hits))

# ---- hybrid: an exchange only ever picks the gateway's own peers -----------
for seed in range(20):
    random.seed(seed)
    rec = recorder()
    hybrid_aggregate(args(0.5), models(seed), topo, meta, KEYS, [0.0] * N, None, {}, recorder=rec)
    donors = {d for d in range(N) for t_ in range(rec.counts.shape[2])
              if rec.counts[d, 2, t_, EXCHANGE] > 0}
    assert donors <= {2, 5, 6}, ("client 2 exchanged with a non-peer", seed, donors)
print("PASS  across 20 seeds, client 2 only ever exchanges with itself, 5 or 6")

# ---- hybrid: isolated / non-gateway clients log nothing at stage 2 ---------
random.seed(0)
rec = recorder()
hybrid_aggregate(args(0.4), models(2), topo, meta, KEYS, [0.3] * N, None, {}, recorder=rec)
for i in (0, 1, 7):
    assert rec.counts[:, i, :, EXCHANGE].sum() == 0 and rec.counts[i, :, :, KEPT_OWN].sum() == 0, i
print("PASS  clients with no inter-cluster link never take part in stage 2")

os.remove(path)
print("\nALL PASS")
