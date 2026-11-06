"""Checks on the communication recorder. Run from the repo root:

    python tests/test_comm_log.py

Needs no dataset, no GPU and no swanlab account. Two properties are asserted:

1. The four modes are *exhaustive* -- every (client, tensor, round) decision is
   counted exactly once, so the totals cannot silently drift from what the
   aggregation did. The expected total differs by pass: the standard pass covers
   every client, while the hybrid's stage 1 covers every client and its stage 2
   only the two hubs.
2. Recording changes *nothing*. The resulting client models are compared
   tensor-by-tensor against a run with `recorder=None` under the same seed, so
   instrumentation cannot perturb the experiment it is measuring.

It also prints the leaf-retention measurement that motivated the work.
"""
import copy, random, sys, types
sys.path.insert(0, ".")
sys.modules.setdefault("swanlab", types.SimpleNamespace(log=lambda *a, **k: None,
                                                        init=lambda *a, **k: None))
import numpy as np, torch

# Seed before the network is built: its initial weights are the base every
# client is perturbed from, so an unseeded init makes the whole run vary.
torch.manual_seed(0)
from Algorithm.Training_DecFus import (standard_aggregate, hybrid_aggregate,
                                       build_two_cluster_topology, build_layer_groups,
                                       resolve_layer_selection)
from Algorithm.comm_log import CommunicationRecorder, AVERAGE, EXCHANGE, KEPT_OWN, BLOCKED
from models.Nets import ResNet8

N, ROUNDS = 20, 6
net = ResNet8(types.SimpleNamespace(num_classes=10, num_channels=3))
topo, meta = build_two_cluster_topology(N)
param_keys = list(net.state_dict().keys())
_, key_to_layer = build_layer_groups(param_keys)
T = len([k for k in param_keys if 'num_batches_tracked' not in k])
args = types.SimpleNamespace(num_users=N, exchange_prob=0.4)

def fresh():
    torch.manual_seed(0)
    out = []
    for _ in range(N):
        sd = copy.deepcopy(net.state_dict())
        for k in param_keys:
            if 'num_batches_tracked' not in k:
                sd[k] = sd[k].float() + torch.randn_like(sd[k].float()) * 0.1
        out.append(sd)
    return out

def drive(fn, allowed, rec, **kw):
    random.seed(7); torch.manual_seed(7)
    models = fresh()
    for it in range(ROUNDS):
        models = fn(args=args, client_models=models, param_keys=param_keys,
                    client_thresholds=[0.75]*N, allowed_layers=allowed,
                    key_to_layer=key_to_layer, recorder=rec, **kw)
        if rec is not None: rec.end_round(it)
    return models

cases = [
    ("standard, full",    standard_aggregate, None,  N*T*ROUNDS,
     dict(network_topology=topo, sample_nums=[100]*N)),
    ("standard, first_1", standard_aggregate,
     resolve_layer_selection(net, param_keys, "first_1")[0], N*T*ROUNDS,
     dict(network_topology=topo, sample_nums=[100]*N)),
    # stage 1 covers every client, stage 2 only the two hubs
    ("hybrid, full",      hybrid_aggregate,   None,  (N + 2)*T*ROUNDS,
     dict(network_topology=topo, topology_meta=meta)),
]

ok = True
for label, fn, allowed, expected, kw in cases:
    rec = CommunicationRecorder("/tmp/_x", N, param_keys, key_to_layer, {})
    with_rec = drive(fn, allowed, rec, **kw)
    c = rec.counts; diag = np.einsum("iitm->itm", c)
    total = int(diag[:, :, AVERAGE].sum() + c[:, :, :, EXCHANGE].sum()
                + diag[:, :, KEPT_OWN].sum() + diag[:, :, BLOCKED].sum())
    exhaustive = total == expected

    without = drive(fn, allowed, None, **kw)
    identical = all(torch.equal(a[k], b[k]) for a, b in zip(with_rec, without) for k in param_keys)

    ok &= exhaustive and identical
    print(f"{label:20s} events {total:6d}/{expected:<6d} exhaustive={exhaustive}  "
          f"unchanged-without-recorder={identical}")

print("\nDoes a degree-1 leaf retain its own tensors on THIS codebase?")
rec = CommunicationRecorder("/tmp/_x", N, param_keys, key_to_layer, {})
drive(standard_aggregate, None, rec, network_topology=topo, sample_nums=[100]*N)
d = np.einsum("iitm->itm", rec.counts)
for leaf0 in (9, 19):
    kept = int(d[leaf0, :, KEPT_OWN].sum())
    exch = int(rec.counts[:, leaf0, :, EXCHANGE].sum())
    avg  = int(d[leaf0, :, AVERAGE].sum())
    print(f"  leaf {leaf0+1:2d}: averaged={avg:4d}  exchanged={exch:4d}  kept_own={kept:4d}"
          f"   (reproduced repo measures kept_own = 0)")
print("\nALL CHECKS PASS" if ok else "\nFAILURE")
