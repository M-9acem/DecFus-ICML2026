"""Communication instrumentation for this repo's DecFus, in the reproduced repo's format.

Why this exists
---------------
This codebase emits a single list of accuracy numbers per run and nothing else,
so every statement about *what the method actually did* -- which layers were
exchanged, which were averaged, whether a leaf ever retains its own weights --
has had to be read off the source rather than measured. This module records it.

What is recorded
----------------
Every aggregation decision lands in exactly one of four modes:

  average    -- the tensor was averaged over the closed neighbourhood
  exchange   -- the tensor was replaced wholesale by one peer's copy
  kept_own   -- the exchange branch drew the client itself
  blocked    -- a layer policy refused the transfer before any branch was taken

`counts[sender, receiver, tensor, mode]` mirrors the reproduced repo's
`counts.npz`, so the same analysis code reads both. As there, an averaging event
logs one op *per contributing sender*, which over-counts the receiver column by
the degree; the diagonal does not, so `counts[c, c, t, AVERAGE]` is the event
count while exchange has a single donor and sums over the sender axis directly.

Granularity, and why it differs from the reproduced repo
--------------------------------------------------------
That repo treats a module as one atomic layer: a BatchNorm's weight, bias and
running statistics are exchanged or averaged together. This repo iterates raw
state_dict entries and computes a similarity per *tensor*, so the four tensors
of one BatchNorm can genuinely receive different modes in the same round. The
recording unit here is therefore the tensor, which is the real unit of decision.
`tensor_to_layer` maps each one onto the module-level layer names the other repo
uses, so counts can be summed to that granularity for comparison -- but a layer
will then show several events per round rather than one, and the two totals are
not directly comparable without dividing by the tensor count per layer.
"""

import json
import os

import numpy as np

AVERAGE, EXCHANGE, KEPT_OWN, BLOCKED = 0, 1, 2, 3
MODE_NAMES = ('average', 'exchange', 'kept_own', 'blocked')


class CommunicationRecorder(object):
    """Accumulates per-tensor aggregation events. Disabled instances cost nothing.

    Call `record_*` from inside an aggregation pass, `end_round(iter)` once per
    round, and `save()` at the end of training.
    """

    def __init__(self, out_dir, num_users, param_keys, key_to_layer, meta):
        self.out_dir = out_dir
        self.num_users = num_users
        # Only tensors the aggregation actually iterates over; it skips these.
        self.tensor_names = [k for k in param_keys if 'num_batches_tracked' not in k]
        self.tensor_index = {name: i for i, name in enumerate(self.tensor_names)}
        self.tensor_to_layer = [key_to_layer.get(n, n.rsplit('.', 1)[0] if '.' in n else n)
                                for n in self.tensor_names]
        self.meta = dict(meta)

        t = len(self.tensor_names)
        self.counts = np.zeros((num_users, num_users, t, len(MODE_NAMES)), dtype=np.int64)
        self.round_rows = []
        self._round = {k: 0 for k in MODE_NAMES}
        self._round['bridge_events'] = 0
        self.rounds_recorded = 0

    # ------------------------------------------------------------- recording --
    def record_average(self, receiver, tensor_name, donors):
        """`donors` is the closed neighbourhood -- include the receiver itself."""
        t = self.tensor_index[tensor_name]
        for d in donors:
            self.counts[d, receiver, t, AVERAGE] += 1
        self._round['average'] += len(donors)

    def record_exchange(self, receiver, tensor_name, donor):
        t = self.tensor_index[tensor_name]
        if donor == receiver:
            self.counts[receiver, receiver, t, KEPT_OWN] += 1
            self._round['kept_own'] += 1
            return
        self.counts[donor, receiver, t, EXCHANGE] += 1
        self._round['exchange'] += 1

    def record_blocked(self, receiver, tensor_name):
        t = self.tensor_index[tensor_name]
        self.counts[receiver, receiver, t, BLOCKED] += 1
        self._round['blocked'] += 1

    def note_bridge_event(self, n=1):
        self._round['bridge_events'] += n

    def end_round(self, round_index):
        row = {'round': round_index}
        row.update(self._round)
        self.round_rows.append(row)
        self._round = {k: 0 for k in MODE_NAMES}
        self._round['bridge_events'] = 0
        self.rounds_recorded += 1

    # ---------------------------------------------------------------- output --
    def save(self):
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        np.savez_compressed(
            os.path.join(self.out_dir, 'counts.npz'),
            counts=self.counts,
            tensor_names=np.array(self.tensor_names),
            tensor_to_layer=np.array(self.tensor_to_layer),
            layer_names=np.array(sorted(set(self.tensor_to_layer))),
        )

        meta = dict(self.meta)
        meta.update({
            'codebase': 'official',
            'granularity': 'param_tensor',
            'mode_names': list(MODE_NAMES),
            'num_clients': self.num_users,
            'num_tensors': len(self.tensor_names),
            'num_layers': len(set(self.tensor_to_layer)),
            'rounds_recorded': self.rounds_recorded,
        })
        with open(os.path.join(self.out_dir, 'comm_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2, sort_keys=True)

        cols = ['round'] + list(MODE_NAMES) + ['bridge_events']
        with open(os.path.join(self.out_dir, 'per_round_totals.csv'), 'w') as f:
            f.write(','.join(cols) + '\n')
            for row in self.round_rows:
                f.write(','.join(str(row[c]) for c in cols) + '\n')

        print('[comm_log] wrote %s (%d rounds, %d tensors, %d clients)' % (
            self.out_dir, self.rounds_recorded, len(self.tensor_names), self.num_users))
