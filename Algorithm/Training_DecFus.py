import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import copy
import numpy as np
import random
import swanlab
from models.Update import DatasetSplit
from utils.utils import save_result
from models.test import test_img
import math
from collections import deque


class DecFusClient(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs),
                                    batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, current_lr):
        net.to(self.args.device)
        net.train()

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=current_lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=current_lr)

        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                model_output = net(images)
                loss = self.loss_func(model_output['output'], labels)
                loss.backward()
                optimizer.step()

        return net.state_dict()


class VarianceStabilityDetector:
    """Variance stability detector based on stability relative to a reference value."""

    def __init__(self, window_size=8, stability_threshold=0.1, protection_rounds=30, ref_start_round=0.1, ref_end_round=0.2):
        self.window_size = window_size
        self.stability_threshold = stability_threshold  # stability threshold
        self.protection_rounds = protection_rounds
        self.variance_history = deque(maxlen=window_size)
        self.is_stable = False
        self.stable_round = None

        self.ref_start_round = ref_start_round
        self.ref_end_round = ref_end_round

        self.reference_variances = []
        self.reference_mean = None



    def update(self, current_variance, current_round):
        """Update the variance history and check for stability."""
        self.variance_history.append(current_variance)

        # Build the reference baseline (mean variance over the reference rounds)
        if self.ref_start_round <= current_round <= self.ref_end_round:
            self.reference_variances.append(current_variance)
        if self.reference_mean is None and current_round > self.ref_end_round:
            # Make sure data was collected during the reference period
            if self.reference_variances:
                self.reference_mean = np.mean(self.reference_variances)
            else:
                # Fallback: if no data was collected for some reason (e.g. the window is too small),
                # use a variance from the history as the reference so that the algorithm can continue
                if self.variance_history:
                    self.reference_mean = self.variance_history[5]

            # Once the reference is set, clamp it from below so that it cannot be too small
            if self.reference_mean is not None:
                self.reference_mean = max(self.reference_mean, 1e-6)

        # Never declare stability during the protection period
        if current_round < self.protection_rounds:
            return False

        # Once stable, stay stable
        if self.is_stable:
            return True

        # Check for stability
        if len(self.variance_history) >= self.window_size and self.reference_mean is not None:
            variances = list(self.variance_history)

            # Standard deviation over the current window
            current_std = np.std(variances)

            # Relative stability: ratio of the standard deviation to the reference mean
            relative_stability = current_std / self.reference_mean

            # Stable once the relative stability drops below the threshold
            if relative_stability < self.stability_threshold:
                self.is_stable = True
                self.stable_round = current_round
                return True

        return False

    def get_relative_stability(self):
        """Return the current relative stability value."""
        if len(self.variance_history) >= self.window_size and self.reference_mean is not None:
            variances = list(self.variance_history)
            current_std = np.std(variances)
            return current_std / self.reference_mean
        return float('inf')


def adaptive_sigmoid_threshold_decay(current_round, args, stability_detector):
    """
    Adaptive cutoff decay driven by variance stability.

    Args:
    current_round: current training round
    args: argument object
    stability_detector: variance stability detector
    """
    initial_threshold = args.initial_threshold
    final_threshold = args.final_threshold
    total_epochs = args.epochs

    if stability_detector.is_stable:
        # After stability is detected, the decay progress is (t - t_s) / (kappa * T), as in Eq. 6 of the paper
        stable_round = stability_detector.stable_round * args.test_round
        rounds_since_stable = current_round - stable_round
        descent_rounds = total_epochs * args.variance_descent_ratio
        descent_progress = rounds_since_stable / descent_rounds

        # Apply the sigmoid function
        steepness = args.sigmoid_steepness
        midpoint = args.sigmoid_midpoint

        sigmoid_value = 1 / (1 + math.exp(-steepness * (descent_progress - midpoint)))
        current_threshold = initial_threshold - sigmoid_value * (initial_threshold - final_threshold)
    else:
        # Before stability is detected, keep the initial cutoff (the detector never fires during the protection period)
        descent_progress = 0.0
        current_threshold = initial_threshold

    # Record for logging
    args.current_threshold = current_threshold
    args.virtual_progress = descent_progress

    return current_threshold


def build_bridge_topology(n: int) -> dict:
    """Two-cluster "bridge" topology (1-indexed client ids, matching this repo's
    convention), for any even n >= 4. Clients 1,2 are hubs (degree
    leaves_per_hub+1); clients 3..n are leaves (degree 1), split evenly between
    the two hubs. Mirrors our own reproduction's
    decfus/topology/registry.py:bridge_topology(n) node-for-node."""
    if n < 4 or n % 2 != 0:
        raise ValueError(f"build_bridge_topology needs an even n >= 4, got {n}")
    leaves_per_hub = (n - 2) // 2
    hub_a, hub_b = 1, 2
    leaves_a = list(range(3, 3 + leaves_per_hub))
    leaves_b = list(range(3 + leaves_per_hub, n + 1))

    topo = {hub_a: [hub_b] + leaves_a, hub_b: [hub_a] + leaves_b}
    for leaf in leaves_a:
        topo[leaf] = [hub_a]
    for leaf in leaves_b:
        topo[leaf] = [hub_b]
    return topo


def build_two_cluster_topology(n=20):
    """Two equal star clusters joined by a single hub-to-hub bridge
    (1-indexed client ids, this repo's convention).

    Each cluster is a pure star: every non-hub client is a degree-1 leaf wired
    only to its own hub, with no intra-cluster connections between leaves. The
    two hubs are joined by the single bridge edge, so every cross-cluster path
    must traverse both hubs.

    For n=20: Cluster A = 1..10 (hub 1, leaves 2..10), Cluster B = 11..20
    (hub 11, leaves 12..20), bridge 1<->11. Matches
    `decfus/topology/registry.py:two_cluster(n)` in the decfus-reproduced repo
    node-for-node after a +1 index shift.
    """
    if n < 8 or n % 4 != 0:
        raise ValueError("build_two_cluster_topology needs n divisible by 4 and >= 8, got {}".format(n))
    half = n // 2
    cluster_a = list(range(1, half + 1))
    cluster_b = list(range(half + 1, n + 1))
    hub_a, hub_b = cluster_a[0], cluster_b[0]
    leaves_a, leaves_b = cluster_a[1:], cluster_b[1:]

    adj = {i: set() for i in range(1, n + 1)}

    def add(i, j):
        adj[i].add(j)
        adj[j].add(i)

    for hub, leaves in ((hub_a, leaves_a), (hub_b, leaves_b)):
        for leaf in leaves:
            add(hub, leaf)
    add(hub_a, hub_b)  # the bridge: the only inter-cluster edge

    topo = {i: sorted(adj[i]) for i in range(1, n + 1)}
    meta = {
        'clusters': dict([(i, 'A') for i in cluster_a] + [(i, 'B') for i in cluster_b]),
        'roles': dict(
            [(i, 'leaf') for i in leaves_a + leaves_b]
            + [(hub_a, 'hub'), (hub_b, 'hub')]
        ),
        'hubs': [hub_a, hub_b],
        'leaves': leaves_a + leaves_b,
        'bridge': (hub_a, hub_b),
        'cluster_members': {'A': cluster_a, 'B': cluster_b},
        'tracked': {'hub_A': hub_a, 'hub_B': hub_b,
                    'leaf_A': leaves_a[-1], 'leaf_B': leaves_b[-1]},
    }
    return topo, meta


def build_layer_groups(param_keys):
    """Group this repo's flat state_dict keys into logical layers.

    This repo has no notion of a "layer" -- it iterates raw state_dict entries,
    so a BatchNorm's weight/bias/running_mean/running_var are four independent
    things. The restricted-exchange experiments need a layer concept, and it has
    to be *the same* concept the decfus-reproduced repo uses or "first 3 layers"
    would mean different things in the two codebases.

    Mirrors LayerPartition's `module` granularity: group by the state_dict key's
    parent scope (everything up to the last '.'). Non-floating buffers such as
    num_batches_tracked are excluded from exchange entirely, as they are there.

    Returns (layer_names_sorted, key_to_layer) where layer_names_sorted is the
    alphabetical grouping order -- depth order is applied separately.
    """
    key_to_layer = {}
    for key in param_keys:
        if 'num_batches_tracked' in key:
            continue
        key_to_layer[key] = key.rsplit('.', 1)[0] if '.' in key else key
    return sorted(set(key_to_layer.values())), key_to_layer


def layer_forward_order(net, layer_names, input_shape=(1, 3, 32, 32)):
    """Order logical layers by when they actually execute in a forward pass.

    Alphabetical order is wrong for anything that talks about "first"/"last"
    layers: a ResNet's classifier head `fc` sorts near the front, ahead of every
    layerN.* block. One hooked forward pass recovers the true order.
    """
    order = {}
    counter = {'n': 0}
    handles = []

    def make_hook(name):
        def hook(_m, _i, _o):
            if name not in order:
                order[name] = counter['n']
                counter['n'] += 1
        return hook

    for name, module in net.named_modules():
        if name:
            handles.append(module.register_forward_hook(make_hook(name)))

    was_training = net.training
    net.eval()
    try:
        device = next(net.parameters()).device
        with torch.no_grad():
            net(torch.zeros(*input_shape, device=device))
    finally:
        for h in handles:
            h.remove()
        net.train(was_training)

    def position(layer_name):
        prefix = layer_name + '.'
        hits = [p for m, p in order.items() if m == layer_name or m.startswith(prefix)]
        return (min(hits) if hits else len(order) + 1, layer_name)

    return sorted(layer_names, key=position)


def resolve_layer_selection(net, param_keys, policy):
    """Layer names a policy permits to be transmitted.

    policy: 'full' | 'first_<k>' | 'last_<k>', taken in forward-execution order.
    """
    layer_names, key_to_layer = build_layer_groups(param_keys)
    if policy in (None, '', 'full'):
        return set(layer_names), key_to_layer

    ordered = layer_forward_order(net, layer_names)
    kind, _, k_str = policy.partition('_')
    if kind not in ('first', 'last') or not k_str.isdigit():
        raise ValueError("layer policy must be 'full', 'first_<k>' or 'last_<k>', got %r" % policy)
    k = int(k_str)
    if not 1 <= k <= len(ordered):
        raise ValueError('policy %r selects %d of %d layers' % (policy, k, len(ordered)))
    chosen = ordered[:k] if kind == 'first' else ordered[-k:]
    return set(chosen), key_to_layer


def standard_aggregate(args, client_models, network_topology, param_keys,
                       client_thresholds, sample_nums, allowed_layers, key_to_layer):
    """This repo's original DecFus aggregation pass, lifted verbatim out of the
    round loop so family B can substitute a different one without an early
    `continue` skipping the variance/cutoff update that follows it.

    The single behavioural addition is the restricted-layer guard; with
    `allowed_layers=None` (the default, 'full' policy) this is byte-identical to
    the original code.
    """
    new_client_models = []
    for idx in range(args.num_users):
        neighbors = network_topology[idx + 1]
        current_model = client_models[idx]
        new_model = copy.deepcopy(current_model)

        # Use the stored cutoff (only updated every test_round rounds)
        individual_threshold = client_thresholds[idx]

        # For each parameter tensor, find the least similar neighbor
        lowest_similarities = []

        for param_name in param_keys:
            if 'num_batches_tracked' in param_name:
                continue

            param_similarities = []
            for neighbor in neighbors:
                neighbor_model = client_models[neighbor - 1]
                similarity = compute_parameter_similarity(
                    current_model[param_name],
                    neighbor_model[param_name]
                )
                param_similarities.append({
                    'neighbor': neighbor,
                    'param_name': param_name,
                    'similarity': similarity
                })

            lowest_sim_item = min(param_similarities, key=lambda x: x['similarity'])
            lowest_similarities.append(lowest_sim_item)

        if lowest_similarities:
            similarity_values = [item['similarity'] for item in lowest_similarities]
            threshold = np.percentile(similarity_values, individual_threshold * 100)

            for sim_item in lowest_similarities:
                neighbor = sim_item['neighbor']
                param_name = sim_item['param_name']
                similarity = sim_item['similarity']

                # Restricted layer exchange: a tensor whose logical layer is not
                # permitted is left alone. `new_model` started as a deepcopy of
                # this client's own state, so skipping here is exactly "keep the
                # locally trained tensor" -- no path can carry a peer's value in.
                # Similarities and the percentile threshold are still computed
                # over *all* layers, matching how the reproduced repo masks its V
                # after construction rather than before.
                if allowed_layers is not None and key_to_layer.get(param_name) not in allowed_layers:
                    continue

                if similarity > threshold:
                    # Above the cutoff: average with all neighbors, sample-weighted.
                    weighted_param = current_model[param_name].clone() * sample_nums[idx]
                    total_samples = sample_nums[idx]

                    for n in neighbors:
                        weighted_param += client_models[n - 1][param_name].clone() * sample_nums[n - 1]
                        total_samples += sample_nums[n - 1]

                    new_model[param_name] = weighted_param / total_samples
                else:
                    if random.random() < args.exchange_prob:
                        new_model[param_name] = client_models[neighbor - 1][param_name].clone()
                    else:
                        possible_choices = [idx]
                        for n in neighbors:
                            if n != neighbor:
                                possible_choices.append(n - 1)

                        random_idx = random.choice(possible_choices)
                        if random_idx == idx:
                            new_model[param_name] = current_model[param_name].clone()
                        else:
                            new_model[param_name] = client_models[random_idx][param_name].clone()

        new_client_models.append(new_model)
    return new_client_models


def hybrid_aggregate(args, client_models, network_topology, topology_meta,
                     param_keys, client_thresholds, allowed_layers, key_to_layer):
    """Experiment family B, in this codebase's idiom.

    Stage 1 -- every client averages with its intra-cluster neighbours only; the
    bridge edge is excluded. Uses the plain unweighted mean this repo's own
    DFLAvg uses (note: NOT the sample-count weighting its DecFus P-group uses,
    and NOT the Metropolis-Hastings weighting the reproduced repo uses -- these
    are genuinely different averaging rules and the comparison should say so).

    Stage 2 -- only the two hubs, and only with each other, run DecFus's
    similarity-based fusion, on the *post-stage-1* states.

    Returns the new client model list.
    """
    hubs = [h - 1 for h in topology_meta['hubs']]  # 0-indexed
    hub_a, hub_b = hubs[0], hubs[1]

    # ---- stage 1: intra-cluster plain averaging ----
    stage1 = []
    for idx in range(args.num_users):
        peers = [n - 1 for n in network_topology[idx + 1]]
        # Drop the bridge: hub-to-hub traffic is stage 2's business.
        peers = [p for p in peers if not (idx in hubs and p in hubs)]
        members = [idx] + peers
        averaged = copy.deepcopy(client_models[idx])
        for key in averaged.keys():
            if averaged[key].dtype in (torch.int32, torch.int64, torch.bool):
                continue
            acc = torch.zeros_like(averaged[key], dtype=torch.float32)
            for m in members:
                acc += client_models[m][key].float()
            averaged[key] = (acc / len(members)).to(averaged[key].dtype)
        stage1.append(averaged)

    # ---- stage 2: DecFus across the bridge, hubs only ----
    new_models = [copy.deepcopy(m) for m in stage1]
    for i, peer in ((hub_a, hub_b), (hub_b, hub_a)):
        sims = []
        for param_name in param_keys:
            if 'num_batches_tracked' in param_name:
                continue
            sims.append({
                'param_name': param_name,
                'similarity': compute_parameter_similarity(
                    stage1[i][param_name], stage1[peer][param_name]),
            })
        if not sims:
            continue
        threshold = np.percentile([s['similarity'] for s in sims],
                                  client_thresholds[i] * 100)
        for item in sims:
            param_name = item['param_name']
            if allowed_layers is not None and key_to_layer.get(param_name) not in allowed_layers:
                continue  # blocked layer: hub keeps its stage-1 result
            if item['similarity'] > threshold:
                # Two-node average across the bridge.
                a = stage1[i][param_name].float()
                b = stage1[peer][param_name].float()
                new_models[i][param_name] = ((a + b) / 2.0).to(stage1[i][param_name].dtype)
            else:
                if random.random() < args.exchange_prob:
                    new_models[i][param_name] = stage1[peer][param_name].clone()
                # Otherwise the only remaining choice is itself (there are no
                # other stage-2 neighbours), i.e. keep the stage-1 value.
    return new_models


def compute_parameter_similarity(param1, param2):
    """Cosine similarity between two parameter tensors."""
    if torch.is_tensor(param1) and torch.is_tensor(param2):
        param1_flat = param1.view(-1)
        param2_flat = param2.view(-1)
        similarity = F.cosine_similarity(
            param1_flat.unsqueeze(0),
            param2_flat.unsqueeze(0)
        )
        return similarity.item()
    return 0.0


def DecFus(args, net_glob, dataset_train, dataset_test, dict_users):
    """DecFus: Decentralized Layer-wise Fusion with dynamic exploration and exploitation."""
    net_glob.train()

    # Initialize records
    acc = []
    loss = []
    train_loss = []

    args.current_threshold = args.initial_threshold

    # One independent variance stability detector per client
    # Note: detectors are only updated every test_round rounds, so protection_rounds is scaled accordingly
    variance_detectors = []
    for i in range(args.num_users):
        variance_detectors.append(VarianceStabilityDetector(
            window_size=args.variance_window_size,
            stability_threshold=args.variance_stability_threshold,  # stability threshold argument
            protection_rounds=int(args.epochs * args.variance_protection_ratio / args.test_round),  # converted to test rounds
            ref_start_round=int(args.epochs * args.variance_reference_start_ratio / args.test_round),
            ref_end_round=int(args.epochs * args.variance_reference_end_ratio / args.test_round)
        ))

    # Network topology
    FIXED_TOPOLOGY = {
        1: [2, 8, 9, 5],
        2: [1, 6, 10, 3],
        3: [2, 8, 10, 4],
        4: [3, 10, 5, 7],
        5: [4, 10, 1, 6],
        6: [5, 2, 9, 7],
        7: [8, 9, 4, 6],
        8: [1, 3, 9, 7],
        9: [8, 7, 1, 6],
        10: [2, 3, 4, 5]
    }
    if args.topology == 'two_cluster':
        # Two equal clusters, genuine intra-cluster circulant wiring, one
        # guaranteed degree-1 leaf per cluster, single hub-to-hub bridge.
        network_topology, topology_meta = build_two_cluster_topology(args.num_users)
    elif args.topology == 'bridge':
        # Two-cluster "bridge" topology, generalized to any even args.num_users
        # (originally added, and still hardcoded at N=10 in build_bridge_topology's
        # docstring example, for the N=10 comparison against our own reproduction):
        # clients 1,2 are hubs (degree leaves_per_hub+1 -- leaves_per_hub
        # same-cluster leaves each + one hub-hub bridge edge), the rest are leaves
        # (degree 1), split evenly between the two hubs. Every path between the two
        # clusters must cross the single 1-2 edge. Matches our own reproduction's
        # decfus/topology/registry.py:bridge_topology(n) node-for-node (hub_a=1,
        # hub_b=2, leaves_a=3..2+leaves_per_hub, leaves_b=3+leaves_per_hub..n).
        network_topology = build_bridge_topology(args.num_users)
        topology_meta = None
    else:
        network_topology = FIXED_TOPOLOGY
        topology_meta = None

    # Initialize client models and sample counts
    client_models = []
    sample_nums = []
    for i in range(args.num_users):
        client_models.append(copy.deepcopy(net_glob.state_dict()))
        sample_nums.append(len(dict_users[i]))

    param_keys = list(client_models[0].keys())
    current_lr = args.lr

    # Restricted layer exchange (experiment family C). 'full' leaves the
    # original behaviour untouched: allowed_layers is None and the guard in the
    # update loop is never taken.
    layer_policy = getattr(args, 'layer_policy', 'full')
    layer_scope = getattr(args, 'layer_scope', 'all')
    if layer_policy in (None, '', 'full'):
        allowed_layers, key_to_layer = None, {}
    else:
        if layer_scope != 'all':
            # The averaging branch below mixes over *all* neighbours at once, so
            # restricting a single edge would mean rebuilding that average
            # without the bridge peer rather than skipping the tensor. Not
            # implemented here; refusing beats silently doing the wrong thing.
            raise NotImplementedError(
                "layer_scope='%s' is not implemented in this codebase; only 'all' is "
                "supported (the reproduced repo supports 'bridge')." % layer_scope
            )
        allowed_layers, key_to_layer = resolve_layer_selection(net_glob, param_keys, layer_policy)
        print('[DecFus] layer policy %r -> %d/%d layers exchangeable: %s' % (
            layer_policy, len(allowed_layers),
            len(build_layer_groups(param_keys)[0]), sorted(allowed_layers)))

    hub_indices = set()
    if topology_meta:
        hub_indices = {h - 1 for h in topology_meta['hubs']}  # 0-indexed

    # Per-client cutoff values
    client_thresholds = [args.initial_threshold] * args.num_users

    # Main training loop
    for iter in range(args.epochs):
        print('*' * 80)
        print('Round {:3d}'.format(iter))

        # Local training
        for idx in range(args.num_users):
            net_local = copy.deepcopy(net_glob)
            net_local.load_state_dict(client_models[idx])
            local = DecFusClient(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(net=net_local, current_lr=current_lr)
            client_models[idx] = copy.deepcopy(w)

        # Evaluate before aggregation (performance right after local training)
        if iter % args.test_round == 0:
            avg_acc, avg_loss, avg_train_loss = 0, 0, 0

            for idx in range(args.num_users):
                net_local = copy.deepcopy(net_glob)
                net_local.load_state_dict(client_models[idx])
                item_acc, item_loss = test_with_loss(net_local, dataset_test, args)
                _, train_item_loss = test_with_loss(net_local, dataset_train, args)

                avg_acc += item_acc
                avg_loss += item_loss
                avg_train_loss += train_item_loss

            avg_acc /= args.num_users
            avg_loss /= args.num_users
            avg_train_loss /= args.num_users

            acc.append(avg_acc)
            loss.append(avg_loss)
            train_loss.append(avg_train_loss)

        # Aggregation. Family B replaces the standard pass entirely; everything
        # after this point (variance detection, cutoff update, logging) runs
        # identically either way -- which is why this is an if/else rather than
        # an early `continue`, as skipping ahead would freeze the DCD cutoff at
        # its initial value for the whole run.
        if getattr(args, 'hybrid', False):
            if not topology_meta:
                raise ValueError('--hybrid requires --topology two_cluster')
            client_models = hybrid_aggregate(
                args, client_models, network_topology, topology_meta,
                param_keys, client_thresholds, allowed_layers, key_to_layer)
        else:
            client_models = standard_aggregate(
                args, client_models, network_topology, param_keys,
                client_thresholds, sample_nums, allowed_layers, key_to_layer)

        # After aggregation, compute the variance and update the cutoff
        if iter % args.test_round == 0:
            # Training loss after aggregation (used for variance detection)
            train_losses_after_exchange = []
            for idx in range(args.num_users):
                net_local = copy.deepcopy(net_glob)
                net_local.load_state_dict(client_models[idx])
                local_dataset = DatasetSplit(dataset_train, dict_users[idx])
                _, train_item_loss = test_with_loss(net_local, local_dataset, args)
                train_losses_after_exchange.append(
                    train_item_loss.item() if torch.is_tensor(train_item_loss) else train_item_loss
                )

            # Each client computes the variance over its neighborhood, then updates its detector and cutoff
            neighbor_train_variances = []
            for idx in range(args.num_users):
                # Neighborhood (including the client itself)
                neighbors = network_topology[idx + 1]
                neighbor_indices = [idx] + [n - 1 for n in neighbors]

                # Neighborhood variance of the post-aggregation training loss
                neighbor_train_losses = [train_losses_after_exchange[i] for i in neighbor_indices]
                neighbor_train_var = np.var(neighbor_train_losses)
                neighbor_train_variances.append(neighbor_train_var)

                # Update the variance detector (the round passed in is iter // args.test_round)
                # update() sets the detector's internal is_stable flag
                variance_detectors[idx].update(neighbor_train_var, iter // args.test_round)

                # Update this client's cutoff
                # adaptive_sigmoid_threshold_decay reads detector.is_stable
                individual_threshold = adaptive_sigmoid_threshold_decay(
                    current_round=iter,
                    args=args,
                    stability_detector=variance_detectors[idx]
                )
                client_thresholds[idx] = individual_threshold

            avg_neighbor_train_variance = np.mean(neighbor_train_variances)

            # Mean cutoff and number of stable clients
            avg_threshold = np.mean(client_thresholds)
            num_stable_clients = sum(1 for detector in variance_detectors if detector.is_stable)

            # Collect all metrics into a single swanlab.log call
            log_dict = {
                # Test accuracy and losses
                "round": iter,
                "average_test_accuracy": avg_acc,
                "average_test_loss": avg_loss,
                "average_train_loss": avg_train_loss,

                # Variance-related metrics
                "avg_neighbor_train_loss_variance": avg_neighbor_train_variance,
                "avg_threshold": avg_threshold,
                "num_stable_clients": num_stable_clients,
            }

            # Per-client metrics
            for client_id in range(args.num_users):
                log_dict[f"client_{client_id}_neighbor_train_variance"] = neighbor_train_variances[client_id]
                log_dict[f"client_{client_id}_threshold"] = client_thresholds[client_id]
                log_dict[f"client_{client_id}_relative_stability"] = variance_detectors[
                    client_id].get_relative_stability()  # relative stability
                log_dict[f"client_{client_id}_is_stable"] = variance_detectors[client_id].is_stable
                if variance_detectors[client_id].stable_round is not None:
                    log_dict[f"client_{client_id}_stable_round"] = variance_detectors[
                                                                       client_id].stable_round * args.test_round  # convert back to actual rounds

            swanlab.log(log_dict)

    # Save results
    save_result(acc, 'test_acc', args)
    save_result(loss, 'test_loss', args)
    save_result(train_loss, 'test_train_loss', args)


def test_with_loss(net_glob, dataset_test, args):
    """Wrapper around the test function."""
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Testing accuracy: {:.2f}".format(acc_test))
    return acc_test, loss_test
