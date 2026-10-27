

import numpy as np



epsilon_DP = 2.0
epsilon = 1e-12


def load_data(graph_path, result_path):
    network, observations = _load_data(graph_path, result_path)
    return network.T.copy(), observations


def randomized_response(observations, h, rng=None):
   
    rng = np.random.default_rng(rng)
    observations = np.asarray(observations)
    return np.logical_xor(observations, rng.random(observations.shape) < h).astype(int)


def hamming_kernel(X, Y=None, sigma=1.0):
    
    X = np.asarray(X)
    Y = X if Y is None else np.asarray(Y)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    return np.exp(-np.count_nonzero(X[:, None, :] != Y[None, :, :], axis=2) / sigma)


def hsic(X, Y, sigma=1.0):
    
    n = len(X)
    K_X = hamming_kernel(X, sigma=sigma)
    K_Y = hamming_kernel(Y, sigma=sigma)
    H = np.eye(n) - np.ones((n, n)) / n
    return np.trace(K_X @ H @ K_Y @ H) / (n - 1) ** 2


def generate_hamming_drf(d, gamma, D, rng=None):
    
    rng = np.random.default_rng(rng)
    masks = rng.random((D, d)) < -np.expm1(-gamma)
    templates = rng.integers(0, 2, size=(D, d))
    return [(np.flatnonzero(mask), template[mask])
            for mask, template in zip(masks, templates)]


class HammingRandomFeatures:
    def __init__(self, gamma=1.0, D=50, random_state=None, h=0.0):
        self.gamma = gamma
        self.D = D
        self.h = h
        self.rng = np.random.default_rng(random_state)

    def fit(self, d):
        self.d_ = d
        self.features_ = generate_hamming_drf(d, self.gamma, self.D, self.rng)
        sizes = np.array([len(S) for S, _ in self.features_])
        self.a_ = np.power(2.0, sizes / 2) / np.sqrt(self.D)
        return self

    def fit_unbias(self, X):
        
        X = np.asarray(X)
        return self.fit(1 if X.ndim == 1 else X.shape[1])

    def transform_unbias(self, X):
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        Phi = np.full((len(X), self.D), 1 / np.sqrt(self.D))
        for l, (S, template) in enumerate(self.features_):
            if len(S):
                factors = ((X[:, S] == template).astype(float) - self.h)
                factors /= (1 - 2 * self.h) * np.sqrt(0.5)
                Phi[:, l] *= np.prod(factors, axis=1)
        return Phi

    def transform(self, X):
        
        Phi = self.transform_unbias(X)
        delta = Phi @ self.a_ - np.sum(Phi * Phi, axis=1)
        if self.h == 0 or self.d_ == 0:
            delta = np.zeros(len(Phi))
        return Phi, delta

    @property
    def zeta(self):
        
        return 2 * sum(len(S) == 1 for S, _ in self.features_) / self.D


def feature_map(nodes, D=50, gamma=1.0, h=0.0, public_seed=42):
    
    nodes = tuple(sorted(nodes))
    seed = np.random.SeedSequence([public_seed, len(nodes), *nodes])
    return HammingRandomFeatures(gamma, D, seed, h).fit(len(nodes))


def hsic_approx(X_feat, Y_feat, delta_X=None, delta_Y=None):
    
    n = len(X_feat)
    delta_X = np.zeros(n) if delta_X is None else delta_X
    delta_Y = np.zeros(n) if delta_Y is None else delta_Y
    Xc = X_feat - X_feat.mean(axis=0, keepdims=True)
    Yc = Y_feat - Y_feat.mean(axis=0, keepdims=True)
    cross = Xc.T @ Yc
    numerator = np.sum(cross * cross)
    numerator += delta_X @ np.sum(Yc * Yc, axis=1)
    numerator += delta_Y @ np.sum(Xc * Xc, axis=1)
    numerator += (1 - 2 / n) * (delta_X @ delta_Y)
    numerator += delta_X.sum() * delta_Y.sum() / n ** 2
    return float(numerator / (n - 1) ** 2)


def binary_self_hsic(X, h=0.0, gamma=1.0):
    
    X = np.asarray(X)
    n = len(X)
    n1 = np.clip((X.sum(axis=0) - n * h) / (1 - 2 * h), 0, n)
    return 4 * (-np.expm1(-gamma)) ** 2 * (n1 * (n - n1) / n) ** 2 / (n - 1) ** 2


def threshold_with_kmeans(weight_matrix):
    
    values = np.asarray(weight_matrix).ravel()
    if not np.any(values > 0):
        return 0.0
    labels = kmeans_zero(values)
    near_zero = values[labels == 0]
    return float(near_zero.max()) if near_zero.size else 0.0


def hsic_prune(hsic_self_vector, record_states, D=50, h=0.0,
               gamma=1.0, public_seed=42, node_features=None):
    
    nodes_num = record_states.shape[1]
    if node_features is None:
        node_features = [feature_map((i,), D, gamma, h, public_seed).transform(
            record_states[:, i]) for i in range(nodes_num)]
    hsic_self_vector[:] = binary_self_hsic(record_states, h, gamma)
    denominators = np.maximum(hsic_self_vector, epsilon)
    scores = np.zeros((nodes_num, nodes_num))
    for i in range(nodes_num):
        Xi, delta_i = node_features[i]
        for j in range(i):
            Xj, delta_j = node_features[j]
            dependence = hsic_approx(Xi, Xj, delta_i, delta_j)
            scores[i, j] = dependence / denominators[i]
            scores[j, i] = dependence / denominators[j]
    tau = threshold_with_kmeans(scores)
    pruned = (scores > tau).astype(int)
    np.fill_diagonal(pruned, 0)
    return pruned, scores


def greedy_parent_search(candidates, score):
    
    selected = ()
    remaining = sorted(candidates)
    best_score = 0.0
    while remaining:
        trials = [(score(tuple(sorted((*selected, j)))), j) for j in remaining]
        next_score, next_parent = max(trials, key=lambda item: item[0])
        if next_score <= best_score:
            break
        selected = tuple(sorted((*selected, next_parent)))
        remaining.remove(next_parent)
        best_score = next_score
    return selected, best_score


def client_infer_network(client_results, D=50, h=0.0, gamma=1.0, public_seed=42):

    nodes_num = client_results.shape[1]
    node_features = [feature_map((i,), D, gamma, h, public_seed).transform(
        client_results[:, i]) for i in range(nodes_num)]
    self_hsic = np.zeros(nodes_num)
    pruned, scores = hsic_prune(self_hsic, client_results, D, h, gamma,
                              public_seed, node_features)
    network = np.zeros((nodes_num, nodes_num), dtype=int)
    confidence = np.zeros(nodes_num)
    for i in range(nodes_num):
        candidates = np.flatnonzero(pruned[i]).tolist()
        cache = {(j,): scores[i, j] for j in candidates}
        Xi, delta_i = node_features[i]

        def score(parents):
            if parents not in cache:
                parent_map = feature_map(parents, D, gamma, h, public_seed)
                Phi, delta = parent_map.transform(client_results[:, list(parents)])
                cache[parents] = hsic_approx(Xi, Phi, delta_i, delta) / max(self_hsic[i], epsilon)
            return cache[parents]

        parents, confidence[i] = greedy_parent_search(candidates, score)
        network[i, list(parents)] = 1
    return network, scores, confidence


def client_status_sum(perturbed_data, h):
    
    return ((perturbed_data - h) / (1 - 2 * h)).sum(axis=0)


def client_calculate_kernel(perturbed_data, child, parents, global_mean,
                            total_beta, parent_map, h):
    
    parents = tuple(sorted(parents))
    Phi, delta = parent_map.transform(perturbed_data[:, list(parents)])
    e = (perturbed_data[:, child] - h) / (1 - 2 * h) - global_mean
    nu = h * (1 - h) / (1 - 2 * h) ** 2
    u = Phi.sum(axis=0)
    g = Phi.T @ e
    r = delta @ (e * e - nu * (1 - 1 / total_beta)) - nu * np.sum(Phi * Phi)
    return u, g, float(r)


def global_confidence(statistics, total_beta, zeta, h):
   
    u = sum(item[0] for item in statistics)
    g = sum(item[1] for item in statistics)
    r = sum(item[2] for item in statistics)
    nu = h * (1 - h) / (1 - 2 * h) ** 2
    return float(zeta * (g @ g + nu / total_beta * (u @ u) + r)
                 / (total_beta - 1) ** 2)


def server_aggregation_with_node_weight(client_network_list, client_weight_list,
                                        client_global_weight_list, return_weights=False):
    
    weighted = np.zeros_like(client_network_list[0], dtype=float)
    denominator = np.zeros(weighted.shape[0])
    for network, local, glob in zip(client_network_list, client_weight_list,
                                    client_global_weight_list):
        weights = np.maximum(local, 0) * np.maximum(glob, 0)
        weighted += weights[:, None] * network
        denominator += weights
    np.divide(weighted, denominator[:, None], out=weighted,
              where=denominator[:, None] != 0)
    np.fill_diagonal(weighted, 0)
    sigma = threshold_with_kmeans(weighted)
    network = (weighted > sigma).astype(int)
    np.fill_diagonal(network, 0)
    return (network, weighted) if return_weights else network


def FIND(client_data_list, D=50, gamma=1.0, epsilon_dp=epsilon_DP,
         random_state=None, public_seed=42):
    
    h = np.exp(-epsilon_dp) / (1 + np.exp(-epsilon_dp))
    rng = np.random.default_rng(random_state)
    perturbed_clients = []
    client_networks = []
    local_confidences = []
    status_sums = []
    client_sizes = []

    # Phase 1: sequential clients, each with a single cached RR realization.
    for observations in client_data_list:
        perturbed = randomized_response(observations, h, rng)
        network, _, confidence = client_infer_network(perturbed, D, h, gamma, public_seed)
        perturbed_clients.append(perturbed)
        client_networks.append(network)
        local_confidences.append(confidence)
        status_sums.append(client_status_sum(perturbed, h))
        client_sizes.append(len(perturbed))

    # Phase 2: server broadcasts means and unique inferred parent sets.
    total_beta = sum(client_sizes)
    global_means = np.sum(status_sums, axis=0) / total_beta
    nodes_num = client_networks[0].shape[0]
    global_confidences = [np.zeros(nodes_num) for _ in client_networks]
    for i in range(nodes_num):
        parent_sets = [tuple(np.flatnonzero(network[i]).tolist()) for network in client_networks]
        estimates = {(): 0.0}
        child_map = feature_map((i,), D, gamma, h, public_seed)
        for parents in dict.fromkeys(parent_sets):
            if not parents:
                continue
            parent_map = feature_map(parents, D, gamma, h, public_seed)
            statistics = []
            for perturbed in perturbed_clients:
                statistics.append(client_calculate_kernel(
                    perturbed, i, parents, global_means[i], total_beta, parent_map, h))
            estimates[parents] = global_confidence(statistics, total_beta, child_map.zeta, h)
        for k, parents in enumerate(parent_sets):
            global_confidences[k][i] = estimates[parents]

    return server_aggregation_with_node_weight(client_networks, local_confidences,
                                              global_confidences)





if __name__ == '__main__':
    graph_path = './network.txt'
    result_path = './observation_data.txt'
    sample_index = [80, 200, 360, 560]
    beta = 800
    ground_truth_network, diffusion_result = load_data(graph_path, result_path)
    result_list = np.split(diffusion_result[:beta], sample_index, axis=0)
    final_network = FIND(result_list)
