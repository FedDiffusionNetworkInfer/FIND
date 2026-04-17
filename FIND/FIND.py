import numpy as np
from sklearn.cluster import KMeans
import math
from orig_utils import *
import time
from collections import defaultdict
import copy
from itertools import combinations




def load_data(graph_path, result_path):
    with open(result_path, 'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        diffusion_result = np.array([[int(state) for state in line] for line in lines])

    nodes_num = diffusion_result.shape[1]

    with open(graph_path, 'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        data = np.array([[int(node) for node in line] for line in lines])
        ground_truth_network = np.zeros((nodes_num, nodes_num))
        edges_num = data.shape[0]
        for i in range(edges_num):
            ground_truth_network[data[i, 0] - 1, data[i, 1] - 1] = 1

    return ground_truth_network, diffusion_result


def krr_3_mechanism(Phi, epsilon, values=[]):
    """
    Fast vectorized 3-ary Randomized Response for matrix Phi.
    Phi: input matrix with values in {v1, v2, v3}
    values: sorted list [v1, v2, v3]
    epsilon: DP parameter
    """

    values = np.array(values)
    values[1] = np.sqrt(1 / Phi.shape[1])
    values[2] = np.sqrt(2 / Phi.shape[1])

    Phi_idx = np.argmin(np.abs(Phi[..., None] - values[None, None, :]), axis=2)

    k = 3
    p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
    q = (1 - p) / (k - 1)


    n, m = Phi_idx.shape


    rnd = np.random.rand(n, m)
    keep = rnd < p


    rand_vals = np.random.randint(0, 3, size=(n, m))


    equal_mask = (rand_vals == Phi_idx)

    rand_vals[equal_mask] = (rand_vals[equal_mask] + 1) % 3


    out_idx = np.where(keep, Phi_idx, rand_vals)


    Phi_noised = values[out_idx]

    return Phi_noised


def hamming_kernel(X, Y=None, sigma=1.0):
    """
    Compute the Hamming kernel between two datasets X and Y.

    Parameters:
    X : ndarray of shape (n_samples, n_features)
        Input binary data.
    Y : ndarray of shape (n_samples, n_features), optional
        Second input binary data. If None, use X itself.
    sigma : float, optional, default=1.0
        Parameter that controls the width of the kernel.

    Returns:
    K : ndarray of shape (n_samples, n_samples)
        Kernel matrix.
    """
    if Y is None:
        Y = X
    if len(X.shape) == 1:
        X = np.reshape(X, (X.shape[0], 1))
    if len(Y.shape) == 1:
        Y = np.reshape(Y, (Y.shape[0], 1))

    # Compute pairwise Hamming distance
    hamming_dist = np.abs(X[:, np.newaxis] - Y).sum(axis=2)  # Hamming distance
    K = np.exp(-hamming_dist / sigma)  # Hamming kernel
    return K






def hsic(X, Y,  sigma=1.0):
    """
    Compute the Hilbert-Schmidt Independence Criterion (HSIC) between two datasets X and Y.

    Parameters:
    X : ndarray of shape (n_samples, n_features)
        First dataset.
    Y : ndarray of shape (n_samples, n_features)
        Second dataset.
    sigma : float, optional, default=1.0
        Parameter for the Gaussian kernel.

    Returns:
    hsic_value : float
        The HSIC value.
    """
    n = X.shape[0]

    K_X = hamming_kernel(X, sigma=sigma)
    K_Y = hamming_kernel(Y, sigma=sigma)

    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    # Compute the HSIC value
    hsic_value = np.trace(K_X @ H @ K_Y @ H) / ((n-1) ** 2)
    return hsic_value


def generate_hamming_drf(d, gamma, D, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    q = 1 - np.exp(-gamma)
    features = []

    for _ in range(D):

        mask = rng.random(d) < q
        S = np.where(mask)[0]

        if len(S) == 0:

            z_S = {}
        else:
            z_S = None

        features.append((S, z_S))

    return features


class ListKeyDict:


    def __init__(self):
        self._data = {}  # 内部使用 tuple -> float 的 dict

    def __setitem__(self, key: list, value: float):
        if not isinstance(key, list):
            raise TypeError("Key must be a list")
        if not isinstance(value, (int, float)):
            raise TypeError("Value must be a float (or int)")
        self._data[tuple(key)] = float(value)

    def __getitem__(self, key: list) -> float:
        if not isinstance(key, list):
            raise TypeError("Key must be a list")
        return self._data[tuple(key)]

    def __delitem__(self, key: list):
        if not isinstance(key, list):
            raise TypeError("Key must be a list")
        del self._data[tuple(key)]

    def __contains__(self, key: list) -> bool:
        if not isinstance(key, list):
            return False
        return tuple(key) in self._data

    def get(self, key: list, default=None) -> float:
        if not isinstance(key, list):
            return default
        return self._data.get(tuple(key), default)

    def keys(self):

        return [list(k) for k in self._data.keys()]

    def values(self):

        return list(self._data.values())

    def items(self):

        return [(list(k), v) for k, v in self._data.items()]

    def clear(self):

        self._data.clear()

    def pop(self, key: list, default=None):

        if not isinstance(key, list):
            raise TypeError("Key must be a list")
        return self._data.pop(tuple(key), default)

    def update(self, other):

        if isinstance(other, ListKeyDict):
            self._data.update(other._data)
        elif isinstance(other, dict):

            self._data.update(other)
        else:

            for k, v in other:
                self[k] = v

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        items = [(list(k), v) for k, v in self._data.items()]
        return f"ListKeyDict({items})"

    def __str__(self):
        return self.__repr__()

class HammingRandomFeatures:
    def __init__(self, gamma=1.0, D=50, random_state=None):
        self.gamma = gamma
        self.D = D
        self.rng = np.random.default_rng(random_state)
        self.features_ = None
        self.value_pools_ = None
        self.d_ = None

    def fit_unbias(self, X):

        X = np.array(X)
        n, d = X.shape
        self.d_ = d
        q = 1 - np.exp(-self.gamma)
        pi = 0.5


        self.value_pools_ = []
        for j in range(d):
            unique_vals = list(set(X[:, j]))
            self.value_pools_.append(unique_vals)


        raw_features = generate_hamming_drf(d, self.gamma, self.D, self.rng)


        self.features_ = []

        base_weight = 1/np.sqrt(self.D)
        for S, _ in raw_features:
            if len(S) == 0:
                z_S = {}
                weight = 1.0
            else:

                z_S = {}
                for i in S:
                    val_pool = self.value_pools_[i]
                    z_S[i] = self.rng.choice(val_pool)

                weight = 1 / np.sqrt(pi)

            self.features_.append((S, z_S, weight))

        self.base_weight_ = base_weight
        return self

    def transform_unbias(self, X):

        X = np.array(X)
        n, d = X.shape
        assert d == self.d_, "Input dimension mismatch"

        Phi = np.zeros((n, self.D))

        sqrt_base_over_D = self.base_weight_

        for j, (S, z_S, weight) in enumerate(self.features_):
            if len(S) == 0:

                Phi[:, j] = sqrt_base_over_D * weight
            else:

                matches = np.ones(n, dtype=bool)
                for i in S:
                    matches &= (X[:, i] == z_S[i])
                Phi[matches, j] = sqrt_base_over_D * weight


        return Phi


def hsic_approx(X_feat, Y_feat):

    n = X_feat.shape[0]

    mu_x = X_feat.mean(axis=0, keepdims=True)

    mu_y = Y_feat.mean(axis=0, keepdims=True)

    Xc = X_feat - mu_x
    Yc = Y_feat - mu_y


    cross = Xc.T @ Yc  # (Dx, Dy)
    hsic = np.sum(cross ** 2) / ((n-1) ** 2)

    return hsic


def approx_hamming_hsic(X, Y,noise_flag = 0,D=50,gamma=1.0):
    if X.size == 0 or Y.size == 0:
        return 0
    if len(X.shape) == 1:
        X = np.reshape(X, (X.shape[0], 1))
    if len(Y.shape) == 1:
        Y = np.reshape(Y, (Y.shape[0], 1))

    hrf_X = HammingRandomFeatures(gamma=gamma, D=D, random_state=42)
    hrf_X.fit_unbias(X)
    Phi_X = hrf_X.transform_unbias(X)


    hrf_Y = HammingRandomFeatures(gamma=gamma, D=D, random_state=42)
    hrf_Y.fit_unbias(Y)
    Phi_Y = hrf_Y.transform_unbias(Y)

    if noise_flag == 1:
        Phi_X = krr_3_mechanism(Phi_X, epsilon_DP)
        Phi_Y = krr_3_mechanism(Phi_Y, epsilon_DP)

    hsic_val = hsic_approx(Phi_X, Phi_Y)
    return hsic_val

def hsic_prune(hsic_self_vector,record_states,D = 20, noise_flag = 0, prune_choice="hsic_score"):
    results_num, nodes_num = record_states.shape
    MI = np.zeros((nodes_num, nodes_num))

    if prune_choice == "hamming":
        for i in range(nodes_num):
            for j in range(i):
                MI[i,j] = hsic(record_states[:,i], record_states[:,j],kernel="hamming")
                MI[j,i] = MI[i,j]
    elif prune_choice == "hsic_score":
        for i in range(nodes_num):
            hsic_self_vector[i] = approx_hamming_hsic(record_states[:, i], record_states[:, i], noise_flag=noise_flag,
                                                      D=D)
            for j in range(i):
                MI[i, j] = approx_hamming_hsic(record_states[:, i], record_states[:, j],noise_flag=noise_flag,D=D)/hsic_self_vector[i]
                MI[j, i] = MI[i, j]
    else:
        raise ValueError("Unsupported kernel type.")

    tmp_MI = MI.reshape((-1, 1))

    estimator = KMeans(n_clusters=2)
    estimator.fit(tmp_MI)
    label_pred = estimator.labels_
    temp_0 = tmp_MI[label_pred == 0]
    temp_1 = tmp_MI[label_pred == 1]

    if np.max(temp_0) > np.max(temp_1):
        tau = np.max(temp_1)
    else:
        tau = np.max(temp_0)
    prune_network = np.zeros((nodes_num, nodes_num))
    prune_network[np.where(MI>tau)] = 1


    for i in range(nodes_num):
        prune_network[i,i]=0

    return prune_network,MI



def hsic_score(X,Y,D=20,noise_flag=0):

    hsic_XY = approx_hamming_hsic(X,Y,noise_flag,D=D)
    hsic_base = approx_hamming_hsic(X,X,noise_flag,D=D)
    return hsic_XY/hsic_base


def numpy2dec(line):
    j = 0
    for m in range(line.size):
        j = j + pow(2, line.size - 1 - m) * line[m]

    return int(j)


def client_infer_network(client_results, D = 50,prune_choice="hsic_score"):
    beta,nodes_num = client_results.shape
    client_network = np.zeros((nodes_num, nodes_num))
    client_weight = np.zeros(nodes_num)
    hsic_self_vector = np.ones(nodes_num)

    pruned_network,hsic_matrix = hsic_prune(hsic_self_vector,client_results,D=D,noise_flag=1, prune_choice=prune_choice)


    for i in range(nodes_num):

        N1 = np.sum(client_results[:, i] == 0)
        N2 = np.sum(client_results[:, i] == 1)
        bound = math.log(2 * N1 * math.log(beta / N1, 2) + 2 * N2 * math.log(beta / N2, 2) + math.log(beta + 1, 2), 2)

        candidate_parents = np.where(pruned_network[:, i] == 1)[0]
        candidate_size = candidate_parents.size

        if candidate_size <= parents_num_limit:
            client_network[candidate_parents, i] = 1
            client_weight[i] = approx_hamming_hsic(client_results[:,i],client_results[:,candidate_parents],1,D)/hsic_self_vector[i]
        else:
            i_mutual_info_index_mi = np.concatenate(
                [np.array([j for j in range(nodes_num)]).reshape([-1, 1]), pruned_network[:, i:i + 1]], axis=1)
            i_mutual_info_index_mi[i, 1] = -1

            i_mutual_info_index_mi = i_mutual_info_index_mi[(i_mutual_info_index_mi[:, 1] * -1).argsort()]

            i_big_mutual_info_index = i_mutual_info_index_mi[:int(parents_num_limit)].astype(int)
            candidate_parents = i_big_mutual_info_index[:, 0]

            par_comb_sets = []
            par_comb_sets.append((np.array([]), 0))
            for k in range(1, int(bound + 1)):

                k_combs = list(combinations(candidate_parents, k))
                for comb in k_combs:
                    score = approx_hamming_hsic(client_results[:, i], client_results[:, comb], 1,D)/hsic_self_vector[i]
                    par_comb_sets.append((np.array(comb), score))

            sorted_sets = sorted(par_comb_sets, key=lambda comb: comb[1], reverse=True)

            for comb in sorted_sets:
                if np.sum(client_network[:, i] >= bound):
                    break
                temp_rel = client_network[:, i]
                temp_rel[comb[0].astype(int)] = 1
                if np.sum(temp_rel) > bound:
                    continue
                client_network[:, i] = temp_rel
                client_weight[i] = comb[1]

    return client_network,hsic_matrix,client_weight







def client_calculate_kernel(client_results, parent_comb_list):
    node_num = client_results.shape[1]
    for i in range(node_num):
        parent_comb_keys = parent_comb_list[i].keys()
        for comb in parent_comb_keys:
            parent_comb_list[i][comb] = hsic_score(client_results[:,i], client_results[:,comb],D=50)




def server_aggregation_with_node_weight(client_network_list, client_weight_list, all_parent_comb_list, client_bata_list):

    client_glob_hsic_list = []
    total_beta = sum(client_bata_list)
    for c in range(len(client_network_list)):
        client_glob_hsic = np.zeros_like(client_weight_list[c])
        for i in range(client_network_list[c].shape[0]):
            parent_comb = np.where(client_network_list[c][i,:]==1)[0]
            for cin in range(len(client_network_list)):
                client_glob_hsic[i] += (client_bata_list[cin]-1)**2 * all_parent_comb_list[cin][i][parent_comb]
            client_glob_hsic[i] = client_glob_hsic[i]/(total_beta-1)**2
        client_glob_hsic_list.append(client_glob_hsic)


    final_network = np.zeros_like(client_network_list[0])
    agg_weight_vector = np.zeros_like(client_weight_list[0])
    for c in range(len(client_network_list)):
        final_network += client_glob_hsic_list[c] * client_weight_list[c] * client_network_list[c]
        agg_weight_vector += client_glob_hsic_list[c] * client_weight_list[c]
    final_network = final_network/(agg_weight_vector + epsilon)

    tmp_MI = final_network.reshape((-1, 1))

    estimator = KMeans(n_clusters=2)
    estimator.fit(tmp_MI)
    label_pred = estimator.labels_
    temp_0 = tmp_MI[label_pred == 0]
    temp_1 = tmp_MI[label_pred == 1]

    if np.max(temp_0) > np.max(temp_1):
        tau = np.max(temp_1)
    else:
        tau = np.max(temp_0)

    final_rounding_network = np.zeros_like(client_network_list[0])
    final_rounding_network[np.where(final_network > tau)] = 1

    for i in range(nodes_num):
        final_rounding_network[i, i] = 0
    return final_rounding_network








def FINA(client_data_list):
    client_num = len(client_data_list)
    client_network_list = []
    client_node_weight_list = []
    client_edge_weight_list = []
    client_beta_list = []
    all_parent_comb_list = []
    parent_comb_list = [ListKeyDict() for i in range(client_data_list[0].shape[1])]

    for c in range(client_num):
        client_network,client_edge_weight,client_node_weight = client_infer_network(client_data_list[c],50,"hsic_score")
        client_network_list.append(client_network)
        client_node_weight_list.append(client_node_weight)
        client_edge_weight_list.append(client_edge_weight)
        client_beta_list.append(client_data_list[c].shape[0])
        for i in range(client_network.shape[0]):
            parent_comb_list[i].__setitem__(np.where(client_network[i, :] == 1)[0].tolist(), -1)
            parent_comb_list[i].__setitem__([i], -1)

    for c in range(client_num):
        all_parent_comb_list.append(copy.deepcopy(parent_comb_list))
        client_calculate_kernel(client_data_list[c],all_parent_comb_list[c])
    final_rounding_network = server_aggregation_with_node_weight(client_network_list, client_node_weight_list, all_parent_comb_list,
                                        client_bata_list):
    return final_rounding_network



def threshold_with_kmeans(weight_matrix):
    nodes_num = weight_matrix.shape[0]

    tmp_MI = weight_matrix.reshape((-1, 1))

    estimator = KMeans(n_clusters=2)
    estimator.fit(tmp_MI)
    label_pred = estimator.labels_
    temp_0 = tmp_MI[label_pred == 0]
    temp_1 = tmp_MI[label_pred == 1]

    if np.max(temp_0) > np.max(temp_1):
        tau = np.max(temp_1)
    else:
        tau = np.max(temp_0)

    return tau




epsilon_DP = 2.0
epsilon=1e-5

if __name__ == '__main__':
    graph_path = './network.txt'
    result_path = './observation_data.txt'
    sample_index = [80, 200, 360, 560]
    beta = 800
    ground_truth_network, diffusion_result = load_data(graph_path, result_path)
    result_list = np.split(diffusion_result, sample_index, axis=0)
    diffusion_result = diffusion_result[0:beta, :]

    final_network = FINA(result_list)








