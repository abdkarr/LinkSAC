import numpy as np
import networkx as nx

from numba import njit
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from scipy import sparse

from linksac import typing


def inject_random_edge_anomalies(
    graph: nx.Graph, anomaly_size: float, rng: typing.RNG_TYPE = None
) -> nx.Graph:

    rng = typing.check_rng(rng)

    # Init the output graph
    graph = graph.copy()
    nx.set_edge_attributes(graph, 0, "anomaly")

    n_edges = graph.number_of_edges()
    n_nodes = graph.number_of_nodes()
    n_anomaly_edges = int(np.floor(n_edges * anomaly_size))

    n_anomalies_added = 0
    while n_anomalies_added < n_anomaly_edges:
        i, j = rng.choice(n_nodes, size=2, replace=False)

        if graph.has_edge(i, j):
            continue

        graph.add_edge(i, j, anomaly=1)
        n_anomalies_added += 1

    return graph


def inject_embedding_edge_anomalies(
    graph: nx.Graph, anomaly_size: float, kth: int, rng: typing.RNG_TYPE = None
) -> nx.Graph:

    rng = typing.check_rng(rng)

    # Init the output graph
    graph = graph.copy()
    nx.set_edge_attributes(graph, 0, "anomaly")

    n_edges = graph.number_of_edges()
    n_nodes = graph.number_of_nodes()
    n_anomaly_edges = int(np.floor(n_edges * anomaly_size))

    # Get Laplacian embeddings
    laplacian = nx.normalized_laplacian_matrix(graph)
    _, embeddings = sparse.linalg.eigsh(laplacian, which="SM", k=16)


    n_anomalies_added = 0
    candidate_cache = {}
    while n_anomalies_added < n_anomaly_edges:
        i = rng.choice(n_nodes)

        if i in candidate_cache:
            anomaly_candidates = candidate_cache[i]
        else:
            dists = np.squeeze(cdist(embeddings[i][None, ...], embeddings))
            max_dist = max([dists[v] for v in graph.neighbors(i)])
            anomaly_candidates = np.where(dists > max_dist)[0]
            if len(anomaly_candidates) > kth:
                anomaly_dists = dists[anomaly_candidates]
                kth_dist = np.partition(anomaly_dists, kth=-kth)[-kth]
                anomaly_candidates = anomaly_candidates[
                    np.where(anomaly_dists >= kth_dist)[0]
                ]

                candidate_cache[i] = anomaly_candidates

        if len(anomaly_candidates) > 0:
            j = rng.choice(anomaly_candidates)
        else:
            continue

        if graph.has_edge(i, j):
            continue

        graph.add_edge(i, j, anomaly=1)
        n_anomalies_added += 1

    return graph


@njit
def _sum_of_powers(x, power):
    n = x.shape[0]
    sum_powers = np.zeros((power, n))
    for i, i_power in enumerate(range(1, power + 1)):
        sum_powers[i] = np.power(x, i_power)

    return sum_powers.sum(0)


@njit
def _estimate_loss_with_delta_eigenvals(
    candidates, vals_org, vecs_org, n_nodes, dim, window_size
):
    loss_est = np.zeros(len(candidates))
    for x in range(len(candidates)):
        i, j = candidates[x]
        vals_est = vals_org + (
            2 * vecs_org[i] * vecs_org[j]
            - vals_org * (vecs_org[i] ** 2 + vecs_org[j] ** 2)
        )

        vals_sum_powers = _sum_of_powers(vals_est, window_size)

        loss_ij = np.sqrt(np.sum(np.sort(vals_sum_powers**2)[: n_nodes - dim]))
        loss_est[x] = loss_ij

    return loss_est


def inject_attack_edge_anomalies(
    graph: nx.Graph,
    anomaly_size: float,
    dim: int = 32,
    window_size: int = 5,
    rng: typing.RNG_TYPE = None,
) -> nx.Graph:

    rng = typing.check_rng(rng)

    adj_matrix = nx.adjacency_matrix(graph)

    n_edges = graph.number_of_edges()
    n_nodes = graph.number_of_nodes()
    n_anomaly_edges = int(np.floor(n_edges * anomaly_size))
    n_candidates = n_anomaly_edges * 10

    # Randomly select unconnected node pairs to consider for anomaly injection
    candidates = rng.integers(0, n_nodes, [n_candidates * 5, 2])
    candidates = candidates[candidates[:, 0] < candidates[:, 1]]
    candidates = candidates[adj_matrix[candidates[:, 0], candidates[:, 1]] == 0]
    candidates = np.array(list(set(map(tuple, candidates))))
    candidates = candidates[:n_candidates]

    # Find perturbations
    deg_matrix = np.diag(adj_matrix.sum(1))
    vals_org, vecs_org = eigh(adj_matrix.toarray(), deg_matrix)

    loss_for_candidates = _estimate_loss_with_delta_eigenvals(
        candidates, vals_org, vecs_org, n_nodes, dim, window_size
    )
    top_flips = candidates[loss_for_candidates.argsort()[-n_anomaly_edges:]]

    graph = graph.copy()
    nx.set_edge_attributes(graph, 0, "anomaly")
    for i in range(n_anomaly_edges):
        graph.add_edge(top_flips[i, 0], top_flips[i, 1], anomaly=1)

    return graph
