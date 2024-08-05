import heapq
import itertools
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import networkx as nx
import numpy as np
from tqdm import tqdm


def f_vectorized(v, F_set, adj_matrix, opposite_color_nodes):
    if len(opposite_color_nodes[v]) == 0:
        return 0

    activated_count = np.sum(
        [(v, node) in F_set or adj_matrix[v, node] for node in opposite_color_nodes[v]]
    )
    return activated_count / len(opposite_color_nodes[v])


def tau_vectorized(R, F_set, adj_matrix, opposite_color_nodes):
    f_values = np.array(
        [f_vectorized(v, F_set, adj_matrix, opposite_color_nodes) for v in R]
    )
    return np.mean(f_values)


def edge_impact(edge, C, F_set, adj_matrix, opposite_color_nodes):
    F_set.add(edge)
    impact = tau_vectorized(C, F_set, adj_matrix, opposite_color_nodes)
    F_set.remove(edge)
    return impact


def compute_initial_impact(edge, C, adj_matrix, opposite_color_nodes):
    impact = edge_impact(edge, C, set(), adj_matrix, opposite_color_nodes)
    return -impact, edge


def get_candidate_edges(G, red_nodes, blue_nodes, perc_k=5):
    red_topk = get_topk_nodes(G, red_nodes, perc_k)
    blue_topk = get_topk_nodes(G, blue_nodes, perc_k)
    candidate_edges = list(itertools.product(red_topk, blue_topk))
    candidate_edges = candidate_edges + [(j, i) for i, j in candidate_edges]
    return set(candidate_edges)


def get_topk_nodes(G, nodes, perc_k=5):
    degrees = {i: G.degree(i) for i in nodes}
    k = int(len(degrees) / 100 * perc_k)
    topk = [i for i, _ in sorted(degrees.items(), key=lambda x: x[1], reverse=True)][:k]
    return topk


def optimize_tau(C, G, k, red_nodes, blue_nodes, perc_k=5):
    candidate_edges = get_candidate_edges(G, red_nodes, blue_nodes, perc_k)
    heap = []
    adj_matrix = nx.to_numpy_array(G)

    opposite_color_nodes = {
        v: [node for node in G.nodes if G.nodes[node]["color"] != G.nodes[v]["color"]]
        for v in G.nodes
    }

    with Pool(cpu_count()) as pool:
        initial_impacts = list(
            pool.starmap(
                compute_initial_impact,
                [
                    (edge, C, adj_matrix, opposite_color_nodes)
                    for edge in candidate_edges
                ],
            ),
            total=len(candidate_edges),
        )

    for impact in initial_impacts:
        heapq.heappush(heap, impact)

    F_set = set()
    best_tau = 0
    for _ in tqdm(range(k)):
        if not heap:
            break
        max_increase, best_edge = heapq.heappop(heap)
        F_set.add(best_edge)
        best_tau -= max_increase

        new_heap = []
        for edge in candidate_edges - F_set:
            if best_edge[0] in edge or best_edge[1] in edge:
                increase = edge_impact(edge, C, F_set, adj_matrix, opposite_color_nodes)
                heapq.heappush(new_heap, (-increase, edge))
        heap = new_heap

    return F_set, best_tau
