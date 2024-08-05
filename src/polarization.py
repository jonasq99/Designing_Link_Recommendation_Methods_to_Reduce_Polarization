import heapq

import networkx as nx
import numpy as np
from tqdm import tqdm


def f_vectorized(v, F_set, G, opposite_color_nodes, existing_edges_map):
    """
    Vectorized function to calculate the probability that a node v activates a node in the set of nodes of the opposite color.
    """
    if len(opposite_color_nodes[v]) == 0:
        return 0

    activated_count = sum(
        1
        for node in opposite_color_nodes[v]
        if (v, node) in F_set
        or (node, v) in F_set
        or existing_edges_map[v].get(node, False)
    )
    return activated_count / len(opposite_color_nodes[v])


def tau_vectorized(R, F_set, G, opposite_color_nodes, existing_edges_map):
    """
    Vectorized function to calculate the average probability that a node in set R activates a node in the set of nodes of the opposite color.
    """
    f_values = np.array(
        [f_vectorized(v, F_set, G, opposite_color_nodes, existing_edges_map) for v in R]
    )
    return np.mean(f_values)


def edge_impact(edge, C, F_set, G, opposite_color_nodes, existing_edges_map):
    """
    Calculate the impact of adding a given edge on tau(C, F, G).
    """
    F_set.add(edge)
    impact = tau_vectorized(C, F_set, G, opposite_color_nodes, existing_edges_map)
    F_set.remove(edge)
    return impact


def optimize_tau(C, G, k):
    """
    Maximize tau(C, F) by adding up to k new edges.
    """
    all_possible_edges = {
        (u, v) for u in G.nodes for v in G.nodes if u != v and not G.has_edge(u, v)
    }
    heap = []

    # Create map of opposite color nodes for each node
    opposite_color_nodes = {
        v: [node for node in G.nodes if G.nodes[node]["color"] != G.nodes[v]["color"]]
        for v in G.nodes
    }
    existing_edges_map = {
        v: {neighbor: G.has_edge(v, neighbor) for neighbor in G.nodes if neighbor != v}
        for v in G.nodes
    }

    # Calculate initial impacts and push to heap
    for edge in tqdm(all_possible_edges):
        increase = edge_impact(
            edge, C, set(), G, opposite_color_nodes, existing_edges_map
        )
        heapq.heappush(heap, (-increase, edge))

    F_set = set()
    best_tau = 0

    for _ in tqdm(range(k)):
        if not heap:
            break
        max_increase, best_edge = heapq.heappop(heap)
        F_set.add(best_edge)
        best_tau -= max_increase

        # Update only affected edges
        new_heap = []
        for edge in all_possible_edges - F_set:
            if best_edge[0] in edge or best_edge[1] in edge:
                increase = edge_impact(
                    edge, C, F_set, G, opposite_color_nodes, existing_edges_map
                )
                heapq.heappush(new_heap, (-increase, edge))
        heap = new_heap

    return F_set, best_tau
