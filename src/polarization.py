import heapq
import itertools
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm


def f_vectorized(
    v: int,
    F_set: Set[Tuple[int, int]],
    adj_matrix: np.ndarray,
    opposite_color_nodes: Dict[int, List[int]],
) -> float:
    """
    Calculate the activation probability for a node v given a set of added edges.

    Parameters:
    v (int): The node for which to calculate the activation probability.
    F_set (Set[Tuple[int, int]]): The set of added edges.
    adj_matrix (np.ndarray): The adjacency matrix of the graph.
    opposite_color_nodes (Dict[int, List[int]]): A dictionary mapping each node
    to its opposite color nodes.

    Returns:
    float: The activation probability for the node v.
    """
    if len(opposite_color_nodes[v]) == 0:
        return 0

    activated_count = np.sum(
        [(v, node) in F_set or adj_matrix[v, node] for node in opposite_color_nodes[v]]
    )
    return activated_count / len(opposite_color_nodes[v])


def tau_vectorized(
    R: List[int],
    F_set: Set[Tuple[int, int]],
    adj_matrix: np.ndarray,
    opposite_color_nodes: Dict[int, List[int]],
) -> float:
    """
    Calculate the average activation probability for a set of nodes R.

    Parameters:
    R (List[int]): The set of nodes for which to calculate the average activation probability.
    F_set (Set[Tuple[int, int]]): The set of added edges.
    adj_matrix (np.ndarray): The adjacency matrix of the graph.
    opposite_color_nodes (Dict[int, List[int]]): A dictionary mapping each node to
    its opposite color nodes.

    Returns:
    float: The average activation probability for the set of nodes R.
    """
    f_values = np.array(
        [f_vectorized(v, F_set, adj_matrix, opposite_color_nodes) for v in R]
    )
    return np.mean(f_values)


def edge_impact(
    edge: Tuple[int, int],
    C: List[int],
    F_set: Set[Tuple[int, int]],
    adj_matrix: np.ndarray,
    opposite_color_nodes: Dict[int, List[int]],
) -> float:
    """
    Calculate the impact of adding a specific edge on the average activation probability.

    Parameters:
    edge (Tuple[int, int]): The edge to add.
    C (List[int]): The set of nodes of a given color.
    F_set (Set[Tuple[int, int]]): The current set of added edges.
    adj_matrix (np.ndarray): The adjacency matrix of the graph.
    opposite_color_nodes (Dict[int, List[int]]): A dictionary mapping each
    node to its opposite color nodes.

    Returns:
    float: The impact of adding the edge.
    """
    F_set.add(edge)
    impact = tau_vectorized(C, F_set, adj_matrix, opposite_color_nodes)
    F_set.remove(edge)
    return impact


def compute_initial_impact(
    edge: Tuple[int, int],
    C: List[int],
    adj_matrix: np.ndarray,
    opposite_color_nodes: Dict[int, List[int]],
) -> Tuple[float, Tuple[int, int]]:
    """
    Compute the initial impact of an edge without any pre-existing edges in the set.

    Parameters:
    edge (Tuple[int, int]): The edge to evaluate.
    C (List[int]): The set of nodes of a given color.
    adj_matrix (np.ndarray): The adjacency matrix of the graph.
    opposite_color_nodes (Dict[int, List[int]]): A dictionary mapping each
    node to its opposite color nodes.

    Returns:
    Tuple[float, Tuple[int, int]]: A tuple containing the negative impact and the edge.
    """
    impact = edge_impact(edge, C, set(), adj_matrix, opposite_color_nodes)
    return -impact, edge


def get_candidate_edges(
    G: nx.Graph, red_nodes: List[int], blue_nodes: List[int], perc_k: int = 5
) -> Set[Tuple[int, int]]:
    """
    Get candidate edges based on top-k nodes by degree from red and blue nodes.

    Parameters:
    G (nx.Graph): The input graph.
    red_nodes (List[int]): List of red nodes.
    blue_nodes (List[int]): List of blue nodes.
    perc_k (int): Percentage of top-k nodes to consider. Defaults to 5.

    Returns:
    Set[Tuple[int, int]]: Set of candidate edges.
    """
    red_topk = get_topk_nodes_by_centrality(G, red_nodes, perc_k)
    blue_topk = get_topk_nodes_by_centrality(G, blue_nodes, perc_k)
    candidate_edges = list(itertools.product(red_topk, blue_topk))
    candidate_edges = candidate_edges + [(j, i) for i, j in candidate_edges]
    return set(candidate_edges)


def get_topk_nodes(G: nx.Graph, nodes: List[int], perc_k: int = 5) -> List[int]:
    """
    Get the top-k nodes by degree from a list of nodes.

    Parameters:
    G (nx.Graph): The input graph.
    nodes (List[int]): List of nodes to evaluate.
    perc_k (int): Percentage of top-k nodes to consider. Defaults to 5.

    Returns:
    List[int]: List of top-k nodes by degree.
    """
    degrees = {i: G.degree(i) for i in nodes}
    k = int(len(degrees) / 100 * perc_k)
    topk = [i for i, _ in sorted(degrees.items(), key=lambda x: x[1], reverse=True)][:k]
    return topk


def get_topk_nodes_by_centrality(
    G: nx.Graph, nodes: List[int], perc_k: int = 5
) -> List[int]:
    """
    Get the top-k nodes by betweenness centrality from a list of nodes.

    Parameters:
    G (nx.Graph): The input graph.
    nodes (List[int]): List of nodes to evaluate.
    perc_k (int): Percentage of top-k nodes to consider. Defaults to 5.

    Returns:
    List[int]: List of top-k nodes by betweenness centrality.
    """
    centrality = nx.betweenness_centrality(G, normalized=True, endpoints=True)
    node_centrality = {i: centrality[i] for i in nodes}
    k = int(len(node_centrality) / 100 * perc_k)
    topk = [
        i for i, _ in sorted(node_centrality.items(), key=lambda x: x[1], reverse=True)
    ][:k]
    return topk


def optimize_tau(
    C: List[int],
    G: nx.Graph,
    k: int,
    red_nodes: List[int],
    blue_nodes: List[int],
    perc_k: int = 5,
) -> Tuple[Set[Tuple[int, int]], float]:
    """
    Optimize the average activation probability by adding up to k edges to the graph.

    Parameters:
    C (List[int]): The set of nodes of a given color.
    G (nx.Graph): The input graph.
    k (int): The maximum number of edges to add.
    red_nodes (List[int]): List of red nodes.
    blue_nodes (List[int]): List of blue nodes.
    perc_k (int): Percentage of top-k nodes to consider for candidate edges. Defaults to 5.

    Returns:
    Tuple[Set[Tuple[int, int]], float]: A tuple containing the set of added edges and the best
    average activation probability.
    """

    candidate_edges = get_candidate_edges(G, red_nodes, blue_nodes, perc_k)
    print(f"Number of candidate edges: {len(candidate_edges)}")
    heap = []
    adj_matrix = nx.to_numpy_array(G)

    opposite_color_nodes = {
        v: [node for node in G.nodes if G.nodes[node]["color"] != G.nodes[v]["color"]]
        for v in G.nodes
    }

    with Pool(cpu_count()) as pool:
        initial_impacts = pool.starmap(
            compute_initial_impact,
            [(edge, C, adj_matrix, opposite_color_nodes) for edge in candidate_edges],
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
