import random
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np


def add_edges(G, seeds, k_nodes, budget):
    """Add a total of `budget` random edges from the seed nodes to k_nodes nodes.
    The weight for each edge is set as 1/in-degree of the target node before adding the edge.

    Args:
        G (nx.DiGraph): The input graph.
        seeds (List[int]): The seed nodes.
        k_nodes (List[int]): The target nodes to potentially connect to.
        budget (int): The total number of edges to add.
    """
    possible_edges = [
        (seed, target_node)
        for seed in seeds
        for target_node in k_nodes
        if not G.has_edge(seed, target_node)
    ]

    # Ensure that budget does not exceed the number of possible edges
    budget = min(budget, len(possible_edges))

    # Randomly select `budget` edges from the possible edges
    selected_edges = random.sample(possible_edges, budget)

    for seed, target_node in selected_edges:
        in_degree = G.in_degree(target_node)
        weight = 1 / in_degree if in_degree > 0 else 1
        G.add_edge(seed, target_node, weight=weight)


def edge_addition_adamic_adar(G, seeds, k, budget):
    graph = G.copy()

    # Convert to undirected graph for Adamic-Adar calculation
    undirected_graph = graph.to_undirected()
    adamic_adar_scores = list(
        nx.adamic_adar_index(
            undirected_graph,
        )
    )

    adamic_adar_scores.sort(key=lambda x: x[2], reverse=True)
    k_nodes = [n[1] for n in adamic_adar_scores[:k]]
    add_edges(graph, seeds, k_nodes, budget)

    return graph


def edge_addition_preferential_attachment(G, seeds, k):
    graph = G.copy()

    for seed in seeds:
        # Calculate the degree of all nodes in the graph
        node_degrees = dict(graph.degree())

        # Generate the cumulative distribution of degrees for random selection
        nodes, degrees = zip(*node_degrees.items())
        total_degree = sum(degrees)
        cumulative_distribution = [
            sum(degrees[: i + 1]) / total_degree for i in range(len(degrees))
        ]

        # Add edges from the seed node to k nodes chosen by the preferential attachment rule
        for _ in range(k):
            random_value = random.random()
            for i, cum_dist in enumerate(cumulative_distribution):
                if random_value <= cum_dist:
                    target_node = nodes[i]
                    # Prevent self-loops and duplicate edges
                    if target_node != seed and not graph.has_edge(seed, target_node):
                        graph.add_edge(seed, target_node)
                        break

    return graph


# Jaccard Coefficient
def edge_addition_jaccard(G, seeds, k):
    graph = G.copy()
    undirected_graph = graph.to_undirected()

    for seed in seeds:
        jaccard_scores = list(
            nx.jaccard_coefficient(
                undirected_graph,
                [(seed, n) for n in undirected_graph.nodes if n != seed],
            )
        )
        jaccard_scores.sort(key=lambda x: x[2], reverse=True)

        for i in range(min(k, len(jaccard_scores))):
            target_node = jaccard_scores[i][1]
            graph.add_edge(seed, target_node)

    return graph


# Degree
def edge_addition_degree(G, seeds, k):
    graph = G.copy()

    for seed in seeds:
        nodes_sorted_by_degree = sorted(
            graph.nodes, key=lambda n: graph.out_degree(n), reverse=True
        )
        for target_node in nodes_sorted_by_degree[:k]:
            if target_node != seed:
                graph.add_edge(seed, target_node)

    return graph


# Harmonic Centrality (Topk)
def edge_addition_topk(G, seeds, k):
    graph = G.copy()
    harmonic_centralities = nx.harmonic_centrality(graph)
    nodes_sorted_by_centrality = sorted(
        harmonic_centralities.items(), key=lambda x: x[1], reverse=True
    )

    for seed in seeds:
        for i in range(min(k, len(nodes_sorted_by_centrality))):
            target_node = nodes_sorted_by_centrality[i][0]
            if target_node != seed:
                graph.add_edge(seed, target_node)

    return graph


# Probabilistic Edge Addition (Prob)
def edge_addition_prob(G, seeds, k):
    graph = G.copy()

    for seed in seeds:
        all_possible_edges = [
            (seed, n) for n in graph.nodes if n != seed and not graph.has_edge(seed, n)
        ]
        if len(all_possible_edges) == 0:
            continue
        random.shuffle(all_possible_edges)
        selected_edges = random.sample(
            all_possible_edges, min(k, len(all_possible_edges))
        )

        for edge in selected_edges:
            graph.add_edge(*edge)

    return graph


# Kempe et al. Seed Selection (KKT)
def edge_addition_kkt(G, seeds, k):
    graph = G.copy()

    for seed in seeds:
        candidates = sorted(
            graph.nodes, key=lambda n: nx.degree_centrality(graph)[n], reverse=True
        )
        for target_node in candidates[:k]:
            if target_node != seed:
                graph.add_edge(seed, target_node)

    return graph


# Random
def edge_addition_random(G, seeds, k):
    graph = G.copy()

    available_nodes = [n for n in graph.nodes if n not in seeds]
    selected_nodes = random.sample(available_nodes, min(k, len(available_nodes)))

    for seed in seeds:
        for target_node in selected_nodes:
            graph.add_edge(seed, target_node)

    return graph


# Define the compute_initial_impact function at the top level
def compute_initial_impact(
    node: int,
    seeds: List[int],
    adj_matrix: np.ndarray,
    opposite_color_nodes: Dict[int, List[int]],
) -> Tuple[float, int]:
    """
    Compute the initial impact of adding a node.
    """
    selected_nodes = {node}
    impact = average_activation_probability(
        seeds, selected_nodes, adj_matrix, opposite_color_nodes
    )
    return -impact, node


def average_activation_probability(
    seeds: List[int],
    selected_nodes: Set[int],
    adj_matrix: np.ndarray,
    opposite_color_nodes: Dict[int, List[int]],
) -> float:
    """
    Calculate the average activation probability for a set of nodes C given selected nodes.
    """
    return np.mean(
        [
            activation_probability(v, selected_nodes, adj_matrix, opposite_color_nodes)
            for v in seeds
        ]
    )


def activation_probability(
    v: int,
    selected_nodes: Set[int],
    adj_matrix: np.ndarray,
    opposite_color_nodes: Dict[int, List[int]],
) -> float:
    """
    Calculate the activation probability for a node v given a set of selected nodes.
    """
    if len(opposite_color_nodes[v]) == 0:
        return 0
    activated_count = np.sum(
        [
            node in selected_nodes or adj_matrix[v, node]
            for node in opposite_color_nodes[v]
        ]
    )
    return activated_count / len(opposite_color_nodes[v])


def edge_addition_custom(G: nx.Graph, seeds: List[int], k: int) -> nx.Graph:
    """
    Add edges from seed nodes to a set of k selected nodes that optimize the average activation probability.

    Parameters:
    G (nx.Graph): The input graph.
    seeds (List[int]): The seed nodes from which to add edges.
    k (int): The maximum number of nodes to connect.

    Returns:
    nx.Graph: The graph with the added edges.
    """

    # Convert the graph to an adjacency matrix
    adj_matrix = nx.to_numpy_array(G)

    # Get opposite color nodes for each node in the graph
    opposite_color_nodes = {
        v: [node for node in G.nodes if G.nodes[node]["color"] != G.nodes[v]["color"]]
        for v in G.nodes
    }

    # Initialize selected nodes set
    selected_nodes = set()

    # Use a priority queue to select the best nodes
    heap = []
    with Pool(cpu_count()) as pool:
        initial_impacts = pool.starmap(
            compute_initial_impact,
            [(node, seeds, adj_matrix, opposite_color_nodes) for node in G.nodes()],
        )

    heap.extend(initial_impacts)
    heap.sort()  # Sort by impact

    for _ in range(min(k, len(heap))):
        impact, node = heap.pop(0)
        selected_nodes.add(node)

    # Add edges from each seed to the selected nodes
    graph_with_edges = G.copy()
    for seed in seeds:
        for target_node in selected_nodes:
            graph_with_edges.add_edge(seed, target_node)

    return graph_with_edges
