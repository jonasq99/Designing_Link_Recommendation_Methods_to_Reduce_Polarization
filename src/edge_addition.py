import heapq
import random
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


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


# Preferential Attachment
def edge_addition_preferential_attachment(G, seeds, k, budget):
    graph = G.copy()
    node_degrees = dict(graph.degree())
    k_nodes = []

    for _ in range(k):
        nodes, degrees = zip(*node_degrees.items())
        total_degree = sum(degrees)
        cumulative_distribution = [
            sum(degrees[: i + 1]) / total_degree for i in range(len(degrees))
        ]

        random_value = random.random()
        for i, cum_dist in enumerate(cumulative_distribution):
            if random_value <= cum_dist:
                target_node = nodes[i]
                if target_node not in k_nodes:
                    k_nodes.append(target_node)
                break

    add_edges(graph, seeds, k_nodes, budget)
    return graph


# Jaccard Coefficient
def edge_addition_jaccard(G, seeds, k, budget):
    graph = G.copy()
    undirected_graph = graph.to_undirected()

    k_nodes = []
    for seed in seeds:
        jaccard_scores = list(
            nx.jaccard_coefficient(
                undirected_graph,
                [(seed, n) for n in undirected_graph.nodes if n != seed],
            )
        )
        jaccard_scores.sort(key=lambda x: x[2], reverse=True)
        k_nodes.extend([j[1] for j in jaccard_scores[:k]])

    k_nodes = list(set(k_nodes))  # Ensure uniqueness
    add_edges(graph, seeds, k_nodes, budget)
    return graph


# Degree
def edge_addition_degree(G, seeds, k, budget):
    graph = G.copy()

    # Get the top k nodes by degree using a heap
    k_nodes = heapq.nlargest(k, graph.nodes, key=graph.degree)

    add_edges(graph, seeds, k_nodes, budget)

    return graph


# Harmonic Centrality (Topk)
def edge_addition_topk(G, seeds, k, budget):
    graph = G.copy()

    # Compute harmonic centralities
    harmonic_centralities = nx.harmonic_centrality(graph)

    # Get the top k nodes by harmonic centrality using a heap
    k_nodes = heapq.nlargest(k, harmonic_centralities, key=harmonic_centralities.get)

    # Call your add_edges function with the top k nodes
    add_edges(graph, seeds, k_nodes, budget)

    return graph


# Probabilistic Edge Addition (Prob)
def edge_addition_prob(G, seeds, k, budget):
    # TODO: function isn't working correctly, since the proba relate to the indegree doesn't really make sense
    graph = G.copy()
    k_nodes = []

    for seed in seeds:
        all_possible_edges = [
            (seed, n) for n in graph.nodes if n != seed and not graph.has_edge(seed, n)
        ]
        if len(all_possible_edges) == 0:
            continue
        random.shuffle(all_possible_edges)
        k_nodes.extend([n[1] for n in all_possible_edges[:k]])

    k_nodes = list(set(k_nodes))  # Ensure uniqueness
    add_edges(graph, seeds, k_nodes[:k], budget)
    return graph


# Kempe et al. Seed Selection (KKT)
def edge_addition_kkt(G, seeds, k, budget):
    graph = G.copy()
    candidates = sorted(
        graph.nodes, key=lambda n: nx.degree_centrality(graph)[n], reverse=True
    )
    k_nodes = candidates[:k]
    add_edges(graph, seeds, k_nodes, budget)
    return graph


# Random Edge Addition
def edge_addition_random(G, seeds, k, budget):
    graph = G.copy()
    available_nodes = [n for n in graph.nodes if n not in seeds]
    selected_nodes = random.sample(available_nodes, min(k, len(available_nodes)))
    add_edges(graph, seeds, selected_nodes, budget)
    return graph


"""
CUSTOM FUNCTION
"""


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


# TODO: Add pruning to this function to only take the nodes into account with a certain degree
def edge_addition_custom(
    G: nx.Graph, seeds: List[int], k: int, budget: int
) -> nx.Graph:
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
    print("Computing initial impact for each node...")

    # Use tqdm to track progress of the initial impact computation
    with Pool(cpu_count()) as pool:
        initial_impacts = list(
            tqdm(
                pool.starmap(
                    compute_initial_impact,
                    [
                        (node, seeds, adj_matrix, opposite_color_nodes)
                        for node in G.nodes()
                    ],
                ),
                total=len(G.nodes()),
            )
        )

    print("Initial impact computation complete.")
    heap.extend(initial_impacts)
    heap.sort()  # Sort by impact

    for _ in range(min(k, len(heap))):
        _, node = heap.pop(0)
        selected_nodes.add(node)

    # Add edges from each seed to the selected nodes
    graph_with_edges = G.copy()
    add_edges(graph_with_edges, seeds, list(selected_nodes), budget)

    return graph_with_edges


def add_edges_color(G, seeds, selected_nodes_per_color, budget, num_colors):
    """
    Add a total of `budget` random edges from the seed nodes to selected nodes, considering color constraints.
    The weight for each edge is set as 1/in-degree of the target node before adding the edge.

    Args:
        G (nx.DiGraph): The input graph.
        seeds (List[int]): The seed nodes.
        selected_nodes_per_color (Dict[int, List[int]]): The top nodes selected from each color.
        budget (int): The total number of edges to add.
        num_colors (int): The number of distinct colors in the graph.
    """
    budget_per_color = budget // num_colors
    edges_added = 0

    all_possible_edges = []

    for color, nodes in selected_nodes_per_color.items():
        possible_edges = [
            (seed, target_node)
            for seed in seeds
            if G.nodes[seed]["color"] != color
            for target_node in nodes
            if not G.has_edge(seed, target_node)
        ]

        # Ensure that budget_per_color does not exceed the number of possible edges
        allocated_budget = min(budget_per_color, len(possible_edges))
        edges_added += allocated_budget

        # Randomly select `allocated_budget` edges from the possible edges
        selected_edges = random.sample(possible_edges, allocated_budget)
        all_possible_edges.extend(
            [edge for edge in possible_edges if edge not in selected_edges]
        )

        for seed, target_node in selected_edges:
            in_degree = G.in_degree(target_node)
            weight = 1 / in_degree if in_degree > 0 else 1
            G.add_edge(seed, target_node, weight=weight)

    # Handle any leftover budget
    leftover_budget = budget - edges_added

    if leftover_budget > 0 and all_possible_edges:
        additional_edges = random.sample(
            all_possible_edges, min(leftover_budget, len(all_possible_edges))
        )

        for seed, target_node in additional_edges:
            in_degree = G.in_degree(target_node)
            weight = 1 / in_degree if in_degree > 0 else 1
            G.add_edge(seed, target_node, weight=weight)


def edge_addition_custom_v2(
    G: nx.Graph, seeds: List[int], k: int, budget: int
) -> nx.Graph:
    """
    Add edges from seed nodes to a set of selected nodes that optimize the average activation probability.

    Parameters:
    G (nx.Graph): The input graph.
    seeds (List[int]): The seed nodes from which to add edges.
    k (int): The maximum number of nodes to connect.
    budget (int): The maximum number of edges to add.

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

    # Initialize a priority queue to select the most impactful nodes per color
    color_impact_nodes = defaultdict(list)

    print("Computing initial impact for each node...")

    # Use tqdm to track progress of the initial impact computation
    with Pool(cpu_count()) as pool:
        initial_impacts = list(
            tqdm(
                pool.starmap(
                    compute_initial_impact,
                    [
                        (node, seeds, adj_matrix, opposite_color_nodes)
                        for node in G.nodes()
                    ],
                ),
                total=len(G.nodes()),
            )
        )

    print(
        "Initial impact computation complete. Grouping nodes by color based on impact..."
    )

    # Group nodes by their color based on impact
    for impact, node in initial_impacts:
        color = G.nodes[node]["color"]
        heapq.heappush(color_impact_nodes[color], (impact, node))

    selected_nodes_per_color = {}
    num_colors = len(color_impact_nodes)
    nodes_per_color = k // num_colors

    for color, nodes in color_impact_nodes.items():
        selected_nodes_per_color[color] = [
            node for _, node in heapq.nsmallest(nodes_per_color, nodes)
        ]

    # Add edges from each seed to the selected nodes from other colors considering the budget
    graph_with_edges = G.copy()
    add_edges_color(
        graph_with_edges, seeds, selected_nodes_per_color, budget, num_colors
    )

    return graph_with_edges
