import heapq
import random
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Set, Tuple

import igraph as ig
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


# Preferential Attachment
def edge_addition_preferential_attachment(G, seeds, k, budget):
    graph = G.copy()
    node_degrees = dict(graph.degree())
    nodes, degrees = zip(*node_degrees.items())
    total_degree = sum(degrees)

    # Compute the probability distribution
    probabilities = np.array(degrees) / total_degree

    # Use a set for faster checking
    k_nodes = set()

    # Randomly select k nodes based on their degree probabilities
    while len(k_nodes) < k:
        target_node = np.random.choice(nodes, p=probabilities)
        k_nodes.add(target_node)

    add_edges(graph, seeds, list(k_nodes), budget)
    return graph


# Jaccard Coefficient
def edge_addition_jaccard(G, seeds, k, budget):
    graph = G.copy()
    undirected_graph = graph.to_undirected()

    # Use a min-heap to keep track of the top k nodes globally by Jaccard score
    top_k_heap = []

    # Parallelize the computation of Jaccard scores for seeds
    def compute_jaccard(seed):
        jaccard_scores = nx.jaccard_coefficient(
            undirected_graph,
            [(seed, n) for n in undirected_graph.nodes if n != seed],
        )
        return list(jaccard_scores)

    # Compute Jaccard scores for all seeds in parallel
    results = Parallel(n_jobs=-1)(delayed(compute_jaccard)(seed) for seed in seeds)

    # Flatten results (list of lists) into a single list
    all_jaccard_scores = [item for sublist in results for item in sublist]

    # Push Jaccard scores into the heap while maintaining the top k
    for _, v, score in all_jaccard_scores:
        if len(top_k_heap) < k:
            heapq.heappush(top_k_heap, (score, v))
        else:
            # Maintain only the top k nodes with the highest scores
            heapq.heappushpop(top_k_heap, (score, v))

    # Extract the nodes from the top k heap
    k_nodes = {node for _, node in top_k_heap}

    # Call add_edges with the selected top k nodes
    add_edges(graph, seeds, list(k_nodes), budget)
    return graph


# Degree
def edge_addition_degree(G, seeds, k, budget):
    graph = G.copy()

    # Get the top k nodes by degree using a heap
    k_nodes = heapq.nlargest(k, graph.nodes, key=graph.degree)

    add_edges(graph, seeds, k_nodes, budget)

    return graph


# Convert NetworkX graph to igraph graph
def nx_to_igraph(nx_graph):
    # Create an empty igraph graph with the same number of nodes
    num_nodes = nx_graph.number_of_nodes()
    ig_graph = ig.Graph(directed=nx_graph.is_directed())
    ig_graph.add_vertices(num_nodes)

    # Add edges to the igraph graph
    edges = [
        (list(nx_graph.nodes()).index(u), list(nx_graph.nodes()).index(v))
        for u, v in nx_graph.edges()
    ]
    ig_graph.add_edges(edges)

    # Add vertex names as attributes (if needed)
    for idx, node in enumerate(nx_graph.nodes()):
        ig_graph.vs[idx]["name"] = node

    return ig_graph


# Harmonic Centrality Calculation for a Single Node in igraph
def harmonic_centrality_single_node(graph, node):
    # Compute shortest paths for the node using igraph
    shortest_paths = graph.shortest_paths(source=node)[0]
    # Harmonic centrality: sum of inverse distances (ignoring zero distances)
    return node, sum(1 / d if d > 0 else 0 for d in shortest_paths)


# Parallel Harmonic Centrality Calculation
def parallel_harmonic_centrality(graph, n_jobs=-1):
    nodes = list(range(graph.vcount()))  # igraph uses node indices

    # Parallel computation of harmonic centrality for each node
    harmonic_centralities = Parallel(n_jobs=n_jobs)(
        delayed(harmonic_centrality_single_node)(graph, node) for node in tqdm(nodes)
    )

    # Convert to a dictionary for easy access
    return dict(harmonic_centralities)


# Main function with NetworkX to igraph conversion
def edge_addition_topk(G_nx, seeds, k, budget, n_jobs=-1):
    # Convert the NetworkX graph to an igraph graph
    G_ig = nx_to_igraph(G_nx)
    graph = G_nx.copy()
    # Parallel computation of harmonic centralities using igraph
    harmonic_centralities = parallel_harmonic_centrality(G_ig, n_jobs=n_jobs)

    # Get the top k nodes by harmonic centrality using a heap
    k_nodes = heapq.nlargest(k, harmonic_centralities, key=harmonic_centralities.get)

    # Add edges based on the top-k nodes (to the original NetworkX graph)
    add_edges(graph, seeds, k_nodes, budget)

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

# Declare global variables
adj_matrix = None
opposite_color_nodes = None
seeds = None


def init_process(adj_matrix_, opposite_color_nodes_, seeds_):
    global adj_matrix
    global opposite_color_nodes
    global seeds
    adj_matrix = adj_matrix_
    opposite_color_nodes = opposite_color_nodes_
    seeds = seeds_


def compute_initial_impact(node: int) -> Tuple[float, int]:
    """
    Compute the initial impact of adding a node.
    """
    selected_nodes = {node}
    impact = average_activation_probability(selected_nodes)
    return -impact, node


def average_activation_probability(selected_nodes: Set[int]) -> float:
    """
    Calculate the average activation probability for the seed nodes given selected nodes.
    """
    return np.mean([activation_probability(v, selected_nodes) for v in seeds])


def activation_probability(v: int, selected_nodes: Set[int]) -> float:
    """
    Calculate the activation probability for a node v given a set of selected nodes.
    """
    opp_nodes = opposite_color_nodes[v]
    if len(opp_nodes) == 0:
        return 0.0
    opp_nodes_array = np.array(opp_nodes)
    in_selected = np.isin(opp_nodes_array, list(selected_nodes))
    adjacency = adj_matrix[v, opp_nodes_array].astype(bool)
    activation = np.logical_or(in_selected, adjacency)
    activated_count = np.sum(activation)
    return activated_count / len(opp_nodes)


def edge_addition_custom(
    G: nx.Graph, seeds: List[int], k: int, budget: int
) -> nx.Graph:
    """
    Add edges from seed nodes to a set of k selected nodes that optimize the average activation probability.

    Parameters:
    G (nx.Graph): The input graph.
    seeds (List[int]): The seed nodes from which to add edges.
    k (int): The maximum number of nodes to connect.
    budget (int): The budget for adding edges.

    Returns:
    nx.Graph: The graph with the added edges.
    """

    # Create a mapping of original node IDs to consecutive integers
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    inverse_node_mapping = {idx: node for node, idx in node_mapping.items()}

    # Use the node mapping to remap the graph's nodes before converting to an adjacency matrix
    remapped_G = nx.relabel_nodes(G, node_mapping)
    adj_matrix_local = nx.to_numpy_array(remapped_G)

    # Get opposite color nodes for each node in the graph
    node_colors = nx.get_node_attributes(G, "color")
    color_groups = defaultdict(list)
    for node, color in node_colors.items():
        color_groups[color].append(node_mapping[node])  # Use mapped node IDs

    # Create the opposite_color_nodes dict using the pre-grouped nodes
    opposite_color_nodes_local = {
        v: color_groups[1 - node_colors[inverse_node_mapping[v]]]
        for v in remapped_G.nodes()
    }

    # Map seeds to remapped node IDs
    seeds_mapped = [node_mapping[s] for s in seeds]

    # Initialize selected nodes set
    selected_nodes = set()

    # Initialize the multiprocessing pool with the initializer function
    heap = []
    with Pool(
        processes=cpu_count(),
        initializer=init_process,
        initargs=(adj_matrix_local, opposite_color_nodes_local, seeds_mapped),
    ) as pool:
        with tqdm(total=len(remapped_G.nodes())) as pbar:
            for result in pool.imap_unordered(
                compute_initial_impact, remapped_G.nodes()
            ):
                heap.append(result)
                pbar.update()

    # Sort heap by impact
    heap.sort()

    # Select top-k nodes based on impact
    for _ in range(min(k, len(heap))):
        _, node = heap.pop(0)
        selected_nodes.add(node)

    # Add edges from each seed to the selected nodes
    graph_with_edges = G.copy()
    selected_nodes_original = [inverse_node_mapping[node] for node in selected_nodes]

    add_edges(
        graph_with_edges,
        seeds,
        selected_nodes_original,
        budget,
    )

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

    # Create a mapping of original node IDs to consecutive integers
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    inverse_node_mapping = {idx: node for node, idx in node_mapping.items()}

    # Remap the graph
    remapped_G = nx.relabel_nodes(G, node_mapping)
    adj_matrix_local = nx.to_numpy_array(remapped_G)

    # Get node colors and color groups
    node_colors = nx.get_node_attributes(G, "color")
    color_groups = defaultdict(list)
    for node, color in node_colors.items():
        color_groups[color].append(node_mapping[node])  # Use mapped node IDs

    # Create the opposite_color_nodes dict using the pre-grouped nodes
    node_colors_mapped = {
        node_mapping[node]: color for node, color in node_colors.items()
    }
    opposite_color_nodes_local = {
        v: color_groups[1 - node_colors_mapped[v]] for v in remapped_G.nodes()
    }

    # Map seeds to remapped node IDs
    seeds_mapped = [node_mapping[s] for s in seeds]

    # Initialize the multiprocessing pool with the initializer function
    with Pool(
        processes=cpu_count(),
        initializer=init_process,
        initargs=(adj_matrix_local, opposite_color_nodes_local, seeds_mapped),
    ) as pool:
        with tqdm(total=len(remapped_G.nodes())) as pbar:
            initial_impacts = []
            for result in pool.imap_unordered(
                compute_initial_impact, remapped_G.nodes()
            ):
                initial_impacts.append(result)
                pbar.update()

    # Initialize a priority queue to select the most impactful nodes per color
    color_impact_nodes = defaultdict(list)

    # Group nodes by their color based on impact
    for impact, node in initial_impacts:
        color = node_colors_mapped[node]
        heapq.heappush(color_impact_nodes[color], (impact, node))

    selected_nodes_per_color = {}
    num_colors = len(color_impact_nodes)
    nodes_per_color = k // num_colors if num_colors > 0 else k

    for color, nodes in color_impact_nodes.items():
        selected_nodes = [
            inverse_node_mapping[node]
            for _, node in heapq.nsmallest(nodes_per_color, nodes)
        ]
        selected_nodes_per_color[color] = selected_nodes

    # Add edges from each seed to the selected nodes from other colors considering the budget
    graph_with_edges = G.copy()
    add_edges_color(
        graph_with_edges, seeds, selected_nodes_per_color, budget, num_colors
    )

    return graph_with_edges


def extract_graph_features(
    G: nx.Graph, seeds: List[int]
) -> Dict[int, Dict[str, float]]:
    """
    Extracts advanced graph-based features for each node, including clustering coefficient, distance to seed nodes,
    and neighborhood overlap.

    Parameters:
    G (nx.Graph): The input graph.
    seeds (List[int]): The seed nodes for the diffusion process.

    Returns:
    Dict[int, Dict[str, float]]: A dictionary mapping each node to its feature set.
    """
    features = {}

    # Compute local clustering coefficient for each node
    clustering = nx.clustering(G)

    # Compute shortest path lengths from each seed node
    seed_distances = {}
    for node in G.nodes:
        try:
            # Try to find the shortest path to the nearest seed node
            seed_distances[node] = min(
                [nx.shortest_path_length(G, source=s, target=node) for s in seeds]
            )
        except nx.NetworkXNoPath:
            # If no path exists, assign a large default value for unreachable nodes
            seed_distances[node] = float("inf")

    # Compute neighborhood overlap with seed nodes
    seed_set = set(seeds)
    neighborhood_overlap = {
        node: (
            len(set(G.neighbors(node)) & seed_set)
            / len(set(G.neighbors(node)) | seed_set)
            if len(set(G.neighbors(node)) | seed_set) > 0
            else 0.0
        )
        for node in G.nodes
    }

    # Compile features into a dictionary for each node
    for node in G.nodes:
        features[node] = {
            "clustering": clustering[node],
            "distance_from_seeds": seed_distances[node],
            "neighborhood_overlap": neighborhood_overlap[node],
            "degree": G.degree[node],
        }

    return features


def score_nodes(features: Dict[int, Dict[str, float]]) -> Dict[int, float]:
    """
    Compute a score for each node based on the features. The scoring function can be tuned based on experiments or
    trained using regression on historical data.

    Parameters:
    features (Dict[int, Dict[str, float]]): The extracted features for each node.

    Returns:
    Dict[int, float]: A dictionary mapping node ID to a computed score.
    """
    node_scores = {}

    for node, feature_set in features.items():
        # Example scoring function (weights can be tuned):
        score = (
            (0.5 * feature_set["clustering"])
            - (0.3 * feature_set["distance_from_seeds"])
            + (0.7 * feature_set["neighborhood_overlap"])
            + (0.4 * feature_set["degree"])
        )

        node_scores[node] = score

    return node_scores


def select_best_nodes_advanced(G: nx.Graph, seeds: List[int], k: int) -> List[int]:
    """
    Select the best k nodes to connect to the seed nodes, based on advanced feature scoring.

    Parameters:
    G (nx.Graph): The input graph.
    seeds (List[int]): The seed nodes.
    k (int): The number of nodes to select.

    Returns:
    List[int]: The selected nodes to connect to the seed nodes.
    """
    # Step 1: Extract features from the graph
    features = extract_graph_features(G, seeds)

    # Step 2: Score the nodes based on their features
    node_scores = score_nodes(features)

    # Step 3: Sort nodes by their score and select the top-k
    sorted_nodes = sorted(node_scores.items(), key=lambda item: item[1], reverse=True)
    selected_nodes = [node for node, score in sorted_nodes[:k]]

    return selected_nodes


def edge_addition_custom_advanced(
    G: nx.Graph, seeds: List[int], k: int, budget: int
) -> nx.Graph:
    """
    Add edges from seed nodes to a set of k selected nodes based on advanced structural scoring.

    Parameters:
    G (nx.Graph): The input graph.
    seeds (List[int]): The seed nodes from which to add edges.
    k (int): The maximum number of nodes to connect.
    budget (int): The budget for adding edges.

    Returns:
    nx.Graph: The graph with the added edges.
    """
    # Step 1: Select the best k nodes using advanced scoring
    selected_nodes = select_best_nodes_advanced(G, seeds, k)

    # Step 2: Add edges from each seed to the selected nodes
    graph_with_edges = G.copy()

    add_edges(
        graph_with_edges,
        seeds,
        selected_nodes,
        budget,
    )

    return graph_with_edges
