import heapq
import random
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Set, Tuple

import community as community_louvain
import igraph as ig
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from src.custom_function import ParallelizedSubmodularInfluenceMaximizer
from src.icm_diffusion import (simulate_diffusion_ICM,
                               simulate_diffusion_ICM_sparse,
                               simulate_diffusion_ICM_vectorized)


def add_edges(G, seeds, k_nodes):
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
        if G.has_node(target_node):
            in_degree = G.in_degree(target_node)
        else:
            raise ValueError(f"Node {target_node} does not exist in the graph.")
        in_degree = G.in_degree(target_node)
        weight = 1 / in_degree if in_degree > 0 else 1
        # weight = 0.5
        G.add_edge(seed, target_node, weight=weight)


# Preferential Attachment
def edge_addition_preferential_attachment(G, seeds, k):
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
        if target_node not in seeds:
            k_nodes.add(target_node)

    return k_nodes


# Jaccard Coefficient
def edge_addition_jaccard(G, seeds, k):
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

    return k_nodes


# Degree
def edge_addition_degree(G, seeds, k):
    graph = G.copy()

    # Get nodes excluding seeds
    non_seed_nodes = [node for node in graph.nodes if node not in seeds]
    
    # Get the top k nodes by degree using a heap, excluding seed nodes
    k_nodes = heapq.nlargest(k, non_seed_nodes, key=graph.degree)

    return k_nodes


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
def edge_addition_topk(G_nx, seeds, k, n_jobs=-1):
    # Convert the NetworkX graph to an igraph graph
    G_ig = nx_to_igraph(G_nx)
    graph = G_nx.copy()
    # Parallel computation of harmonic centralities using igraph
    harmonic_centralities = parallel_harmonic_centrality(G_ig, n_jobs=n_jobs)

    # Get the top k nodes by harmonic centrality using a heap
    k_nodes = heapq.nlargest(k, harmonic_centralities, key=harmonic_centralities.get)

    return k_nodes


# Random Edge Addition
def edge_addition_random(G, seeds, k):
    graph = G.copy()
    available_nodes = [n for n in graph.nodes if n not in seeds]
    selected_nodes = random.sample(available_nodes, min(k, len(available_nodes)))
    return selected_nodes


#########
# Custom
#########
def edge_addition_custom(G, seeds, k):
    graph = G.copy()
    # Initialize the submodular influence maximizer
    influencer = ParallelizedSubmodularInfluenceMaximizer(graph)

    # Select seed nodes using different methods
    # greedy_seeds = influencer.greedy_submodular_selection(k=5)
    lazy_greedy_seeds = influencer.lazy_greedy_selection(k=k, initial_seeds=seeds)

    return lazy_greedy_seeds


def evaluate_nodes(args: Tuple[nx.Graph, List[int], int]) -> Tuple[int, float]:
    """
    Evaluate the influence spread of adding a target node.

    Args:
        args: Tuple containing (graph, seeds, target_node)

    Returns:
        Tuple of (target_node, avg_color_activation_count)
    """
    graph, seeds, target = args
    temp_seeds = seeds + [target]
    results = simulate_diffusion_ICM_vectorized(graph, temp_seeds, 500, verbose=False)
    return target, results["avg_color_activation_count"]


def edge_addition_custom_v2(
    G: nx.Graph, seeds: List[int], k: int, n_processes: int = None
) -> nx.Graph:
    """
    Parallelized version of edge_addition_custom_v2.

    Args:
        G (nx.Graph): The input graph.
        seeds (List[int]): Seed nodes to start the diffusion process.
        k (int): Number of nodes to select.
        budget (int): Budget for edge addition.
        n_processes (int, optional): Number of processes to use. Defaults to CPU count - 1.

    Returns:
        nx.Graph: The modified graph after adding edges.
    """
    graph = G.copy()

    # Set up multiprocessing
    if n_processes is None:
        n_processes = max(1, cpu_count() - 2)

    # Initialize target nodes
    target_nodes = list(set(G.nodes) - set(seeds))
    target_nodes = [node for node in target_nodes if G.out_degree(node) > 0]

    # Precompute the initial influence in parallel
    precompute_args = [(graph, seeds, target) for target in target_nodes]
    with Pool(processes=n_processes) as pool:
        initial_influence_results = list(
            tqdm(
                pool.imap(evaluate_nodes, precompute_args),
                total=len(precompute_args),
                desc="Precomputing influence",
            )
        )

    # Convert results to a dictionary
    initial_influence = {node: score for node, score in initial_influence_results}

    # Sort the target nodes by their initial influence
    sorted_targets = sorted(
        target_nodes, key=lambda x: initial_influence[x], reverse=True
    )

    selected_nodes = seeds.copy()

    for _ in tqdm(range(k), desc="Selecting nodes"):
        # Evaluate the top 5*k nodes
        target_candidates = sorted_targets[: 5 * k]
        candidate_args = [
            (graph, selected_nodes, target) for target in target_candidates
        ]

        with Pool(processes=n_processes) as pool:
            candidate_results = list(
                tqdm(
                    pool.imap(evaluate_nodes, candidate_args),
                    total=len(candidate_args),
                    desc="Evaluating candidates",
                )
            )

        # Convert results to a dictionary
        target_scores = {node: score for node, score in candidate_results}

        # Sort the candidates and select the best node
        best_nodes = sorted(target_scores, key=target_scores.get, reverse=True)
        best_node = best_nodes[0]

        # Add the best node to the selected nodes
        selected_nodes.append(best_node)
        sorted_targets.remove(best_node)

    return selected_nodes


def evaluate_target_node(args: Tuple[nx.Graph, List[int], int]) -> Tuple[int, Dict]:
    """
    Evaluate a single target node through diffusion simulation.

    Args:
        args: Tuple containing (graph, seeds, target_node)

    Returns:
        Tuple of (target_node, results_dict)
    """
    graph, seeds, target = args
    # Temporarily treat the target as the only seed node
    temp_seeds = seeds + [target]
    # Simulate diffusion for this configuration
    results = simulate_diffusion_ICM_vectorized(graph, temp_seeds, 500, verbose=False)

    return target, {
        "color_activation": results["avg_color_activation_count"],
        "activated_nodes": results["avg_activated_nodes"],
    }


def edge_addition_custom_v4(
    G: nx.Graph, seeds: List[int], k: int, n_processes: int = None
) -> nx.Graph:
    """
    Parallelized version of edge_addition_custom_v4 that identifies the best nodes
    to connect all seed nodes with to minimize polarization.

    Args:
        G (nx.Graph): The input graph.
        seeds (List[int]): Seed nodes to start the diffusion process.
        k (int): Number of simulations for evaluating diffusion impact.
        budget (int): Total number of nodes to connect the seed nodes to.
        n_processes (int, optional): Number of processes to use. Defaults to CPU count - 1.

    Returns:
        nx.Graph: The modified graph after adding connections to the best nodes.
    """
    graph = G.copy()

    # Set up multiprocessing
    if n_processes is None:
        n_processes = max(1, cpu_count() - 2)  # Leave one CPU free for system

    # Prepare target candidates
    target_candidates = list(set(graph.nodes) - set(seeds))

    # Remove nodes with out-degree equal to 0
    target_candidates = [
        node for node in target_candidates if graph.out_degree(node) > 0
    ]

    # Prepare arguments for parallel processing
    eval_args = [(graph, seeds, target) for target in target_candidates]

    # Create process pool and run parallel evaluations
    with Pool(processes=n_processes) as pool:
        # Use tqdm to show progress bar
        results = list(
            tqdm(
                pool.imap(evaluate_target_node, eval_args),
                total=len(target_candidates),
                desc="Evaluating nodes",
            )
        )

    # Process results
    node_scores = {}
    for target, scores in results:
        node_scores[target] = scores["color_activation"]  # + scores['activated_nodes']

    # Sort all nodes by their combined scores (higher is better)
    best_nodes = sorted(node_scores, key=node_scores.get, reverse=True)[:k]

    return best_nodes


def edge_addition_custom_v5(
    G: nx.Graph, seeds: List[int], k: int, n_processes: int = None
) -> nx.Graph:
    """
    Parallelized version of edge_addition_custom_v4 that identifies the best nodes
    to connect all seed nodes with to minimize polarization.

    Args:
        G (nx.Graph): The input graph.
        seeds (List[int]): Seed nodes to start the diffusion process.
        k (int): Number of simulations for evaluating diffusion impact.
        budget (int): Total number of nodes to connect the seed nodes to.
        n_processes (int, optional): Number of processes to use. Defaults to CPU count - 1.

    Returns:
        nx.Graph: The modified graph after adding connections to the best nodes.
    """
    graph = G.copy()

    # Prepare target candidates
    target_candidates = list(set(graph.nodes) - set(seeds))

    # Reduce the size of target candidates by removing nodes with a low out-degree
    target_candidates = [
        node for node in target_candidates if graph.out_degree(node) > 0
    ]

    # Initialize dictionary to store node scores
    node_scores = {}

    # Evaluate each target node
    for target in tqdm(target_candidates, desc="Evaluating nodes"):
        # Temporarily treat the target as the only seed node
        temp_seeds = seeds + [target]
        # Simulate diffusion for this configuration
        results = simulate_diffusion_ICM_vectorized(
            graph, temp_seeds, 500, verbose=False
        )
        # Store the average activation count for this node
        node_scores[target] = results["avg_color_activation_count"]

    # Sort all nodes by their scores (higher is better)
    best_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    best_nodes = [node for node, score in best_nodes[:k]]

    return best_nodes
