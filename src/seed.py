import heapq
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import networkx as nx
import numpy as np
from tqdm import tqdm


def seed_random(G, n):
    """Select seeds randomly
    Args:
        G (networkx.DiGraph): Directed graph;
        n (int): number of seeds;
    Returns:
        seeds: (list) [#seed]: randomly selected seed nodes;
    """
    nodes = list(G.nodes())
    seeds = np.random.choice(nodes, size=n, replace=False)
    return seeds.tolist()


def seed_degree(G, n):
    """Select seeds by degree policy
    Args:
        G (networkx.DiGraph): Directed graph;
        n (int): number of seeds;
    Returns:
        seeds: (list) [#seed]: selected seed nodes index;
    """
    degree_dict = dict(G.degree())
    seeds = [
        node
        for node, _ in sorted(
            degree_dict.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ]
    return seeds


def seed_polarized(G, n, color=0):
    """Select seeds by polarized policy
    Args:
        G (networkx.DiGraph): Directed graph;
        n (int): number of seeds;
        color (int): color of the nodes one wants to select as seeds;
    Returns:
        seeds: (list) [#seed]: selected seed nodes index;
    """
    seeds = []
    color_nodes = [node for node in G.nodes() if G.nodes[node]["color"] == color]
    if len(color_nodes) <= n:
        raise ValueError(
            "Number of seeds should be less than the number of nodes, from the specified color"
        )
    else:
        seeds = np.random.choice(color_nodes, size=n, replace=False)
        return seeds.tolist()


def seed_influence_maximization(
    G, n, l=200, num_workers=4, degree_threshold=1, verbose=False
):
    """Select seeds using the influence maximization policy with optimizations
    Args:
        G (networkx.DiGraph): Directed graph;
        n (int): number of seeds;
        l (int): number of influence simulations (default: 500);
        num_workers (int): number of parallel workers (default: 4);
        degree_threshold (int): minimum degree threshold for nodes to be considered;
        verbose (bool): whether to print progress information (default: False);
    Returns:
        seeds (list): selected seed nodes index;
    """

    # Prune graph based on degree threshold
    pruned_G = G.subgraph(
        [node for node in G.nodes if G.degree(node) >= degree_threshold]
    ).copy()

    seed_set = []
    covered = [set() for _ in range(l)]
    influence_cache = {}  # Cache BFS results for nodes
    node_priority_queue = []  # Priority queue to track best nodes

    # Initialize the priority queue with influence estimates
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                compute_influence, pruned_G, node, covered, influence_cache
            ): node
            for node in pruned_G.nodes
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Initial influence calculation",
            unit="node",
        ):
            node = futures[future]
            influence = future.result()
            heapq.heappush(
                node_priority_queue, (-influence, node)
            )  # Max-heap using negative influence

    # Start selecting seeds
    with tqdm(total=n, desc="Selecting seeds", unit="seed") as pbar:
        while len(seed_set) < n:
            _, best_node = heapq.heappop(node_priority_queue)

            # If the node has already been selected, skip
            if best_node in seed_set:
                continue

            # Update covered nodes with the influence of the best node
            for i in range(l):
                covered[i].update(influence_cache[best_node])

            # Add the best node to the seed set
            seed_set.append(best_node)
            pbar.update(1)  # Update progress bar

            # Lazy update of priority queue only for affected nodes
            affected_nodes = set(pruned_G.neighbors(best_node)) - set(seed_set)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        compute_influence, pruned_G, node, covered, influence_cache
                    ): node
                    for node in affected_nodes
                }

                for future in as_completed(futures):
                    node = futures[future]
                    influence = future.result()
                    heapq.heappush(
                        node_priority_queue, (-influence, node)
                    )  # Max-heap using negative influence

            if verbose:
                print(
                    f"Selected node {best_node} with influence {influence_cache[best_node]}"
                )

    return seed_set


def compute_influence(G, node, covered, influence_cache):
    """Compute the influence of a node with potential parallelization.
    Args:
        G (networkx.DiGraph): Directed graph;
        node: node to compute influence for;
        covered (list): list of sets tracking influenced nodes in each simulation;
        influence_cache (dict): cache of BFS results for nodes;
    Returns:
        influence (int): calculated influence for the node.
    """
    if node not in influence_cache:
        influence_cache[node] = bfs_influence(G, node)

    influence = 0
    for cov in covered:
        influence += len(influence_cache[node] - cov)

    return influence


def bfs_influence(G, start_node):
    """Perform BFS to simulate influence spread and cache the results.
    Args:
        G (networkx.DiGraph): Directed graph;
        start_node: node to start BFS from;
    Returns:
        visited (set): set of influenced nodes;
    """
    visited = set()
    queue = deque([start_node])  # Use deque for efficient popping

    while queue:
        node = queue.popleft()  # deque.popleft() is O(1)
        if node not in visited:
            visited.add(node)
            neighbors = set(G.neighbors(node)) - visited
            queue.extend(neighbors)

    return visited


# TODO: Implement the following functions
def seed_centrality(G, n, centrality_type="closeness"):
    """Select seeds by centrality policy
    Args:
        G (networkx.DiGraph): Directed graph;
        n (int): number of seeds;
        centrality_type (str): type of centrality measure ('closeness' or 'betweenness');
    Returns:
        seeds: (list) [#seed]: selected seed nodes index;
    """
    if centrality_type == "closeness":
        centrality_dict = nx.closeness_centrality(G)
    elif centrality_type == "betweenness":
        centrality_dict = nx.betweenness_centrality(G)
    else:
        raise ValueError(
            "Unsupported centrality type. Use 'closeness' or 'betweenness'."
        )

    seeds = [
        node
        for node, _ in sorted(
            centrality_dict.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ]
    return seeds


def seed_centrality_mixed(G, n):
    """Select seeds by a mixed centrality policy
    Args:
        G (networkx.DiGraph): Directed graph;
        n (int): number of seeds;
    Returns:
        seeds: (list) [#seed]: selected seed nodes index;
    """
    # Calculate centrality measures
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    # Normalize centrality measures to the [0, 1] range
    max_closeness = max(closeness_centrality.values())
    max_betweenness = max(betweenness_centrality.values())

    # Calculate mixed centrality as an average of normalized centrality measures
    mixed_centrality = {
        node: (
            closeness_centrality[node] / max_closeness
            + betweenness_centrality[node] / max_betweenness
        )
        / 2
        for node in G.nodes()
    }

    # Select top-n nodes based on mixed centrality
    seeds = [
        node
        for node, _ in sorted(
            mixed_centrality.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ]

    return seeds


def seed_polarized_degree(G, n, color=0):
    """Select seeds by polarized degree policy
    Args:
        G (networkx.DiGraph): Directed graph;
        n (int): number of seeds;
        color (int): color of the nodes one wants to select as seeds;
    Returns:
        seeds: (list) [#seed]: selected seed nodes index;
    """
    color_nodes = [node for node in G.nodes() if G.nodes[node]["color"] == color]
    if len(color_nodes) < n:
        raise ValueError(
            "Number of seeds should be less than the number of nodes from the specified color"
        )

    degree_dict = {node: G.degree(node) for node in color_nodes}
    seeds = [
        node
        for node, _ in sorted(
            degree_dict.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ]
    return seeds


def seed_polarized_centrality(G, n, color=0, centrality_type="closeness"):
    """Select seeds by polarized centrality policy
    Args:
        G (networkx.DiGraph): Directed graph;
        n (int): number of seeds;
        color (int): color of the nodes one wants to select as seeds;
        centrality_type (str): type of centrality measure ('closeness' or 'betweenness');
    Returns:
        seeds: (list) [#seed]: selected seed nodes index;
    """
    color_nodes = [node for node in G.nodes() if G.nodes[node]["color"] == color]
    if len(color_nodes) < n:
        raise ValueError(
            "Number of seeds should be less than the number of nodes from the specified color"
        )

    if centrality_type == "closeness":
        centrality_dict = nx.closeness_centrality(G.subgraph(color_nodes))
    elif centrality_type == "betweenness":
        centrality_dict = nx.betweenness_centrality(G.subgraph(color_nodes))
    else:
        raise ValueError(
            "Unsupported centrality type. Use 'closeness' or 'betweenness'."
        )

    seeds = [
        node
        for node, _ in sorted(
            centrality_dict.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ]
    return seeds


def seed_polarized_centrality_mixed(G, n, color=0):
    """Select seeds by a mixed polarized centrality policy
    Args:
        G (networkx.DiGraph): Directed graph;
        n (int): number of seeds;
        color (int): color of the nodes one wants to select as seeds;
    Returns:
        seeds: (list) [#seed]: selected seed nodes index;
    """
    # Filter nodes by color
    color_nodes = [node for node in G.nodes() if G.nodes[node]["color"] == color]
    if len(color_nodes) < n:
        raise ValueError(
            "Number of seeds should be less than the number of nodes from the specified color"
        )

    # Calculate centrality measures for the subgraph
    subgraph = G.subgraph(color_nodes)
    closeness_centrality = nx.closeness_centrality(subgraph)
    betweenness_centrality = nx.betweenness_centrality(subgraph)

    # Normalize centrality measures to the [0, 1] range
    max_closeness = max(closeness_centrality.values())
    max_betweenness = max(betweenness_centrality.values())

    # Calculate mixed centrality as an average of normalized centrality measures
    mixed_centrality = {
        node: (
            closeness_centrality[node] / max_closeness
            + betweenness_centrality[node] / max_betweenness
        )
        / 2
        for node in subgraph.nodes()
    }

    # Select top-n nodes based on mixed centrality
    seeds = [
        node
        for node, _ in sorted(
            mixed_centrality.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ]

    return seeds
