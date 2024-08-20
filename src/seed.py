import networkx as nx
import numpy as np


def seed_random(G, n, random_seed=42):
    """Select seeds randomly
    Args:
        G (networkx.DiGraph): Directed graph;
        n (int): number of seeds;
    Returns:
        seeds: (list) [#seed]: randomly selected seed nodes;
    """
    if random_seed:
        np.random.seed(random_seed)
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


def seed_mia(G, n, theta=0.5):
    """Select seeds by MIA (Maximum Influence Arborescence) policy.
    Args:
        G (networkx.DiGraph): Directed graph;
        n (int): number of seeds;
        theta (float): threshold for influence propagation (default is 0.5);
    Returns:
        seeds: (list) [#seed]: selected seed nodes index;
    """
    mia_scores = {}

    for node in G.nodes():
        mia_scores[node] = calculate_mia_score(G, node, theta)

    seeds = [
        node
        for node, _ in sorted(
            mia_scores.items(), key=lambda item: item[1], reverse=True
        )[:n]
    ]

    return seeds


def calculate_mia_score(G, node, theta):
    """Calculate MIA centrality score for a node using networkx.
    Args:
        G (networkx.DiGraph): Directed graph;
        node (int): Node for which to calculate MIA centrality;
        theta (float): threshold for influence propagation.
    Returns:
        mia_score: (float) MIA centrality score for the node;
    """
    visited_nodes = set()
    influence_threshold = 1  # Influence threshold for propagation
    mia_score = 0

    # Use BFS instead of DFS for efficiency
    for neighbor, path_prob in bfs_mia(G, node, influence_threshold, theta):
        if path_prob >= theta:
            visited_nodes.add(neighbor)
            mia_score += 1

    return mia_score


def bfs_mia(G, start_node, influence_threshold, theta):
    """Breadth-First Search (BFS) for MIA centrality calculation.
    Args:
        G (networkx.DiGraph): Directed graph;
        start_node (int): Starting node for BFS;
        influence_threshold (float): Threshold of influence propagation;
        theta (float): threshold for influence propagation.
    Yields:
        neighbor, path_prob: Neighbor node and the corresponding path probability.
    """
    queue = [(start_node, 1)]  # (current_node, path_probability)
    while queue:
        current_node, path_prob = queue.pop(0)
        for neighbor in G.successors(current_node):
            new_path_prob = path_prob * (influence_threshold / G.in_degree(neighbor))
            if new_path_prob >= theta:
                yield neighbor, new_path_prob
                queue.append((neighbor, new_path_prob))
