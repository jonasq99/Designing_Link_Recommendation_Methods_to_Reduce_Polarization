import random

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


def seed_influence_maximization(G, n, l=500, random_seed=42, verbose=False):
    """Select seeds using the influence maximization policy
    Args:
        G (networkx.DiGraph): Directed graph;
        n (int): number of seeds;
        l (int): number of influence simulations (default: 100);
        random_seed (int): random seed for reproducibility (default: 42);
        verbose (bool): whether to print progress information (default: False);
    Returns:
        seeds (list): selected seed nodes index;
    """
    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)

    seed_set = []
    covered = [set() for _ in range(l)]

    while len(seed_set) < n:
        best_node = None
        best_influence = -1

        for node in G.nodes:
            if node in seed_set:
                continue

            influence = estimate_influence(G, node, covered)
            if influence > best_influence:
                best_influence = influence
                best_node = node

        seed_set.append(best_node)

        if verbose:
            print(
                f"Selected node {best_node} with estimated influence {best_influence}"
            )

    return seed_set


def estimate_influence(G, node, covered):
    """Estimate the influence of a node by simulating the spread of influence
    Args:
        G (networkx.DiGraph): Directed graph;
        node: node to estimate influence for;
        covered (list): list of sets tracking influenced nodes in each simulation;
    Returns:
        influence (int): estimated influence count;
    """
    influence = 0
    for i, cov in enumerate(covered):
        if node not in cov:
            influenced_nodes = bfs_influence(G, node)
            influence += len(influenced_nodes)
            cov.update(influenced_nodes)
    return influence


def bfs_influence(G, start_node):
    """Perform BFS to simulate influence spread
    Args:
        G (networkx.DiGraph): Directed graph;
        start_node: node to start BFS from;
    Returns:
        visited (set): set of influenced nodes;
    """
    visited = set()
    queue = [start_node]

    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            neighbors = set(G.neighbors(node)) - visited
            queue.extend(neighbors)

    return visited


# TODO: Implement the following functions
def seed_centrality(G, n):
    pass


def seed_polarized_degree(G, n, color=0):
    pass


def seed_polarized_centrality(G, n, color=0):
    # suggestions closeness centrality betweens centrality
    pass
