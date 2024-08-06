import numpy as np


def seed_random(G, n, random_seed=None):
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
