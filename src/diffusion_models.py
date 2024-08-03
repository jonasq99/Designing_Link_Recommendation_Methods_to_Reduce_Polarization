import numpy as np


def independent_cascade_model(G, seeds, threshold):
    """Perform diffusion using the Independent Cascade model.
    The probability of node u activating node v is threshold / in_degree(v).
    Args:
        G (networkx.DiGraph): Directed graph;
        seeds (list) [#seed]: selected seeds;
        threshold (float): influent threshold, between 0 and 1;
    Return:
        final_actived_node (int): count of influent nodes;
    """
    # Initialize node status
    nodes_status = {node: 0 for node in G.nodes}  # 0: inactive, 1: active, 2: processed
    for seed in seeds:
        nodes_status[seed] = 1

    active_nodes = seeds[:]

    while active_nodes:
        new_active_nodes = []

        for node in active_nodes:
            for neighbor in G.successors(node):
                if nodes_status[neighbor] == 0:
                    in_degree = G.in_degree(neighbor)
                    probability = threshold / in_degree
                    if np.random.rand() < probability:
                        new_active_nodes.append(neighbor)

        for node in active_nodes:
            nodes_status[node] = 2  # Mark old active nodes as processed
        for node in new_active_nodes:
            nodes_status[node] = 1  # Mark new active nodes

        active_nodes = new_active_nodes

    final_active_nodes = [node for node, status in nodes_status.items() if status == 2]
    final_active_node_count = len(final_active_nodes)

    return final_active_node_count, final_active_nodes


def linear_threshold_model(G, seeds, thresholds):
    """Perform diffusion using the Linear Threshold model
    Args:
        G (networkx.DiGraph): Directed graph with edge weights;
        seeds (list) [#seed]: selected seeds;
        thresholds (dict) {node: threshold}: threshold for each node;
    Returns:
        final_active_node_count (int): count of influent nodes;
        final_active_nodes (list): list of influent nodes;
    """
    # Initialize node status
    nodes_status = {node: 0 for node in G.nodes}  # 0: inactive, 1: active
    for seed in seeds:
        nodes_status[seed] = 1

    newly_active_nodes = seeds[:]

    while newly_active_nodes:
        new_active_nodes = []

        for node in newly_active_nodes:
            for neighbor in G.successors(node):
                if nodes_status[neighbor] == 0:  # If neighbor is inactive
                    total_weight = sum(
                        G[nbr][neighbor].get("weight", 0.1)
                        for nbr in G.predecessors(neighbor)
                        if nodes_status[nbr] == 1
                    )
                    if total_weight >= thresholds[neighbor]:
                        new_active_nodes.append(neighbor)

        for node in new_active_nodes:
            nodes_status[node] = 1  # Mark new active nodes

        newly_active_nodes = new_active_nodes

    final_active_nodes = [node for node, status in nodes_status.items() if status == 1]
    final_active_node_count = len(final_active_nodes)

    return final_active_node_count, final_active_nodes
