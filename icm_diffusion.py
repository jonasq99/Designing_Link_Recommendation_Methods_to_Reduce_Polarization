import concurrent.futures
from collections import defaultdict

import networkx as nx
import numpy as np
from tqdm import tqdm


def independent_cascade_model(G, seeds):
    """Perform diffusion using the Independent Cascade model."""
    nodes_status = {node: 0 for node in G.nodes}  # 0: inactive, 1: active, 2: processed
    color_activation_count = 0  # Counter to track activations between different colors

    for seed in seeds:
        nodes_status[seed] = 1

    active_nodes = seeds[:]

    while active_nodes:
        new_active_nodes = []
        random_values = np.random.rand(len(G))

        for node in active_nodes:
            for i, neighbor in enumerate(G.successors(node)):
                if nodes_status[neighbor] == 0:
                    # Use the precomputed weight as the probability
                    probability = G[node][neighbor]["weight"]
                    if random_values[i] <= probability:
                        new_active_nodes.append(neighbor)

                        # Check the color of the nodes and update the count if colors differ
                        if G.nodes[node]["color"] != G.nodes[neighbor]["color"]:
                            color_activation_count += 1

        for node in active_nodes:
            nodes_status[node] = 2  # Mark old active nodes as processed
        for node in new_active_nodes:
            nodes_status[node] = 1  # Mark new active nodes

        active_nodes = new_active_nodes

    total_activated_nodes = sum(1 for status in nodes_status.values() if status == 2)

    # Use defaultdict for counting
    color_activated_nodes = defaultdict(int)
    # Filter nodes with status == 2 and count their colors
    for node, status in nodes_status.items():
        if status == 2:
            color_activated_nodes[G.nodes[node]["color"]] += 1

    return total_activated_nodes, color_activation_count, color_activated_nodes


def simulate_diffusion_single_run(G, seeds):
    """Helper function to run a single diffusion simulation."""
    return independent_cascade_model(G, seeds)


def simulate_diffusion_ICM(G, seeds, num_simulations):
    """Simulate diffusion using the Independent Cascade model with parallel execution."""
    activated_nodes_list = []
    color_activation_counts = []
    color_activated_nodes_list = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(simulate_diffusion_single_run, G, seeds)
            for _ in range(num_simulations)
        ]
        """for future in tqdm(
            concurrent.futures.as_completed(futures), total=num_simulations
        ):"""
        for future in concurrent.futures.as_completed(futures):
            activated_nodes, color_activation_count, color_activated_nodes = (
                future.result()
            )
            activated_nodes_list.append(activated_nodes)
            color_activation_counts.append(color_activation_count)
            color_activated_nodes_list.append(color_activated_nodes)

    avg_activated_nodes = np.mean(activated_nodes_list)
    std_dev_activated_nodes = np.std(activated_nodes_list)

    avg_color_activation_count = np.mean(color_activation_counts)
    std_dev_color_activation_count = np.std(color_activation_counts)

    # Calculate average and std deviation for activated nodes by color
    avg_color_activated_nodes = {
        color: np.mean([counts[color] for counts in color_activated_nodes_list])
        for color in color_activated_nodes_list[0]
    }
    std_dev_color_activated_nodes = {
        color: np.std([counts[color] for counts in color_activated_nodes_list])
        for color in color_activated_nodes_list[0]
    }

    # Combine all results into a single dictionary
    results = {
        "avg_activated_nodes": avg_activated_nodes,
        "std_dev_activated_nodes": std_dev_activated_nodes,
        "avg_color_activation_count": avg_color_activation_count,
        "std_dev_color_activation_count": std_dev_color_activation_count,
        "avg_color_activated_nodes": avg_color_activated_nodes,
        "std_dev_color_activated_nodes": std_dev_color_activated_nodes,
    }

    return results
