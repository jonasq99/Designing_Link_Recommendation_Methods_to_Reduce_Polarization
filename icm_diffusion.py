import concurrent.futures

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

    return total_activated_nodes, color_activation_count


def simulate_diffusion_single_run(G, seeds):
    """Helper function to run a single diffusion simulation."""
    return independent_cascade_model(G, seeds)


def edge_based_polarization_score(G):
    """
    Calculate the polarization score based on the proportion of edges between different color groups.

    A higher score indicates higher polarization, meaning more edges connect nodes from different groups.
    """
    E_total = G.number_of_edges()
    E_diff = sum(1 for u, v in G.edges if G.nodes[u]["color"] != G.nodes[v]["color"])
    polarization_score = E_diff / E_total if E_total > 0 else 0
    return polarization_score


def modularity_based_polarization_score(G):
    """
    Calculate the modularity-based polarization score.

    A higher score indicates lower polarization, meaning most edges are within the same group (same color),
    suggesting strong community structure. A lower or negative score indicates higher polarization.
    """
    partition = {
        node: 0 if G.nodes[node]["color"] == "color1" else 1 for node in G.nodes
    }
    modularity_score = nx.algorithms.community.modularity(
        G,
        [
            list(filter(lambda x: partition[x] == 0, partition)),
            list(filter(lambda x: partition[x] == 1, partition)),
        ],
    )
    return modularity_score


def homophily_based_polarization_score(G):
    """
    Calculate the polarization score based on the proportion of neighbors within the same group.

    A higher score indicates higher polarization, meaning there is less local cohesion within groups.
    """
    homophily_scores = []
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        if neighbors:
            same_color_neighbors = sum(
                1
                for neighbor in neighbors
                if G.nodes[neighbor]["color"] == G.nodes[node]["color"]
            )
            homophily_scores.append(same_color_neighbors / len(neighbors))
    polarization_score = 1 - np.mean(homophily_scores)
    return polarization_score


def simulate_diffusion_ICM(G, seeds, num_simulations):
    """Simulate diffusion using the Independent Cascade model with parallel execution."""
    activated_nodes_list = []
    color_activation_counts = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(simulate_diffusion_single_run, G, seeds)
            for _ in range(num_simulations)
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=num_simulations
        ):
            activated_nodes, color_activation_count = future.result()
            activated_nodes_list.append(activated_nodes)
            color_activation_counts.append(color_activation_count)

    avg_activated_nodes = np.mean(activated_nodes_list)
    std_dev_activated_nodes = np.std(activated_nodes_list)

    avg_color_activation_count = np.mean(color_activation_counts)
    std_dev_color_activation_count = np.std(color_activation_counts)

    """# Calculate polarization scores after the simulations
    edge_polarization = edge_based_polarization_score(G)
    modularity_polarization = modularity_based_polarization_score(G)
    homophily_polarization = homophily_based_polarization_score(G)
    
    "polarization_scores": {
            "edge_based_polarization": edge_polarization,
            "modularity_based_polarization": modularity_polarization,
            "homophily_based_polarization": homophily_polarization,
        },"""

    # Combine all results into a single dictionary
    results = {
        "avg_activated_nodes": avg_activated_nodes,
        "std_dev_activated_nodes": std_dev_activated_nodes,
        "avg_color_activation_count": avg_color_activation_count,
        "std_dev_color_activation_count": std_dev_color_activation_count,
    }

    return results
