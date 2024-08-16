import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyvis.network import Network


def independent_cascade_model(G, seeds, threshold, random_seed=None):
    """Perform diffusion using the Independent Cascade model."""
    if random_seed:
        np.random.seed(random_seed)

    # Initialize node status
    nodes_status = {node: 0 for node in G.nodes}  # 0: inactive, 1: active, 2: processed
    for seed in seeds:
        nodes_status[seed] = 1

    active_nodes = seeds[:]
    iteration_activations = [len(active_nodes)]
    store_activations = [active_nodes]

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
        iteration_activations.append(len(new_active_nodes))
        store_activations.append(new_active_nodes)

    final_active_nodes = [node for node, status in nodes_status.items() if status == 2]
    final_active_node_count = len(final_active_nodes)

    return (
        final_active_node_count,
        final_active_nodes,
        iteration_activations,
        store_activations,
    )


def linear_threshold_model(G, seeds, thresholds, random_seed=None):
    """Perform diffusion using the Linear Threshold model."""
    if random_seed:
        np.random.seed(random_seed)

    # Initialize node status
    nodes_status = {node: 0 for node in G.nodes}  # 0: inactive, 1: active
    for seed in seeds:
        nodes_status[seed] = 1

    newly_active_nodes = seeds[:]
    iteration_activations = [len(newly_active_nodes)]
    store_activations = [newly_active_nodes]

    # Precompute degrees to avoid repeated degree lookups
    degrees = dict(G.degree())

    # Precompute active statuses
    active_nodes = {node for node, status in nodes_status.items() if status == 1}

    while newly_active_nodes:
        new_active_nodes = []

        # Create a dictionary to hold the sum of weights for each neighbor
        weights = {neighbor: 0 for neighbor in G.nodes if nodes_status[neighbor] == 0}

        for node in newly_active_nodes:
            for neighbor in G.successors(node):
                if nodes_status[neighbor] == 0:  # If neighbor is inactive
                    if degrees[neighbor]:  # Check if degree is not zero
                        weight = 1 / float(degrees[neighbor])
                        total_weight = sum(
                            weight
                            for each in G.neighbors(neighbor)
                            if each in active_nodes
                        )

                        weights[neighbor] += total_weight

        for neighbor, weight in weights.items():
            if weight >= thresholds[neighbor]:
                new_active_nodes.append(neighbor)

        for node in new_active_nodes:
            nodes_status[node] = 1  # Mark new active nodes

        newly_active_nodes = new_active_nodes
        iteration_activations.append(len(newly_active_nodes))
        store_activations.append(newly_active_nodes)

        # Update active nodes set
        active_nodes.update(new_active_nodes)

    final_active_nodes = [node for node, status in nodes_status.items() if status == 1]
    final_active_node_count = len(final_active_nodes)

    return (
        final_active_node_count,
        final_active_nodes,
        iteration_activations,
        store_activations,
    )


def plot_diffusion_results(iteration_activations_list, labels):
    """Plot the number of new nodes activated after each iteration for different approaches."""

    # rolling sum of the activations
    for i, activations in enumerate(iteration_activations_list):
        iteration_activations_list[i] = np.cumsum(activations)
    plt.figure(figsize=(10, 6))
    for activations, label in zip(iteration_activations_list, labels):
        plt.plot(activations, marker="o", label=label)

    plt.title("Diffusion Process - New Nodes Activated Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Number of active Nodes")
    plt.legend()
    plt.grid(True)
    plt.show()


def illustrate_diffusion(G, store_activations, snapshots=4):
    """
    Illustrate the diffusion process by showing activated nodes up to certain snapshots.

    Parameters:
    - G: The graph object.
    - store_activations: List of lists containing the nodes that got activated at each iteration.
    - snapshots: Number of snapshots to take during the process.
    """
    # Ensure that the first and last points are included
    total_iterations = len(store_activations)
    if snapshots > 2:
        # Generate snapshot points, ensuring the first and last are included
        snapshot_points = np.linspace(
            0, total_iterations - 1, snapshots - 2, endpoint=True, dtype=int
        ).tolist()
        snapshot_points = [0] + snapshot_points + [total_iterations - 1]
    else:
        snapshot_points = [0, total_iterations - 1]

    # Create subplots
    plt.figure(figsize=(20, 10))
    pos = nx.spring_layout(G)  # Layout for consistent node positioning
    cumulative_activated_nodes = set()  # Track all activated nodes up to each snapshot

    for i, point in enumerate(snapshot_points):
        # Accumulate activations up to the current snapshot
        for j in range(point + 1):
            cumulative_activated_nodes.update(store_activations[j])

        # Draw the graph with activated nodes up to this point
        plt.subplot(1, len(snapshot_points), i + 1)
        nx.draw(G, pos, with_labels=True, node_color="lightgrey", edge_color="grey")
        nx.draw_networkx_nodes(
            G, pos, nodelist=cumulative_activated_nodes, node_color="orange"
        )
        plt.title(f"Snapshot {i + 1}: Iteration {point}")

    plt.suptitle("Diffusion Process - Snapshots of Node Activation")
    plt.show()


def illustrate_diffusion_pyvis(G, store_activations, snapshots=4):
    """
    Illustrate the diffusion process using pyvis by showing snapshots of activated nodes at different stages.

    Parameters:
    - G: The graph on which diffusion is happening.
    - store_activations: A list where each entry represents the new nodes activated during each iteration.
    - snapshots: The number of snapshots to take (default is 4).
    """

    # Determine which iterations to capture based on the number of snapshots
    total_iterations = len(store_activations)
    snapshot_points = np.linspace(0, total_iterations - 1, snapshots, dtype=int)

    # Precompute fixed positions for nodes
    pos = nx.spring_layout(G)

    for i, point in enumerate(snapshot_points):
        net = Network(notebook=True, height="750px", width="100%", layout=False)

        # Add nodes with fixed positions
        for node in G.nodes():
            x, y = (
                pos[node][0] * 1000,
                pos[node][1] * 1000,
            )  # Scaling for better visualization
            color = (
                "red"
                if any(node in store_activations[j] for j in range(point + 1))
                else "blue"
            )
            net.add_node(n_id=node, label=str(node), color=color, x=x, y=y, fixed=True)

        # Add edges
        for edge in G.edges():
            net.add_edge(edge[0], edge[1])

        # Generate the snapshot HTML
        net.show(f"diffusion_snapshot_{i + 1}.html")
