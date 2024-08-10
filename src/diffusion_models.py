import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


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

    final_active_nodes = [node for node, status in nodes_status.items() if status == 2]
    final_active_node_count = len(final_active_nodes)

    return final_active_node_count, final_active_nodes, iteration_activations


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

        # Update active nodes set
        active_nodes.update(new_active_nodes)

    final_active_nodes = [node for node, status in nodes_status.items() if status == 1]
    final_active_node_count = len(final_active_nodes)

    return final_active_node_count, final_active_nodes, iteration_activations


def plot_diffusion_results(iteration_activations_list, labels):
    """Plot the number of new nodes activated after each iteration for different approaches."""
    plt.figure(figsize=(10, 6))
    for activations, label in zip(iteration_activations_list, labels):
        plt.plot(activations, marker="o", label=label)

    plt.title("Diffusion Process - New Nodes Activated Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Number of New Nodes Activated")
    plt.legend()
    plt.grid(True)
    plt.show()


def animate_diffusion_process(G, seed_sets, models, model_names, steps=6):
    """Generate snapshots of the diffusion process and animate information spread."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Define colors
    colors = ["#1f78b4", "#33a02c", "#e31a1c"]

    def update(num):
        ax.clear()
        model_idx = num // steps
        step = num % steps

        _, _, iteration_activations = models[model_idx](
            G, seed_sets[model_idx], threshold=0.1
        )
        active_nodes = set()

        for i in range(min(step + 1, len(iteration_activations))):
            _, active_nodes_at_step, _ = models[model_idx](
                G, seed_sets[model_idx], threshold=0.1
            )
            active_nodes.update(
                active_nodes_at_step[: sum(iteration_activations[: i + 1])]
            )

        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            ax=ax,
            node_color="lightgray",
            with_labels=True,
            node_size=500,
            font_size=10,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=active_nodes,
            node_color=colors[model_idx],
            ax=ax,
            node_size=500,
        )

        ax.set_title(f"{model_names[model_idx]} - Step {step + 1}")

    ani = animation.FuncAnimation(
        fig, update, frames=len(models) * steps, interval=1000, repeat=True
    )
    plt.show()
