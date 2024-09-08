import concurrent.futures
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import networkx as nx
cimport cython
cimport numpy as np

# Use the types from numpy for better performance
ctypedef np.float64_t FLOAT64
ctypedef np.int32_t INT32

@cython.boundscheck(False)
@cython.wraparound(False)
def independent_cascade_model(G, list seeds):
    """Perform diffusion using the Independent Cascade model."""
    
    cdef dict nodes_status = {node: 0 for node in G.nodes}  # 0: inactive, 1: active, 2: processed
    cdef int color_activation_count = 0  # Counter to track activations between different colors
    cdef list active_nodes = seeds[:]
    cdef list new_active_nodes
    cdef float probability
    cdef int node, neighbor
    cdef np.ndarray[FLOAT64, ndim=1] random_values
    cdef int i
    
    for seed in seeds:
        nodes_status[seed] = 1

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

    return total_activated_nodes, color_activation_count, dict(color_activated_nodes)


@cython.boundscheck(False)
@cython.wraparound(False)
def simulate_diffusion_single_run(G, list seeds):
    """Helper function to run a single diffusion simulation."""
    return independent_cascade_model(G, seeds)


@cython.boundscheck(False)
@cython.wraparound(False)
def simulate_diffusion_ICM(G, list seeds, int num_simulations):
    """Simulate diffusion using the Independent Cascade model with parallel execution."""
    
    cdef list activated_nodes_list = []
    cdef list color_activation_counts = []
    cdef list color_activated_nodes_list = []
    cdef int activated_nodes, color_activation_count
    cdef dict color_activated_nodes

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(simulate_diffusion_single_run, G, seeds)
            for _ in range(num_simulations)
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=num_simulations
        ):
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
        color: np.mean([counts.get(color, 0) for counts in color_activated_nodes_list])
        for color in color_activated_nodes_list[0]
    }
    std_dev_color_activated_nodes = {
        color: np.std([counts.get(color, 0) for counts in color_activated_nodes_list])
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