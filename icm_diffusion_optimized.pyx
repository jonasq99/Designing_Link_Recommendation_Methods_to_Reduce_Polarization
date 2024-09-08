import numpy as np
import cython
from libc.stdlib cimport rand
from collections import defaultdict
from cython.parallel import prange
cimport cython

@cython.boundscheck(False)  # Disable bounds-checking for faster array access
@cython.wraparound(False)  # Disable negative indexing
cpdef tuple independent_cascade_model(G, list seeds):
    """Perform diffusion using the Independent Cascade model."""
    
    cdef dict nodes_status = {node: 0 for node in G.nodes}  # 0: inactive, 1: active, 2: processed
    cdef int color_activation_count = 0  # Counter to track activations between different colors

    cdef int seed
    for seed in seeds:
        nodes_status[seed] = 1

    cdef list active_nodes = seeds[:]
    cdef list new_active_nodes
    cdef int node, neighbor, i
    cdef int num_successors
    cdef double probability

    while active_nodes:
        new_active_nodes = []
        num_successors = len(G)

        cdef np.ndarray random_values = np.random.rand(num_successors)

        for node in active_nodes:
            for i, neighbor in enumerate(G.successors(node)):
                if nodes_status[neighbor] == 0:
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

    cdef int total_activated_nodes = sum(1 for status in nodes_status.values() if status == 2)

    # Use defaultdict for counting
    cdef defaultdict color_activated_nodes = defaultdict(int)
    # Filter nodes with status == 2 and count their colors
    for node, status in nodes_status.items():
        if status == 2:
            color_activated_nodes[G.nodes[node]["color"]] += 1

    return total_activated_nodes, color_activation_count, color_activated_nodes


cpdef tuple simulate_diffusion_single_run(G, list seeds):
    """Helper function to run a single diffusion simulation."""
    return independent_cascade_model(G, seeds)


cpdef dict simulate_diffusion_ICM(G, list seeds, int num_simulations):
    """Simulate diffusion using the Independent Cascade model with parallel execution."""
    
    cdef list activated_nodes_list = []
    cdef list color_activation_counts = []
    cdef list color_activated_nodes_list = []

    # Cython's parallel loop (prange)
    for i in prange(num_simulations, nogil=True):
        activated_nodes, color_activation_count, color_activated_nodes = simulate_diffusion_single_run(G, seeds)
        activated_nodes_list.append(activated_nodes)
        color_activation_counts.append(color_activation_count)
        color_activated_nodes_list.append(color_activated_nodes)

    cdef double avg_activated_nodes = np.mean(activated_nodes_list)
    cdef double std_dev_activated_nodes = np.std(activated_nodes_list)

    cdef double avg_color_activation_count = np.mean(color_activation_counts)
    cdef double std_dev_color_activation_count = np.std(color_activation_counts)

    # Calculate average and std deviation for activated nodes by color
    cdef dict avg_color_activated_nodes = {
        color: np.mean([counts[color] for counts in color_activated_nodes_list])
        for color in color_activated_nodes_list[0]
    }
    cdef dict std_dev_color_activated_nodes = {
        color: np.std([counts[color] for counts in color_activated_nodes_list])
        for color in color_activated_nodes_list[0]
    }

    # Combine all results into a single dictionary
    cdef dict results = {
        "avg_activated_nodes": avg_activated_nodes,
        "std_dev_activated_nodes": std_dev_activated_nodes,
        "avg_color_activation_count": avg_color_activation_count,
        "std_dev_color_activation_count": std_dev_color_activation_count,
        "avg_color_activated_nodes": avg_color_activated_nodes,
        "std_dev_color_activated_nodes": std_dev_color_activated_nodes,
    }

    return results