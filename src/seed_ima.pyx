# cython: language_level=3
import cython
from cython.parallel import prange
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import deque  # Import deque here

@cython.boundscheck(False)
@cython.wraparound(False)
def seed_influence_maximization(
    G, int n, int l=200, int num_workers=4, verbose=False
):
    """
    Select seeds using the influence maximization policy without pruning
    Args:
        G (networkx.DiGraph): Directed graph;
        n (int): number of seeds;
        l (int): number of influence simulations (default: 200);
        num_workers (int): number of parallel workers (default: 4);
        verbose (bool): whether to print progress information (default: False);
    Returns:
        seeds (list): selected seed nodes index;
    """
    import networkx as nx  # Import here to avoid issues with Cython
    cdef dict node_to_idx = {}
    cdef dict idx_to_node = {}
    cdef int idx = 0
    cdef object node

    # Map nodes to integer indices
    for node in G.nodes():
        node_to_idx[node] = idx
        idx_to_node[idx] = node
        idx += 1

    cdef int num_nodes = idx
    cdef list adjacency_list = [ [] for _ in range(num_nodes) ]

    # Create adjacency list with integer indices
    for node in G.nodes():
        idx = node_to_idx[node]
        adjacency_list[idx] = [node_to_idx[neighbor] for neighbor in G.neighbors(node)]

    # Initialize variables
    cdef list seed_set = []
    cdef list covered = [set() for _ in range(l)]
    cdef dict influence_cache = {}
    cdef list node_priority_queue = []
    cdef int influence, best_idx, negative_influence

    # Initialize the priority queue with influence estimates
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                compute_influence, adjacency_list, idx, covered, influence_cache
            ): idx
            for idx in range(num_nodes)
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Initial influence calculation",
            unit="node",
        ):
            idx = futures[future]
            influence = future.result()
            heapq.heappush(node_priority_queue, (-influence, idx))

    # Start selecting seeds
    with tqdm(total=n, desc="Selecting seeds", unit="seed") as pbar:
        while len(seed_set) < n:
            negative_influence, best_idx = heapq.heappop(node_priority_queue)

            # If the node has already been selected, skip
            if best_idx in seed_set:
                continue

            # Update covered nodes with the influence of the best node
            for i in range(l):
                covered[i].update(influence_cache[best_idx])

            # Add the best node to the seed set
            seed_set.append(best_idx)
            pbar.update(1)  # Update progress bar

            # Lazy update of priority queue only for affected nodes
            affected_indices = set(adjacency_list[best_idx]) - set(seed_set)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        compute_influence, adjacency_list, idx, covered, influence_cache
                    ): idx
                    for idx in affected_indices
                }

                for future in as_completed(futures):
                    idx = futures[future]
                    influence = future.result()
                    heapq.heappush(node_priority_queue, (-influence, idx))


    # Map indices back to nodes
    seed_set_nodes = [idx_to_node[idx] for idx in seed_set]

    return seed_set_nodes

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_influence(adjacency_list, int idx, covered, influence_cache):
    """
    Compute the influence of a node with potential parallelization.
    Args:
        adjacency_list (list): adjacency list of the graph;
        idx (int): node index to compute influence for;
        covered (list): list of sets tracking influenced nodes in each simulation;
        influence_cache (dict): cache of BFS results for nodes;
    Returns:
        influence (int): calculated influence for the node.
    """
    cdef int influence = 0
    cdef set node_influence_set

    if idx not in influence_cache:
        influence_cache[idx] = bfs_influence(adjacency_list, idx)

    node_influence_set = influence_cache[idx]
    for cov in covered:
        influence += len(node_influence_set - cov)

    return influence

@cython.boundscheck(False)
@cython.wraparound(False)
def bfs_influence(adjacency_list, int start_idx):
    """
    Perform BFS to simulate influence spread and cache the results.
    Args:
        adjacency_list (list): adjacency list representation of the graph;
        start_idx (int): index of node to start BFS from;
    Returns:
        visited (set): set of influenced node indices;
    """
    cdef set visited = set()
    cdef list queue = [start_idx]
    cdef int idx, neighbor_idx, i = 0
    cdef list neighbors

    while i < len(queue):
        idx = queue[i]
        i += 1
        if idx not in visited:
            visited.add(idx)
            neighbors = adjacency_list[idx]
            for neighbor_idx in neighbors:
                if neighbor_idx not in visited:
                    queue.append(neighbor_idx)
    return visited