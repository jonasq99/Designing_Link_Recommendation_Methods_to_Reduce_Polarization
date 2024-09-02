def seed_influence_maximization(G, n, l=200, num_workers=4, verbose=False):
    """Select seeds using the influence maximization policy
    Args:
        G (networkx.DiGraph): Directed graph;
        n (int): number of seeds;
        l (int): number of influence simulations (default: 500);
        num_workers (int): number of parallel workers (default: 4);
        verbose (bool): whether to print progress information (default: False);
    Returns:
        seeds (list): selected seed nodes index;
    """

    seed_set = []
    covered = [set() for _ in range(l)]
    influence_cache = {}  # Cache BFS results for nodes

    with tqdm(total=n, desc="Selecting seeds", unit="seed") as pbar:
        while len(seed_set) < n:
            best_node = None
            best_influence = -1

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_node = {
                    executor.submit(
                        compute_influence, G, node, covered, influence_cache
                    ): node
                    for node in G.nodes
                    if node not in seed_set
                }

                for future in as_completed(future_to_node):
                    node = future_to_node[future]
                    temp_influence = future.result()

                    # Choose the node with the highest influence
                    if temp_influence > best_influence:
                        best_influence = temp_influence
                        best_node = node

            # Update covered nodes with the influence of the best node
            for i in range(l):
                covered[i].update(influence_cache[best_node])

            # Add the best node to the seed set
            seed_set.append(best_node)
            pbar.update(1)  # Update progress bar

            if verbose:
                print(
                    f"Selected node {best_node} with estimated influence {best_influence}"
                )

    return seed_set


def compute_influence(G, node, covered, influence_cache):
    """Compute the influence of a node with potential parallelization.
    Args:
        G (networkx.DiGraph): Directed graph;
        node: node to compute influence for;
        covered (list): list of sets tracking influenced nodes in each simulation;
        influence_cache (dict): cache of BFS results for nodes;
    Returns:
        influence (int): calculated influence for the node.
    """
    if node not in influence_cache:
        influence_cache[node] = bfs_influence(G, node)

    influence = 0
    for i in range(len(covered)):
        influence += len(influence_cache[node] - covered[i])

    return influence


def bfs_influence(G, start_node):
    """Perform BFS to simulate influence spread and cache the results.
    Args:
        G (networkx.DiGraph): Directed graph;
        start_node: node to start BFS from;
    Returns:
        visited (set): set of influenced nodes;
    """
    visited = set()
    queue = deque([start_node])  # Use deque for efficient popping

    while queue:
        node = queue.popleft()  # deque.popleft() is O(1)
        if node not in visited:
            visited.add(node)
            neighbors = set(G.neighbors(node)) - visited
            queue.extend(neighbors)

    return visited
