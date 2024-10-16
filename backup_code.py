# Declare global variables
adj_matrix = None
opposite_color_nodes = None
seeds = None


def init_process(adj_matrix_, opposite_color_nodes_, seeds_):
    global adj_matrix
    global opposite_color_nodes
    global seeds
    adj_matrix = adj_matrix_
    opposite_color_nodes = opposite_color_nodes_
    seeds = seeds_


def compute_initial_impact(node: int) -> Tuple[float, int]:
    """
    Compute the initial impact of adding a node.
    """
    selected_nodes = {node}
    impact = average_activation_probability(selected_nodes)
    return -impact, node


def average_activation_probability(selected_nodes: Set[int]) -> float:
    """
    Calculate the average activation probability for the seed nodes given selected nodes.
    """
    return np.mean([activation_probability(v, selected_nodes) for v in seeds])


def activation_probability(v: int, selected_nodes: Set[int]) -> float:
    """
    Calculate the activation probability for a node v given a set of selected nodes.
    """
    opp_nodes = opposite_color_nodes[v]
    if len(opp_nodes) == 0:
        return 0.0
    opp_nodes_array = np.array(opp_nodes)
    in_selected = np.isin(opp_nodes_array, list(selected_nodes))
    adjacency = adj_matrix[v, opp_nodes_array].astype(bool)
    activation = np.logical_or(in_selected, adjacency)
    activated_count = np.sum(activation)
    return activated_count / len(opp_nodes)


def edge_addition_custom(
    G: nx.Graph, seeds: List[int], k: int, budget: int
) -> nx.Graph:
    """
    Add edges from seed nodes to a set of k selected nodes that optimize the average activation probability.

    Parameters:
    G (nx.Graph): The input graph.
    seeds (List[int]): The seed nodes from which to add edges.
    k (int): The maximum number of nodes to connect.
    budget (int): The budget for adding edges.

    Returns:
    nx.Graph: The graph with the added edges.
    """

    # Create a mapping of original node IDs to consecutive integers
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    inverse_node_mapping = {idx: node for node, idx in node_mapping.items()}

    # Use the node mapping to remap the graph's nodes before converting to an adjacency matrix
    remapped_G = nx.relabel_nodes(G, node_mapping)
    adj_matrix_local = nx.to_numpy_array(remapped_G)

    # Get opposite color nodes for each node in the graph
    node_colors = nx.get_node_attributes(G, "color")
    color_groups = defaultdict(list)
    for node, color in node_colors.items():
        color_groups[color].append(node_mapping[node])  # Use mapped node IDs

    # Create the opposite_color_nodes dict using the pre-grouped nodes
    opposite_color_nodes_local = {
        v: color_groups[1 - node_colors[inverse_node_mapping[v]]]
        for v in remapped_G.nodes()
    }

    # Map seeds to remapped node IDs
    seeds_mapped = [node_mapping[s] for s in seeds]

    # Initialize selected nodes set
    selected_nodes = set()

    # Initialize the multiprocessing pool with the initializer function
    heap = []
    with Pool(
        processes=cpu_count(),
        initializer=init_process,
        initargs=(adj_matrix_local, opposite_color_nodes_local, seeds_mapped),
    ) as pool:
        with tqdm(total=len(remapped_G.nodes())) as pbar:
            for result in pool.imap_unordered(
                compute_initial_impact, remapped_G.nodes()
            ):
                heap.append(result)
                pbar.update()

    # Sort heap by impact
    heap.sort()

    # Select top-k nodes based on impact
    for _ in range(min(k, len(heap))):
        _, node = heap.pop(0)
        selected_nodes.add(node)

    # Add edges from each seed to the selected nodes
    graph_with_edges = G.copy()
    selected_nodes_original = [inverse_node_mapping[node] for node in selected_nodes]

    add_edges(
        graph_with_edges,
        seeds,
        selected_nodes_original,
        budget,
    )

    return graph_with_edges


"---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"


# Declare global variables
adj_matrix = None
opposite_color_nodes = None
seeds = None


def init_process(adj_matrix_, opposite_color_nodes_, seeds_):
    global adj_matrix
    global opposite_color_nodes
    global seeds
    adj_matrix = adj_matrix_
    opposite_color_nodes = opposite_color_nodes_
    seeds = seeds_


def compute_marginal_gain(node: int, selected_nodes: Set[int]) -> Tuple[float, int]:
    """
    Compute the marginal gain of adding a node to the already selected nodes.
    """
    impact_before = average_activation_probability(selected_nodes)
    selected_nodes_with_node = selected_nodes | {node}  # Add the node to the set
    impact_after = average_activation_probability(selected_nodes_with_node)
    marginal_gain = impact_after - impact_before
    return -marginal_gain, node  # Use negative for a min-heap


def average_activation_probability(selected_nodes: Set[int]) -> float:
    """
    Calculate the average activation probability for the seed nodes given selected nodes.
    """
    return np.mean([activation_probability(v, selected_nodes) for v in seeds])


def activation_probability(v: int, selected_nodes: Set[int]) -> float:
    """
    Calculate the activation probability for a node v given a set of selected nodes.
    """
    opp_nodes = opposite_color_nodes[v]
    if len(opp_nodes) == 0:
        return 0.0
    opp_nodes_array = np.array(opp_nodes)
    in_selected = np.isin(opp_nodes_array, list(selected_nodes))
    adjacency = adj_matrix[v, opp_nodes_array].astype(bool)
    activation = np.logical_or(in_selected, adjacency)
    activated_count = np.sum(activation)
    return activated_count / len(opp_nodes)


from functools import partial


def edge_addition_custom(
    G: nx.Graph, seeds: List[int], k: int, budget: int
) -> nx.Graph:
    """
    Add edges from seed nodes to a set of k selected nodes that optimize the average activation probability.

    Parameters:
    G (nx.Graph): The input graph.
    seeds (List[int]): The seed nodes from which to add edges.
    k (int): The maximum number of nodes to connect.
    budget (int): The budget for adding edges.

    Returns:
    nx.Graph: The graph with the added edges.
    """

    # Step 1: Centrality Measures for Preselection
    centrality = nx.degree_centrality(G)

    # Select the top 25% of nodes based on centrality
    top_central_nodes = sorted(centrality, key=centrality.get, reverse=True)[
        : int(0.25 * len(G.nodes))
    ]

    # Create a mapping of original node IDs to consecutive integers
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    inverse_node_mapping = {idx: node for node, idx in node_mapping.items()}

    # Use the node mapping to remap the graph's nodes before converting to an adjacency matrix
    remapped_G = nx.relabel_nodes(G, node_mapping)
    adj_matrix_local = nx.to_numpy_array(remapped_G)

    # Get opposite color nodes for each node in the graph
    node_colors = nx.get_node_attributes(G, "color")
    color_groups = defaultdict(list)
    for node, color in node_colors.items():
        color_groups[color].append(node_mapping[node])  # Use mapped node IDs

    # Create the opposite_color_nodes dict using the pre-grouped nodes
    opposite_color_nodes_local = {
        v: color_groups[1 - node_colors[inverse_node_mapping[v]]]
        for v in remapped_G.nodes()
    }

    # Map seeds to remapped node IDs
    seeds_mapped = [node_mapping[s] for s in seeds]

    # Initialize selected nodes set
    selected_nodes = set()

    # Step 2: Greedy Selection with Marginal Gains
    heap = []

    # Create a partial function that includes `selected_nodes`
    partial_compute_marginal_gain = partial(
        compute_marginal_gain, selected_nodes=selected_nodes
    )

    with Pool(
        processes=cpu_count(),
        initializer=init_process,
        initargs=(adj_matrix_local, opposite_color_nodes_local, seeds_mapped),
    ) as pool:
        with tqdm(total=len(top_central_nodes)) as pbar:
            for result in pool.imap_unordered(
                partial_compute_marginal_gain, top_central_nodes
            ):
                heap.append(result)
                pbar.update()

    # Step 3: Sort heap by marginal gain and store impact scores
    heap.sort()

    # Step 4: Select top-k nodes
    for _ in range(min(k, len(heap))):
        _, node = heap.pop(0)
        selected_nodes.add(node)

    # Convert selected node IDs back to original node IDs
    selected_nodes_original = [inverse_node_mapping[node] for node in selected_nodes]

    # Step 5: Call the add_edges function
    graph_with_edges = G.copy()

    # Pass selected nodes, seeds, budget, and impact scores to the add_edges function
    add_edges(
        graph_with_edges,
        seeds,
        selected_nodes_original,
        budget,
    )

    return graph_with_edges
