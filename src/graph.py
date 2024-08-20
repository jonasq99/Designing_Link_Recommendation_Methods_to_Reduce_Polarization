import random

import networkx as nx
from sklearn.cluster import SpectralClustering


def graph_loader(path):
    """Load data from path, and sets the weight of the edges as 1/in-degree of the target node.
    Args:
        path (str): path to data files;
    Returns:
        G (networkx.DiGraph): Directed graph;
    """
    G = nx.read_edgelist(path, create_using=nx.DiGraph(), nodetype=int, data=False)
    print(f"Number of Nodes: {G.number_of_nodes()}")
    print(f"Number of Edges: {G.number_of_edges()}")
    for node in G.nodes():
        in_degree = G.in_degree(node)
        for neighbor in G.successors(node):
            G[node][neighbor]["weight"] = 1 / in_degree if in_degree > 0 else 1
    return G


def random_color_graph(G, colors=(1, 0)):
    """
    Randomly colors the nodes of the graph G with the specified colors.
    """
    for node in G.nodes:
        G.nodes[node]["color"] = random.choice(colors)


def spectral_bipartition_coloring(G):
    """
    Perform spectral clustering to color the graph nodes into two communities in place.

    Args:
        G (networkx.Graph): The input graph. The function will add a 'color' attribute
                            to each node in the graph.

    Returns:
        None: The graph G is modified in place with the 'color' attribute set for each node.
    """
    # Convert the graph to an adjacency matrix
    adjacency_matrix = nx.to_numpy_array(G)

    # Perform spectral clustering with 2 clusters
    sc = SpectralClustering(
        2,
        affinity="precomputed",
        n_init=100,
        assign_labels="discretize",
        random_state=42,
    )
    labels = sc.fit_predict(adjacency_matrix)

    # Assign colors based on labels
    for node, label in zip(G.nodes(), labels):
        G.nodes[node]["color"] = 0 if label == 0 else 1


def create_polarized_graph(num_nodes, intra_group_connectness, inter_group_connectness):
    """Create a polarized directed graph with two distinct groups of nodes. And set the
    weight of the edges as 1/in-degree of the target node."""
    # Create two groups of nodes
    group_1_size = num_nodes // 2
    group_2_size = num_nodes - group_1_size

    # Create two subgraphs with high intra-group connectness
    G1 = nx.gnp_random_graph(group_1_size, intra_group_connectness, directed=True)
    G2 = nx.gnp_random_graph(group_2_size, intra_group_connectness, directed=True)

    # Relabel nodes to avoid overlap
    G2 = nx.relabel_nodes(G2, lambda x: x + group_1_size)

    # Create a new graph and combine the two subgraphs
    G = nx.DiGraph()
    G.add_nodes_from(G1.nodes(data=True))
    G.add_nodes_from(G2.nodes(data=True))
    G.add_edges_from(G1.edges(data=True))
    G.add_edges_from(G2.edges(data=True))

    # Add edges between the two groups with lower inter-group connectness
    for node_1 in G1.nodes():
        for node_2 in G2.nodes():
            if random.random() < inter_group_connectness:
                G.add_edge(node_1, node_2)

    # Set the weight of the edges as 1/in-degree of the target node
    for node in G.nodes():
        in_degree = G.in_degree(node)
        for neighbor in G.successors(node):
            G[node][neighbor]["weight"] = 1 / in_degree if in_degree > 0 else 1

    return G
