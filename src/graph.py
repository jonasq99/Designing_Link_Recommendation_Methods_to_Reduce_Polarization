import random

import networkx as nx


def graph_loader(path):
    """Load data from path
    Args:
        path (str): path to data files;
    Returns:
        G (networkx.DiGraph): Directed graph;
    """
    G = nx.read_edgelist(path, create_using=nx.DiGraph(), nodetype=int, data=False)
    print("Number of Nodes: {}".format(G.number_of_nodes()))
    print("Number of Edges: {}".format(G.number_of_edges()))
    return G


def random_color_graph(G, colors=("red", "blue")):
    """
    Randomly colors the nodes of the graph G with the specified colors.
    """
    for node in G.nodes:
        G.nodes[node]["color"] = random.choice(colors)


def create_polarized_graph(num_nodes, intra_group_connectness, inter_group_connectness):
    """Create a polarized directed graph with two distinct groups of nodes."""
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

    return G
