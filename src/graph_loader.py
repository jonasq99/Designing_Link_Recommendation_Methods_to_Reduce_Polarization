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
