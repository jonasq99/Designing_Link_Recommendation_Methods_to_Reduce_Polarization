import random
from collections import Counter

import networkx as nx
from node2vec import Node2Vec
from sklearn.cluster import KMeans, SpectralClustering


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
        G.nodes[node]["color"] = label


def k_means_partition_coloring(G, n_clusters=2):
    """
    Perform K-Means clustering on node embeddings to color the graph nodes into communities in place.

    Args:
        G (networkx.Graph): The input graph. The function will add a 'color' attribute
                            to each node in the graph.
        n_clusters (int): The number of clusters to form.

    Returns:
        None: The graph G is modified in place with the 'color' attribute set for each node.
    """
    # Generate node embeddings using Node2Vec
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Extract embeddings for each node
    embeddings = [model.wv[str(node)] for node in G.nodes()]

    # Perform K-Means clustering on the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Assign colors based on cluster labels
    for node, label in zip(G.nodes(), labels):
        G.nodes[node]["color"] = label


def spectral_partition_coloring(G, num_groups=2):
    """
    Perform spectral clustering to color the graph nodes into a specified number of communities in place.
    Tries to balance the group sizes.

    Args:
        G (networkx.Graph): The input graph. The function will add a 'color' attribute
                            to each node in the graph.
        num_groups (int): The number of groups to partition the nodes into.

    Returns:
        None: The graph G is modified in place with the 'color' attribute set for each node.
    """
    # Convert the graph to an adjacency matrix
    adjacency_matrix = nx.to_numpy_array(G)

    # Perform spectral clustering with the specified number of clusters
    sc = SpectralClustering(
        num_groups,
        affinity="precomputed",
        n_init=100,
        assign_labels="discretize",
        random_state=42,
    )
    labels = sc.fit_predict(adjacency_matrix)

    # Count the number of nodes in each cluster
    label_counts = Counter(labels)
    desired_size = len(G.nodes) // num_groups

    # Sort labels by their size (smallest to largest)
    sorted_labels = sorted(label_counts, key=label_counts.get)

    # Balance the group sizes
    for i in range(len(sorted_labels) - 1):
        current_label = sorted_labels[i]
        next_label = sorted_labels[i + 1]

        while (
            label_counts[current_label] < desired_size
            and label_counts[next_label] > desired_size
        ):
            # Find a node to swap from next_label to current_label
            swap_node = next(
                node for node, label in enumerate(labels) if label == next_label
            )
            labels[swap_node] = current_label
            label_counts[current_label] += 1
            label_counts[next_label] -= 1

    # Assign colors to nodes
    for node, label in zip(G.nodes(), labels):
        G.nodes[node]["color"] = label


def create_polarized_graph_multiple(
    num_nodes, num_groups, intra_group_connectness, inter_group_connectness
):
    """Create a polarized directed graph with multiple distinct groups of nodes. And set the
    weight of the edges as 1/in-degree of the target node."""
    # Calculate the size of each group
    group_sizes = [
        num_nodes // num_groups + (1 if x < num_nodes % num_groups else 0)
        for x in range(num_groups)
    ]

    # Create a list to store each subgraph
    subgraphs = []
    current_node_index = 0

    # Create subgraphs with high intra-group connectness
    for size in group_sizes:
        G_sub = nx.gnp_random_graph(size, intra_group_connectness, directed=True)
        # Relabel nodes to avoid overlap
        G_sub = nx.relabel_nodes(G_sub, lambda x, idx=current_node_index: x + idx)
        subgraphs.append(G_sub)
        current_node_index += size

    # Create a new graph and combine the subgraphs
    G = nx.DiGraph()
    for subgraph in subgraphs:
        G.add_nodes_from(subgraph.nodes(data=True))
        G.add_edges_from(subgraph.edges(data=True))

    # Add edges between the groups with lower inter-group connectness
    for i in range(len(subgraphs)):
        for j in range(i + 1, len(subgraphs)):
            for node_1 in subgraphs[i].nodes():
                for node_2 in subgraphs[j].nodes():
                    if random.random() < inter_group_connectness:
                        G.add_edge(node_1, node_2)
                    if random.random() < inter_group_connectness:
                        G.add_edge(node_2, node_1)  # Add edge in both directions

    # Set the weight of the edges as 1/in-degree of the target node
    for node in G.nodes():
        in_degree = G.in_degree(node)
        for neighbor in G.successors(node):
            G[node][neighbor]["weight"] = 1 / in_degree if in_degree > 0 else 1

    print(f"Number of Nodes: {G.number_of_nodes()}")
    print(f"Number of Edges: {G.number_of_edges()}")
    return G


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

    print(f"Number of Nodes: {G.number_of_nodes()}")
    print(f"Number of Edges: {G.number_of_edges()}")
    return G
