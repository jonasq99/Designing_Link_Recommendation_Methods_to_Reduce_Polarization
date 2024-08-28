"""
                "Edge-based Polarization",
                "Modularity-based Polarization",
                "Homophily-based Polarization",
                round(original_results_dict["polarization_scores"]["edge_based_polarization"], 3),
                round(original_results_dict["polarization_scores"]["modularity_based_polarization"], 3),
                round(original_results_dict["polarization_scores"]["homophily_based_polarization"], 3),
                
                
                
                    "Edge-based Polarization",
                    "Modularity-based Polarization",
                    "Homophily-based Polarization",
                    round(modified_results_dict["polarization_scores"]["edge_based_polarization"], 3),
                    round(modified_results_dict["polarization_scores"]["modularity_based_polarization"], 3),
                    round(modified_results_dict["polarization_scores"]["homophily_based_polarization"], 3),"""


def edge_based_polarization_score(G):
    """
    Calculate the polarization score based on the proportion of edges between different color groups.

    A higher score indicates higher polarization, meaning more edges connect nodes from different groups.
    """
    E_total = G.number_of_edges()
    E_diff = sum(1 for u, v in G.edges if G.nodes[u]["color"] != G.nodes[v]["color"])
    polarization_score = E_diff / E_total if E_total > 0 else 0
    return polarization_score


def modularity_based_polarization_score(G):
    """
    Calculate the modularity-based polarization score.

    A higher score indicates lower polarization, meaning most edges are within the same group (same color),
    suggesting strong community structure. A lower or negative score indicates higher polarization.
    """
    partition = {
        node: 0 if G.nodes[node]["color"] == "color1" else 1 for node in G.nodes
    }
    modularity_score = nx.algorithms.community.modularity(
        G,
        [
            list(filter(lambda x: partition[x] == 0, partition)),
            list(filter(lambda x: partition[x] == 1, partition)),
        ],
    )
    return modularity_score


def homophily_based_polarization_score(G):
    """
    Calculate the polarization score based on the proportion of neighbors within the same group.

    A higher score indicates higher polarization, meaning there is less local cohesion within groups.
    """
    homophily_scores = []
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        if neighbors:
            same_color_neighbors = sum(
                1
                for neighbor in neighbors
                if G.nodes[neighbor]["color"] == G.nodes[node]["color"]
            )
            homophily_scores.append(same_color_neighbors / len(neighbors))
    polarization_score = 1 - np.mean(homophily_scores)
    return polarization_score

    """# Calculate polarization scores after the simulations
    edge_polarization = edge_based_polarization_score(G)
    modularity_polarization = modularity_based_polarization_score(G)
    homophily_polarization = homophily_based_polarization_score(G)
    
    "polarization_scores": {
            "edge_based_polarization": edge_polarization,
            "modularity_based_polarization": modularity_polarization,
            "homophily_based_polarization": homophily_polarization,
        },"""
