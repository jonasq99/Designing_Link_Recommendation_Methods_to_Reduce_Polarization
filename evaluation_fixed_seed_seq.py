from multiprocessing import Pool, cpu_count
from typing import Dict, List, Set

import pandas as pd
from sympy import true

from src.edge_addition_seq import (
    edge_addition_custom,
    edge_addition_custom_v2,
    edge_addition_custom_v4,
    edge_addition_custom_v5,
    edge_addition_degree,
    edge_addition_jaccard,
    edge_addition_preferential_attachment,
    edge_addition_random,
    edge_addition_topk,
)
from src.icm_diffusion import simulate_diffusion_ICM, simulate_diffusion_ICM_vectorized


def evaluate_graph_modifications_incremental(
    G, seeds, k_values, max_iter, verbose=True
):
    """
    Evaluate graph modifications incrementally for different k values.

    Args:
        G (nx.Graph): Input graph
        seeds (List[int]): Seed nodes
        k_values (List[int]): List of k values to evaluate incrementally
        max_iter (int): Maximum iterations for ICM diffusion
        verbose (bool): Whether to print progress

    Returns:
        pd.DataFrame: Results for each k value
    """

    def create_metrics_df(results_dict, graph_name, current_k):
        avg_metrics = [
            {
                "Metric": "Avg Activated Nodes",
                "Graph Modification": graph_name,
                "K Value": current_k,
                "Value": round(results_dict["avg_activated_nodes"], 3),
            },
            {
                "Metric": "Avg Color Activation Count",
                "Graph Modification": graph_name,
                "K Value": current_k,
                "Value": round(results_dict["avg_color_activation_count"], 3),
            },
        ]
        std_metrics = [
            {
                "Metric": "Activated Nodes Std Dev",
                "Graph Modification": graph_name,
                "K Value": current_k,
                "Value": round(results_dict["std_dev_activated_nodes"], 3),
            },
            {
                "Metric": "Color Activation Count Std Dev",
                "Graph Modification": graph_name,
                "K Value": current_k,
                "Value": round(results_dict["std_dev_color_activation_count"], 3),
            },
        ]

        for color, avg_count in results_dict["avg_color_activated_nodes"].items():
            avg_metrics.append(
                {
                    "Metric": f"Avg Activated Nodes, Color ({color})",
                    "Graph Modification": graph_name,
                    "K Value": current_k,
                    "Value": round(avg_count, 3),
                }
            )

        for color, std_dev_count in results_dict[
            "std_dev_color_activated_nodes"
        ].items():
            std_metrics.append(
                {
                    "Metric": f"Activated Nodes Std Dev, Color ({color})",
                    "Graph Modification": graph_name,
                    "K Value": current_k,
                    "Value": round(std_dev_count, 3),
                }
            )

        return pd.DataFrame(avg_metrics + std_metrics)

    # Simulate diffusion on the original graph
    if verbose:
        print("    Running evaluation for original graph")
    original_results_dict = simulate_diffusion_ICM_vectorized(
        G, seeds, max_iter, verbose
    )
    original_results = create_metrics_df(original_results_dict, "Original Graph", 0)

    # Get graph information
    graph_info = {
        "Metric": ["Number of Nodes", "Number of Edges"],
        "Graph Modification": ["Original Graph", "Original Graph"],
        "K Value": [0, 0],
        "Value": [G.number_of_nodes(), G.number_of_edges()],
    }
    graph_info_df = pd.DataFrame(graph_info)

    # Define modification functions
    modification_functions = {
        "Custom": edge_addition_custom_v4,
        "Random": edge_addition_random,
        "Degree": edge_addition_degree,
        "PrefAtt": edge_addition_preferential_attachment,
        "Jaccard": edge_addition_jaccard,
        "TopK": edge_addition_topk,
    }

    # Track results for each method and k value
    combined_results = original_results.copy()

    for method_name, mod_func in modification_functions.items():
        if verbose:
            print(f"    Running evaluation for method: {method_name}")

        modified_graph = G.copy()

        # Get new target nodes for this k value
        new_targets = list(mod_func(G, seeds, k_values[-1]))
        prev_increment = 0

        for k in k_values:
            if verbose:
                print(f"        Evaluating k = {k}")

            # Add only the new edges for the difference in target nodes
            new_nodes = new_targets[prev_increment : k + 1]

            # Update the increment
            prev_increment = k + 1

            # Add edges from seeds to new target nodes
            for seed in seeds:
                for target_node in new_nodes:
                    if not modified_graph.has_edge(seed, target_node):
                        in_degree = modified_graph.in_degree(target_node)
                        weight = 1 / in_degree if in_degree > 0 else 1
                        modified_graph.add_edge(seed, target_node, weight=weight)

            # Evaluate current state
            modified_results_dict = simulate_diffusion_ICM_vectorized(
                modified_graph, seeds, max_iter, verbose
            )

            # Create and append results
            step_results = create_metrics_df(modified_results_dict, method_name, k)
            combined_results = pd.concat(
                [combined_results, step_results], ignore_index=True
            )

            # Update graph information
            mod_graph_info = {
                "Metric": ["Number of Nodes", "Number of Edges"],
                "Graph Modification": [method_name, method_name],
                "K Value": [k, k],
                "Value": [
                    modified_graph.number_of_nodes(),
                    modified_graph.number_of_edges(),
                ],
            }
            graph_info_df = pd.concat(
                [graph_info_df, pd.DataFrame(mod_graph_info)], ignore_index=True
            )

    # Combine graph information and results
    final_results = pd.concat([graph_info_df, combined_results], ignore_index=True)

    # Pivot the DataFrame
    final_results_pivot = final_results.pivot_table(
        index=["Graph Modification", "K Value"], columns="Metric", values="Value"
    ).reset_index()

    return final_results_pivot


def evaluate_all_seeds_incremental(
    G, seed_functions_eval, k_values, max_iter, name="", verbose=True
):
    """
    Evaluate all seed functions with incremental k values.

    Args:
        G (nx.Graph): Input graph
        seed_functions_eval (Dict[str, List[int]]): Dictionary of seed functions
        k_values (List[int]): List of k values to evaluate incrementally
        max_iter (int): Maximum iterations for ICM
        name (str): Name prefix for saving results
        verbose (bool): Whether to print progress

    Returns:
        pd.DataFrame: Combined results for all seed functions
    """
    combined_results = []

    for seed_name, seed in seed_functions_eval.items():
        print(f"Running evaluation for seed function: {seed_name}")
        final_results = evaluate_graph_modifications_incremental(
            G, seed, k_values, max_iter, verbose
        )

        # Add seed function identifier
        final_results["Seed Function"] = seed_name

        # Save intermediate results
        final_results.to_csv(
            f"results/store/{name + '_' if name else ''}{seed_name}_incremental.csv",
            index=False,
        )

        combined_results.append(final_results)

    # Combine all results
    all_results_df = pd.concat(combined_results, ignore_index=True)

    # Define modification order
    modification_order = [
        "Original Graph",
        "PrefAtt",
        "Jaccard",
        "Degree",
        "TopK",
        "Random",
        "Custom",
    ]

    # Set up categorical ordering
    all_results_df["Graph Modification"] = pd.Categorical(
        all_results_df["Graph Modification"],
        categories=modification_order,
        ordered=True,
    )

    # Sort results
    all_results_df = all_results_df.sort_values(
        by=["Seed Function", "Graph Modification", "K Value"]
    )

    # Reorder columns
    base_columns = [
        "Seed Function",
        "Graph Modification",
        "K Value",
        "Avg Activated Nodes",
        "Activated Nodes Std Dev",
        "Avg Color Activation Count",
        "Color Activation Count Std Dev",
        "Number of Nodes",
        "Number of Edges",
    ]

    # Get and sort color-related columns
    color_avg_columns = sorted(
        [
            col
            for col in all_results_df.columns
            if col.startswith("Avg Activated Nodes, Color (")
        ]
    )
    color_std_columns = sorted(
        [
            col
            for col in all_results_df.columns
            if col.startswith("Activated Nodes Std Dev, Color (")
        ]
    )

    # Interleave avg and std columns
    ordered_columns = []
    for avg_col in color_avg_columns:
        std_col = avg_col.replace("Avg Activated Nodes", "Activated Nodes Std Dev")
        ordered_columns.append(avg_col)
        if std_col in color_std_columns:
            ordered_columns.append(std_col)

    # Set final column order
    final_column_order = (
        base_columns[:3]  # Seed Function, Graph Modification, K Value
        + base_columns[3:5]  # Avg and Std Dev Activated Nodes
        + ordered_columns  # Color-specific columns
        + base_columns[5:]  # Remaining base columns
    )

    # Reorder columns and reset index
    all_results_df = all_results_df[final_column_order].reset_index(drop=True)

    return all_results_df
