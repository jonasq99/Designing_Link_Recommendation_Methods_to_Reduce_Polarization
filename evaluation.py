from multiprocessing import Pool, cpu_count
from typing import Dict, List, Set

import pandas as pd

from src.edge_addition import (
    edge_addition_custom,
    edge_addition_custom_v2,
    edge_addition_degree,
    edge_addition_jaccard,
    edge_addition_kkt,
    edge_addition_preferential_attachment,
    edge_addition_random,
    edge_addition_topk,
)
from src.icm_diffusion_optimized import simulate_diffusion_ICM


def evaluate_graph_modifications(G, seeds, k, max_iter, budget):
    def create_metrics_df(results_dict, graph_name):
        avg_metrics = [
            {
                "Metric": "Avg Activated Nodes",
                "Graph Modification": graph_name,
                "Value": round(results_dict["avg_activated_nodes"], 3),
            },
            {
                "Metric": "Avg Color Activation Count",
                "Graph Modification": graph_name,
                "Value": round(results_dict["avg_color_activation_count"], 3),
            },
        ]
        std_metrics = [
            {
                "Metric": "Activated Nodes Std Dev",
                "Graph Modification": graph_name,
                "Value": round(results_dict["std_dev_activated_nodes"], 3),
            },
            {
                "Metric": "Color Activation Count Std Dev",
                "Graph Modification": graph_name,
                "Value": round(results_dict["std_dev_color_activation_count"], 3),
            },
        ]

        for color, avg_count in results_dict["avg_color_activated_nodes"].items():
            avg_metrics.append(
                {
                    "Metric": f"Avg Activated Nodes, Color ({color})",
                    "Graph Modification": graph_name,
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
                    "Value": round(std_dev_count, 3),
                }
            )

        return pd.DataFrame(avg_metrics + std_metrics)

    # Simulate diffusion on the original graph
    original_results_dict = simulate_diffusion_ICM(G, seeds, max_iter)
    original_results = create_metrics_df(original_results_dict, "Original Graph")

    # Get graph information
    graph_info = {
        "Metric": ["Number of Nodes", "Number of Edges"],
        "Graph Modification": ["Original Graph", "Original Graph"],
        "Value": [G.number_of_nodes(), G.number_of_edges()],
    }
    graph_info_df = pd.DataFrame(graph_info)

    # Define modification functions
    modification_functions = {
        "Degree": edge_addition_degree,
        "PrefAtt": edge_addition_preferential_attachment,
        "Jaccard": edge_addition_jaccard,
        "TopK": edge_addition_topk,
        "KKT": edge_addition_kkt,
        "Random": edge_addition_random,
        "Custom": edge_addition_custom,
        "Custom V2": edge_addition_custom_v2,
    }

    # Evaluate each graph modification
    combined_results = original_results.copy()
    for method_name, mod_func in modification_functions.items():
        modified_graph = mod_func(G, seeds, k, budget)
        modified_results_dict = simulate_diffusion_ICM(modified_graph, seeds, max_iter)
        adapted_results = create_metrics_df(modified_results_dict, method_name)
        combined_results = pd.concat(
            [combined_results, adapted_results], ignore_index=True
        )

        # Update graph information
        mod_graph_info = {
            "Metric": ["Number of Nodes", "Number of Edges"],
            "Graph Modification": [method_name, method_name],
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

    # Pivot the DataFrame to get the desired format
    final_results_pivot = final_results.pivot(
        index="Graph Modification", columns="Metric", values="Value"
    ).reset_index()

    return final_results_pivot


def evaluate_all_seeds(G, seed_functions, seed_size, k, max_iter, budget):
    combined_results = []
    all_results_df = pd.DataFrame()

    for seed_name, seed_func in seed_functions.items():
        print(f"Running evaluation for seed function: {seed_name}")
        seed = seed_func(G, seed_size)
        final_results = evaluate_graph_modifications(G, seed, k, max_iter, budget)

        # Add a new column to identify the seed function used
        final_results["Seed Function"] = seed_name

        # Reset the index and ensure "Graph Modification" is the first column, if it exists
        final_results = final_results.reset_index(drop=True)

        # Move "Seed Function" to be the first column
        columns = ["Seed Function"] + [
            col for col in final_results.columns if col != "Seed Function"
        ]
        final_results = final_results[columns]

        # Append results to the list
        combined_results.append(final_results)

    # Combine all results into a single DataFrame
    all_results_df = pd.concat(combined_results, ignore_index=True)

    # Define the desired order for the "Graph Modification" column
    modification_order = [
        "Original Graph",
        "PrefAtt",
        "Jaccard",
        "Degree",
        "TopK",
        "KKT",
        "Random",
        "Custom",
        "Custom V2",
    ]
    all_results_df["Graph Modification"] = pd.Categorical(
        all_results_df["Graph Modification"],
        categories=modification_order,
        ordered=True,
    )
    all_results_df = all_results_df.sort_values(
        by=["Seed Function", "Graph Modification"]
    )

    # Reorder columns based on your desired structure
    base_columns = [
        "Seed Function",
        "Graph Modification",
        "Avg Activated Nodes",
        "Activated Nodes Std Dev",
        "Avg Color Activation Count",
        "Color Activation Count Std Dev",
        "Number of Nodes",
        "Number of Edges",
    ]

    # Get all specific color-related columns
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

    # Create the final column order by interleaving avg and std columns
    ordered_columns = []
    for avg_col in color_avg_columns:
        std_col = avg_col.replace("Avg Activated Nodes", "Activated Nodes Std Dev")
        ordered_columns.append(avg_col)
        if std_col in color_std_columns:
            ordered_columns.append(std_col)

    # Final column order
    final_column_order = (
        base_columns[:2] + base_columns[2:4] + ordered_columns + base_columns[4:]
    )

    # Reorder the DataFrame columns
    all_results_df = all_results_df[final_column_order]

    # Reset the index if needed
    all_results_df = all_results_df.reset_index(drop=True)

    return all_results_df
