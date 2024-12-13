import copy
import multiprocessing as mp
from typing import Callable, Dict, List, Set

import networkx as nx
import numpy as np
from tqdm import tqdm


class ParallelizedSubmodularInfluenceMaximizer:
    def __init__(self, graph: nx.DiGraph):
        """
        Initialize the Submodular Influence Maximization algorithm with caching and parallel simulation.

        :param graph: NetworkX directed graph representing the social network
        """
        self.graph = graph
        self.color_attribute = "color"

        # Cache to store influence simulation results
        self._influence_cache: Dict[frozenset, float] = {}

    def _single_simulation(
        self, seed_nodes: List[str], cross_color_bias: bool = True
    ) -> int:
        """
        Single Independent Cascade Model simulation run.

        :param seed_nodes: Initial seed nodes
        :param cross_color_bias: Enhance cross-color activation probability
        :return: Number of nodes activated in this simulation
        """
        sim_graph = copy.deepcopy(self.graph)
        active_nodes = set(seed_nodes)
        newly_active = set(seed_nodes)
        cross_color_activations = 0

        while newly_active:
            next_newly_active = set()

            for node in newly_active:
                for neighbor in sim_graph.neighbors(node):
                    if neighbor not in active_nodes:
                        # Base activation probability
                        weight = sim_graph[node][neighbor].get("weight", 0.5)

                        # Cross-color bias for reducing polarization
                        if cross_color_bias:
                            node_color = self.graph.nodes[node].get(
                                self.color_attribute
                            )
                            neighbor_color = self.graph.nodes[neighbor].get(
                                self.color_attribute
                            )

                            if node_color != neighbor_color:
                                # Increase activation probability for cross-color interactions
                                weight *= 1.5
                                cross_color_activations += 1

                        if np.random.random() < weight:
                            next_newly_active.add(neighbor)
                            active_nodes.add(neighbor)

            newly_active = next_newly_active

        return len(active_nodes)

    def _cached_parallel_influence_simulation(
        self,
        seed_nodes: List[str],
        num_simulations: int = 500,
        cross_color_bias: bool = True,
        num_processes: int = None,
        progress_bar: bool = False,
    ) -> float:
        """
        Cached version of parallel independent cascade simulation.

        :param seed_nodes: Initial seed nodes
        :param num_simulations: Number of simulation runs
        :param cross_color_bias: Enhance cross-color activation probability
        :param num_processes: Number of parallel processes
        :param progress_bar: Show progress of simulations
        :return: Average number of nodes activated
        """
        # Convert seed nodes to a frozenset for hashability
        seed_set = frozenset(seed_nodes)

        # Check if result is already cached
        if seed_set in self._influence_cache:
            return self._influence_cache[seed_set]

        # If not cached, run the parallel simulation
        result = self.parallel_independent_cascade_simulation(
            list(seed_set),
            num_simulations,
            cross_color_bias,
            num_processes,
            progress_bar,
        )

        # Cache the result
        self._influence_cache[seed_set] = result
        return result

    def parallel_independent_cascade_simulation(
        self,
        seed_nodes: List[str],
        num_simulations: int = 500,
        cross_color_bias: bool = True,
        num_processes: int = None,
        progress_bar: bool = True,
    ) -> float:
        """
        Parallelized Independent Cascade Model simulation.

        :param seed_nodes: Initial seed nodes
        :param num_simulations: Number of simulation runs
        :param cross_color_bias: Enhance cross-color activation probability
        :param num_processes: Number of parallel processes (None uses all available cores)
        :param progress_bar: Show progress of simulations
        :return: Average number of nodes activated
        """
        # Use all available cores if not specified
        if num_processes is None:
            num_processes = mp.cpu_count()

        # Prepare arguments for parallel processing
        simulation_args = [
            (seed_nodes, cross_color_bias) for _ in range(num_simulations)
        ]

        # Use multiprocessing Pool for parallel execution
        with mp.Pool(processes=num_processes) as pool:
            # Use tqdm for progress tracking if enabled
            if progress_bar:
                activations = list(
                    tqdm(
                        pool.starmap(self._single_simulation, simulation_args),
                        total=num_simulations,
                        desc="Parallel ICM Simulations",
                    )
                )
            else:
                activations = pool.starmap(self._single_simulation, simulation_args)

        # Calculate and return average activations
        return np.mean(activations)

    def monotone_submodular_marginal_gain(
        self,
        seed_set: Set[str],
        candidate_node: str,
        influence_function: Callable[[Set[str]], float],
    ) -> float:
        """
        Calculate the monotone submodular marginal gain of adding a node.
        Uses cached influence simulation to improve performance.

        :param seed_set: Current set of seed nodes
        :param candidate_node: Node being considered for addition
        :param influence_function: Function to measure influence spread
        :return: Marginal gain of adding the candidate node
        """
        # Compute influence before and after adding the node
        influence_before = influence_function(seed_set)
        influence_after = influence_function(seed_set.union({candidate_node}))

        return influence_after - influence_before

    def calculate_graph_density(self) -> float:
        """
        Calculate the edge density of the graph.

        :return: Graph edge density
        """
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()

        # Maximum possible edges in a directed graph
        max_edges = num_nodes * (num_nodes - 1)

        return num_edges / max_edges if max_edges > 0 else 0

    def dynamic_lazy_greedy_threshold(
        self, k: int, node_count: int, graph_density: float
    ) -> int:
        """
        Dynamically calculate an appropriate threshold for lazy greedy selection.

        :param k: Number of seed nodes to select
        :param node_count: Total number of nodes in the graph
        :param graph_density: Edge density of the graph
        :return: Calculated threshold value
        """

        # Adjust threshold based on graph density
        if graph_density < 0.01:  # Sparse graph
            threshold = max(5, int(node_count * 0.05))
        elif graph_density < 0.1:  # Moderately sparse
            threshold = max(5, int(node_count * 0.1))
        elif graph_density < 0.5:  # Dense graph
            threshold = max(5, int(node_count * 0.2))
        else:  # Very dense graph
            threshold = max(5, int(node_count * 0.3))

        # Further adjust based on k
        if k < 3:
            # For small k, use a smaller threshold
            threshold = min(threshold, max(5, int(node_count * 0.05)))
        elif k > 10:
            # For larger k, use a larger threshold
            threshold = min(threshold * 2, node_count)

        return int(threshold)

    def lazy_greedy_selection(
        self,
        k: int = 5,
        initial_seeds: Set[str] = None,
        adaptive_threshold: bool = True,
        progress_bar: bool = True,
        num_processes: int = None,
    ) -> List[str]:
        """
        Enhanced Lazy Greedy algorithm with optional initial seed set and dynamic threshold.

        :param k: Total number of seed nodes to select
        :param initial_seeds: Optional set of seed nodes to start with
        :param adaptive_threshold: Use dynamic threshold calculation
        :param progress_bar: Show progress of seed selection
        :param num_processes: Number of parallel processes for influence simulation
        :return: List of selected seed nodes
        """

        def influence_function(seed_set):
            """Wrapper for parallel influence simulation."""
            return self._cached_parallel_influence_simulation(
                list(seed_set), num_processes=num_processes, progress_bar=False
            )

        # Initialize seed set or use empty set
        selected_seeds = set(initial_seeds) if initial_seeds is not None else set()

        # Calculate graph characteristics
        node_count = self.graph.number_of_nodes()
        graph_density = self.calculate_graph_density()

        # Determine threshold
        if adaptive_threshold:
            threshold = self.dynamic_lazy_greedy_threshold(k, node_count, graph_density)
        else:
            # Fallback to original fixed threshold
            threshold = max(5, int(node_count * 0.1))

        # Get candidate nodes, excluding already selected seeds
        candidate_nodes = [
            node for node in self.graph.nodes() if node not in selected_seeds
        ]

        # Precompute initial marginal gains with progress bar
        marginal_gains = {}
        precompute_progress = tqdm(
            candidate_nodes,
            desc="Precomputing Marginal Gains",
            disable=not progress_bar,
        )
        for node in precompute_progress:
            marginal_gains[node] = self.monotone_submodular_marginal_gain(
                selected_seeds, node, influence_function
            )

        # Sort nodes by initial marginal gain
        sorted_nodes = sorted(
            candidate_nodes, key=lambda x: marginal_gains[x], reverse=True
        )

        # Use tqdm for progress tracking
        selection_range = tqdm(
            range(k),
            desc="Lazy Greedy Seed Selection",
            disable=not progress_bar,
        )
        for _ in selection_range:
            best_node = None
            best_marginal_gain = float("-inf")

            # Limit the search space based on lazy evaluation
            candidates = sorted_nodes[:threshold]

            # Inner progress bar for candidate evaluation
            candidate_progress = tqdm(
                candidates, desc="Evaluating Candidates", disable=not progress_bar
            )

            for node in candidate_progress:
                if node not in selected_seeds:
                    marginal_gain = self.monotone_submodular_marginal_gain(
                        selected_seeds, node, influence_function
                    )

                    # Update progress description with current best gain
                    if progress_bar:
                        candidate_progress.set_postfix(
                            {
                                "Best Gain": f"{best_marginal_gain:.4f}",
                                "Current Node": node,
                            }
                        )

                    if marginal_gain > best_marginal_gain:
                        best_marginal_gain = marginal_gain
                        best_node = node

            if best_node:
                selected_seeds.add(best_node)
                sorted_nodes.remove(best_node)

                # Update outer progress bar description
                if progress_bar:
                    selection_range.set_postfix(
                        {
                            "Selected": list(selected_seeds),
                            "Last Gain": f"{best_marginal_gain:.4f}",
                        }
                    )
        return list(selected_seeds)

    def clear_influence_cache(self):
        """
        Clear the influence simulation cache.
        Useful when graph structure or node attributes change.
        """
        self._influence_cache.clear()

    def get_cached_influences(self) -> Dict[frozenset, float]:
        """
        Retrieve the current cached influence values.

        :return: Dictionary of cached influence simulations
        """
        return dict(self._influence_cache)
