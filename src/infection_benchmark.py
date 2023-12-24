"""
Code from https://github.com/m30m/gnn-explainability
"""

import random
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data


# generate an infection benchmark dataset
def create_infection_dataset(
    num_nodes: int,
    infected_portion: float,
    edge_prob: float,
    max_dist: int,
    seed: int = 0,
) -> Data:
    """
    Create a ER graph with 1000 nodes.
    - 50 nodes are infected
    - unique_solution_nodes: nodes that have a unique shortest path from the source
    - unique_solution_explanations: the unique shortest path from the source
    - labels: the distance from the source
    - features: 2 features, 1 if the node is infected, 0 otherwise
    - infected_portion: the portion of infected nodes
    """

    g = nx.erdos_renyi_graph(num_nodes, edge_prob, directed=True, seed=seed)
    num_infected_nodes = int(num_nodes * infected_portion)
    random.seed(seed)
    infected_nodes = random.sample(
        g.nodes(), num_infected_nodes
    )  # Sample 50 nodes from the graph
    g.add_node("X")  # dummy node for easier computation, will be removed in the end
    for u in infected_nodes:
        g.add_edge("X", u)  # Add an edge from the dummy node to the infected nodes

    # Compute the shortest path length from the dummy node to all other nodes
    shortest_path_length = nx.single_source_shortest_path_length(g, "X")
    unique_solution_nodes = []
    unique_solution_explanations = []
    labels = []
    infected = []  # 0 if infected, 1 otherwise
    features = np.zeros((num_nodes, 2))
    for i in range(num_nodes):
        if i == "X":
            continue
        length = shortest_path_length.get(i, 100) - 1  # 100 is inf distance
        labels.append(min(max_dist + 1, length))
        col = 0 if i in infected_nodes else 1
        features[i, col] = 1
        infected.append(col)
        if 0 < length <= max_dist:
            path_iterator = iter(nx.all_shortest_paths(g, "X", i))
            unique_shortest_path = next(path_iterator)
            if next(path_iterator, 0) != 0:
                continue
            unique_shortest_path.pop(0)  # pop 'X' node
            if len(unique_shortest_path) == 0:
                continue
            unique_solution_explanations.append(unique_shortest_path)
            unique_solution_nodes.append(i)
    g.remove_node("X")  # remove the dummy node
    data = from_networkx(g)
    data.x = torch.tensor(features, dtype=torch.float)
    data.y = torch.tensor(labels)
    data.unique_solution_nodes = unique_solution_nodes
    data.unique_solution_explanations = unique_solution_explanations
    data.num_classes = 1 + max_dist + 1
    data.infected = torch.tensor(infected)
    return data
