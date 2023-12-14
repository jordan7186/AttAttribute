from attention_analysis_utils import (
    get_computation_graph,
    get_nodes_per_level_from_comp_graph_full,
    reindex_nodes_per_level,
    translate_comp_graph,
    get_att_dict_per_layer,
    get_node_positions,
    get_edge_weights_dict,
    get_nodes_per_level_from_comp_graph,
    get_attention_raw_dict,
    process_attention_dict,
)
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils.convert import to_networkx
from torch import Tensor


def visualize_comp_graph_with_attention(data, att_dict, num_layers, target_idx):
    comp_graph = get_computation_graph(
        edge_index=data.edge_index, k=num_layers, target_idx=target_idx
    )
    (
        nodes_per_level_original,
        num_nodes_per_level,
        true_node_label,
    ) = get_nodes_per_level_from_comp_graph_full(comp_graph=comp_graph)
    nodes_per_level_new = reindex_nodes_per_level(
        nodes_per_level_original, num_nodes_per_level
    )
    comp_graph_new = translate_comp_graph(
        comp_graph=comp_graph,
        nodes_per_level_new=nodes_per_level_new,
        nodes_per_level_original=nodes_per_level_original,
    )
    layer_att_dict = get_att_dict_per_layer(
        comp_graph=comp_graph, comp_graph_new=comp_graph_new, att_dict=att_dict
    )
    visualize_computation_graph(
        layer_att_dict=layer_att_dict,
        nodes_per_level_new=nodes_per_level_new,
        true_node_label=true_node_label,
    )


# Let's turn the process of drawing the local graph with the ground truth path into a function
def draw_local_comp_graph_with_ground_truth_path(
    data: Data, hops: int, target_idx: int, ground_truth: bool = True
) -> None:
    # First assert that the target index does have a unique ground truth path
    assert (
        target_idx in data.unique_solution_nodes
    ), "Target index does not have a unique ground truth path"
    # Get the local k hop subgraph
    subgraph_nodes, _, _, inv = k_hop_subgraph(
        node_idx=target_idx,
        num_hops=hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
    )
    # Convert nodes and edges to lists
    subgraph_nodes = subgraph_nodes.tolist()
    subgraph_edges = data.edge_index[:, inv].tolist()
    # Transform subgraph_edges to a list of tuples
    subgraph_edges_tup = [
        (subgraph_edges[0][i], subgraph_edges[1][i])
        for i in range(len(subgraph_edges[0]))
    ]
    # Create a directed graph
    G = nx.DiGraph()
    G.add_edges_from(subgraph_edges_tup)
    if ground_truth:
        # Get index of target node in data.unique_solution_nodes
        target_idx_in_unique_solution_nodes = data.unique_solution_nodes.index(
            target_idx
        )
        # Get the ground truth path for target node
        ground_truth_path = data.unique_solution_explanations[
            target_idx_in_unique_solution_nodes
        ]
        # Convert the ground truth path to a list of tuples
        ground_truth_path_tup = [
            (ground_truth_path[i], ground_truth_path[i + 1])
            for i in range(len(ground_truth_path) - 1)
        ]
    # Draw the graph with subgraph_nodes as node labels
    # Highlight the path from ground_truth_path_tup with red edges
    # Also highlight the target node with a different color
    plt.figure(figsize=(6, 6), dpi=120)
    pos = nx.spring_layout(G, seed=0)
    nx.draw(
        G,
        pos=pos,
        node_color="#3bbcd9",
        node_size=1000,
        font_size=20,
        width=2,
        edgecolors="black",
        linewidths=2,
        edge_color="black",
        arrowstyle="-|>",
        labels={node: node for node in subgraph_nodes},
        with_labels=True,
        font_weight="bold",
    )
    if ground_truth:
        nx.draw_networkx_edges(
            G,
            pos=pos,
            edgelist=ground_truth_path_tup,
            edge_color="red",
            width=2,
            arrows=True,
            arrowsize=20,
            node_size=1000,
        )
    nx.draw_networkx_nodes(
        G, pos=pos, nodelist=[target_idx], node_color="red", node_size=1000
    )
    plt.show()


def visualize_computation_graph(
    layer_att_dict,
    nodes_per_level_new,
    true_node_label,
    with_labels=True,
    arrowsize=10,
    arrowstyle="-|>",
):
    # Visualize the computation graph with the attention weights, pre-defined node positions
    comp_G = nx.DiGraph()
    for layer, att_dict in layer_att_dict.items():
        for edge, att in att_dict.items():
            comp_G.add_edge(edge[0], edge[1], weight=att)

    # Get the edge weights
    edge_weights = [comp_G[u][v]["weight"] for u, v in comp_G.edges()]

    # Get the node positions
    node_pos = get_node_positions(nodes_per_level_new=nodes_per_level_new)

    # Draw the graph with the node positions and the edge weights
    # Define the width and height of the figure adaptively with the number of nodes
    figsize = (int(len(nodes_per_level_new[-1]) * 1), len(nodes_per_level_new) * 1.5)
    fig = plt.figure(figsize=figsize, dpi=120)
    # Set axis in the figure
    ax = fig.add_axes([0, 0, 1, 1])

    # Set the colormap
    cmap = plt.cm.coolwarm
    axfig = nx.draw(
        comp_G,
        pos=node_pos,
        node_color="#3bbcd9",
        node_size=1000,
        font_size=20,
        edgecolors="black",
        linewidths=2,
        edge_color=edge_weights,
        width=2,
        edge_vmin=0,
        edge_vmax=1,
        edge_cmap=cmap,
        arrowstyle=arrowstyle,
        arrowsize=arrowsize,
        labels=true_node_label,
        with_labels=with_labels,
        connectionstyle="angle3",
        font_weight="bold",
        ax=ax,
    )
    # Also show the colormap as legend on the side
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    plt.colorbar(sm, ax=ax)
    plt.show()


def draw_infection_data_original_labels(data, ax=None, figsize=(16, 16), dpi=200):
    # Get the number of classes for data
    num_classes = data.y.max().item() + 1
    # Set the size of the figure with dpi=200
    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Create a new axis if ax is None
    if ax is None:
        ax = fig.add_subplot(111)
    # Convert the data to networkx graph
    g = to_networkx(data)
    # Create a colormap for the labels
    cmap = plt.get_cmap("autumn", data.num_classes)
    # Create a list of colors for the nodes
    # where its color is determined by the label.
    node_colors = [cmap(data.y[i].item() / num_classes) for i in range(data.num_nodes)]
    # unique_solution_explanations = {
    #     node: f"{data.y[node].item()}"
    #     for  node in data.unique_solution_nodes
    # }
    # Draw the graph with node outline as black
    nx.draw(
        g,
        pos=nx.kamada_kawai_layout(g),
        node_color=node_colors,
        edgecolors="black",
        ax=ax,
        # labels=unique_solution_explanations,
        with_labels=True,
        arrows=True,
    )
    # Draw the legend
    # Create a list of handles for the legend
    handles = [
        plt.Line2D(
            [],
            [],
            color=cmap(i / num_classes),
            marker="o",
            linestyle="",
            label=f"Length {i}",
        )
        for i in range(data.num_classes)
    ]
    # Explicitly mention infected nodes in legend
    handles[0].set_label(f"Infected nodes")
    # Replace the highest category by appending + symbol
    handles[-1].set_label(f"Length {data.num_classes - 1}+")

    # Make some adjustments
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=15,
    )
    # Remove the axis
    ax.axis("off")
    # Set the title, including the number of nodes, number of ground truth shortest paths
    # and the number of unique shortest paths
    ax.set_title(
        f"Number of nodes: {data.num_nodes}\n"
        f"Number of unique shortest paths: {len(data.unique_solution_nodes)}",
        fontsize=20,
    )
    # Show the plot
    plt.show()


def draw_infection_data(data, ax=None, figsize=(16, 16), dpi=200):
    # Get the number of classes for data
    num_classes = data.y.max().item() + 1
    # Set the size of the figure with dpi=200
    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Create a new axis if ax is None
    if ax is None:
        ax = fig.add_subplot(111)
    # Convert the data to networkx graph
    g = to_networkx(data)
    # Create a colormap for the labels
    cmap = plt.get_cmap("autumn", data.num_classes)
    # Create a list of colors for the nodes
    # where its color is determined by the label.
    node_colors = [cmap(data.y[i].item() / num_classes) for i in range(data.num_nodes)]
    unique_solution_explanations = {
        node: f"{data.y[node].item()}" for node in data.unique_solution_nodes
    }
    # Draw the graph with node outline as black
    nx.draw(
        g,
        pos=nx.kamada_kawai_layout(g),
        node_color=node_colors,
        edgecolors="black",
        ax=ax,
        labels=unique_solution_explanations,
        arrows=True,
    )
    # Draw the legend
    # Create a list of handles for the legend
    handles = [
        plt.Line2D(
            [],
            [],
            color=cmap(i / num_classes),
            marker="o",
            linestyle="",
            label=f"Length {i}",
        )
        for i in range(data.num_classes)
    ]
    # Explicitly mention infected nodes in legend
    handles[0].set_label(f"Infected nodes")
    # Replace the highest category by appending + symbol
    handles[-1].set_label(f"Length {data.num_classes - 1}+")

    # Make some adjustments
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=15,
    )
    # Remove the axis
    ax.axis("off")
    # Set the title, including the number of nodes, number of ground truth shortest paths
    # and the number of unique shortest paths
    ax.set_title(
        f"Number of nodes: {data.num_nodes}\n"
        f"Number of unique shortest paths: {len(data.unique_solution_nodes)}",
        fontsize=20,
    )
    # Show the plot
    plt.show()


"""
Prepare & draw the computation graph
We perform several things during the prep stage:
1. Re-define the node indices to a ascending order, starting from 0 (root / target node) and ending with the last node in the last layer.
2. Equip the edges with the attention weights (we will accept the output of get_attention_weights_dict as input).
3. Equip the nodes with the node positions in the figure.
"""


# Previously prep_for_comp_graph_vis
def vis_comp_graph(comp_graph: Dict, att_tuple: Tuple, subset: Tensor) -> None:
    # First, get the nodes per level from comp_graph
    (
        nodes_per_level_original,
        num_nodes_per_level,
        true_node_label,
    ) = get_nodes_per_level_from_comp_graph(comp_graph, subset)

    # Second, reindex the nodes_per_level to a ascending order, starting from 0 (root / target node) and ending with the last node in the last layer.
    nodes_per_level_new = reindex_nodes_per_level(
        nodes_per_level_original, num_nodes_per_level
    )

    # Third, get the node positions in the figure
    node_pos_new = get_node_positions(nodes_per_level_new)

    # Fourth, reindex the comp_graph
    new_comp_graph = translate_comp_graph(
        comp_graph, nodes_per_level_new, nodes_per_level_original
    )

    # Fifth, equip the edges with the attention weights
    new_att_dict = get_edge_weights_dict(comp_graph, new_comp_graph, att_tuple)

    # Now, we can visualize the graph. First, make a networkx graph, then use matplotlib to visualize it.
    comp_G = nx.DiGraph()
    comp_G.add_edges_from(new_att_dict)

    for u, v, d in comp_G.edges(data=True):
        d["weight"] = new_att_dict[(u, v)]

    edges, weights = zip(*nx.get_edge_attributes(comp_G, "weight").items())
    plt.figure(
        figsize=(len(nodes_per_level_new[-1]) + 2, 2 * len(nodes_per_level_new)),
        dpi=150,
    )
    nx.draw(
        comp_G,
        pos=node_pos_new,
        labels=true_node_label,
        with_labels=True,
        node_size=2000,
        font_size=30,
        font_color="white",
        edge_cmap=plt.cm.Blues,
        width=2,
        edge_color=weights,
        edgelist=edges,
    )
    plt.show()


def visualizer_automatic(data: Data, model, num_layers: int, target_idx: int):
    att_dict_raw = get_attention_raw_dict(model, data)
    att_dict = process_attention_dict(att_dict_raw)

    comp_graph = get_computation_graph(
        edge_index=data.edge_index, k=num_layers, target_idx=target_idx
    )
    (
        nodes_per_level_original,
        num_nodes_per_level,
        true_node_label,
    ) = get_nodes_per_level_from_comp_graph_full(comp_graph=comp_graph)
    nodes_per_level_new = reindex_nodes_per_level(
        nodes_per_level_original, num_nodes_per_level
    )
    comp_graph_new = translate_comp_graph(
        comp_graph=comp_graph,
        nodes_per_level_new=nodes_per_level_new,
        nodes_per_level_original=nodes_per_level_original,
    )
    layer_att_dict = get_att_dict_per_layer(
        comp_graph=comp_graph, comp_graph_new=comp_graph_new, att_dict=att_dict
    )
    visualize_computation_graph(
        layer_att_dict=layer_att_dict,
        nodes_per_level_new=nodes_per_level_new,
        true_node_label=true_node_label,
    )
