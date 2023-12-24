# Tools for analyzing attention weights
import matplotlib.pyplot as plt
import torch
import networkx as nx
from torch_geometric.utils import remove_self_loops, add_self_loops, get_num_hops
from torch_geometric.data import Data
from typing import List, Optional, Tuple, Dict
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
import matplotlib.pyplot as plt
import torch
import networkx as nx
from torch_geometric.utils import (
    remove_self_loops,
    add_self_loops,
    get_num_hops,
    k_hop_subgraph,
)


def k_hop_subgraph_modified(
    node_idx: int,
    num_hops: int,
    edge_index: Tensor,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[Tensor]]:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_index = add_self_loops(remove_self_loops(edge_index)[0], num_nodes=num_nodes)[
        0
    ]
    col, row = edge_index

    node_mask: Tensor = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask: Tensor = row.new_empty(row.size(0), dtype=torch.bool)

    node_idx_Tensor: Tensor = torch.tensor([node_idx], device=row.device).flatten()
    subsets: List[Tensor] = [node_idx_Tensor]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    edge_index = edge_index[:, edge_mask]

    node_idx_Tensor = row.new_full((num_nodes,), -1)
    node_idx_Tensor[subset] = torch.arange(subset.size(0), device=row.device)
    edge_index = node_idx_Tensor[edge_index]

    new_levels = []
    for level in subsets:
        new_levels.append(node_idx_Tensor[level])

    return subset, edge_index, inv, edge_mask, new_levels


# Before acquiring the full computation graph, get the attention weights of the target node
@torch.no_grad()
def get_attention_weights_karate(
    k_hop_edge_index: Tensor, subset: Tensor, model, data: Data
) -> Tuple:
    k_hop_edge_index = k_hop_edge_index.to("cuda")
    k_hop_edge_index = remove_self_loops(k_hop_edge_index)[0]
    new_x = data.x[subset].to("cuda")
    model(new_x, k_hop_edge_index, return_att=True)
    return model.att


# Deprecated
# @torch.no_grad()
# def get_attention_weights(
#     k_hop_edge_index: Tensor, subset: Tensor, model, data: Data
# ) -> Tuple:
#     k_hop_edge_index = k_hop_edge_index.to("cuda")
#     k_hop_edge_index = remove_self_loops(k_hop_edge_index)[0]
#     new_x = data.x[subset].to("cuda")
#     model(
#         new_x, k_hop_edge_index, return_att=True
#     )  # just run the model once to get the attention weights
#     num_layers = get_num_hops(model)
#     if num_layers == 2:
#         att = (model.att1, model.att2)
#     elif num_layers == 3:
#         att = (model.att1, model.att2, model.att3)
#     else:
#         raise ValueError("The model is not GAT_L2 or GAT_L3")
#     return att


def get_computation_graph(
    edge_index: Tensor, k: int, target_idx: int
) -> dict[int, Tensor]:
    """
    Returns the computation graph of the k-hop subgraph.
    We are expecting the inputs to be the ORIGINAL DATA INDICES.
    The first row of edge_index is the source, and the second row is the target.
    """
    edge_index = add_self_loops(remove_self_loops(edge_index)[0])[0]
    # Initialize the computation graph dictionary
    comp_graph = {}
    # Start by getting the neighbors of the target node
    curr_edge_index = edge_index[:, edge_index[1] == target_idx]
    comp_graph[0] = curr_edge_index.cpu()
    # Repeat the process for the rest of the layers recursively.
    for layer in range(1, k):
        # Get the neighbors of the current layer
        new_targets = curr_edge_index[0, :]
        # Initialize the final edge index tensor
        final_edge_index = []
        for target in new_targets:
            # Get the neighbors of the new targets
            temp_edge_index = edge_index[:, edge_index[1] == target]
            final_edge_index.append(temp_edge_index)
        final_edge_index = torch.cat(final_edge_index, dim=1).cpu()
        # Add the new edge_index to the computation graph dictionary
        comp_graph[layer] = final_edge_index
        curr_edge_index = final_edge_index

    return comp_graph


def get_computation_graph_local(
    edge_index: Tensor, k: int, shifted_target_idx: int
) -> dict[int, Tensor]:
    """
    Returns the computation graph of the k-hop subgraph.
    We are expecting the inputs to be the output of k_hop_subgraph_modified.
    Therefore, the target node index is always 0.
    The first row of edge_index is the source, and the second row is the target.
    """
    edge_index = add_self_loops(remove_self_loops(edge_index)[0])[0]
    # Initialize the computation graph dictionary
    comp_graph = {}
    # Start by getting the neighbors of the target node
    curr_edge_index = edge_index[:, edge_index[1] == shifted_target_idx]
    comp_graph[0] = curr_edge_index.cpu()
    # Repeat the process for the rest of the layers recursively.
    for layer in range(1, k):
        # Get the neighbors of the current layer
        new_targets = curr_edge_index[0, :]
        # Initialize the final edge index tensor
        final_edge_index = []
        for target in new_targets:
            # Get the neighbors of the new targets
            temp_edge_index = edge_index[:, edge_index[1] == target]
            final_edge_index.append(temp_edge_index)
        final_edge_index = torch.cat(final_edge_index, dim=1).cpu()
        # Add the new edge_index to the computation graph dictionary
        comp_graph[layer] = final_edge_index
        curr_edge_index = final_edge_index

    return comp_graph


# Draw the attention weights of the target node for a given layer
def draw_attention_weights(att_tuple: Tuple, layer: int) -> None:
    curr_edge_index = att_tuple[layer][0]
    edge_with_att = []
    for idx, att in enumerate(att_tuple[layer][1].flatten().cpu()):
        edge_with_att.append(
            (curr_edge_index[0][idx].item(), curr_edge_index[1][idx].item(), att.item())
        )
    # Create a temporary graph to draw the attention weights
    G_temp = nx.Graph()
    G_temp.add_weighted_edges_from(edge_with_att)
    # Draw the graph with weighted edges as the attention weights with color
    pos = nx.spring_layout(G_temp, seed=42)
    edges = G_temp.edges()
    weights = [G_temp[u][v]["weight"] for u, v in edges]
    plt.figure(figsize=(10, 10), dpi=100)
    nx.draw_networkx_nodes(G_temp, pos, node_color="tab:blue", node_size=200)
    nx.draw_networkx_labels(G_temp, pos)
    nx.draw_networkx_edges(
        G=G_temp,
        pos=pos,
        edgelist=edges,
        edge_color=weights,
        edge_cmap=plt.cm.Blues,
        width=2,
    )
    plt.show()


# Return the attention weights as a dictionary
def get_attention_weights_dict(att_tuple: Tuple, layer: int) -> Dict:
    curr_edge_index = att_tuple[layer][0]
    edge_with_att = []
    for idx, att in enumerate(att_tuple[layer][1].flatten().cpu()):
        edge_with_att.append(
            (
                (curr_edge_index[0][idx].item(), curr_edge_index[1][idx].item()),
                att.item(),
            )
        )
    return dict(edge_with_att)


# Deprecated
def get_attention_weights_dict_all(att_tuple: Tuple) -> Dict:
    att_dict = {}
    for layer in range(len(att_tuple)):
        att_dict[layer] = get_attention_weights_dict(att_tuple=att_tuple, layer=layer)
    return att_dict



def translate_comp_graph(comp_graph, nodes_per_level_new, nodes_per_level_original, degree):
    new_comp_graph = {}
    for key, value in comp_graph.items():
        curr_tgt = value[1]
        new_src = nodes_per_level_new[key + 1]
        new_tgt = torch.zeros_like(curr_tgt)

        curr_idx = 0
        degree_counter = 0
        for idx, node in enumerate(curr_tgt):
            # Check node corresponds to the nodes_per_level_original[key][curr_idx]
            candidate_node = nodes_per_level_original[key][curr_idx]
            degree_candidate_node = degree[candidate_node] + 1 # Add 1 to account for self-loop
            if node == candidate_node and degree_counter < degree_candidate_node:
                new_tgt[idx] = nodes_per_level_new[key][curr_idx]
                degree_counter += 1
            # If not, check if node corresponds to the nodes_per_level_original[key][curr_idx+1]
            else:
                curr_idx += 1
                degree_counter = 1 # Reset degree counter, but used up one degree
                new_tgt[idx] = nodes_per_level_new[key][curr_idx]

        new_comp_graph[key] = torch.stack([new_src, new_tgt], dim=0)
    return new_comp_graph


# Deprecated
def translate_comp_graph_deprecated(comp_graph, nodes_per_level_new, nodes_per_level_original):
    new_comp_graph = {}
    for key, value in comp_graph.items():
        curr_tgt = value[1]
        new_src = nodes_per_level_new[key + 1]
        new_tgt = torch.zeros_like(curr_tgt)

        curr_idx = 0
        for idx, node in enumerate(curr_tgt):
            # Fix bug?
            # Check node corresponds to the nodes_per_level_original[key][curr_idx]
            if node == nodes_per_level_original[key][curr_idx]:
                new_tgt[idx] = nodes_per_level_new[key][curr_idx]
            # If not, check if node corresponds to the nodes_per_level_original[key][curr_idx+1]
            else:
                curr_idx += 1
                new_tgt[idx] = nodes_per_level_new[key][curr_idx]

        new_comp_graph[key] = torch.stack([new_src, new_tgt], dim=0)
    return new_comp_graph


def get_node_positions(nodes_per_level_new):
    node_pos_new = {}
    # Define the y position of the nodes in each level
    y_pos = torch.linspace(1, -1, len(nodes_per_level_new))
    for idx, nodes in enumerate(nodes_per_level_new):
        # Define the x position of the nodes in each level
        x_pos = torch.linspace(-1, 1, len(nodes))
        for node_idx, node in enumerate(nodes):
            node_pos_new[node.item()] = (x_pos[node_idx].item(), y_pos[idx].item())
    node_pos_new[0] = (0, 1)
    return node_pos_new


def get_edge_weights_dict(comp_graph, new_comp_graph, att_tuple):
    new_att_dict = {}
    for layer in range(len(att_tuple)):
        curr_new_edge_index = new_comp_graph[layer]
        curr_edge_index = comp_graph[layer]
        curr_att_data = att_tuple[-1 - layer]

        # Convert curr_att_data to a dictionary
        curr_att_dict = {}
        for idx in range(curr_att_data[0].shape[1]):
            curr_edge = (
                curr_att_data[0][0][idx].item(),
                curr_att_data[0][1][idx].item(),
            )
            curr_att_dict[curr_edge] = curr_att_data[1][idx].item()

        num_edges = curr_edge_index.shape[1]

        for idx in range(num_edges):
            curr_edge = (curr_edge_index[0][idx].item(), curr_edge_index[1][idx].item())
            curr_att = curr_att_dict[curr_edge]
            curr_new_edge = (
                curr_new_edge_index[0][idx].item(),
                curr_new_edge_index[1][idx].item(),
            )
            new_att_dict[curr_new_edge] = curr_att
    return new_att_dict


def get_nodes_per_level_from_comp_graph(comp_graph, subset):
    # Get list of nodes for each level, in the original node index
    nodes_per_level_original = [
        comp_graph[0][1][0].cpu().view(1)
    ]  # Initialize the list with the target node
    for idx in range(len(comp_graph)):
        nodes_per_level_original.append(comp_graph[idx][0].cpu())
    # And get the number of nodes in each level
    num_nodes_per_level = torch.tensor(
        [len(node_tensor) for node_tensor in nodes_per_level_original]
    )
    # And the true node label (for visualization)
    nodes_per_level_original_concat = torch.cat(nodes_per_level_original, dim=0)
    true_node_label = dict(
        zip(
            range(len(nodes_per_level_original_concat)),
            subset[nodes_per_level_original_concat].tolist(),
        )
    )
    return nodes_per_level_original, num_nodes_per_level, true_node_label


def reindex_nodes_per_level(nodes_per_level_original, num_nodes_per_level):
    # Redefine the node indices
    nodes_per_level_new = []
    for idx, num_nodes in enumerate(num_nodes_per_level):
        nodes_per_level_new.append(
            torch.arange(
                (num_nodes_per_level[:idx]).sum(),
                (num_nodes_per_level[: idx + 1]).sum(),
            )
        )
    return nodes_per_level_new


@torch.no_grad()
def get_attention_raw_dict(model, data) -> Dict:
    num_layers = get_num_hops(model)
    model(data.x, data.edge_index, return_att=True)
    attention_dict = {}
    for layer in range(num_layers):
        attention_dict[layer] = model.att[layer]

    return attention_dict


def average_attention_heads(
    att: List[Tuple[Tensor, Tensor]]
) -> List[Tuple[Tensor, Tensor]]:
    att_new = []
    for layer in att:
        curr_att_weight = layer[1]
        avg_att_weight = curr_att_weight.mean(dim=1).view(-1, 1)
        att_new.append((layer[0], avg_att_weight))

    return att_new


@torch.no_grad()
def get_attention_raw_dict_multihead(model, data) -> Dict:
    num_layers = get_num_hops(model)
    model(data.x, data.edge_index, return_att=True)
    att = model.att
    att = average_attention_heads(att)
    attention_dict = {}
    for layer in range(num_layers):
        attention_dict[layer] = att[layer]

    return attention_dict


def process_attention_dict(att_dict: Dict) -> Dict:
    att_dict_processed = {}
    for layer, att in att_dict.items():
        curr_edge_index = att[0]
        curr_att_tensor = att[1]
        curr_att_dict = {}
        for idx in range(curr_edge_index.shape[1]):
            curr_edge = (curr_edge_index[0][idx].item(), curr_edge_index[1][idx].item())
            curr_att_dict[curr_edge] = curr_att_tensor[idx].item()
        att_dict_processed[layer] = curr_att_dict
    return att_dict_processed


def get_nodes_per_level_from_comp_graph_full(comp_graph):
    # Get list of nodes for each level, in the original node index
    nodes_per_level_original = [
        comp_graph[0][1][0].view(1)
    ]  # Initialize the list with the target node
    for idx in range(len(comp_graph)):
        nodes_per_level_original.append(comp_graph[idx][0])
    # And get the number of nodes in each level
    num_nodes_per_level = torch.tensor(
        [len(node_tensor) for node_tensor in nodes_per_level_original]
    )
    # And the true node label (for visualization)
    nodes_per_level_original_concat = torch.cat(nodes_per_level_original, dim=0)
    true_node_label = dict(
        zip(
            range(len(nodes_per_level_original_concat)),
            nodes_per_level_original_concat.tolist(),
        )
    )
    return nodes_per_level_original, num_nodes_per_level, true_node_label


def get_att_dict_per_layer(comp_graph, comp_graph_new, att_dict):
    layer_att_dict = {}
    for curr_layer in range(len(comp_graph)):
        curr_att_dict = att_dict[curr_layer]
        curr_edge_index = comp_graph[curr_layer]
        curr_edge_index_new = comp_graph_new[curr_layer]

        curr_layer_att_dict = {}
        for idx in range(curr_edge_index_new.shape[1]):
            curr_edge = (curr_edge_index[0][idx].item(), curr_edge_index[1][idx].item())
            curr_edge_new = (
                curr_edge_index_new[0][idx].item(),
                curr_edge_index_new[1][idx].item(),
            )
            curr_att = curr_att_dict[curr_edge]
            curr_layer_att_dict[curr_edge_new] = curr_att
        layer_att_dict[curr_layer] = curr_layer_att_dict
    return layer_att_dict


# First, we need to find the intermediate edges for each edge in target_edge_new_dict
# Aim at devising a function that returns the intermediate edges for a given edge
# Assuming we already know the depth of the edge in the comp_graph_new
def return_intermediate_edges(
    comp_graph_new: Dict[int, Tensor], edge: Tuple, depth: int
) -> List[Tuple]:
    root_idx = 0
    # If there are no intermediate edges, return [(-1, -1)]
    if edge[1] == root_idx:
        return [(-1, -1)]
    # If there are intermediate edges, return a list of intermediate edges
    intermediate_edges = []
    curr_edge = edge
    for curr_depth in range(depth - 1, -1, -1):
        find_idx = curr_edge[1]
        # Get the location where find_idx appear in comp_graph_new[curr_depth]
        find_idx_location = comp_graph_new[curr_depth][0].tolist().index(find_idx)
        curr_edge = tuple(comp_graph_new[curr_depth].t()[find_idx_location].tolist())
        intermediate_edges.append(curr_edge)

    return intermediate_edges



# Now let's make this into a function
# Despite the name, this function returns all two ATTATTTRIBUTE, ATTATTTRIBUTE_sim
def get_ATTATTTRIBUTE_edge(
    comp_graph: Dict[int, Tensor],
    comp_graph_new: Dict[int, Tensor],
    layer_att_dict: Dict[int, Dict[Tuple, float]],
    target_edge: Tuple[int, int],
    verbose: bool = False,
) -> Tuple[float, float]:
    # First, get location_dict
    location_dict = {}
    for depth in comp_graph.items():
        occurance_idx = []
        for idx, edge in enumerate(depth[1].t().tolist()):
            if edge == list(target_edge):
                occurance_idx.append(idx)
        location_dict[depth[0]] = occurance_idx

    # Second, get target_edge_new_dict
    target_edge_new_dict = {}
    for depth in location_dict.items():
        target_edge_new = []
        for idx in depth[1]:
            target_edge_new.append(tuple(comp_graph_new[depth[0]][:, idx].tolist()))
        target_edge_new_dict[depth[0]] = target_edge_new

    # Third, get target_edge_att_dict
    target_edge_att_dict = {}
    for depth in target_edge_new_dict.items():
        for edge in depth[1]:
            target_edge_att_dict[edge] = layer_att_dict[depth[0]][edge]

    if verbose:
        formatted_list = [
            float("%.4f" % item) for item in list(target_edge_att_dict.values())
        ]
        print(
            f"Naive attention for each occurance of edge {target_edge}: {formatted_list}"
        )
        print(
            f"ATTATTRIBUTE_sim for edge {target_edge}: {sum(target_edge_att_dict.values()):.4f}"
        )

    # Fourth, get target_edge_att_dict_new
    target_edge_att_dict_new = {}
    for depth, curr_edge_list in target_edge_new_dict.items():
        for curr_edge in curr_edge_list:
            # Find the intermediate edges for the current edge
            intermediate_edges = return_intermediate_edges(
                comp_graph_new, curr_edge, depth
            )
            # If intermediate_edges is [(-1, -1)], just return the original attention value
            if intermediate_edges == [(-1, -1)]:
                target_edge_att_dict_new[curr_edge] = target_edge_att_dict[curr_edge]
            # Otherwise, multiply the attention value of the edge with the attention values of the intermediate edges
            else:
                curr_att = target_edge_att_dict[curr_edge]
                curr_depth = depth - 1
                for intermediate_edge in intermediate_edges:
                    curr_att *= layer_att_dict[curr_depth][intermediate_edge]
                    curr_depth -= 1
                target_edge_att_dict_new[curr_edge] = curr_att

    if verbose:
        print(
            f"ATTATTTRIBUTE for edge {target_edge}: {sum(target_edge_att_dict_new.values()):.4f}"
        )

    # Fifth, return the sum of all the values in target_edge_att_dict_new
    return sum(target_edge_att_dict_new.values()), sum(target_edge_att_dict.values())


def get_AVGATT_edge(att: List, edge: Tuple[int]) -> float:
    # First, get the index of the edge in each edge_index in att
    att_list = []
    for curr_att in att:
        for idx, curr_edge in enumerate(list(zip(curr_att[0][0], curr_att[0][1]))):
            if curr_edge == edge:
                att_list.append(idx)
                break

    # Then, get the attention value of the edge in each edge_index in att
    att_value_list = []
    for layer, curr_att in enumerate(att):
        att_value_list.append(curr_att[1][att_list[layer]].item())

    assert len(att_value_list) > 0, f"Edge {edge} not found in attention list"

    # Calculate the average attention value of the edge in all edge_index in att
    return sum(att_value_list) / len(att_value_list)


def return_edges_in_k_hop(
    data: Data, target_idx: int, hop: int, self_loops: bool = False
) -> List[Tuple[int, int]]:
    r"""Returns all edges in :obj:`data` that are connected to :obj:`target_idx`
    and lie within a :obj:`hop` distance.

    Args:
        data (Data): The graph data object.
        target_idx (int): The central node.
        hop (int): The number of hops.
        add_self_loops (bool, optional): If set to :obj:`True`, will add self-loops
            in the returned edge indices. (default: :obj:`False`)
    """
    assert hop > 0
    if self_loops:
        data.edge_index = add_self_loops(remove_self_loops(data.edge_index)[0])[0]

    _, _, _, inv = k_hop_subgraph(
        node_idx=target_idx,
        num_hops=hop,
        edge_index=data.edge_index,
        relabel_nodes=True,
    )

    return data.edge_index[:, inv].t().tolist()


from typing import Tuple, Dict

def attattribute(target_edge: Tuple, ref_node: int, att_matrix_dict: Dict) -> float:
    """
    Calculates the AttAttribute score for a given target edge and reference node.
    """
    # Get the indices of the target edge
    src_idx = target_edge[0]
    tgt_idx = target_edge[1]

    # Get the number of hops
    num_of_hops = len(att_matrix_dict)

    if num_of_hops == 1:
        result = att_matrix_dict[0][tgt_idx, src_idx] if tgt_idx == ref_node else 0
    elif num_of_hops == 2:
        result = att_matrix_dict[1][ref_node, tgt_idx] * att_matrix_dict[0][tgt_idx, src_idx]
        result += att_matrix_dict[1][tgt_idx, src_idx] if tgt_idx == ref_node else 0
    elif num_of_hops == 3:
        result = (torch.sparse.mm(att_matrix_dict[2], att_matrix_dict[1]))[ref_node, tgt_idx].item() * att_matrix_dict[0][tgt_idx, src_idx]
        result += att_matrix_dict[2][ref_node, tgt_idx] * att_matrix_dict[1][tgt_idx, src_idx]
        result += att_matrix_dict[2][tgt_idx, src_idx] if tgt_idx == ref_node else 0
    else:
        raise NotImplementedError("This function only supports up to 3-hop attention.")
    
    return result

def attattribute_sim(target_edge: Tuple, ref_node: int, att_matrix_dict: Dict, att_matrix_dict_sim: Dict) -> float:
    """
    Calculates the AttAttribute_sim score for a given target edge and reference node.
    """
    # Get the indices of the target edge
    src_idx = target_edge[0]
    tgt_idx = target_edge[1]

    # Get the number of hops
    num_of_hops = len(att_matrix_dict)

    if num_of_hops == 1:
        result = att_matrix_dict[0][tgt_idx, src_idx] if tgt_idx == ref_node else 0
    elif num_of_hops == 2:
        result = att_matrix_dict_sim[1][ref_node, tgt_idx] * att_matrix_dict[0][tgt_idx, src_idx]
        result += att_matrix_dict[1][tgt_idx, src_idx] if tgt_idx == ref_node else 0
    elif num_of_hops == 3:
        result = (torch.sparse.mm(att_matrix_dict_sim[2], att_matrix_dict_sim[1]))[ref_node, tgt_idx].item() * att_matrix_dict[0][tgt_idx, src_idx]
        result += att_matrix_dict_sim[2][ref_node, tgt_idx] * att_matrix_dict[1][tgt_idx, src_idx]
        result += att_matrix_dict[2][tgt_idx, src_idx] if tgt_idx == ref_node else 0
    else:
        raise NotImplementedError("This function only supports up to 3-hop attention.")
    
    return result

def avgatt(target_edge: Tuple, ref_node: int, att_matrix_dict: Dict) -> float:
    """
    Calculates the AvgAtt score for a given target edge and reference node.
    """
    # Get the indices of the target edge
    src_idx = target_edge[0]
    tgt_idx = target_edge[1]

    # Get the number of hops
    num_of_hops = len(att_matrix_dict)

    if num_of_hops == 1:
        result = att_matrix_dict[0][tgt_idx, src_idx] if tgt_idx == ref_node else 0
    elif num_of_hops == 2:
        result = (att_matrix_dict[0][tgt_idx, src_idx] + att_matrix_dict[1][tgt_idx, src_idx]) / 2
    elif num_of_hops == 3:
        result = (att_matrix_dict[0][tgt_idx, src_idx] + att_matrix_dict[1][tgt_idx, src_idx] + att_matrix_dict[2][tgt_idx, src_idx]) / 3
    else:
        raise NotImplementedError("This function only supports up to 3-hop attention.")
    
    return result

@torch.no_grad()
def generate_att_dict(model, data) -> Dict:
    """
    Generates a dictionary of attention matrices from a model.
    """
    _ = model(data.x, data.edge_index, return_att=True)
    num_nodes = data.num_nodes
    att = model.att
    att_matrix_dict = {}
    for idx, att_info in enumerate(att):
        att_matrix_dict[idx] = torch.sparse_coo_tensor(att_info[0], att_info[1].squeeze(), size=(num_nodes, num_nodes)).t()
    return att_matrix_dict

@torch.no_grad()
def generate_att_dict_sim(model, data) -> Dict:    
    """
    Generates a dictionary of attention matrices from a model.
    """
    _ = model(data.x, data.edge_index, return_att=True)
    num_nodes = data.num_nodes
    att = model.att
    att_matrix_dict_sim = {}
    for idx, att_info in enumerate(att):
        att_matrix_dict_sim[idx] = torch.sparse_coo_tensor(att_info[0], torch.ones_like(att_info[1].squeeze()), size=(num_nodes, num_nodes)).t()
    return att_matrix_dict_sim