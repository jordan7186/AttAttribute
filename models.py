"""
This file contains the GAT models used in the experiments.
"""
# Import the required libraries
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# From the source code from GATConv
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value


# Define the GAT model, with 2 hidden layers
class GAT_L2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout=0):
        super().__init__()
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            concat=False,
            add_self_loops=True,
        )
        self.att = []

    def forward(self, x, edge_index, return_att: bool = False):
        if return_att:
            x, att1 = self.conv1(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            x, att2 = self.conv2(x, edge_index, return_attention_weights=True)
            self.att = [att1, att2]
        else:
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = self.conv2(x, edge_index)
        return x


# Define the GAT model, with 3 hidden layers
class GAT_L3(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout=0):
        super().__init__()
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )
        self.conv3 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )
        self.att = []

    def forward(self, x, edge_index, return_att: bool = False):
        if return_att:
            x, att1 = self.conv1(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            x, att2 = self.conv2(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            x, att3 = self.conv3(x, edge_index, return_attention_weights=True)
            self.att = [att1, att2, att3]
        else:
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = self.conv2(x, edge_index)
            x = F.elu(x)
            x = self.conv3(x, edge_index)
        return x


class GAT_karate(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT_karate, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=1)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1)

    def forward(self, x, edge_index):
        x, att1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x, att2 = self.conv2(x, edge_index, return_attention_weights=True)
        return x, (att1, att2)


class GATConv_mask(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = "mean",
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(
                in_channels,
                heads * out_channels,
                bias=False,
                weight_initializer="glorot",
            )
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(
                in_channels[0], heads * out_channels, False, weight_initializer="glorot"
            )
            self.lin_dst = Linear(
                in_channels[1], heads * out_channels, False, weight_initializer="glorot"
            )

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(
                edge_dim, heads * out_channels, bias=False, weight_initializer="glorot"
            )
            self.att_edge = Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter("att_edge", None)

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights=None,
        mask_edge=None,
    ):
        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index,
                    edge_attr,
                    fill_value=self.fill_value,
                    num_nodes=num_nodes,
                )
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form"
                    )

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        # Identify indices of edges that need to be masked
        if mask_edge is not None:
            mask_edge_index = (edge_index[0] == mask_edge[0]) & (
                edge_index[1] == mask_edge[1]
            )
            alpha[mask_edge_index] = 0

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def edge_update(
        self,
        alpha_j: Tensor,
        alpha_i: OptTensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )


# Since the previous models have defined on GAT_L2 and GAT_L3, we need a way to
# copy the model parameters from those models to the ones defined from
# GATConv_intervention.


class GAT_L2_intervention(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, heads, dropout=0, **kwargs
    ):
        super().__init__()
        self.conv1 = GATConv_mask(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            **kwargs,
        )
        self.conv2 = GATConv_mask(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            concat=False,
            add_self_loops=True,
            **kwargs,
        )
        self.att = []

    def forward(self, x, edge_index, return_att: bool = False, mask_edge: Tuple = None):
        if return_att:
            x, att1 = self.conv1(
                x, edge_index, return_attention_weights=True, mask_edge=mask_edge
            )
            x = F.elu(x)
            x, att2 = self.conv2(
                x, edge_index, return_attention_weights=True, mask_edge=mask_edge
            )
            self.att = [att1, att2]
        else:
            x = self.conv1(x, edge_index, mask_edge=mask_edge)
            x = F.elu(x)
            x = self.conv2(x, edge_index, mask_edge=mask_edge)
        return x


class GAT_L3_intervention(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, heads, dropout=0, **kwargs
    ):
        super().__init__()
        self.conv1 = GATConv_mask(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            **kwargs,
        )
        self.conv2 = GATConv_mask(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            **kwargs,
        )
        self.conv3 = GATConv_mask(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
            **kwargs,
        )
        self.att = []

    def forward(self, x, edge_index, return_att: bool = False, mask_edge: Tuple = None):
        if return_att:
            x, att1 = self.conv1(
                x, edge_index, return_attention_weights=True, mask_edge=mask_edge
            )
            x = F.elu(x)
            x, att2 = self.conv2(
                x, edge_index, return_attention_weights=True, mask_edge=mask_edge
            )
            x = F.elu(x)
            x, att3 = self.conv3(
                x, edge_index, return_attention_weights=True, mask_edge=mask_edge
            )
            self.att = [att1, att2, att3]
        else:
            x = self.conv1(x, edge_index, mask_edge=mask_edge)
            x = F.elu(x)
            x = self.conv2(x, edge_index, mask_edge=mask_edge)
            x = F.elu(x)
            x = self.conv3(x, edge_index, mask_edge=mask_edge)
        return x
