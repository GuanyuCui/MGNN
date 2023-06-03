import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import MessagePassing, Linear, MLP, GCNConv, SGConv, GCN2Conv, PointNetConv
from torch_geometric.nn.conv import APPNP as _APPNP
from torch_geometric.utils import degree, add_self_loops
from math import log

# A single layer linear model (with bias).
class MyLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(MyLinear, self).__init__()
        self.dropout = dropout
        self.linear = Linear(in_channels = in_channels, out_channels = out_channels)

    def forward(self, x, edge_index):
        x = self.linear(x)
        x = F.dropout(x, p = self.dropout, training = self.training, inplace = True)
        return x

# An MLP.
class MyMLP(torch.nn.Module):
    def __init__(self, channel_list, dropout):
        super(MyMLP, self).__init__()
        self.mlp = MLP(channel_list = channel_list, dropout = dropout, norm = None)
    
    def forward(self, x, edge_index):
        return self.mlp(x)

class SGC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, 
                out_channels, K, dropout):
        super(SGC, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.K = K
        self.dropout = dropout

        self.conv = SGConv(in_channels = self.in_channels, out_channels = self.out_channels, K = self.K, cached = True)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.dropout(x, p = self.dropout, training = self.training, inplace = True)
        return x

class APPNP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, 
                out_channels, K, 
                alpha, dropout):
        super(APPNP, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.K = K
        self.alpha = alpha
        self.dropout = dropout

        self.initial = MLP(channel_list = [self.in_channels, self.hidden_channels, self.hidden_channels], dropout = self.dropout, norm = None)
        self.conv = _APPNP(K = self.K, alpha = self.alpha, cached = True)
        self.final = Linear(in_channels = self.hidden_channels, out_channels = self.out_channels, weight_initializer = 'glorot')

    def forward(self, x, edge_index):
        x = self.initial(x)
        x = self.conv(x, edge_index)
        x = self.final(x)
        return x

class GCNII(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                num_layers, alpha, theta, dropout):
        super(GCNII, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.alpha = alpha
        self.theta = theta
        self.dropout = dropout

        self.initial = MLP(channel_list = [self.in_channels, self.hidden_channels, self.hidden_channels], dropout = self.dropout, norm = None)
        self.convs = ModuleList()
        for i in range(self.num_layers):
            self.convs.append(GCN2Conv(channels = self.hidden_channels, alpha = self.alpha, theta = self.theta, layer = i + 1, cached = True))
        self.final = Linear(in_channels = self.hidden_channels, out_channels = self.out_channels, weight_initializer = 'glorot')

    def forward(self, x, edge_index):
        x = self.initial(x)
        x_0 = x
        for i in range(self.num_layers):
            x = self.convs[i].forward(x = x, x_0 = x_0, edge_index = edge_index)
            x = F.dropout(x, p = self.dropout, training = self.training, inplace = True)
            x = F.relu(x, inplace = True)
        del x_0
        x = self.final(x)
        return x


class MGNNAttention(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout, method = 'concat'):
        super(MGNNAttention, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.method = method

        assert self.method in ['cosine', 'concat', 'bilinear']

        self.initial = MLP(channel_list = [self.in_channels, self.hidden_channels, self.hidden_channels], dropout = self.dropout, norm = None)
        self.a = Linear(in_channels = 2 * self.hidden_channels, out_channels = 1, bias = False, weight_initializer = 'glorot')
        self.W = Linear(in_channels = self.hidden_channels, out_channels = self.hidden_channels, bias = False, weight_initializer = 'glorot')

    def forward(self, x, edge_index):
        x = self.initial(x)
        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]
        del x

        if self.method == 'cosine':
            edge_attention = F.cosine_similarity(x_i, x_j, dim = 1)
        elif self.method == 'concat':
            edge_attention = torch.tanh(self.a(torch.cat([x_i, x_j], dim = 1))).flatten()
        elif self.method == 'bilinear':
            edge_attention = torch.tanh(torch.sum(x_i * self.W(x_j), dim = 1))
        del x_i, x_j
        return edge_attention

class MGNNConv(MessagePassing):
    def __init__(self, channels, alpha, beta, theta, layer, eps = 1e-5):
        super(MGNNConv, self).__init__(aggr = 'add')
        self.alpha = alpha
        self.beta = beta
        self.gamma = log(theta / layer + 1)
        self.eps = eps
        self.channels = channels

        self.linear = Linear(in_channels = channels, out_channels = channels, bias = True, weight_initializer = 'glorot')

    def forward(self, x, x_0, edge_index, edge_metric, norm, deg_inv_sqrt):
        x = self.alpha * x_0 + self.propagate(x = x, edge_index = edge_index, edge_metric = edge_metric, norm = norm, deg_inv_sqrt = deg_inv_sqrt.reshape(-1, 1))
        x = (1 - self.gamma) * x + self.gamma * self.linear.forward(x)
        return x

    def message(self, x_i, x_j, edge_metric, norm, deg_inv_sqrt_i, deg_inv_sqrt_j):
        topological_message = norm.view(-1, 1) * x_j
        positional_message = norm.view(-1, 1) * edge_metric.view(-1, 1) * (x_i - x_j) / ( (torch.norm((deg_inv_sqrt_i.view(-1, 1) * x_i - deg_inv_sqrt_j.view(-1, 1) * x_j), p = 2, dim = 1) + self.eps).view(-1, 1) )
        return (1 - self.alpha) * topological_message + self.beta * positional_message

class MGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, 
                out_channels, num_layers, 
                alpha, beta, theta, dropout, 
                attention_method = 'concat', initial = 'Linear', eps = 1e-5):
        super(MGNN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.dropout = dropout
        self.attention_method = attention_method
        self.initial_method = initial
        self.eps = eps

        self.attention = MGNNAttention(in_channels = self.hidden_channels, hidden_channels = self.hidden_channels, dropout = self.dropout, method = self.attention_method)
        if self.initial_method == 'Linear':
            self.initial = Linear(in_channels = self.in_channels, out_channels = self.hidden_channels, bias = True, weight_initializer = 'glorot')
        elif self.initial_method == 'MLP':
            self.initial = MLP(channel_list = [self.in_channels, self.hidden_channels, self.hidden_channels], dropout = self.dropout, norm = None)
        elif self.initial_method == 'GC':
            self.initial = GCNConv(in_channels = self.in_channels, out_channels = self.hidden_channels, bias = True, add_self_loops = False, normalize = False)

        self.convs = ModuleList()
        for i in range(self.num_layers):
            self.convs.append(MGNNConv(channels = self.hidden_channels, alpha = self.alpha, beta = self.beta, theta = self.theta, layer = i + 1))
        self.final = Linear(in_channels = self.hidden_channels, out_channels = self.out_channels, bias = True, weight_initializer = 'glorot')

        self.norm_cache = None
        self.deg_inv_sqrt_cache = None

    def forward(self, x, edge_index, edge_metric = None):
        if self.norm_cache is None:
            row, col = edge_index
            deg = degree(col, x.size(0), dtype = x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            self.norm_cache = (deg_inv_sqrt[row] * deg_inv_sqrt[col])
            self.deg_inv_sqrt_cache = deg_inv_sqrt
        # Embedding.
        if self.initial_method == 'GC':
            x = self.initial(x, edge_index)
        else:
            x = self.initial(x)
        x_0 = x
        # Learn edge attention from original data.
        edge_attention = self.attention(x = x_0, edge_index = edge_index)
        # Calculate the edge metric using edge attention and z^{(0)} = f(x).
        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]

        edge_metric = (1 - edge_attention) / (1 + edge_attention + self.eps) * torch.norm(x_i - x_j, p = 2, dim = 1)
        # Graph Propagation.
        for i in range(self.num_layers):
            x = self.convs[i].forward(x = x, x_0 = x_0, edge_index = edge_index, edge_metric = edge_metric, norm = self.norm_cache, deg_inv_sqrt = self.deg_inv_sqrt_cache)
            x = F.dropout(x, p = self.dropout, training = self.training, inplace = True)
            x = F.relu(x, inplace = True)
        # Classification.
        x = self.final(x)
        return x


@torch.jit._overload
def pgnn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def pgnn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def pgnn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=False, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t, deg_inv_sqrt

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

        return edge_index, edge_weight, deg_inv_sqrt


def calc_M(f, edge_index, edge_weight, deg_inv_sqrt, num_nodes, mu, p):
        if isinstance(edge_index, SparseTensor):
            row, col, edge_weight = edge_index.coo()
        else:
            row, col = edge_index[0], edge_index[1]

        ## calculate M
        graph_grad = torch.pow(edge_weight, 0.5).view(-1, 1) * (deg_inv_sqrt[row].view(-1, 1) * f[row] - deg_inv_sqrt[col].view(-1, 1) * f[col])
        graph_grad = torch.pow(torch.norm(graph_grad, dim=1), p-2)
        M = edge_weight * graph_grad
        M.masked_fill_(M == float('inf'), 0)
        alpha = (deg_inv_sqrt.pow(2) * scatter_add(M, col, dim=0, dim_size=num_nodes) + (2*mu)/p).pow(-1)
        beta = 4*mu / p * alpha
        M_ = alpha[row] * deg_inv_sqrt[row] * M * deg_inv_sqrt[col]
        return M_, beta




from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


class pGNNConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor, Tensor]]
    _cached_adj_t: Optional[Tuple[SparseTensor, Tensor]]

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 mu: float,
                 p: float,
                 K: int,
                 improved: bool = False, 
                 cached: bool = False,
                 add_self_loops: bool = False, 
                 normalize: bool = True,
                 bias: bool = True, 
                 return_M_: bool = False,
                 **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(pGNNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mu = mu
        self.p = p
        self.K = K
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.return_M_ = return_M_

        self.lin1 = torch.nn.Linear(in_channels, out_channels, bias=bias)

        if return_M_:
            self.new_edge_attr = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        num_nodes = x.size(self.node_dim)
        
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight, deg_inv_sqrt = pgnn_norm(  # yapf: disable
                        edge_index, edge_weight, num_nodes,
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight, deg_inv_sqrt)
                else:
                    edge_index, edge_weight, deg_inv_sqrt = cache[0], cache[1], cache[2]
            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index, deg_inv_sqrt = pgnn_norm(  # yapf: disable
                        edge_index, edge_weight, num_nodes,
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = (edge_index, deg_inv_sqrt)
                else:
                    edge_index, deg_inv_sqrt = cache[0], cache[1]

        out = x
        for _ in range(self.K):
            edge_attr, beta = calc_M(out, edge_index, edge_weight, deg_inv_sqrt, num_nodes, self.mu, self.p)
            out = self.propagate(edge_index, x=out, edge_weight=edge_attr, size=None) + beta.view(-1, 1) * x
            
        out = self.lin1(out)

        if self.return_M_:
            self.new_edge_attr = edge_attr
            
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class pGNN(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 num_hid=16, 
                 mu=0.1,
                 p=2,
                 K=2,
                 dropout=0.5,
                 cached=True):
        super(pGNN, self).__init__()
        self.dropout = dropout
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.conv1 = pGNNConv(num_hid, out_channels, mu, p, K, cached=cached)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)        
        return F.log_softmax(x, dim=1)


class PointNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                num_layers, dropout):
        super(PointNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.initial = MLP(channel_list = [self.in_channels, self.hidden_channels, self.hidden_channels], dropout = self.dropout, norm = None)
        self.pos_gen = MLP(channel_list = [self.hidden_channels, self.hidden_channels, 3], dropout = self.dropout, norm = None)

        self.local_nn = MLP(channel_list = [self.hidden_channels + 3, self.hidden_channels], dropout = self.dropout, norm = None)
        self.global_nn = MLP(channel_list = [self.hidden_channels, self.hidden_channels], dropout = self.dropout, norm = None) 

        self.convs = ModuleList()
        for i in range(self.num_layers):
            self.convs.append(PointNetConv(local_nn = self.local_nn, global_nn = self.global_nn))
        self.final = Linear(in_channels = self.hidden_channels, out_channels = self.out_channels, weight_initializer = 'glorot')

    def forward(self, x, edge_index):
        x = self.initial(x)
        x_0 = x
        pos = self.pos_gen(x_0)
        for i in range(self.num_layers):
            x = self.convs[i].forward(x = x, edge_index = edge_index, pos = pos)
            x = F.dropout(x, p = self.dropout, training = self.training, inplace = True)
            x = F.relu(x, inplace = True)
        x = self.final(x)
        return x