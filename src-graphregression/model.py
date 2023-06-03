import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn.models import GCN, GAT, GIN
from torch_geometric.nn import MessagePassing, Linear, MLP, GCNConv, SGConv, GCN2Conv, PointNetConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn.conv import APPNP as _APPNP
from torch_geometric.utils import degree, add_self_loops
from math import log

class MyGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, 
                out_channels, num_layers, dropout):
        super(MyGCN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.initial = Linear(in_channels = self.in_channels, out_channels = self.hidden_channels, bias = True, weight_initializer = 'glorot')
        self.convs = GCN(in_channels = self.hidden_channels, hidden_channels = self.hidden_channels, out_channels = self.hidden_channels, num_layers = self.num_layers, dropout = self.dropout)
        self.final = Linear(in_channels = self.hidden_channels, out_channels = self.out_channels, bias = True, weight_initializer = 'glorot')
        
    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        x = self.initial(x)
        x = self.convs(x = x, edge_index = edge_index)
        # Pooling.
        x = global_add_pool(x, batch = batch).squeeze()
        # Classification.
        x = self.final(x).squeeze()
        return x

class MyGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, 
                out_channels, num_layers, dropout):
        super(MyGAT, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.initial = Linear(in_channels = self.in_channels, out_channels = self.hidden_channels, bias = True, weight_initializer = 'glorot')
        self.convs = GAT(in_channels = self.hidden_channels, hidden_channels = self.hidden_channels, out_channels = self.hidden_channels, num_layers = self.num_layers, dropout = self.dropout)
        self.final = Linear(in_channels = self.hidden_channels, out_channels = self.out_channels, bias = True, weight_initializer = 'glorot')
        
    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch
        
        x = self.initial(x)
        x = self.convs(x = x, edge_index = edge_index)
        # Pooling.
        x = global_add_pool(x, batch = batch).squeeze()
        # Classification.
        x = self.final(x).squeeze()
        return x

class MyGIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, 
                out_channels, num_layers, dropout):
        super(MyGIN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.initial = Linear(in_channels = self.in_channels, out_channels = self.hidden_channels, bias = True, weight_initializer = 'glorot')
        self.convs = GIN(in_channels = self.hidden_channels, hidden_channels = self.hidden_channels, out_channels = self.hidden_channels, num_layers = self.num_layers, dropout = self.dropout)
        self.final = Linear(in_channels = self.hidden_channels, out_channels = self.out_channels, bias = True, weight_initializer = 'glorot')
        
    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch
        
        x = self.initial(x)
        x = self.convs(x = x, edge_index = edge_index)
        # Pooling.
        x = global_add_pool(x, batch = batch).squeeze()
        # Classification.
        x = self.final(x).squeeze()
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

    def forward(self, data, edge_metric = None):
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch
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
        # Pooling.
        x = global_add_pool(x, batch = batch).squeeze()
        # Classification.
        x = self.final(x).squeeze()
        return x

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
        
    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch
        
        x = self.initial(x)
        x_0 = x
        pos = self.pos_gen(x_0)
        for i in range(self.num_layers):
            x = self.convs[i].forward(x = x, edge_index = edge_index, pos = pos)
            x = F.dropout(x, p = self.dropout, training = self.training, inplace = True)
            x = F.relu(x, inplace = True)
        # Pooling.
        x = global_add_pool(x, batch = batch).squeeze()
        x = self.final(x).squeeze()
        return x