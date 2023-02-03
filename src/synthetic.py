import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import stochastic_blockmodel_graph, to_networkx, degree
from torch_geometric.nn import MessagePassing

from math import log

class SyntheticDataset(Dataset):
	def __init__(self):
		# self.num_node_features = 0
		# self.num_features = 0
		pass

	@property
	def raw_file_names(self):
		r"""The name of the files in the :obj:`self.raw_dir` folder that must
		be present in order to skip downloading."""
		pass

	@property
	def processed_file_names(self):
		r"""The name of the files in the :obj:`self.processed_dir` folder that
		must be present in order to skip processing."""
		pass

	def download(self):
		r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
		pass

	def process(self):
		r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
		pass

	def len(self) -> int:
		r"""Returns the number of graphs stored in the dataset."""
		pass

	def get(self, idx: int) -> Data:
		r"""Gets the data object at index :obj:`idx`."""
		pass

def get_synthetic_data(block_sizes: list, edge_probs: list, means: list, stddevs: list):
	edge_index = stochastic_blockmodel_graph(block_sizes = block_sizes, edge_probs = edge_probs, directed = False)
	xs, ys = [], []
	i = 0
	for block_size, mean, stddev in zip(block_sizes, means, stddevs):
		_ = np.random.normal(scale = stddev, size = (block_size, 2))
		_[:, 0] = _[:, 0] + mean[0]
		_[:, 1] = _[:, 1] + mean[1]
		ys.append([i] * block_size)
		xs.append(_)
		i += 1
		
	x = np.concatenate(xs)
	y = np.concatenate(ys)
	data = Data(x = Tensor(x), y = Tensor(y).long(), edge_index = edge_index, num_nodes = sum(block_sizes))
	dataset = SyntheticDataset()
	dataset.data = data
	dataset.num_classes = len(block_sizes)
	G = to_networkx(data)
	return data, G, dataset

def get_edge_attention(edge_index, y):
	edge_attention = np.zeros(len(edge_index[0]))
	l = 0
	for i, j in zip(edge_index[0], edge_index[1]):
		edge_attention[l] = 1 if y[i] == y[j] else -1
		l += 1
	return edge_attention

def draw_graph(filename : str, title : str, G : nx.Graph, x : np.array, node_colors : list, edge_colors : list):
	pos = {}
	for i in range(len(x)):
		pos[i] = (x[i, 0], x[i, 1])
	plt.figure(figsize = (5, 5))
	plt.xlabel(title)
	nx.draw_networkx(G, pos = pos, arrows = False, with_labels = False, node_size = 20, node_color = node_colors, edge_color = edge_colors, alpha = 0.8, width = 0.5)
	plt.savefig(filename, bbox_inches = 'tight', dpi = 300, format = 'png')
	plt.close()


class SGCPropagationLayer(MessagePassing):
	def __init__(self):
		super(SGCPropagationLayer, self).__init__(aggr = 'add')
	def forward(self, x, edge_index, norm = None):
		if norm is None:
			row, col = edge_index
			deg = degree(col, x.size(0), dtype = x.dtype)
			deg_inv_sqrt = deg.pow(-0.5)
			deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
			norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
		return self.propagate(x = x, edge_index = edge_index, norm = norm)
	def message(self, x_j, norm):
		return norm.view(-1, 1) * x_j

class APPNPPropagationLayer(MessagePassing):
	def __init__(self, alpha):
		super(APPNPPropagationLayer, self).__init__(aggr = 'add')
		self.alpha = alpha

	def forward(self, x, x_0, edge_index, norm = None):
		if norm is None:
			row, col = edge_index
			deg = degree(col, x.size(0), dtype = x.dtype)
			deg_inv_sqrt = deg.pow(-0.5)
			deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
			norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
		return self.alpha * x_0 + (1 - self.alpha) * self.propagate(x = x, edge_index = edge_index, norm = norm)
	
	def message(self, x_j, norm):
		return norm.view(-1, 1) * x_j

class MGNNPropagationLayer(MessagePassing):
    def __init__(self, alpha, beta, eps = 1e-5):
        super(MGNNPropagationLayer, self).__init__(aggr = 'add')
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, x, x_0, edge_index, edge_metric, norm, deg_inv_sqrt):
        x = self.alpha * x_0 + self.propagate(x = x, edge_index = edge_index, edge_metric = edge_metric, norm = norm, deg_inv_sqrt = deg_inv_sqrt.reshape(-1, 1))
        return x

    def message(self, x_i, x_j, edge_metric, norm, deg_inv_sqrt_i, deg_inv_sqrt_j):
        topological_message = norm.view(-1, 1) * x_j
        positional_message = norm.view(-1, 1) * edge_metric.view(-1, 1) * (x_i - x_j) / ( (torch.norm((deg_inv_sqrt_i.view(-1, 1) * x_i - deg_inv_sqrt_j.view(-1, 1) * x_j), p = 2, dim = 1) + self.eps).view(-1, 1) )
        return (1 - self.alpha) * topological_message + self.beta * positional_message

if __name__ == '__main__': 
	block_sizes = [50, 50, 50, 50]
	edge_probs = [[0.3, 0.05, 0.05, 0.05], [0.05, 0.3, 0.05, 0.05], [0.05, 0.05, 0.3, 0.05], [0.05, 0.05, 0.05, 0.3]]
	# edge_probs = [[0.05, 0.2, 0.2, 0.2], [0.2, 0.05, 0.2, 0.2], [0.2, 0.2, 0.05, 0.2], [0.2, 0.2, 0.2, 0.05]]
	means = [(0, 0), (1, 0), (0, 1), (1, 1)]
	stddevs = [1, 1, 1, 1]

	# Generate synthetic datasets.
	data, G, _ = get_synthetic_data(block_sizes, edge_probs, means, stddevs)

	# Calculate edge attentions and colors.
	edge_attention = get_edge_attention(data.edge_index, data.y)
	node_colors = ['blue'] * block_sizes[0] + ['yellow'] * block_sizes[1] + ['green'] * block_sizes[2] + ['cyan'] * block_sizes[3]
	edge_colors = []
	for _ in edge_attention:
		edge_colors.append('black' if _ == 1 else 'red')
	# Set edge metrics.
	edge_metric = []
	for _ in edge_attention:
		edge_metric.append(0 if _ == 1 else 5)
	edge_metric = Tensor(edge_metric)

	SGCProp = SGCPropagationLayer()
	APPNPProp = APPNPPropagationLayer(alpha = 0.1)
	MGNNProp = MGNNPropagationLayer(alpha = 5e-2, beta = 0.5)

	n_layers = 8
	x_SGC = data.x.clone()
	x_APPNP = data.x.clone()
	x_MGNN = data.x.clone()

	row, col = data.edge_index
	deg = degree(col, data.x.size(0), dtype = data.x.dtype)
	deg_inv_sqrt = deg.pow(-0.5)
	deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
	norm = (deg_inv_sqrt[row] * deg_inv_sqrt[col])

	if not os.path.exists('out'):
		os.mkdir('out')

	# Draw the original graph.
	draw_graph('./out/SGC-0.png', 'original', G, data.x, node_colors, edge_colors)
	draw_graph('./out/APPNP-0.png', 'original', G, data.x, node_colors, edge_colors)
	draw_graph('./out/MGNN-0.png', 'original', G, data.x, node_colors, edge_colors)

	for i in list(range(1, n_layers + 1)):
		x_SGC = SGCProp.forward(x_SGC, data.edge_index, edge_metric)
		draw_graph('out/SGC-{}.png'.format(i), 'SGC-{}'.format(i), G, x_SGC, node_colors, edge_colors)

		x_APPNP = APPNPProp.forward(x_APPNP, data.x, data.edge_index)
		draw_graph('out/APPNP-{}.png'.format(i), 'APPNP-{}'.format(i), G, x_APPNP, node_colors, edge_colors)

		x_MGNN = MGNNProp.forward(x_MGNN, data.x, data.edge_index, edge_metric, norm, deg_inv_sqrt)
		draw_graph('out/MGNN-{}.png'.format(i), 'MGNN-{}'.format(i), G, x_MGNN, node_colors, edge_colors)