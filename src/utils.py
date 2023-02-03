from torch_geometric.nn.models import GCN, GAT, LINKX
from torch_geometric.datasets import Planetoid, CoraFull, Amazon, Coauthor, WebKB, WikipediaNetwork, Actor, DeezerEurope, WikiCS, LINKXDataset
from ogb.nodeproppred import PygNodePropPredDataset
from model import MyLinear, MyMLP, SGC, APPNP, GCNII, MGNN, pGNN

def get_model(model : str, dataset, args):
	if model == 'Linear':
		model = MyLinear(in_channels = dataset.data.num_features, out_channels = dataset.num_classes, 
					dropout = args['dropout'])
	elif model == 'MLP':
		model = MyMLP(channel_list = [dataset.data.num_features] + [args['hidden_dim']] * (args['num_layers'] - 1) + [dataset.num_classes],
					dropout = args['dropout'])
	elif model == 'GCN':
		model = GCN(in_channels = dataset.data.num_features, 
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes, 
					num_layers = args['num_layers'], 
					dropout = args['dropout'])
	elif model == 'SGC':
		model = SGC(in_channels = dataset.data.num_features, 
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes, 
					K = args['num_layers'], 
					dropout = args['dropout'])
	elif model == 'GAT':
		model = GAT(in_channels = dataset.data.num_features,
					hidden_channels = args['hidden_dim'],
					out_channels = dataset.num_classes,
					num_layers = args['num_layers'],
					dropout = args['dropout'])
	elif model == 'APPNP':
		model = APPNP(in_channels = dataset.data.num_features,
					hidden_channels = args['hidden_dim'],
					out_channels = dataset.num_classes,
					K = args['num_layers'],
					alpha = args['alpha'],
					dropout = args['dropout'])
	elif model == 'GCNII':
		model = GCNII(in_channels = dataset.data.num_features, 
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes, 
					num_layers = args['num_layers'], 
					alpha = args['alpha'],
					theta = args['theta'],
					dropout = args['dropout'])
	elif model == 'MGNN':
		model = MGNN(in_channels = dataset.data.num_features, 
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes, 
					num_layers = args['num_layers'], 
					alpha = args['alpha'],
					beta = args['beta'],
					theta = args['theta'],
					dropout = args['dropout'],
					attention_method = args['attention_method'],
					initial = args['initial'])
	elif model == 'LINKX':
		model = LINKX(num_nodes = args['num_nodes'],
					in_channels = dataset.data.num_features,
					hidden_channels = args['hidden_dim'], 
					out_channels = dataset.num_classes,
					num_layers = args['num_layers'],
					dropout = args['dropout'])
	elif model == 'pGNN':
		model = pGNN(in_channels = dataset.data.num_features, 
                	out_channels = dataset.num_classes,
                	num_hid = args['hidden_dim'], 
                	mu = args['alpha'],
                 	p = args['theta'],
                 	K = args['num_layers'],
                	dropout = args['dropout'])
	return model

def get_dataset(root : str, name : str):
	if name in ['Cora', 'CiteSeer', 'PubMed']:
		dataset = Planetoid(root = root, name = name)
	elif name == 'CoraFull':
		dataset = CoraFull(root = root)
	elif name in ['Computers', 'Photo']:
		dataset = Amazon(root = root, name = name)
	elif name in ['CS', 'Physics']:
		dataset = Coauthor(root = root, name = name)
	elif name in ['Cornell', 'Texas', 'Wisconsin']:
		dataset = WebKB(root = root, name = name)
	elif name in ['Chameleon', 'Squirrel']:
		dataset = WikipediaNetwork(root = root, name = name.lower())
	elif name == 'Actor':
		dataset = Actor(root = root)
	elif name == 'DeezerEurope':
		dataset = DeezerEurope(root = root)
	elif name == 'WikiCS':
		dataset = WikiCS(root = root, is_undirected = True)
	elif name in ['genius']:
		dataset = LINKXDataset(root = root, name = name)
	elif name in ['ogbn-arxiv', 'ogbn-products']:
		dataset = PygNodePropPredDataset(name = name, root = root)
	else:
		raise Exception('Unknown dataset.')

	return dataset