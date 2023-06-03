from torch_geometric.datasets import ZINC
from ogb.nodeproppred import PygNodePropPredDataset
from model import MyGCN, MyGAT, MyGIN, MGNN, PointNet

def get_model(model : str, dataset, args):
	if model == 'GCN':
		model = MyGCN(in_channels = dataset.data.num_features, 
					hidden_channels = args['hidden_dim'], 
					out_channels = 1, 
					num_layers = args['num_layers'], 
					dropout = args['dropout'])
	elif model == 'GAT':
		model = MyGAT(in_channels = dataset.data.num_features,
					hidden_channels = args['hidden_dim'],
					out_channels = 1,
					num_layers = args['num_layers'],
					dropout = args['dropout'])
	elif model == 'GIN':
		model = MyGIN(in_channels = dataset.data.num_features,
					hidden_channels = args['hidden_dim'],
					out_channels = 1,
					num_layers = args['num_layers'],
					dropout = args['dropout'])
	elif model == 'MGNN':
		model = MGNN(in_channels = dataset.data.num_features, 
					hidden_channels = args['hidden_dim'], 
					out_channels = 1, 
					num_layers = args['num_layers'], 
					alpha = args['alpha'],
					beta = args['beta'],
					theta = args['theta'],
					dropout = args['dropout'],
					attention_method = args['attention_method'],
					initial = args['initial'])
	elif model == 'PointNet':
		model = PointNet(in_channels = dataset.data.num_features,
					hidden_channels = args['hidden_dim'], 
					out_channels = 1, 
					num_layers = args['num_layers'],
					dropout = args['dropout'])
	return model

def get_dataset(root : str, name : str):
	if name == 'ZINC':
		ZINC_train = ZINC(root = root, subset = True, split = 'train')
		ZINC_val = ZINC(root = root, subset = True, split = 'val')
		ZINC_test = ZINC(root = root, subset = True, split = 'test')
		return (ZINC_train, ZINC_val, ZINC_test)
	else:
		raise Exception('Unknown dataset.')

	return dataset