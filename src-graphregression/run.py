import os
import random
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.profile import count_parameters
from torch_geometric.loader import DataLoader
from utils import get_model, get_dataset

def train(model, dataloader, loss_fn, optimizer, device):
    # Training.
    model.train()
    
    for data in dataloader:
        optimizer.zero_grad()
        data = data.to(device)
        # Forward pass.
        pred = model(data)
        loss = loss_fn(pred, data.y)
        # Backward pass.
        # with torch.autograd.detect_anomaly():
        loss.backward()
        optimizer.step()


@torch.no_grad()
def infer(model, dataloader, device):
    # Inference.
    model.eval()
    error = 0.0
    for data in dataloader:
        data = data.to(device)
        # Forward pass.
        pred = model.forward(data)
        error += (pred - data.y).abs().sum().item()

    return error / len(dataloader.dataset)

def run(args, verbose: bool = True):
    # torch.autograd.set_detect_anomaly(True)
    # Seed.
    if args['seed'] != -1:
        random.seed(args['seed'])
        os.environ['PYTHONHASHSEED'] = str(args['seed'])
        np.random.seed(args['seed'])
        torch.manual_seed(args['seed'])
        torch.cuda.manual_seed(args['seed'])
        torch.cuda.manual_seed_all(args['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Device.
    device = (args['device'] if torch.cuda.is_available() else 'cpu')

    if verbose:
        print('Using device: {}'.format(device))
        print(args)

    times = args['times']
    epochs = args['epochs']

    val_maes = []
    test_maes = []
    for time in range(times):
        if device != 'cpu':
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

        # Dataset & model.
        train_dataset, val_dataset, test_dataset = get_dataset(root = '../data/{}'.format(args['dataset']), name = args['dataset'])

        train_loader = DataLoader(train_dataset, batch_size = args['batch'], shuffle = False)
        val_loader = DataLoader(val_dataset, batch_size = args['batch'], shuffle = False) 
        test_loader = DataLoader(test_dataset, batch_size = args['batch'], shuffle = False)
   
        model = get_model(model = args['model'], dataset = train_dataset, args = args).to(device)

        # if verbose:
        #     print(f'#Nodes: {graph.num_nodes}')
        #     print(f'#Edges: {len(graph.edge_index[0])}')
        #     print(f'#Features: {graph.num_features}')
        #     print(f'#Classes: {dataset.num_classes}')
        #     raise

        optimizer = torch.optim.Adam(params = model.parameters(), lr = args['lr'], weight_decay = args['weight_decay'])        
        loss_fn = torch.nn.L1Loss()

        if verbose:
            print('Training time: {}'.format(time))
        best_val_mae = float('inf')

        patience = 0

        for epoch in tqdm(range(epochs)) if verbose else range(epochs):
            train(model = model, dataloader = train_loader, loss_fn = loss_fn, optimizer = optimizer, device = device)
            val_mae = infer(model = model, dataloader = val_loader, device = device)
            if val_mae < best_val_mae:
                patience = 0
                best_val_mae = val_mae
                test_mae = infer(model = model, dataloader = test_loader, device = device)
            else:
                patience += 1
                if patience > args['early_stopping']:
                    break
        if verbose:
            print('Training done. Test_mae (on best valid_mae = {:.4f}) = {:.4f}\n'.format(best_val_mae, test_mae))
        val_maes.append(val_mae)
        test_maes.append(test_mae)

    val_maes = np.array(val_maes)
    val_mean = np.mean(val_maes)
    val_stddev = np.std(val_maes)

    test_maes = np.array(test_maes)
    test_mean = np.mean(test_maes)
    test_stddev = np.std(test_maes)
    if verbose:
        print('Mean = {:.4f}%; Stddev = {:.4f}%'.format(test_mean, test_stddev))

    return val_mean, val_stddev, test_mean, test_stddev