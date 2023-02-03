import optuna
from run import run
from argparse import ArgumentParser

global args

def objective(trail):
    params = {
        'lr': trail.suggest_categorical('lr', [1e-3, 5e-3, 1e-2, 5e-2]),
        'weight_decay': trail.suggest_categorical('weight_decay', [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
        'dropout': trail.suggest_categorical('dropout', [0.0, 0.1, 0.3, 0.5]),
        'num_layers': trail.suggest_categorical('num_layers', [2, 4, 8]),
        'alpha': trail.suggest_float('alpha', 0, 1, step = 0.01),
        #'alpha': trail.suggest_float('alpha', 0, 0.2, step = 0.01),
        'beta': trail.suggest_float('beta', 0, 1, step = 0.01),
        'theta': trail.suggest_categorical('theta', [0.5, 1.0, 1.5]),
        #'theta': trail.suggest_categorical('theta', [1.5]),
        'attention_method': trail.suggest_categorical('attention_method', ['concat', 'bilinear'])
    }

    args['lr'] = params['lr']
    args['weight_decay'] = params['weight_decay']
    args['dropout'] = params['dropout']
    args['num_layers'] = params['num_layers']
    args['alpha'] = params['alpha']
    args['beta'] = params['beta']
    args['theta'] = params['theta']
    args['attention_method'] = params['attention_method']

    val_mean, _, test_mean, _ = run(args, verbose = False)

    return test_mean


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic.
    parser.add_argument('--model', type = str, default = '', help = 'Model to use.')
    parser.add_argument('--dataset', type = str, default = '', help = 'Dataset to use.')

    # Training.
    parser.add_argument('--seed', type = int, default = 42, help = 'Random seed.')
    parser.add_argument('--device', type = str, default = 'cuda', help = 'Training device.')
    parser.add_argument('--times', type = int, default = 10, help = 'Training times.')
    parser.add_argument('--epochs', type = int, default = 1500, help = 'Number of epochs to train.')
    parser.add_argument('--early_stopping', type = int, default = 100, help = 'Early stopping.')
    parser.add_argument('--setting', type = str, default = '60/20/20', help = 'Training node setting (\'public\', \'GCN\', \'semi\', \'48/32/20\' or \'60/20/20\').')

    # Model.
    parser.add_argument('--hidden_dim', type = int, default = 64, help = 'Hidden dimension.')
    parser.add_argument('--initial', type = str, default = 'Linear', help = 'Initial embedding method for the MGNN model.')

    parser.add_argument('--trails', type = int, default = 500, help = 'Number of hyper-tuning trails.')

    args = vars(parser.parse_args())

    study = optuna.create_study(direction = 'maximize', sampler = optuna.samplers.TPESampler(), pruner = optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials = args['trails'])
