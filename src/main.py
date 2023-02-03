from argparse import ArgumentParser
from run import run

if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic.
    parser.add_argument('--model', type = str, default = 'MGNN', help = 'Model to use.')
    parser.add_argument('--dataset', type = str, default = 'Cora', help = 'Dataset to use.')

    # Training.
    parser.add_argument('--seed', type = int, default = 42, help = 'Random seed.')
    parser.add_argument('--device', type = str, default = 'cuda', help = 'Training device.')
    parser.add_argument('--times', type = int, default = 10, help = 'Training times.')
    parser.add_argument('--epochs', type = int, default = 1500, help = 'Number of epochs to train.')
    parser.add_argument('--early_stopping', type = int, default = 100, help = 'Early stopping.')
    parser.add_argument('--lr', type = float, default = 1e-2, help = 'Learning rate.')
    parser.add_argument('--weight_decay', type = float, default = 1e-4, help = 'Weight decay.')
    parser.add_argument('--dropout', type = float, default = 0.5, help = 'Dropout rate.')
    parser.add_argument('--setting', type = str, default = '60/20/20', help = 'Training node setting (\'public\', \'GCN\', \'semi\', \'48/32/20\' or \'60/20/20\').')

    # Model.
    parser.add_argument('--hidden_dim', type = int, default = 64, help = 'Hidden dimension.')
    parser.add_argument('--num_layers', type = int, default = 2, help = 'Number of convolution layers or propagation hops.')
    parser.add_argument('--alpha', type = float, default = 0.5, help = 'Alpha value for the propagation.')
    parser.add_argument('--beta', type = float, default = 0.5, help = 'Beta value for the propagation.')
    parser.add_argument('--theta', type = float, default = 0.5, help = 'Theta value for the propagation.')
    parser.add_argument('--attention_method', type = str, default = 'concat', help = 'Attention method for the MGNNAttention layer.')
    parser.add_argument('--initial', type = str, default = 'Linear', help = 'Initial embedding method for the MGNN model.')

    args = vars(parser.parse_args())

    run(args = args, verbose = True)