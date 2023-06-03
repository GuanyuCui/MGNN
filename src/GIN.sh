python main.py --model GIN --dataset Cora --lr 5e-3 --weight_decay 1e-5 --dropout 0.5 --num_layers 2 # -> 85.76 +- 1.18
python main.py --model GIN --dataset CiteSeer --lr 1e-3 --weight_decay 1e-2 --dropout 0.5 --num_layers 2 # -> 72.83 +- 1.47
python main.py --model GIN --dataset PubMed --lr 1e-2 --weight_decay 1e-3 --dropout 0.5 --num_layers 2 # -> 87.15 +- 0.43
python main.py --model GIN --dataset CoraFull --lr 1e-3 --weight_decay 1e-5 --dropout 0.3 --num_layers 2 # -> 67.11 +- 0.46
python main.py --model GIN --dataset CS --lr 5e-3 --weight_decay 5e-3 --dropout 0.5 --num_layers 2 # -> 91.81 +- 0.34
python main.py --model GIN --dataset Physics --lr 5e-3 --weight_decay 5e-4 --dropout 0.3 --num_layers 2 --device cuda:1 # -> 94.66 +- 2.04
python main.py --model GIN --dataset Cornell --lr 1e-3 --weight_decay 1e-4 --dropout 0.1 --num_layers 2 # -> 49.46 +- 7.46
python main.py --model GIN --dataset Texas --lr 1e-3 --weight_decay 1e-2 --dropout 0.0 --num_layers 2 # -> 64.05 +- 4.20
python main.py --model GIN --dataset Wisconsin --lr 1e-3 --weight_decay 1e-2 --dropout 0.1 --num_layers 2 # -> 57.20 +- 6.82
python main.py --model GIN --dataset Chameleon --lr 1e-3 --weight_decay 1e-3 --dropout 0.5 --num_layers 2 # -> 46.73 +- 13.51
python main.py --model GIN --dataset Squirrel --lr 1e-3 --weight_decay 1e-3 --dropout 0.0 --num_layers 2 # -> 20.96 +- 2.08
python main.py --model GIN --dataset Actor --lr 1e-3 --weight_decay 1e-4 --dropout 0.0 --num_layers 2 # -> 26.09 +- 1.75
python main.py --model GIN --dataset WikiCS --lr 1e-3 --weight_decay 0.0 --dropout 0.3 --num_layers 2 # -> 66.08 +- 22.77
python main.py --model GIN --dataset ogbn-arxiv --lr 1e-3 --weight_decay 1e-5 --dropout 0.1 --num_layers 2 --device cuda:1 # -> 66.60 +- 0.53