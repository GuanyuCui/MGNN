python main.py --model PointNet --dataset Cora --lr 5e-3 --weight_decay 0.0 --dropout 0.5 --num_layers 2 # -> 84.43 +- 1.94
python main.py --model PointNet --dataset CiteSeer --lr 5e-3 --weight_decay 0.0 --dropout 0.5 --num_layers 2 # -> 72.83 +- 1.38
python main.py --model PointNet --dataset PubMed --lr 5e-3 --weight_decay 1e-4 --dropout 0.5 --num_layers 2 # -> 89.18 +- 0.38
python main.py --model PointNet --dataset CoraFull --lr 5e-3 --weight_decay 0.0 --dropout 0.5 --num_layers 2 # -> 63.32 +- 1.02
python main.py --model PointNet --dataset CS --lr 1e-2 --weight_decay 5e-4 --dropout 0.3 --num_layers 2 # -> 93.13 +- 0.42
python main.py --model PointNet --dataset Physics --lr 5e-3 --weight_decay 5e-4 --dropout 0.0 --num_layers 2 # -> 96.37 +- 0.27
python main.py --model PointNet --dataset Cornell --lr 1e-2 --weight_decay 1e-3 --dropout 0.3 --num_layers 2 # -> 71.08 +- 5.93
python main.py --model PointNet --dataset Texas --lr 5e-3 --weight_decay 1e-4 --dropout 0.3 --num_layers 8 # -> 82.16 +- 6.30
python main.py --model PointNet --dataset Wisconsin --lr 1e-2 --weight_decay 5e-3 --dropout 0.5 --num_layers 2 # -> 81.60 +- 3.98
python main.py --model PointNet --dataset Chameleon --lr 5e-3 --weight_decay 5e-3 --dropout 0.1 --num_layers 2 # -> 63.76 +- 2.52
python main.py --model PointNet --dataset Squirrel --lr 1e-3 --weight_decay 5e-3 --dropout 0.1 --num_layers 2 # -> 47.39 +- 8.31
python main.py --model PointNet --dataset Actor --lr 1e-3 --weight_decay 5e-5 --dropout 0.0 --num_layers 4 # -> 36.41 +- 1.23
python main.py --model PointNet --dataset WikiCS --lr 5e-3 --weight_decay 1e-5 --dropout 0.3 --num_layers 2 # -> 84.09 +- 0.86
python main.py --model PointNet --dataset ogbn-arxiv --lr 5e-3 --weight_decay 0.0 --dropout 0.1 --num_layers 4 --device cuda:1 # -> 69.89 +- 0.59