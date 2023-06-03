python main.py --model GAT --dataset Cora --lr 1e-2 --weight_decay 5e-3 --dropout 0.5 --num_layers 2
python main.py --model GAT --dataset CiteSeer --lr 1e-2 --weight_decay 1e-2 --dropout 0.3 --num_layers 2
python main.py --model GAT --dataset PubMed --lr 1e-2 --weight_decay 1e-4 --dropout 0.3 --num_layers 2
python main.py --model GAT --dataset CoraFull --lr 5e-3 --weight_decay 1e-4 --dropout 0.5 --num_layers 2
python main.py --model GAT --dataset CS --lr 1e-2 --weight_decay 1e-4 --dropout 0.1 --num_layers 2
python main.py --model GAT --dataset Physics --lr 1e-2 --weight_decay 1e-4 --dropout 0.3 --num_layers 2
python main.py --model GAT --dataset Cornell --lr 1e-2 --weight_decay 5e-5 --dropout 0.0 --num_layers 2
python main.py --model GAT --dataset Texas --lr 1e-3 --weight_decay 1e-5 --dropout 0.0 --num_layers 8
python main.py --model GAT --dataset Wisconsin --lr 1e-2 --weight_decay 1e-2 --dropout 0.1 --num_layers 2
python main.py --model GAT --dataset Chameleon --lr 1e-2 --weight_decay 5e-5 --dropout 0.0 --num_layers 2
python main.py --model GAT --dataset Squirrel --lr 5e-2 --weight_decay 5e-5 --dropout 0.1 --num_layers 2
python main.py --model GAT --dataset Actor --lr 1e-3 --weight_decay 5e-4 --dropout 0.1 --num_layers 2
python main.py --model GAT --dataset WikiCS --lr 1e-2 --weight_decay 1e-5 --dropout 0.3 --num_layers 2
python main.py --model GAT --dataset ogbn-arxiv --lr 1e-2 --weight_decay 1e-5 --dropout 0.1 --num_layers 4 --device cuda:1