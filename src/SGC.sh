python main.py --model SGC --dataset Cora --lr 1e-2 --weight_decay 1e-3 --dropout 0.5 --num_layers 2
python main.py --model SGC --dataset CiteSeer --lr 1e-2 --weight_decay 5e-3 --dropout 0.3 --num_layers 2
python main.py --model SGC --dataset PubMed --lr 5e-2 --weight_decay 1e-5 --dropout 0.5 --num_layers 2
python main.py --model SGC --dataset CoraFull --lr 5e-2 --weight_decay 1e-4 --dropout 0.0 --num_layers 2 --device cuda:1
python main.py --model SGC --dataset CS --lr 5e-2 --weight_decay 1e-4 --dropout 0.1 --num_layers 2 --device cuda:1
python main.py --model SGC --dataset Cornell --lr 1e-2 --weight_decay 5e-3 --dropout 0.3 --num_layers 2
python main.py --model SGC --dataset Texas --lr 5e-3 --weight_decay 5e-5 --dropout 0.5 --num_layers 2
python main.py --model SGC --dataset Wisconsin --lr 5e-2 --weight_decay 1e-2 --dropout 0.5 --num_layers 2
python main.py --model SGC --dataset Chameleon --lr 5e-2 --weight_decay 0.0 --dropout 0.0 --num_layers 2
python main.py --model SGC --dataset Squirrel --lr 5e-2 --weight_decay 0.0 --dropout 0.1 --num_layers 2
python main.py --model SGC --dataset Actor --lr 1e-2 --weight_decay 1e-4 --dropout 0.5 --num_layers 2
python main.py --model SGC --dataset WikiCS --lr 5e-2 --weight_decay 0.0 --dropout 0.3 --num_layers 2
python main.py --model SGC --dataset ogbn-arxiv --lr 5e-2 --weight_decay 0.0 --dropout 0.0 --num_layers 4 --device cuda:1