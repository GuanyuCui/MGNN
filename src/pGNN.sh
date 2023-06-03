python main.py --model pGNN --dataset Cora --lr 1e-3 --weight_decay 5e-3 --dropout 0.3 --num_layers 4 --alpha 0.08 --theta 2.0
python main.py --model pGNN --dataset CiteSeer --lr 5e-3 --weight_decay 5e-3 --dropout 0.1 --num_layers 2 --alpha 0.79 --theta 2.0
python main.py --model pGNN --dataset PubMed --lr 5e-3 --weight_decay 5e-5 --dropout 0.5 --num_layers 8 --alpha 0.63 --theta 1.5
python main.py --model pGNN --dataset CoraFull --lr 1e-3 --weight_decay 1e-3 --dropout 0.3 --num_layers 4 --alpha 0.08 --theta 1.0
python main.py --model pGNN --dataset CS --lr 1e-2 --weight_decay 1e-3 --dropout 0.1 --num_layers 4 --alpha 0.33 --theta 1.0
python main.py --model pGNN --dataset Physics --lr 5e-2 --weight_decay 5e-4 --dropout 0.5 --num_layers 4 --alpha 0.8 --theta 2.0
python main.py --model pGNN --dataset Cornell --lr 5e-2 --weight_decay 1e-2 --dropout 0.0 --num_layers 8 --alpha 0.9 --theta 0.5
python main.py --model pGNN --dataset Texas --lr 1e-2 --weight_decay 1e-2 --dropout 0.1 --num_layers 4 --alpha 0.94 --theta 1.0
python main.py --model pGNN --dataset Wisconsin --lr 1e-2 --weight_decay 5e-3 --dropout 0.3 --num_layers 2 --alpha 0.87 --theta 0.5
python main.py --model pGNN --dataset Chameleon --lr 1e-2 --weight_decay 0.0 --dropout 0.5 --num_layers 2 --alpha 0.00 --theta 1.5
python main.py --model pGNN --dataset Squirrel --lr 1e-2 --weight_decay 1e-4 --dropout 0.5 --num_layers 2 --alpha 0.00 --theta 0.5
python main.py --model pGNN --dataset Actor --lr 5e-2 --weight_decay 5e-3 --dropout 0.5 --num_layers 4 --alpha 1.0 --theta 2.0
python main.py --model pGNN --dataset WikiCS --lr 5e-3 --weight_decay 5e-5 --dropout 0.5 --num_layers 2 --alpha 0.09 --theta 1.5
python main.py --model pGNN --dataset ogbn-arxiv --lr 1e-2 --weight_decay 1e-4 --dropout 0.1 --num_layers 8 --alpha 0.22 --theta 2.0 --device cuda:1