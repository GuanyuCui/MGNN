python main.py --model GCNII --dataset Cora --lr 1e-3 --weight_decay 1e-5 --dropout 0.5 --num_layers 8 --alpha 0.17 --theta 0.5
python main.py --model GCNII --dataset CiteSeer --lr 5e-3 --weight_decay 1e-5 --dropout 0.3 --num_layers 8 --alpha 0.31 --theta 0.5
python main.py --model GCNII --dataset PubMed --lr 1e-2 --weight_decay 1e-3 --dropout 0.5 --num_layers 2 --alpha 0.35 --theta 0.5
python main.py --model GCNII --dataset CoraFull --lr 5e-3 --weight_decay 0.0 --dropout 0.3 --num_layers 8 --alpha 0.17 --theta 0.5
python main.py --model GCNII --dataset CS --lr 5e-3 --weight_decay 0.0 --dropout 0.5 --num_layers 8 --alpha 0.57 --theta 0.5
python main.py --model GCNII --dataset Physics --lr 1e-3 --weight_decay 5e-5 --dropout 0.5 --num_layers 2 --alpha 0.34 --theta 1.0
python main.py --model GCNII --dataset Cornell --lr 5e-3 --weight_decay 1e-3 --dropout 0.5 --num_layers 8 --alpha 0.96 --theta 1.5
python main.py --model GCNII --dataset Texas --lr 5e-2 --weight_decay 5e-3 --dropout 0.3 --num_layers 4 --alpha 0.96 --theta 1.0
python main.py --model GCNII --dataset Wisconsin --lr 1e-2 --weight_decay 5e-3 --dropout 0.5 --num_layers 2 --alpha 0.93 --theta 1.5
python main.py --model GCNII --dataset Chameleon --lr 1e-2 --weight_decay 1e-3 --dropout 0.0 --num_layers 2 --alpha 0.00 --theta 1.0
python main.py --model GCNII --dataset Squirrel --lr 5e-2 --weight_decay 1e-4 --dropout 0.1 --num_layers 2 --alpha 0.00 --theta 1.0
python main.py --model GCNII --dataset Actor --lr 5e-3 --weight_decay 1e-2 --dropout 0.3 --num_layers 8 --alpha 0.83 --theta 1.5
python main.py --model GCNII --dataset WikiCS --lr 1e-2 --weight_decay 5e-5 --dropout 0.3 --num_layers 2 --alpha 0.10 --theta 0.5
python main.py --model GCNII --dataset ogbn-arxiv --lr 1e-2 --weight_decay 1e-5 --dropout 0.1 --num_layers 8 --alpha 0.07 --theta 1.5 --device cuda:1