# python main.py --model MGNN --dataset Cora --lr 1e-2 --weight_decay 1e-3 --dropout 0.5 --num_layers 8 --alpha 0.15 --beta 0.98 --theta 1.0 --attention_method concat
# python main.py --model MGNN --dataset CiteSeer --lr 5e-3 --weight_decay 1e-2 --dropout 0.3 --num_layers 4 --alpha 0.32 --beta 0.0 --theta 0.5 --attention_method concat
# python main.py --model MGNN --dataset PubMed --lr 1e-2 --weight_decay 1e-3 --dropout 0.5 --num_layers 4 --alpha 0.39 --beta 0.34 --theta 1.5 --attention_method bilinear --initial MLP
# python main.py --model MGNN --dataset CoraFull --lr 5e-2 --weight_decay 5e-5 --dropout 0.1 --num_layers 8 --alpha 0.12 --beta 0.16 --theta 0.5 --attention_method concat
# python main.py --model MGNN --dataset CS --lr 1e-2 --weight_decay 5e-5 --dropout 0.5 --num_layers 4 --alpha 0.46 --beta 0.82 --theta 1.0 --attention_method concat
# python main.py --model MGNN --dataset Physics --lr 1e-2 --weight_decay 1e-5 --dropout 0.3 --num_layers 8 --alpha 0.3 --beta 0.81 --theta 0.5 --attention_method concat --device cuda:1
# python main.py --model MGNN --dataset Cornell --lr 5e-2 --weight_decay 5e-3 --dropout 0.5 --num_layers 2 --alpha 0.72 --beta 1.00 --theta 0.5 --attention_method concat
# python main.py --model MGNN --dataset Texas --lr 5e-2 --weight_decay 1e-2 --dropout 0.1 --num_layers 2 --alpha 0.70 --beta 0.0 --theta 0.5 --attention_method concat
# python main.py --model MGNN --dataset Wisconsin --lr 1e-2 --weight_decay 5e-3 --dropout 0.5 --num_layers 2 --alpha 0.81 --beta 0.0 --theta 1.5 --attention_method concat --initial MLP --device cuda:1 -> 88.20
# python main.py --model MGNN --dataset Wisconsin --lr 1e-2 --weight_decay 1e-2 --dropout 0.5 --num_layers 2 --alpha 0.73 --beta 0.00 --theta 1.5 --attention_method concat --initial MLP --device cuda:1 -> 88.40
# python main.py --model MGNN --dataset Chameleon --lr 1e-2 --weight_decay 5e-4 --dropout 0.0 --num_layers 2 --alpha 0.01 --beta 0.00 --theta 1.0 --attention_method bilinear --initial MLP --device cuda:1
# python main.py --model MGNN --dataset Squirrel --lr 1e-3 --weight_decay 1e-4 --dropout 0.0 --num_layers 8 --alpha 0.94 --beta 0.9 --theta 1.5 --attention_method bilinear --device cuda:1
# python main.py --model MGNN --dataset Actor --lr 5e-2 --weight_decay 0.0 --dropout 0.1 --num_layers 8 --alpha 0.02 --beta 0.28 --theta 1.5 --attention_method concat
# python main.py --model MGNN --dataset WikiCS --lr 1e-2 --weight_decay 5e-5 --dropout 0.5 --num_layers 2 --alpha 0.09 --beta 0.12 --theta 1.0 --attention_method concat --device cuda:1
# python main.py --model MGNN --dataset ogbn-arxiv --lr 5e-3 --weight_decay 0.0 --dropout 0.1 --num_layers 4 --alpha 0.12 --beta 0.63 --attention_method concat --initial MLP --device cuda:1
