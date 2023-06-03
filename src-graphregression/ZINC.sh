python main.py --model GCN --dataset ZINC --lr 5e-3 --weight_decay 1e-4 --dropout 0.0 --num_layers 8 # -> 0.6143 +- 0.0200
python main.py --model GAT --dataset ZINC --lr 1e-3 --weight_decay 0.0 --dropout 0.0 --num_layers 8 # -> 0.6046 +- 0.0167
python main.py --model GIN --dataset ZINC --lr 1e-3 --weight_decay 1e-4 --dropout 0.1 --num_layers 8 # -> 0.4410 +- 0.0065
python main.py --model PointNet --dataset ZINC --lr 5e-3 --weight_decay 1e-5 --dropout 0.0 --num_layers 8 # -> 0.5293 +- 0.0138
python main.py --model MGNN --dataset ZINC --lr 1e-3 --weight_decay 0.0 --dropout 0.0 --num_layers 8 --alpha 0.35 --beta 0.66 --theta 1.5 --attention_method bilinear # -> 0.5074 +- 0.0055
python main.py --model MGNN --dataset ZINC --lr 5e-3 --weight_decay 0.0 --dropout 0.0 --num_layers 4 --alpha 0.37 --beta 0.65 --theta 1.5 --attention_method bilinear --initial MLP # -> 0.4751 +- 0.0112
