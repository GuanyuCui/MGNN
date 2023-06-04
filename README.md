# MetricGNN (MGNN)

This repository contains the reference implementation code for the paper "MGNN: Graph Neural Networks Inspired by the Distance Geometry Problem" (KDD 2023).

Note: Our experiments were conducted on a server equipped with a TITAN V GPU and a Quadro RTX 8000 GPU, with CUDA version 10.3.
The hyperparameters used in our experiments may depend on the specific machine configuration and could vary with different versions of PyTorch due to the random number generator.
You may need to adjust these parameters when running the code on different machines.

## Package Requirements
- matplotlib==3.5.1
- networkx==2.8
- numpy==1.21.4
- ogb==1.3.6
- torch==1.8.1+cu101
- torch_geometric==2.1.0.post1
- torch_scatter==2.0.7
- torch_sparse==0.6.11
- tqdm==4.62.3

## Basic Usage
```bash
cd src 
```
or 
```bash
cd src-graphregression
```
for graph regression experiments. Then type:
```bash
python main.py (arguments)
```

## Run Experiments
To run our experiments, please follow the steps below.

- Install requirements.
```bash
pip install -r requirements.txt
```

- Run the 'Arranging Nodes with the Given Metric Matrices' experiment.
```bash
python synthetic.py
```
Then the output figures are generated in the 'out' folder.

- Run the 'Supervised Node Classification' experiment.
```bash
cd src
chmod +x *.sh
```
Then run the shell script of any specific model.

- Run the 'Graph Regression' experiment.
```bash
cd src-graphregression
chmod +x ZINC.sh
./ZINC.sh
```

## Credits

The code of the pGNN model is borrowed from the [official implementation](https://github.com/guoji-fu/pGNNs).
