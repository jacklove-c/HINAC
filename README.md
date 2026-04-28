# HINAC: Representation Learning On Heterogeneous Information Networks with Graph Transformer



## 1. Descriptions
The repository is organised as follows:

- dataset/: the original data of four benchmark dataset.
- run.py: multi-class node classificaiton of HINAC.
- run_multi.py: multi-label node classification of HINAC on IMDB.
- model.py: implementation of HINAC.
- utils/: contains tool functions.


## 2. Requirements

- Python==3.9.0
- Pytorch==1.12.0
- Networkx==2.8.4
- numpy==1.22.3
- dgl==0.9.0
- scikit-learn==1.1.1
- scipy==1.7.3
- openai
- sentence_transformers

## 3. Running experiments

We train our model using NVIDIA 3090 GPU.

For node classification with offline evaluation:
- python run.py --dataset DBLP --len-seq 50 --dropout 0.5 --beta 0.1 --temperature 2 --num-hade-layers 5
- python run_multi.py --dataset IMDB --len-seq 20 --beta 0.1 --temperature 0.1
- python run.py --dataset Freebase --num-gnns 3 --len-seq 30 --num-layers 3 --dropout 0 --beta 0.5 --temperature 0.2
- python run.py --dataset AMiner --len-seq 80 --num-gnns 3 --num-layers 4 --temperature 0.5
- python run.py --dataset ACM  --num-gnns 2 --num-layers 4 --temperature 1.5 --beta 0.5 --dropout 0.25 --num-heads 2 --feats-type 0

For reproducing our results in the paper and applying HINAC to other datasets, you need to tune the values of key parameters like 'num-gnns','num-layers','len-seq', 'dropout', 'temperature', 'beta' and 'num-hade-layers' in your experimental environment. 
## 4. Citation
