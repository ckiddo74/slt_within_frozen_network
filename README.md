# Partially Frozen Random Networks Contain Compact Strong Lottery Tickets

We search for compact SLTs within frozen networks.

## Installation
- CNN experiment
```bash
cd image_classification
conda env create -f requirement.yml
```
- GNN experiment
```bash
cd node_classification
conda env create -f requirement.yml
```

## Experiments (Example)

- Conv6 & CIFAR-10 w/ 50% freezing ratio (25% pruning ratio:25% locking ratio)
```bash
python3 main.py \
--config configs/cifar10-conv6-global_ep-0.5-epl-p_0.25-r_0.25.yaml \
--dataset_dir "your cifar-10 directory"
```
- GIN & OGBN-Arxiv w/ 50% freezing ratio (0% pruning ratio:40% locking ratio)
```bash
python3 main.py \
--config ogbn_arxiv-gin_based_net-global_ep-0.2-epl_p_0.0-r_0.4.yaml \
--dataset_dir "your OGBN-Arxiv directory"
```

## Options
Please see `utils/parser.py`
