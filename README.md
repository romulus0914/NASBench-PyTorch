# NASBench PyTorch
A PyTorch implementation of *training* of NASBench

# Overview
A PyTroch implementation of *training* of NASBench dataset: [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://arxiv.org/abs/1902.09635).
NASBench dataset contains 423,624 unique neural networks exhaustively generated and evaluated from a fixed graph-based search space.

# Architecture
![archtecture](./assets/architecture.png)

# Results
![param-time-acc](./assets/param_time_acc.png)

# Disclaimer
Modified from [NASBench: A Neural Architecture Search Dataset and Benchmark](https://github.com/google-research/nasbench).
*graph_util.py* and *model_spec.py* are directly copied from the original repo.
**Please noted that this repo is only used to train one possible architecture in the search space, not to generate all possible graphs and trained**
