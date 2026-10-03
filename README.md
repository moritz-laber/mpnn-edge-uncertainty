# Code for the paper MPNNs on Random Graphs: Quantifying the Effects of Edge Uncertainty on Node Embeddings


## Setup

Experiments were run with `python 3.12.2`. You can install the required packages in the correct version using `requirements.txt`. The user is responsible for correctly adapting paths in all scripts and utility files.

## Experiments on synthetic graphs

Experiments for synthetic graphs from random graph models requires extracting the required utility files from `/utilities` and running in the following order all scripts in

1. `/experiments`, this samples random graphs from the random graph models and conduct forward passes of the MPNN.
2. `/errors`, this compares the empirical and analytical preactivation distributions.
3. `/determinitic`, this evaluates the deterministic forward pass.

## Experiments on real-world graphs

Experiments on real-world graphs requires unziping the datasets in `data.zip`, extracting the required utility files from `/utilities` and running in teh following order all scripts in

1. `/realworld/train_*.py`, trains and saves the MPNN models.
2. `/realworld/experiment_*.py`, samples the random graph models and conducts forward passes.
3. `/realworld/predictive_ergm_comparison.py`, compares DC-SBM and CL on held-out dyads.

## Plots

Plotting requires extracting the required utility files from `/utilities` and running the relevant scripts from `/plots`.