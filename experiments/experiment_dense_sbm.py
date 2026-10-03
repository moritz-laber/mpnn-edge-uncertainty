"""
experiment_sparse_sbm.py

This script is for experiments on SBM(n,g,P) where
- the number of nodes n diverges
- the number of groups is constant at 2
- the connection probability p_ab ~ 1
"""

import numpy as np
import os
from graphs import sample_sbm, edgelist_to_adjacency
from models_numpy import RWGCN
import scipy.special as special
import sys

# PARAMETERS
num_graphs = 1000                                    # number of graphs to generate.
n = int(sys.argv[1])                                 # number of nodes.
P = np.asarray(                                      # connection probability
    [[0.04, 0.01],
     [0.01, 0.04]]
)

groups = np.asarray(                                 # group assignment
    [0 if i<n/2 else 1 for i in range(n)]
)

shift = 5
features = [16, 16, 16, 16]                    # number of features including input and output. Saving assumed that these are the same across layers.
activation = lambda x: np.maximum(0.0, x)      # activation function applied after each layer.
use_bias = False                               # whether to include a bias term in the linear layers.

out_dir = "./results/sbm/dense/"              # output directory: outputs will be stored at out_dir/results_n=n_p=p.npz.
seed = 1234                                    # seed for random number generator for features and graphs
model_seed = 5678                              # seed for random number generator for model weights
verbose = True                                 # whether to print progress information during the experiment.

def main():

    # create a random generators
    rng = np.random.default_rng(seed=seed)
    model_rng = np.random.default_rng(seed=model_seed)

    # sample the input features
    X = rng.normal(size=(n, features[0]))
    X[groups == 0, :] -= shift
    X[groups == 1, :] += shift

    # initialize the GCN model
    model = RWGCN(
        model_rng,
        features=features,
        activations=(len(features) - 1) * [activation],
        use_bias=use_bias
    )
    
    preactivations = [[] for _ in range(len(features))]
    embeddings = [[] for _ in range(len(features))]
    for ig in range(num_graphs):

        # progress
        if verbose: print(f"Graph {ig+1}/{num_graphs}", end="\r", flush=True)
        
        # sample a new random graph
        edges = sample_sbm(rng, groups, P)
        A = edgelist_to_adjacency(n, edges)

        # compute embeddings and preactivations at each layer of the GCN
        pre, emb = model(X, A)
        for l in range(len(features)):
            preactivations[l].append(pre[l])
            embeddings[l].append(emb[l])
    
    # create single arrays of embeddings and preactivations across all graphs and layers
    preactivations = [np.stack(pre, axis=0) for pre in preactivations]
    embeddings = [np.stack(emb, axis=0) for emb in embeddings]

    # save the preactivations and embeddings
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir)
        except OSError as e:
            print(f"Error creating directory {out_dir}: {e}")

    save_dict = {}
    for l, x in enumerate(preactivations):
        save_dict[f"preactivations_{l}"] = x
    
    for l, x in enumerate(embeddings):
        save_dict[f"embeddings_{l}"] = x

    for l, layer in enumerate(model.layers):
        save_dict[f"weight_{l}"] = layer.weight

    save_dict['X'] = X
    save_dict['P'] = P
        
    np.savez(
        os.path.join(out_dir, f"results_n={n}.npz"),
        **save_dict,
    )

if __name__ == "__main__":
    main()