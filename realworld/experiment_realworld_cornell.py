"""
experiment_dcsbm_real_network.py

Load a trained GCN on a real network, fit a DC-SBM to the observed graph,
copy the trained JAX weights into the NumPy implementation, and evaluate
the model on graphs sampled from the fitted DC-SBM.
"""

import os
import numpy as np
import scipy as sp
import jax
import jax.numpy as jnp
import equinox as eqx

from training import load_data
from models_jax import RWGCN as JaxRWGCN
from models_numpy import RWGCN as NumpyRWGCN
from graphs import sample_dcsbm, edgelist_to_adjacency, fit_dcsbm


# PARAMETERS
dataset_name = "cornell"

components = 64
num_graphs = 1000
path_to_data = f"./data/{dataset_name}.npz"
model_path = f"./models/{dataset_name}.eqx"

model_seed = 1014
split = 0
num_hidden = 64
use_bias = False
seed = 1234

out_dir = "./results/realworld/"
verbose = True

def main():

    rng = np.random.default_rng(seed)

    print("load data")
    X, y, A, A_sp, train_idx, val_idx, test_idx, pca = load_data(
        path_to_data,
        components=components,
        split=split
    )

    # Convert data used by NumPy model
    X = np.asarray(X)
    y = np.asarray(y)

    # SciPy adjacency used to fit DC-SBM
    A_sp = A_sp.tocsr()

    n = X.shape[0]
    num_features = X.shape[1]
    num_classes = int(np.max(y)) + 1

    features = [
        num_features,
        num_hidden,
        num_hidden,
        num_classes
    ]

    print(f"n = {n}")
    print(f"num features = {num_features}")
    print(f"num classes = {num_classes}")

    print("load trained model")
    
    model_key = jax.random.PRNGKey(model_seed)
    jax_model = JaxRWGCN(
        model_key,
        features=features,
        activations=[
            jax.nn.relu,
            jax.nn.relu,
            jax.nn.identity
        ],
        use_bias=use_bias
    )

    jax_model = eqx.tree_deserialise_leaves(
        model_path,
        jax_model
    )

    numpy_model = NumpyRWGCN(
        rng=np.random.default_rng(model_seed),
        features=features,
        activations=[
            lambda x : np.maximum(0.0, x),
            lambda x : np.maximum(0.0, x),
            lambda x: x
        ],
        use_bias=use_bias
    )

    for jax_layer, numpy_layer in zip(jax_model.layers, numpy_model.layers):

        numpy_layer.weight = np.asarray(jax_layer.weight).copy()
        if jax_model.use_bias:
            numpy_layer.bias = np.asarray(jax_layer.bias).copy()
        else:
            numpy_layer.bias = np.zeros(jax_layer.weight.shape[1])

    ### make one forward pass on cora ###
    pre, emb = numpy_model(X, A_sp)
    logits = emb[-1]
    pred_realworld = np.argmax(logits, axis=-1)
    acc_train_orig = np.mean(pred_realworld[train_idx] == y[train_idx])
    acc_test_orig = np.mean(pred_realworld[test_idx] == y[test_idx])
    acc_val_orig = np.mean(pred_realworld[val_idx] == y[val_idx])

    save_dict = {
        'accuracy_train_realworld' : acc_train_orig,
        'accuracy_test_realworld' : acc_test_orig,
        'accuracy_val_realworld' : acc_val_orig,
        'predicted_groups' : pred_realworld
    }

    for l, x in enumerate(pre):
        save_dict[f"preactivations_{l}_realworld"] = x
    for l, x in enumerate(emb):
        save_dict[f"embeddings_{l}_realworld"] = x
    for l, layer in enumerate(numpy_model.layers):
        save_dict[f"weight_{l}"] = layer.weight

    print("fit DC-SBM")
    groups = y.astype(int)

    kappa, B = fit_dcsbm(
        A_sp,
        pred_realworld
    )

    print("mean degree:", kappa.mean())
    print("B:")
    print(B)
    
    ### SAMPLING ###
    num_layers = len(features) - 1

    preactivations = [[] for _ in range(num_layers + 1)]
    embeddings = [[] for _ in range(num_layers + 1)]
    accuracies_test = []
    accuracies_train = []
    accuracies_val = []
    for ig in range(num_graphs):

        if verbose:
            print(
                f"Graph {ig + 1}/{num_graphs}",
                end="\r"
            )

        # sample graph from fitted DC-SBM
        edges = sample_dcsbm(
            rng,
            kappa,
            pred_realworld,
            B
        )
        
        A_sample = edgelist_to_adjacency(
            n,
            edges
        ).tocsr()

        # run the trained model
        pre, emb = numpy_model(
            X,
            A_sample
        )

        logits = emb[-1]
        pred = np.argmax(logits, axis=-1)
        accuracies_train.append(np.mean(pred[train_idx] == y[train_idx]))
        accuracies_test.append(np.mean(pred[test_idx] == y[test_idx]))
        accuracies_val.append(np.mean(pred[val_idx] == y[val_idx]))

        for l in range(num_layers + 1):
            preactivations[l].append(pre[l])
            embeddings[l].append(emb[l])

    if verbose:
        print()


    # ------------------------------------------------------------
    # Stack results
    # ------------------------------------------------------------
    preactivations = [
        np.stack(x, axis=0)
        for x in preactivations
    ]

    embeddings = [
        np.stack(x, axis=0)
        for x in embeddings
    ]


    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    os.makedirs(
        out_dir,
        exist_ok=True
    )
    
    save_dict["kappa"] = kappa
    save_dict["B"] = B
    save_dict["groups"] = groups

    for l, x in enumerate(preactivations):
        save_dict[f"preactivations_{l}"] = x

    for l, x in enumerate(embeddings):
        save_dict[f"embeddings_{l}"] = x

    save_dict["accuracy_train"] = np.asarray(accuracies_train)
    save_dict["accuracy_test"] = np.asarray(accuracies_test)
    save_dict["accuracy_val"] = np.asarray(accuracies_val)

    np.savez(
        os.path.join(
            out_dir,
            f"results_{dataset_name}.npz"
        ),
        **save_dict
    )

    print("saved")


if __name__ == "__main__":
    main()