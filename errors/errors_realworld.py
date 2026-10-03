"""
REALWORLD
"""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from training import load_data
from models_jax import RWGCN as JaxRWGCN
from models_numpy import RWGCN as NumpyRWGCN, deterministic_update
from graphs import connection_probability_dcsbm
from utils import marginal_wasserstein_errors, sigma_errors, mean_errors
from utils import m, d, Sig
import os

# parameters
output_dir = './theory/realworld'
model_path = './models'

seed = 1243
use_bias = False

split = 0
components = {
    'actor' : 64,
    'chameleon' : 128,
    'citeseer' : 32,
    'computers' : 32,
    'cora' : 128,
    'cornell' : 64,
    'photo' : 32,
    'pubmed' : 32,
    'squirrel' : 64,
    'texas' : 32,
    'wisconsin' : 64
}

num_hidden = 64

datasets = [
    'actor',
    'chameleon',
    'citeseer',
    'computers',
    'cora',
    'cornell',
    'photo',
    'pubmed',
    'squirrel',
    'texas',
    'wisconsin'
]

model_seed = 1014

def main():
    
    for dataset in datasets:

        print(dataset)
        
        # resamble the features
        X, y, A, A_sp, train_idx, val_idx, test_idx, pca = load_data(
            f'./data/{dataset}.npz',
            components=components[dataset],
            split=split
        )

        # load the models
        n = X.shape[0]
        num_features = X.shape[1]
        num_classes = int(np.max(y)) + 1
    
        features = [
            num_features,
            num_hidden,
            num_hidden,
            num_classes
        ]

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
            f'{model_path}/{dataset}.eqx',
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

        # load the preactivations
        result = np.load(f'./results/realworld/results_{dataset}.npz')
        hs = result['preactivations_1']
        kappas = result['kappa']
        B = result['B']
        groups = result['predicted_groups']

        # compute the connection probability
        p = connection_probability_dcsbm(groups, kappas, B)
        np.fill_diagonal(p, 1.0)

        # compute the theory for the first layer preactivations
        w = numpy_model.layers[0].weight.copy()
        ms = np.dot(m(p, X), w)
        Sigs = np.einsum('ab, iac, cd -> ibd', w, Sig(p,X), w)
        sigs = np.sqrt(np.diagonal(Sigs, axis1=-2, axis2=-1))

        # compute the errors in the means (for each node)
        errors_mean = mean_errors(hs, ms)

        # compute the errors in the covariance matrix (for each node)
        errors_sig = sigma_errors(hs, Sigs)
        
        # average over nodes and features
        errors_wass = marginal_wasserstein_errors(hs, ms, Sigs)

        # difference between real-world and x
        pre_error = []
        emb_error = []
        det_pre_error = []
        det_emb_error = []
        pre_rand = [None, result['preactivations_1'], result['preactivations_2'], result['preactivations_3']]
        emb_rand = [None, result['embeddings_1'], result['embeddings_2'], result['embeddings_3']]
        
        # deterministic update
        det_pre, det_emb = deterministic_update(X, p, [layer.weight for layer in numpy_model.layers], [activation for activation in numpy_model.activations])
        
        for l in range(1,4):
            pre_error.append(np.linalg.norm(pre[l][None,:,:] - pre_rand[l], axis=-1))
            emb_error.append(np.linalg.norm(emb[l][None,:,:] - emb_rand[l], axis=-1))
            det_emb_error.append(np.linalg.norm(emb[l] - det_emb[l-1], axis=-1))
            det_pre_error.append(np.linalg.norm(pre[l] - det_pre[l-1], axis=-1))
        
        # mean error
        pre_error_m = np.linalg.norm(pre[1] - ms, axis=-1)

        # save
        np.savez(
            
            f'{output_dir}/theory_and_errors_{dataset}.npz',     
            
            # node-wise errors: (n,)
            errors_mean=errors_mean,
            errors_sig=errors_sig,
            errors_wass=errors_wass,

            # theoretical statistics: (n, f)
            means_theory=ms,
            sigs_theory=sigs,

            # emb errors
            emb_error_1 = emb_error[0],
            emb_error_2 = emb_error[1],
            emb_error_3 = emb_error[2],
            
            # pre
            pre_error_1 = pre_error[0],
            pre_error_2 = pre_error[1],
            pre_error_3 = pre_error[2],
            pre_error_m = pre_error_m,

            # det
            det_pre_error_1 = det_pre_error[0],
            det_pre_error_2 = det_pre_error[1],
            det_pre_error_3 = det_pre_error[2],
            det_emb_error_1 = det_emb_error[0],
            det_emb_error_2 = det_emb_error[1],
            det_emb_error_3 = det_emb_error[2],

            # 
            groups = groups,
            kappas = kappas
        
        )

if __name__ == '__main__':
    main()