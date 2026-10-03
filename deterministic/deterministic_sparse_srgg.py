"""
SPARSE SBM
"""

import numpy as np
from graphs import connection_probability_srgg, average_degree_to_c
from models_numpy import deterministic_update
from utils import m
import os

# parameters
output_dir = './deterministic/srgg'
ns = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
seed = 1234
f = 16
beta = 1.9

def main():
    
    for n in ns:

        print(n)
        
        # resamble the features
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, f))
        
        # load the preactivations and weight
        result = np.load(f'./results/srgg/sparse/results_n={n}.npz')
        emp_embeddings = [result['embeddings_1'], result['embeddings_2'], result['embeddings_3']]
        weights = [result['weight_0'], result['weight_1'], result['weight_2']]
        activations = [lambda x : np.maximum(0.0, x), lambda x : np.maximum(0.0, x), lambda x : np.maximum(0.0, x)]

        # compute the connection probability
        coords = result['positions']
        kbar = np.sqrt(n)
        c = average_degree_to_c(kbar, beta, num_nodes=n)
        p = connection_probability_srgg(coords, c, beta)
        np.fill_diagonal(p, 1.0)

        # compute the deterministic forward pass
        _, det_embeddings = deterministic_update(X, p, weights, activations)

        errors = []
        for l in range(0, 3):
            errors.append(np.linalg.norm(emp_embeddings[l] - det_embeddings[l][None, :, :], axis=-1))

        # save
        np.savez(
            
            f'{output_dir}/sparse/deterministic_n={n}.npz',

            det_emb_1 = det_embeddings[0],
            det_emb_2 = det_embeddings[1],
            det_emb_3 = det_embeddings[2],
            
            # node-wise errors: (n,)
            errors_1 = errors[0],
            errors_2 = errors[1],
            errors_3 = errors[2],
    
            # theoretical statistics: (n, f)
            means_theory=m(p,X),
        )

if __name__ == '__main__':
    main()