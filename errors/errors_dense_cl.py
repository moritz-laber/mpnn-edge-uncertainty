"""
DENSE CL
"""

import numpy as np
from graphs import connection_probability_cl
from utils import marginal_wasserstein_errors, sigma_errors, mean_errors
from utils import m, d, Sig
import os

# parameters
output_dir = './theory/cl'
ns = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

seed = 1234
f = 16
shift = 5

def main():
    
    for n in ns:

        print(n)

        
        # resamble the features
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, f))
        
        # load the preactivations and weight
        result = np.load(f'./results/cl/dense/results_n={n}.npz')
        hs = result['preactivations_1']
        w = result['weight_0']
        kappas = result['kappas']

        # compute the connection probability
        p = connection_probability_cl(kappas)
        np.fill_diagonal(p, 1.0)

        # compute the theory for the first layer preactivations
        ms = np.dot(m(p, X), w)
        Sigs = np.einsum('ab, iac, cd -> ibd', w, Sig(p,X), w)
        sigs = np.sqrt(np.diagonal(Sigs, axis1=-2, axis2=-1))

        # compute the errors in the means (for each node)
        errors_mean = mean_errors(hs, ms)

        # compute the errors in the covariance matrix (for each node)
        errors_sig = sigma_errors(hs, Sigs)
        
        # average over nodes and features
        errors_wass = marginal_wasserstein_errors(hs, ms, Sigs)

        # save
        np.savez(
            
            f'{output_dir}/dense/theory_and_errors_n={n}.npz',     
            
            # node-wise errors: (n,)
            errors_mean=errors_mean,
            errors_sig=errors_sig,
            errors_wass=errors_wass,

            # theoretical statistics: (n, f)
            means_theory=ms,
            sigs_theory=sigs,
        )

if __name__ == '__main__':
    main()