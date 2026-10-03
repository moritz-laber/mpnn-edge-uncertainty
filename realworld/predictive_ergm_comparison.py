"""
Predictive DC-SBM vs. CL comparison
"""

import os
import numpy as np

from training import load_data
from utils import predictive_loglik_split


seed = 1234
split = 0

test_fraction = 0.20

num_splits = 100

max_test_dyads = 5_000_000

eps = 1e-12

out_dir = './predictive_comparison'

components = {
    'actor': 64,
    'chameleon': 128,
    'citeseer': 32,
    'computers': 32,
    'cora': 128,
    'cornell': 64,
    'photo': 32,
    'pubmed': 32,
    'squirrel': 64,
    'texas': 32,
    'wisconsin': 64,
}


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
    'wisconsin',
]


def main():

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        seed
    )

    for dataset in datasets:

        print()
        print('=' * 60)
        print(dataset)
        print('=' * 60)

        # load data
        X, y, A, A_sp, train_idx, val_idx, test_idx, pca = load_data(
            f'./data/{dataset}.npz',
            components=components[dataset],
            split=split,
        )

        # predicted labels
        model_result = np.load(
            f'./results/realworld/results_{dataset}.npz'
        )

        groups = model_result[
            'predicted_groups'
        ].astype(int)

        results = []
        for s in range(num_splits):

            result = predictive_loglik_split(
                A_sp,
                groups,
                rng,
                test_fraction=test_fraction,
                max_test_dyads=max_test_dyads,
                eps=eps,
            )

            results.append(
                result
            )

            print(
                f'  split {s + 1:3d}: '
                f'S_dyad = {result["score_dyad"]:.8f}, '
                f'S_edge = {result["score_edge"]:.4f}, '
                f'test edges = {result["num_test_edges"]}, '
                f'test dyads = {result["num_test_dyads"]}'
            )


        score_dyad = np.asarray([
            result['score_dyad']
            for result in results
        ])

        score_edge = np.asarray([
            result['score_edge']
            for result in results
        ])

        score_dyad_edges = np.asarray([
            result['score_dyad_edges']
            for result in results
        ])

        score_dyad_nonedges = np.asarray([
            result['score_dyad_nonedges']
            for result in results
        ])

        ll_dcsbm = np.asarray([
            result['ll_dcsbm']
            for result in results
        ])

        ll_cl = np.asarray([
            result['ll_cl']
            for result in results
        ])

        delta_ll = np.asarray([
            result['delta_ll']
            for result in results
        ])

        delta_ll_edges = np.asarray([
            result['delta_ll_edges']
            for result in results
        ])

        delta_ll_nonedges = np.asarray([
            result['delta_ll_nonedges']
            for result in results
        ])

        num_test_edges = np.asarray([
            result['num_test_edges']
            for result in results
        ])

        num_test_nonedges = np.asarray([
            result['num_test_nonedges']
            for result in results
        ])

        num_test_dyads = np.asarray([
            result['num_test_dyads']
            for result in results
        ])

        num_zero_edges_dcsbm = np.asarray([
            result['num_zero_edges_dcsbm']
            for result in results
        ])

        num_zero_edges_cl = np.asarray([
            result['num_zero_edges_cl']
            for result in results
        ])

        num_one_nonedges_dcsbm = np.asarray([
            result['num_one_nonedges_dcsbm']
            for result in results
        ])

        num_one_nonedges_cl = np.asarray([
            result['num_one_nonedges_cl']
            for result in results
        ])

        assert np.allclose(
            score_dyad,
            score_dyad_edges
            + score_dyad_nonedges,
        )

        np.savez(
            os.path.join(
                out_dir,
                f'predictive_loglik_dcsbm_vs_cl_{dataset}.npz',
            ),

            # scores
            score_dyad=score_dyad,
            score_edge=score_edge,
            score_dyad_edges=score_dyad_edges,
            score_dyad_nonedges=score_dyad_nonedges,

            # likelihoods
            ll_dcsbm=ll_dcsbm,
            ll_cl=ll_cl,
            delta_ll=delta_ll,
            delta_ll_edges=delta_ll_edges,
            delta_ll_nonedges=delta_ll_nonedges,

            # test set
            num_test_edges=num_test_edges,
            num_test_nonedges=num_test_nonedges,
            num_test_dyads=num_test_dyads,

            # diagnostics
            num_zero_edges_dcsbm=num_zero_edges_dcsbm,
            num_zero_edges_cl=num_zero_edges_cl,
            num_one_nonedges_dcsbm=num_one_nonedges_dcsbm,
            num_one_nonedges_cl=num_one_nonedges_cl,
        )


if __name__ == '__main__':
    main()