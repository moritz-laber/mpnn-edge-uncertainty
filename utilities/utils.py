import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
from graphs import edgelist_to_adjacency, fit_dcsbm, connection_probability_dcsbm_pairs, connection_probability_cl_pairs, bernoulli_loglik

def d(p):

    return 1.0 + np.sum(p, axis=1) - p.diagonal()

def m(p, X):

    ds = d(p)
    ms = np.einsum('ij, jk -> ik', p, X)
    ms /= ds[:, None]
    
    return ms

def Sig(p, X):
    
    ds = d(p)
    ms = m(p, X)

    R = p * (1.0 - p)
    np.fill_diagonal(R, 0.0)

    q = R @ X
    c = np.sum(R, axis=1)

    T = np.einsum(
        'ij,jd,je->ide',
        R, X, X,
        optimize=True
    )

    S = (
        T
        - np.einsum('id,ie->ide', q, ms)
        - np.einsum('id,ie->ide', ms, q)
        + np.einsum('i,id,ie->ide', c, ms, ms)
    )

    S /= ds[:, None, None]**2

    return S

def mean_errors(hs, ms):

    return np.linalg.norm(np.mean(hs, axis=0) - ms, axis=-1)

def sigma_errors(hs, Sigs):

    # compute the empirical covariance
    h_mean = np.mean(hs, axis=0)
    hc = hs - h_mean[None, :, :]
    Sig_emp = np.einsum('rif,rig->ifg', hc, hc) / (hs.shape[0] - 1)

    return np.linalg.norm(Sig_emp - Sigs, ord='fro', axis=(-2, -1))

def marginal_wasserstein_errors(hs, ms, Sigs):

    # determine number of repetitions
    R = hs.shape[0]

    # empirical quantiles
    h_sorted = np.sort(hs, axis=0)

    # theoretical quantiles
    q = stats.norm.ppf((np.arange(R) + 0.5) / R)

    vars_ = np.diagonal(Sigs, axis1=-2, axis2=-1)
    stds = np.sqrt(np.maximum(vars_, 0.0))
    
    h_theory = (
        ms[None, :, :]
        + q[:, None, None] * stds[None, :, :]
    )
    
    # compute W1 for every node and feature
    W1_marginals = np.mean(
        np.abs(h_sorted - h_theory),
        axis=0
    )

    return W1_marginals.mean(axis=-1)

def funcbin(x, y, nbins=12, q=(2.5, 97.5)):

    nonzero = x > 0
    x = x[nonzero]
    y = y[nonzero]

    bins = np.geomspace(np.min(x), np.max(x) + 1, num=nbins + 1)
    centers = np.sqrt(bins[:-1] * bins[1:])

    ymean = np.full(nbins, np.nan)
    ylo = np.full(nbins, np.nan)
    yhi = np.full(nbins, np.nan)

    for b in range(nbins):
        mask = (bins[b] <= x) & (x < bins[b + 1])

        if np.any(mask):
            ymean[b] = np.mean(y[mask])
            ylo[b], yhi[b] = np.percentile(y[mask], q)

    return centers, ymean, ylo, yhi

def plotsetup(texpath):

    os.environ["PATH"] = os.path.expanduser(texpath) + ":" + os.environ["PATH"]
    mpl.rcParams.update({
        "text.usetex": True,
    
        # Avoid Type 3 fonts from Matplotlib itself
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    
        # Use a scalable LaTeX font
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}\usepackage{amssymb}",
    })

def pair_to_rank(i, j):

    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)

    hi = np.maximum(i, j)
    lo = np.minimum(i, j)

    return hi * (hi - 1) // 2 + lo

def rank_to_pair(r):

    r = np.asarray(r, dtype=np.int64)

    j = np.floor(
        (1.0 + np.sqrt(1.0 + 8.0 * r)) / 2.0
    ).astype(np.int64)

    t = j * (j - 1) // 2

    # Correct rare floating-point rounding errors
    too_high = t > r
    while np.any(too_high):
        j[too_high] -= 1
        t = j * (j - 1) // 2
        too_high = t > r

    too_low = ((j + 1) * j // 2) <= r
    while np.any(too_low):
        j[too_low] += 1
        t = j * (j - 1) // 2
        too_low = ((j + 1) * j // 2) <= r

    i = r - t

    return i, j

def predictive_loglik_split(
    A,
    groups,
    rng,
    test_fraction=0.20,
    max_test_dyads=5_000_000,
    eps=1e-12,
):
    """
    Compare predictive Bernoulli log-likelihood of the DC-SBM
    against Chung-Lu on held-out dyads.

    Both models use the same fitted degree parameters kappa.
    Therefore, the comparison isolates the predictive information
    contributed by the block structure B_{g_i,g_j}.

    Returns both

        score_dyad =
            (LL_DCSBM - LL_CL) / number_of_test_dyads

    and

        score_edge =
            (LL_DCSBM - LL_CL) / number_of_test_edges.
    """

    A = A.tocsr()
    groups = np.asarray(groups, dtype=int)

    n = A.shape[0]

    # total number of possible unordered dyads
    num_dyads = (
        n * (n - 1) // 2
    )

    # ----------------------------------------------------------
    # Extract observed edges
    # ----------------------------------------------------------

    # Only take upper triangle so each undirected edge appears once.
    upper = sp.sparse.triu(
        A,
        k=1,
    ).tocoo()

    edges = np.column_stack(
        [
            upper.row,
            upper.col,
        ]
    ).astype(np.int64)

    # Map observed edges to unique dyad indices.
    edge_ranks = pair_to_rank(
        edges[:, 0],
        edges[:, 1],
    )

    # ----------------------------------------------------------
    # Sample held-out dyads
    # ----------------------------------------------------------

    num_test_dyads = min(
        int(round(
            test_fraction * num_dyads
        )),
        max_test_dyads,
    )

    num_test_dyads = max(
        num_test_dyads,
        1,
    )

    test_ranks = rng.choice(
        num_dyads,
        size=num_test_dyads,
        replace=False,
        shuffle=False,
    )

    # ----------------------------------------------------------
    # Construct training edge set and test labels
    # ----------------------------------------------------------

    # Which observed edges fall in the held-out dyads?
    is_test_edge = np.isin(
        edge_ranks,
        test_ranks,
    )

    train_edges = edges[
        ~is_test_edge
    ]

    # Test outcomes:
    # y_test = 1 -> observed edge
    # y_test = 0 -> observed nonedge
    y_test = np.isin(
        test_ranks,
        edge_ranks,
    ).astype(float)

    edge_mask = (
        y_test == 1
    )

    nonedge_mask = (
        y_test == 0
    )

    num_test_edges = int(
        edge_mask.sum()
    )

    num_test_nonedges = int(
        nonedge_mask.sum()
    )

    if num_test_edges == 0:
        raise RuntimeError(
            "No edges occurred in the test dyads."
        )

    # ----------------------------------------------------------
    # Construct training adjacency
    # ----------------------------------------------------------

    A_train = edgelist_to_adjacency(
        n,
        train_edges,
    ).tocsr()

    num_train_dyads = (
        num_dyads
        - num_test_dyads
    )

    train_fraction = (
        num_train_dyads
        / num_dyads
    )

    # ----------------------------------------------------------
    # Fit DC-SBM
    # ----------------------------------------------------------

    kappa_train, B = fit_dcsbm(
        A_train,
        groups,
    )

    # Correct degree parameters for the fraction of observed dyads.
    kappa = (
        kappa_train
        / train_fraction
    )

    # ----------------------------------------------------------
    # Compute probabilities on held-out dyads
    # ----------------------------------------------------------

    i_test, j_test = rank_to_pair(
        test_ranks
    )

    # DC-SBM:
    #
    # p_ij = min(
    #     1,
    #     kappa_i kappa_j B_{g_i,g_j} / K
    # )
    p_dcsbm = connection_probability_dcsbm_pairs(
        i_test,
        j_test,
        groups,
        kappa,
        B,
    )

    # Chung-Lu:
    #
    # p_ij = min(
    #     1,
    #     kappa_i kappa_j / K
    # )
    #
    # IMPORTANT: exactly the same kappa is used for both models.
    p_cl = connection_probability_cl_pairs(
        i_test,
        j_test,
        kappa,
    )

    # ----------------------------------------------------------
    # Predictive Bernoulli log-likelihood
    # ----------------------------------------------------------

    ll_dcsbm = bernoulli_loglik(
        y_test,
        p_dcsbm,
        eps=eps,
    )

    ll_cl = bernoulli_loglik(
        y_test,
        p_cl,
        eps=eps,
    )

    delta_ll = (
        ll_dcsbm
        - ll_cl
    )

    # ----------------------------------------------------------
    # Primary and alternative normalizations
    # ----------------------------------------------------------

    score_dyad = (
        delta_ll
        / num_test_dyads
    )

    score_edge = (
        delta_ll
        / num_test_edges
    )

    # ----------------------------------------------------------
    # Decompose improvement into edges and nonedges
    # ----------------------------------------------------------

    p_dc_clip = np.clip(
        p_dcsbm,
        eps,
        1.0 - eps,
    )

    p_cl_clip = np.clip(
        p_cl,
        eps,
        1.0 - eps,
    )

    delta_ll_edges = np.sum(
        np.log(
            p_dc_clip[edge_mask]
        )
        - np.log(
            p_cl_clip[edge_mask]
        )
    )

    delta_ll_nonedges = np.sum(
        np.log1p(
            -p_dc_clip[nonedge_mask]
        )
        - np.log1p(
            -p_cl_clip[nonedge_mask]
        )
    )

    score_dyad_edges = (
        delta_ll_edges
        / num_test_dyads
    )

    score_dyad_nonedges = (
        delta_ll_nonedges
        / num_test_dyads
    )

    # ----------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------

    # These should mostly cancel between CL and DC-SBM if caused
    # by a zero kappa, since both models use exactly the same kappa.
    num_zero_edges_dcsbm = int(
        np.sum(
            edge_mask
            & (p_dcsbm == 0.0)
        )
    )

    num_zero_edges_cl = int(
        np.sum(
            edge_mask
            & (p_cl == 0.0)
        )
    )

    num_one_nonedges_dcsbm = int(
        np.sum(
            nonedge_mask
            & (p_dcsbm == 1.0)
        )
    )

    num_one_nonedges_cl = int(
        np.sum(
            nonedge_mask
            & (p_cl == 1.0)
        )
    )

    return {
        # primary scores
        'score_dyad': score_dyad,
        'score_edge': score_edge,

        # decomposition
        'score_dyad_edges': score_dyad_edges,
        'score_dyad_nonedges': score_dyad_nonedges,

        # likelihoods
        'll_dcsbm': ll_dcsbm,
        'll_cl': ll_cl,
        'delta_ll': delta_ll,
        'delta_ll_edges': delta_ll_edges,
        'delta_ll_nonedges': delta_ll_nonedges,

        # test-set sizes
        'num_test_edges': num_test_edges,
        'num_test_nonedges': num_test_nonedges,
        'num_test_dyads': num_test_dyads,

        # diagnostics
        'num_zero_edges_dcsbm': num_zero_edges_dcsbm,
        'num_zero_edges_cl': num_zero_edges_cl,
        'num_one_nonedges_dcsbm': num_one_nonedges_dcsbm,
        'num_one_nonedges_cl': num_one_nonedges_cl,
    }