"""
graphs.y

Functions to generate and manipulate graphs.
"""

import numpy as np
import scipy as sp
import math
from typing import List, Tuple, Optional
import warnings

def edgelist_to_adjacency(n:int, edges:List[Tuple[int, int]]) -> sp.sparse.coo_matrix:
    """Convert an edgelist to an adjacency matrix.
    
    Input
    n : the number of nodes in the graph
    edges : a list of tuples (i, j) representing an undirected edge from node i to node j.
    
    Output
    A : the adjacency matrix of the graph, in sparse COO format.
    """
    
    row, col = zip(*edges)
    data = np.ones(len(edges))

    A = sp.sparse.coo_matrix((data, (row, col)), shape=(n, n))
    A += A.T

    return A

def sample_gnp(rng:np.random.Generator, num_nodes:int, p:float) -> List[Tuple[int, int]]:
    """Generate a random graph from the model G(n, p).
    
    Import 
    rng : a numpy random generator
    num_nodes : the number of nodes in the graph
    p : the probability of an edge between any two nodes

    Output
    edges : a list of tuples (i, j) representing an undirected edge from node i to node j.
    """

    # determine the maximum number of possible edges
    mmax = num_nodes * (num_nodes - 1) // 2

    # sample edges using the geometric distribution to skip over non-edges
    edges = []
    m = -1
    while True:

        # sample the step
        step = rng.geometric(p)
        m += step

        if m >= mmax:
            break
        else:
            
            # find the first node
            i = int((1 + math.isqrt(1 + 8 * m)) // 2)
            ti = i * (i - 1) // 2

            # correction of rare rounding errors due to integer square root.
            while ti > m:
                i -= 1
                ti = i * (i - 1) // 2
            while (i + 1) * i // 2 <= m:
                i += 1
                ti = i * (i - 1) // 2
            
            # find the second node
            j = int(m - ti)

            edges.append((i, j))

    return edges

def sample_sbm(rng: np.random.Generator, groups: np.ndarray, P: np.ndarray) -> List[Tuple[int, int]]:
    """Sample an SBM with arbitrary node-to-group assignments."""

    num_groups = P.shape[0]
    nodes = [np.where(groups == a)[0] for a in range(num_groups)]

    edges = []
    for a in range(num_groups):
        for b in range(a, num_groups):

            p = P[a, b]
            if p <= 0:
                continue

            nodes_a = nodes[a]
            nodes_b = nodes[b]
            na = len(nodes_a)
            nb = len(nodes_b)

            # within group
            if a == b:
                mmax = na * (na - 1) // 2
                m = -1

                while True:
                    m += rng.geometric(p)

                    if m >= mmax:
                        break

                    # unrank m into local edge (i, j), j < i
                    i = (1 + math.isqrt(1 + 8 * m)) // 2
                    ti = i * (i - 1) // 2

                    while ti > m:
                        i -= 1
                        ti = i * (i - 1) // 2

                    while (i + 1) * i // 2 <= m:
                        i += 1
                        ti = i * (i - 1) // 2

                    j = m - ti

                    edges.append((
                        int(nodes_a[i]),
                        int(nodes_a[j])
                    ))

            # between groups
            else:
                mmax = na * nb
                m = -1

                while True:
                    m += rng.geometric(p)

                    if m >= mmax:
                        break

                    i, j = divmod(m, nb)

                    edges.append((
                        int(nodes_a[i]),
                        int(nodes_b[j])
                    ))

    return edges

def sample_srgg(rng:np.random.Generator, x:np.ndarray, c:float, beta:float)->List[Tuple[int,int]]:
    """Sample a 1-dimensional soft random geometric graph with polynomially decaying connection probability.
    
    Input
    rng : a numpy random generator.
    num_nodes : the number of nodes in the graph.
    c : the connection length scale of the srgg.
    beta : the inverse temperature parameter.

    Output
    edges : the edge list of the generated graph.
    """

    # type check and sanity check
    num_nodes = x.shape[0]
    c = float(c)
    beta = float(beta)

    assert num_nodes > 0, "num_nodes must be a positive integer."
    assert beta > 0, "beta must be a positive number."

    # compute pairwise distances on the circle
    q = rng.uniform(0, 1., size=num_nodes*(num_nodes - 1)//2)
    idx = 0
    edges = []
    for i in range(num_nodes):
        
        # compute the distance on the circle from i to all j > i .
        dij = num_nodes/2. - np.abs(num_nodes/2. - np.abs(x[i] - x[i+1:]))

        # compute the connection probability
        pij = np.minimum(np.power(c / dij, beta), 1.0)

        # determine which edges to create
        js = np.where(q[idx:idx + num_nodes - i - 1] < pij)[0] + i + 1
        idx += num_nodes - i - 1

        # add edges to the list
        if len(js) > 0:
            edges.extend(list(zip(len(js)*[i], js.tolist())))

    return edges

def average_degree_srgg(c:float, beta:float, num_nodes:int=None) -> float:
    """Compute the average degree in the SRGG model in the thermodynamic limit or with first
       order finite size correction.
    
    Input
    c : the length scale in the SRGG model.
    beta : inverse temperature in the SRGG model.
    num_nodes : the number of nodes if finite-size correction is desired (set None otherwise).

    Output
    kbar : the average degree in the SRGG model in the thermodynamic limit.
    """

    if num_nodes is None:
        kbar = 2.0 * c * (beta / (beta - 1.0))
    else:
        kbar = (2.0 * (num_nodes - 1.0) * c) * (beta - (2.0 * c/ num_nodes)**(beta - 1.0)) / (num_nodes * (beta - 1.0))

    return kbar

def average_degree_to_c(kbar:float, beta:float, num_nodes:Optional[int]=None)->float:
    """Converts the average degree into a a connection scale c for the SRGG, using
       either the thermodynamic limit formula or the first-order finite size correction.
       
    Input
    kbar : the average degree
    beta : the inverse temperature
    num_nodes : (optional) the number of nodes, if supplied the formula for kbar with first order correction is numerically inverted.   
    
    Output
    c : the connection scale such that a random node in SRGG(n, c, beta) has average degree kbar.
    """

    # sanity check
    assert kbar > 0, "kbar needs to be non-negative."
    assert 1.0 < beta < 2.0, "only the regime beta in (1.0, 2.0) is supported"
    if num_nodes is not None:
        assert num_nodes > 1, "num_nodes needs to be non-negative."

    # thermodynamic limit (as initial guess or output)
    c0 = (beta - 1.0) / (2.0 * beta) * kbar

    # find c at finite size using numerical root finding.
    if num_nodes is not None:
        _obj = lambda c : average_degree_srgg(c, beta, num_nodes=num_nodes) - kbar
        solution = sp.optimize.root_scalar(_obj, bracket=[1e-15, num_nodes/2.0], x0=c0)
    
        if solution.converged:
            c = solution.root
        else:
            c = c0
            warnings.warn("Numerical root finding failed. Falling back to thermodynamic limit value.")
    else:
        c = c0
    
    return c

def sample_pareto(kbar:float, gamma:float, num_samples:int, rng:np.random.Generator)->np.ndarray:
    """Sample a Pareto distribution with exponent gamma and mean kbar.
    
    Input
    kbar : the mean of the Pareto distribution
    gamma : the pdf exponent of the Pareto distribution (must be > 2)
    num_samples : the number of samples to draw
    rng : a numpy random generator.
    
    Output
    ks : an array of samples from the Pareto distribution.
    """

    assert gamma > 2.0, "This implementation assumes finite mean, i.e., gamma > 2.0 is required"
    assert kbar > 0.0, "kbar must be positive."

    kmin = kbar * (gamma - 2.0) / (gamma - 1.0)
    u = rng.uniform(size=num_samples)

    return kmin * (1.0 - u) ** (-1.0 / (gamma - 1.0))

def sample_cl(rng:np.random.Generator, kappas:np.ndarray)->List[Tuple[int,int]]:
    """Sample the soft configuration model in (clipped) Chung-Lu approximation, p_ij = min(1, kappa_i kappa_j / sum(kappa)) 

    Input
    rng : a numpy random generator.
    kappas : the expected degree parameters
             (equal to expected degrees asymptotically if graphs are not too degree heterogeneous)

    Output
    edges : the edge list of the generated graph.   
    """

    # 
    n = kappas.shape[0]
    K = np.sum(kappas)

    assert np.all(kappas >= 0), "kappas need to be non-negative."

    # sort
    order = np.argsort(kappas)[::-1]
    kappas_sorted = kappas[order]

    edges = []
    for i in range(n - 1):
        j = i + 1
        while j < n:

            # compute the connection probability
            p = min(kappas_sorted[i] * kappas_sorted[j] / K, 1.0)

            if p == 0.0:
                break
            elif p == 1.0:
                edges.append((int(order[i]), int(order[j])))
                j += 1
                continue
            else:

                # sample geometric step taking into account
                # that numpy samples geometric on {1, 2, ...}
                # and we need {0, 1, ...}
                delta = rng.geometric(p) - 1
                j += delta
                if j >= n:
                    break

                # adjust with rejection sampling
                q = min(kappas_sorted[i] * kappas_sorted[j] / K, 1.0)
                if rng.uniform() < q/p:
                    edges.append((int(order[i]), int(order[j])))

                # even it fails we move to the next node
                j += 1

    return edges

def sample_dcsbm(rng: np.random.Generator, kappa: np.ndarray, groups: np.ndarray, B: np.ndarray) -> List[Tuple[int, int]]:
    """Sample the DC-SBM in clipped CL approximation,
    p_ij = min(1, kappa_i kappa_j B_ab / sum(kappa)).
    """

    num_groups = B.shape[0]
    K = np.sum(kappa)

    # nodes in each group, sorted by decreasing kappa
    nodes = []
    kappas = []

    for a in range(num_groups):
        idx = np.where(groups == a)[0]
        idx = idx[np.argsort(kappa[idx])[::-1]]

        nodes.append(idx)
        kappas.append(kappa[idx])

    edges = []

    for a in range(num_groups):
        for b in range(a, num_groups):

            Bab = B[a, b]
            if Bab <= 0:
                continue

            nodes_a = nodes[a]
            nodes_b = nodes[b]
            kappa_a = kappas[a]
            kappa_b = kappas[b]

            na = len(nodes_a)
            nb = len(nodes_b)

            # within group
            if a == b:
                for i in range(na - 1):
                    j = i + 1

                    while j < na:
                        p = min(kappa_a[i] * kappa_a[j] * Bab / K, 1.0)

                        if p == 0:
                            break

                        if p == 1:
                            edges.append((int(nodes_a[i]), int(nodes_a[j])))
                            j += 1
                            continue

                        j += rng.geometric(p) - 1

                        if j >= na:
                            break

                        q = min(kappa_a[i] * kappa_a[j] * Bab / K, 1.0)

                        if rng.uniform() < q / p:
                            edges.append((int(nodes_a[i]), int(nodes_a[j])))

                        j += 1

            # between groups
            else:
                for i in range(na):
                    j = 0

                    while j < nb:
                        p = min(kappa_a[i] * kappa_b[j] * Bab / K, 1.0)

                        if p == 0:
                            break

                        if p == 1:
                            edges.append((int(nodes_a[i]), int(nodes_b[j])))
                            j += 1
                            continue

                        j += rng.geometric(p) - 1

                        if j >= nb:
                            break

                        q = min(kappa_a[i] * kappa_b[j] * Bab / K, 1.0)

                        if rng.uniform() < q / p:
                            edges.append((int(nodes_a[i]), int(nodes_b[j])))

                        j += 1

    return edges

def fit_dcsbm(A, groups):
    """Fit the DC-SBM

        p_ij = min(1, kappa_i kappa_j B_ab / sum(kappa))

    using observed node degrees for kappa and observed block edge
    counts for B.
    """

    groups = np.asarray(groups, dtype=int)

    n = A.shape[0]
    num_groups = groups.max() + 1

    # observed node degrees
    kappa = np.asarray(A.sum(axis=1)).ravel()
    K = kappa.sum()

    # group indicator matrix
    S = sp.sparse.csr_matrix(
        (
            np.ones(n),
            (np.arange(n), groups)
        ),
        shape=(n, num_groups)
    )

    # block edge counts
    #
    # Off-diagonal:
    #     E[a,b] = number of edges between a and b
    #
    # Diagonal:
    #     E[a,a] = twice the number of internal edges
    E = (S.T @ A @ S).toarray()

    # total degree in each group
    Ka = np.bincount(
        groups,
        weights=kappa,
        minlength=num_groups
    )

    # infer block affinity matrix
    denom = np.outer(Ka, Ka)
    B = np.divide(
        K * E,
        denom,
        out=np.zeros_like(E, dtype=float),
        where=denom > 0
    )

    return kappa, B

def connection_probability_gnp(n, p):

    return p * np.ones((n,n))

def connection_probability_cl(kappas):

    return np.minimum(1.0, np.einsum('i,j->ij', kappas, kappas)/np.sum(kappas))

def connection_probability_sbm(groups, P):

    return P[groups[:, None], groups[None, :]]

def connection_probability_dcsbm(groups, kappa, B):

    BB = B[groups[:, None], groups[None, :]]
    p = np.minimum(1.0, np.einsum('i,ij,j -> ij', kappa, BB, kappa)/np.sum(kappa))

    return p

def connection_probability_srgg(x, c, beta):

    num_nodes = x.shape[0]

    dx = np.abs(x[:, None] - x[None, :])
    dij = num_nodes / 2. - np.abs(num_nodes / 2. - dx)

    return np.minimum(np.power(c / dij, beta), 1.0)

def connection_probability_dcsbm_pairs(
    i,
    j,
    groups,
    kappa,
    B,
):

    K = np.sum(kappa)

    p = (
        kappa[i]
        * kappa[j]
        * B[groups[i], groups[j]]
        / K
    )

    return np.minimum(
        p,
        1.0,
    )

def connection_probability_cl_pairs(
    i,
    j,
    kappa,
):

    K = np.sum(kappa)

    p = (
        kappa[i]
        * kappa[j]
        / K
    )

    return np.minimum(
        p,
        1.0,
    )

def bernoulli_loglik(y, p, eps=1e-12):

    p = np.clip(
        p,
        eps,
        1.0 - eps,
    )

    return np.sum(
        np.where(
            y == 1,
            np.log(p),
            np.log1p(-p),
        )
    )