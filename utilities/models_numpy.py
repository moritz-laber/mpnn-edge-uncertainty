"""
models_numpy.py

GCN implementation in numpy for untrained models without overhead.
"""

import numpy as np
import math
from typing import List, Tuple, Callable
import scipy as sp
from utils import m

class Linear:
    """Simple linear layer with weight and optional bias.
    
    Input
    rng : a numpy random generator
    in_features : the number of input features
    out_features : the number of output features
    use_bias : whether to include a bias term

    Attributes
    weight : the weight matrix of shape (out_features, in_features)
    bias : the bias vector of shape (out_features,)
    use_bias : whether to include a bias term
    """

    weight : np.ndarray
    bias : np.ndarray
    use_bias : bool

    def __init__(self, rng: np.random.Generator, in_features: int, out_features: int, use_bias:bool=False):
        self.weight = rng.normal(size=(in_features, out_features)) * math.sqrt(2.0 / in_features)
        self.bias = np.zeros(out_features)
        self.use_bias = use_bias
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Apply the linear layer to the input x.
        
        Input
        X : a numpy array of shape (num_nodes, in_features)

        Output
        Y : a numpy array of shape (num_nodes, out_features)
        """

        if self.use_bias:
            return np.dot(X, self.weight) + self.bias
        else:
            return np.dot(X, self.weight)

class GCN:
    """Graph Convolutional Network that allows to track embeddings and preactivations at each layer.
    
    Input
    rng : a numpy random generator
    features : a list of integers specifying the number of features at each layer, including input and output
    activations : a list of activation functions to apply after each layer. Needs to be 1 shorted than features.
    use_bias : whether to include a bias term in the linear layers.

    Attributes
    layers : a list of Linear layers
    activations : a list of activation functions to apply after each layer
    use_bias : whether to include a bias term in the linear layers
    """

    layers : List[Linear]
    activations : List[Callable[[np.ndarray], np.ndarray]]
    use_bias : bool

    def __init__(self, rng: np.random.Generator, features: List[int], activations: List[Callable[[np.ndarray], np.ndarray]], use_bias:bool=False):
        
        assert len(features) == len(activations) + 1, "Number of layers must be one more than number of activations"

        self.use_bias = use_bias
        self.layers = [Linear(rng, features[i], features[i+1], self.use_bias) for i in range(len(features)-1)]
        self.activations = activations
    
    def __call__(self, X: np.ndarray, A: sp.sparse.csr_matrix) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Call the GCN on the input features X and adjacency matrix A, and return the preactivations and embeddings at each layer.
        
        Input
        X : a numpy array of shape (num_nodes, in_features) containing the input features
        A : a scipy sparse adjacency matrix of shape (num_nodes, num_nodes) containing the graph structure.

        Output
        preactivations : a list of numpy arrays containing the preactivations at each layer, including a dummy entry NaN for the input.
        embeddings : a list of numpy arrays containing the embeddings at each layer, including the input X.
        """

        # compute the vector of degrees and its inverse square root for normalization
        k = np.asarray(A.sum(axis=1)).flatten()
        kinvsqrt = np.power(k + 1, -0.5)

        # compute the forward pass and keep track of the preactivations and embeddings at each layer
        preactivations = [np.nan * np.ones_like(X)]
        embeddings = [X.copy()]
        for layer, activation in zip(self.layers, self.activations):
            
            # propagation
            X = kinvsqrt[:, None] * X
            X = X + A @ X
            X = kinvsqrt[:, None] * X

            # linear transformation
            X = layer(X)
            preactivations.append(X.copy())

            # activation
            X = activation(X)
            embeddings.append(X.copy())
        
        return preactivations, embeddings

class RWGCN:
    """Graph Convolutional Network with random walk normalization that allows to track embeddings and preactivations at each layer.
    
    Input
    rng : a numpy random generator
    features : a list of integers specifying the number of features at each layer, including input and output
    activations : a list of activation functions to apply after each layer. Needs to be 1 shorted than features.
    use_bias : whether to include a bias term in the linear layers.

    Attributes
    layers : a list of Linear layers
    activations : a list of activation functions to apply after each layer
    use_bias : whether to include a bias term in the linear layers
    """

    layers : List[Linear]
    activations : List[Callable[[np.ndarray], np.ndarray]]
    use_bias : bool

    def __init__(self, rng: np.random.Generator, features: List[int], activations: List[Callable[[np.ndarray], np.ndarray]], use_bias:bool=False):
        
        assert len(features) == len(activations) + 1, "Number of layers must be one more than number of activations"

        self.use_bias = use_bias
        self.layers = [Linear(rng, features[i], features[i+1], self.use_bias) for i in range(len(features)-1)]
        self.activations = activations
    
    def __call__(self, X: np.ndarray, A: sp.sparse.csr_matrix) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Call the GCN on the input features X and adjacency matrix A, and return the preactivations and embeddings at each layer.
        
        Input
        X : a numpy array of shape (num_nodes, in_features) containing the input features
        A : a scipy sparse adjacency matrix of shape (num_nodes, num_nodes) containing the graph structure.

        Output
        preactivations : a list of numpy arrays containing the preactivations at each layer, including a dummy entry NaN for the input.
        embeddings : a list of numpy arrays containing the embeddings at each layer, including the input X.
        """

        # compute the vector of degrees and its inverse for normalization
        k = np.asarray(A.sum(axis=1)).flatten()
        kinv = np.power(k + 1, -1.0)

        # compute the forward pass and keep track of the preactivations and embeddings at each layer
        preactivations = [np.nan * np.ones_like(X)]
        embeddings = [X.copy()]
        for layer, activation in zip(self.layers, self.activations):
            
            # propagation
            X = X + A @ X
            X = kinv[:, None] * X

            # linear transformation
            X = layer(X)
            preactivations.append(X.copy())

            # activation
            X = activation(X)
            embeddings.append(X.copy())
        
        return preactivations, embeddings

def deterministic_update(X, p, weights, activations):

    preactivations = []
    embeddings = []
    for w, activation in zip(weights, activations):
        
        X = m(p, X)
        X = np.dot(X, w)
        preactivations.append(X.copy())
        X = activation(X)
        embeddings.append(X.copy())

    return preactivations, embeddings
        