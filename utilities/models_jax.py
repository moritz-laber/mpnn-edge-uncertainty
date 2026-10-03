"""
models_jax.py

GCN implementation in jax.numpy, slight overhead but trainable.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import jax.experimental.sparse as sparse
from typing import List, Tuple, Callable

class Linear(eqx.Module):
    """Simple linear layer with weight and optional bias.
    
    Input
    key : a jax random key
    in_features : the number of input features
    out_features : the number of output features
    use_bias : whether to include a bias term

    Attributes
    weight : the weight matrix of shape (out_features, in_features)
    bias : the bias vector of shape (out_features,)
    use_bias : whether to include a bias term
    """

    weight: jnp.ndarray
    bias: jnp.ndarray | None 
    use_bias: bool 

    def __init__(self, key, in_features, out_features, use_bias=False):
        self.use_bias = use_bias
        self.weight = jax.random.normal(key, (in_features, out_features)) * jnp.sqrt(2.0 / in_features)
        
        if self.use_bias:
            self.bias = jnp.zeros(out_features)
        else:
            self.bias = None

    def __call__(self, X):
        """Apply the linear layer to the input x.
        
        Input
        X : a numpy array of shape (num_nodes, in_features)

        Output
        Y : a numpy array of shape (num_nodes, out_features)
        """

        if self.use_bias:
            return jnp.dot(X, self.weight) + self.bias
        else:
            return jnp.dot(X, self.weight)

class GCN(eqx.Module):
    """Graph Convolutional Network that allows to track embeddings and preactivations at each layer.
    
    Input
    key : a jax random key
    features : a list of integers specifying the number of features at each layer, including input and output
    activations : a list of activation functions to apply after each layer. Needs to be 1 shorted than features.
    use_bias : whether to include a bias term in the linear layers.

    Attributes
    layers : a list of Linear layers
    activations : a list of activation functions to apply after each layer
    use_bias : whether to include a bias term in the linear layers
    """

    layers : List[Linear]
    activations : List[Callable[[jnp.ndarray], jnp.ndarray]]
    use_bias : bool

    def __init__(self, key, features, activations, use_bias=False):

        assert len(features) == len(activations) + 1, "Number of layers must be one more than number of activations"

        self.activations = activations
        self.use_bias = use_bias

        self.layers = []
        for l in range(len(features) - 1):
            key, key_ = jax.random.split(key)
            self.layers.append(Linear(key_, features[l], features[l+1], use_bias=self.use_bias))
    
    @eqx.filter_jit
    def __call__(self, X: jnp.ndarray, A: sparse.BCOO) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        """Call the GCN on the input features X and adjacency matrix A, and return the preactivations and embeddings at each layer.
        
        Input
        X : a numpy array of shape (num_nodes, in_features) containing the input features
        A : a BCOO sparse adjacency matrix of shape (num_nodes, num_nodes) containing the graph structure.

        Output
        preactivations : a list of numpy arrays containing the preactivations at each layer, including a dummy entry NaN for the input.
        embeddings : a list of numpy arrays containing the embeddings at each layer, including the input X.
        """

        # compute the vector of degrees and its inverse square root
        k = A.sum(axis=1).todense().flatten()
        kinvsqrt = jax.lax.rsqrt(k + 1.0)

        # compute the forward pass and keep track of the preactivations and embeddings at each layer
        preactivations = [jnp.nan * jnp.ones_like(X)]
        embeddings = [X.copy()]

        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            
            # propagation
            X = kinvsqrt[:, None] * X
            X = X + A @ X
            X = kinvsqrt[:, None] * X

            # linear transformation
            X = layer(X)
            preactivations.append(jax.lax.stop_gradient(X))

            # activation
            X = activation(X)
            if i < len(self.layers) - 1:
                embeddings.append(jax.lax.stop_gradient(X))
            else:
                embeddings.append(X)

        return preactivations, embeddings

class RWGCN(eqx.Module):
    """Graph Convolutional Network with random walk normalization that allows to track embeddings and preactivations at each layer.
    
    Input
    key : a jax random key
    features : a list of integers specifying the number of features at each layer, including input and output
    activations : a list of activation functions to apply after each layer. Needs to be 1 shorted than features.
    use_bias : whether to include a bias term in the linear layers.

    Attributes
    layers : a list of Linear layers
    activations : a list of activation functions to apply after each layer
    use_bias : whether to include a bias term in the linear layers
    """

    layers : List[Linear]
    activations : List[Callable[[jnp.ndarray], jnp.ndarray]]
    use_bias : bool

    def __init__(self, key, features, activations, use_bias=False):

        assert len(features) == len(activations) + 1, "Number of layers must be one more than number of activations"

        self.activations = activations
        self.use_bias = use_bias

        self.layers = []
        for l in range(len(features) - 1):
            key, key_ = jax.random.split(key)
            self.layers.append(Linear(key_, features[l], features[l+1], use_bias=self.use_bias))
    
    @eqx.filter_jit
    def __call__(self, X: jnp.ndarray, A: sparse.BCOO) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        """Call the GCN on the input features X and adjacency matrix A, and return the preactivations and embeddings at each layer.
        
        Input
        X : a numpy array of shape (num_nodes, in_features) containing the input features
        A : a BCOO sparse adjacency matrix of shape (num_nodes, num_nodes) containing the graph structure.

        Output
        preactivations : a list of numpy arrays containing the preactivations at each layer, including a dummy entry NaN for the input.
        embeddings : a list of numpy arrays containing the embeddings at each layer, including the input X.
        """

        # compute the vector of degrees and its inverse root
        k = A.sum(axis=1).todense().flatten()
        kinv = 1.0 / (k + 1.0)

        # compute the forward pass and keep track of the preactivations and embeddings at each layer
        preactivations = [jnp.nan * jnp.ones_like(X)]
        embeddings = [X.copy()]

        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            
            # propagation
            X = X + A @ X
            X = kinv[:, None] * X

            # linear transformation
            X = layer(X)
            preactivations.append(jax.lax.stop_gradient(X))

            # activation
            X = activation(X)
            if i < len(self.layers) - 1:
                embeddings.append(jax.lax.stop_gradient(X))
            else:
                embeddings.append(X)

        return preactivations, embeddings