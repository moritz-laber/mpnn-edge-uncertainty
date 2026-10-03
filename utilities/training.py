import jax
import jax.experimental.sparse as sparse
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from sklearn.decomposition import PCA
from graphs import edgelist_to_adjacency

def cross_entropy(model, X, A, y, train_idx):

    _, embeddings = model(X, A)
    logits = embeddings[-1]

    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits[train_idx], y[train_idx]))

@eqx.filter_jit
def train_step(model, opt_state, optimizer, X, A, y, train_idx):

    loss, grads = eqx.filter_value_and_grad(cross_entropy)(model, X, A, y, train_idx)

    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))

    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss

@eqx.filter_jit
def accuracy(model, X, A, y, idx):

    _, embeddings = model(X, A)
    logits = embeddings[-1]

    pred = jnp.argmax(logits, axis=-1)

    return jnp.mean(pred[idx] == y[idx])

def train(model, X, A, y, train_idx, val_idx, learning_rate=1e-2, weight_decay=1e-4, num_epochs=1000, patience=100):

    # init optimizer
    optimizer = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    opt_state = optimizer.init(
        eqx.filter(model, eqx.is_array)
    )


    best_model = model
    best_val = -jnp.inf
    epochs_without_improvement = 0

    history = {
        "loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    for epoch in range(num_epochs):

        print(f'{epoch} / {num_epochs}', end='\r')

        model, opt_state, loss = train_step(
            model,
            opt_state,
            optimizer,
            X,
            A,
            y,
            train_idx,
        )

        train_acc = accuracy(
            model, X, A, y, train_idx
        )

        val_acc = accuracy(
            model, X, A, y, val_idx
        )

        history["loss"].append(float(loss))
        history["train_accuracy"].append(float(train_acc))
        history["val_accuracy"].append(float(val_acc))

        if val_acc > best_val:
            best_val = val_acc
            best_model = model
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    return best_model, history

def load_data(path_to_data, split=0, components=None):

    # load data
    data = np.load(path_to_data, allow_pickle=True)

    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int32)

    edge_list = np.asarray(data["edge_list"], dtype=np.int32)

    train_mask = data["train_split"]
    val_mask = data["val_split"]
    test_mask = data["test_split"]

    n = X.shape[0]

    # dimensionality reduction
    if components is not None:
        pca = PCA(
            n_components=components,
            whiten=False,
            svd_solver="full",
        )

        X = pca.fit_transform(X)
        X = X.astype(np.float32)
    else:
        pca = None

    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True, ddof=0)
    X = (X - mean) / std
    X = X.astype(np.float32)

    # create adjacency matrix
    edge_set = set()
    for i,j in edge_list.T:

        i = int(i)
        j = int(j)

        if i == j:
            continue

        edge_set.add((min(i,j), max(i,j)))

    edges = list(edge_set)

    A_sp = edgelist_to_adjacency(n, edges).tocsr()
    A = sparse.BCOO.from_scipy_sparse(A_sp)

    # train and test split
    if train_mask.ndim == 2:
        train_mask = train_mask[:, split]

    if val_mask.ndim == 2:
        val_mask = val_mask[:, split]

    if test_mask.ndim == 2:
        test_mask = test_mask[:, split]
    
    train_idx = jnp.asarray(
        np.flatnonzero(train_mask),
        dtype=jnp.int32,
    )

    val_idx = jnp.asarray(
        np.flatnonzero(val_mask),
        dtype=jnp.int32,
    )

    test_idx = jnp.asarray(
        np.flatnonzero(test_mask),
        dtype=jnp.int32,
    )

    X = jnp.asarray(X)
    y = jnp.asarray(y)

    return (
        X,
        y,
        A,
        A_sp,
        train_idx,
        val_idx,
        test_idx,
        pca
    )