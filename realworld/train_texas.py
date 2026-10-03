from training import load_data, train, accuracy
from models_jax import RWGCN
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

components = 32
path_to_data = './data/texas.npz'
model_path = './models/texas.eqx'
history_path = './histories/texas.npz'
model_seed = 1014
split = 0
num_hidden = 64
use_bias = False
learning_rate = 5e-3
weight_decay = 5e-4
patience = 100
num_epochs = 1000

# load the dataset
print('load data')
X, y, A, A_sp, train_idx, val_idx, test_idx, pca = load_data(path_to_data, components=components, split=split)
num_features = X.shape[1]
num_classes = int(jnp.max(y)) + 1
print(f'num classes {num_classes}')

# define the model
model_key = jax.random.PRNGKey(model_seed)
gcn = RWGCN(model_key, features=[num_features, num_hidden, num_hidden, num_classes], activations=[jax.nn.relu, jax.nn.relu, jax.nn.identity], use_bias=use_bias)

print('train')
gcn, history = train(
    gcn,
    X,
    A,
    y,
    train_idx,
    val_idx,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    patience=patience,
    num_epochs=num_epochs
)

test_acc = accuracy(
    gcn,
    X,
    A,
    y,
    test_idx,
)

print('save')
# save the model and the results
eqx.tree_serialise_leaves(model_path, gcn)

history_dict = {k: np.asarray(v) for k, v in history.items()}
history_dict['test_accuracy'] = float(test_acc)
history_dict['random'] = float(1.0 / num_classes)

np.savez(history_path, **history_dict)