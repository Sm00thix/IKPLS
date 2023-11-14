from ikpls.jax_ikpls_alg_1 import PLS # For this example, we will use IKPLS Algorithm #1. The interface for IKPLS Algorithm #2 is identical.
import jax.numpy as jnp
import numpy as np

N = 100  # Number of samples.
K = 50  # Number of features.
M = 10  # Number of targets.
A = 20  # Number of latent variables (PLS components).
splits = np.random.randint(
    0, 5, size=N
)  # Randomly assign each sample to one of 5 splits.

# Using float64 is important for numerical stability.
X = np.random.uniform(size=(N, K)).astype(np.float64)
Y = np.random.uniform(size=(N, M)).astype(np.float64)

# Function to apply mean centering to X and Y based on training data.
def cross_val_preprocessing(X_train: jnp.ndarray,Y_train: jnp.ndarray,X_val: jnp.ndarray,Y_val: jnp.ndarray,) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    print('Preprocessing function will be JIT compiled...')
    x_mean = X_train.mean(axis=0, keepdims=True)
    X_train -= x_mean
    X_val -= x_mean
    y_mean = Y_train.mean(axis=0, keepdims=True)
    Y_train -= y_mean
    Y_val -= y_mean
    return X_train, Y_train, X_val, Y_val

def mse_per_component_and_best_components(Y_true: jnp.ndarray, Y_pred: jnp.ndarray) -> jnp.ndarray:
    # Y_true has shape (N, M), Y_pred has shape (A, N, M).
    e = Y_true - Y_pred # Shape (A, N, M)
    se = e**2 # Shape (A, N, M)
    mse = jnp.mean(se, axis=-2) # Shape (A, M)
    best_num_components = jnp.argmin(mse, axis=0) + 1 # Shape (M,)
    return (mse, best_num_components)




if __name__ == "__main__":
    pass