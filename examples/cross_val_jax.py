from ikpls.jax_ikpls_alg_1 import (
    PLS,
)  # For this example, we will use IKPLS Algorithm #1. The interface for IKPLS Algorithm #2 is identical.
import jax.numpy as jnp
import numpy as np
import jax
from typing import Tuple


# Function to apply mean centering to X and Y based on training data.
def cross_val_preprocessing(
    X_train: jnp.ndarray,
    Y_train: jnp.ndarray,
    X_val: jnp.ndarray,
    Y_val: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    print(
        "Preprocessing function will be JIT compiled..."
    )  # The internals of .cv() in JAX are JIT compiled. That includes the preprocessing function.
    x_mean = X_train.mean(axis=0, keepdims=True)
    X_train -= x_mean
    X_val -= x_mean
    y_mean = Y_train.mean(axis=0, keepdims=True)
    Y_train -= y_mean
    Y_val -= y_mean
    return X_train, Y_train, X_val, Y_val


def mse_per_component_and_best_components(
    Y_true: jnp.ndarray, Y_pred: jnp.ndarray
) -> jnp.ndarray:
    print(
        "Metric function will be JIT compiled..."
    )  # The internals of .cv() in JAX are JIT compiled. That includes the metric function.
    # Y_true has shape (N, M), Y_pred has shape (A, N, M).
    e = Y_true - Y_pred  # Shape (A, N, M)
    se = e**2  # Shape (A, N, M)
    mse = jnp.mean(se, axis=-2)  # Shape (A, M)
    best_num_components = jnp.argmin(mse, axis=0) + 1  # Shape (M,)
    return (mse, best_num_components)


if __name__ == "__main__":
    """
    NOTE: Every time a training or validation split has a different size from the previously encountered one, recompilation will occur.
    This is because the JIT compiler must generate a new function for each unique input shape.
    Thus, if all splits have the same shape, JIT compilation happens only once.
    """
    N = 100  # Number of samples.
    K = 50  # Number of features.
    M = 10  # Number of targets.
    A = 20  # Number of latent variables (PLS components).
    splits = np.arange(100) % 5  # Randomly assign each sample to one of 5 splits.

    # Using float64 is important for numerical stability.
    X = np.random.uniform(size=(N, K)).astype(np.float64)
    Y = np.random.uniform(size=(N, M)).astype(np.float64)

    jax_pls_alg_1 = PLS(verbose=True)

    metric_names = ["mse", "best_num_components"]
    metric_values_dict = jax_pls_alg_1.cv(
        X,
        Y,
        A,
        cv_splits=splits,
        preprocessing_function=cross_val_preprocessing,
        metric_function=mse_per_component_and_best_components,
        metric_names=metric_names,
    )

    """
    list of length 5 where each element is an array of shape (A, M) = (20, 10)
    corresponding to the mse output of mse_per_component_and_best_components for each split.
    """
    mse_for_each_split = metric_values_dict["mse"]
    mse_for_each_split = np.array(
        mse_for_each_split
    )  # shape (n_splits, A, M) = (5, 20, 10)

    """
    list of length 5 where each element is an array of shape (M,) = (10,)
    corresponding to the best_num_components output of mse_per_component_and_best_components for each split.
    """
    best_num_components_for_each_split = metric_values_dict["best_num_components"]
    best_num_components_for_each_split = np.array(
        best_num_components_for_each_split
    )  # shape (n_splits, M) = (5, 10)

    """
    # The -1 in the index is due to the fact that mse_for_each_split is 0-indexed but the number of components go from 1 to A.
    This could also have been implemented using jax.numpy in mse_per_component_and_best_components directly as part of the metric function.
    """
    best_mse_for_each_split = np.amin(mse_for_each_split, axis=-2) # (n_splits, M) shape (5, 10)
    equivalent_best_mse_for_each_split = np.array(
        [
            mse_for_each_split[
                i, best_num_components_for_each_split[i] - 1, np.arange(M)
            ]
            for i in range(len(best_num_components_for_each_split))
        ]
    )  # shape (n_splits, M) = (5, 10)
    (best_mse_for_each_split == equivalent_best_mse_for_each_split).all()  # True
    pass