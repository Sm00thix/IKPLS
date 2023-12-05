from typing import Union

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import cross_validate

from ikpls.fast_cross_validation.numpy_ikpls import PLS


def mse_for_each_target(Y_true, Y_pred):
    # We can return anything we want. Here, we will return the lowest MSE for each target and the number of components that achieves that lowest MSE.
    # Y_true has shape (N, M)
    # Y_pred has shape (A, N, M)
    e = Y_true - Y_pred  # Shape (A, N, M)
    se = e**2  # Shape (A, N, M)
    mse = np.mean(se, axis=-2)  # Compute the mean over samples. Shape (A, M).
    row_idxs = np.argmin(
        mse, axis=0
    )  # The number of components that minimizes the MSE for each target. Shape (M,).
    lowest_mses = mse[
        row_idxs, np.arange(mse.shape[1])
    ]  # The lowest MSE for each target. Shape (M,).
    num_components = (
        row_idxs + 1
    )  # Indices are 0-indexed but number of components is 1-indexed.
    mse_names = [
        f"lowest_mse_target_{i}" for i in range(lowest_mses.shape[0])
    ]  # List of names for the lowest MSE values.
    num_components_names = [  # List of names for the number of components that achieves the lowest MSE for each target.
        f"num_components_lowest_mse_target_{i}" for i in range(lowest_mses.shape[0])
    ]
    all_names = mse_names + num_components_names  # List of all names.
    all_values = np.concatenate((lowest_mses, num_components))  # Array of all values.
    return dict(zip(all_names, all_values))


if __name__ == "__main__":
    N = 100  # Number of samples.
    K = 50  # Number of features.
    M = 10  # Number of targets.
    A = 20  # Number of latent variables (PLS components).
    splits = np.random.randint(
        0, 5, size=N
    )  # Randomly assign each sample to one of 5 splits.
    number_of_splits = np.unique(splits).shape[0]

    X = np.random.uniform(size=(N, K)).astype(np.float64)
    Y = np.random.uniform(size=(N, M)).astype(np.float64)

    np_pls_alg_1_fast_cv = PLS(
        algorithm=1
    )  # For this example, we will use IKPLS Algorithm #1. The interface for IKPLS Algorithm #2 is identical.
    np_pls_alg_1_fast_cv_results = np_pls_alg_1_fast_cv.cross_validate(
        X=X,
        Y=Y,
        A=A,
        cv_splits=splits,
        metric_function=mse_for_each_target,
        center=True,
        n_jobs=-1,
        verbose=10,
    )

    lowest_val_mses = np.array(
        [
            [
                np_pls_alg_1_fast_cv_results[i][f"lowest_mse_target_{j}"]
                for i in range(number_of_splits)
            ]
            for j in range(M)
        ]
    )  # Shape (M, splits) = (10, number_of_splits). Lowest MSE for each target for each split.

    best_num_components = np.array(
        [
            [
                np_pls_alg_1_fast_cv_results[i][f"num_components_lowest_mse_target_{j}"]
                for i in range(number_of_splits)
            ]
            for j in range(M)
        ]
    )  # Shape (M, splits) = (10, number_of_splits). Number of components that achieves the lowest MSE for each target for each split.
