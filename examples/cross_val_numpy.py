from typing import Union

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import cross_validate

from ikpls.numpy_ikpls import PLS


class PLSWithPreprocessing(
    PLS
):  # We can simply inherit from the numpy implementation to override the fit and predict methods to include preprocessing.
    def __init__(self, algorithm: int = 1, dtype: np.float_ = np.float64) -> None:
        super().__init__(algorithm, dtype)

    def fit(
        self, X: npt.ArrayLike, Y: npt.ArrayLike, A: int
    ) -> None:  # Override the fit method to include mean centering of X and Y.
        self.X_mean = np.mean(X, axis=0)
        self.Y_mean = np.mean(Y, axis=0)
        X -= self.X_mean
        Y -= self.Y_mean
        return super().fit(X, Y, A)

    def predict(  # Override the predict method to include mean centering of X and Y based on values encountered in fit.
        self, X: npt.ArrayLike, A: Union[None, int] = None
    ) -> npt.NDArray[np.float_]:
        return super().predict(X - self.X_mean, A) + self.Y_mean


def cv_splitter(
    splits: npt.NDArray,
):  # Splits is a 1D array of integers indicating the split number for each sample.
    uniq_splits = np.unique(splits)
    for split in uniq_splits:
        train_idxs = np.nonzero(splits != split)[0]
        val_idxs = np.nonzero(splits == split)[0]
        yield train_idxs, val_idxs


def mse_for_each_target(estimator, X, Y_true, **kwargs):
    # We must return a dict of singular values. Let's choose the number of components that achieves the lowest MSE value for each target and return both MSE and the number of components.
    # Y_true has shape (N, M)
    Y_pred = estimator.predict(X, **kwargs)  # Shape (A, N, M)
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

    np_pls_alg_1 = PLSWithPreprocessing(
        algorithm=1
    )  # For this example, we will use IKPLS Algorithm #1. The interface for IKPLS Algorithm #2 is identical.
    fit_params = {"A": A}
    np_pls_alg_1_results = cross_validate(
        np_pls_alg_1,
        X,
        Y,
        cv=cv_splitter(splits),
        scoring=mse_for_each_target,  # We want to return the MSE for each target and the number of components that achieves the lowest MSE for each target.
        fit_params=fit_params,  # We want to pass the number of components to the fit method.
        return_estimator=False,  # We don't need the estimators themselves, just the MSEs and the best number of components.
        n_jobs=-1,  # Use all available CPU cores.
    )

    lowest_val_mses = np.array(
        [np_pls_alg_1_results[f"test_lowest_mse_target_{i}"] for i in range(M)]
    )  # Shape (M, splits) = (10, number_of_splits). Lowest MSE for each target for each split.

    best_num_components = np.array(
        [
            np_pls_alg_1_results[f"test_num_components_lowest_mse_target_{i}"]
            for i in range(M)
        ]
    )  # Shape (M, splits) = (10, number_of_splits). Number of components that achieves the lowest MSE for each target for each split.
