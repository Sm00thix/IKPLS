from numpy import ndarray
from sklearn.cross_decomposition import PLSRegression as SK_PLS
import numpy as np
import numpy.typing as npt
import jax.numpy as jnp
from sklearn.model_selection import KFold, cross_validate
from timeit import Timer
from timeit import default_timer


class SK_PLS_All_Components(SK_PLS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(
        self,
        X,
        Y,
    ):  # Override fit method to store regression matrices for all possible number of components. This is MUCH faster than fitting an entirely new model for every single number of components
        super().fit(X, Y)
        B = np.empty(
            (self.n_components, self.n_features_in_, self.y_loadings_.shape[0])
        )
        for i in range(B.shape[0]):
            B_at_component_i = np.dot(
                self.x_rotations_[..., : i + 1],
                self.y_loadings_[..., : i + 1].T,
            )
            B[i] = B_at_component_i
        self.B = B

    def predict(
        self, X: npt.ArrayLike
    ) -> (
        ndarray
    ):  # Override predict function to give a prediction for each number of components up to A. This is MUCH faster than calling the predict function n_components number of times with different number of components.
        return (X - self._x_mean) / self._x_std @ self.B + self.intercept_


def gen_random_data(n, m, k):
    rng = np.random.default_rng(seed=42)
    X = rng.random((n, m), dtype=np.float64)
    Y = rng.random((n, k), dtype=np.float64)
    return X, Y


def mse_protein_moisture(estimator, X, Y_true, **kwargs):
    # We must return a dict of singular values. Let's choose the number of components that achieves the lowest MSE value for each target and return both MSE and the number of components.
    Y_pred = estimator.predict(
        X, **kwargs
    )  # Shape is n_components x num_samples x num_targets
    e = Y_true - Y_pred
    se = e**2
    mse = np.mean(se, axis=-2)  # Compute the mean over samples
    row_idxs = np.argmin(
        mse, axis=0
    )  # The number of components that minimizes the MSE for each target.
    lowest_mses = mse[
        row_idxs, np.arange(mse.shape[1])
    ]  # The lowest MSE for each target.
    num_components = (
        row_idxs + 1
    )  # Indices are 0-indexed but number of components is 1-indexed.
    mse_names = [f"lowest_mse_target_{i}" for i in range(lowest_mses.shape[0])]
    num_components_names = [
        f"num_components_lowest_mse_target_{i}" for i in range(lowest_mses.shape[0])
    ]
    all_names = mse_names + num_components_names
    all_values = np.concatenate((lowest_mses, num_components))
    return dict(zip(all_names, all_values))


def jax_mse_protein_moisture(Y_true, Y_pred):
    e = Y_true - Y_pred
    se = e**2
    mse = jnp.mean(se, axis=-2)  # Compute the mean over samples
    row_idxs = jnp.argmin(
        mse, axis=0
    )  # The number of components that minimizes the MSE for each target.
    lowest_mses = mse[
        row_idxs, jnp.arange(mse.shape[1])
    ]  # The lowest MSE for each target.
    num_components = (
        row_idxs + 1
    )  # Indices are 0-indexed but number of components is 1-indexed.
    all_values = jnp.concatenate((lowest_mses, num_components))
    return all_values


def jax_metric_names(K):
    mse_names = [f"lowest_mse_target_{i}" for i in range(K)]
    num_components_names = [f"num_components_lowest_mse_target_{i}" for i in range(K)]
    all_names = mse_names + num_components_names
    return all_names


def cross_val_cpu_pls(pls, X, Y, n_splits, fit_params, n_jobs, verbose):
    cv = KFold(n_splits=n_splits, shuffle=False)
    t = Timer(
        stmt="scores = cross_validate(pls, X, Y, cv=cv, scoring=mse_protein_moisture, return_estimator=False, fit_params=fit_params, n_jobs=n_jobs, verbose=verbose, )",
        timer=default_timer,
        globals=locals() | globals(),
    )
    return t.timeit(number=1)


def single_fit_cpu_pls(pls, X, Y, fit_params=None):
    t = Timer(
        stmt="pls.fit(X, Y, **fit_params)",
        timer=default_timer,
        globals=locals() | globals(),
    )
    return t.timeit(number=1)


def jax_preprocessing_function(X_train, Y_train, X_val, Y_val):
    return X_train, Y_train, X_val, Y_val


def cross_val_gpu_pls(pls, X, Y, n_components, n_splits, show_progress):
    cv_splits = np.zeros(X.shape[0])
    for i in range(X.shape[0] % n_splits):
        split_size = X.shape[0] // n_splits + 1
        cv_splits[i * split_size : (i + 1) * split_size] = i
    prev_max_idx = (X.shape[0] % n_splits) * (X.shape[0] // n_splits + 1)
    for i in range(n_splits - X.shape[0] % n_splits):
        split_size = X.shape[0] // n_splits
        cv_splits[
            prev_max_idx + i * split_size : prev_max_idx + (i + 1) * split_size
        ] = (i + X.shape[0] % n_splits)
    t = Timer(
        stmt="pls.cv(X=X, Y=Y, A=n_components, cv_splits=cv_splits, preprocessing_function=jax_preprocessing_function, metric_function=jax_mse_protein_moisture, metric_names=jax_metric_names(Y.shape[1]), show_progress=show_progress)",
        timer=default_timer,
        globals=locals() | globals(),
    )
    return t.timeit(number=1)


def single_fit_gpu_pls(pls, X, Y, n_components):
    t = Timer(
        stmt="pls.fit(X, Y, A=n_components)",
        timer=default_timer,
        globals=locals() | globals(),
    )
    return t.timeit(number=1)


if __name__ == "__main__":
    X, Y = gen_random_data(9, 100, 10)
    from algorithms.jax_ikpls_alg_1 import PLS as JAX_PLS_Alg_1

    pls = SK_PLS_All_Components()
    n_components = 20
    n_splits = 5
    cross_val_cpu_pls(pls, X, Y, n_splits, {})