import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from functools import partial
import abc
from typing import Tuple, Union, Any
from collections.abc import Callable
from tqdm import tqdm
import numpy as np
import warnings


class PLSBase(abc.ABC):
    """
    Implements partial least-squares regression using Improved Kernel PLS by Dayal and MacGregor: https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23
    """

    def __init__(self) -> None:
        self.name = "PLS"

    def weight_warning(self, arg, _transforms):
        i, norm = arg
        if np.isclose(norm, 0, atol=np.finfo(np.float64).eps, rtol=0):
            warnings.warn(
                f"Weight is close to zero. Results with A = {i} component(s) or higher may be unstable."
            )

    @partial(jax.jit, static_argnums=0)
    def compute_regression_coefficients(
        self, b_last: jnp.ndarray, r: jnp.ndarray, q: jnp.ndarray
    ) -> jnp.ndarray:
        b = b_last + r @ q.T
        return b

    @abc.abstractmethod
    @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def _get_initial_matrices(
        self, A, K, M
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        B = jnp.zeros(shape=(A, K, M), dtype=jnp.float64)
        W = jnp.zeros(shape=(A, K), dtype=jnp.float64)
        P = jnp.zeros(shape=(A, K), dtype=jnp.float64)
        Q = jnp.zeros(shape=(A, M), dtype=jnp.float64)
        R = jnp.zeros(shape=(A, K), dtype=jnp.float64)
        return B, W, P, Q, R

    @partial(jax.jit, static_argnums=0)
    def _compute_XT(self, X: jnp.ndarray) -> jnp.ndarray:
        return X.T

    @partial(jax.jit, static_argnums=0)
    def _compute_initial_XTY(self, XT: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        return XT @ Y

    @partial(jax.jit, static_argnums=0)
    def _compute_XTX(self, XT: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        return XT @ X

    @abc.abstractmethod
    @partial(jax.jit, static_argnums=(0, 3))
    def stateless_fit(
        self, X: jnp.ndarray, Y: jnp.ndarray, A: int
    ) -> Union[
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        Tuple[
            jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
        ],
    ]:
        pass

    @abc.abstractmethod
    def _step_1(self):
        pass

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def _step_2(
        self, XTY: jnp.ndarray, M: jnp.ndarray, K: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.float64]:
        print("Tracing step 2...")
        if M == 1:
            norm = jla.norm(XTY)
            w = XTY / norm
        else:
            if M < K:
                XTYTXTY = XTY.T @ XTY
                eig_vals, eig_vecs = jla.eigh(XTYTXTY)
                q = eig_vecs[:, -1:]
                q = q.reshape(-1, 1)
                w = XTY @ q
                norm = jla.norm(w)
                w = w / norm
            else:
                XTYYTX = XTY @ XTY.T
                eig_vals, eig_vecs = jla.eigh(XTYYTX)
                w = eig_vecs[:, -1:]
                norm = eig_vals[-1]
        return w, norm

    @partial(
        jax.jit,
        static_argnums=(
            0,
            1,
        ),
    )
    def _step_3(
        self, A: int, w: jnp.ndarray, P: jnp.ndarray, R: jnp.ndarray
    ) -> jnp.ndarray:
        print("Tracing step 3...")
        r = jnp.copy(w)
        r, P, w, R = jax.lax.fori_loop(0, A, self._step_3_body, (r, P, w, R))
        return r

    @partial(jax.jit, static_argnums=0)
    def _step_3_body(
        self, j: int, carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        print("Tracing step 3 loop body...")
        r, P, w, R = carry
        r = r - P[j].reshape(-1, 1).T @ w * R[j].reshape(-1, 1)
        return r, P, w, R

    @abc.abstractmethod
    def _step_4(self):
        pass

    @partial(jax.jit, static_argnums=0)
    def _step_5(
        self, XTY: jnp.ndarray, p: jnp.ndarray, q: jnp.ndarray, tTt: jnp.ndarray
    ) -> jnp.ndarray:
        print("Tracing step 5...")
        return XTY - (p @ q.T) * tTt

    @abc.abstractmethod
    def _main_loop_body(self):
        pass

    @abc.abstractmethod
    def fit(self, X: jnp.ndarray, Y: jnp.ndarray, A: int) -> None:
        pass

    @partial(jax.jit, static_argnums=(0, 3))
    def stateless_predict(
        self, X: jnp.ndarray, B: jnp.ndarray, A: None | int = None
    ) -> jnp.ndarray:
        print("Tracing stateless predict...")
        """
        Parameters:
        X: Predictor variables matrix (N x K)
        B: Regression coefficient matrix (A x K x M) or (K x M)
        A: Integer number of components to use in the prediction or None. If None, return the predictions for every component. Defaults to the maximum number of components, the model was fitted with.

        Returns:
        Y_hat: Predicted response variables matrix (N x M) or (A x N x M)
        """

        if A is None:
            return X @ B
        else:
            return X @ B[A - 1]

    def predict(self, X: jnp.ndarray, A: None | int = None) -> jnp.ndarray:
        """
        Parameters:
        X: Predictor variables matrix (N x K)
        A: Integer number of components to use in the prediction or None. If None, return the predictions for every component. Defaults to the maximum number of components, the model was fitted with.

        Returns:
        Y_hat: Predicted response variables matrix (N x M) or (A x N x M)
        """

        if A is None:
            return X @ self.B
        else:
            return X @ self.B[A - 1]

    @partial(jax.jit, static_argnums=(0, 3, 6))
    def stateless_fit_predict_eval(
        self,
        X_train: jnp.ndarray,
        Y_train: jnp.ndarray,
        A: int,
        X_test: jnp.ndarray,
        Y_test: jnp.ndarray,
        metric_function: Callable[[jnp.ndarray, jnp.ndarray], Any],
    ) -> Tuple[jnp.int64, jnp.float64]:
        """
        Parameters:
        X: Predictor variables matrix (N x K)
        Y: Response variables matrix (N x M)
        A: Number of components in the PLS model
        metric_function: Callable that takes two array of shape (N x M) and returns Any.
        """

        matrices = self.stateless_fit(X_train, Y_train, A)
        B = matrices[0]
        Y_pred = self.stateless_predict(X_test, B)
        return metric_function(Y_test, Y_pred)

    def cv(
        self,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        A: int,
        cv_splits: jnp.ndarray,
        preprocessing_function: Callable[
            [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ],
        metric_funtion: Callable[[jnp.ndarray, jnp.ndarray], Any],
        metric_names: list[str],
    ) -> dict[str, Any]:
        """
        Perform cross validation on using jnp.unique(cv_splits) splits. For each split, (X_train, Y_train, X_val, Y_val) = preprocessing_function(X_train, Y_train, X_val, Y_val) and metric_function(Y_val, self.stateless_predict(X_val, B)) where B is the regression matrix derived from self.stateless_fit(X_train, Y_train, A).

        Parameters:
        X: Predictor variables matrix (N x K)
        Y: Response variables matrix (N x M)
        A: Number of components in the PLS model
        cv_splits: An array of length N assigning a split to corresponding rows of X and Y. Each split will be used exactly once for validation with the remaining splits used for training.
        preprocessing_function: Callable that takes X_train, Y_train, X_val, Y_val and returns (X_train, Y_train, X_val, Y_val)
        metric_function: Callable that takes two arrays Y_true of shape (N x M) and Y_pred of shape (A x N x M) and returns Any.
        metric_names: List of strings with names to assign to each output of metric_function.

        Returns:
        Dictionairy with keys from metric_names and values that are outputs of metric_function(Y_true, Y_pred).
        """
        metric_value_lists = [[] for _ in metric_names]
        unique_splits = jnp.unique(cv_splits)
        for split in tqdm(unique_splits):
            train_idxs = jnp.nonzero(cv_splits != split)[0]
            val_idxs = jnp.nonzero(cv_splits == split)[0]
            metric_values = self._inner_cv(
                X, Y, train_idxs, val_idxs, A, preprocessing_function, metric_funtion
            )
            metric_value_lists = self.update_metric_value_lists(
                metric_value_lists, metric_values
            )
        return self.finalize_metric_values(metric_value_lists, metric_names)

    @partial(jax.jit, static_argnums=(0, 5, 6, 7))
    def _inner_cv(
        self,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        train_idxs: jnp.ndarray,
        val_idxs: jnp.ndarray,
        A: int,
        preprocessing_function: Callable[
            [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ],
        metric_function: Callable[[jnp.ndarray, jnp.ndarray], Tuple[Any]],
    ):
        print("Tracing inner CV...")
        X_train = jnp.take(X, train_idxs, axis=0)
        Y_train = jnp.take(Y, train_idxs, axis=0)

        X_val = jnp.take(X, val_idxs, axis=0)
        Y_val = jnp.take(Y, val_idxs, axis=0)
        X_train, Y_train, X_val, Y_val = preprocessing_function(
            X_train, Y_train, X_val, Y_val
        )
        metric_values = self.stateless_fit_predict_eval(
            X_train, Y_train, A, X_val, Y_val, metric_function
        )
        return metric_values

    def update_metric_value_lists(self, metric_value_lists, metric_values):
        for j, m in enumerate(metric_values):
            metric_value_lists[j].append(m)
        return metric_value_lists

    def finalize_metric_values(
        self, metrics_results: list[list[Any]], metric_names: list[str]
    ):
        metrics = {}
        for name, lst in zip(metric_names, metrics_results):
            metrics[name] = lst
        return metrics
