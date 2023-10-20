import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from functools import partial
import abc
from typing import Tuple, Callable, Union, Any
from tqdm import tqdm
import numpy as np


class PLSBase(abc.ABC):
    """
    Implements partial least-squares regression using Improved Kernel PLS by Dayal and MacGregor: https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23
    """

    def __init__(self) -> None:
        self.name = "PLS"

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
    def _fit_helper(
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
    def _step_2(self, XTY: jnp.ndarray, M: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
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
                w = w / jla.norm(w)
            else:
                XTYYTX = XTY @ XTY.T
                eig_vals, eig_vecs = jla.eigh(XTYYTX)
                w = eig_vecs[:, -1:]
        return w

    @partial(jax.jit, static_argnums=(0,))
    def _step_3(
        self, i: int, w: jnp.ndarray, P: jnp.ndarray, R: jnp.ndarray
    ) -> jnp.ndarray:
        print("Tracing step 3...")
        r = jnp.copy(w)
        r, P, w, R = jax.lax.fori_loop(0, i, self._step_3_body, (r, P, w, R))
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

        matrices = self._fit_helper(X_train, Y_train, A)
        B = matrices[0]
        Y_pred = X_test @ B
        return metric_function(Y_test, Y_pred)

    @partial(jax.jit, static_argnums=(0, 4, 5))
    def _inner_loocv(
        self,
        i: int,
        X_train_val: jnp.ndarray,
        Y_train_val: jnp.ndarray,
        A: int,
        metric_function: Callable[[jnp.ndarray, jnp.ndarray], Any],
    ) -> Any:
        print("Tracing inner LOOCV...")
        all_indices = jnp.arange(X_train_val.shape[0])
        train_indices = jnp.nonzero(all_indices != i, size=X_train_val.shape[0] - 1)[0]
        X_train = jnp.take(X_train_val, train_indices, axis=0)
        Y_train = jnp.take(Y_train_val, train_indices, axis=0)
        X_val = jnp.take(X_train_val, jnp.array([i]), axis=0)
        Y_val = jnp.take(Y_train_val, jnp.array([i]), axis=0)
        metric_values = self.stateless_fit_predict_eval(
            X_train, Y_train, A, X_val, Y_val, metric_function
        )
        return metric_values

    def loocv(
        self,
        X_train_val: jnp.ndarray,
        Y_train_val: jnp.ndarray,
        A: int,
        metric_function: Callable[[jnp.ndarray, jnp.ndarray], Any],
        metric_names: list[str],
    ) -> dict[str]:
        metric_value_lists = [[] for _ in metric_names]
        i = 0  # The first iteration includes JIT compilation so let's keep it outside tqdm's timing estimate
        metric_values = self._inner_loocv(
            i, X_train_val, Y_train_val, A, metric_function
        )
        metric_value_lists = self.update_metric_value_lists(
            metric_value_lists, metric_values
        )
        for i in tqdm(
            range(1, X_train_val.shape[0]), initial=1, total=X_train_val.shape[0]
        ):
            metric_values = self._inner_loocv(
                i, X_train_val, Y_train_val, A, metric_function
            )
            metric_value_lists = self.update_metric_value_lists(
                metric_value_lists, metric_values
            )
        return self.finalize_metric_values(metric_value_lists, metric_names)

    def update_metric_value_lists(self, metric_value_lists, metric_values):
        for j, m in enumerate(metric_values):
            metric_value_lists[j].append(m)
        return metric_value_lists

    def finalize_metric_values(self, metrics_results, metric_names):
        metrics = {}
        for name, lst in zip(metric_names, metrics_results):
            metrics[name] = np.asarray(lst)
        return metrics

    @partial(jax.jit, static_argnums=(0,))
    def _sample_with_replacement(
        self,
        prng_key: Union[jnp.ndarray, jax.Array],
        X_train: jnp.ndarray,
        Y_train: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        print("Tracing inner bootstrap...")
        X_train = jax.random.choice(
            prng_key, X_train, shape=(X_train.shape[0],), axis=0
        )
        Y_train = jax.random.choice(
            prng_key, Y_train, shape=(Y_train.shape[0],), axis=0
        )
        return X_train, Y_train

    def bootstrap(
        self,
        seed: int,
        num_iters: int,
        A: int,
        X_train: jnp.ndarray,
        Y_train: jnp.ndarray,
        X_val: jnp.ndarray,
        Y_val: jnp.ndarray,
        metric_function: Callable[[jnp.ndarray, jnp.ndarray], Any],
        metric_names: list[str],
    ) -> dict[str]:
        metric_value_lists = [[] for _ in metric_names]
        prng_key = jax.random.PRNGKey(seed)
        prng_array = jax.random.split(prng_key, num_iters)

        # The first iteration includes JIT compilation so let's keep it outside tqdm's timing estimate
        X_train_bootsrap, Y_train_bootstrap = self._sample_with_replacement(
            prng_array[0], X_train, Y_train
        )
        metric_values = self.stateless_fit_predict_eval(
            X_train_bootsrap, Y_train_bootstrap, A, X_val, Y_val, metric_function
        )
        metric_value_lists = self.update_metric_value_lists(
            metric_value_lists, metric_values
        )
        for key in tqdm(prng_array[1:], initial=1, total=prng_array.shape[0]):
            X_train_bootsrap, Y_train_bootstrap = self._sample_with_replacement(
                key, X_train, Y_train
            )
            metric_values = self.stateless_fit_predict_eval(
                X_train_bootsrap, Y_train_bootstrap, A, X_val, Y_val, metric_function
            )
            metric_value_lists = self.update_metric_value_lists(
                metric_value_lists, metric_values
            )
        return self.finalize_metric_values(metric_value_lists, metric_names)
