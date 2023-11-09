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
    Description
    -----------
    Implements an abstract class for partial least-squares regression using Improved Kernel PLS by Dayal and MacGregor: https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23.

    Implementations of concrete classes exist for both Improved Kernel PLS Algorithm #1 and Improved Kernel PLS Algorithm #2.

    Parameters
    ----------
    `reverse_differentiable`: bool, optional (default=False). Whether to make the implementation end-to-end differentiable. The differentiable version is slightly slower. Results among the two versions are identical.

    `name` : str, optional (default=\"Improved Kernel PLS Algorithm\"). Assigns a name to the instance of the class.

    `verbose` : bool, optional (default=False). If True, each sub-function will print when it will be JIT compiled. This can be useful to track if recompilation is triggered due to passing inputs with different shapes.
    """

    def __init__(
        self,
        name: str = "Improved Kernel PLS Algorithm",
        reverse_differentiable: bool = False,
        verbose: bool = False,
    ) -> None:
        self.name = name
        self.reverse_differentiable = reverse_differentiable
        self.verbose = verbose

    def _weight_warning(self, arg, *args):
        """
        Description
        -----------
        Display a warning message if the weight is close to zero.

        Parameters
        ----------
        `arg` : tuple
            A tuple containing the component index and the weight norm.

        `*args` : Any
            Placeholder for unused arguments.

        Warns
        -----
        `UserWarning`
            If the weight norm is below machine epsilon for float64, a warning message is displayed.

        Notes
        -----
        This method issues a warning if the weight becomes close to zero during the PLS algorithm. It provides a hint about potential instability in results with a higher number of components.
        """
        i, norm = arg
        if np.isclose(norm, 0, atol=np.finfo(np.float64).eps, rtol=0):
            warnings.warn(
                f"Weight is close to zero. Results with A = {i} component(s) or higher may be unstable."
            )

    @partial(jax.jit, static_argnums=0)
    def _compute_regression_coefficients(
        self, b_last: jnp.ndarray, r: jnp.ndarray, q: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Description
        -----------
        Compute the regression coefficients in the PLS algorithm.

        Parameters
        ----------
        `b_last` : Array of shape (K, M)
            The previous regression coefficient matrix.

        `r` : Array of shape (K, 1)
            The orthogonal weight vector for the current component.

        `q` : Array of shape (M, 1)
            The loadings vector for the response variables.

        Returns
        -------
        `b` : Array of shape (K, M)
            The updated regression coefficient matrix for the current component.

        Notes
        -----
        This method computes the regression coefficients matrix for the current component in the PLS algorithm, incorporating the orthogonal weight vector and loadings vector.
        """
        b = b_last + r @ q.T
        return b

    @abc.abstractmethod
    @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def _get_initial_matrices(
        self, A, K, M
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Description
        -----------
        Initialize the matrices used in the PLS algorithm.

        Parameters
        ----------
        `A` : int
            Number of components in the PLS model.

        `K` : int
            Number of predictor variables.

        `M` : int
            Number of response variables.

        Returns
        -------
        A tuple of initialized matrices:
        `B` : Array of shape (A, K, M)
            PLS regression coefficients tensor.

        `W` : Array of shape (A, K)
            PLS weights matrix for X.

        `P` : Array of shape (A, K)
            PLS loadings matrix for X.

        `Q` : Array of shape (A, M)
            PLS Loadings matrix for Y.

        `R` : Array of shape (A, K)
            PLS weights matrix to compute scores T directly from the original X.

        Notes
        -----
        This abstract method is responsible for initializing various matrices used in the PLS algorithm, including regression coefficients, weights, loadings, and orthogonal weights.
        """
        B = jnp.zeros(shape=(A, K, M), dtype=jnp.float64)
        W = jnp.zeros(shape=(A, K), dtype=jnp.float64)
        P = jnp.zeros(shape=(A, K), dtype=jnp.float64)
        Q = jnp.zeros(shape=(A, M), dtype=jnp.float64)
        R = jnp.zeros(shape=(A, K), dtype=jnp.float64)
        return B, W, P, Q, R

    @partial(jax.jit, static_argnums=0)
    def _compute_XT(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Description
        -----------
        Compute the transposed predictor variable matrix.

        Parameters
        ----------
        `X` : Array of shape (N, K)
            Predictor variables matrix.

        Returns
        -------
        `XT` : Array of shape (K, N)
            Transposed predictor variables matrix.

        Notes
        -----
        This method calculates the transposed predictor variables matrix from the original predictor variables matrix.
        """
        return X.T

    @partial(jax.jit, static_argnums=0)
    def _compute_initial_XTY(self, XT: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        """
        Description
        -----------
        Compute the initial cross-covariance matrix of the predictor variables and the response variables.

        Parameters
        ----------
        `XT` : Array of shape (K, N)
            Transposed predictor variables matrix.

        `Y` : Array of shape (N, M)
            Response variables matrix.

        Returns
        -------
        `XTY` : Array of shape (K, M)
            Initial cross-covariance matrix of the predictor variables and the response variables.

        Notes
        -----
        This method calculates the initial cross-covariance matrix of the predictor variables and the response variables.
        """
        return XT @ Y

    @partial(jax.jit, static_argnums=0)
    def _compute_XTX(self, XT: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """
        Description
        -----------
        Compute the product of the transposed predictor variables matrix and the predictor variables matrix.

        Parameters
        ----------
        `XT` : Array of shape (K, N)
            Transposed predictor variables matrix.

        `X` : Array of shape (N, K)
            Predictor variables matrix.

        Returns
        -------
        `XTX` : Array of shape (K, K)
            Product of transposed predictor variables and predictor variables.

        Notes
        -----
        This method calculates the product of the transposed predictor variables matrix and the predictor variables matrix.
        """
        return XT @ X

    @abc.abstractmethod
    @partial(jax.jit, static_argnums=0)
    def _step_1(self):
        """
        Description
        -----------
        Abstract method representing the first step in the PLS algorithm. This step should be implemented in concrete PLS classes.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method represents the first step of the PLS algorithm and should be implemented in concrete PLS classes.
        """

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def _step_2(
        self, XTY: jnp.ndarray, M: jnp.ndarray, K: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.float64]:
        """
        Description
        -----------
        The second step of the PLS algorithm. Computes the next weight vector and the associated norm.

        Parameters
        ----------
        `XTY` : Array of shape (K, M)
            The cross-covariance matrix of the predictor variables and the response variables.

        `M` : int
            Number of response variables.

        `K` : int
            Number of predictor variables.

        Returns
        -------
        `w` : Array of shape (K, 1)
            The next weight vector for the PLS algorithm.

        `norm` : float
            The l2 norm of the weight vector `w`.

        Notes
        -----
        This method computes the next weight vector `w` for the PLS algorithm and its associated norm. It is an essential step in the PLS algorithm.
        """
        if self.verbose:
            print(f"_step_2 for {self.name} will be JIT compiled...")
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

    @partial(jax.jit, static_argnums=(0, 1))
    def _step_3(
        self, i: int, w: jnp.ndarray, P: jnp.ndarray, R: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Description
        -----------
        The third step of the PLS algorithm. Computes the orthogonal weight vectors.

        Parameters
        ----------
        `i` : int
            The current component number in the PLS algorithm.

        `w` : Array of shape (K, 1)
            The current weight vector.

        `P` : Array of shape (A, K)
            The loadings matrix for the predictor variables.

        `R` : Array of shape (A, K)
            The weights matrix to compute scores `T` directly from the original predictor variables.

        Returns
        -------
        `r` : Array of shape (K, 1)
            The orthogonal weight vector for the current component.

        Notes
        -----
        This method computes the orthogonal weight vector `r` for the current component in the PLS algorithm. It is a key step for calculating the loadings and weights matrices.
        """
        if self.verbose:
            print(f"_step_3 for {self.name} will be JIT compiled...")
        r = jnp.copy(w)
        r, P, w, R = jax.lax.fori_loop(0, i, self._step_3_body, (r, P, w, R))
        return r

    @partial(jax.jit, static_argnums=0)
    def _step_3_body(
        self, j: int, carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Description
        -----------
        The body of the third step of the PLS algorithm. Iteratively computes orthogonal weight vectors.

        Parameters
        ----------
        `j` : int
            The current iteration index.

        `carry` : Tuple of arrays
            A tuple containing weight vectors and matrices used in the PLS algorithm.

        Returns
        -------
        Updated weight vectors and matrices for the PLS algorithm.

        Notes
        -----
        This method is the body of the third step of the PLS algorithm and iteratively computes orthogonal weight vectors used in the PLS algorithm.
        """
        if self.verbose:
            print(f"_step_3_body for {self.name} will be JIT compiled...")
        r, P, w, R = carry
        r = r - P[j].reshape(-1, 1).T @ w * R[j].reshape(-1, 1)
        return r, P, w, R

    @abc.abstractmethod
    @partial(jax.jit, static_argnums=0)
    def _step_4(self):
        """
        Description
        -----------
        Abstract method representing the fourth step in the PLS algorithm. This step should be implemented in concrete PLS classes.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method represents the fourth step of the PLS algorithm and should be implemented in concrete PLS classes.
        """

    @partial(jax.jit, static_argnums=0)
    def _step_5(
        self, XTY: jnp.ndarray, p: jnp.ndarray, q: jnp.ndarray, tTt: jnp.ndarray
    ) -> jnp.ndarray:
        if self.verbose:
            print(f"_step_5 for {self.name} will be JIT compiled...")
        return XTY - (p @ q.T) * tTt

    @abc.abstractmethod
    @partial(jax.jit, static_argnums=0)
    def _main_loop_body(self):
        """
        Description
        -----------
        Abstract method representing the main loop body in the PLS algorithm. This method should be implemented in concrete PLS classes.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method represents the main loop body of the PLS algorithm and should be implemented in concrete PLS classes.
        """

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
        """
        Parameters
        ----------
        `X` : Array of shape (N, K)
            Predictor variables. The precision should be at least float64 for reliable results.

        `Y` : Array of shape (N, M)
            Response variables. The precision should be at least float64 for reliable results.

        `A` : int
            Number of components in the PLS model.

        Returns
        -------
        `B` : Array of shape (A, K, M)
            PLS regression coefficients tensor.

        `W` : Array of shape (A, K)
            PLS weights matrix for X.

        `P` : Array of shape (A, K)
            PLS loadings matrix for X.

        `Q` : Array of shape (A, M)
            PLS Loadings matrix for Y.

        `R` : Array of shape (A, K)
            PLS weights matrix to compute scores T directly from original X.

        `T` : Array of shape (A, N)
            PLS scores matrix of X. Only Returned for Improved Kernel PLS Algorithm #1.

        Warns
        -----
        `UserWarning`.
            If at any point during iteration over the number of components `A`, the residual goes below machine precision for jnp.float64.

        See Also
        --------
        fit : Performs the same operation but stores the output matrices in the class instance instead of returning them.

        Notes
        -----
        For optimization purposes, the internal representation of all matrices (except B) is transposed from the usual representation.
        """

    @abc.abstractmethod
    def fit(self, X: jnp.ndarray, Y: jnp.ndarray, A: int) -> None:
        """
        Description
        -----------
        Fits Improved Kernel PLS Algorithm #1 on `X` and `Y` using `A` components.

        Parameters
        ----------
        `X` : Array of shape (N, K)
            Predictor variables. The precision should be at least float64 for reliable results.

        `Y` : Array of shape (N, M)
            Response variables. The precision should be at least float64 for reliable results.

        `A` : int
            Number of components in the PLS model.

        Assigns
        -------
        `self.B` : Array of shape (A, K, M)
            PLS regression coefficients tensor.

        `self.W` : Array of shape (K, A)
            PLS weights matrix for X.

        `self.P` : Array of shape (K, A)
            PLS loadings matrix for X.

        `self.Q` : Array of shape (M, A)
            PLS Loadings matrix for Y.

        `self.R` : Array of shape (K, A)
            PLS weights matrix to compute scores T directly from original X.

        `self.T` : Array of shape (N, A)
            PLS scores matrix of X. Only assigned for Improved Kernel PLS Algorithm #1.

        Returns
        -------
        `None`.

        Warns
        -----
        `UserWarning`.
            If at any point during iteration over the number of components `A`, the residual goes below machine precision for jnp.float64.

        See Also
        --------
        `stateless_fit` : Performs the same operation but returns the output matrices instead of storing them in the class instance.
        """

    @partial(jax.jit, static_argnums=(0, 3))
    def stateless_predict(
        self, X: jnp.ndarray, B: jnp.ndarray, n_components: None | int = None
    ) -> jnp.ndarray:
        """
        Description
        -----------
        Predicts with Improved Kernel PLS Algorithm #1 on `X` with `B` using `n_components` components. If `n_components` is None, then predictions are returned for all number of components.

        Parameters
        ----------
        `X` : Array of shape (N, K)
            Predictor variables. The precision should be at least float64 for reliable results.

        `B` : Array of shape (A, K, M)
            PLS regression coefficients tensor.

        `n_components` : int or None, optional
            Number of components in the PLS model. If None, then all number of components are used.

        Returns
        -------
        `Y_pred` : Array of shape (N, M) or (A, N, M)
            If `n_components` is an int, then an array of shape (N, M) with the predictions for that specific number of components is used. If `n_components` is None, returns a prediction for each number of components up to `A`.

        See Also
        --------
        `predict` : Performs the same operation but uses the class instance of `B`.
        """
        if self.verbose:
            print(f"stateless_predict for {self.name} will be JIT compiled...")
        if n_components is None:
            return X @ B
        else:
            return X @ B[n_components - 1]

    def predict(self, X: jnp.ndarray, n_components: None | int = None) -> jnp.ndarray:
        """
        Description
        -----------
        Predicts with Improved Kernel PLS Algorithm #1 on `X` with `B` using `n_components` components. If `n_components` is None, then predictions are returned for all number of components.

        Parameters
        ----------
        `X` : Array of shape (N, K)
            Predictor variables. The precision should be at least float64 for reliable results.

        `n_components` : int or None, optional
            Number of components in the PLS model. If None, then all number of components are used.

        Returns
        -------
        `Y_pred` : Array of shape (N, M) or (A, N, M)
            If `n_components` is an int, then an array of shape (N, M) with the predictions for that specific number of components is used. If `n_components` is None, returns a prediction for each number of components up to `A`.

        See Also
        --------
        `stateless_predict` : Performs the same operation but uses an input `B` instead of the one stored in the class instance.
        """
        if n_components is None:
            return X @ self.B
        else:
            return X @ self.B[n_components - 1]

    @partial(jax.jit, static_argnums=(0, 3, 6))
    def stateless_fit_predict_eval(
        self,
        X_train: jnp.ndarray,
        Y_train: jnp.ndarray,
        A: int,
        X_test: jnp.ndarray,
        Y_test: jnp.ndarray,
        metric_function: Callable[[jnp.ndarray, jnp.ndarray], Any],
    ) -> Any:
        """
        Description
        -----------
        Calls `B = stateless_fit(X_train, Y_train, A)[0]`. Then Calls `Y_pred = stateless_predict(X_test, B)`. `Y_pred` is an array of shape (A, N, M). Then evaluates and returns the result of `metric_function(Y_test, Y_pred)`.

        Parameters
        ----------
        `X_train` : Array of shape (N_train, K)
            Predictor variables. The precision should be at least float64 for reliable results.

        `Y_train` : Array of shape (N_train, M)
            Response variables. The precision should be at least float64 for reliable results.

        `A` : int
            Number of components in the PLS model.

        `X_test` : Array of shape (N_test, K)
            Predictor variables. The precision should be at least float64 for reliable results.

        `Y_test` : Array of shape (N_test, M)
            Response variables. The precision should be at least float64 for reliable results.

        `metric_function` : Callable receiving arrays `Y_test` of shape (N, M) and `Y_pred` (A, N, M) and returns Any
            Computes a metric based on true values `Y_test` and predicted values `Y_pred`. `Y_pred` contains a prediction for all `A` components.

        Returns
        -------
        `metric_function(Y_test, Y_pred)` : Any.

        See Also
        --------
        `stateless_fit` : Fits on `X_train` and `Y_train` using `A` components. Then returns the internal matrices instead of storing them in the class instance.

        `stateless_predict` : Computes `Y_pred` given predictor variables `X` and regression tensor `B` and optionally `A` components.
        """
        if self.verbose:
            print(f"stateless_fit_predict_eval for {self.name} will be JIT compiled...")

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
        metric_function: Callable[[jnp.ndarray, jnp.ndarray], Any],
        metric_names: list[str],
        show_progress=True,
    ) -> dict[str, Any]:
        """
        Description
        -----------
        Performs cross-validation for the Partial Least-Squares (PLS) model on given data.

        Parameters
        ----------
        `X` : Array of shape (N, K)
            Predictor variables. The precision should be at least float64 for reliable results.

        `Y` : Array of shape (N, M)
            Response variables. The precision should be at least float64 for reliable results.

        `A` : int
            Number of components in the PLS model.

        `cv_splits` : Array of shape (N,)
            An array defining cross-validation splits. Each unique value in `cv_splits` corresponds to a different fold.

        `preprocessing_function` : Callable receiving arrays `X_train`, `Y_train`, `X_val`, and `Y_val`
            A function that preprocesses the training and validation data for each fold. It should return preprocessed arrays for `X_train`, `Y_train`, `X_val`, and `Y_val`.

        `metric_function` : Callable receiving arrays `Y_test` and `Y_pred` and returning Any
            Computes a metric based on true values `Y_test` and predicted values `Y_pred`. `Y_pred` contains a prediction for all `A` components.

        `metric_names` : list of str
            A list of names for the metrics used for evaluation.

        `show_progress` : bool, optional (default=True)
            If True, displays a progress bar for the cross-validation.

        Returns
        -------
        `metrics` : dict[str, Any]
            A dictionary containing evaluation metrics for each metric specified in `metric_names`. The keys are metric names, and the values are lists of metric values for each cross-validation fold.

        See Also
        --------
        `_inner_cv` : Performs cross-validation for a single fold and computes evaluation metrics.

        `_update_metric_value_lists` : Updates lists of metric values for each metric and fold.

        `_finalize_metric_values` : Organizes and finalizes the metric values into a dictionary for the specified metric names.

        stateless_fit_predict_eval : Fits the PLS model, makes predictions, and evaluates metrics for a given fold.

        Notes
        -----
        This method is used to perform cross-validation on the PLS model with different data splits and evaluate its performance using user-defined metrics.
        """
        metric_value_lists = [[] for _ in metric_names]
        unique_splits = jnp.unique(cv_splits)
        for split in tqdm(unique_splits, disable=not show_progress):
            train_idxs = jnp.nonzero(cv_splits != split)[0]
            val_idxs = jnp.nonzero(cv_splits == split)[0]
            metric_values = self._inner_cv(
                X, Y, train_idxs, val_idxs, A, preprocessing_function, metric_function
            )
            metric_value_lists = self._update_metric_value_lists(
                metric_value_lists, metric_values
            )
        return self._finalize_metric_values(metric_value_lists, metric_names)

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
        """
        Description
        -----------
        Performs cross-validation for a single fold of the data and computes evaluation metrics.

        Parameters
        ----------
        `X` : Array of shape (N, K)
            Predictor variables. The precision should be at least float64 for reliable results.

        `Y` : Array of shape (N, M)
            Response variables. The precision should be at least float64 for reliable results.

        `train_idxs` : Array of shape (N_train,)
            Indices of data points in the training set.

        `val_idxs` : Array of shape (N_val,)
            Indices of data points in the validation set.

        `A` : int
            Number of components in the PLS model.

        `preprocessing_function` : Callable receiving arrays `X_train`, `Y_train`, `X_val`, and `Y_val`
            A function that preprocesses the training and validation data for each fold. It should return preprocessed arrays for `X_train`, `Y_train`, `X_val`, and `Y_val.

        `metric_function` : Callable receiving arrays `Y_test` and `Y_pred` and returning Any
            Computes a metric based on true values `Y_test` and predicted values `Y_pred`. `Y_pred` contains a prediction for all `A` components.

        Returns
        -------
        `metric_values` : Tuple of Any
            A tuple of metric values based on the true and predicted values for a single fold.

        Notes
        -----
        This method performs cross-validation for a single fold of the data, including preprocessing, fitting, predicting, and evaluating the PLS model.
        """
        if self.verbose:
            print(f"_inner_cv for {self.name} will be JIT compiled...")
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

    def _update_metric_value_lists(self, metric_value_lists, metric_values):
        """
        Description
        -----------
        Updates lists of metric values for each metric and fold during cross-validation.

        Parameters
        ----------
        `metric_value_lists` : list of list of Any
            Lists of metric values for each metric and fold.

        `metric_values` : list of Any
            Metric values for a single fold.

        Returns
        -------
        `metric_value_lists` : list of list of Any
            Updated lists of metric values for each metric and fold.

        Notes
        -----
        This method updates the lists of metric values for each metric and fold during cross-validation.
        """
        for j, m in enumerate(metric_values):
            metric_value_lists[j].append(m)
        return metric_value_lists

    def _finalize_metric_values(
        self, metrics_results: list[list[Any]], metric_names: list[str]
    ):
        """
        Description
        -----------
        Organizes and finalizes the metric values into a dictionary for the specified metric names.

        Parameters
        ----------
        `metrics_results` : list of list of Any
            Lists of metric values for each metric and fold.

        `metric_names` : list of str
            A list of names for the metrics used for evaluation.

        Returns
        -------
        `metrics` : dict[str, Any]
            A dictionary containing evaluation metrics for each metric specified in `metric_names`. The keys are metric names, and the values are lists of metric values for each cross-validation fold.

        Notes
        -----
        This method organizes and finalizes the metric values into a dictionary for the specified metric names, making it easy to analyze the cross-validation results.
        """
        metrics = {}
        for name, lst in zip(metric_names, metrics_results):
            metrics[name] = lst
        return metrics
