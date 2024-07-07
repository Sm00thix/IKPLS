"""
Implements an abstract class for partial least-squares regression using Improved Kernel
PLS by Dayal and MacGregor.

Implementations of concrete classes exist for both Improved Kernel PLS Algorithm #1
and Improved Kernel PLS Algorithm #2.

For more details, refer to the paper:
"Improved Kernel Partial Least Squares Regression" by Dayal and MacGregor.

Author: Ole-Christian Galbo Engstrøm
E-mail: ole.e@di.ku.dk
"""

import abc
import warnings
from collections.abc import Callable
from functools import partial
from typing import Any, Tuple, Union

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import numpy as np
from jax.typing import ArrayLike, DTypeLike
from numpy import typing as npt
from tqdm import tqdm


class PLSBase(abc.ABC):
    """
    Implements an abstract class for partial least-squares regression using Improved
    Kernel PLS by Dayal and MacGregor:
    https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

    Implementations of concrete classes exist for both Improved Kernel PLS Algorithm #1
    and Improved Kernel PLS Algorithm #2.

    Parameters
    ----------
    center_X : bool, default=True
        Whether to center the predictor variables (X) before fitting. If True, then the
        mean of the training data is subtracted from the predictor variables.

    center_Y : bool, default=True
        Whether to center the response variables (Y) before fitting. If True, then the
        mean of the training data is subtracted from the response variables.

    scale_X : bool, default=True
        Whether to scale the predictor variables (X) before fitting. If True, then the
        data is scaled using Bessel's correction for the unbiased estimate of the
        sample standard deviation.

    scale_Y : bool, default=True
        Whether to scale the response variables (Y) before fitting. If True, then the
        data is scaled using Bessel's correction for the unbiased estimate of the
        sample standard deviation.

    copy : bool, optional, default=True
        Whether to copy `X` and `Y` in fit before potentially applying centering and
        scaling. If True, then the data is copied before fitting. If False, and `dtype`
        matches the type of `X` and `Y`, then centering and scaling is done inplace,
        modifying both arrays.

    dtype : jnp.float, optional, default=jnp.float64
        The float datatype to use in computation of the PLS algorithm. Using a lower
        precision than float64 will yield significantly worse results when using an
        increasing number of components due to propagation of numerical errors.

    reverse_differentiable: bool, optional, default=False
        Whether to make the implementation end-to-end differentiable. The
        differentiable version is slightly slower. Results among the two versions are
        identical.

    verbose : bool, optional, default=False
        If True, each sub-function will print when it will be JIT compiled. This can be
        useful to track if recompilation is triggered due to passing inputs with
        different shapes.

    Notes
    -----
    Any centering and scaling is undone before returning predictions with `fit` to
    ensure that predictions are on the original scale. If both centering and scaling
    are True, then the data is first centered and then scaled.
    """

    def __init__(
        self,
        center_X: bool = True,
        center_Y: bool = True,
        scale_X: bool = True,
        scale_Y: bool = True,
        copy: bool = True,
        dtype: DTypeLike = jnp.float64,
        reverse_differentiable: bool = False,
        verbose: bool = False,
    ) -> None:
        self.center_X = center_X
        self.center_Y = center_Y
        self.scale_X = scale_X
        self.scale_Y = scale_Y
        self.copy = copy
        self.dtype = dtype
        self.eps = jnp.finfo(self.dtype).eps
        self.reverse_differentiable = reverse_differentiable
        self.verbose = verbose
        self.name = "Improved Kernel PLS Algorithm"
        self.B = None
        self.W = None
        self.P = None
        self.Q = None
        self.R = None
        self.X_mean = None
        self.Y_mean = None
        self.X_std = None
        self.Y_std = None

    def _weight_warning(self, arg: Tuple[npt.NDArray[np.int_], npt.NDArray[np.float_]]):
        """
        Display a warning message if the weight is close to zero.

        Parameters
        ----------
        arg : tuple
            A tuple containing the component index and the weight norm.

        Warns
        -----
        UserWarning.
            If the weight norm is below machine epsilon, a warning message is
            displayed.

        Notes
        -----
        This method issues a warning if the weight becomes close to zero during the PLS
        algorithm. It provides a hint about potential instability in results with a
        higher number of components.
        """
        i, norm = arg
        if np.isclose(norm, 0, atol=np.finfo(self.dtype).eps, rtol=0):
            with warnings.catch_warnings():
                warnings.simplefilter("always", UserWarning)
                warnings.warn(
                    f"Weight is close to zero. Results with A = {i} component(s) or "
                    "higher may be unstable."
                )

    @partial(jax.jit, static_argnums=0)
    def _compute_regression_coefficients(
        self, b_last: jax.Array, r: jax.Array, q: jax.Array
    ) -> jax.Array:
        """
        Compute the regression coefficients in the PLS algorithm.

        Parameters
        ----------
        b_last : Array of shape (K, M)
            The previous regression coefficient matrix.

        r : Array of shape (K, 1)
            The orthogonal weight vector for the current component.

        q : Array of shape (M, 1)
            The loadings vector for the response variables.

        Returns
        -------
        b : Array of shape (K, M)
            The updated regression coefficient matrix for the current component.

        Notes
        -----
        This method computes the regression coefficients matrix for the current
        component in the PLS algorithm, incorporating the orthogonal weight vector and
        loadings vector.
        """
        b = b_last + r @ q.T
        return b

    @partial(jax.jit, static_argnums=0)
    def _initialize_input_matrices(self, X: jax.Array, Y: jax.Array):
        """
        Initialize the input matrices used in the PLS algorithm.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables matrix.

        Y : Array of shape (N, M) or (N,)
            Response variables matrix.
        """
        X = jnp.asarray(X, dtype=self.dtype)
        Y = jnp.asarray(Y, dtype=self.dtype)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        return X, Y

    @partial(jax.jit, static_argnums=0)
    def get_mean(self, A: ArrayLike):
        """
        Get the mean of the a matrix.

        Parameters
        ----------
        A : Array of shape (N, K) or (N, M)
            Predictor variables matrix or response variables matrix.

        Returns
        -------
        A_mean : Array of shape (1, K) or (1, M)
            Mean of the predictor variables or response variables.
        """
        if self.verbose:
            print(f"get_means for {self.name} will be JIT compiled...")

        A_mean = jnp.mean(A, axis=0, dtype=self.dtype, keepdims=True)
        return A_mean

    @partial(jax.jit, static_argnums=0)
    def get_std(self, A: ArrayLike):
        """
        Get the standard deviation of a matrix.

        Parameters
        ----------
        A : Array of shape (N, K) or (N, M)
            Predictor variables matrix or response variables matrix.

        Returns
        -------
        A_std : Array of shape (1, K) or (1, M)
            Sample standard deviation of the predictor variables or response variables.
        """
        if self.verbose:
            print(f"get_stds for {self.name} will be JIT compiled...")

        A_std = jnp.std(A, axis=0, dtype=self.dtype, keepdims=True, ddof=1)
        A_std = jnp.where(jnp.abs(A_std) <= self.eps, 1, A_std)
        return A_std

    @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 7))
    def _center_scale_input_matrices(
        self,
        X: jax.Array,
        Y: jax.Array,
        center_X: bool,
        center_Y: bool,
        scale_X: bool,
        scale_Y: bool,
        copy: bool,
    ):
        """
        Preprocess the input matrices based on the centering and scaling parameters.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables matrix.

        Y : Array of shape (N, M)
            Response variables matrix.

        center_X : bool
            Whether to center the predictor variables (X) before fitting.

        center_Y : bool
            Whether to center the response variables (Y) before fitting.

        scale_X : bool
            Whether to scale the predictor variables (X) before fitting.

        scale_Y : bool
            Whether to scale the response variables (Y) before fitting.

        Returns
        -------
        X : Array of shape (N, K)
            Potentially centered and potentially scaled variables matrix.

        Y : Array of shape (N, M)
            Potentially centered and potentially scaled response variables matrix.
        """
        if self.verbose:
            print(f"_preprocess_input_matrices for {self.name} will be JIT compiled...")

        if (center_X or scale_X) and copy:
            X = X.copy()

        if (center_Y or scale_Y) and copy:
            Y = Y.copy()

        if center_X:
            X_mean = self.get_mean(X)
            X = X - X_mean
        else:
            X_mean = None

        if center_Y:
            Y_mean = self.get_mean(Y)
            Y = Y - Y_mean
        else:
            Y_mean = None

        if scale_X:
            X_std = self.get_std(X)
            X = X / X_std
        else:
            X_std = None

        if scale_Y:
            Y_std = self.get_std(Y)
            Y = Y / Y_std
        else:
            Y_std = None

        return X, Y, X_mean, Y_mean, X_std, Y_std

    @abc.abstractmethod
    @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def _get_initial_matrices(
        self, A: int, K: int, M: int
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Initialize the matrices used in the PLS algorithm.

        Parameters
        ----------
        A : int
            Number of components in the PLS model.

        K : int
            Number of predictor variables.

        M : int
            Number of response variables.

        Returns
        -------
        A tuple of initialized matrices:
        B : Array of shape (A, K, M)
            PLS regression coefficients tensor.

        W : Array of shape (A, K)
            PLS weights matrix for X.

        P : Array of shape (A, K)
            PLS loadings matrix for X.

        Q : Array of shape (A, M)
            PLS Loadings matrix for Y.

        R : Array of shape (A, K)
            PLS weights matrix to compute scores T directly from the original X.

        Notes
        -----
        This abstract method is responsible for initializing various matrices used in
        the PLS algorithm, including regression coefficients, weights, loadings, and
        orthogonal weights.
        """
        B = jnp.zeros(shape=(A, K, M), dtype=self.dtype)
        W = jnp.zeros(shape=(A, K), dtype=self.dtype)
        P = jnp.zeros(shape=(A, K), dtype=self.dtype)
        Q = jnp.zeros(shape=(A, M), dtype=self.dtype)
        R = jnp.zeros(shape=(A, K), dtype=self.dtype)
        return B, W, P, Q, R

    @partial(jax.jit, static_argnums=0)
    def _compute_XT(self, X: jax.Array) -> jax.Array:
        """
        Compute the transposed predictor variable matrix.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables matrix.

        Returns
        -------
        XT : Array of shape (K, N)
            Transposed predictor variables matrix.

        Notes
        -----
        This method calculates the transposed predictor variables matrix from the
        original predictor variables matrix.
        """
        return X.T

    @partial(jax.jit, static_argnums=0)
    def _compute_initial_XTY(self, XT: jax.Array, Y: jax.Array) -> jax.Array:
        """
        Compute the initial cross-covariance matrix of the predictor variables and the
        response variables.

        Parameters
        ----------
        XT : Array of shape (K, N)
            Transposed predictor variables matrix.

        Y : Array of shape (N, M)
            Response variables matrix.

        Returns
        -------
        XTY : Array of shape (K, M)
            Initial cross-covariance matrix of the predictor variables and the response
            variables.

        Notes
        -----
        This method calculates the initial cross-covariance matrix of the predictor
        variables and the response variables.
        """
        return XT @ Y

    @partial(jax.jit, static_argnums=0)
    def _compute_XTX(self, XT: jax.Array, X: jax.Array) -> jax.Array:
        """
        Compute the product of the transposed predictor variables matrix and the
        predictor variables matrix.

        Parameters
        ----------
        XT : Array of shape (K, N)
            Transposed predictor variables matrix.

        X : Array of shape (N, K)
            Predictor variables matrix.

        Returns
        -------
        XTX : Array of shape (K, K)
            Product of transposed predictor variables and predictor variables.

        Notes
        -----
        This method calculates the product of the transposed predictor variables matrix
        and the predictor variables matrix.
        """
        return XT @ X

    @abc.abstractmethod
    @partial(jax.jit, static_argnums=0)
    def _step_1(self, X: jax.Array, Y: jax.Array):
        """
        Abstract method representing the first step in the PLS algorithm. This step
        should be implemented in concrete PLS classes.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method represents the first step of the PLS algorithm and should be
        implemented in concrete PLS classes.
        """

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def _step_2(
        self, XTY: jax.Array, M: int, K: int
    ) -> Tuple[jax.Array, DTypeLike]:
        """
        The second step of the PLS algorithm. Computes the next weight vector and the
        associated norm.

        Parameters
        ----------
        XTY : Array of shape (K, M)
            The cross-covariance matrix of the predictor variables and the response
            variables.

        M : int
            Number of response variables.

        K : int
            Number of predictor variables.

        Returns
        -------
        w : Array of shape (K, 1)
            The next weight vector for the PLS algorithm.

        norm : float
            The l2 norm of the weight vector `w`.

        Notes
        -----
        This method computes the next weight vector `w` for the PLS algorithm and its
        associated norm.
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

    def _step_3_base(self, i: int, w: jax.Array, P: jax.Array, R: jax.Array) -> jax.Array:
        """
        The third step of the PLS algorithm. Computes the orthogonal weight vectors.

        Parameters
        ----------
        i : int
            The current component number in the PLS algorithm.

        w : Array of shape (K, 1)
            The current weight vector.

        P : Array of shape (A, K)
            The loadings matrix for the predictor variables.

        R : Array of shape (A, K)
            The weights matrix to compute scores `T` directly from the original
            predictor variables.

        Returns
        -------
        r : Array of shape (K, 1)
            The orthogonal weight vector for the current component.

        Notes
        -----
        This method computes the orthogonal weight vector `r` for the current component
        in the PLS algorithm. It is a key step for calculating the loadings and weights
        matrices.
        """
        if self.verbose:
            print(f"_step_3 for {self.name} will be JIT compiled...")
        r = jnp.copy(w)
        r, P, w, R = jax.lax.fori_loop(0, i, self._step_3_body, (r, P, w, R))
        return r

    def _step_3(self, i: int, w: jax.Array, P: jax.Array, R: jax.Array) -> jax.Array:
        """
        This is an API to the third step of the PLS algorithm. Computes the orthogonal
        weight vectors.

        Parameters
        ----------
        i : int
            The current component number in the PLS algorithm.

        w : Array of shape (K, 1)
            The current weight vector.

        P : Array of shape (A, K)
            The loadings matrix for the predictor variables.

        R : Array of shape (A, K)
            The weights matrix to compute scores `T` directly from the original
            predictor variables.

        Returns
        -------
        r : Array of shape (K, 1)
            The orthogonal weight vector for the current component.

        Notes
        -----
        This method compiles _step_3_base which in turn computes the orthogonal weight
        vector `r` for the current component in the PLS algorithm. It is a key step for
        calculating the loadings and weights matrices.

        See Also
        --------
        _step_3_base : The third step of the PLS algorithm.
        """
        if self.reverse_differentiable:
            jax.jit(self._step_3_base, static_argnums=(0, 1))
            return self._step_3_base(i, w, P, R)
        else:
            jax.jit(self._step_3_base, static_argnums=(0))
            return self._step_3_base(i, w, P, R)

    @partial(jax.jit, static_argnums=0)
    def _step_3_body(
        self, j: int, carry: Tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        The body of the third step of the PLS algorithm. Iteratively computes
        orthogonal weight vectors.

        Parameters
        ----------
        j : int
            The current iteration index.

        carry : Tuple of arrays
            A tuple containing weight vectors and matrices used in the PLS algorithm.

        Returns
        -------
        carry : Tuple of arrays
            Updated weight vectors and matrices used in the PLS algorithm.

        Notes
        -----
        This method is the body of the third step of the PLS algorithm and iteratively
        computes orthogonal weight vectors used in the PLS algorithm.
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
        Abstract method representing the fourth step in the PLS algorithm. This step
        should be implemented in concrete PLS classes.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method represents the fourth step of the PLS algorithm and should be
        implemented in concrete PLS classes.
        """

    @partial(jax.jit, static_argnums=0)
    def _step_5(
        self, XTY: jax.Array, p: jax.Array, q: jax.Array, tTt: jax.Array
    ) -> jax.Array:
        if self.verbose:
            print(f"_step_5 for {self.name} will be JIT compiled...")
        return XTY - (p @ q.T) * tTt

    @abc.abstractmethod
    @partial(jax.jit, static_argnums=0)
    def _main_loop_body(self):
        """
        Abstract method representing the main loop body in the PLS algorithm. This
        method should be implemented in concrete PLS classes.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method represents the main loop body of the PLS algorithm and should be
        implemented in concrete PLS classes.
        """

    @abc.abstractmethod
    @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 7, 8))
    def stateless_fit(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        A: int,
        center_X: bool = True,
        center_Y: bool = True,
        scale_X: bool = True,
        scale_Y: bool = True,
        copy: bool = True,
    ) -> Union[
        Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ]:
        """
        Fits Improved Kernel PLS Algorithm #1 on `X` and `Y` using `A` components.
        Returns the internal matrices instead of storing them in the class instance.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        Y : Array of shape (N, M) or (N,)
            Response variables.

        A : int
            Number of components in the PLS model.

        center_X : bool, default=True
            Whether to center `X` before fitting by subtracting its row of
            column-wise means from each row.

        center_Y : bool, default=True
            Whether to center `Y` before fitting by subtracting its row of
            column-wise means from each row.

        scale_X : bool, default=True
            Whether to scale `X` before fitting by dividing each row with the row of
            `X`'s column-wise standard deviations. Bessel's correction for the unbiased
            estimate of the sample standard deviation is used.

        scale_Y : bool, default=True
            Whether to scale `Y` before fitting by dividing each row with the row of
            `X`'s column-wise standard deviations. Bessel's correction for the unbiased
            estimate of the sample standard deviation is used.

        copy : bool, optional, default=True
            Whether to copy `X` and `Y` in fit before potentially applying centering
            and scaling. If True, then the data is copied before fitting. If False, and
            `dtype` matches the type of `X` and `Y`, then centering and scaling is done
            inplace, modifying both arrays.

        Returns
        -------
        B : Array of shape (A, K, M)
            PLS regression coefficients tensor.

        W : Array of shape (A, K)
            PLS weights matrix for X.

        P : Array of shape (A, K)
            PLS loadings matrix for X.

        Q : Array of shape (A, M)
            PLS Loadings matrix for Y.

        R : Array of shape (A, K)
            PLS weights matrix to compute scores T directly from original X.

        T : Array of shape (A, N)
            PLS scores matrix of X. Only Returned for Improved Kernel PLS Algorithm #1.

        X_mean : Array of shape (1, K) or None
            Mean of the predictor variables `center_X` is True, otherwise None.

        Y_mean : Array of shape (1, M) or None
            Mean of the response variables `center_Y` is True, otherwise None.

        X_std : Array of shape (1, K) or None
            Sample standard deviation of the predictor variables `scale_X` is True,
            otherwise None.

        Y_std : Array of shape (1, M) or None
            Sample standard deviation of the response variables `scale_Y` is True,
            otherwise None.

        Warns
        -----
        UserWarning.
            If at any point during iteration over the number of components `A`, the
            residual goes below machine epsilon.

        See Also
        --------
        fit : Performs the same operation but stores the output matrices in the class
        instance instead of returning them.

        Notes
        -----
        For optimization purposes, the internal representation of all matrices
        (except B) is transposed from the usual representation.
        """

    @abc.abstractmethod
    def fit(self, X: ArrayLike, Y: ArrayLike, A: int) -> None:
        """
        Fits Improved Kernel PLS Algorithm #1 on `X` and `Y` using `A` components.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        Y : Array of shape (N, M) or (N,)
            Response variables.

        A : int
            Number of components in the PLS model.

        Attributes
        ----------
        B : Array of shape (A, K, M)
            PLS regression coefficients tensor.

        W : Array of shape (K, A)
            PLS weights matrix for X.

        P : Array of shape (K, A)
            PLS loadings matrix for X.

        Q : Array of shape (M, A)
            PLS Loadings matrix for Y.

        R : Array of shape (K, A)
            PLS weights matrix to compute scores T directly from original X.

        T : Array of shape (N, A)
            PLS scores matrix of X. Only assigned for Improved Kernel PLS Algorithm #1.

        Returns
        -------
        None.

        Warns
        -----
        UserWarning.
            If at any point during iteration over the number of components `A`, the
            residual goes below machine epsilon.

        See Also
        --------
        stateless_fit : Performs the same operation but returns the output matrices
        instead of storing them in the class instance.
        """

    @partial(jax.jit, static_argnums=(0, 3))
    def stateless_predict(
        self,
        X: ArrayLike,
        B: jax.Array,
        n_components: Union[None, int] = None,
        X_mean: Union[None, jax.Array] = None,
        X_std: Union[None, jax.Array] = None,
        Y_mean: Union[None, jax.Array] = None,
        Y_std: Union[None, jax.Array] = None,
    ) -> jax.Array:
        """
        Predicts with Improved Kernel PLS Algorithm #1 on `X` with `B` using
        `n_components` components. If `n_components` is None, then predictions are
        returned for all number of components.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        B : Array of shape (A, K, M)
            PLS regression coefficients tensor.

        n_components : int or None, optional
            Number of components in the PLS model. If None, then all number of
            components are used.

        X_mean : Array of shape (1, K) or None, optional, default=None
            Mean of the predictor variables. If None, then no mean is subtracted from
            `X`.

        X_std : Array of shape (1, K) or None, optional, default=None
            Sample standard deviation of the predictor variables. If None, then no
            scaling is applied to `X`.

        Y_mean : Array of shape (1, M) or None, optional, default=None
            Mean of the response variables. If None, then no mean is subtracted from
            `Y`.

        Y_std : Array of shape (1, M) or None, optional, default=None
            Sample standard deviation of the response variables. If None, then no
            scaling is applied to `Y`.

        Returns
        -------
        Y_pred : Array of shape (N, M) or (A, N, M)
            If `n_components` is an int, then an array of shape (N, M) with the
            predictions for that specific number of components is used. If
            `n_components` is None, returns a prediction for each number of components
            up to `A`.

        See Also
        --------
        predict : Performs the same operation but uses the class instances of `B`,
        `X_mean`, `X_std`, `Y_mean`, and `Y_std` instead of the ones passed as
        arguments.
        """
        X = jnp.asarray(X, dtype=self.dtype)
        if self.verbose:
            print(f"stateless_predict for {self.name} will be JIT compiled...")

        if X_mean is not None:
            X = X - X_mean
        if X_std is not None:
            X = X / X_std

        if n_components is None:
            Y_pred = X @ B
        else:
            Y_pred = X @ B[n_components - 1]

        if Y_std is not None:
            Y_pred = Y_pred * Y_std
        if Y_mean is not None:
            Y_pred = Y_pred + Y_mean
        return Y_pred

    def predict(self, X: ArrayLike, n_components: Union[None, int] = None) -> jax.Array:
        """
        Predicts with Improved Kernel PLS Algorithm #1 on `X` with `B` using
        `n_components` components. If `n_components` is None, then predictions are
        returned for all number of components.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        n_components : int or None, optional
            Number of components in the PLS model. If None, then all number of
            components are used.

        Returns
        -------
        Y_pred : Array of shape (N, M) or (A, N, M)
            If `n_components` is an int, then an array of shape (N, M) with the
            predictions for that specific number of components is used. If
            `n_components` is None, returns a prediction for each number of components
            up to `A`.

        See Also
        --------
        stateless_predict : Performs the same operation but uses inputs `B`, `X_mean`,
        `X_std`, `Y_mean`, and `Y_std` instead of the ones stored in the class
        instance.
        """
        return self.stateless_predict(
            X, self.B, n_components, self.X_mean, self.X_std, self.Y_mean, self.Y_std
        )

    @partial(jax.jit, static_argnums=(0, 3, 6, 7, 8, 9, 10, 11))
    def stateless_fit_predict_eval(
        self,
        X_train: ArrayLike,
        Y_train: ArrayLike,
        A: int,
        X_test: ArrayLike,
        Y_test: ArrayLike,
        metric_function: Callable[[jax.Array, jax.Array], Any],
        center_X: bool = True,
        center_Y: bool = True,
        scale_X: bool = True,
        scale_Y: bool = True,
        copy: bool = True,
    ) -> Any:
        """
        Computes `B` with `stateless_fit`. Then computes `Y_pred` with
        `stateless_predict`. `Y_pred` is an array of shape (A, N, M). Then evaluates
        and returns the result of `metric_function(Y_test, Y_pred)`.

        Parameters
        ----------
        X_train : Array of shape (N_train, K)
            Predictor variables.

        Y_train : Array of shape (N_train, M) or (N_train,)
            Response variables.

        A : int
            Number of components in the PLS model.

        X_test : Array of shape (N_test, K)
            Predictor variables.

        Y_test : Array of shape (N_test, M) or (N_test,)
            Response variables.

        metric_function : Callable receiving arrays `Y_test` of shape (N, M) and
        `Y_pred` (A, N, M) and returns Any
            Computes a metric based on true values `Y_test` and predicted values
            `Y_pred`. `Y_pred` contains a prediction for all `A` components.

        center_X : bool, default=True
            Whether to center `X` before fitting by subtracting its row of
            column-wise means from each row.

        center_Y : bool, default=True
            Whether to center `Y` before fitting by subtracting its row of
            column-wise means from each row.

        scale_X : bool, default=True
            Whether to scale `X` before fitting by dividing each row with the row of
            `X`'s column-wise standard deviations. Bessel's correction for the unbiased
            estimate of the sample standard deviation is used.

        scale_Y : bool, default=True
            Whether to scale `Y` before fitting by dividing each row with the row of
            `X`'s column-wise standard deviations. Bessel's correction for the unbiased
            estimate of the sample standard deviation is used.

        copy : bool, optional, default=True
            Whether to copy `X_train` and `Y_train` in stateless_fit before potentially
            applying centering and scaling. If True, then the data is copied before
            fitting. If False, and `dtype` matches the type of `X` and `Y`, then
            centering and scaling is done inplace, modifying both arrays.

        Returns
        -------
        metric_function(Y_test, Y_pred) : Any.

        See Also
        --------
        stateless_fit : Fits on `X_train` and `Y_train` using `A` components while
        optionally performing centering and scaling. Then returns the internal matrices
        instead of storing them in the class instance.

        stateless_predict : Computes `Y_pred` given predictor variables `X` and
        regression tensor `B` and optionally `A` components.
        """
        if self.verbose:
            print(f"stateless_fit_predict_eval for {self.name} will be JIT compiled...")

        X_train, Y_train = self._initialize_input_matrices(X=X_train, Y=Y_train)
        X_test, Y_test = self._initialize_input_matrices(X=X_test, Y=Y_test)

        matrices = self.stateless_fit(
            X=X_train,
            Y=Y_train,
            A=A,
            center_X=center_X,
            center_Y=center_Y,
            scale_X=scale_X,
            scale_Y=scale_Y,
            copy=copy,
        )
        B = matrices[0]
        X_mean, Y_mean, X_std, Y_std = matrices[-4:]
        Y_pred = self.stateless_predict(
            X_test, B, X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std
        )
        return metric_function(Y_test, Y_pred)

    def cross_validate(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        A: int,
        cv_splits: ArrayLike,
        preprocessing_function: Callable[
            [jax.Array, jax.Array, jax.Array, jax.Array],
            Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ],
        metric_function: Callable[[jax.Array, jax.Array], Any],
        metric_names: list[str],
        show_progress=True,
    ) -> dict[str, Any]:
        """
        Performs cross-validation for the Partial Least-Squares (PLS) model on given
        data. `preprocessing_function` will be applied before any potential centering
        and scaling as determined by `self.center_X`, `self.center_Y`, `self.scale_X`,
        and `self.scale_Y`. Any such potential centering and scaling is applied for
        each split using training set statistics to avoid data leakage from the
        validation set.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        Y : Array of shape (N, M) or (N,)
            Response variables.

        A : int
            Number of components in the PLS model.

        cv_splits : Array of shape (N,)
            An array defining cross-validation splits. Each unique value in `cv_splits`
            corresponds to a different fold.

        preprocessing_function : Callable receiving arrays `X_train`, `Y_train`,
        `X_val`, and `Y_val`
            A function that preprocesses the training and validation data for each
            fold. It should return preprocessed arrays for `X_train`, `Y_train`,
            `X_val`, and `Y_val`.

        metric_function : Callable receiving arrays `Y_test` and `Y_pred` and returning
        Any
            Computes a metric based on true values `Y_test` and predicted values
            `Y_pred`. `Y_pred` contains a prediction for all `A` components.

        metric_names : list of str
            A list of names for the metrics used for evaluation.

        show_progress : bool, optional, default=True
            If True, displays a progress bar for the cross-validation.

        Returns
        -------
        metrics : dict[str, Any]
            A dictionary containing evaluation metrics for each metric specified in
            `metric_names`. The keys are metric names, and the values are lists of
            metric values for each cross-validation fold.

        See Also
        --------
        _inner_cv : Performs cross-validation for a single fold and computes evaluation
        metrics.

        _update_metric_value_lists : Updates lists of metric values for each metric and
        fold.

        _finalize_metric_values : Organizes and finalizes the metric values into a
        dictionary for the specified metric names.

        stateless_fit_predict_eval : Fits the PLS model, makes predictions, and
        evaluates metrics for a given fold.

        Notes
        -----
        This method is used to perform cross-validation on the PLS model with different
        data splits and evaluate its performance using user-defined metrics.
        """
        X = jnp.asarray(X, dtype=self.dtype)
        Y = jnp.asarray(Y, dtype=self.dtype)
        cv_splits = jnp.asarray(cv_splits, dtype=jnp.int64)
        metric_value_lists = [[] for _ in metric_names]
        unique_splits = jnp.unique(cv_splits)
        for split in tqdm(unique_splits, disable=not show_progress):
            train_idxs = jnp.nonzero(cv_splits != split)[0]
            val_idxs = jnp.nonzero(cv_splits == split)[0]
            metric_values = self._inner_cross_validate(
                X=X,
                Y=Y,
                train_idxs=train_idxs,
                val_idxs=val_idxs,
                A=A,
                preprocessing_function=preprocessing_function,
                metric_function=metric_function,
                center_X=self.center_X,
                center_Y=self.center_Y,
                scale_X=self.scale_X,
                scale_Y=self.scale_Y,
                copy=self.copy,
            )
            metric_value_lists = self._update_metric_value_lists(
                metric_value_lists, metric_names, metric_values
            )
        return self._finalize_metric_values(metric_value_lists, metric_names)

    @partial(jax.jit, static_argnums=(0, 5, 6, 7, 8, 9, 10, 11, 12))
    def _inner_cross_validate(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        train_idxs: jax.Array,
        val_idxs: jax.Array,
        A: int,
        preprocessing_function: Callable[
            [jax.Array, jax.Array, jax.Array, jax.Array],
            Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ],
        metric_function: Callable[[jax.Array, jax.Array], Any],
        center_X: bool = True,
        center_Y: bool = True,
        scale_X: bool = True,
        scale_Y: bool = True,
        copy: bool = True,
    ):
        """
        Performs cross-validation for a single fold of the data and computes evaluation
        metrics.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        Y : Array of shape (N, M)
            Response variables.

        train_idxs : Array of shape (N_train,)
            Indices of data points in the training set.

        val_idxs : Array of shape (N_val,)
            Indices of data points in the validation set.

        A : int
            Number of components in the PLS model.

        preprocessing_function : Callable receiving arrays `X_train`, `Y_train`,
        `X_val`, and `Y_val`
            A function that preprocesses the training and validation data for each
            fold. It should return preprocessed arrays for `X_train`, `Y_train`,
            `X_val`, and `Y_val`.

        metric_function : Callable receiving arrays `Y_test` and `Y_pred` and returning
        Any
            Computes a metric based on true values `Y_test` and predicted values
            `Y_pred`. `Y_pred` contains a prediction for all `A` components.

        center_X : bool, default=True
            Whether to center `X` before fitting by subtracting its row of
            column-wise means from each row.

        center_Y : bool, default=True
            Whether to center `Y` before fitting by subtracting its row of
            column-wise means from each row.

        scale_X : bool, default=True
            Whether to scale `X` before fitting by dividing each row with the row of
            `X`'s column-wise standard deviations. Bessel's correction for the unbiased
            estimate of the sample standard deviation is used.

        scale_Y : bool, default=True
            Whether to scale `Y` before fitting by dividing each row with the row of
            `X`'s column-wise standard deviations. Bessel's correction for the unbiased
            estimate of the sample standard deviation is used.

        copy : bool, optional, default=True
            Whether to copy `X_train` and `Y_train` in stateless_fit before potentially
            applying centering and scaling. If True, then the data is copied before
            fitting. If False, and `dtype` matches the type of `X` and `Y`, then
            centering and scaling is done inplace, modifying both arrays.

        Returns
        -------
        metric_values : Any
            metric values based on the true and predicted values for a single fold.

        Notes
        -----
        This method performs cross-validation for a single fold of the data, including
        preprocessing, fitting, predicting, and evaluating the PLS model.
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
            X_train=X_train,
            Y_train=Y_train,
            A=A,
            X_test=X_val,
            Y_test=Y_val,
            metric_function=metric_function,
            center_X=center_X,
            center_Y=center_Y,
            scale_X=scale_X,
            scale_Y=scale_Y,
            copy=copy,
        )
        return metric_values

    def _update_metric_value_lists(
        self,
        metric_value_lists: list[list[Any]],
        metric_names: list[str],
        metric_values: Any,
    ):
        """
        Updates lists of metric values for each metric and fold during
        cross-validation.

        Parameters
        ----------
        metric_value_lists : list of list of Any
            Lists of metric values for each metric and fold.

        metric_values : list of Any
            Metric values for a single fold.

        Returns
        -------
        metric_value_lists : list of list of Any
            Updated lists of metric values for each metric and fold.

        Notes
        -----
        This method updates the lists of metric values for each metric and fold during
        cross-validation.
        """
        if len(metric_names) == 1:
            metric_value_lists[0].append(metric_values)
        else:
            for i in range(len(metric_names)):
                metric_value_lists[i].append(metric_values[i])
        return metric_value_lists

    def _finalize_metric_values(
        self, metrics_results: list[list[Any]], metric_names: list[str]
    ):
        """
        Organizes and finalizes the metric values into a dictionary for the specified
        metric names.

        Parameters
        ----------
        metrics_results : list of list of Any
            Lists of metric values for each metric and fold.

        metric_names : list of str
            A list of names for the metrics used for evaluation.

        Returns
        -------
        metrics : dict[str, list[Any]]
            A dictionary containing evaluation metrics for each metric specified in
            `metric_names`. The keys are metric names, and the values are lists of
            metric values for each cross-validation fold.

        Notes
        -----
        This method organizes and finalizes the metric values into a dictionary for the
        specified metric names, making it easy to analyze the cross-validation results.
        """
        metrics = {}
        for name, lst_of_metric_value_for_each_split in zip(
            metric_names, metrics_results
        ):
            metrics[name] = lst_of_metric_value_for_each_split
        return metrics
