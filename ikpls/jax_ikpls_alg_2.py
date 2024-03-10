from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental import host_callback
from jax.typing import ArrayLike, DTypeLike

from ikpls.jax_ikpls_base import PLSBase


class PLS(PLSBase):
    """
    Implements partial least-squares regression using Improved Kernel PLS Algorithm #2 by Dayal and MacGregor: https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23.

    Parameters
    ----------
    center : bool, optional, default=True
        Whether to center the data before fitting. If True, then the mean of the training data is subtracted from the data. If False, then the data is assumed to be already centered.

    scale : bool, optional, default=True
        Whether to scale the data before fitting. If True, then the data is scaled using Bessel's correction for the unbiased estimate of the sample standard deviation. If False, then the data is assumed to be already scaled.

    copy : bool, optional, default=True
        Whether to copy `X` and `Y` in fit before potentially applying centering and scaling. If True, then the data is copied before fitting. If False, and `dtype` matches the type of `X` and `Y`, then centering and scaling is done inplace, modifying both arrays.

    dtype : DTypeLike, optional, default=jnp.float64
        The float datatype to use in computation of the PLS algorithm. Using a lower precision than float64 will yield significantly worse results when using an increasing number of components due to propagation of numerical errors.

    reverse_differentiable: bool, optional, default=False
        Whether to make the implementation end-to-end differentiable. The differentiable version is slightly slower. Results among the two versions are identical.

    verbose : bool, optional, default=False
        If True, each sub-function will print when it will be JIT compiled. This can be useful to track if recompilation is triggered due to passing inputs with different shapes.
    """

    def __init__(
        self,
        center: bool = True,
        scale: bool = True,
        copy: bool = True,
        dtype: DTypeLike = jnp.float64,
        reverse_differentiable: bool = False,
        verbose: bool = False,
    ) -> None:
        self.name = "Improved Kernel PLS Algorithm #2"
        super().__init__(
            center=center,
            scale=scale,
            copy=copy,
            dtype=dtype,
            reverse_differentiable=reverse_differentiable,
            verbose=verbose,
        )

    def _get_initial_matrices(
        self, X: ArrayLike, Y: ArrayLike, A: int
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Initialize the matrices and arrays needed for the PLS algorithm. This method is part of the PLS fitting process.

        Parameters
        ----------
        A : int
            Number of components in the PLS model.

        K : int
            Number of predictor variables.

        M : int
            Number of response variables.

        N : int
            Number of samples.

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
        """
        if self.verbose:
            print(f"_get_initial_matrices for {self.name} will be JIT compiled...")
        return super()._get_initial_matrices(X, Y, A)

    @partial(jax.jit, static_argnums=(0,))
    def _step_1(self, X: ArrayLike, Y: ArrayLike) -> Tuple[jax.Array, jax.Array]:
        """
        Perform the first step of Improved Kernel PLS Algorithm #2.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        Y : Array of shape (N, M)
            Response variables.

        Returns
        -------
        XTX : Array of shape (K, K)
            Product of transposed predictor variables and predictor variables.

        XTY : Array of shape (K, M)
            Initial cross-covariance matrix of the predictor variables and the response variables.
        """
        if self.verbose:
            print(f"_step_1 for {self.name} will be JIT compiled...")
        XT = self._compute_XT(X)
        XTX = self._compute_XTX(XT, X)
        XTY = self._compute_initial_XTY(XT, Y)
        return XTX, XTY

    @partial(jax.jit, static_argnums=(0,))
    def _step_4(
        self, XTX: jax.Array, XTY: jax.Array, r: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Perform the fourth step of Improved Kernel PLS Algorithm #2.

        Parameters
        ----------
        XTX : Array of shape (K, K)
            XTX product.

        XTY : Array of shape (K, M)
            XTY product.

        r : Array of shape (K, K)
            PLS weight vector.

        Returns
        -------
        tTt : Array of shape (1, 1)
            tTt value.

        p : Array of shape (K, K)
            p matrix.

        q : Array of shape (K, M)
            q matrix.
        """
        if self.verbose:
            print(f"_step_4 for {self.name} will be JIT compiled...")
        rXTX = r.T @ XTX
        tTt = rXTX @ r
        p = rXTX.T / tTt
        q = (r.T @ XTY).T / tTt
        return tTt, p, q

    @partial(jax.jit, static_argnums=(0, 1, 5, 6, 9))
    def _main_loop_body(
        self,
        A: int,
        i: int,
        XTX: jax.Array,
        XTY: jax.Array,
        M: int,
        K: int,
        P: jax.Array,
        R: jax.Array,
        reverse_differentiable: bool,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Execute the main loop body of Improved Kernel PLS Algorithm #2. This function performs various steps of the PLS algorithm for each component.

        Parameters
        ----------
        A : int
            Number of components in the PLS model.

        i : int
            Current iteration step.

        XTX : Array of shape (K, K)
            XTX product.

        XTY : Array of shape (K, M)
            XTY product.

        M : int
            Number of response variables.

        K : int
            Number of predictor variables.

        P : Array of shape (K, K)
            PLS loadings matrix for X.

        R : Array of shape (K, A)
            PLS weights matrix to compute scores T directly from original X.

        reverse_differentiable : bool
            Whether to use a reverse_differentiable version of the algorithm.

        Returns
        -------
        XTY : Array of shape (K, M)
            Updated XTY product.

        w : Array of shape (K, K)
            w matrix.

        p : Array of shape (K, K)
            p matrix.

        q : Array of shape (K, M)
            q matrix.

        r : Array of shape (K, K)
            PLS weight vector.
        """
        if self.verbose:
            print(f"_main_loop_body for {self.name} will be JIT compiled...")
        # step 2
        w, norm = self._step_2(XTY, M, K)
        host_callback.id_tap(self._weight_warning, [i, norm])
        # step 3
        if reverse_differentiable:
            r = self._step_3(A, w, P, R)
        else:
            r = self._step_3(i, w, P, R)
        # step 4
        tTt, p, q = self._step_4(XTX, XTY, r)
        # step 5
        XTY = self._step_5(XTY, p, q, tTt)
        return XTY, w, p, q, r

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

        X_mean : Array of shape (1, K) or None
            Mean of X. If centering is not performed, this is None.

        Y_mean : Array of shape (1, M) or None
            Mean of Y. If centering is not performed, this is None.

        X_std : Array of shape (1, K) or None
            Sample standard deviation of X. If scaling is not performed, this is None.

        Y_std : Array of shape (1, M) or None
            Sample standard deviation of Y. If scaling is not performed, this is None.

        Returns
        -------
        None.

        Warns
        -----
        UserWarning.
            If at any point during iteration over the number of components `A`, the residual goes below machine epsilon.

        See Also
        --------
        stateless_fit : Performs the same operation but returns the output matrices instead of storing them in the class instance.
        """
        self.B, W, P, Q, R, self.X_mean, self.Y_mean, self.X_std, self.Y_std = (
            self.stateless_fit(X, Y, A, self.center, self.scale, self.copy)
        )
        self.W = W.T
        self.P = P.T
        self.Q = Q.T
        self.R = R.T

    @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))
    def stateless_fit(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        A: int,
        center: bool = True,
        scale: bool = True,
        copy: bool = True,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Fits Improved Kernel PLS Algorithm #1 on `X` and `Y` using `A` components. Returns the internal matrices instead of storing them in the class instance.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables. Its dtype will be converted to float64 for reliable results.

        Y : Array of shape (N, M) or (N,)
            Response variables. Its dtype will be converted to float64 for reliable results.

        A : int
            Number of components in the PLS model.

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

        X_mean : Array of shape (1, K) or None
            Mean of X. If centering is not performed, this is None.

        Y_mean : Array of shape (1, M) or None
            Mean of Y. If centering is not performed, this is None.

        X_std : Array of shape (1, K) or None
            Sample standard deviation of X. If scaling is not performed, this is None.

        Y_std : Array of shape (1, M) or None
            Sample standard deviation of Y. If scaling is not performed, this is None.

        Warns
        -----
        UserWarning.
            If at any point during iteration over the number of components `A`, the residual goes below machine epsilon.

        See Also
        --------
        fit : Performs the same operation but stores the output matrices in the class instance instead of returning them.

        Notes
        -----
        For optimization purposes, the internal representation of all matrices (except B) is transposed from the usual representation.
        """
        if self.verbose:
            print(f"stateless_fit for {self.name} will be JIT compiled...")

        X, Y = self._initialize_input_matrices(X, Y)
        X, Y, X_mean, Y_mean, X_std, Y_std = self._center_scale_input_matrices(
            X, Y, center, scale, copy
        )

        # Get shapes
        N, K = X.shape
        M = Y.shape[1]

        # Initialize matrices
        B, W, P, Q, R = self._get_initial_matrices(A, K, M)

        # step 1
        XTX, XTY = self._step_1(X, Y)

        # steps 2-6
        for i in range(A):
            XTY, w, p, q, r = self._main_loop_body(
                A, i, XTX, XTY, M, K, P, R, self.reverse_differentiable
            )
            W = W.at[i].set(w.squeeze())
            P = P.at[i].set(p.squeeze())
            Q = Q.at[i].set(q.squeeze())
            R = R.at[i].set(r.squeeze())
            b = self._compute_regression_coefficients(B[i - 1], r, q)
            B = B.at[i].set(b)

        return B, W, P, Q, R, X_mean, Y_mean, X_std, Y_std
