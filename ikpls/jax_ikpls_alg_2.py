from ikpls.jax_ikpls_base import PLSBase
import jax
from jax.experimental import host_callback
import jax.numpy as jnp
from functools import partial
from typing import Tuple


class PLS(PLSBase):
    """
    Description
    -----------
    Implements partial least-squares regression using Improved Kernel PLS Algorithm #2 by Dayal and MacGregor: https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23.

    Parameters
    ----------
    `reverse_differentiable`: bool, optional (default=False). Whether to make the implementation end-to-end differentiable. The differentiable version is slightly slower. Results among the two versions are identical.

    `verbose` : bool, optional (default=False). If True, each sub-function will print when it will be JIT compiled. This can be useful to track if recompilation is triggered due to passing inputs with different shapes.
    """

    def __init__(
        self, reverse_differentiable: bool = False, verbose: bool = False
    ) -> None:
        name = "Improved Kernel PLS Algorithm #2"
        super().__init__(
            name=name, reverse_differentiable=reverse_differentiable, verbose=verbose
        )

    def _get_initial_matrices(
        self, X: jnp.ndarray, Y: jnp.ndarray, A: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Description
        -----------
        Initialize the matrices and arrays needed for the PLS algorithm. This method is part of the PLS fitting process.

        Parameters
        ----------
        `A` : int
            Number of components in the PLS model.

        `K` : int
            Number of predictor variables.

        `M` : int
            Number of response variables.

        `N` : int
            Number of samples.

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
        """
        if self.verbose:
            print(f"_get_initial_matrices for {self.name} will be JIT compiled...")
        return super()._get_initial_matrices(X, Y, A)

    @partial(jax.jit, static_argnums=(0,))
    def _step_1(
        self, X: jnp.ndarray, Y: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Description
        -----------
        Perform the first step of Improved Kernel PLS Algorithm #2.

        Parameters
        ----------
        `X` : Array of shape (N, K)
            Predictor variables. The precision should be at least float64 for reliable results.

        `Y` : Array of shape (N, M)
            Response variables. The precision should be at least float64 for reliable results.

        Returns
        -------
        `XTX` : Array of shape (K, K)
            Product of transposed predictor variables and predictor variables.

        `XTY` : Array of shape (K, M)
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
        self, XTX: jnp.ndarray, XTY: jnp.ndarray, r: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Description
        -----------
        Perform the fourth step of Improved Kernel PLS Algorithm #2.

        Parameters
        ----------
        `XTX` : Array of shape (K, K)
            XTX product.

        `XTY` : Array of shape (K, M)
            XTY product.

        `r` : Array of shape (K, K)
            PLS weight vector.

        Returns
        -------
        `tTt` : Array of shape (1, 1)
            tTt value.

        `p` : Array of shape (K, K)
            p matrix.

        `q` : Array of shape (K, M)
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
        XTX: jnp.ndarray,
        XTY: jnp.ndarray,
        M: int,
        K: int,
        P: jnp.ndarray,
        R: jnp.ndarray,
        reverse_differentiable: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Description
        -----------
        Execute the main loop body of Improved Kernel PLS Algorithm #2. This function performs various steps of the PLS algorithm for each component.

        Parameters
        ----------
        `A` : int
            Number of components in the PLS model.

        `i` : int
            Current iteration step.

        `XTX` : Array of shape (K, K)
            XTX product.

        `XTY` : Array of shape (K, M)
            XTY product.

        `M` : int
            Number of response variables.

        `K` : int
            Number of predictor variables.

        `P` : Array of shape (K, K)
            PLS loadings matrix for X.

        `R` : Array of shape (K, A)
            PLS weights matrix to compute scores T directly from original X.

        `reverse_differentiable` : bool
            Whether to use a reverse_differentiable version of the algorithm.

        Returns
        -------
        `XTY` : Array of shape (K, M)
            Updated XTY product.

        `w` : Array of shape (K, K)
            w matrix.

        `p` : Array of shape (K, K)
            p matrix.

        `q` : Array of shape (K, M)
            q matrix.

        `r` : Array of shape (K, K)
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
        self.B, W, P, Q, R = self.stateless_fit(X, Y, A)
        self.W = W.T
        self.P = P.T
        self.Q = Q.T
        self.R = R.T

    @partial(jax.jit, static_argnums=(0, 3))
    def stateless_fit(
        self, X: jnp.ndarray, Y: jnp.ndarray, A: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        if self.verbose:
            print(f"stateless_fit for {self.name} will be JIT compiled...")

        # Get shapes
        N, K = X.shape
        M = Y.shape[1]

        # Initialize matrices
        B, W, P, Q, R = self._get_initial_matrices(A, K, M)

        # step 1
        XTX, XTY = self._step_1(X, Y)

        # steps 2-5
        for i in range(A):
            XTY, w, p, q, r = self._main_loop_body(
                A, i, XTX, XTY, M, K, P, R, self.reverse_differentiable
            )
            W = W.at[i].set(w.squeeze())
            P = P.at[i].set(p.squeeze())
            Q = Q.at[i].set(q.squeeze())
            R = R.at[i].set(r.squeeze())

            # step 6
            b = self._compute_regression_coefficients(B[i - 1], r, q)
            B = B.at[i].set(b)

        return B, W, P, Q, R
