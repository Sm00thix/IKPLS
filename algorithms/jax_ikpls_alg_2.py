from algorithms.jax_ikpls_base import PLSBase
import jax
from jax.experimental import host_callback
import jax.numpy as jnp
from functools import partial
from typing import Tuple


class PLS(PLSBase):
    """
    Implements partial least-squares regression using Improved Kernel PLS by Dayal and MacGregor: https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

    Parameters:
    differentiable: Bool. Whether to make the implementation end-to-end differentiable. The differentiable version is slightly slower. Results among the two versions are identical. Defaults to False
    """

    def __init__(self, differentiable: bool = False) -> None:
        super().__init__(differentiable=differentiable)

    def _get_initial_matrices(
        self, X: jnp.ndarray, Y: jnp.ndarray, A: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        print("Tracing initial matrices...")
        return super()._get_initial_matrices(X, Y, A)

    @partial(jax.jit, static_argnums=(0,))
    def _step_1(
        self, X: jnp.ndarray, Y: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        print("Tracing step 1...")
        XT = self._compute_XT(X)
        XTX = self._compute_XTX(XT, X)
        XTY = self._compute_initial_XTY(XT, Y)
        return XTX, XTY

    @partial(jax.jit, static_argnums=(0,))
    def _step_4(
        self, XTX: jnp.ndarray, XTY: jnp.ndarray, r: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        print("Tracing step 4...")
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
        differentiable: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        print("Tracing loop body...")
        # step 2
        w, norm = self._step_2(XTY, M, K)
        host_callback.id_tap(self.weight_warning, [i, norm])
        # step 3
        if differentiable:
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
        Parameters:
        X: Predictor variables matrix (N x K)
        Y: Response variables matrix (N x M)
        A: Number of components in the PLS model

        Sets:
        self.B: PLS regression coefficients matrix (A x K x M)
        self.W: PLS weights matrix for X (K x A)
        self.P: PLS loadings matrix for X (K x A)
        self.Q: PLS Loadings matrix for Y (M x A)
        self.R: PLS weights matrix to compute scores T directly from original X (K x A)
        """
        self.B, _W, _P, _Q, _R = self.stateless_fit(X, Y, A)
        self.W = _W.T
        self.P = _P.T
        self.Q = _Q.T
        self.R = _R.T

    def stateless_fit(
        self, X: jnp.ndarray, Y: jnp.ndarray, A: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Parameters:
        X: Predictor variables matrix (N x K)
        Y: Response variables matrix (N x M)
        A: Number of components in the PLS model

        Returns:
        A tuple of:
        B: PLS regression coefficients matrix (A x K x M)
        W: PLS weights matrix for X (K x A)
        P: PLS loadings matrix for X (K x A)
        Q: PLS Loadings matrix for Y (M x A)
        R: PLS weights matrix to compute scores T directly from original X (K x A)

        Note that the internal representation of all matrices (except B) is transposed for optimization purposes.
        """

        @partial(jax.jit, static_argnums=(2, 3))
        def helper(
            X: jnp.ndarray, Y: jnp.ndarray, A: int, differentiable: bool
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
                    A, i, XTX, XTY, M, K, P, R, differentiable
                )
                W = W.at[i].set(w.squeeze())
                P = P.at[i].set(p.squeeze())
                Q = Q.at[i].set(q.squeeze())
                R = R.at[i].set(r.squeeze())

                # step 6
                b = self.compute_regression_coefficients(B[i - 1], r, q)
                B = B.at[i].set(b)

            return B, W, P, Q, R

        return helper(X=X, Y=Y, A=A, differentiable=self.differentiable)
