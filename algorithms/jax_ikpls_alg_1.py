from algorithms.jax_ikpls_base import PLSBase
import jax
from jax.experimental import host_callback
import jax.numpy as jnp
from functools import partial
from typing import Tuple


class PLS(PLSBase):
    """
    Implements partial least-squares regression using Improved Kernel PLS by Dayal and MacGregor: https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23
    """

    def __init__(self) -> None:
        super().__init__()

    def _get_initial_matrices(
        self, A: int, K: int, M: int, N: int
    ) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]:
        print("Tracing initial matrices...")
        B, W, P, Q, R = super()._get_initial_matrices(A, K, M)
        T = jnp.empty(shape=(A, N), dtype=jnp.float64)
        return B, W, P, Q, R, T

    @partial(jax.jit, static_argnums=(0,))
    def _step_1(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        print("Tracing step 1...")
        return self._compute_initial_XTY(X.T, Y)

    @partial(jax.jit, static_argnums=(0,))
    def _step_4(self, X: jnp.ndarray, XTY: jnp.ndarray, r: jnp.ndarray):
        print("Tracing step 4...")
        t = X @ r
        tT = t.T
        tTt = tT @ t
        p = (tT @ X).T / tTt
        q = (r.T @ XTY).T / tTt
        return tTt, p, q, t
    
    @partial(jax.jit, static_argnums=(0, 1, 5, 6))
    def _main_loop_body(
        self,
        A: int,
        i: int,
        X: jnp.ndarray,
        XTY: jnp.ndarray,
        M: int,
        K: int,
        P: jnp.ndarray,
        R: jnp.ndarray,
    ) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]:
        print("Tracing loop body...")
        # step 2
        w, norm = self._step_2(XTY, M, K)
        host_callback.id_tap(self.weight_warning, [i, norm])
        # step 3
        r = self._step_3(A, w, P, R)
        # step 4
        tTt, p, q, t = self._step_4(X, XTY, r)
        # step 5
        XTY = self._step_5(XTY, p, q, tTt)
        return XTY, w, p, q, r, t

    # @partial(jax.jit, static_argnums=(0, 4, 5))
    # def _main_loop_body(
    #     self,
    #     i: int,
    #     X: jnp.ndarray,
    #     XTY: jnp.ndarray,
    #     M: int,
    #     K: int,
    #     P: jnp.ndarray,
    #     R: jnp.ndarray,
    # ) -> Tuple[
    #     jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    # ]:
    #     print("Tracing loop body...")
    #     # step 2
    #     w, norm = self._step_2(XTY, M, K)
    #     host_callback.id_tap(self.weight_warning, [i, norm])
    #     # step 3
    #     r = self._step_3(i, w, P, R)
    #     # step 4
    #     tTt, p, q, t = self._step_4(X, XTY, r)
    #     # step 5
    #     XTY = self._step_5(XTY, p, q, tTt)
    #     return XTY, w, p, q, r, t

    def fit(self, X: jnp.ndarray, Y: jnp.ndarray, A: int) -> None:
        self.B, _W, _P, _Q, _R, _T = self.stateless_fit(X, Y, A)
        self.W = _W.T
        self.P = _P.T
        self.Q = _Q.T
        self.R = _R.T
        self.T = _T.T

    @partial(jax.jit, static_argnums=(0, 3))
    def stateless_fit(
        self, X: jnp.ndarray, Y: jnp.ndarray, A: int
    ) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]:
        """
        Parameters:
        X: Predictor variables matrix (N x K)
        Y: Response variables matrix (N x M)
        A: Number of components in the PLS model


        Contains:
        B: PLS regression coefficients matrix (A x K x M)
        W: PLS weights matrix for X (K x A)
        P: PLS loadings matrix for X (K x A)
        Q: PLS Loadings matrix for Y (M x A)
        R: PLS weights matrix to compute scores T directly from original X (K x A)
        T: PLS scores matrix of X (N x A)
        """

        # Get shapes
        N, K = X.shape
        M = Y.shape[1]

        # Initialize matrices
        B, W, P, Q, R, T = self._get_initial_matrices(A, K, M, N)

        # step 1
        XTY = self._step_1(X, Y)

        for i in range(A):
            XTY, w, p, q, r, t = self._main_loop_body(A, i, X, XTY, M, K, P, R)
            W = W.at[i].set(w.squeeze())
            P = P.at[i].set(p.squeeze())
            Q = Q.at[i].set(q.squeeze())
            R = R.at[i].set(r.squeeze())
            T = T.at[i].set(t.squeeze())
            b = self.compute_regression_coefficients(B[i - 1], r, q)
            B = B.at[i].set(b)

        return B, W, P, Q, R, T
