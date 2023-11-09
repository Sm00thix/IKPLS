import numpy as np
import numpy.linalg as la
import numpy.typing as npt
import warnings
from sklearn.base import BaseEstimator


class PLS(BaseEstimator):
    """
    Implements partial least-squares regression using Improved Kernel PLS by Dayal and MacGregor: https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

    Parameters:
    `algorithm` : int
        Whether to use Improved Kernel PLS Algorithm #1 or #2. Defaults to 1.
    `dtype` : np.float_, default=numpy.float64
        The float datatype to use in computation of the PLS algorithm. Using a lower precision than float64 will yield significantly worse results when using an increasing number of components due to propagation of numerical errors.
    """

    def __init__(self, algorithm: int = 1, dtype: np.float_ = np.float64) -> None:
        self.algorithm = algorithm
        self.dtype = dtype
        self.name = f"Improved Kernel PLS Algorithm #{algorithm}"

    def fit(self, X: npt.ArrayLike, Y: npt.ArrayLike, A: int) -> None:
        """
        Description
        -----------
        Fits Improved Kernel PLS Algorithm #1 on `X` and `Y` using `A` components.

        Parameters
        ----------
        `X` : Array of shape (N, K)
            Predictor variables.

        `Y` : Array of shape (N, M)
            Response variables.

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
        """
        X = np.array(X, dtype=self.dtype)
        Y = np.array(Y, dtype=self.dtype)

        N, K = X.shape
        M = Y.shape[1]

        self.B = np.zeros(shape=(A, K, M), dtype=self.dtype)
        W = np.zeros(shape=(A, K), dtype=self.dtype)
        P = np.zeros(shape=(A, K), dtype=self.dtype)
        Q = np.zeros(shape=(A, M), dtype=self.dtype)
        R = np.zeros(shape=(A, K), dtype=self.dtype)
        self.W = W.T
        self.P = P.T
        self.Q = Q.T
        self.R = R.T
        if self.algorithm == 1:
            T = np.zeros(shape=(A, N), dtype=self.dtype)
            self.T = T.T
        self.A = A
        self.N = N
        self.K = K
        self.M = M

        # Step 1
        XTY = X.T @ Y

        # Used for algorithm #2
        if self.algorithm == 2:
            XTX = X.T @ X

        for i in range(A):
            # Step 2
            if M == 1:
                norm = la.norm(XTY, ord=2)
                if np.isclose(norm, 0, atol=np.finfo(np.float64).eps, rtol=0):
                    warnings.warn(
                        f"Weight is close to zero. Stopping fitting after A = {i} component(s)."
                    )
                    break
                w = XTY / norm
            else:
                if M < K:
                    XTYTXTY = XTY.T @ XTY
                    eig_vals, eig_vecs = la.eigh(XTYTXTY)
                    q = eig_vecs[:, -1:]
                    w = XTY @ q
                    norm = la.norm(w)
                    if np.isclose(norm, 0, atol=np.finfo(np.float64).eps, rtol=0):
                        warnings.warn(
                            f"Weight is close to zero. Stopping fitting after A = {i} component(s)."
                        )
                        break
                    w = w / norm
                else:
                    XTYYTX = XTY @ XTY.T
                    eig_vals, eig_vecs = la.eigh(XTYYTX)
                    norm = eig_vals[-1]
                    if np.isclose(norm, 0, atol=np.finfo(np.float64).eps, rtol=0):
                        warnings.warn(
                            f"Weight is close to zero. Stopping fitting after A = {i} component(s)."
                        )
                        break
                    w = eig_vecs[:, -1:]
            W[i] = w.squeeze()

            # Step 3
            r = np.copy(w)
            for j in range(i):
                r = r - P[j].reshape(-1, 1).T @ w * R[j].reshape(-1, 1)
            R[i] = r.squeeze()

            # Step 4
            if self.algorithm == 1:
                t = X @ r
                T[i] = t.squeeze()
                tTt = t.T @ t
                p = (t.T @ X).T / tTt
            elif self.algorithm == 2:
                rXTX = r.T @ XTX
                tTt = rXTX @ r
                p = rXTX.T / tTt
            q = (r.T @ XTY).T / tTt
            P[i] = p.squeeze()
            Q[i] = q.squeeze()

            # Step 5
            XTY = XTY - (p @ q.T) * tTt

            # Compute regression coefficients
            self.B[i] = self.B[i - 1] + r @ q.T

    def predict(
        self, X: npt.ArrayLike, n_components: None | int = None
    ) -> npt.NDArray[np.float_]:
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
        """
        X = np.array(X, dtype=self.dtype)
        if n_components is None:
            return X @ self.B
        return X @ self.B[n_components - 1]
