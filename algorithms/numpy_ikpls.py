import numpy as np
import numpy.linalg as la
import numpy.typing as npt
import warnings
from sklearn.base import BaseEstimator


class PLS(BaseEstimator):
    """
    Implements partial least-squares regression using Improved Kernel PLS by Dayal and MacGregor: https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

    Parameters:
    algorithm: Whether to use algorithm #1 or #2. Defaults to #1.
    dtype: The float datatype to use in computation of the PLS algorithm. Defaults to numpy.float64. Using a lower precision will yield significantly worse results when using an increasing number of components due to propagation of numerical errors.
    """

    def __init__(self, algorithm: int = 1, dtype: np.float_ = np.float64) -> None:
        self.algorithm = algorithm
        self.dtype = dtype
        self.name = "PLS"

    def fit(self, X: npt.ArrayLike, Y: npt.ArrayLike, A: int) -> None:
        """
        Parameters:
        X: Predictor variables matrix (N x K)
        Y: Response variables matrix (N x M)
        A: Number of components in the PLS model

        Returns:
        None

        Sets:
        self.B: PLS regression coefficients matrix (A x K x M)
        self.W: PLS weights matrix for X (K x A)
        self.P: PLS loadings matrix for X (K x A)
        self.Q: PLS Loadings matrix for Y (M x A)
        self.R: PLS weights matrix to compute scores T directly from original X (K x A)
        if algorithm is 1, then also sets self.T which is a PLS scores matrix of X (N x A)

        """
        X = np.array(X, dtype=self.dtype)
        Y = np.array(Y, dtype=self.dtype)

        N, K = X.shape
        M = Y.shape[1]

        self.B = np.zeros(shape=(A, K, M), dtype=self.dtype)
        self._W = np.zeros(shape=(A, K), dtype=self.dtype)
        self._P = np.zeros(shape=(A, K), dtype=self.dtype)
        self._Q = np.zeros(shape=(A, M), dtype=self.dtype)
        self._R = np.zeros(shape=(A, K), dtype=self.dtype)
        self.W = self._W.T
        self.P = self._P.T
        self.Q = self._Q.T
        self.R = self._R.T
        if self.algorithm == 1:
            self._T = np.zeros(shape=(A, N), dtype=self.dtype)
            self.T = self._T.T
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
                print(norm)
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
                    print(norm)
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
                    print(norm)
                    if np.isclose(norm, 0, atol=np.finfo(np.float64).eps, rtol=0):
                        warnings.warn(
                            f"Weight is close to zero. Stopping fitting after A = {i} component(s)."
                        )
                        break
                    w = eig_vecs[:, -1:]
            self._W[i] = w.squeeze()

            # Step 3
            r = np.copy(w)
            for j in range(i):
                r = r - self._P[j].reshape(-1, 1).T @ w * self._R[j].reshape(-1, 1)
            self._R[i] = r.squeeze()

            # Step 4
            if self.algorithm == 1:
                t = X @ r
                self._T[i] = t.squeeze()
                tTt = t.T @ t
                p = (t.T @ X).T / tTt
            elif self.algorithm == 2:
                rXTX = r.T @ XTX
                tTt = rXTX @ r
                p = rXTX.T / tTt
            q = (r.T @ XTY).T / tTt
            self._P[i] = p.squeeze()
            self._Q[i] = q.squeeze()

            # Step 5
            XTY = XTY - (p @ q.T) * tTt

            # Compute regression coefficients
            self.B[i] = self.B[i - 1] + r @ q.T

    def predict(self, X: npt.ArrayLike, A: None | int = None) -> npt.NDArray[np.float_]:
        """
        Parameters:
        X: Predictor variables matrix (N x K)
        A: Integer number of components to use in the prediction or None. If None, return the predictions for every component. Defaults to the maximum number of components, the model was fitted with.

        Returns:
        Y_hat: Predicted response variables matrix (N x M) if A is int or (A x N x M) if a is None.
        """
        X = np.array(X, dtype=self.dtype)
        if A is None:
            return X @ self.B
        return X @ self.B[A - 1]
