"""
Contains the PLS Class which implements partial least-squares regression using Improved
Kernel PLS by Dayal and MacGregor:
https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

The PLS class subclasses scikit-learn's BaseEstimator to ensure compatibility with e.g.
scikit-learn's cross_validate. It is written using NumPy.

Author: Ole-Christian Galbo Engstrøm
E-mail: ole.e@di.ku.dk
"""

import warnings
from typing import Union

import numpy as np
import numpy.linalg as la
import numpy.typing as npt
from sklearn.base import BaseEstimator


class PLS(BaseEstimator):
    """
    Implements partial least-squares regression using Improved Kernel PLS by Dayal and
    MacGregor:
    https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

    Parameters
    ----------
    algorithm : int, default=1
        Whether to use Improved Kernel PLS Algorithm #1 or #2.

    center_X : bool, default=True
        Whether to center `X` before fitting by subtracting its row of
        column-wise means from each row.

    center_Y : bool, default=True
        Whether to center `Y` before fitting by subtracting its row of
        column-wise means from each row.

    scale_X : bool, default=True
        Whether to scale `X` before fitting by dividing each row with the row of `X`'s
        column-wise standard deviations. Bessel's correction for the unbiased estimate
        of the sample standard deviation is used.

    scale_Y : bool, default=True
        Whether to scale `Y` before fitting by dividing each row with the row of `X`'s
        column-wise standard deviations. Bessel's correction for the unbiased estimate
        of the sample standard deviation is used.

    copy : bool, default=True
        Whether to copy `X` and `Y` in fit before potentially applying centering and
        scaling. If True, then the data is copied before fitting. If False, and `dtype`
        matches the type of `X` and `Y`, then centering and scaling is done inplace,
        modifying both arrays.

    dtype : numpy.float, default=numpy.float64
        The float datatype to use in computation of the PLS algorithm. Using a lower
        precision than float64 will yield significantly worse results when using an
        increasing number of components due to propagation of numerical errors.

    Raises
    ------
    ValueError
        If `algorithm` is not 1 or 2.

    Notes
    -----
    Any centering and scaling is undone before returning predictions to ensure that
    predictions are on the original scale. If both centering and scaling are True, then
    the data is first centered and then scaled.
    """

    def __init__(
        self,
        algorithm: int = 1,
        center_X: bool = True,
        center_Y: bool = True,
        scale_X: bool = True,
        scale_Y: bool = True,
        copy: bool = True,
        dtype: np.float_ = np.float64,
    ) -> None:
        self.algorithm = algorithm
        self.center_X = center_X
        self.center_Y = center_Y
        self.scale_X = scale_X
        self.scale_Y = scale_Y
        self.copy = copy
        self.dtype = dtype
        self.eps = np.finfo(dtype).eps
        self.name = f"Improved Kernel PLS Algorithm #{algorithm}"
        if self.algorithm not in [1, 2]:
            raise ValueError(
                f"Invalid algorithm: {self.algorithm}. Algorithm must be 1 or 2."
            )
        self.A = None
        self.N = None
        self.K = None
        self.M = None
        self.B = None
        self.W = None
        self.P = None
        self.Q = None
        self.R = None
        self.T = None
        self.X_mean = None
        self.Y_mean = None
        self.X_std = None
        self.Y_std = None

    def _weight_warning(self, i: int) -> None:
        """
        Warns the user that the weight is close to zero.

        Parameters
        ----------
        i : int
            Number of components.

        Returns
        -------
        None.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("always", UserWarning)
            warnings.warn(
                f"Weight is close to zero. Results with A = {i} component(s) or higher"
                " may be unstable."
            )

    def fit(self, X: npt.ArrayLike, Y: npt.ArrayLike, A: int) -> None:
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
            If at any point during iteration over the number of components `A`, the
            residual goes below machine precision for np.float64.
        """
        X = np.asarray(X, dtype=self.dtype)
        Y = np.asarray(Y, dtype=self.dtype)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if (self.center_X or self.scale_X) and self.copy:
            X = X.copy()

        if (self.center_Y or self.scale_Y) and self.copy:
            Y = Y.copy()

        if self.center_X:
            self.X_mean = X.mean(axis=0, dtype=self.dtype, keepdims=True)
            X -= self.X_mean

        if self.center_Y:
            self.Y_mean = Y.mean(axis=0, dtype=self.dtype, keepdims=True)
            Y -= self.Y_mean

        if self.scale_X:
            self.X_std = X.std(axis=0, ddof=1, dtype=self.dtype, keepdims=True)
            self.X_std[np.abs(self.X_std) <= self.eps] = 1
            X /= self.X_std

        if self.scale_Y:
            self.Y_std = Y.std(axis=0, ddof=1, dtype=self.dtype, keepdims=True)
            self.Y_std[np.abs(self.Y_std) <= self.eps] = 1
            Y /= self.Y_std

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
                if np.isclose(norm, 0, atol=self.eps, rtol=0):
                    self._weight_warning(i)
                    break
                w = XTY / norm
            else:
                if M < K:
                    XTYTXTY = XTY.T @ XTY
                    eig_vals, eig_vecs = la.eigh(XTYTXTY)
                    q = eig_vecs[:, -1:]
                    w = XTY @ q
                    norm = la.norm(w)
                    if np.isclose(norm, 0, atol=self.eps, rtol=0):
                        self._weight_warning(i)
                        break
                    w = w / norm
                else:
                    XTYYTX = XTY @ XTY.T
                    eig_vals, eig_vecs = la.eigh(XTYYTX)
                    norm = eig_vals[-1]
                    if np.isclose(norm, 0, atol=self.eps, rtol=0):
                        self._weight_warning(i)
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
        self, X: npt.ArrayLike, n_components: Union[None, int] = None
    ) -> npt.NDArray[np.float_]:
        """
        Predicts with Improved Kernel PLS Algorithm #1 on `X` with `B` using
        `n_components` components. If `n_components` is None, then predictions are
        returned for all number of components.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        n_components : int or None, optional, default=None.
            Number of components in the PLS model. If None, then all number of
            components are used.

        Returns
        -------
        Y_pred : Array of shape (N, M) or (A, N, M)
            If `n_components` is an int, then an array of shape (N, M) with the
            predictions for that specific number of components is used. If
            `n_components` is None, returns a prediction for each number of components
            up to `A`.
        """
        X = np.asarray(X, dtype=self.dtype)
        if self.center_X:
            X = X - self.X_mean
        if self.scale_X:
            X = X / self.X_std

        if n_components is None:
            Y_pred = X @ self.B
        else:
            Y_pred = X @ self.B[n_components - 1]

        if self.scale_Y:
            Y_pred = Y_pred * self.Y_std
        if self.center_Y:
            Y_pred = Y_pred + self.Y_mean
        return Y_pred
