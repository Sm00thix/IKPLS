import warnings
from typing import Union

import joblib
import numpy as np
import numpy.linalg as la
import numpy.typing as npt
from joblib import Parallel, delayed


class PLS:
    """
    Implements fast cross-validation with partial least-squares regression using Improved Kernel PLS by Dayal and MacGregor: https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23

    Parameters
    ----------
    algorithm : int
        Whether to use Improved Kernel PLS Algorithm #1 or #2. Defaults to 1.

    dtype : numpy.float, default=numpy.float64
        The float datatype to use in computation of the PLS algorithm. Using a lower precision than float64 will yield significantly worse results when using an increasing number of components due to propagation of numerical errors.

    Raises
    ------
    ValueError
        If `algorithm` is not 1 or 2.
    """

    def __init__(self, algorithm: int = 1, dtype: np.float_ = np.float64) -> None:
        self.algorithm = algorithm
        self.dtype = dtype
        self.name = f"Improved Kernel PLS Algorithm #{algorithm}"
        if self.algorithm not in [1, 2]:
            raise ValueError(
                f"Invalid algorithm: {self.algorithm}. Algorithm must be 1 or 2."
            )

    def _stateless_fit(
        self,
        validation_indices: npt.ArrayLike,
    ) -> Union[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
    ]:
        """
        Fits Improved Kernel PLS Algorithm #1 on `X` and `Y` using `A` components.

        Parameters
        ----------
        validation_indices : Array of shape (N,)
            Boolean array defining indices into X and Y corresponding to validation samples.

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

        training_X_mean : Array of shape (1, K)
            Mean row of training X. Will be an array of zeros if `self.center` is False.

        training_Y_mean : Array of shape (1, M)
            Mean row of training Y. Will be an array of zeros if `self.center` is False.

        training_X_std : Array of shape (1, K)
            Sample standard deviation row of training X. Will be an array of ones if `self.scale` is False. Any zero standard deviations will be replaced with ones.

        training_Y_std : Array of shape (1, M)
            Sample standard deviation row of training Y. Will be an array of ones if `self.scale` is False. Any zero standard deviations will be replaced with ones.

        Warns
        -----
        UserWarning.
            If at any point during iteration over the number of components `A`, the residual goes below machine precision for jnp.float64.
        """

        B = np.zeros(shape=(self.A, self.K, self.M), dtype=self.dtype)
        WT = np.zeros(shape=(self.A, self.K), dtype=self.dtype)
        PT = np.zeros(shape=(self.A, self.K), dtype=self.dtype)
        QT = np.zeros(shape=(self.A, self.M), dtype=self.dtype)
        RT = np.zeros(shape=(self.A, self.K), dtype=self.dtype)
        W = WT.T
        P = PT.T
        Q = QT.T
        R = RT.T
        if self.algorithm == 1:
            TT = np.zeros(
                shape=(self.A, self.N - np.sum(validation_indices)), dtype=self.dtype
            )
            T = TT.T

        # Extract training XTY
        validation_X = self.X[validation_indices]
        validation_Y = self.Y[validation_indices]
        if self.center:
            validation_size = np.sum(validation_indices)
            training_size = self.N - validation_size
            N_total_over_N_train = self.N / training_size
            N_val_over_N_train = validation_size / training_size
            training_X_mean = (
                N_total_over_N_train * self.X_mean
                - N_val_over_N_train * np.mean(validation_X, axis=0, keepdims=True)
            )
            training_Y_mean = (
                N_total_over_N_train * self.Y_mean
                - N_val_over_N_train * np.mean(validation_Y, axis=0, keepdims=True)
            )
            if self.scale:
                train_sum_X = self.sum_X - np.sum(validation_X, axis=0, keepdims=True)
                train_sum_sq_X = self.sum_sq_X - np.sum(
                    validation_X**2, axis=0, keepdims=True
                )
                training_X_std = np.sqrt(
                    1
                    / (training_size - 1)
                    * (
                        -2 * training_X_mean * train_sum_X
                        + training_size * training_X_mean**2
                        + train_sum_sq_X
                    )
                )
                training_X_std[training_X_std == 0] = 1
                train_sum_Y = self.sum_Y - np.sum(validation_Y, axis=0, keepdims=True)
                train_sum_sq_Y = self.sum_sq_Y - np.sum(
                    validation_Y**2, axis=0, keepdims=True
                )
                training_Y_std = np.sqrt(
                    1
                    / (training_size - 1)
                    * (
                        -2 * training_Y_mean * train_sum_Y
                        + training_size * training_Y_mean**2
                        + train_sum_sq_Y
                    )
                )
                training_Y_std[training_Y_std == 0] = 1
        # Subtract the validation set's contribution
        training_XTY = self.XTY - validation_X.T @ validation_Y
        if self.center:
            # Apply the training set centering
            training_XTY = training_XTY - training_size * (
                training_X_mean.T @ training_Y_mean
            )
            if self.scale:
                # Apply the training set scaling
                training_XTY = training_XTY / (training_X_std.T @ training_Y_std)
        if self.algorithm == 1:
            training_X = self.X[~validation_indices]
            if self.center:
                # Apply the training set centering
                training_X = training_X - training_X_mean
                if self.scale:
                    # Apply the training set scaling
                    training_X = training_X / training_X_std
        else:
            # Used for algorithm #2
            # Subtract the validation set's contribution
            training_XTX = self.XTX - validation_X.T @ validation_X
            if self.center:
                # Apply the training set centering
                training_XTX = training_XTX - training_size * (
                    training_X_mean.T @ training_X_mean
                )
                if self.scale:
                    # Apply the training set scaling
                    training_XTX = training_XTX / (training_X_std.T @ training_X_std)

        for i in range(self.A):
            # Step 2
            if self.M == 1:
                norm = la.norm(training_XTY, ord=2)
                if np.isclose(norm, 0, atol=np.finfo(np.float64).eps, rtol=0):
                    warnings.warn(
                        f"Weight is close to zero. Stopping fitting after A = {i} component(s)."
                    )
                    break
                w = training_XTY / norm
            else:
                if self.M < self.K:
                    training_XTYTtraining_XTY = training_XTY.T @ training_XTY
                    eig_vals, eig_vecs = la.eigh(training_XTYTtraining_XTY)
                    q = eig_vecs[:, -1:]
                    w = training_XTY @ q
                    norm = la.norm(w)
                    if np.isclose(norm, 0, atol=np.finfo(np.float64).eps, rtol=0):
                        warnings.warn(
                            f"Weight is close to zero. Stopping fitting after A = {i} component(s)."
                        )
                        break
                    w = w / norm
                else:
                    training_XTYYTX = training_XTY @ training_XTY.T
                    eig_vals, eig_vecs = la.eigh(training_XTYYTX)
                    norm = eig_vals[-1]
                    if np.isclose(norm, 0, atol=np.finfo(np.float64).eps, rtol=0):
                        warnings.warn(
                            f"Weight is close to zero. Stopping fitting after A = {i} component(s)."
                        )
                        break
                    w = eig_vecs[:, -1:]
            WT[i] = w.squeeze()

            # Step 3
            r = np.copy(w)
            for j in range(i):
                r = r - PT[j].reshape(-1, 1).T @ w * RT[j].reshape(-1, 1)
            RT[i] = r.squeeze()

            # Step 4
            if self.algorithm == 1:
                t = training_X @ r
                TT[i] = t.squeeze()
                tTt = t.T @ t
                p = (t.T @ training_X).T / tTt
            elif self.algorithm == 2:
                rtraining_XTX = r.T @ training_XTX
                tTt = rtraining_XTX @ r
                p = rtraining_XTX.T / tTt
            q = (r.T @ training_XTY).T / tTt
            PT[i] = p.squeeze()
            QT[i] = q.squeeze()

            # Step 5
            training_XTY = training_XTY - (p @ q.T) * tTt

            # Compute regression coefficients
            B[i] = B[i - 1] + r @ q.T
        if not self.center:
            training_X_mean = np.zeros((1, self.K))
            training_Y_mean = np.zeros((1, self.M))
        if not self.scale:
            training_X_std = np.ones((1, self.K))
            training_Y_std = np.ones((1, self.M))
        if self.algorithm == 1:
            return (
                B,
                W,
                P,
                Q,
                R,
                T,
                training_X_mean,
                training_Y_mean,
                training_X_std,
                training_Y_std,
            )
        else:
            return (
                B,
                W,
                P,
                Q,
                R,
                training_X_mean,
                training_Y_mean,
                training_X_std,
                training_Y_std,
            )

    def _stateless_predict(
        self,
        indices: npt.ArrayLike,
        B: npt.ArrayLike,
        training_X_mean: npt.ArrayLike,
        training_Y_mean: npt.ArrayLike,
        training_X_std: npt.ArrayLike,
        training_Y_std: npt.ArrayLike,
        n_components: Union[None, int] = None,
    ) -> npt.NDArray[np.float_]:
        """
        Predicts with Improved Kernel PLS Algorithm #1 on `X` with `B` using `n_components` components. If `n_components` is None, then predictions are returned for all number of components.

        Parameters
        ----------
        indices : Array of shape (N,)
            Boolean array defining indices into X and Y corresponding to samples on which to predict.

        B : Array of shape (A, K, M)
            PLS regression coefficients tensor.

        training_X_mean : Array of shape (1, K)
            Mean row of training X. If self.center is False, then this should be an array of zeros.

        training_Y_mean : Array of shape (1, M)
            Mean row of training Y. If self.center is False, then this should be an array of zeros.

        training_X_std : Array of shape (1, K)
            Sample standard deviation row of training X. If self.scale is False, then this should be an array of ones. Any zero standard deviations should be replaced with ones.

        training_Y_std : Array of shape (1, M)
            Sample standard deviation row of training Y. If self.scale is False, then this should be an array of ones. Any zero standard deviations should be replaced with ones.

        n_components : int or None, optional
            Number of components in the PLS model. If None, then all number of components are used.

        Returns
        -------
        Y_pred : Array of shape (N_pred, M) or (A, N_pred, M)
            If `n_components` is an int, then an array of shape (N_pred, M) with the predictions for that specific number of components is used. If `n_components` is None, returns a prediction for each number of components up to `A`.
        """

        predictor_variables = self.X[indices]
        # Apply the potential training set centering
        predictor_variables = (predictor_variables - training_X_mean) / training_X_std
        if n_components is None:
            Y_pred = predictor_variables @ B
        else:
            Y_pred = predictor_variables @ B[n_components - 1]
        # Add the potential training set bias
        return Y_pred * training_Y_std + training_Y_mean

    def _stateless_fit_predict_eval(self, validation_indices, metric_function):
        """
        Fits Improved Kernel PLS Algorithm #1 or #2 on `X` or `XTX`, `XTY` and `Y` using `A` components, predicts on `X` and evaluates predictions using `metric_function`. The fit is performed on the training set defined by `validation_indices`. The prediction is performed on the validation set defined by `validation_indices`.

        Parameters
        ----------
        validation_indices : Array of shape (N,)
            Boolean array defining indices into X and Y corresponding to validation samples.

        metric_function : Callable receiving arrays `Y_true` and `Y_pred` and returning Any

        Returns
        -------
        metric : Any
            The result of evaluating `metric_function` on the validation set.
        """
        matrices = self._stateless_fit(validation_indices)
        B = matrices[0]
        training_X_mean = matrices[-4]
        training_Y_mean = matrices[-3]
        training_X_std = matrices[-2]
        training_Y_std = matrices[-1]
        Y_pred = self._stateless_predict(
            validation_indices,
            B,
            training_X_mean,
            training_Y_mean,
            training_X_std,
            training_Y_std,
        )
        return metric_function(self.Y[validation_indices], Y_pred)

    def cross_validate(
        self,
        X,
        Y,
        A,
        cv_splits,
        metric_function,
        center=False,
        scale=False,
        n_jobs=-1,
        verbose=10,
    ):
        """
        Cross-validates the PLS model using `cv_splits` splits on `X` and `Y` with `n_components` components evaluating results with `metric_function`.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        Y : Array of shape (N, M)
            Target variables.

        A : int
            Number of components in the PLS model.

        cv_splits : Array of shape (N,)
            An array defining cross-validation splits. Each unique value in `cv_splits` corresponds to a different fold.

        metric_function : Callable receiving arrays `Y_test` and `Y_pred` and returning Any
            Computes a metric based on true values `Y_test` and predicted values `Y_pred`. `Y_pred` contains a prediction for all `A` components.

        mean_centering : bool, optional default=True
            Whether to mean X and Y across the sample axis before fitting. The mean is subtracted from X and Y before fitting and added back to the predictions. This implementation ensures that no data leakage occurs between training and validation sets.

        center : bool, optional default=False
            Whether to center `X` and `Y` before fitting by subtracting a mean row from each. The centering is computed on the training set for each fold to avoid data leakage. The centering is undone before returning predictions. Setting this to True while using multiple jobs will increase the memory consumption as each job will then have to keep its own copy of the data with its specific centering.

        scale : bool, optional default=False, only used if `center` is True
            Whether to scale `X` and `Y` before fitting by dividing each row by its standard deviation. The scaling is computed on the training set for each fold to avoid data leakage. The scaling is undone before returning predictions. Setting this to True while using multiple jobs will increase the memory consumption as each job will then have to keep its own copy of the data with its specific scaling.

        n_jobs : int, optional default=-1
            Number of parallel jobs to use. A value of -1 will use the minimum of all available cores and the number of unique values in `cv_splits`.

        verbose : int, optional default=10
            Controls verbosity of parallel jobs.

        Returns
        -------
        metrics : list of Any
            A list of the results of evaluating `metric_function` on each fold.
        """

        self.X = np.asarray(X, dtype=self.dtype)
        self.Y = np.asarray(Y, dtype=self.dtype)
        self.A = A
        self.center = center
        self.scale = scale
        self.N, self.K = X.shape
        self.M = Y.shape[1]
        unique_splits = np.unique(cv_splits)

        if n_jobs == -1:
            n_jobs = min(joblib.cpu_count(), unique_splits.size)

        print(
            f"Cross-validating Improved Kernel PLS Algorithm {self.algorithm} with {A} components on {len(unique_splits)} unique splits using {n_jobs} parallel processes."
        )

        if self.center:
            # We can compute these once for the entire dataset and subtract the validation parts during cross-validation.
            self.X_mean = np.mean(self.X, axis=0, keepdims=True)
            self.Y_mean = np.mean(self.Y, axis=0, keepdims=True)
            if self.scale:
                self.sum_X = np.sum(self.X, axis=0, keepdims=True)
                self.sum_Y = np.sum(self.Y, axis=0, keepdims=True)
                self.sum_sq_X = np.sum(self.X**2, axis=0, keepdims=True)
                self.sum_sq_Y = np.sum(self.Y**2, axis=0, keepdims=True)

        if verbose > 0:
            print("Computing total XTY...")
        self.XTY = self.X.T @ self.Y
        if verbose > 0:
            print("Done!")
        if self.algorithm == 2:
            if verbose > 0:
                print("Computing total XTX...")
            self.XTX = self.X.T @ self.X
            if verbose > 0:
                print("Done!")

        def worker(split):
            validation_indices = cv_splits == split
            return self._stateless_fit_predict_eval(validation_indices, metric_function)

        metrics = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(worker)(split) for split in unique_splits
        )

        return metrics
