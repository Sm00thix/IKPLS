import numpy as np

from ikpls.numpy_ikpls import PLS

if __name__ == "__main__":
    N = 100  # Number of samples.
    K = 50  # Number of features.
    M = 10  # Number of targets.
    A = 20  # Number of latent variables (PLS components).

    # Using float64 is important for numerical stability.
    X = np.random.uniform(size=(N, K)).astype(np.float64)
    Y = np.random.uniform(size=(N, M)).astype(np.float64)

    # The other PLS algorithms and implementations have the same interface for fit() and predict().
    np_ikpls_alg_1 = PLS(algorithm=1)
    np_ikpls_alg_1.fit(X, Y, A)

    y_pred = np_ikpls_alg_1.predict(
        X
    )  # Has shape (A, N, M) = (20, 100, 10). Contains a prediction for all possible number of components up to and including A.
    y_pred_20_components = np_ikpls_alg_1.predict(
        X, n_components=20
    )  # Has shape (N, M) = (100, 10).
    (y_pred_20_components == y_pred[19]).all()  # True

    # The internal model parameters can be accessed as follows:
    np_ikpls_alg_1.B  # Regression coefficients tensor of shape (A, K, M) = (20, 50, 10).
    np_ikpls_alg_1.W  # X weights matrix of shape (K, A) = (50, 20).
    np_ikpls_alg_1.P  # X loadings matrix of shape (K, A) = (50, 20).
    np_ikpls_alg_1.Q  # Y loadings matrix of shape (M, A) = (10, 20).
    np_ikpls_alg_1.R  # X rotations matrix of shape (K, A) = (50, 20).
    np_ikpls_alg_1.T  # X scores matrix of shape (N, A) = (100, 20). This is only computed for IKPLS Algorithm #1.
