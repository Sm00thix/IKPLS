import numpy as np

from ikpls.jax_ikpls_alg_1 import PLS

if __name__ == "__main__":
    N = 100  # Number of samples.
    K = 50  # Number of features.
    M = 10  # Number of targets.
    A = 20  # Number of latent variables (PLS components).

    X = np.random.uniform(size=(N, K)).astype(np.float64)
    Y = np.random.uniform(size=(N, M)).astype(np.float64)

    jax_ikpls_alg_1 = PLS()
    jax_ikpls_alg_1.fit(X, Y, A)

    y_pred = jax_ikpls_alg_1.predict(
        X
    )  # Has shape (A, N, M) = (20, 100, 10). Contains a prediction for all possible numbers of components up to and including A.
    y_pred_20_components = jax_ikpls_alg_1.predict(
        X, n_components=20
    )  # Has shape (N, M) = (100, 10).
    np.allclose(
        y_pred_20_components, y_pred[19], atol=0, rtol=1e14
    )  # True. Exact equality might not hold due to numerical differences.

    # The internal model parameters can be accessed as follows:
    jax_ikpls_alg_1.B  # Regression coefficients tensor of shape (A, K, M) = (20, 50, 10).
    jax_ikpls_alg_1.W  # X weights matrix of shape (K, A) = (50, 20).
    jax_ikpls_alg_1.P  # X loadings matrix of shape (K, A) = (50, 20).
    jax_ikpls_alg_1.Q  # Y loadings matrix of shape (M, A) = (10, 20).
    jax_ikpls_alg_1.R  # X rotations matrix of shape (K, A) = (50, 20).
    jax_ikpls_alg_1.T  # X scores matrix of shape (N, A) = (100, 20). This is only computed for IKPLS Algorithm #1.
