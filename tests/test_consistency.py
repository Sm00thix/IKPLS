from sklearn.cross_decomposition import PLSRegression as SkPLS
from algorithms.jax_ikpls_alg_1 import PLS as JAX_Alg_1
from algorithms.jax_ikpls_alg_2 import PLS as JAX_Alg_2
from algorithms.numpy_ikpls import PLS as NpPLS

# import load_data
from . import load_data

import numpy as np
import numpy.typing as npt
from jax import numpy as jnp
from numpy.testing import assert_allclose


class TestClass:
    csv = load_data.load_csv()
    raw_spectra = load_data.load_spectra()

    def load_X(self):
        return np.copy(self.raw_spectra)

    def load_Y(self, values: list[str]) -> npt.NDArray[np.float_]:
        target_values = self.csv[values].to_numpy()
        return target_values

    def fit_models(self, X, Y, n_components):
        jnp_X = jnp.array(X)
        jnp_Y = jnp.array(Y)
        sk_pls = SkPLS(n_components=n_components)
        np_pls_alg_1 = NpPLS(algorithm=1)
        np_pls_alg_2 = NpPLS(algorithm=2)
        jax_pls_alg_1 = JAX_Alg_1()
        jax_pls_alg_2 = JAX_Alg_2()

        sk_pls.fit(X=X, Y=Y)
        np_pls_alg_1.fit(X=X, Y=Y, A=n_components)
        np_pls_alg_2.fit(X=X, Y=Y, A=n_components)
        jax_pls_alg_1.fit(X=jnp_X, Y=jnp_Y, A=n_components)
        jax_pls_alg_2.fit(X=jnp_X, Y=jnp_Y, A=n_components)

        # Reconstruct SkPLS regression matrix for all components
        sk_B = np.empty(np_pls_alg_1.B.shape)
        for i in range(sk_B.shape[0]):
            sk_B_at_component_i = np.dot(
                sk_pls.x_rotations_[..., : i + 1],
                sk_pls.y_loadings_[..., : i + 1].T,
            )
            sk_B[i] = sk_B_at_component_i
        return sk_pls, sk_B, np_pls_alg_1, np_pls_alg_2, jax_pls_alg_1, jax_pls_alg_2

    def assert_matrix_orthogonal(self, M, atol, rtol):
        MTM = np.dot(M.T, M)
        assert_allclose(MTM, np.diag(np.diag(MTM)), atol=atol, rtol=rtol)

    def check_internal_matrices(
        self,
        sk_pls,
        sk_B,
        np_pls_alg_1,
        np_pls_alg_2,
        jax_pls_alg_1,
        jax_pls_alg_2,
        atol,
        rtol,
        n_good_components=-1,
    ):
        if n_good_components == -1:
            n_good_components = np_pls_alg_1.A
        # X weights
        assert_allclose(
            np.abs(np_pls_alg_1.W[..., :n_good_components]),
            np.abs(sk_pls.x_weights_[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_2.W[..., :n_good_components]),
            np.abs(sk_pls.x_weights_[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np.array(jax_pls_alg_1.W)[..., :n_good_components]),
            np.abs(sk_pls.x_weights_[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np.array(jax_pls_alg_2.W)[..., :n_good_components]),
            np.abs(sk_pls.x_weights_[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )

        # X loadings
        assert_allclose(
            np.abs(np_pls_alg_1.P[..., :n_good_components]),
            np.abs(sk_pls.x_loadings_[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_2.P[..., :n_good_components]),
            np.abs(sk_pls.x_loadings_[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np.array(jax_pls_alg_1.P)[..., :n_good_components]),
            np.abs(sk_pls.x_loadings_[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np.array(jax_pls_alg_2.P)[..., :n_good_components]),
            np.abs(sk_pls.x_loadings_[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )

        # Y loadings
        assert_allclose(
            np.abs(np_pls_alg_1.Q[..., :n_good_components]),
            np.abs(sk_pls.y_loadings_[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_2.Q[..., :n_good_components]),
            np.abs(sk_pls.y_loadings_[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np.array(jax_pls_alg_1.Q)[..., :n_good_components]),
            np.abs(sk_pls.y_loadings_[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np.array(jax_pls_alg_2.Q)[..., :n_good_components]),
            np.abs(sk_pls.y_loadings_[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )

        # X rotations
        assert_allclose(
            np.abs(np_pls_alg_1.R[..., :n_good_components]),
            np.abs(sk_pls.x_rotations_[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_2.R[..., :n_good_components]),
            np.abs(sk_pls.x_rotations_[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np.array(jax_pls_alg_1.R[..., :n_good_components])),
            np.abs(sk_pls.x_rotations_[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np.array(jax_pls_alg_2.R[..., :n_good_components])),
            np.abs(sk_pls.x_rotations_[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )

        # X scores - only computed by IKPLS Algorithm #1
        assert_allclose(
            np.abs(np_pls_alg_1.T[..., :n_good_components]),
            np.abs(sk_pls.x_scores_[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np.array(jax_pls_alg_1.T[..., :n_good_components])),
            np.abs(sk_pls.x_scores_[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )

        # Regression matrices
        assert_allclose(
            np_pls_alg_1.B[:n_good_components],
            sk_B[:n_good_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np_pls_alg_2.B[:n_good_components],
            sk_B[:n_good_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(jax_pls_alg_1.B)[:n_good_components],
            sk_B[:n_good_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(jax_pls_alg_2.B)[:n_good_components],
            sk_B[:n_good_components],
            atol=atol,
            rtol=rtol,
        )

    def check_predictions(
        self,
        sk_B,
        np_pls_alg_1,
        np_pls_alg_2,
        jax_pls_alg_1,
        jax_pls_alg_2,
        X,
        atol,
        rtol,
        n_good_components=-1,
    ):
        if n_good_components == -1:
            n_good_components = np_pls_alg_1.A
        # Check predictions for each and all possible number of components.
        sk_all_preds = X @ sk_B
        assert_allclose(
            np_pls_alg_1.predict(X)[:n_good_components],
            sk_all_preds[:n_good_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np_pls_alg_2.predict(X)[:n_good_components],
            sk_all_preds[:n_good_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(jax_pls_alg_1.predict(X))[:n_good_components],
            sk_all_preds[:n_good_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(jax_pls_alg_2.predict(X))[:n_good_components],
            sk_all_preds[:n_good_components],
            atol=atol,
            rtol=rtol,
        )

        # Check predictions using the largest good number of components.
        sk_final_pred = sk_all_preds[n_good_components - 1]
        assert_allclose(
            np_pls_alg_1.predict(X, A=n_good_components),
            sk_final_pred,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np_pls_alg_2.predict(X, A=n_good_components),
            sk_final_pred,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(jax_pls_alg_1.predict(X, A=n_good_components)),
            sk_final_pred,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(jax_pls_alg_2.predict(X, A=n_good_components)),
            sk_final_pred,
            atol=atol,
            rtol=rtol,
        )

    def check_orthogonality_properties(
        self,
        np_pls_alg_1,
        np_pls_alg_2,
        jax_pls_alg_1,
        jax_pls_alg_2,
        atol,
        rtol,
        n_good_components=-1,
    ):
        if n_good_components == -1:
            n_good_components = np_pls_alg_1.A
        # X weights should be orthogonal
        self.assert_matrix_orthogonal(
            np_pls_alg_1.W[..., :n_good_components], atol=atol, rtol=rtol
        )
        self.assert_matrix_orthogonal(
            np_pls_alg_2.W[..., :n_good_components], atol=atol, rtol=rtol
        )
        self.assert_matrix_orthogonal(
            np.array(jax_pls_alg_1.W)[..., :n_good_components], atol=atol, rtol=rtol
        )
        self.assert_matrix_orthogonal(
            np.array(jax_pls_alg_2.W)[..., :n_good_components], atol=atol, rtol=rtol
        )

        # X scores (only computed by algorithm 1) should be orthogonal
        self.assert_matrix_orthogonal(
            np_pls_alg_1.T[..., :n_good_components], atol=atol, rtol=rtol
        )
        self.assert_matrix_orthogonal(
            np.array(jax_pls_alg_1.T)[..., :n_good_components], atol=atol, rtol=rtol
        )

    def check_equality_properties(
        self, np_pls_alg_1, jax_pls_alg_1, X, atol, rtol, n_good_components=-1
    ):
        if n_good_components == -1:
            n_good_components = np_pls_alg_1.A
        # X can be reconstructed by multiplying X scores (T) and the transpose of X loadings (P)
        assert_allclose(
            np.dot(
                np_pls_alg_1.T[..., :n_good_components],
                np_pls_alg_1.P[..., :n_good_components].T,
            ),
            X,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.dot(
                np.array(jax_pls_alg_1.T[..., :n_good_components]),
                np.array(jax_pls_alg_1.P[..., :n_good_components]).T,
            ),
            X,
            atol=atol,
            rtol=rtol,
        )

        # X multiplied by X rotations (R) should be equal to X scores (T)
        assert_allclose(
            np.dot(X, np_pls_alg_1.R[..., :n_good_components]),
            np_pls_alg_1.T[..., :n_good_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.dot(X, np.array(jax_pls_alg_1.R[..., :n_good_components])),
            np.array(jax_pls_alg_1.T[..., :n_good_components]),
            atol=atol,
            rtol=rtol,
        )

    def test_pls_1(self):
        """
        Test PLS1.
        """
        X = self.load_X()
        X -= np.mean(
            X, axis=0
        )  # SkLearn's PLS implementation always centers its X input. This ensures that the X input is always centered for all algorithms.
        X /= np.std(X, axis=0)  # Let's also scale them for better numerical stability
        Y = self.load_Y(["Protein"])
        Y -= np.mean(Y, axis=0, keepdims=True) # SkLearn's PLS implementation always centers its Y input. This ensures that the Y input is always centered for all algorithms.
        Y /= np.std(Y, axis=0, keepdims=True)  # Scale for numerical stability
        n_components = 25
        assert Y.shape[1] == 1
        (
            sk_pls,
            sk_B,
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
        ) = self.fit_models(X=X, Y=Y, n_components=n_components)
        self.check_equality_properties(
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            X=X,
            atol=1e-1,
            rtol=1e-5,
        )
        self.check_orthogonality_properties(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-1,
            rtol=0,
        )
        # self.check_internal_matrices(atol=1e-2, rtol=1e-2)
        self.check_predictions(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            X=X,
            atol=1e-8,
            rtol=0,
        )  # PLS1 is very numerically stable for protein.

    def test_pls_2_m_less_k(self):
        """
        Test PLS2 where the number of targets is less than the number of features (M < K).
        """
        X = self.load_X()
        X -= np.mean(
            X, axis=0
        )  # SkLearn's PLS implementation always centers its X input. This ensures that the X input is always centered for all algorithms.
        X /= np.std(X, axis=0)  # Let's also scale them for better numerical stability
        Y = self.load_Y(
            [
                "Rye_Midsummer",
                "Wheat_H1",
                "Wheat_H3",
                "Wheat_H4",
                "Wheat_H5",
                "Wheat_Halland",
                "Wheat_Oland",
                "Wheat_Spelt",
                "Moisture",
                "Protein",
            ]
        )
        Y -= np.mean(Y, axis=0, keepdims=True) # SkLearn's PLS implementation always centers its Y input. This ensures that the Y input is always centered for all algorithms.
        Y /= np.std(Y, axis=0, keepdims=True)  # Scale for numerical stability
        assert Y.shape[1] > 1
        assert Y.shape[1] < X.shape[1]
        n_components = 25
        (
            sk_pls,
            sk_B,
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
        ) = self.fit_models(X=X, Y=Y, n_components=n_components)
        self.check_equality_properties(
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            X=X,
            atol=1e-1,
            rtol=1e-5,
        )
        self.check_orthogonality_properties(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-1,
            rtol=0,
        )
        # self.check_internal_matrices(atol=1e-1, rtol=1e-1)
        self.check_predictions(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            X=X,
            atol=1e-2,
            rtol=0,
        )  # PLS2 is not as numerically stable as PLS1.

    def test_pls_2_m_eq_k(self):
        """
        Test PLS2 where the number of targets is equal to the number of features (M = K).
        """
        X = self.load_X()
        X = X[..., :10]
        X -= np.mean(
            X, axis=0
        )  # SkLearn's PLS implementation always centers its X input. This ensures that the X input is always centered for all algorithms.
        X /= np.std(X, axis=0)  # Let's also scale them for better numerical stability
        Y = self.load_Y(
            [
                "Rye_Midsummer",
                "Wheat_H1",
                "Wheat_H3",
                "Wheat_H4",
                "Wheat_H5",
                "Wheat_Halland",
                "Wheat_Oland",
                "Wheat_Spelt",
                "Moisture",
                "Protein",
            ]
        )
        Y -= np.mean(Y, axis=0, keepdims=True)  # SkLearn's PLS implementation always centers its Y input. This ensures that the Y input is always centered for all algorithms.
        Y /= np.std(Y, axis=0, keepdims=True)  # Scale for numerical stability
        assert Y.shape[1] > 1
        assert Y.shape[1] == X.shape[1]
        n_components = 10
        (
            sk_pls,
            sk_B,
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
        ) = self.fit_models(X=X, Y=Y, n_components=n_components)
        self.check_equality_properties(
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            X=X,
            atol=1e-1,
            rtol=1e-5,
        )
        self.check_orthogonality_properties(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-1,
            rtol=0,
        )
        # self.check_internal_matrices(atol=1e-1, rtol=1e-1)
        self.check_predictions(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            X=X,
            atol=1e-2,
            rtol=0,
        )  # PLS2 is not as numerically stable as PLS1.

    def test_pls_2_m_greater_k(self):
        """
        Test PLS2 where the number of targets is greater than the number of features (M > K).
        """
        X = self.load_X()
        X = X[..., :9]
        X -= np.mean(
            X, axis=0
        )  # SkLearn's PLS implementation always centers its X input. This ensures that the X input is always centered for all algorithms.
        X /= np.std(X, axis=0)  # Let's also scale them for better numerical stability
        Y = self.load_Y(
            [
                "Rye_Midsummer",
                "Wheat_H1",
                "Wheat_H3",
                "Wheat_H4",
                "Wheat_H5",
                "Wheat_Halland",
                "Wheat_Oland",
                "Wheat_Spelt",
                "Moisture",
                "Protein",
            ]
        )
        Y -= np.mean(Y, axis=0, keepdims=True)  # SkLearn's PLS implementation always centers its Y input. This ensures that the Y input is always centered for all algorithms.
        Y /= np.std(Y, axis=0, keepdims=True)  # Scale for numerical stability
        assert Y.shape[1] > 1
        assert Y.shape[1] > X.shape[1]
        n_components = 9
        (
            sk_pls,
            sk_B,
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
        ) = self.fit_models(X=X, Y=Y, n_components=n_components)
        self.check_equality_properties(
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            X=X,
            atol=1e-1,
            rtol=1e-5,
        )
        self.check_orthogonality_properties(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-1,
            rtol=0,
        )
        # self.check_internal_matrices(atol=1e-1, rtol=1e-1)
        self.check_predictions(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            X=X,
            atol=1e-2,
            rtol=0,
        )  # PLS2 is not as numerically stable as PLS1.

    def test_early_stop_fitting_pls_1(self):
        """
        The NumPy implementations will stop iterating through components if the residual comes close to 0.
        """
        vectors = np.array(
            [np.arange(2, 7, dtype=np.float64) ** (i + 1) for i in range(5)]
        )
        X = np.tile(vectors, reps=(5000, 3))
        X -= np.mean(X, axis=0)
        X /= np.std(X, axis=0)  # Let's also scale them for better numerical stability

        Y = np.sum(X, axis=1, keepdims=True)
        Y -= np.mean(Y, axis=0, keepdims=True)  # SkLearn's PLS implementation always centers its Y input. This ensures that the Y input is always centered for all algorithms.
        Y /= np.std(Y, axis=0, keepdims=True)  # Scale for numerical stability
        n_components = 10
        n_good_components = 4  # X has a rank of 4.

        assert Y.shape[1] == 1
        (
            sk_pls,
            sk_B,
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
        ) = self.fit_models(X=X, Y=Y, n_components=n_components)
        self.check_equality_properties(
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            X=X,
            atol=1e-1,
            rtol=1e-5,
            n_good_components=n_good_components,
        )
        self.check_orthogonality_properties(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-1,
            rtol=0,
            n_good_components=n_good_components,
        )
        # self.check_internal_matrices(atol=1e-2, rtol=1e-2)
        self.check_predictions(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            X=X,
            atol=1e-8,
            rtol=0,
            n_good_components=n_good_components,
        )  # PLS1 is very numerically stable for protein.

    def test_early_stop_fitting_pls_2_m_less_k(self):
        """
        The NumPy implementations will stop iterating through components if the residual comes close to 0.
        """
        vectors = np.array(
            [np.arange(2, 7, dtype=np.float64) ** (i + 1) for i in range(5)]
        )
        X = np.tile(vectors, reps=(5000, 3))
        X -= np.mean(X, axis=0)
        X /= np.std(X, axis=0)  # Let's also scale them for better numerical stability

        Y = np.hstack(
            (np.sum(X, axis=1, keepdims=True), np.sum(X, axis=1, keepdims=True) ** 2)
        )
        Y -= np.mean(Y, axis=0, keepdims=True)  # SkLearn's PLS implementation always centers its Y input. This ensures that the Y input is always centered for all algorithms.
        Y /= np.std(Y, axis=0, keepdims=True)  # Scale for numerical stability
        n_components = 10
        n_good_components = 4  # X has a rank of 4.

        assert Y.shape[1] > 1
        assert Y.shape[1] < X.shape[1]
        (
            sk_pls,
            sk_B,
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
        ) = self.fit_models(X=X, Y=Y, n_components=n_components)
        self.check_equality_properties(
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            X=X,
            atol=1e-1,
            rtol=1e-5,
            n_good_components=n_good_components,
        )
        self.check_orthogonality_properties(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-1,
            rtol=0,
            n_good_components=n_good_components,
        )
        # self.check_internal_matrices(atol=1e-1, rtol=1e-1)
        self.check_predictions(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            X=X,
            atol=1e-2,
            rtol=0,
            n_good_components=n_good_components,
        )  # PLS2 is not as numerically stable as PLS1.

    def test_early_stop_fitting_pls_2_m_eq_k(self):
        """
        The NumPy implementations will stop iterating through components if the residual comes close to 0.
        """
        vectors = np.array(
            [np.arange(2, 7, dtype=np.float64) ** (i + 1) for i in range(5)]
        )
        X = np.tile(vectors, reps=(5000, 3))
        X -= np.mean(X, axis=0)
        X /= np.std(X, axis=0)  # Let's also scale them for better numerical stability

        Y = np.hstack(
            [np.sum(X, axis=1, keepdims=True) ** (i + 1) for i in range(15)]
        )
        Y -= np.mean(Y, axis=0, keepdims=True)  # SkLearn's PLS implementation always centers its Y input. This ensures that the Y input is always centered for all algorithms.
        Y /= np.std(Y, axis=0, keepdims=True)  # Scale for numerical stability
        n_components = 10
        n_good_components = 4  # X has a rank of 4.

        assert Y.shape[1] > 1
        assert Y.shape[1] == X.shape[1]
        (
            sk_pls,
            sk_B,
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
        ) = self.fit_models(X=X, Y=Y, n_components=n_components)
        self.check_equality_properties(
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            X=X,
            atol=1e-1,
            rtol=1e-5,
            n_good_components=n_good_components,
        )
        self.check_orthogonality_properties(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-1,
            rtol=0,
            n_good_components=n_good_components,
        )
        # self.check_internal_matrices(atol=1e-1, rtol=1e-1)
        self.check_predictions(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            X=X,
            atol=1e-2,
            rtol=0,
            n_good_components=n_good_components,
        )  # PLS2 is not as numerically stable as PLS1.

    def test_early_stop_fitting_pls_2_m_greater_k(self):
        """
        The NumPy implementations will stop iterating through components if the residual comes close to 0.
        """
        vectors = np.array(
            [np.arange(2, 7, dtype=np.float64) ** (i + 1) for i in range(5)]
        )
        X = np.tile(vectors, reps=(5000, 3))
        X -= np.mean(X, axis=0)
        X /= np.std(X, axis=0)  # Let's also scale them for better numerical stability

        Y = np.hstack(
            [np.sum(X, axis=1, keepdims=True) ** (i + 1) for i in range(20)]
        )
        Y -= np.mean(Y, axis=0, keepdims=True)  # SkLearn's PLS implementation always centers its Y input. This ensures that the Y input is always centered for all algorithms.
        Y /= np.std(Y, axis=0, keepdims=True)  # Scale for numerical stability
        n_components = 10
        n_good_components = 4  # X has a rank of 4.

        assert Y.shape[1] > 1
        assert Y.shape[1] > X.shape[1]
        (
            sk_pls,
            sk_B,
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
        ) = self.fit_models(X=X, Y=Y, n_components=n_components)
        self.check_equality_properties(
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            X=X,
            atol=1e-1,
            rtol=1e-5,
            n_good_components=n_good_components,
        )
        self.check_orthogonality_properties(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-1,
            rtol=0,
            n_good_components=n_good_components,
        )
        # self.check_internal_matrices(atol=1e-1, rtol=1e-1)
        self.check_predictions(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            X=X,
            atol=1e-2,
            rtol=0,
            n_good_components=n_good_components,
        )  # PLS2 is not as numerically stable as PLS1.


if __name__ == "__main__":
    tc = TestClass()
    tc.test_pls_1()
    tc.test_pls_2_m_less_k()
    tc.test_pls_2_m_eq_k()
    tc.test_pls_2_m_greater_k()
    tc.test_early_stop_fitting_pls_1()  # Stop after 4 components. Here, own algorithms fails to stop early. Norm is constant at approx. 1e-14.
    tc.test_early_stop_fitting_pls_2_m_less_k()  # Stop after 4 components. Here, own algorithms fails to stop early. Norm explodes.
    tc.test_early_stop_fitting_pls_2_m_eq_k()  # Stop after 4 components
    tc.test_early_stop_fitting_pls_2_m_greater_k()  # Fix denne. Lykkes ikke med at skaffe early stopping
