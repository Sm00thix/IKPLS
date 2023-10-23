from sklearn.cross_decomposition import PLSRegression as SkPLS
from algorithms.jax_ikpls_alg_1 import PLS as JAX_Alg_1
from algorithms.jax_ikpls_alg_2 import PLS as JAX_Alg_2
from algorithms.numpy_ikpls import PLS as NpPLS

# import load_data
from . import load_data

import pytest
import numpy as np
import numpy.linalg as la
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

    def check_x_weights(
        self,
        sk_pls,
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

    def check_x_loadings(
        self,
        sk_pls,
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

    def check_y_loadings(
        self,
        sk_pls,
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

    def check_x_rotations(
        self,
        sk_pls,
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

    def check_x_scores(  # X scores - not computed by IKPLS Algorithm #2
        self,
        sk_pls,
        np_pls_alg_1,
        jax_pls_alg_1,
        atol,
        rtol,
        n_good_components=-1,
    ):
        if n_good_components == -1:
            n_good_components = np_pls_alg_1.A
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

    def check_regression_matrices(
        self,
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
        Y -= np.mean(
            Y, axis=0, keepdims=True
        )  # SkLearn's PLS implementation always centers its Y input. This ensures that the Y input is always centered for all algorithms.
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

        self.check_x_weights(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-8,
            rtol=1e-5,
        )

        self.check_x_loadings(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-8,
            rtol=1e-5,
        )

        self.check_y_loadings(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-8,
            rtol=1e-5,
        )

        self.check_x_rotations(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-8,
            rtol=1e-5,
        )

        self.check_x_scores(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            atol=1e-3,
            rtol=0,
        )

        self.check_regression_matrices(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-8,
            rtol=1e-5,
        )

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
        Y -= np.mean(
            Y, axis=0, keepdims=True
        )  # SkLearn's PLS implementation always centers its Y input. This ensures that the Y input is always centered for all algorithms.
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

        self.check_x_weights(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=2e-3,
            rtol=0,
        )

        self.check_x_loadings(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=2e-3,
            rtol=0,
        )

        self.check_y_loadings(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=0.14,
            rtol=0,
        )

        self.check_x_rotations(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=2e-3,
            rtol=0,
        )

        self.check_x_scores(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            atol=2e-3,
            rtol=0,
        )

        self.check_regression_matrices(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=0.06,
            rtol=0,
        )
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
        Y -= np.mean(
            Y, axis=0, keepdims=True
        )  # SkLearn's PLS implementation always centers its Y input. This ensures that the Y input is always centered for all algorithms.
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

        self.check_x_weights(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=2e-3,
            rtol=0,
        )

        self.check_x_loadings(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=2e-3,
            rtol=0,
        )

        self.check_y_loadings(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=0.23,
            rtol=0,
        )

        self.check_x_rotations(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=2e-3,
            rtol=0,
        )

        self.check_x_scores(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            atol=3e-4,
            rtol=0,
        )

        self.check_regression_matrices(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=0.11,
            rtol=0,
        )
        self.check_predictions(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            X=X,
            atol=2e-3,
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
        Y -= np.mean(
            Y, axis=0, keepdims=True
        )  # SkLearn's PLS implementation always centers its Y input. This ensures that the Y input is always centered for all algorithms.
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

        self.check_x_weights(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=2e-3,
            rtol=0,
        )

        self.check_x_loadings(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=2e-3,
            rtol=0,
        )

        self.check_y_loadings(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=0.17,
            rtol=0,
        )

        self.check_x_rotations(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=2e-3,
            rtol=0,
        )

        self.check_x_scores(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            atol=3e-4,
            rtol=0,
        )

        self.check_regression_matrices(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=0.09,
            rtol=0,
        )
        self.check_predictions(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            X=X,
            atol=2e-3,
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
        Y -= np.mean(
            Y, axis=0, keepdims=True
        )  # SkLearn's PLS implementation always centers its Y input. This ensures that the Y input is always centered for all algorithms.
        Y /= np.std(Y, axis=0, keepdims=True)  # Scale for numerical stability
        n_components = 10
        n_good_components = 4  # X has a rank of 4.
        assert la.matrix_rank(X) == 4
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

        self.check_x_weights(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=5e-4,
            rtol=2e-3,
            n_good_components=n_good_components,
        )

        self.check_x_loadings(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-8,
            rtol=1e-5,
            n_good_components=n_good_components,
        )

        self.check_y_loadings(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-8,
            rtol=1e-5,
            n_good_components=n_good_components,
        )

        self.check_x_rotations(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=5e-4,
            rtol=0,
            n_good_components=n_good_components,
        )

        self.check_x_scores(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            atol=2e-4,
            rtol=0,
            n_good_components=n_good_components,
        )

        self.check_regression_matrices(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-8,
            rtol=1e-5,
            n_good_components=n_good_components,
        )
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
        )

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
        Y -= np.mean(
            Y, axis=0, keepdims=True
        )  # SkLearn's PLS implementation always centers its Y input. This ensures that the Y input is always centered for all algorithms.
        Y /= np.std(Y, axis=0, keepdims=True)  # Scale for numerical stability
        n_components = 10
        n_good_components = 4  # X has a rank of 4.
        assert la.matrix_rank(X) == 4
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

        self.check_x_weights(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-8,
            rtol=1e-5,
            n_good_components=n_good_components,
        )

        self.check_x_loadings(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-8,
            rtol=1e-5,
            n_good_components=n_good_components,
        )

        self.check_y_loadings(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-8,
            rtol=1e-5,
            n_good_components=n_good_components,
        )

        self.check_x_rotations(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=0.26,
            rtol=0,
            n_good_components=n_good_components,
        )

        self.check_x_scores(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            atol=2e-4,
            rtol=0,
            n_good_components=n_good_components,
        )

        self.check_regression_matrices(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=0.33,
            rtol=0,
            n_good_components=n_good_components,
        )
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
        )

    def test_early_stop_fitting_pls_2_m_eq_k(self):
        """
        The NumPy implementations will stop iterating through components if the residual comes close to 0.
        """
        vectors = np.array(
            [np.arange(2, 7, dtype=np.float64) ** (i + 1) for i in range(5)]
        )
        X = np.tile(vectors, reps=(5000, 2))
        X -= np.mean(X, axis=0)
        X /= np.std(X, axis=0)  # Let's also scale them for better numerical stability

        Y = np.hstack([np.sum(X, axis=1, keepdims=True) ** (i + 1) for i in range(10)])
        Y -= np.mean(
            Y, axis=0, keepdims=True
        )  # SkLearn's PLS implementation always centers its Y input. This ensures that the Y input is always centered for all algorithms.
        Y /= np.std(Y, axis=0, keepdims=True)  # Scale for numerical stability
        n_components = 10
        n_good_components = 4  # X has a rank of 4.
        assert la.matrix_rank(X) == 4
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

        self.check_x_weights(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-8,
            rtol=1e-5,
            n_good_components=n_good_components,
        )

        self.check_x_loadings(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-8,
            rtol=1e-5,
            n_good_components=n_good_components,
        )

        self.check_y_loadings(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-8,
            rtol=1e-5,
            n_good_components=n_good_components,
        )

        self.check_x_rotations(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=0.33,
            rtol=0,
            n_good_components=n_good_components,
        )

        self.check_x_scores(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            atol=2e-4,
            rtol=0,
            n_good_components=n_good_components,
        )

        self.check_regression_matrices(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=0.3,
            rtol=0,
            n_good_components=n_good_components,
        )
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
        )

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

        Y = np.hstack([np.sum(X, axis=1, keepdims=True) ** (i + 1) for i in range(20)])
        Y -= np.mean(
            Y, axis=0, keepdims=True
        )  # SkLearn's PLS implementation always centers its Y input. This ensures that the Y input is always centered for all algorithms.
        Y /= np.std(Y, axis=0, keepdims=True)  # Scale for numerical stability
        n_components = 10
        n_good_components = 4  # X has a rank of 4.
        assert la.matrix_rank(X) == 4
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

        self.check_x_weights(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-8,
            rtol=1e-5,
            n_good_components=n_good_components,
        )

        self.check_x_loadings(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-8,
            rtol=1e-5,
            n_good_components=n_good_components,
        )

        self.check_y_loadings(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-8,
            rtol=1e-5,
            n_good_components=n_good_components,
        )

        self.check_x_rotations(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=0.38,
            rtol=0.69,
            n_good_components=n_good_components,
        )

        self.check_x_scores(
            sk_pls=sk_pls,
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            atol=2e-4,
            rtol=0,
            n_good_components=n_good_components,
        )

        self.check_regression_matrices(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=0.22,
            rtol=0,
            n_good_components=n_good_components,
        )
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
        )

    def test_sanity_check_pls_regression(
        self,
    ):  # Taken from SkLearn's test suite and modified to include own algorithms.
        from sklearn.datasets import load_linnerud

        d = load_linnerud()
        X = d.data  # Shape = (20,3)
        X -= np.mean(X, axis=0)
        X /= np.std(X, axis=0)
        Y = d.target  # Shape = (20,3)
        Y -= np.mean(Y, axis=0)
        Y /= np.std(Y, axis=0)
        n_components = X.shape[1]  # 3
        (
            sk_pls,
            sk_B,
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
        ) = self.fit_models(X=X, Y=Y, n_components=n_components)

        # Check for orthogonal X weights.
        self.assert_matrix_orthogonal(sk_pls.x_weights_, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(np_pls_alg_1.W, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(np_pls_alg_2.W, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(jax_pls_alg_1.W, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(jax_pls_alg_2.W, atol=1e-8, rtol=0)

        # Check for orthogonal X scores - not computed by Algorithm #2.
        self.assert_matrix_orthogonal(sk_pls.x_scores_, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(np_pls_alg_1.T, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(jax_pls_alg_1.T, atol=1e-8, rtol=0)

        # Check invariants.
        self.check_equality_properties(
            np_pls_alg_1=np_pls_alg_1,
            jax_pls_alg_1=jax_pls_alg_1,
            X=X,
            atol=1e-8,
            rtol=1e-5,
        )

        expected_x_weights = np.array(
            [
                [-0.61330704, -0.00443647, 0.78983213],
                [-0.74697144, -0.32172099, -0.58183269],
                [-0.25668686, 0.94682413, -0.19399983],
            ]
        )

        expected_x_loadings = np.array(
            [
                [-0.61470416, -0.24574278, 0.78983213],
                [-0.65625755, -0.14396183, -0.58183269],
                [-0.51733059, 1.00609417, -0.19399983],
            ]
        )

        expected_y_loadings = np.array(
            [
                [+0.32456184, 0.29892183, 0.20316322],
                [+0.42439636, 0.61970543, 0.19320542],
                [-0.13143144, -0.26348971, -0.17092916],
            ]
        )

        # Check for expected X weights
        assert_allclose(
            np.abs(sk_pls.x_weights_), np.abs(expected_x_weights), atol=1e-8, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_1.W), np.abs(expected_x_weights), atol=2e-6, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_2.W), np.abs(expected_x_weights), atol=2e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_1.W), np.abs(expected_x_weights), atol=2e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_2.W), np.abs(expected_x_weights), atol=2e-6, rtol=0
        )

        # Check for expected X loadings
        assert_allclose(
            np.abs(sk_pls.x_loadings_), np.abs(expected_x_loadings), atol=1e-8, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_1.P), np.abs(expected_x_loadings), atol=2e-6, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_2.P), np.abs(expected_x_loadings), atol=2e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_1.P), np.abs(expected_x_loadings), atol=2e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_2.P), np.abs(expected_x_loadings), atol=2e-6, rtol=0
        )

        # Check for expected Y loadings
        assert_allclose(
            np.abs(sk_pls.y_loadings_), np.abs(expected_y_loadings), atol=1e-8, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_1.Q), np.abs(expected_y_loadings), atol=2e-6, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_2.Q), np.abs(expected_y_loadings), atol=2e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_1.Q), np.abs(expected_y_loadings), atol=2e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_2.Q), np.abs(expected_y_loadings), atol=2e-6, rtol=0
        )

        # Check that sign flip is consistent and exact across loadings and weights
        sk_x_loadings_sign_flip = np.sign(sk_pls.x_loadings_ / expected_x_loadings)
        sk_x_weights_sign_flip = np.sign(sk_pls.x_weights_ / expected_x_weights)
        sk_y_loadings_sign_flip = np.sign(sk_pls.y_loadings_ / expected_y_loadings)
        assert np.allclose(
            sk_x_loadings_sign_flip, sk_x_weights_sign_flip, atol=0, rtol=0
        )
        assert np.allclose(
            sk_x_loadings_sign_flip, sk_y_loadings_sign_flip, atol=0, rtol=0
        )

        np_alg_1_x_loadings_sign_flip = np.sign(np_pls_alg_1.P / expected_x_loadings)
        np_alg_1_x_weights_sign_flip = np.sign(np_pls_alg_1.W / expected_x_weights)
        np_alg_1_y_loadings_sign_flip = np.sign(np_pls_alg_1.Q / expected_y_loadings)
        assert np.allclose(
            np_alg_1_x_loadings_sign_flip, np_alg_1_x_weights_sign_flip, atol=0, rtol=0
        )
        assert np.allclose(
            np_alg_1_x_loadings_sign_flip, np_alg_1_y_loadings_sign_flip, atol=0, rtol=0
        )

        np_alg_2_x_loadings_sign_flip = np.sign(np_pls_alg_2.P / expected_x_loadings)
        np_alg_2_x_weights_sign_flip = np.sign(np_pls_alg_2.W / expected_x_weights)
        np_alg_2_y_loadings_sign_flip = np.sign(np_pls_alg_2.Q / expected_y_loadings)
        assert np.allclose(
            np_alg_2_x_loadings_sign_flip, np_alg_2_x_weights_sign_flip, atol=0, rtol=0
        )
        assert np.allclose(
            np_alg_2_x_loadings_sign_flip, np_alg_2_y_loadings_sign_flip, atol=0, rtol=0
        )

        jax_alg_1_x_loadings_sign_flip = np.sign(jax_pls_alg_1.P / expected_x_loadings)
        jax_alg_1_x_weights_sign_flip = np.sign(jax_pls_alg_1.W / expected_x_weights)
        jax_alg_1_y_loadings_sign_flip = np.sign(jax_pls_alg_1.Q / expected_y_loadings)
        assert np.allclose(
            jax_alg_1_x_loadings_sign_flip,
            jax_alg_1_x_weights_sign_flip,
            atol=0,
            rtol=0,
        )
        assert np.allclose(
            jax_alg_1_x_loadings_sign_flip,
            jax_alg_1_y_loadings_sign_flip,
            atol=0,
            rtol=0,
        )

        jax_alg_2_x_loadings_sign_flip = np.sign(jax_pls_alg_2.P / expected_x_loadings)
        jax_alg_2_x_weights_sign_flip = np.sign(jax_pls_alg_2.W / expected_x_weights)
        jax_alg_2_y_loadings_sign_flip = np.sign(jax_pls_alg_2.Q / expected_y_loadings)
        assert np.allclose(
            jax_alg_2_x_loadings_sign_flip,
            jax_alg_2_x_weights_sign_flip,
            atol=0,
            rtol=0,
        )
        assert np.allclose(
            jax_alg_2_x_loadings_sign_flip,
            jax_alg_2_y_loadings_sign_flip,
            atol=0,
            rtol=0,
        )

    def test_sanity_check_pls_regression_constant_column_Y(
        self,
    ):  # Taken from SkLearn's test suite and modified to include own algorithms.
        from sklearn.datasets import load_linnerud

        d = load_linnerud()
        X = d.data  # Shape = (20,3)
        X -= np.mean(X, axis=0)
        X /= np.std(X, axis=0)
        Y = d.target  # Shape = (20,3)
        Y[:, 0] = 1  # Set the first column to a constant
        Y -= np.mean(Y, axis=0)
        Y[:, 1:] /= np.std(
            Y[:, 1:], axis=0
        )  # The standard deviation of the first column is 0 by construction. Do not attempt division on that column.
        n_components = X.shape[1]  # 3
        (
            sk_pls,
            sk_B,
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
        ) = self.fit_models(X=X, Y=Y, n_components=n_components)

        expected_x_weights = np.array(
            [
                [-0.6273573, 0.007081799, 0.7786994],
                [-0.7493417, -0.277612681, -0.6011807],
                [-0.2119194, 0.960666981, -0.1794690],
            ]
        )

        expected_x_loadings = np.array(
            [
                [-0.6273512, -0.22464538, 0.7786994],
                [-0.6643156, -0.09871193, -0.6011807],
                [-0.5125877, 1.01407380, -0.1794690],
            ]
        )

        expected_y_loadings = np.array(
            [
                [0.0000000, 0.0000000, 0.0000000],
                [0.4357300, 0.5828479, 0.2174802],
                [-0.1353739, -0.2486423, -0.1810386],
            ]
        )

        # Check for expected X weights
        assert_allclose(
            np.abs(sk_pls.x_weights_), np.abs(expected_x_weights), atol=5e-8, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_1.W), np.abs(expected_x_weights), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_2.W), np.abs(expected_x_weights), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_1.W), np.abs(expected_x_weights), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_2.W), np.abs(expected_x_weights), atol=3e-6, rtol=0
        )

        # Check for expected X loadings
        assert_allclose(
            np.abs(sk_pls.x_loadings_), np.abs(expected_x_loadings), atol=5e-8, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_1.P), np.abs(expected_x_loadings), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_2.P), np.abs(expected_x_loadings), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_1.P), np.abs(expected_x_loadings), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_2.P), np.abs(expected_x_loadings), atol=3e-6, rtol=0
        )

        # Check for expected Y loadings
        assert_allclose(
            np.abs(sk_pls.y_loadings_), np.abs(expected_y_loadings), atol=5e-8, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_1.Q), np.abs(expected_y_loadings), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(np_pls_alg_2.Q), np.abs(expected_y_loadings), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_1.Q), np.abs(expected_y_loadings), atol=3e-6, rtol=0
        )
        assert_allclose(
            np.abs(jax_pls_alg_2.Q), np.abs(expected_y_loadings), atol=3e-6, rtol=0
        )

        # Check for orthogonal X weights.
        self.assert_matrix_orthogonal(sk_pls.x_weights_, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(np_pls_alg_1.W, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(np_pls_alg_2.W, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(jax_pls_alg_1.W, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(jax_pls_alg_2.W, atol=1e-8, rtol=0)

        # Check for orthogonal X scores - not computed by Algorithm #2.
        self.assert_matrix_orthogonal(sk_pls.x_scores_, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(np_pls_alg_1.T, atol=1e-8, rtol=0)
        self.assert_matrix_orthogonal(jax_pls_alg_1.T, atol=1e-8, rtol=0)

        # Check that sign flip is consistent and exact across loadings and weights. Ignore the first column of Y which will be a column of zeros (due to mean centering of its constant value).
        sk_x_loadings_sign_flip = np.sign(sk_pls.x_loadings_ / expected_x_loadings)
        sk_x_weights_sign_flip = np.sign(sk_pls.x_weights_ / expected_x_weights)
        sk_y_loadings_sign_flip = np.sign(
            sk_pls.y_loadings_[1:] / expected_y_loadings[1:]
        )
        assert np.allclose(
            sk_x_loadings_sign_flip, sk_x_weights_sign_flip, atol=0, rtol=0
        )
        assert np.allclose(
            sk_x_loadings_sign_flip[1:], sk_y_loadings_sign_flip, atol=0, rtol=0
        )

        np_alg_1_x_loadings_sign_flip = np.sign(np_pls_alg_1.P / expected_x_loadings)
        np_alg_1_x_weights_sign_flip = np.sign(np_pls_alg_1.W / expected_x_weights)
        np_alg_1_y_loadings_sign_flip = np.sign(
            np_pls_alg_1.Q[1:] / expected_y_loadings[1:]
        )
        assert np.allclose(
            np_alg_1_x_loadings_sign_flip, np_alg_1_x_weights_sign_flip, atol=0, rtol=0
        )
        assert np.allclose(
            np_alg_1_x_loadings_sign_flip[1:],
            np_alg_1_y_loadings_sign_flip,
            atol=0,
            rtol=0,
        )

        np_alg_2_x_loadings_sign_flip = np.sign(np_pls_alg_2.P / expected_x_loadings)
        np_alg_2_x_weights_sign_flip = np.sign(np_pls_alg_2.W / expected_x_weights)
        np_alg_2_y_loadings_sign_flip = np.sign(
            np_pls_alg_2.Q[1:] / expected_y_loadings[1:]
        )
        assert np.allclose(
            np_alg_2_x_loadings_sign_flip, np_alg_2_x_weights_sign_flip, atol=0, rtol=0
        )
        assert np.allclose(
            np_alg_2_x_loadings_sign_flip[1:],
            np_alg_2_y_loadings_sign_flip,
            atol=0,
            rtol=0,
        )

        jax_alg_1_x_loadings_sign_flip = np.sign(jax_pls_alg_1.P / expected_x_loadings)
        jax_alg_1_x_weights_sign_flip = np.sign(jax_pls_alg_1.W / expected_x_weights)
        jax_alg_1_y_loadings_sign_flip = np.sign(
            jax_pls_alg_1.Q[1:] / expected_y_loadings[1:]
        )
        assert np.allclose(
            jax_alg_1_x_loadings_sign_flip,
            jax_alg_1_x_weights_sign_flip,
            atol=0,
            rtol=0,
        )
        assert np.allclose(
            jax_alg_1_x_loadings_sign_flip,
            jax_alg_1_y_loadings_sign_flip[1:],
            atol=0,
            rtol=0,
        )

        jax_alg_2_x_loadings_sign_flip = np.sign(jax_pls_alg_2.P / expected_x_loadings)
        jax_alg_2_x_weights_sign_flip = np.sign(jax_pls_alg_2.W / expected_x_weights)
        jax_alg_2_y_loadings_sign_flip = np.sign(
            jax_pls_alg_2.Q[1:] / expected_y_loadings[1:]
        )
        assert np.allclose(
            jax_alg_2_x_loadings_sign_flip,
            jax_alg_2_x_weights_sign_flip,
            atol=0,
            rtol=0,
        )
        assert np.allclose(
            jax_alg_2_x_loadings_sign_flip,
            jax_alg_2_y_loadings_sign_flip[1:],
            atol=0,
            rtol=0,
        )

# if __name__ == "__main__":
#     tc = TestClass()
#     tc.test_pls_1()
#     tc.test_pls_2_m_less_k()
#     tc.test_pls_2_m_eq_k()
#     tc.test_pls_2_m_greater_k()
#     tc.test_early_stop_fitting_pls_1()  # Stop after 4 components. Here, own algorithms fails to stop early. Norm is constant at approx. 1e-14.
#     tc.test_early_stop_fitting_pls_2_m_less_k()  # Stop after 4 components. Here, own algorithms fails to stop early. Norm explodes.
#     tc.test_early_stop_fitting_pls_2_m_eq_k()  # Stop after 4 components
#     tc.test_early_stop_fitting_pls_2_m_greater_k()  # Fix denne. Lykkes ikke med at skaffe early stopping
#     tc.test_sanity_check_pls_regression()
#     tc.test_sanity_check_pls_regression_constant_column_Y()

# TODO: Make individual tests for each of the internal matrices. (DONE)
# TODO: Add the remaining tests from SkLearn's test suite.
# TODO: Check that results are consistent across CPU and GPU implementations.
# TODO: Check that cross validation results match those achieved by SkLearn.
# TODO: Implement general purpose cross validation for GPU algorithms.

# TODO: Use pytest.warns as context manager.
# TODO: Implement constant Y test from SkLearn's test suite.