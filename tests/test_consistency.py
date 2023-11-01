from sklearn.cross_decomposition import PLSRegression as SkPLS
from algorithms.jax_ikpls_alg_1 import PLS as JAX_Alg_1
from algorithms.jax_ikpls_alg_2 import PLS as JAX_Alg_2
from algorithms.numpy_ikpls import PLS as NpPLS

# import load_data

from . import load_data

import pytest
import numpy as np
import numpy.typing as npt
import jax
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
        x_mean = X.mean(axis=0)
        X -= x_mean
        y_mean = Y.mean(axis=0)
        Y -= y_mean
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std
        jnp_X = jnp.array(X)
        jnp_Y = jnp.array(Y)
        sk_pls = SkPLS(
            n_components=n_components, scale=False, copy=True
        )  # Do not rescale again.
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
        diff = (
            np_pls_alg_1.predict(X)[:n_good_components]
            - sk_all_preds[:n_good_components]
        )
        max_atol = np.amax(diff)
        max_rtol = np.amax(diff / np.abs(sk_all_preds[:n_good_components]))
        print(f"Max atol: {max_atol}\nMax rtol:{max_rtol}")
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

    def check_cpu_gpu_equality(
        self,
        np_pls_alg_1,
        np_pls_alg_2,
        jax_pls_alg_1,
        jax_pls_alg_2,
        n_good_components=-1,
    ):
        if n_good_components == -1:
            n_good_components = np_pls_alg_1.A

        atol = 0
        rtol = 1e-4
        # Regression matrices
        assert_allclose(
            np_pls_alg_1.B[:n_good_components],
            np.array(jax_pls_alg_1.B[:n_good_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np_pls_alg_2.B[:n_good_components],
            np.array(jax_pls_alg_2.B[:n_good_components]),
            atol=atol,
            rtol=rtol,
        )

        # X weights
        assert_allclose(
            np.abs(np_pls_alg_1.W[..., :n_good_components]),
            np.abs(np.array(jax_pls_alg_1.W[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_2.W[..., :n_good_components]),
            np.abs(np.array(jax_pls_alg_2.W[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )

        # X loadings
        assert_allclose(
            np.abs(np_pls_alg_1.P[..., :n_good_components]),
            np.abs(np.array(jax_pls_alg_1.P[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_2.P[..., :n_good_components]),
            np.abs(np.array(jax_pls_alg_2.P[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )

        # Y loadings
        assert_allclose(
            np.abs(np_pls_alg_1.Q[..., :n_good_components]),
            np.abs(np.array(jax_pls_alg_1.Q[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_2.Q[..., :n_good_components]),
            np.abs(np.array(jax_pls_alg_2.Q[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )

        # X rotations
        assert_allclose(
            np.abs(np_pls_alg_1.R[..., :n_good_components]),
            np.abs(np.array(jax_pls_alg_1.R[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np_pls_alg_2.R[..., :n_good_components]),
            np.abs(np.array(jax_pls_alg_2.R[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )

        # X scores - only computed by Algorithm #1
        assert_allclose(
            np.abs(np_pls_alg_1.T[..., :n_good_components]),
            np.abs(np.array(jax_pls_alg_1.T[..., :n_good_components])),
            atol=atol,
            rtol=rtol,
        )

    def test_pls_1(self):
        """
        Test PLS1.
        """
        X = self.load_X()
        Y = self.load_Y(["Protein"])
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

        self.check_cpu_gpu_equality(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
        )

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
            rtol=1e-5,
        )  # PLS1 is very numerically stable for protein.

    def test_pls_2_m_less_k(self):
        """
        Test PLS2 where the number of targets is less than the number of features (M < K).
        """
        X = self.load_X()
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

        self.check_cpu_gpu_equality(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
        )

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

        self.check_cpu_gpu_equality(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
        )

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

        self.check_regression_matrices(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-8,
            rtol=0.1,
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

        self.check_cpu_gpu_equality(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
        )

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

        self.check_regression_matrices(
            sk_B=sk_B,
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
            atol=1e-8,
            rtol=2e-2,
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

    def test_sanity_check_pls_regression(
        self,
    ):  # Taken from SkLearn's test suite and modified to include own algorithms.
        from sklearn.datasets import load_linnerud

        d = load_linnerud()
        X = d.data  # Shape = (20,3)
        Y = d.target  # Shape = (20,3)
        n_components = X.shape[1]  # 3
        (
            sk_pls,
            sk_B,
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
        ) = self.fit_models(X=X, Y=Y, n_components=n_components)

        self.check_cpu_gpu_equality(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
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
        assert_allclose(sk_x_loadings_sign_flip, sk_x_weights_sign_flip, atol=0, rtol=0)
        assert_allclose(
            sk_x_loadings_sign_flip, sk_y_loadings_sign_flip, atol=0, rtol=0
        )

        np_alg_1_x_loadings_sign_flip = np.sign(np_pls_alg_1.P / expected_x_loadings)
        np_alg_1_x_weights_sign_flip = np.sign(np_pls_alg_1.W / expected_x_weights)
        np_alg_1_y_loadings_sign_flip = np.sign(np_pls_alg_1.Q / expected_y_loadings)
        assert_allclose(
            np_alg_1_x_loadings_sign_flip, np_alg_1_x_weights_sign_flip, atol=0, rtol=0
        )
        assert_allclose(
            np_alg_1_x_loadings_sign_flip, np_alg_1_y_loadings_sign_flip, atol=0, rtol=0
        )

        np_alg_2_x_loadings_sign_flip = np.sign(np_pls_alg_2.P / expected_x_loadings)
        np_alg_2_x_weights_sign_flip = np.sign(np_pls_alg_2.W / expected_x_weights)
        np_alg_2_y_loadings_sign_flip = np.sign(np_pls_alg_2.Q / expected_y_loadings)
        assert_allclose(
            np_alg_2_x_loadings_sign_flip, np_alg_2_x_weights_sign_flip, atol=0, rtol=0
        )
        assert_allclose(
            np_alg_2_x_loadings_sign_flip, np_alg_2_y_loadings_sign_flip, atol=0, rtol=0
        )

        jax_alg_1_x_loadings_sign_flip = np.sign(jax_pls_alg_1.P / expected_x_loadings)
        jax_alg_1_x_weights_sign_flip = np.sign(jax_pls_alg_1.W / expected_x_weights)
        jax_alg_1_y_loadings_sign_flip = np.sign(jax_pls_alg_1.Q / expected_y_loadings)
        assert_allclose(
            jax_alg_1_x_loadings_sign_flip,
            jax_alg_1_x_weights_sign_flip,
            atol=0,
            rtol=0,
        )
        assert_allclose(
            jax_alg_1_x_loadings_sign_flip,
            jax_alg_1_y_loadings_sign_flip,
            atol=0,
            rtol=0,
        )

        jax_alg_2_x_loadings_sign_flip = np.sign(jax_pls_alg_2.P / expected_x_loadings)
        jax_alg_2_x_weights_sign_flip = np.sign(jax_pls_alg_2.W / expected_x_weights)
        jax_alg_2_y_loadings_sign_flip = np.sign(jax_pls_alg_2.Q / expected_y_loadings)
        assert_allclose(
            jax_alg_2_x_loadings_sign_flip,
            jax_alg_2_x_weights_sign_flip,
            atol=0,
            rtol=0,
        )
        assert_allclose(
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
        Y = d.target  # Shape = (20,3)
        Y[:, 0] = 1  # Set the first column to a constant
        n_components = X.shape[1]  # 3
        (
            sk_pls,
            sk_B,
            np_pls_alg_1,
            np_pls_alg_2,
            jax_pls_alg_1,
            jax_pls_alg_2,
        ) = self.fit_models(X=X, Y=Y, n_components=n_components)

        self.check_cpu_gpu_equality(
            np_pls_alg_1=np_pls_alg_1,
            np_pls_alg_2=np_pls_alg_2,
            jax_pls_alg_1=jax_pls_alg_1,
            jax_pls_alg_2=jax_pls_alg_2,
        )

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
        assert_allclose(sk_x_loadings_sign_flip, sk_x_weights_sign_flip, atol=0, rtol=0)
        assert_allclose(
            sk_x_loadings_sign_flip[1:], sk_y_loadings_sign_flip, atol=0, rtol=0
        )

        np_alg_1_x_loadings_sign_flip = np.sign(np_pls_alg_1.P / expected_x_loadings)
        np_alg_1_x_weights_sign_flip = np.sign(np_pls_alg_1.W / expected_x_weights)
        np_alg_1_y_loadings_sign_flip = np.sign(
            np_pls_alg_1.Q[1:] / expected_y_loadings[1:]
        )
        assert_allclose(
            np_alg_1_x_loadings_sign_flip, np_alg_1_x_weights_sign_flip, atol=0, rtol=0
        )
        assert_allclose(
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
        assert_allclose(
            np_alg_2_x_loadings_sign_flip, np_alg_2_x_weights_sign_flip, atol=0, rtol=0
        )
        assert_allclose(
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
        assert_allclose(
            jax_alg_1_x_loadings_sign_flip,
            jax_alg_1_x_weights_sign_flip,
            atol=0,
            rtol=0,
        )
        assert_allclose(
            jax_alg_1_x_loadings_sign_flip[1:],
            jax_alg_1_y_loadings_sign_flip,
            atol=0,
            rtol=0,
        )

        jax_alg_2_x_loadings_sign_flip = np.sign(jax_pls_alg_2.P / expected_x_loadings)
        jax_alg_2_x_weights_sign_flip = np.sign(jax_pls_alg_2.W / expected_x_weights)
        jax_alg_2_y_loadings_sign_flip = np.sign(
            jax_pls_alg_2.Q[1:] / expected_y_loadings[1:]
        )
        assert_allclose(
            jax_alg_2_x_loadings_sign_flip,
            jax_alg_2_x_weights_sign_flip,
            atol=0,
            rtol=0,
        )
        assert_allclose(
            jax_alg_2_x_loadings_sign_flip[1:],
            jax_alg_2_y_loadings_sign_flip,
            atol=0,
            rtol=0,
        )

    def check_pls_constant_y(
        self, X, Y
    ):  # Taken from SkLearn's test suite and modified to include own algorithms.
        ## Taken from self.fit_models() to check each individual algorithm for early stopping.
        """Checks warning when y is constant."""
        x_mean = X.mean(axis=0)
        X -= x_mean
        y_mean = Y.mean(axis=0)
        Y -= y_mean
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std
        jnp_X = jnp.array(X)
        jnp_Y = jnp.array(Y)
        n_components = 2
        sk_pls = SkPLS(n_components=n_components, scale=False)  # Do not rescale again.
        np_pls_alg_1 = NpPLS(algorithm=1)
        np_pls_alg_2 = NpPLS(algorithm=2)
        jax_pls_alg_1 = JAX_Alg_1()
        jax_pls_alg_2 = JAX_Alg_2()

        sk_msg = "Y residual is constant at iteration"
        with pytest.warns(UserWarning, match=sk_msg):
            sk_pls.fit(X=X, Y=Y)
            assert_allclose(sk_pls.x_rotations_, 0)

        msg = "Weight is close to zero."
        with pytest.warns(UserWarning, match=msg):
            np_pls_alg_1.fit(X=X, Y=Y, A=n_components)
            assert_allclose(np_pls_alg_1.R, 0)
        with pytest.warns(UserWarning, match=msg):
            np_pls_alg_2.fit(X=X, Y=Y, A=n_components)
            assert_allclose(np_pls_alg_2.R, 0)
        with pytest.warns(UserWarning, match=msg):
            jax_pls_alg_1.fit(X=jnp_X, Y=jnp_Y, A=n_components)
        with pytest.warns(UserWarning, match=msg):
            jax_pls_alg_2.fit(X=jnp_X, Y=jnp_Y, A=n_components)

    def test_pls_1_constant_y(self):
        rng = np.random.RandomState(42)
        X = rng.rand(100, 3)
        Y = np.zeros(shape=(100, 1))
        assert Y.shape[1] == 1
        self.check_pls_constant_y(X, Y)

    def test_pls_2_m_less_k_constant_y(self):
        rng = np.random.RandomState(42)
        X = rng.rand(100, 3)
        Y = np.zeros(shape=(100, 2))
        assert Y.shape[1] > 1
        assert Y.shape[1] < X.shape[1]
        self.check_pls_constant_y(X, Y)

    def test_pls_2_m_eq_k_constant_y(self):
        rng = np.random.RandomState(42)
        X = rng.rand(100, 3)
        Y = np.zeros(shape=(100, 3))
        assert Y.shape[1] > 1
        assert Y.shape[1] == X.shape[1]
        self.check_pls_constant_y(X, Y)

    def test_pls_2_m_greater_k_constant_y(self):
        rng = np.random.RandomState(42)
        X = rng.rand(100, 3)
        Y = np.zeros(shape=(100, 4))
        assert Y.shape[1] > 1
        assert Y.shape[1] > X.shape[1]
        self.check_pls_constant_y(X, Y)

    def check_gradient_pls(
        self,
        X,
        Y,
        num_components,
        filter_size,
        val_atol,
        val_rtol,
        grad_atol,
        grad_rtol,
    ):
        """
        Tests that the gradient propagation works for PLS. The input spectra are convolved with a filter. We compute the gradients of RMSE loss w.r.t. the parameters of the preprocessing filter.
        """
        # Taken from self.fit_models() to check each individual algorithm for early stopping.
        x_mean = X.mean(axis=0)
        X -= x_mean
        y_mean = Y.mean(axis=0)
        Y -= y_mean
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std

        jnp_X = jnp.array(X, dtype=jnp.float64)
        jnp_Y = jnp.array(Y, dtype=jnp.float64)

        pls_alg_1 = JAX_Alg_1()
        pls_alg_2 = JAX_Alg_2()

        uniform_filter = jnp.ones(filter_size) / filter_size

        # Preprocessing convolution filter for which we will obtain the gradients.
        @jax.jit
        def apply_1d_convolution(matrix, conv_filter):
            convolved_rows = jax.vmap(
                lambda row: jnp.convolve(row, conv_filter, "valid")
            )(matrix)
            return convolved_rows

        # Loss function which we want to minimize.
        @jax.jit
        def rmse(y_true, y_pred):
            e = y_true - y_pred
            se = e**2
            mse = jnp.mean(se)
            rmse = jnp.sqrt(mse)
            return rmse

        # Function to differentiate.
        def preprocess_fit_rmse(X, Y, pls_alg, A=None):
            @jax.jit
            def helper(conv_filter):
                filtered_X = apply_1d_convolution(X, conv_filter)
                matrices = pls_alg.stateless_fit(filtered_X, Y, A)
                B = matrices[0]
                Y_pred = pls_alg.stateless_predict(filtered_X, B, A)
                rmse_loss = rmse(Y, Y_pred)
                return jnp.squeeze(rmse_loss)

            return helper

        # Compute values and gradients for algorithm #1
        grad_fun = jax.value_and_grad(
            preprocess_fit_rmse(jnp_X, jnp_Y, pls_alg_1, num_components), argnums=0
        )
        output_val_alg_1, grad_alg_1 = grad_fun(uniform_filter)

        # Compute the gradient and output value for a single number of components
        grad_fun = jax.value_and_grad(
            preprocess_fit_rmse(jnp_X, jnp_Y, pls_alg_2, num_components), argnums=0
        )
        output_val_alg_2, grad_alg_2 = grad_fun(uniform_filter)

        # Check that outputs and gradients of algorithm 1 and 2 are identical
        assert_allclose(
            np.array(grad_alg_1), np.array(grad_alg_2), atol=grad_atol, rtol=grad_rtol
        )
        assert_allclose(
            np.array(output_val_alg_1),
            np.array(output_val_alg_2),
            atol=val_atol,
            rtol=val_rtol,
        )

        # Check that no NaNs are encountered in the gradient
        assert jnp.all(~jnp.isnan(grad_alg_1))
        assert jnp.all(~jnp.isnan(grad_alg_2))

        # Check that the gradient actually flows all the way through
        zeros = jnp.zeros(filter_size, dtype=jnp.float64)
        assert jnp.any(jnp.not_equal(grad_alg_1, zeros))
        assert jnp.any(jnp.not_equal(grad_alg_2, zeros))

    def test_gradient_pls_1(self):
        X = self.load_X()
        Y = self.load_Y(["Protein"])
        num_components = 25
        filter_size = 7
        assert Y.shape[1] == 1
        self.check_gradient_pls(
            X=X,
            Y=Y,
            num_components=num_components,
            filter_size=filter_size,
            val_atol=0,
            val_rtol=1e-5,
            grad_atol=0,
            grad_rtol=1e-5,
        )

    def test_gradient_pls_2_m_less_k(self):
        X = self.load_X()
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
        num_components = 25
        filter_size = 7
        assert Y.shape[1] > 1
        assert (
            Y.shape[1] < X.shape[1] - filter_size + 1
        )  # The output of the convolution preprocessing is what is actually fed as input to the PLS algorithms.
        self.check_gradient_pls(
            X=X,
            Y=Y,
            num_components=num_components,
            filter_size=filter_size,
            val_atol=0,
            val_rtol=1e-5,
            grad_atol=0,
            grad_rtol=1e-5,
        )

    def test_gradient_pls_2_m_eq_k(self):
        X = self.load_X()
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
        filter_size = 7
        X = X[..., : 10 + filter_size - 1]
        num_components = 10
        assert Y.shape[1] > 1
        assert (
            Y.shape[1] == X.shape[1] - filter_size + 1
        )  # The output of the convolution preprocessing is what is actually fed as input to the PLS algorithms.
        self.check_gradient_pls(
            X=X,
            Y=Y,
            num_components=num_components,
            filter_size=filter_size,
            val_atol=0,
            val_rtol=1e-5,
            grad_atol=0,
            grad_rtol=1e-5,
        )

    def test_gradient_pls_2_m_greater_k(self):
        X = self.load_X()
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
        filter_size = 7
        X = X[..., : 9 + filter_size - 1]
        num_components = 9
        assert Y.shape[1] > 1
        assert (
            Y.shape[1] > X.shape[1] - filter_size + 1
        )  # The output of the convolution preprocessing is what is actually fed as input to the PLS algorithms.
        self.check_gradient_pls(
            X=X,
            Y=Y,
            num_components=num_components,
            filter_size=filter_size,
            val_atol=0,
            val_rtol=1e-5,
            grad_atol=0,
            grad_rtol=1e-5,
        )

    def check_cross_val_pls(self, X, Y, splits, atol, rtol):
        """
        Test that we can cross validate to get RMSE and best number of components for each target and for each split
        """
        from sklearn.model_selection import cross_validate

        def cross_val_preprocessing(X_train, Y_train, X_val, Y_val):
            x_mean = X_train.mean(axis=0, keepdims=True)
            X_train -= x_mean
            X_val -= x_mean
            y_mean = Y_train.mean(axis=0, keepdims=True)
            Y_train -= y_mean
            Y_val -= y_mean
            return X_train, Y_train, X_val, Y_val

        n_components = X.shape[1]

        sk_pls = SkPLS(n_components=n_components, scale=False)
        jax_pls_alg_1 = JAX_Alg_1()
        jax_pls_alg_2 = JAX_Alg_2()

        def cv_splitter(splits):
            uniq_splits = np.unique(splits)
            for split in uniq_splits:
                train_idxs = np.nonzero(splits != split)[0]
                val_idxs = np.nonzero(splits == split)[0]
                yield train_idxs, val_idxs

        def rmse_per_component(Y_true, Y_pred):
            e = Y_true - Y_pred
            se = e**2
            mse = np.mean(se, axis=-2)
            rmse = np.sqrt(mse)
            return rmse

        def jax_rmse_per_component(y_true, y_pred):
            e = y_true - y_pred
            se = e**2
            mse = jnp.mean(se, axis=-2)
            rmse = jnp.sqrt(mse)
            return (rmse,)

        jnp_splits = jnp.array(splits)

        # Calibrate SkPLS
        sk_results = cross_validate(
            sk_pls, X, Y, cv=cv_splitter(splits), return_estimator=True, n_jobs=-1
        )
        sk_models = sk_results["estimator"]
        # Extract regression matrices for SkPLS for all possible number of components and make a prediction with the regression matrices at all possible number of components.
        sk_Bs = np.empty((len(sk_models), n_components, X.shape[1], Y.shape[1]))
        sk_preds = np.empty((len(sk_models), n_components, X.shape[0], Y.shape[1]))
        for i, sk_model in enumerate(sk_models):
            for j in range(sk_Bs.shape[1]):
                sk_B_at_component_j = np.dot(
                    sk_model.x_rotations_[..., : j + 1],
                    sk_model.y_loadings_[..., : j + 1].T,
                )
                sk_Bs[i, j] = sk_B_at_component_j
            sk_pred = (X - sk_models[i]._x_mean) / sk_models[i]._x_std @ sk_Bs[
                i
            ] + sk_models[i].intercept_
            sk_preds[i] = sk_pred
            assert_allclose(
                sk_pred[-1], sk_models[i].predict(X), atol=0, rtol=0
            )  # Sanity check. SkPLS also uses the maximum number of components in its predict method.

        # Compute RMSE on the validation predictions
        sk_pls_rmses = np.empty((len(sk_models), n_components, Y.shape[1]))
        for i in range(len(sk_models)):
            val_idxs = val_idxs = np.nonzero(splits == i)[0]
            Y_true = Y[val_idxs]
            Y_pred = sk_preds[i, :, val_idxs, ...].swapaxes(0, 1)
            val_rmses = rmse_per_component(Y_true, Y_pred)
            sk_pls_rmses[i] = val_rmses

        # Calibrate NPPLS
        # Since SkLearn's cross_validate does not allow for a preprocessing function to be executed on each split, we have to get a bit creative.
        # This is because SkLearn's PLS implementation always mean centers X and Y based on the training data and uses these mean values in its predictions.
        # We subclass the original implementation and modify it to mimic the SkLearn implementation's behavior so that we can accurately compare results.
        class NpPLSWithPreprocessing(NpPLS):
            def __init__(
                self, algorithm: int = 1, dtype: np.float_ = np.float64
            ) -> None:
                super().__init__(algorithm, dtype)

            def fit(self, X: npt.ArrayLike, Y: npt.ArrayLike, A: int) -> None:
                self.X_mean = np.mean(X, axis=0)
                self.Y_mean = np.mean(Y, axis=0)
                X -= self.X_mean
                Y -= self.Y_mean
                return super().fit(X, Y, A)

            def predict(
                self, X: npt.ArrayLike, A: None | int = None
            ) -> npt.NDArray[np.float_]:
                return super().predict(X - self.X_mean, A) + self.Y_mean

        np_pls_alg_1 = NpPLSWithPreprocessing(algorithm=1)
        np_pls_alg_2 = NpPLSWithPreprocessing(algorithm=2)

        fit_params = {"A": n_components}
        np_pls_alg_1_results = cross_validate(
            np_pls_alg_1,
            X,
            Y,
            cv=cv_splitter(splits),
            scoring=lambda *args, **kwargs: 0,
            fit_params=fit_params,
            return_estimator=True,
            n_jobs=-1,
        )
        np_pls_alg_1_models = np_pls_alg_1_results["estimator"]
        np_pls_alg_2_results = cross_validate(
            np_pls_alg_2,
            X,
            Y,
            cv=cv_splitter(splits),
            scoring=lambda *args, **kwargs: 0,
            fit_params=fit_params,
            return_estimator=True,
            n_jobs=-1,
        )
        np_pls_alg_2_models = np_pls_alg_2_results["estimator"]

        # Compute RMSE on the validation predictions
        np_pls_alg_1_rmses = np.empty(
            (len(np_pls_alg_1_models), n_components, Y.shape[1])
        )
        np_pls_alg_2_rmses = np.empty(
            (len(np_pls_alg_2_models), n_components, Y.shape[1])
        )
        for i in range(len(np_pls_alg_1_models)):
            val_idxs = val_idxs = np.nonzero(splits == i)[0]
            Y_true = Y[val_idxs]
            Y_pred_alg_1 = np_pls_alg_1_models[i].predict(X[val_idxs])
            Y_pred_alg_2 = np_pls_alg_2_models[i].predict(X[val_idxs])
            val_rmses_alg_1 = rmse_per_component(Y_true, Y_pred_alg_1)
            val_rmses_alg_2 = rmse_per_component(Y_true, Y_pred_alg_2)
            np_pls_alg_1_rmses[i] = val_rmses_alg_1
            np_pls_alg_2_rmses[i] = val_rmses_alg_2

        # Calibrate JAX_PLS
        jax_pls_alg_1_results = jax_pls_alg_1.cv(
            X,
            Y,
            n_components,
            jnp_splits,
            cross_val_preprocessing,
            jax_rmse_per_component,
            ["RMSE"],
        )
        jax_pls_alg_2_results = jax_pls_alg_2.cv(
            X,
            Y,
            n_components,
            jnp_splits,
            cross_val_preprocessing,
            jax_rmse_per_component,
            ["RMSE"],
        )

        # Check that best number of components in terms of minimizing validation RMSE for each split is equal among all algorithms
        unique_splits = np.unique(splits).astype(int)
        sk_best_num_components = [
            [np.argmin(sk_pls_rmses[split][..., i]) for split in unique_splits]
            for i in range(Y.shape[1])
        ]
        np_pls_alg_1_best_num_components = [
            [np.argmin(np_pls_alg_1_rmses[split][..., i]) for split in unique_splits]
            for i in range(Y.shape[1])
        ]
        np_pls_alg_2_best_num_components = [
            [np.argmin(np_pls_alg_2_rmses[split][..., i]) for split in unique_splits]
            for i in range(Y.shape[1])
        ]
        jax_pls_alg_1_best_num_components = [
            [
                np.argmin(jax_pls_alg_1_results["RMSE"][split][..., i])
                for split in unique_splits
            ]
            for i in range(Y.shape[1])
        ]
        jax_pls_alg_2_best_num_components = [
            [
                np.argmin(jax_pls_alg_2_results["RMSE"][split][..., i])
                for split in unique_splits
            ]
            for i in range(Y.shape[1])
        ]

        assert sk_best_num_components == np_pls_alg_1_best_num_components
        assert sk_best_num_components == np_pls_alg_2_best_num_components
        assert sk_best_num_components == jax_pls_alg_1_best_num_components
        assert sk_best_num_components == jax_pls_alg_2_best_num_components

        # Check that the RMSE achieved is similar
        sk_best_rmses = [
            [np.amin(sk_pls_rmses[split][..., i]) for split in unique_splits]
            for i in range(Y.shape[1])
        ]
        np_pls_alg_1_best_rmses = [
            [np.amin(np_pls_alg_1_rmses[split][..., i]) for split in unique_splits]
            for i in range(Y.shape[1])
        ]
        np_pls_alg_2_best_rmses = [
            [np.amin(np_pls_alg_2_rmses[split][..., i]) for split in unique_splits]
            for i in range(Y.shape[1])
        ]
        jax_pls_alg_1_best_rmses = [
            [
                np.amin(jax_pls_alg_1_results["RMSE"][split][..., i])
                for split in unique_splits
            ]
            for i in range(Y.shape[1])
        ]
        jax_pls_alg_2_best_rmses = [
            [
                np.amin(jax_pls_alg_2_results["RMSE"][split][..., i])
                for split in unique_splits
            ]
            for i in range(Y.shape[1])
        ]

        assert_allclose(np_pls_alg_1_best_rmses, sk_best_rmses, atol=atol, rtol=rtol)
        assert_allclose(np_pls_alg_2_best_rmses, sk_best_rmses, atol=atol, rtol=rtol)
        assert_allclose(jax_pls_alg_1_best_rmses, sk_best_rmses, atol=atol, rtol=rtol)
        assert_allclose(jax_pls_alg_2_best_rmses, sk_best_rmses, atol=atol, rtol=rtol)

    def test_cross_val_pls_1(self):
        X = self.load_X()
        Y = self.load_Y(["Protein"])
        splits = self.load_Y(["split"])  # Contains 3 splits of differfent sizes
        assert Y.shape[1] == 1
        self.check_cross_val_pls(X, Y, splits, atol=0, rtol=1e-5)

    def test_cross_val_pls_2_m_less_k(self):
        X = self.load_X()
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
        splits = self.load_Y(["split"])  # Contains 3 splits of differfent sizes
        assert Y.shape[1] > 1
        assert Y.shape[1] < X.shape[1]
        self.check_cross_val_pls(X, Y, splits, atol=0, rtol=2e-4)

    def test_cross_val_pls_2_m_eq_k(self):
        X = self.load_X()
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
        splits = self.load_Y(["split"])  # Contains 3 splits of differfent sizes
        X = X[..., :10]
        assert Y.shape[1] > 1
        assert Y.shape[1] == X.shape[1]
        self.check_cross_val_pls(X, Y, splits, atol=0, rtol=1e-5)

    def test_cross_val_pls_2_m_greater_k(self):
        X = self.load_X()
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
        splits = self.load_Y(["split"])  # Contains 3 splits of differfent sizes
        X = X[..., :9]
        assert Y.shape[1] > 1
        assert Y.shape[1] > X.shape[1]
        self.check_cross_val_pls(X, Y, splits, atol=0, rtol=3e-5)


if __name__ == "__main__":
    tc = TestClass()
    tc.test_pls_1()
    tc.test_pls_2_m_less_k()
    tc.test_pls_2_m_eq_k()
    tc.test_pls_2_m_greater_k()
    tc.test_sanity_check_pls_regression()
    tc.test_sanity_check_pls_regression_constant_column_Y()
    tc.test_pls_1_constant_y()
    tc.test_pls_2_m_less_k_constant_y()
    tc.test_pls_2_m_eq_k_constant_y()
    tc.test_pls_2_m_greater_k_constant_y()
    tc.test_gradient_pls_1()
    tc.test_gradient_pls_2_m_less_k()
    tc.test_gradient_pls_2_m_eq_k()
    tc.test_gradient_pls_2_m_greater_k()
    tc.test_cross_val_pls_1()
    tc.test_cross_val_pls_2_m_less_k()
    tc.test_cross_val_pls_2_m_eq_k()
    tc.test_cross_val_pls_2_m_greater_k()