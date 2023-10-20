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

    def load_target_values(self, values: list[str]) -> npt.NDArray[np.float_]:
        target_values = self.csv[values].to_numpy()
        return target_values

    def fit_models(self):
        self.sk_pls = SkPLS(n_components=self.n_components, scale=False)
        self.np_pls_alg_1 = NpPLS(algorithm=1)
        self.np_pls_alg_2 = NpPLS(algorithm=2)
        self.jax_pls_alg_1 = JAX_Alg_1()
        self.jax_pls_alg_2 = JAX_Alg_2()

        self.sk_pls.fit(X=self.spectra, Y=self.targets)
        self.np_pls_alg_1.fit(X=self.spectra, Y=self.targets, A=self.n_components)
        self.np_pls_alg_2.fit(X=self.spectra, Y=self.targets, A=self.n_components)
        self.jax_pls_alg_1.fit(
            X=self.jnp_spectra, Y=self.jnp_targets, A=self.n_components
        )
        self.jax_pls_alg_2.fit(
            X=self.jnp_spectra, Y=self.jnp_targets, A=self.n_components
        )

        # Reconstruct SkPLS regression matrix for all components
        sk_B = np.empty(self.np_pls_alg_1.B.shape)
        for i in range(sk_B.shape[0]):
            sk_B_at_component_i = np.dot(
                self.sk_pls.x_rotations_[..., : i + 1],
                self.sk_pls.y_loadings_[..., : i + 1].T,
            )
            sk_B[i] = sk_B_at_component_i
        self.sk_B = sk_B

    def assert_matrix_orthogonal(self, M, atol, rtol):
        MTM = np.dot(M.T, M)
        assert_allclose(MTM, np.diag(np.diag(MTM)), atol=atol, rtol=rtol)

    def check_internal_matrices(self, atol, rtol, max_n_components=-1):
        if max_n_components == -1:
            max_n_components = self.np_pls_alg_1.A
        # X weights
        assert_allclose(
            np.abs(self.np_pls_alg_1.W[..., :max_n_components]),
            np.abs(self.sk_pls.x_weights_[..., :max_n_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(self.np_pls_alg_2.W[..., :max_n_components]),
            np.abs(self.sk_pls.x_weights_[..., :max_n_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np.array(self.jax_pls_alg_1.W)[..., :max_n_components]),
            np.abs(self.sk_pls.x_weights_[..., :max_n_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np.array(self.jax_pls_alg_2.W)[..., :max_n_components]),
            np.abs(self.sk_pls.x_weights_[..., :max_n_components]),
            atol=atol,
            rtol=rtol,
        )

        # X loadings
        assert_allclose(
            np.abs(self.np_pls_alg_1.P[..., :max_n_components]),
            np.abs(self.sk_pls.x_loadings_[..., :max_n_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(self.np_pls_alg_2.P[..., :max_n_components]),
            np.abs(self.sk_pls.x_loadings_[..., :max_n_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np.array(self.jax_pls_alg_1.P)[..., :max_n_components]),
            np.abs(self.sk_pls.x_loadings_[..., :max_n_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np.array(self.jax_pls_alg_2.P)[..., :max_n_components]),
            np.abs(self.sk_pls.x_loadings_[..., :max_n_components]),
            atol=atol,
            rtol=rtol,
        )

        # Y loadings
        assert_allclose(
            np.abs(self.np_pls_alg_1.Q[..., :max_n_components]),
            np.abs(self.sk_pls.y_loadings_[..., :max_n_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(self.np_pls_alg_2.Q[..., :max_n_components]),
            np.abs(self.sk_pls.y_loadings_[..., :max_n_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np.array(self.jax_pls_alg_1.Q)[..., :max_n_components]),
            np.abs(self.sk_pls.y_loadings_[..., :max_n_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np.array(self.jax_pls_alg_2.Q)[..., :max_n_components]),
            np.abs(self.sk_pls.y_loadings_[..., :max_n_components]),
            atol=atol,
            rtol=rtol,
        )

        # X rotations
        assert_allclose(
            np.abs(self.np_pls_alg_1.R[..., :max_n_components]),
            np.abs(self.sk_pls.x_rotations_[..., :max_n_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(self.np_pls_alg_2.R[..., :max_n_components]),
            np.abs(self.sk_pls.x_rotations_[..., :max_n_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np.array(self.jax_pls_alg_1.R[..., :max_n_components])),
            np.abs(self.sk_pls.x_rotations_[..., :max_n_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np.array(self.jax_pls_alg_2.R[..., :max_n_components])),
            np.abs(self.sk_pls.x_rotations_[..., :max_n_components]),
            atol=atol,
            rtol=rtol,
        )

        # X scores - only computed by IKPLS Algorithm #1
        assert_allclose(
            np.abs(self.np_pls_alg_1.T[..., :max_n_components]),
            np.abs(self.sk_pls.x_scores_[..., :max_n_components]),
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.abs(np.array(self.jax_pls_alg_1.T[..., :max_n_components])),
            np.abs(self.sk_pls.x_scores_[..., :max_n_components]),
            atol=atol,
            rtol=rtol,
        )

        # Regression matrices
        assert_allclose(
            self.np_pls_alg_1.B[:max_n_components],
            self.sk_B[:max_n_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            self.np_pls_alg_2.B[:max_n_components],
            self.sk_B[:max_n_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(self.jax_pls_alg_1.B)[:max_n_components],
            self.sk_B[:max_n_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(self.jax_pls_alg_2.B)[:max_n_components],
            self.sk_B[:max_n_components],
            atol=atol,
            rtol=rtol,
        )

    def check_predictions(self, atol, rtol, max_n_components=-1):
        if max_n_components == -1:
            max_n_components = self.np_pls_alg_1.A
        # Check predictions for each and all possible number of components.
        sk_all_preds = self.spectra @ self.sk_B
        assert_allclose(
            self.np_pls_alg_1.predict(self.spectra)[:max_n_components]
            + self.mean_targets,
            sk_all_preds[:max_n_components] + self.mean_targets,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            self.np_pls_alg_2.predict(self.spectra)[:max_n_components]
            + self.mean_targets,
            sk_all_preds[:max_n_components] + self.mean_targets,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(self.jax_pls_alg_1.predict(self.spectra))[:max_n_components]
            + self.mean_targets,
            sk_all_preds[:max_n_components] + self.mean_targets,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(self.jax_pls_alg_2.predict(self.spectra))[:max_n_components]
            + self.mean_targets,
            sk_all_preds[:max_n_components] + self.mean_targets,
            atol=atol,
            rtol=rtol,
        )

        # Check predictions using the largest good number of components.
        sk_final_pred = sk_all_preds[max_n_components-1]
        assert_allclose(
            self.np_pls_alg_1.predict(self.spectra, A=max_n_components)
            + self.mean_targets,
            sk_final_pred + self.mean_targets,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            self.np_pls_alg_2.predict(self.spectra, A=max_n_components)
            + self.mean_targets,
            sk_final_pred + self.mean_targets,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(self.jax_pls_alg_1.predict(self.spectra, A=max_n_components))
            + self.mean_targets,
            sk_final_pred + self.mean_targets,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.array(self.jax_pls_alg_2.predict(self.spectra, A=self.n_components))
            + self.mean_targets,
            sk_final_pred + self.mean_targets,
            atol=atol,
            rtol=rtol,
        )

    def check_orthogonality_properties(self, atol, rtol, max_n_components=-1):
        if max_n_components == -1:
            max_n_components = self.np_pls_alg_1.A
        # X weights should be orthogonal
        self.assert_matrix_orthogonal(
            self.np_pls_alg_1.W[..., :max_n_components], atol=atol, rtol=rtol
        )
        self.assert_matrix_orthogonal(
            self.np_pls_alg_2.W[..., :max_n_components], atol=atol, rtol=rtol
        )
        self.assert_matrix_orthogonal(
            np.array(self.jax_pls_alg_1.W)[..., :max_n_components], atol=atol, rtol=rtol
        )
        self.assert_matrix_orthogonal(
            np.array(self.jax_pls_alg_2.W)[..., :max_n_components], atol=atol, rtol=rtol
        )

        # X scores (only computed by algorithm 1) should be orthogonal
        self.assert_matrix_orthogonal(
            self.np_pls_alg_1.T[..., :max_n_components], atol=atol, rtol=rtol
        )
        self.assert_matrix_orthogonal(
            np.array(self.jax_pls_alg_1.T)[..., :max_n_components], atol=atol, rtol=rtol
        )

    def check_equality_properties(self, atol, rtol, max_n_components=-1):
        if max_n_components == -1:
            max_n_components = self.np_pls_alg_1.A
        # X can be reconstructed by multiplying X scores (T) and the transpose of X loadings (P)
        assert_allclose(
            np.dot(
                self.np_pls_alg_1.T[..., :max_n_components],
                self.np_pls_alg_1.P[..., :max_n_components].T,
            ),
            self.spectra,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.dot(
                np.array(self.jax_pls_alg_1.T[..., :max_n_components]),
                np.array(self.jax_pls_alg_1.P[..., :max_n_components]).T,
            ),
            self.spectra,
            atol=atol,
            rtol=rtol,
        )

        # X multiplied by X rotations (R) should be equal to X scores (T)
        assert_allclose(
            np.dot(self.spectra, self.np_pls_alg_1.R[..., :max_n_components]),
            self.np_pls_alg_1.T[..., :max_n_components],
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            np.dot(
                self.spectra, np.array(self.jax_pls_alg_1.R[..., :max_n_components])
            ),
            np.array(self.jax_pls_alg_1.T[..., :max_n_components]),
            atol=atol,
            rtol=rtol,
        )

    def test_pls_1(self):
        """
        Test PLS1.
        """
        self.spectra = self.raw_spectra - np.mean(
            self.raw_spectra, axis=0
        )  # SkLearn's PLS implementation always centers its X input. This ensures that the X input is always centered for all algorithms.
        self.spectra /= np.std(
            self.spectra, axis=0
        )  # Let's also scale them for better numerical stability
        self.jnp_spectra = jnp.array(self.spectra)
        self.targets = self.load_target_values(["Protein"])
        self.mean_targets = np.mean(self.targets, axis=0, keepdims=True)
        self.targets -= (
            self.mean_targets
        )  # SkLearn's PLS implementation always centers its Y input. This ensures that the Y input is always centered for all algorithms.
        # self.targets /= np.std(
        #     self.targets, axis=0
        # )  # Scale target features to have unit variance so that no single feature dominates in variance.
        self.jnp_targets = jnp.array(self.targets)
        self.n_components = 25
        assert self.targets.shape[1] == 1
        self.fit_models()
        self.check_equality_properties(atol=1e-1, rtol=1e-5)
        self.check_orthogonality_properties(atol=1e-1, rtol=0)
        # self.check_internal_matrices(atol=1e-2, rtol=1e-2)
        self.check_predictions(
            atol=1e-8, rtol=0
        )  # PLS1 is very numerically stable for protein.

    def test_pls_2_m_less_k(self):
        """
        Test PLS2 where the number of targets is less than the number of features (M < K).
        """
        self.spectra = self.raw_spectra - np.mean(
            self.raw_spectra, axis=0
        )  # SkLearn's PLS implementation always centers its X input. This ensures that the X input is always centered for all algorithms.
        self.spectra /= np.std(
            self.spectra, axis=0
        )  # Let's also scale them for better numerical stability
        self.jnp_spectra = jnp.array(self.spectra)
        self.targets = self.load_target_values(
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
        self.mean_targets = np.mean(self.targets, axis=0, keepdims=True)
        self.targets -= (
            self.mean_targets
        )  # SkLearn's PLS implementation always centers its Y input. This ensures that the Y input is always centered for all algorithms.
        # self.targets /= np.std(
        #     self.targets, axis=0
        # )  # Scale target features to have unit variance so that no single feature dominates in variance.
        assert self.targets.shape[1] < self.spectra.shape[1]
        self.jnp_targets = jnp.array(self.targets)
        self.n_components = 25
        self.fit_models()
        self.check_equality_properties(atol=1e-1, rtol=1e-5)
        self.check_orthogonality_properties(atol=1e-1, rtol=0)
        # self.check_internal_matrices(atol=1e-1, rtol=1e-1)
        self.check_predictions(
            atol=5e-3, rtol=0
        )  # PLS2 is not as numerically stable as PLS1.

    def test_pls_2_m_eq_k(self):
        """
        Test PLS2 where the number of targets is equal to the number of features (M = K).
        """
        self.spectra = self.raw_spectra[..., :10] - np.mean(
            self.raw_spectra[..., :10], axis=0
        )  # SkLearn's PLS implementation always centers its X input. This ensures that the X input is always centered for all algorithms.
        self.spectra /= np.std(
            self.spectra, axis=0
        )  # Let's also scale them for better numerical stability
        self.jnp_spectra = jnp.array(self.spectra)
        self.targets = self.load_target_values(
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
        self.mean_targets = np.mean(self.targets, axis=0, keepdims=True)
        self.targets -= (
            self.mean_targets
        )  # SkLearn's PLS implementation always centers its Y input. This ensures that the Y input is always centered for all algorithms.
        # self.targets /= np.std(
        #     self.targets, axis=0
        # )  # Scale target features to have unit variance so that no single feature dominates in variance.
        assert self.targets.shape[1] == self.spectra.shape[1]
        self.jnp_targets = jnp.array(self.targets)
        self.n_components = 10
        self.fit_models()
        self.check_equality_properties(atol=1e-1, rtol=1e-5)
        self.check_orthogonality_properties(atol=1e-1, rtol=0)
        # self.check_internal_matrices(atol=1e-1, rtol=1e-1)
        self.check_predictions(
            atol=5e-3, rtol=0
        )  # PLS2 is not as numerically stable as PLS1.

    def test_pls_2_m_greater_k(self):
        """
        Test PLS2 where the number of targets is greater than the number of features (M > K).
        """
        self.spectra = self.raw_spectra[..., :9] - np.mean(
            self.raw_spectra[..., :9], axis=0
        )  # SkLearn's PLS implementation always centers its X input. This ensures that the X input is always centered for all algorithms.
        self.spectra /= np.std(
            self.spectra, axis=0
        )  # Let's also scale them for better numerical stability
        self.jnp_spectra = jnp.array(self.spectra)
        self.targets = self.load_target_values(
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
        self.mean_targets = np.mean(self.targets, axis=0, keepdims=True)
        self.targets -= (
            self.mean_targets
        )  # SkLearn's PLS implementation always centers its Y input. This ensures that the Y input is always centered for all algorithms.
        # self.targets /= np.std(
        #     self.targets, axis=0
        # )  # Scale target features to have unit variance so that no single feature dominates in variance.
        assert self.targets.shape[1] > self.spectra.shape[1]
        self.jnp_targets = jnp.array(self.targets)
        self.n_components = 9
        self.fit_models()
        self.check_equality_properties(atol=1e-1, rtol=1e-5)
        self.check_orthogonality_properties(atol=1e-1, rtol=0)
        # self.check_internal_matrices(atol=1e-1, rtol=1e-1)
        self.check_predictions(
            atol=5e-3, rtol=0
        )  # PLS2 is not as numerically stable as PLS1.

    def test_early_stop_fitting_pls_1(self):
        """
        The NumPy implementations will stop iterating through components if the residual comes close to 0.
        """
        subset_spectra = self.raw_spectra[..., :3]
        lines = np.arange(self.raw_spectra.shape[0] * 22).reshape(
            self.raw_spectra.shape[0], 22
        )
        self.spectra = np.hstack((subset_spectra, lines))
        self.spectra -= np.mean(self.spectra, axis=0)
        self.spectra /= np.std(self.spectra, axis=0)
        self.jnp_spectra = jnp.array(self.spectra)

        self.targets = self.spectra[..., 0:1]
        self.mean_targets = np.mean(self.targets, axis=0)
        self.targets -= self.mean_targets
        self.jnp_targets = jnp.array(self.targets)
        self.n_components = 25
        max_n_components = 4  # There are 3 independent vectors and 22 vectors which are all linearly dependent on eachother, giving the problem a rank of 4.

        assert self.targets.shape[1] == 1
        self.fit_models()
        self.check_equality_properties(
            atol=1e-1, rtol=1e-5, max_n_components=max_n_components
        )
        self.check_orthogonality_properties(
            atol=1e-1, rtol=0, max_n_components=max_n_components
        )
        # self.check_internal_matrices(
        #     atol=1e-2, rtol=1e-2, max_n_components=max_n_components
        # )  # SkLearn does not necessarily stop early. Let the next prediction test be the judge of whether the early stopping is good or not.
        self.check_predictions(
            atol=1e-8, rtol=0, max_n_components=max_n_components
        )  # PLS1 is very numerically stable for protein.

    def test_early_stop_fitting_pls_2_m_less_k(self):
        """
        The NumPy implementations will stop iterating through components if the residual comes close to 0.
        """
        subset_spectra = self.raw_spectra[..., :3]
        lines = np.arange(self.raw_spectra.shape[0] * 22).reshape(
            self.raw_spectra.shape[0], 22
        )
        self.spectra = np.hstack((subset_spectra, lines))
        self.spectra -= np.mean(self.spectra, axis=0)
        self.spectra /= np.std(self.spectra, axis=0)
        self.jnp_spectra = jnp.array(self.spectra)

        self.targets = self.spectra[..., :10]
        self.mean_targets = np.mean(self.targets, axis=0)
        self.targets -= self.mean_targets
        self.jnp_targets = jnp.array(self.targets)
        self.n_components = 25
        max_n_components = 4  # There are 3 independent vectors and 22 vectors which are all linearly dependent on eachother, giving the problem a rank of 4.

        assert self.targets.shape[1] < self.spectra.shape[1]
        self.fit_models()
        self.check_equality_properties(
            atol=1e-1, rtol=1e-5, max_n_components=max_n_components
        )
        self.check_orthogonality_properties(
            atol=1e-1, rtol=0, max_n_components=max_n_components
        )
        # self.check_internal_matrices(
        #     atol=1e-1, rtol=1e-1, max_n_components=max_n_components
        # )  # SkLearn does not necessarily stop early. Let the next prediction test be the judge of whether the early stopping is good or not.
        self.check_predictions(
            atol=5e-3, rtol=0, max_n_components=max_n_components
        )  # PLS2 is not as numerically stable as PLS1.

    def test_early_stop_fitting_pls_2_m_eq_k(self):
        """
        The NumPy implementations will stop iterating through components if the residual comes close to 0.
        """
        subset_spectra = self.raw_spectra[..., :3]
        lines = np.arange(self.raw_spectra.shape[0] * 22).reshape(
            self.raw_spectra.shape[0], 22
        )
        self.spectra = np.hstack((subset_spectra, lines))
        self.spectra -= np.mean(self.spectra, axis=0)
        self.spectra /= np.std(self.spectra, axis=0)
        self.jnp_spectra = jnp.array(self.spectra)

        # self.targets = self.load_target_values(["Protein"])
        self.targets = self.spectra
        self.mean_targets = np.mean(self.targets, axis=0)
        self.targets -= self.mean_targets
        self.jnp_targets = jnp.array(self.targets)
        self.n_components = 25
        max_n_components = 4  # There are 3 independent vectors and 22 vectors which are all linearly dependent on eachother, giving the problem a rank of 4.

        assert self.targets.shape[1] == self.spectra.shape[1]
        self.fit_models()
        self.check_equality_properties(
            atol=1e-1, rtol=1e-5, max_n_components=max_n_components
        )
        self.check_orthogonality_properties(
            atol=1e-1, rtol=0, max_n_components=max_n_components
        )
        # self.check_internal_matrices(
        #     atol=1e-1, rtol=1e-1, max_n_components=max_n_components
        # )
        self.check_predictions(
            atol=5e-3, rtol=0, max_n_components=max_n_components
        )  # PLS2 is not as numerically stable as PLS1.

    def test_early_stop_fitting_pls_2_m_greater_k(self):
        """
        The NumPy implementations will stop iterating through components if the residual comes close to 0.
        """
        subset_spectra = self.raw_spectra[..., :3]
        lines = np.arange(self.raw_spectra.shape[0] * 22).reshape(
            self.raw_spectra.shape[0], 22
        )
        self.spectra = np.hstack((subset_spectra, lines))
        self.spectra -= np.mean(self.spectra, axis=0)
        self.spectra /= np.std(self.spectra, axis=0)
        self.jnp_spectra = jnp.array(self.spectra)

        # self.targets = self.load_target_values(["Protein"])
        self.targets = self.spectra
        self.mean_targets = np.mean(self.targets, axis=0)
        self.targets -= self.mean_targets
        self.jnp_targets = jnp.array(self.targets)
        self.n_components = 24
        max_n_components = 4  # There are 3 independent vectors and 22 vectors which are all linearly dependent on eachother, giving the problem a rank of 4.

        self.spectra = self.spectra[..., :24]
        self.jnp_spectra = self.jnp_spectra[..., :24]

        assert self.targets.shape[1] > self.spectra.shape[1]
        self.fit_models()
        self.check_equality_properties(atol=1e-1, rtol=1e-5, max_n_components=max_n_components)
        self.check_orthogonality_properties(atol=1e-1, rtol=0, max_n_components=max_n_components)
        # self.check_internal_matrices(atol=1e-1, rtol=1e-1, max_n_components=max_n_components) # SkLearn does not necessarily stop early. Let the next prediction test be the judge of whether the early stopping is good or not.
        self.check_predictions(
            atol=5e-3, rtol=0, max_n_components=max_n_components
        )  # PLS2 is not as numerically stable as PLS1.


if __name__ == "__main__":
    tc = TestClass()
    tc.test_pls_1()
    tc.test_pls_2_m_less_k()
    tc.test_pls_2_m_eq_k()
    tc.test_pls_2_m_greater_k()
    tc.test_early_stop_fitting_pls_1() # Stop after 4 components. Here, own algorithms fails to stop early. Norm is constant at approx. 1e-14.
    tc.test_early_stop_fitting_pls_2_m_less_k()  # Stop after 4 components. Here, own algorithms fails to stop early. Norm explodes.
    tc.test_early_stop_fitting_pls_2_m_eq_k()  # Stop after 4 components
    tc.test_early_stop_fitting_pls_2_m_greater_k() # Fix denne. Lykkes ikke med at skaffe early stopping
