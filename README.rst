Improved Kernel Partial Least Squares (IKPLS) and Fast Cross-Validation
=======================================================================

.. image:: https://img.shields.io/pypi/v/ikpls.svg
   :target: https://pypi.python.org/pypi/ikpls/
   :alt: PyPI Version
.. image:: https://img.shields.io/pypi/l/ikpls.svg
   :target: https://pypi.python.org/pypi/ikpls/
   :alt: License
.. image:: https://readthedocs.org/projects/ikpls/badge/?version=latest
   :target: https://ikpls.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://github.com/Sm00thix/IKPLS/actions/workflows/workflow.yml/badge.svg
   :target: https://github.com/Sm00thix/IKPLS/actions/workflows/workflow.yml
   :alt: Build Status

Fast CPU, GPU, and TPU Python implementations of Improved Kernel PLS Algorithm #1 and Algorithm #2 by Dayal and MacGregor [1]_. Improved Kernel PLS is both fast [2]_ and numerically stable [3]_.
The CPU implementations use NumPy [4]_ and subclass BaseEstimator from scikit-learn [5]_, allowing integration into scikit-learn's ecosystem of machine learning algorithms and pipelines. For example, the CPU implementations can be used with scikit-learn's `cross_validate <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html>`_.
The GPU and TPU implementations use Google's JAX [6]_. JAX supports automatic differentiation while allowing CPU, GPU, and TPU execution. This implies that the JAX implementations can be combined with deep learning approaches, as the PLS fit is differentiable.

The documentation is available at https://ikpls.readthedocs.io/en/latest/; examples can be found at https://github.com/Sm00thix/IKPLS/tree/main/examples.

.. [1] `Dayal, B. S., & MacGregor, J. F. (1997). Improved PLS algorithms. Journal of Chemometrics: A Journal of the Chemometrics Society, 11(1), 73-85`_.
.. [2] `Alin, A. (2009). Comparison of PLS algorithms when the number of objects is much larger than the number of variables. Statistical papers, 50, 711-720`_.
.. [3] `Andersson, M. (2009). A comparison of nine PLS1 algorithms. Journal of Chemometrics: A Journal of the Chemometrics Society, 23(10), 518-529`_.
.. [4] `NumPy`_.
.. [5] `scikit-learn`_.
.. [6] `JAX`_.

.. _Dayal, B. S., & MacGregor, J. F. (1997). Improved PLS algorithms. Journal of Chemometrics\: A Journal of the Chemometrics Society, 11(1), 73-85: https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23?
.. _Alin, A. (2009). Comparison of PLS algorithms when the number of objects is much larger than the number of variables. Statistical papers, 50, 711-720: https://doi.org/10.1007/s00362-009-0251-7
.. _Andersson, M. (2009). A comparison of nine PLS1 algorithms. Journal of Chemometrics\: A Journal of the Chemometrics Society, 23(10), 518-529: https://doi.org/10.1002/cem.1248
.. _NumPy: https://numpy.org/
.. _scikit-learn: https://scikit-learn.org/stable/
.. _JAX: https://jax.readthedocs.io/en/latest/

Extremely Fast Cross-Validation
-------------------------------
In addition to the implementations mentioned above, this package contains the novel, fast cross-validation algorithms mentioned in [7]_ using both IKPLS algorithms.
The fast cross-validation algorithms benefit both IKPLS Algorithms and especially Algorithm #2.
The fast cross-validation algorithms are mathematically equivalent to the classical cross-validation algorithm. Still, they are much quicker if cross-validation splits exceed 3.
The fast cross-validation algorithms correctly handle (column-wise) centering and scaling of the X and Y input matrices using training set means and standard deviations to avoid data leakage from the validation set.
This centering and scaling can be enabled by setting the center parameter to True and the scale parameter to True, respectively.

.. [7] `Engstrøm, O.-C. G. (2024). Shortcutting Cross-Validation: Efficiently Deriving Column-Wise Centered and Scaled Training Set $\\mathbf{X}^\\mathbf{T}\\mathbf{X}$ and $\\mathbf{X}^\\mathbf{T}\\mathbf{Y}$ Without Full Recomputation of Matrix Products or Statistical Moments`_.

.. _Engstrøm, O.-C. G. (2024). Shortcutting Cross-Validation\: Efficiently Deriving Column-Wise Centered and Scaled Training Set $\\mathbf{X}^\\mathbf{T}\\mathbf{X}$ and $\\mathbf{X}^\\mathbf{T}\\mathbf{Y}$ Without Full Recomputation of Matrix Products or Statistical Moments: https://arxiv.org/abs/2401.13185

Pre-requisites
--------------

The JAX implementations support running on both CPU, GPU, and TPU. To use the GPU or TPU, follow the instructions from the `JAX Installation Guide
<https://jax.readthedocs.io/en/latest/installation.html>`_.

To ensure that JAX implementations use Float64, set the environment variable JAX_ENABLE_X64=True as per the `Current Gotchas
<https://github.com/google/jax#current-gotchas>`_.

Installation
------------

-  | Install the package for Python3 using the following command:
   | ``$ pip3 install ikpls``
-  | Now you can import the NumPy and JAX implementations with:
   | ``from ikpls.numpy_ikpls import PLS as NpPLS``
   | ``from ikpls.jax_ikpls_alg_1 import PLS as JAXPLS_Alg_1``
   | ``from ikpls.jax_ikpls_alg_2 import PLS as JAXPLS_Alg_2``
   | ``from ikpls.fast_cross_validation.numpy_ikpls import PLS as NpPLS_FastCV``


Quick Start
-----------
Use the ikpls package for PLS modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   

  .. code:: python

    import numpy as np

    from ikpls.numpy_ikpls import PLS

    N = 100  # Number of samples.
    K = 50  # Number of features.
    M = 10  # Number of targets.
    A = 20  # Number of latent variables (PLS components).

    X = np.random.uniform(size=(N, K)).astype(np.float64)
    Y = np.random.uniform(size=(N, M)).astype(np.float64)

    # The other PLS algorithms and implementations, except for NpPLS_FastCV, have the same interface for fit() and predict().
    np_ikpls_alg_1 = PLS(algorithm=1)
    np_ikpls_alg_1.fit(X, Y, A)

    y_pred = np_ikpls_alg_1.predict(X) # Has shape (A, N, M) = (20, 100, 10). Contains a prediction for all possible numbers of components up to and including A.
    y_pred_20_components = np_ikpls_alg_1.predict(X, n_components=20) # Has shape (N, M) = (100, 10).
    (y_pred_20_components == y_pred[19]).all() # True

    # The internal model parameters can be accessed as follows:
    np_ikpls_alg_1.B  # Regression coefficients tensor of shape (A, K, M) = (20, 50, 10).
    np_ikpls_alg_1.W  # X weights matrix of shape (K, A) = (50, 20).
    np_ikpls_alg_1.P  # X loadings matrix of shape (K, A) = (50, 20).
    np_ikpls_alg_1.Q  # Y loadings matrix of shape (M, A) = (10, 20).
    np_ikpls_alg_1.R  # X rotations matrix of shape (K, A) = (50, 20).
    np_ikpls_alg_1.T  # X scores matrix of shape (N, A) = (100, 20). This is only computed for IKPLS Algorithm #1.

Examples
~~~~~~~~

In `examples <https://github.com/Sm00thix/IKPLS/tree/main/examples>`_ you will find:

- `Fit and Predict with NumPy. <https://github.com/Sm00thix/IKPLS/tree/main/examples/fit_predict_numpy.py>`_

- `Fit and Predict with JAX. <https://github.com/Sm00thix/IKPLS/tree/main/examples/fit_predict_jax.py>`_

- `Cross-validate with NumPy. <https://github.com/Sm00thix/IKPLS/tree/main/examples/cross_val_numpy.py>`_

- `Cross-validate with NumPy and fast cross-validation. <https://github.com/Sm00thix/IKPLS/tree/main/examples/fast_cross_val_numpy.py>`_

- `Cross-validate with JAX. <https://github.com/Sm00thix/IKPLS/tree/main/examples/cross_val_jax.py>`_

- `Compute the gradient of a preprocessing convolution filter with respect to the RMSE between the target value and the value predicted by PLS after fitting with JAX. <https://github.com/Sm00thix/IKPLS/tree/main/examples/gradient_jax.py>`_

