# Improved Kernel PLS
Fast CPU, GPU, and TPU Python implementations of Improved Kernel PLS Algorithm #1 and Algorithm #2 by Dayal and MacGregor[^1]. Improved Kernel PLS has been shown to be both fast[^2] and numerically stable[^3].
The CPU implementations are made using NumPy[^4] and subclass BaseEstimator from scikit-learn[^5] allowing integration into scikit-learn's ecosystem of machine learning algorithms and pipelines. For example, the CPU implementations can be used with scikit-learn's [`cross_validate`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html).
The GPU and TPU implementations are made using Google's JAX[^6]. While allowing CPU, GPU, and TPU execution, automatic differentiation is also supported by JAX. This implies that the JAX implementations can be used together with deep learning approaches as the PLS fit is differentiable.

[^1]: [Dayal, B. S., & MacGregor, J. F. (1997). Improved PLS algorithms. Journal of Chemometrics: A Journal of the Chemometrics Society, 11(1), 73-85.](https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23)
[^2]: [Alin, A. (2009). Comparison of PLS algorithms when number of objects is much larger than number of variables. Statistical papers, 50, 711-720.](https://link.springer.com/content/pdf/10.1007/s00362-009-0251-7.pdf)
[^3]: [Andersson, M. (2009). A comparison of nine PLS1 algorithms. Journal of Chemometrics: A Journal of the Chemometrics Society, 23(10), 518-529.](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/pdf/10.1002/cem.1248?)
[^4]: [NumPy.](https://numpy.org/)
[^5]: [scikit-learn.](https://scikit-learn.org/stable/)
[^6]: [JAX.](https://jax.readthedocs.io/en/latest/)

# Improved Kernel PLS
Fast CPU, GPU, and TPU Python implementations of Improved Kernel PLS Algorithm #1 and Algorithm #2 by Dayal and MacGregor[^1]. Improved Kernel PLS has been shown to be both fast[^2] and numerically stable[^3].
The CPU implementations are made using NumPy[^4] and subclass BaseEstimator from scikit-learn[^5] allowing integration into scikit-learn's ecosystem of machine learning algorithms and pipelines. For example, the CPU implementations can be used with scikit-learn's [`cross_validate`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html).
The GPU and TPU implementations are made using Google's JAX[^6]. While allowing CPU, GPU, and TPU execution, automatic differentiation is also supported by JAX. This implies that the JAX implementations can be used together with deep learning approaches as the PLS fit is differentiable.

[^1]: [Dayal, B. S., & MacGregor, J. F. (1997). Improved PLS algorithms. Journal of Chemometrics: A Journal of the Chemometrics Society, 11(1), 73-85.](https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23)
[^2]: [Alin, A. (2009). Comparison of PLS algorithms when number of objects is much larger than number of variables. Statistical papers, 50, 711-720.](https://link.springer.com/content/pdf/10.1007/s00362-009-0251-7.pdf)
[^3]: [Andersson, M. (2009). A comparison of nine PLS1 algorithms. Journal of Chemometrics: A Journal of the Chemometrics Society, 23(10), 518-529.](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/pdf/10.1002/cem.1248?)
[^4]: [NumPy.](https://numpy.org/)
[^5]: [scikit-learn.](https://scikit-learn.org/stable/)
[^6]: [JAX.](https://jax.readthedocs.io/en/latest/)


## Pre-requisites
The JAX implementations support running on both CPU, GPU, and TPU. To use the GPU or TPU, follow the instructions from the [JAX Installation Guide](https://jax.readthedocs.io/en/latest/installation.html).
The JAX implementations support running on both CPU, GPU, and TPU. To use the GPU or TPU, follow the instructions from the [JAX Installation Guide](https://jax.readthedocs.io/en/latest/installation.html).
To ensure that JAX implementations use Float64, set the environment variable JAX_ENABLE_X64=True as per the [Current Gotchas](https://github.com/google/jax#current-gotchas).

## Installation
* Install the package for Python3 using the following command:  
``$ pip3 install ikpls``
* Now you can import the NumPy and JAX implementations with:
```python
from ikpls.numpy_ikpls import PLS as NpPLS
from ikpls.jax_ikpls_alg_1 import PLS as JAXPLS_Alg_1
from ikpls.jax_ikpls_alg_2 import PLS as JAXPLS_Alg_2
```

## Quick Start
### Use the ikpls package for PLS modelling
```python
from ikpls.numpy_ikpls import PLS
import numpy as np

N = 100 # Number of samples.
K = 50 # Number of features.
M = 10 # Number of targets.
A = 20 # Number of latent variables (PLS components).

# Using float64 is important for numerical stability.
X = np.random.uniform(size=(N, K)).astype(np.float64)
Y = np.random.uniform(size=(N, M)).astype(np.float64)

# The other PLS algorithms and implementations have the same interface for fit() and predict().
np_ikpls_alg_1 = PLS(algorithm=1)
np_ikpls_alg_1.fit(X, Y, A)

y_pred = np_ikpls_alg_1.predict(X) # Has shape (A, N, M) = (20, 100, 10). Contains a prediction for all possible number of components up to and including A.
y_pred_20_components = np_ikpls_alg_1.predict(X, n_components=20) # Has shape (N, M) = (100, 10).
(y_pred_20_components == y_pred[19]).all() # True

# The internal model parameters can be accessed as follows:
np_ikpls_alg_1.B # Regression coefficients tensor of shape (A, K, M) = (20, 50, 10).
np_ikpls_alg_1.W # X weights matrix of shape (K, A) = (50, 20).
np_ikpls_alg_1.P # X loadings matrix of shape (K, A) = (50, 20).
np_ikpls_alg_1.Q # Y loadings matrix of shape (M, A) = (10, 20).
np_ikpls_alg_1.R # X rotations matrix of shape (K, A) = (50, 20).
np_ikpls_alg_1.T # X scores matrix of shape (N, A) = (100, 20). This is only computed for IKPLS Algorithm #1.
```

## Examples
In [examples](https://github.com/Sm00thix/IKPLS/tree/main/examples) you will find:
* [Example](https://github.com/Sm00thix/IKPLS/tree/main/examples/fit_predict_numpy.py) of fitting and predicting with the NumPy implementations.
* [Example](https://github.com/Sm00thix/IKPLS/tree/main/examples/fit_predict_jax.py) of fitting and predicting with the JAX implementations.
* [Example](https://github.com/Sm00thix/IKPLS/tree/main/examples/cross_val_numpy.py) of cross validating with the NumPy implementations.
* [Example](https://github.com/Sm00thix/IKPLS/tree/main/examples/cross_val_jax.py) of cross validating with the JAX implementations.
* [Example](https://github.com/Sm00thix/IKPLS/tree/main/examples/gradient_jax.py) of computing the gradient of a preprocessing filter with respect to the RMSE between the target value and the value predicted by PLS after fitting.