from ikpls.jax_ikpls_alg_1 import PLS as JAX_Alg_1
from jax import numpy as jnp
import numpy as np
import jax
from typing import Callable, Union

# Preprocessing convolution filter for which we will obtain the gradients.
@jax.jit
def apply_1d_convolution(
    X: jnp.ndarray, conv_filter: jnp.ndarray
) -> jnp.ndarray:
    # X is a matrix of shape (N, K) = (100, 100)
    # conv_filter is a vector of shape (filter_size,) = (7,)
    convolved_rows = jax.vmap(
        lambda row: jnp.convolve(row, conv_filter, "valid")
    )(X) # Shape (N, K - filter_size + 1) = (100, 94)
    return convolved_rows 

# Loss function which we want to minimize. Computes the mean squared error between the true and predicted values, averaged over all samples and targets.
@jax.jit
def mean_squared_error(Y_true: jnp.ndarray, Y_pred: jnp.ndarray) -> float:
    # Y_true is a matrix of shape (N, M) = (100, 10)
    # Y_pred is a matrix of shape (N, M) = (100, 10) or (A, N, M) = (20, 100, 10)
    e = Y_true - Y_pred # Shape (N, M) or (A, N, M)
    se = e**2 # Shape (N, M) or (A, N, M)
    mse = jnp.mean(se, axis=(-2, -1)) # Shape () or (A,)
    return mse

# Function to differentiate.
def convolve_fit_mse(
    X: jnp.ndarray, Y: jnp.ndarray, pls_alg, A: int, n_components: Union[int, None] = None
) -> Callable[[jnp.ndarray], float]:
    @jax.jit
    def helper(conv_filter):
        filtered_X = apply_1d_convolution(X, conv_filter)
        matrices = pls_alg.stateless_fit(filtered_X, Y, A) # We must use stateless_fit() because we are using JAX's autodiff.
        B = matrices[0] # Extract the regression matrix
        Y_pred = pls_alg.stateless_predict(filtered_X, B, n_components) # Predict the values.
        mse_loss = mean_squared_error(Y, Y_pred)
        return mse_loss
    return helper

if __name__ == '__main__':
    N = 100
    K = 100
    M = 10
    A = 20

    # Generate random data. Using float64 is important for numerical stability.
    jnp_X = jnp.array(np.random.uniform(size=(N, K)), dtype=jnp.float64)
    jnp_Y = jnp.array(np.random.uniform(size=(N, M)), dtype=jnp.float64)

    filter_size = 7 # Filter size for convolution
    conv_filter = jnp.array(np.random.rand(filter_size)) # Random filter

    diff_pls_alg_1 = JAX_Alg_1(reverse_differentiable=True, verbose=True)
    
    # Compute values and gradients for the conv_filter using mean squared error as the loss function and IKPLS Algorithm #1 as the PLS algorithm with exactly 20 components. The gradient is computed using backwards mode differentiation.
    grad_fun = jax.grad(
        convolve_fit_mse(jnp_X, jnp_Y, diff_pls_alg_1, A=A, n_components=A), argnums=0
    )
    # Compute the gradient of the mean_squared_error of Y_true and Y_pred (with 20 components) with respect to the weights of the convolution filter.
    grad_alg_1 = grad_fun(conv_filter)

    """
    Compute gradients (Jacobian matrix) for the conv_filter using mean squared error as the loss function and IKPLS Algorithm #1 as the PLS algorithm with all number of components from 1 to 20.
    The gradients are computed using forwards mode differentiation. We could also use backwards mode differentiation (jacrev). But forward mode differentiation is faster when the Jacobian is tall.
    """
    jac_fun = jax.jacfwd(convolve_fit_mse(jnp_X, jnp_Y, diff_pls_alg_1, A=A, n_components=None), argnums=0)
    # Compute the gradient of the mean_squared_error of Y_true and Y_pred (from 1 to 20 components) with respect to the weights of the convolution filter.
    jac_alg_1 = jac_fun(conv_filter)
    np.allclose(jac_alg_1[A], grad_alg_1, atol=0) # True