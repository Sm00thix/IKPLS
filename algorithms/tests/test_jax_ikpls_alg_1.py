from algorithms.jax_ikpls_alg_1 import PLS
import jax.numpy as jnp
import numpy as np

def test_fit_predict():
    X = jnp.array(np.random.uniform(size=(10,10)))
    Y = jnp.array(np.random.uniform(size=(10,2)))
    p = PLS()
    p.fit(X, Y, 5)
    p.predict(X)