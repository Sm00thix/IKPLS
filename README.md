# IKPLS
Fast CPU and GPU Python implementations of Improved Kernel PLS by Dayal and MacGregor (1997).

## Pre-requisites
The JAX implementations support running on both CPU and GPU. To use the GPU, follow the instructions from the [JAX Installation Guide](https://jax.readthedocs.io/en/latest/installation.html).
To ensure that JAX implementations use Float64, set the environment variable JAX_ENABLE_X64=True as per the [Current Gotchas](https://github.com/google/jax#current-gotchas).