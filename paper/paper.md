---
title: 'Improved Kernel Partial Least Squares for Python: Fast CPU and GPU Implementations with NumPy and JAX'
tags:
  - Python
  - PLS
  - latent variables
  - multivariate statistics
  - cross-validation
  - deep learning
authors:
  - name: Ole-Christian Galbo Engstrøm
    orcid: 0000-0002-7906-4589
    affiliation: "1, 2, 3"
    corresponding: true
  - name: Erik Schou Dreier
    orcid: 0000-0001-9784-7504
    affiliation: "1"
  - name: Birthe Møller Jespersen
    orcid: 0000-0002-8695-1450
    affiliation: "4"
  - name: Kim Steenstrup Pedersen
    orcid: 0000-0003-3713-0960
    affiliation: "2, 5"
affiliations:
  - name: FOSS Analytical A/S, Denmark
    index: 1
  - name: Department of Computer Science (DIKU), University of Copenhagen, Denmark
    index: 2
  - name: Department of Food Science (UCPH FOOD), University of Copenhagen, Denmark
    index: 3
  - name: UCL University College, Denmark
    index: 4
  - name: Natural History Museum of Denmark (NHMD), University of Copenhagen, Denmark
    index: 5
date: 24 November 2023
bibliography: paper.bib
---

# Summary


# Statement of need

PLS (partial least squares) [@wold1966estimation] is a standard method in chemometrics. PLS can be used as a regression model, PLS-R (PLS regression), [@wold1983food] [@wold2001pls] or a classification model, PLS-DA (PLS discriminant analysis), [@barker2003partial]. PLS takes as input a matrix $\mathbf{X}$ with shape $(N, K)$ of predictor variables and a matrix $\mathbf{Y}$ with shape $(N, M)$ of response variables. PLS decomposes $\mathbf{X}$ and $\mathbf{Y}$ into $A$ latent variables (also called components), which are linear combinations of the original $\mathbf{X}$ and $\mathbf{Y}$. Choosing the optimal number of components, $A$, depends on the input data and varies from task to task. Additionally, selecting the optimal preprocessing method is challenging to assess before model validation [@rinnan2009review] [@sorensen2021nir] but is required for achieving optimal performance [@du2022quantitative]. The optimal number of components and the optimal preprocessing method are typically chosen by cross-validation. However, depending on the size of $\mathbf{X}$ and $\mathbf{Y}$, cross-validation may be very computationally expensive.

This work introduces the Python software package, `ikpls`, with novel, fast implementations of Improved Kernel PLS (IKPLS) Algorithm #1 and Algorithm #2 by Dayal and MacGregor [@dayal1997improved], which have previously been shown to be fast [@alin2009comparison] and numerically stable [@andersson2009comparison]. The implementations introduced in this work use NumPy [@harris2020array] and JAX [@jax2018github]. The NumPy implementations can be executed on CPUs, and the JAX implementations can be executed on CPUs, GPUs, and TPUs. The JAX implementations are also end-to-end differentiable, allowing integration into deep learning methods. This work compares the execution time of the implementations on input data of varying shapes. It reveals that choosing the implementation that best fits the data will yield orders of magnitude faster execution than the common NIPALS (Nonlinear Iterative Partial Least Squares) [@wold1966estimation] implementation of PLS, which is the one implemented by scikit-learn [@scikit-learn], a large machine learning library for Python. With the implementations introduced in this work, choosing the optimal number of components and the optimal preprocessing becomes much more feasible than previously. Indeed, derivatives of this work have previously been applied to do this precisely [@engstrom2023improving] [@engstrom2023analyzing].

# Implementations

This section covers the implementation of the algorithms. Improved Kernel PLS [@dayal1997improved] comes in two variants: Algorithm #1 and Algorithm #2. Algorithm #1 uses the input matrix $\mathbf{X}$ of shape $(N, K)$ directly while Algorithm #2 starts by computing $X^{T}X$ of shape $(K, K)$. After this initial step, the algorithms are almost identical. Thus, if $K < N$, Algorithm #2 requires less computation after this initial step. The implementations compute internal matrices $\mathbf{W}$ ($\mathbf{X}$ weights) of shape (K, A), $\mathbf{P}$ ($\mathbf{X}$ loadings) of shape (K, A), $\mathbf{Q}$ ($\mathbf{Y}$ loadings) of shape (M, A), $\mathbf{R}$ ($\mathbf{X}$ rotations) of shape (K, A) and a tensor $\mathbf{B}$ (regression coefficients) of shape (A, K, M). Algorithm #1 also computes $\mathbf{T}$ ($\mathbf{X}$ scores) of shape $(N, A)$.

The implementations introduced in this work have been tested for equivalency against NIPALS from scikit-learn using a dataset of NIR (near-infrared) spectra from [@dreier2022hyperspectral] and tests from scikit-learn's own PLS test-suite.

## NumPy

A Python class implementing both IKPLS algorithms using NumPy is available in `ikpls` as `ikpls.numpy_ikpls.PLS`. Pass in `algorithm=1` or `algorithm=2` in the constructor to choose between algorithms. The class subclasses [`BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html) from scikit-learn, allowing the IKPLS class to be used in combination with e.g., scikit-learn's [`cross_validate`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html) for a simple interface to parallel cross-validation with user-defined metric functions.

## JAX

A Python class for each IKPLS algorithm using JAX is available in `ikpls` as `ikpls.jax_ikpls_alg_1.PLS` and `ikpls.jax_ikpls_alg_2.PLS`. JAX combines Autograd [@maclaurin2015autograd] with [XLA (Accelerated Linear Algebra)](https://www.tensorflow.org/xla). Using XLA, JAX compiles instructions and optimizes them for high-performance computation, allowing execution on CPUs, GPUs, and TPUs.

Using Autograd, JAX enables automatic differentiation of the IKPLS algorithms. JAX supports both forward and backward mode differentiation. The implementations in this work are naturally compatible with forward mode differentiation. This work also offers implementations that are compatible with backward mode differentiation. While mathematically equivalent to the other implementations, the backward mode differentiation compatible implementations incur a slight increase in computation time that increases with the number of components, $A$. Therefore, the backward mode differentiable algorithms should only be used if their specific functionality is desired. The differentiation, either backward or forward, allows users to combine PLS with deep learning techniques. For example, the input data may be preprocessed with a convolution filter. The differentiation allows computing the gradient of the convolution filter with respect to a loss function that measures the discrepancy between the PLS prediction and some ground truth value. [This](https://github.com/Sm00thix/IKPLS/blob/main/examples/gradient_jax.py) is an example of such a use case.

Fitting a PLS model consists exclusively of matrix and vector operations. Therefore, the JAX implementations of IKPLS were explicitly made with the idea that massively parallel hardware, such as GPUs and TPUs, optimized for these operations could be used. To this end, custom implementations of cross-validation using JAX were made part of the IKPLS classes. In particular, this ensures that the input data only needs to be transferred to GPU/TPU memory once and will be partitioned for cross-validation segments on this device. Additionally, the implementations allow for evaluating user-defined metric functions. JAX ensures these functions are compiled and executed on-device, enabling maximum utilization of the massively parallel hardware.

# Benchmarks

This section compares the execution times of the `ikpls` implementations with scikit-learn's NIPALS implementation. The comparisons are made with varying shapes for $\mathbf{X}$ and $\mathbf{Y}$ and variable number of components, $A$. Additionally, the comparisons are made using a single fit and leave-one-out cross-validation. To estimate the execution time in a realistic scenario, the mean squared error and the number of components that minimize this error are computed during cross-validation and returned hereafter for subsequent analysis by the user. The execution times reported for the JAX implementations include the time spent compiling the instructions and sending the input data to the GPU on which the cross-validation is computed. While some sources on the internet claim that JAX is faster than NumPy on CPU, we did not find that to be the case for our scenarios. Therefore, we exclusively test the JAX implementations on GPU. When cross-validating with the NumPy CPU implementations and scikit-learn's NIPALS, we use 32 parallel jobs corresponding to one for each available thread on the CPU that we used. We use only eight parallel jobs when performing cross-validating with 1 million samples due to memory constraints. In practice, however, we had 100% CPU utilization using eight jobs for such a large dataset due to the parallel nature of NumPy operations.

The benchmarks use randomly generated data. The random seed is fixed to give all implementations the same random data. The default parameters for the benchmarks are $N=10,000$, $K=500$, and $A=30$. We benchmark using a single target variable $M=1$ and multiple target variables with $M=10$. PLS with $M=1$ is commonly referred to as PLS1 and PLS2 with $M>1$

In an attempt to give guidelines for algorithm choice for the most common use cases, we report the execution time of the implementations with varying values for each of the parameters above. Specifically, we define a list of values for each parameter to take while the rest of the parameters maintain their default settings. We use 

$N \in [10^1, 10^2, 10^3, 10^4, 10^5, 10^6]$, $K \in [30, 50, 10^2, 5\cdot 10^2, 10^3, 5\cdot 10^3, 10^4]$, $A \in [10, 20, 30, 50, 100, 200, 500]$, and $M \in [1, 10]$.

All the experiments are executed on the hardware shown in \autoref{tab:hardware} on a machine running Ubuntu 22.04 Jammy Jellyfish.

: Hardware used in the execution time experiments. \label{tab:hardware}

| Component   | Name                      |
|-------------|---------------------------|
| Motherboard | ASUS PRO WS X570-ACE      |
| CPU         | AMD Ryzen 9 5950X         |
| CPU Cooler  | NZXT Kraken X73           |
| GPU         | NVIDIA GeForce RTX3090 Ti |
| RAM         | 4x32GB, DDR4, 3.2GHz, C16 |

![Timings. A circle indicates that the experiment was run until the end and the time reported is exact. A square indicates that the experiment was run until the time per iterations had stabilized and used to forecast the time usage if the experiment was run to completion.\label{fig:timings}](timings.png)

# Algorithmic improvement for cross-validation

Cross-validating PLS algorithms has an inherent redundant structure. Each cross-validation iteration invovles operations on subsets of $\mathbf{X}$ (Algorithm #1) or $\mathbf{X^{T}X}$ (Algorithm #2) and $\mathbf{X^{T}Y}$ that typically have a large overlap with subsets from other iterations. Here, we provide some insight into how to avoid redundant operations yielding a dramatic speedup in cross-validation. These insights are inspired by related insights from [@stefansson2019orders] and [@liland2020much]. They show how to achieve a similar speedup for feature selection with algorithms using $\mathbf{X^{T}X}$ and cross-validation with algorithms using $\mathbf{XX^{T}}$, respectively.

We propose an algorithmic improvement that consists of combining the training and validation data to compute the full $\mathbf{X^{T}X}$ and $\mathbf{X^{T}Y}$ only once in the beginning and then subtract outer vector products from these during each cross-validation iteration, avoiding recomputation of the full matrix products.

Using IKPLS Algorithm 2 (for Algorithm 1, ignore any terms related to $\mathbf{X^{T}X}$), the improved algorithm for cross-validation is defined by the following pseudo-code:

1. Compute $\mathbf{X^{T}X}$ and $\mathbf{X^{T}Y}$. Set cross-validation split $i=0$

2. Let $V_{i}$ denote the set of validation indices for cross-validation split $i$, i.e., the samples that should be used for validation instead of training.

3. Copy $\mathbf{X^{T}X}$ and $\mathbf{X^{T}Y}$ into new matrices $\mathbf{(X^{T}X)^{i}}$ and $\mathbf{(X^{T}Y)^{i}}$.

4. We must remove the contribution of all samples, $\mathbf{X}_{j}$ and $\mathbf{Y}_{j}$ for $j \in $V_{i}$.
   This can be done with the following operations  
   for $j$ in $V_{i}$:  
   &nbsp;&nbsp;$\mathbf{(X^{T}X)^{i}}$ = $\mathbf{(X^{T}X)^{i}}$ - $\mathbf{X}_{j}^{T}\mathbf{X}_{j}$  
   &nbsp;&nbsp;$\mathbf{(X^{T}Y)^{i}}$ = $\mathbf{(X^{T}Y)^{i}}$ - $\mathbf{X}_{j}^{T}\mathbf{Y}_{j}$

5. Fit PLS with $\mathbf{(X^{T}X)^{i}}$ and $\mathbf{(X^{T}Y)^{i}}$ which now only contains samples with an index not in $V_{i}$.

6. Evaluate the calibrated PLS model on the validation data as per usual:  
   for $j$ in $V_{i}$:  
    &nbsp;&nbsp;predict on the validation samples $\mathbf{X}_{j}$ and evaluate the predictions against validation targets $\mathbf{Y}_{j}$

7. Terminate if there are no more cross-validation splits. Otherwise increment the split counter: $i = i + 1$ and go to step 2.

This algorithm avoids recomputing the full $\mathbf{X^{T}X}$ and $\mathbf{X^{T}Y}$ for each cross-validation iteration which would require $N \times K^2$ and $N \times K \times M$ multiplications per cross-validation split, respectively. Instead, we compute in each cross-validation iteration $\mathbf{X}_{j}^{T}\mathbf{X}_{j}$ and $\mathbf{X}_{j}^{T}\mathbf{Y}_{j}$, requiring only $K^2$ and $K \times M$ for each $j \in V_{i}$. Thus, if $N > |V_{i}|$, the latter approach is faster. As $|V_{i}|$ can never be larger than $N$ when performing cross-validation, the proposed algorithm is always faster than recomputing the full $\mathbf{X^{T}X}$ and $\mathbf{X^{T}Y}$. The achieved speedup is proportional to the number of cross-validation splits. In the most extreme case of leave-one-out cross-validation, $|V_{i}|=1$ for all cross-validation splits $i$, and a speedup of order $N$ is achieved.  

The caveat with this algorithm, and in fact the reason for not having implemented it in the `ikpls` package, is that preprocessing methods dependent on multiple samples (such as feature centering and scaling) allow a single row in $\mathbf{X}$ to affect the full $\mathbf{X^{T}X}$ and $\mathbf{X^{T}Y}$ and a single row in $\mathbf{Y}$ to affect the full $\mathbf{X^{T}Y}$. These effects must be taken into account in step 4 of the proposed algorithm to avoid data leakage between training and validation splits. The authors conceive no easy way to take this into account but welcome any future contributions adressing this issue.

# Acknowledgements

This work is part of an industrial Ph.D. project receiving funding from FOSS Analytical A/S and The Innovation Fund Denmark. Grant Number: 1044-00108B.

# References