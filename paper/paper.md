---
title: 'IKPLS: Improved Kernel Partial Least Squares and Fast Cross-Validation Algorithms for Python with CPU and GPU Implementations Using NumPy and JAX'
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
  - name : Martin Holm Jensen
    orcid: 0009-0002-4478-1534
    affliation: "1"
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
date: 09 January 2024
bibliography: paper.bib
---

# Summary
The `ikpls` software package introduces fast and versatile implementations of both versions of Improved Kernel Partial Least Squares (IKPLS) [@dayal1997improved]. `ikpls` also contains all three fast cross-validation algorithms presented by Engstrøm and Jensen [@engstrom2024shortcutting] combined with both IKPLS algorithms. PLS is a widely used method in chemometrics for regression (PLS-R) and classification (PLS-DA) tasks, requiring selecting optimal components and preprocessing methods. However, conventional implementations, such as NIPALS in scikit-learn, can be computationally expensive.

`ikpls` offers NumPy-based CPU and JAX-based CPU/GPU/TPU implementations. The JAX implementations are also differentiable, allowing seamless integration with deep learning techniques. This versatility enables users to handle diverse data dimensions efficiently.

Benchmarks demonstrate the superior performance of `ikpls` compared to scikit-learn's NIPALS across various scenarios, emphasizing the importance of choosing the proper implementation for specific data characteristics. Additionally, the software addresses the redundant structure in cross-validation, proposing an algorithmic improvement for substantial speedup without recomputing total matrix products.

In conclusion, `ikpls` empowers researchers and practitioners in chemometrics and related fields with efficient, scalable, and end-to-end differentiable tools for PLS modeling, facilitating optimal component selection and preprocessing decisions by offering implementations of

1. both variants of IKPLS for CPUs;
2. both variants of IKPLS for GPUs, both of which are end-to-end differentiable, allowing integration with deep learning models;
3. a new algorithm for cross-validation that yields a substantial speedup if the training set is larger than the validation set.

# Statement of need

PLS (partial least squares) [@wold1966estimation] is a standard method in chemometrics. PLS can be used as a regression model, PLS-R (PLS regression), [@wold1983food] [@wold2001pls] or a classification model, PLS-DA (PLS discriminant analysis), [@barker2003partial]. PLS takes as input a matrix $\mathbf{X}$ with dimension $(N, K)$ of predictor variables and a matrix $\mathbf{Y}$ with dimension $(N, M)$ of response variables. PLS decomposes $\mathbf{X}$ and $\mathbf{Y}$ into $A$ latent variables (also called components), which are linear combinations of the original $\mathbf{X}$ and $\mathbf{Y}$. Choosing the optimal number of components, $A$, depends on the input data and varies from task to task. Additionally, selecting the optimal preprocessing method is challenging to assess before model validation [@rinnan2009review] [@sorensen2021nir] but is required for achieving optimal performance [@du2022quantitative]. The optimal number of components and the optimal preprocessing method are typically chosen by cross-validation. However, depending on the size of $\mathbf{X}$ and $\mathbf{Y}$, cross-validation may be very computationally expensive. The implementations of the fast cross-validation algorithms [@engstrom2024shortcutting] will significantly reduce the computational cost of cross-validation.

This work introduces the Python software package, `ikpls`, with novel, fast implementations of Improved Kernel PLS (IKPLS) Algorithm #1 and Algorithm #2 by Dayal and MacGregor [@dayal1997improved], which have previously been compared with other PLS algorithms and shown to be fast [@alin2009comparison] and numerically stable [@andersson2009comparison]. The implementations introduced in this work use NumPy [@harris2020array] and JAX [@jax2018github]. The NumPy implementations can be executed on CPUs, and the JAX implementations can be executed on CPUs, GPUs, and TPUs. The JAX implementations are also end-to-end differentiable, allowing integration into deep learning methods. This work compares the execution time of the implementations on input data of varying dimensions. It reveals that choosing the implementation that best fits the data will yield orders of magnitude faster execution than the common NIPALS (Nonlinear Iterative Partial Least Squares) [@wold1966estimation] implementation of PLS, which is the one implemented by scikit-learn [@scikit-learn], a large machine learning library for Python. With the implementations introduced in this work, choosing the optimal number of components and the optimal preprocessing becomes much more feasible than previously. Indeed, derivatives of this work have previously been applied to do this precisely [@engstrom2023improving] [@engstrom2023analyzing].

Other implementations of other PLS algorithms with NumPy and scikit-learn exist, even for more specialized tasks such as multiblock PLS [@baum2019multiblock]. These implementations, however, are not as fast as IKPLS [@alin2009comparison]. Implementations of IKPLS exist in R and MATLAB. Indeed, the original implementation was given in MATLAB [@dayal1997improved]. To the best of the authors' knowledge, however, there are no Python implementations of IKPLS that simultaneously correctly handle all possible dimensions of $\mathbf{X}$ and $\mathbf{Y}$. To the best of the authors' knowledge, no other PLS algorithms exist in JAX, nor do implementations of IKPLS in other frameworks with automatic differentiation.

# Implementations

Improved Kernel PLS [@dayal1997improved] comes in two variants: Algorithm #1 and Algorithm #2. The implementations compute internal matrices $\mathbf{W}$ ($\mathbf{X}$ weights) of dimension (K, A), $\mathbf{P}$ ($\mathbf{X}$ loadings) of dimension (K, A), $\mathbf{Q}$ ($\mathbf{Y}$ loadings) of dimension (M, A), $\mathbf{R}$ ($\mathbf{X}$ rotations) of dimension (K, A) and a tensor $\mathbf{B}$ (regression coefficients) of dimension (A, K, M). Algorithm #1 also computes $\mathbf{T}$ ($\mathbf{X}$ scores) of dimension $(N, A)$.

The implementations introduced in this work have been tested for equivalency against NIPALS from scikit-learn using a dataset of NIR (near-infrared) spectra from [@dreier2022hyperspectral] and tests from scikit-learn's own PLS test-suite.

The authors provide [examples](https://github.com/Sm00thix/IKPLS/blob/main/examples/) for all the core functionality, including use-cases for fitting, predicting, cross-validating with custom metrics and custom preprocessing on CPU and GPU, and propagating gradients through the PLS fitting subsequent prediction.

## NumPy

A Python class implementing both IKPLS algorithms using NumPy is available in `ikpls`. The class subclasses [`BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html) from scikit-learn, allowing the IKPLS class to be used in combination with e.g., scikit-learn's [`cross_validate`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html) for a simple interface to parallel cross-validation with user-defined metric functions.

A Python class implementing both IKPLS algorithms with our algorithmic improvement for cross-validation is also available in `ikpls`. This class does not subclass any other class; instead, it relies on its own parallel cross-validation scheme, which is fast and memory efficient.

## JAX

A Python class for each IKPLS algorithm using JAX is available in `ikpls`. JAX combines Autograd [@maclaurin2015autograd] with [XLA (Accelerated Linear Algebra)](https://www.tensorflow.org/xla). Using XLA, JAX compiles instructions and optimizes them for high-performance computation, allowing execution on CPUs, GPUs, and TPUs.

Using Autograd, JAX enables automatic differentiation of the IKPLS algorithms. JAX supports both forward and backward mode differentiation. The implementations in this work are naturally compatible with forward mode differentiation. This work also offers implementations that are compatible with backward mode differentiation. While mathematically equivalent to the other implementations, the backward mode differentiation compatible implementations incur a slight increase in computation time that increases with the number of components, $A$. The differentiation, either backward or forward, allows users to combine PLS with deep learning techniques. Additionally, the implementations allow for evaluating user-defined metric functions. JAX ensures these functions are compiled and executed on-device, enabling maximum utilization of the massively parallel hardware.

# Benchmarks

The `ikpls` implementations are benchmarked against scikit-learn's NIPALS implementation. The comparisons are made with varying dimensions for $\mathbf{X}$ and $\mathbf{Y}$ and variable number of components, $A$. Additionally, the comparisons are made using a single fit and leave-one-out cross-validation. To estimate the execution time in a realistic scenario, the mean squared error and the number of components that minimize this error are computed during cross-validation and returned hereafter for subsequent analysis by the user. The execution times reported for the JAX implementations include the time spent compiling the instructions and sending the input data to the GPU on which the cross-validation is computed. While some sources on the internet claim that JAX is faster than NumPy on CPU, we did not find that to be the case for our scenarios. Therefore, we exclusively test the JAX implementations on GPU. When cross-validating with the NumPy CPU implementations and scikit-learn's NIPALS, we use 32 parallel jobs corresponding to one for each available thread on the CPU that we used. We use only eight parallel jobs for the CPU experiments when performing cross-validation with 1 million samples due to memory consumption increased by scikit-learn's cross_validate. In practice, however, we had 100% CPU utilization using eight jobs for such a large dataset due to the parallel nature of NumPy operations. IKPLS #2 (fast cross-validation) is more memory efficient when $N > K$, as only the $\mathbf{X}^\mathbf{T}\mathbf{X}$ matrix of dimensions $K \times K = 500 \times 500$ had to be transferred to each parallel job, allowing 32 parallel jobs without exhausting the memory resources.

The benchmarks use randomly generated data. The random seed is fixed to give all implementations the same random data. The default parameters for the benchmarks are $N=10,000$, $K=500$, and $A=30$. We benchmark using a single target variable $M=1$ and multiple target variables with $M=10$. PLS with $M=1$ is commonly referred to as PLS1 and PLS2 with $M>1$.

In an attempt to give guidelines for algorithm choice for the most common use cases, we report the execution time of the implementations with varying values for each of the parameters above. Specifically, we define a list of values for each parameter to take while the rest of the parameters maintain their default settings. We use 

$N \in [10^1, 10^2, 10^3, 10^4, 10^5, 10^6]$, $K \in [30, 50, 10^2, 5\cdot 10^2, 10^3, 5\cdot 10^3, 10^4]$, $A \in [10, 20, 30, 50, 100, 200, 500]$, and $M \in [1, 10]$.

All the experiments are executed on the hardware shown in \autoref{tab:hardware} on a machine running Ubuntu 22.04 Jammy Jellyfish.

: Hardware used in the execution time experiments. \label{tab:hardware}

| Component   | Name                                 |
|-------------|--------------------------------------|
| Motherboard | ASUS PRO WS X570-ACE                 |
| CPU         | AMD Ryzen 9 5950X                    |
| CPU Cooler  | NZXT Kraken X73                      |
| GPU         | NVIDIA GeForce RTX3090 Ti, CUDA 11.8 |
| RAM         | 4x32GB, DDR4, 3.2GHz, C16            |

![Results of our timing experiments. We vary $N$, $K$, and $A$ in the first, second, and third columns. The first two rows are PLS1. The last two rows are PLS2. The first and third rows are single-fit. The second and fourth rows are leave-one-out cross-validation, computing the mean squared error and best number of components for each validation split. A circle indicates that the experiment was run until the end, and the time reported is exact. A square means that the experiment was run until the time per iteration had stabilized and used to forecast the time usage if the experiment was run to completion.\label{fig:timings}](timings.png)

Based on the results in \autoref{fig:timings}, we will give general guidelines for choosing an algorithm. The choice of CPU versus GPU may vary depending on the specific hardware a user has available. The guidelines apply to both PLS1 and PLS2.

If the user needs only to perform a single fit (i.e., no cross-validation), the user should opt for either of the CPU IKPLS implementations. In this case, if $N \gg K$, choose IKPLS #2 over IKPLS #1. The user should consider using a GPU only if $N > 10^6$ or $K > 10^4$. While the GPU generally scales better than the CPU with increasing $N$ and $K$, it scales worse with increasing $A$. Even for a single fit, the implementations of IKPLS are typically one to two orders of magnitude faster than scikit-learn's NIPALS.

In the case of having to perform cross-validation, the user should also always choose one of the implementations of IKPLS. In the most extreme case in our experiments, the CPU implementation of IKPLS #2 using fast cross-validation is a staggering six orders of magnitude (1 million) times faster than scikit-learn's NIPALS. However, even in more typical scenarios, the best choice of the IKPLS algorithm and implementation is typically several orders of magnitude faster than scikit-learn's NIPALS.

In the case of cross-validation, if the validation split size is larger than the training split size, the user should opt for IKPLS #2 using fast cross-validation unless $K > 10^3$ in which case the user should opt for IKPLS #1 on the GPU. If $N < 10^5$ and $N < K$, the user should opt for IKPLS #1 on the CPU, and if the validation splits are smaller than the training splits, fast cross-validation should be used.

The user should consider another algorithm only when applying preprocessing techniques incompatible with fast cross-validation (described in the next section). The CPU implementation of IKPLS #2 is a suitable choice unless $N > 10^4$ or $K > 10^3$, in which case GPU IKPLS #1 is the fastest.

As a final note, fast cross-validation offers a speedup, which is significant for IKPLS #2 but less profound for IKPLS #1. If $M$ is large, the speedup for IKPLS #1 will be more profound.

# Acknowledgements

This work is part of an industrial Ph.D. project receiving funding from FOSS Analytical A/S and The Innovation Fund Denmark. Grant Number: 1044-00108B.

# References