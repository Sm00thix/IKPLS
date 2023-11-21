---
title: 'Improved Kernel Partial Least Squares for Python: Fast CPU and GPU Implementations with NumPy and JAX'
tags:
  - Python
  - PLS
  - latent variables
  - multivariate statistics
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
date: 20 November 2023
bibliography: paper.bib
---

# Summary


# Statement of need

PLS (partial least squares) [@wold1966estimation] is a standard method in chemometrics. PLS can be used as a regression model, PLS-R (PLS regression), [@wold1983food] [@wold2001pls] or a classification model, PLS-DA (PLS discriminant analysis), [@barker2003partial]. PLS takes as input a matrix $\mathbf{X}$ with shape $(N, K)$ of predictor variables and a matrix $\mathbf{Y}$ with shape $(N, M)$ of response variables. PLS decomposes $\mathbf{X}$ and $\mathbf{Y}$ into $A$ latent variables (also called components), which are linear combinations of the original $\mathbf{X}$ and $\mathbf{Y}$. Choosing the optimal number of components, $A$, depends on the input data and varies from task to task. Additionally, selecting the optimal preprocessing method is challenging to assess before model validation [@rinnan2009review] [@sorensen2021nir] but is required for achieving optimal performance [@du2022quantitative]. The optimal number of components and the optimal preprocessing method are typically chosen by cross-validation. However, depending on the size of $\mathbf{X}$ and $\mathbf{Y}$, cross-validation may be very computationally expensive. This work introduces novel, fast implementations of Improved Kernel PLS Algorithm #1 and Algorithm #2 by Dayal and MacGregor [@dayal1997improved], which have previously been shown to be fast [@alin2009comparison] and numerically stable [@andersson2009comparison]. The implementations introduced in this work use NumPy [@harris2020array] and JAX [@jax2018github]. The NumPy implementations can be executed on CPUs, and the JAX implementations can be executed on CPUs, GPUs, and TPUs. The JAX implementations are also end-to-end differentiable, allowing integration into deep learning methods. This work compares the execution time of the implementations on input data of varying shapes. It reveals that choosing the implementation that best fits the data will yield orders of magnitude faster execution than the common NIPALS (Nonlinear Iterative Partial Least Squares) [@wold1966estimation] implementation of PLS, which is the one implemented by scikit-learn [@scikit-learn], a large machine learning library for Python. With the implementations introduced in this work, choosing the optimal number of components and the optimal preprocessing becomes much more feasible than previously. Indeed, derivatives of this work have previously been applied to do this precisely [@engstrom2023improving] [@engstrom2023analyzing].

# Implementations

This section covers the implementation of the algorithms. Improved Kernel PLS [@dayal1997improved] comes in two variants: Algorithm #1 and Algorithm #2. Algorithm #1 uses the input matrix $\mathbf{X}$ of shape $(N, K)$ directly while Algorithm #2 starts by computing $X^{T}X$ of shape $(K, K)$. After this initial step, the algorithms are almost identical. Thus, if $K < N$, Algorithm #2 requires less computation after this initial step. The implementations compute internal matrices $\mathbf{W}$ ($\mathbf{X}$ weights) of shape (K, A), $\mathbf{P}$ ($\mathbf{X}$ loadings) of shape (K, A), $\mathbf{Q}$ ($\mathbf{Y}$ loadings) of shape (M, A), $\mathbf{R}$ ($\mathbf{X}$ rotations) of shape (K, A) and a tensor $\mathbf{B}$ (regression coefficients) of shape (A, K, M). Algorithm #1 also computes $\mathbf{T}$ ($\mathbf{X}$ scores) of shape $(N, A)$.

The implementations introduced in this work have been tested for equivalency against NIPALS from scikit-learn using a dataset of NIR (near-infrared) spectra from [@dreier2022hyperspectral] and tests from scikit-learn's own PLS test-suite.

## NumPy

This section is for the NumPy implementations. Details about subclassing BaseEstimator so it can be used in cross_validate.

## JAX

This section is for the JAX implementations. Details about data living on GPU, preprocessing, metric functions, and end-to-end-differentiability.

# Benchmarks

This section is for the benchmark figures.

# Possible further improvements

Write about shortcut here.

# Acknowledgements

This work is part of an industrial Ph.D. project receiving funding from FOSS Analytical A/S and The Innovation Fund Denmark. Grant Number: 1044-00108B.

# References