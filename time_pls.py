import argparse

from ikpls.jax_ikpls_alg_1 import PLS as JAX_PLS_Alg_1
from ikpls.jax_ikpls_alg_2 import PLS as JAX_PLS_Alg_2
from ikpls.numpy_ikpls import PLS as NP_PLS
from ikpls.fast_cross_validation.numpy_ikpls import PLS as NP_PLS_FCV
from timings.timings import (
    SK_PLS_All_Components,
    cross_val_cpu_pls,
    fast_cross_val_cpu_pls,
    cross_val_gpu_pls,
    gen_random_data,
    single_fit_cpu_pls,
    single_fit_gpu_pls,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model",
        type=str,
        help="Model to use. Must be either 'sk', 'np1', 'np2', 'jax1', 'jax2', 'diffjax1', 'diffjax2'.",
    )
    parser.add_argument("-n_components", type=int, help="Number of components to use.")
    parser.add_argument(
        "-n_splits", type=int, help="Number of splits to use in cross-validation."
    )
    parser.add_argument(
        "-n_jobs",
        type=int,
        help="Number of parallel jobs to use. Only used for CPU implementations. A value of -1 will use all available cores.",
    )
    parser.add_argument("-n", type=int, help="Number of samples to generate.")
    parser.add_argument("-k", type=int, help="Number of features to generate.")
    parser.add_argument("-m", type=int, help="Number of targets to generate.")
    args = parser.parse_args()
    config = vars(args)
    model = config["model"]
    n_components = config["n_components"]
    n_splits = config["n_splits"]
    n = config["n"]
    k = config["k"]
    m = config["m"]

    X, Y = gen_random_data(n, k, m)
    if "jax" in model:
        if model == "jax1":
            pls = JAX_PLS_Alg_1()
            name = "JAX Improved Kernel PLS Algorithm #1"
        elif model == "jax2":
            pls = JAX_PLS_Alg_2()
            name = "JAX Improved Kernel PLS Algorithm #2"
        elif model == "diffjax1":
            pls = JAX_PLS_Alg_1(reverse_differentiable=True)
            name = (
                "JAX Improved Kernel PLS Algorithm #1 (backwards mode differentiable)"
            )
        elif model == "diffjax2":
            pls = JAX_PLS_Alg_2(reverse_differentiable=True)
            name = (
                "JAX Improved Kernel PLS Algorithm #2 (backwards mode differentiable)"
            )
        if n_splits == 1:
            print(
                f"Fitting {name} with {n_components} components on {n} samples with {k} features and {m} targets."
            )
            time = single_fit_gpu_pls(pls, X, Y, n_components)
        else:
            print(
                f"Fitting {name} with {n_components} components using {n_splits}-fold cross-validation on {n} samples with {k} features and {m} targets."
            )
            time = cross_val_gpu_pls(
                pls, X, Y, n_components, n_splits, show_progress=True
            )
        print(f"Time: {time}")
    else:
        n_jobs = config["n_jobs"]
        if model == "sk":
            pls = SK_PLS_All_Components(n_components=n_components)
            fit_params = {} if n_splits == 1 else None
            name = "scikit-learn NIPALS"
        elif model == "np1":
            pls = NP_PLS(algorithm=1)
            fit_params = {"A": n_components}
            name = "NumPy Improved Kernel PLS Algorithm #1"
        elif model == "np2":
            pls = NP_PLS(algorithm=2)
            fit_params = {"A": n_components}
            name = "NumPy Improved Kernel PLS Algorithm #2"
        elif model == "fastnp1":
            pls = NP_PLS_FCV(algorithm=1)
            name = "NumPy Improved Kernel PLS Algorithm #1 (fast cross-validation)"
        elif model == "fastnp2":
            pls = NP_PLS_FCV(algorithm=2)
            name = "NumPy Improved Kernel PLS Algorithm #2 (fast cross-validation)"
        else:
            raise ValueError(
                f"Unknown model: {model}. Must be one of 'sk', 'np1', 'np2', 'fastnp1', 'fastnp2', 'jax1', 'jax2', 'diffjax1', 'diffjax2'."
            )

        if n_splits == 1:
            print(
                f"Fitting {name} with {n_components} components on {n} samples with {k} features and {m} targets."
            )
            time = single_fit_cpu_pls(pls, X, Y, fit_params)
        else:
            print(
                f"Fitting {name} with {n_components} components using {n_splits}-fold cross-validation on {n} samples with {k} features and {m} targets. Using {n_jobs} concurrent workers."
            )
            if model.startswith("fast"):
                time = fast_cross_val_cpu_pls(
                    pls, X, Y, n_components, n_splits=n_splits, n_jobs=n_jobs, verbose=1
                )
            else:
                time = cross_val_cpu_pls(
                    pls, X, Y, n_splits, fit_params, n_jobs=n_jobs, verbose=1
                )
        print(f"Time: {time}")

    try:
        with open("timings/timings.csv", "x") as f:
            f.write("model,n_components,n_splits,n,k,m,time\n")
            f.write(f"{model},{n_components},{n_splits},{n},{k},{m},{time}\n")
    except FileExistsError:
        with open("timings/timings.csv", "a") as f:
            f.write(f"{model},{n_components},{n_splits},{n},{k},{m},{time}\n")
