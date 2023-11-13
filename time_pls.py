import argparse
from timings.timings import (
    single_fit_cpu_pls,
    single_fit_gpu_pls,
    cross_val_cpu_pls,
    cross_val_gpu_pls,
    gen_random_data,
    SK_PLS_All_Components,
)
from ikpls.numpy_ikpls import PLS as NP_PLS
from ikpls.jax_ikpls_alg_1 import PLS as JAX_PLS_Alg_1
from ikpls.jax_ikpls_alg_2 import PLS as JAX_PLS_Alg_2

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
    parser.add_argument("-n_jobs", type=int, help="Number of parallel jobs to use. Only used for CPU implementations. A value of -1 will use all available cores.")
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
            time = cross_val_gpu_pls(pls, X, Y, n_components, n_splits, show_progress=True)
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
        else:
            raise ValueError(
                f"Unknown model: {model}. Must be one of 'sk', 'np1', 'np2', 'jax1', 'jax2', 'diffjax1', 'diffjax2'."
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
            time = cross_val_cpu_pls(pls, X, Y, n_splits, fit_params, n_jobs=n_jobs, verbose=1)
        print(f"Time: {time}")

    try:
        with open("timings/timings.csv", "x") as f:
            f.write("model,n_components,n_splits,n,m,k,time\n")
            f.write(f"{model},{n_components},{n_splits},{n},{k},{m},{time}\n")
    except FileExistsError:
        with open("timings/timings.csv", "a") as f:
            f.write(f"{model},{n_components},{n_splits},{n},{k},{m},{time}\n")

    # Freeze values:
    # 1. n_components = 30
    # 2. n_splits = {1, LOOCV} # The overhead of JIT-compilation is already negligible at 1e4 samples.
    # 3. n = 10000
    # 4. k = 500
    # 5. m = {1, 10}

    # Dynamic values:
    # 1. n_components = {10, 20, 30, 50, 100, 200, 500, 1000}
    # 1. n = {1e1, 1e2, 1e3, 1e4, 1e5, 1e6}
    # 3. k = {20, 50, 100, 500, 1000, 5000, 10000}

    # TODO: Save timings to file
    # Format should be a csv with something like: model, n_components, n_splits, n, m, k, time

    # TODO: Add opportunity to run only the first max_splits number of splits in cross-validation. Running them all to completion with all values will take a long time. Simply setting n_splits to a lower value will not work because the splits will have wrong sizes.
