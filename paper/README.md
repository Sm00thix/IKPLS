# This README describes how to generate benchmark figures.
- Benchmarking a single fit will report the time to fit a single PLS model.
- Benchmarking a cross-validation will report the total time taken to:
  - Fit the model on every training partition.
  - Compute the mean squared error (MSE) for every PLS component on every validation partition.

## To reproduce timings/timings.png, the figure showing benchmarks in paper.md:
  - Execute the following command in your terminal:

    ```bash
    python3 plot_timings.py
    ```

    This will read the contents of timings/timings.csv to generate timings/timings.png.

## To make your own benchmarks
  - Delete timings/timings.csv or empty its contents to keep your benchmarks distinct from the ones provided.
  - Execute time_pls.py with arguments specifying:
    - The PLS algorithm to use.
    - The number of components to use.
    - The number of cross-validation splits to use, if any.
    - The number of parallel jobs to use.
    - The shapes of X and Y.
  - For example, to benchmark the fast cross-validation algorithm with the NumPy implementation of IKPLS algorithm #2 using leave-one-out cross-validation with
  1 million samples, 500 features, 10 targets, 30 PLS components, using all available CPU cores for parallel cross-validation, execute the following command in your terminal:

    ```bash
    python3 time_pls.py -model fastnp2 -n 1000000 -k 500 -m 10 -n_components 30 -n_splits 1000000 -n_jobs -1
    ```

    This will run the experiment and append the result as a line in timings/timings.csv.
  - After executing the desired benchmarks, execute plot_timings.py to generate timings/timings.png with your benchmark results.
  - Once a benchmark has finished, you can safely run another, the result of which will be appended to timings/timings.csv
  - Execute the following command in your terminal to get an overview and description of the different arguments and their meaning:

    ```bash
    python3 time_pls.py -h
    ```
