# This README describes how to generate benchmark figures.
- Benchmarking a single fit will report the time to fit a single PLS model.
- Benchmarking a cross-validation will report the total time taken to:
  - Fit the model on every training partition.
  - Compute the mean squared error (MSE) for every PLS component on every validation partition.

## To reproduce timings/timings.png, the figure showing benchmarks in paper.md:
- Execute plot_timings.py will read the contents of timings/timings.csv to generate timings/timings.png.

## To make your own benchmarks
- Delete timings/timings.csv or empty its contents to keep your benchmarks distinct from the ones provided.
- Execute time_pls.py with arguments specifying:
  - The PLS algorithm to use.
  - The number of components to use.
  - The number of cross-validation splits to use, if any.
  - The number of parallel jobs to use.
  - The shapes of X and Y.

Execute time_pls.py with -h to get an overview of the different arguments and their meaning.
Executing time_pls.py with the above arguments will run the experiment and append the result as a line in timings/timings.csv.
After executing the desired benchmarks, execute plot_timings.py to generate timings/timings.png with your benchmark results.