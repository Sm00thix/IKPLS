# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
# ---

# %% [markdown]
# # how to reproduce the results in the paper
#
# **NOTE** you may need to `pip install jupytext` in order to open this file as a notebook in Jupyter
#
# this Jupyter notebook uses the file `timings/timings.csv`, as shipped with the paper repo  
# it allows to reproduce the same runs locally, and to gather your own performance measurements
#
# it stores its own results in a file named `timings/user_timings.csv`  
# these are used as a cache - i.e. not re-run if already present; you can simply delete or move this file if you want to re-start from scratch

# %%
# this one must not change
REF_TIMINGS = "timings/timings.csv"

# default - see next cell for how to change it
OUR_TIMINGS = "timings/user_timings.csv"

# %%
# provide a way to choose the output from the command line
# but argparse won't work from Jupyter, so:

try:
    # this is defined in a Jupyter / IPython environment
    # in this case just change OUR_TIMINGS above
    get_ipython()
except:
    # not in IPython/notebook - so we run from the command line
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", default=OUR_TIMINGS, help="filename for the csv output")
    args = parser.parse_args()
    OUR_TIMINGS = args.output

print(f"using {OUR_TIMINGS=}")

# %% [markdown]
# ## loading the paper timings
#
# to get a list of what would need to be done

# %%
import pandas as pd

# %%
paper = pd.read_csv(REF_TIMINGS)
paper.shape

# %%
paper.head()

# %% [markdown]
# ### dropping the njobs column
#
# we won't need this

# %%
paper.drop(columns=['njobs'], inplace=True)
paper.shape

# %% [markdown]
# ## loading previous runs
#
# to avoid re-running them if already done

# %%
try:
    previous = (
        pd.read_csv(OUR_TIMINGS)
        .rename(columns={'time': 'previous'})
        # this is unchanged between focus and previous
        # so avoid duplication in the merge below
        .drop(columns=['inferred'])
    )
except:
    previous = pd.DataFrame()

if len(previous):
    print(f"we have {len(previous)} previous runs already")
else:
    print(f"restarting from scratch")

# %% [markdown]
# ## running again
#
# focus is a selection of the runs to reproduce - e.g. we will use it optionnally later on if we can only run on a CPU

# %%
focus = paper.copy()

def status():
    print(f"we are focusing on {len(focus)} runs")

status()

# %% [markdown]
# ### isolating lines doable on a CPU (optional)

# %%
focus = focus[ ~ focus.model.str.contains('jax')]
status()

# %% [markdown]
# ### compute missing runs: merge both tables
#
# and ignore entries with a previous time

# %%
# what to join on
KEYS = "model|n_components|n_splits|n|k|m".split("|")

if not len(previous):
    missing = focus
else:
    join = pd.merge(focus, previous, on=KEYS, how='left')
    missing = join[join.previous.isna()]


# %%
missing

# %% [markdown]
# ### ignore inferred entries

# %%
inferred = missing[missing.inferred == True]
f"there were {len(inferred)} runs in the data"

# %%
missing = missing[~ (missing.inferred == True)]

# %%
f"we still have {len(missing)} runs to carry out"

# %% [markdown]
# ### go

# %%
import os
import numpy as np

for index, t in enumerate(missing.itertuples()):
    if t.time > 30:
        print(f"skipping run ({t.model} x {t.n_components} x {t.n_splits}) that has {t.time=} > 30")
        continue
    # print(t)
    estimate = "" if np.isnan(t.inferred) or not t.inferred else " --estimate"
    command = (f"python3 time_pls.py -o {OUR_TIMINGS}"
               f" -model {t.model} -n_components {t.n_components}" 
               f" -n_splits {t.n_splits} -n {t.n} -k{t.k} -m {t.m} -n_jobs -1{estimate}")
    print(command)
    os.system(command)
