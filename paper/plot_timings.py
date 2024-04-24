"""
Plot the benchmarks stored in timings/timings.csv. The plot is saved as
timings/timings.png.

The timings.csv file contains the following columns:
- model: The model name.
- n_components: The number of components.
- n_splits: The number of splits in the cross-validation.
- n: The number of samples.
- k: The number of X features.
- m: The number of Y targets.
- time: The time taken to fit the model.
- inferred: Whether the time was inferred by preemptively stopping the benchmark once
    the time per iteration was stable or by running the benchmark to completion.
- n_jobs: This column is unused. It contains only a single non-empty entry which was
    manually entered. The scikit-learn implementation of NIPALS uses a lot of memory
    when cross-validating and the machine had to use 8 cores instead of 32 to avoid
    running out of memory.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.transforms import blended_transform_factory

import ikpls


def remove_rows_where_all_values_except_time_are_same(df):
    """
    Remove rows from a DataFrame where all values except the 'time' column are the same.

    Paramters
    ---------
    df : pd.DataFrame
        The input DataFrame.
        
    Returns:
        pd.DataFrame: The DataFrame with duplicate rows removed.
    """
    df = df.drop_duplicates(subset=df.columns[:-1])
    return df


def get_name_x_t_dict(df, x_name, constants_dict, single_fit_or_loocv):
    """
    Get a dictionary containing the x, t, and inferred values for each model name.

    Paramters
    ---------
    df : pd.DataFrame
        The input DataFrame.
    x_name : str
        The name of the x column.
    constants_dict : dict
        A dictionary containing the constant values for the plot.
    single_fit_or_loocv : str
        The type of fit to consider. Must be 'single_fit' or 'loocv'.

    Returns:
        dict: A dictionary containing the x, t, and inferred values for each model
        name.
    """
    model_names = ["sk", "np1", "np2", "fastnp1", "fastnp2", "jax1", "jax2"]
    for name, value in constants_dict.items():
        df = df[df[name] == value]
    name_x_t_dict = {}
    for model_name in model_names:
        sub_df = df[df["model"] == model_name]
        if single_fit_or_loocv == "single_fit":
            sub_df = sub_df[sub_df["n_splits"] == 1]
        elif single_fit_or_loocv == "loocv":
            sub_df = sub_df[sub_df["n_splits"] != 1]
        else:
            raise ValueError(
                f"single_fit_or_loocv must be 'single_fit' or 'loocv'. but got: "
                f"{single_fit_or_loocv}"
            )
        x = sub_df[x_name].values
        t = sub_df["time"].values
        inferred = sub_df["inferred"].values
        if model_name == "sk":
            model_name = "scikit-learn NIPALS (CPU)"
        elif model_name == "np1":
            model_name = "NumPy IKPLS #1 (CPU)"
        elif model_name == "np2":
            model_name = "NumPy IKPLS #2 (CPU)"
        elif model_name == "jax1":
            model_name = "JAX IKPLS #1 (GPU)"
        elif model_name == "jax2":
            model_name = "JAX IKPLS #2 (GPU)"
        elif model_name == "fastnp1":
            model_name = "NumPy IKPLS #1 (fast cross-validation)"
        elif model_name == "fastnp2":
            model_name = "NumPy IKPLS #2 (fast cross-validation)"
        name_x_t_dict[model_name] = {"x": x, "t": t, "inferred": inferred}
    return name_x_t_dict


def plot_timings(
    ax,
    name_x_t_dict,
    xlabel,
    constants_dict,
):
    """
    Plot the timings on a given axis.

    Paramters
    ---------
    ax : plt.Axes
        The axis to plot on.
    name_x_t_dict : dict
        A dictionary containing the x, t, and inferred values for each model name.
    xlabel : str
        The x-axis label.
    constants_dict : dict
        A dictionary containing the constant values for the plot.

    Returns:
        None
    """
    fixed_points = [1, 60, 3600, 86400, 604800, 2592000, 31536000, 315360000]
    fixed_points_labels = [
        "1 second",
        "1 minute",
        "1 hour",
        "1 day",
        "1 week",
        "30 days",
        "365 days",
        "3650 days",
    ]
    min_t = np.inf
    max_t = -np.inf
    np1_color = None
    np2_color = None
    for name, x_t_dict in name_x_t_dict.items():
        x = x_t_dict["x"]
        t = x_t_dict["t"]
        inferred = x_t_dict["inferred"]
        sorted_indices = np.argsort(x)
        x = x[sorted_indices]
        t = t[sorted_indices]
        inferred = inferred[sorted_indices]
        if "fast" in name:
            linestyle = "--"
        else:
            linestyle = "-"
        if np1_color is not None and "NumPy IKPLS #1" in name:
            color = np1_color
            curve = ax.loglog(x, t, linestyle, color=color, label=name)
        elif np2_color is not None and "NumPy IKPLS #2" in name:
            color = np2_color
            curve = ax.loglog(x, t, linestyle, color=color, label=name)
        else:
            curve = ax.loglog(x, t, linestyle, label=name)
        color = curve[0].get_color()
        if "NumPy IKPLS #1" in name and np1_color is None:
            np1_color = color
        elif "NumPy IKPLS #2" in name and np2_color is None:
            np2_color = color
        for point_x, point_t, point_inferred in zip(x, t, inferred):
            if np.isnan(point_inferred) or not point_inferred:
                ax.loglog(point_x, point_t, "o", color=color)
            else:
                ax.loglog(point_x, point_t, "s", color=color)
        if name not in legend_dict:
            legend_dict[name] = True
        try:
            if min_t > np.min(t):
                min_t = np.min(t)
            if max_t < np.max(t):
                max_t = np.max(t)
        except:
            pass
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for fp, fpl in zip(fixed_points, fixed_points_labels):
        if min_t <= fp <= max_t:
            ax.axhline(fp, color="k", linestyle="--", linewidth=1)
            ax.text(
                0.25, fp, fpl, fontsize=10, ha="center", va="bottom", transform=trans
            )
    if xlabel == "n_components":
        xlabel = "a"
    xlabel = xlabel.upper()
    title_str = ""
    f = mticker.ScalarFormatter(useMathText=True)
    for name, value in constants_dict.items():
        if name == "n_splits":
            continue
        elif name == "n_components":
            name = "a"
        name = name.upper()
        if value >= 1000:
            value = f.format_data(value)
        title_str += f"${name}={value}$, "
    title_str = title_str[:-2]
    ax.set_title(title_str, fontsize=10)


if __name__ == "__main__":
    df = pd.read_csv("timings/user_timings.csv")
    df = remove_rows_where_all_values_except_time_are_same(df)

    plt.rcParams.update({"font.size": 10})
    fig, axs = plt.subplots(4, 3, figsize=(15, 15))
    legend_dict = {}

    # Single fit
    constants_dict_n_single_fit_pls1 = {"n_components": 30, "k": 500, "m": 1}
    name_x_t_dict_n_single_fit_pls1 = get_name_x_t_dict(
        df, "n", constants_dict_n_single_fit_pls1, "single_fit"
    )
    plot_timings(
        axs[0, 0],
        name_x_t_dict_n_single_fit_pls1,
        "N",
        constants_dict_n_single_fit_pls1,
    )

    constants_dict_k_single_fit_pls1 = {"n_components": 30, "n": 10000, "m": 1}
    name_x_t_dict_k_single_fit_pls1 = get_name_x_t_dict(
        df, "k", constants_dict_k_single_fit_pls1, "single_fit"
    )
    plot_timings(
        axs[0, 1],
        name_x_t_dict_k_single_fit_pls1,
        "K",
        constants_dict_k_single_fit_pls1,
    )

    constants_dict_nc_single_fit_pls1 = {"n": 10000, "k": 500, "m": 1}
    name_x_t_dict_nc_single_fit_pls1 = get_name_x_t_dict(
        df, "n_components", constants_dict_nc_single_fit_pls1, "single_fit"
    )
    plot_timings(
        axs[0, 2],
        name_x_t_dict_nc_single_fit_pls1,
        "A",
        constants_dict_nc_single_fit_pls1,
    )

    constants_dict_n_single_fit_pls2 = {"n_components": 30, "k": 500, "m": 10}
    name_x_t_dict_n_single_fit_pls2 = get_name_x_t_dict(
        df, "n", constants_dict_n_single_fit_pls2, "single_fit"
    )
    plot_timings(
        axs[2, 0],
        name_x_t_dict_n_single_fit_pls2,
        "N",
        constants_dict_n_single_fit_pls2,
    )

    constants_dict_k_single_fit_pls2 = {"n_components": 30, "n": 10000, "m": 10}
    name_x_t_dict_k_single_fit_pls2 = get_name_x_t_dict(
        df, "k", constants_dict_k_single_fit_pls2, "single_fit"
    )
    plot_timings(
        axs[2, 1],
        name_x_t_dict_k_single_fit_pls2,
        "K",
        constants_dict_k_single_fit_pls2,
    )

    constants_dict_nc_single_fit_pls2 = {"n": 10000, "k": 500, "m": 10}
    name_x_t_dict_nc_single_fit_pls2 = get_name_x_t_dict(
        df, "n_components", constants_dict_nc_single_fit_pls2, "single_fit"
    )
    plot_timings(
        axs[2, 2],
        name_x_t_dict_nc_single_fit_pls2,
        "A",
        constants_dict_nc_single_fit_pls2,
    )

    # LOOCV
    constants_dict_n_loocv_pls1 = {"n_components": 30, "k": 500, "m": 1}
    name_x_t_dict_n_loocv_pls1 = get_name_x_t_dict(
        df, "n", constants_dict_n_loocv_pls1, "loocv"
    )
    plot_timings(
        axs[1, 0],
        name_x_t_dict_n_loocv_pls1,
        "N",
        constants_dict_n_loocv_pls1,
    )

    constants_dict_k_loocv_pls1 = {"n_components": 30, "n": 10000, "m": 1}
    name_x_t_dict_k_loocv_pls1 = get_name_x_t_dict(
        df, "k", constants_dict_k_loocv_pls1, "loocv"
    )
    plot_timings(
        axs[1, 1],
        name_x_t_dict_k_loocv_pls1,
        "K",
        constants_dict_k_loocv_pls1,
    )

    constants_dict_nc_loocv_pls1 = {"n": 10000, "k": 500, "m": 1}
    name_x_t_dict_nc_loocv_pls1 = get_name_x_t_dict(
        df, "n_components", constants_dict_nc_loocv_pls1, "loocv"
    )
    plot_timings(
        axs[1, 2],
        name_x_t_dict_nc_loocv_pls1,
        "A",
        constants_dict_nc_loocv_pls1,
    )

    constants_dict_n_loocv_pls2 = {"n_components": 30, "k": 500, "m": 10}
    name_x_t_dict_n_loocv_pls2 = get_name_x_t_dict(
        df, "n", constants_dict_n_loocv_pls2, "loocv"
    )
    plot_timings(
        axs[3, 0],
        name_x_t_dict_n_loocv_pls2,
        "N",
        constants_dict_n_loocv_pls2,
    )

    constants_dict_k_loocv_pls2 = {"n_components": 30, "n": 10000, "m": 10}
    name_x_t_dict_k_loocv_pls2 = get_name_x_t_dict(
        df, "k", constants_dict_k_loocv_pls2, "loocv"
    )
    plot_timings(
        axs[3, 1],
        name_x_t_dict_k_loocv_pls2,
        "K",
        constants_dict_k_loocv_pls2,
    )

    constants_dict_nc_loocv_pls2 = {"n": 10000, "k": 500, "m": 10}
    name_x_t_dict_nc_loocv_pls2 = get_name_x_t_dict(
        df, "n_components", constants_dict_nc_loocv_pls2, "loocv"
    )
    plot_timings(
        axs[3, 2],
        name_x_t_dict_nc_loocv_pls2,
        "A",
        constants_dict_nc_loocv_pls2,
    )

    fig.supylabel("Time (s)")
    space = " " * 60
    fig.supxlabel(
        "$N$ (no. samples)"
        + space
        + "$K$ (no. X features)"
        + space
        + "$A$ (no. components)"
    )
    handles, labels = axs[3, 0].get_legend_handles_labels()
    sk_handle = handles[0]
    sk_label = labels[0]
    handles[:2] = handles[1:3]
    labels[:2] = labels[1:3]
    handles[2] = sk_handle
    labels[2] = sk_label

    rect1 = patches.Rectangle(
        (0.04, 0.47),
        0.92,
        0.43,
        transform=fig.transFigure,
        linewidth=1,
        edgecolor="b",
        facecolor="none",
    )
    rect2 = patches.Rectangle(
        (0.04, 0.039),
        0.92,
        0.43,
        transform=fig.transFigure,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )

    rect11 = patches.Rectangle(
        (0.045, 0.692),
        0.91,
        0.205,
        transform=fig.transFigure,
        linewidth=1,
        edgecolor="c",
        facecolor="none",
    )
    rect12 = patches.Rectangle(
        (0.045, 0.476),
        0.91,
        0.205,
        transform=fig.transFigure,
        linewidth=1,
        edgecolor="m",
        facecolor="none",
    )

    rect21 = patches.Rectangle(
        (0.045, 0.258),
        0.91,
        0.205,
        transform=fig.transFigure,
        linewidth=1,
        edgecolor="c",
        facecolor="none",
    )
    rect22 = patches.Rectangle(
        (0.045, 0.042),
        0.91,
        0.205,
        transform=fig.transFigure,
        linewidth=1,
        edgecolor="m",
        facecolor="none",
    )

    first_legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        fancybox=True,
        shadow=True,
        ncol=len(labels) // 2,
        prop={"size": 9},
        bbox_to_anchor=(0.5, 0.95),
        bbox_transform=plt.gcf().transFigure,
    )
    plt.gca().add_artist(first_legend)
    
    ikpls_version_text = f"ikpls version: {ikpls.__version__}"

    fig.text(
        x=0.02,  # Adjust the x-coordinate to control horizontal placement
        y=0.98,  # Adjust the y-coordinate to control vertical placement
        s=ikpls_version_text,  # The text string to display
        fontsize=9,  # Adjust font size as needed
        ha='left',  # Horizontal alignment: 'left', 'center', or 'right'
        va='top',  # Vertical alignment: 'top', 'center', or 'bottom'
        transform=fig.transFigure  # Use figure-level coordinates
    )

    plt.legend(
        handles=[rect1, rect2, rect11, rect12],
        labels=["PLS1", "PLS2", "Single Fit", "LOOCV"],
        loc="lower center",
        fancybox=True,
        shadow=True,
        ncol=2,
        prop={"size": 9},
        bbox_to_anchor=(0.5, 0.95),
        bbox_transform=plt.gcf().transFigure,
    )

    fig.patches.extend([rect1, rect2, rect11, rect12, rect21, rect22])
    # fig.tight_layout()
    plt.subplots_adjust(bottom=0.06, hspace=0.3, wspace=0.2, left=0.1, right=0.95)
    plt.savefig(f"timings/user_timings.png")
