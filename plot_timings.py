import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import patches


def remove_rows_where_all_values_except_time_are_same(df):
    df = df.drop_duplicates(subset=df.columns[:-1])
    return df


def get_name_x_t_dict(df, x_name, constants_dict, single_fit_or_loocv):
    model_names = ["sk", "np1", "np2", "jax1", "jax2", "diffjax1", "diffjax2"]
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
                f"single_fit_or_loocv must be 'single_fit' or 'loocv'. but got: {single_fit_or_loocv}"
            )
        x = sub_df[x_name].values
        t = sub_df["time"].values
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
        elif model_name == "diffjax1":
            model_name = "JAX IKPLS #1 (GPU, BMD)"
        elif model_name == "diffjax2":
            model_name = "JAX IKPLS #2 (GPU, BMD)"
        name_x_t_dict[model_name] = {"x": x, "t": t}
    return name_x_t_dict


def plot_timings(
    ax, name_x_t_dict, xlabel, constants_dict, log_scale_x, log_scale_t, single_fit_or_loocv
):
    # fig, ax = plt.subplots()
    for name, x_t_dict in name_x_t_dict.items():
        x = x_t_dict["x"]
        t = x_t_dict["t"]
        sorted_indices = np.argsort(x)
        x = x[sorted_indices]
        t = t[sorted_indices]
        if log_scale_t:
            if log_scale_x:
                ax.loglog(x, t, "o-", label=name)
            else:
                ax.semilogy(x, t, "o-", label=name)
        else:
            if log_scale_x:
                ax.semilogx(x, t, "o-", label=name)
            else:
                ax.plot(x, t, "o-", label=name)
        if name not in legend_dict:
            legend_dict[name] = True
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel("Time (s)")
    if xlabel == "n_components":
        xlabel = "a"
    xlabel = xlabel.upper()
    # title_str = f"Time vs. {xlabel} for "
    title_str = ""
    for name, value in constants_dict.items():
        if name == "n_splits":
            continue
        elif name == "n_components":
            name = "a"
        name = name.upper()
        title_str += f"${name}$={value}, "
    title_str = title_str[:-2]
    ax.set_title(title_str, fontsize=10)
    # ax.legend()
    # pls1_or_pls2 = "pls1" if constants_dict["m"] == 1 else "pls2"
    # plt.savefig(f"timings/{single_fit_or_loocv}_{pls1_or_pls2}_{xlabel}.png")


if __name__ == "__main__":
    df = pd.read_csv("timings/timings.csv")
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
        True,
        True,
        "single_fit",
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
        True,
        True,
        "single_fit",
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
        True,
        True,
        "single_fit",
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
        True,
        True,
        "single_fit",
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
        True,
        True,
        "single_fit",
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
        True,
        True,
        "single_fit",
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
        True,
        True,
        "loocv",
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
        True,
        True,
        "loocv",
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
        True,
        True,
        "loocv",
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
        True,
        True,
        "loocv",
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
        True,
        True,
        "loocv",
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
        True,
        True,
        "loocv",
    )

    
    fig.supylabel("Time (s)")
    space = " "*85
    fig.supxlabel("$N$" + space + "$K$" + space + "$A$")
    # axs[0, 0].twiny().set_xlabel("Time vs. N")
    # axs[0, 1].twiny().set_xlabel("Time vs. K")
    # axs[0, 2].twiny().set_xlabel("Time vs. A")
    # axs[3, 0].set_xlabel("$N$")
    # axs[3, 1].set_xlabel("$K$")
    # axs[3, 2].set_xlabel("$A$")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    

    rect1 = patches.Rectangle((0.04, 0.47), 0.92, 0.43, transform=fig.transFigure, linewidth=1, edgecolor='b', facecolor='none')
    rect2 = patches.Rectangle((0.04, 0.039), 0.92, 0.43, transform=fig.transFigure, linewidth=1, edgecolor='r', facecolor='none')
    
    rect11 = patches.Rectangle((0.045, 0.692), 0.91, 0.205, transform=fig.transFigure, linewidth=1, edgecolor='c', facecolor='none')
    rect12 = patches.Rectangle((0.045, 0.476), 0.91, 0.205, transform=fig.transFigure, linewidth=1, edgecolor='m', facecolor='none')

    rect21 = patches.Rectangle((0.045, 0.258), 0.91, 0.205, transform=fig.transFigure, linewidth=1, edgecolor='c', facecolor='none')
    rect22 = patches.Rectangle((0.045, 0.042), 0.91, 0.205, transform=fig.transFigure, linewidth=1, edgecolor='m', facecolor='none')

    first_legend = fig.legend(handles, labels, loc='upper center', fancybox=True, shadow=True, ncol=len(labels), prop={'size': 9}, bbox_to_anchor=(0.5, 0.95), bbox_transform=plt.gcf().transFigure)
    ax = plt.gca().add_artist(first_legend)
    plt.legend(handles=[rect1, rect2, rect11, rect12], labels=["PLS1", "PLS2", "Single Fit", "LOOCV"], loc='lower center', fancybox=True, shadow=True, ncol=2, prop={'size': 9}, bbox_to_anchor=(0.5, 0.95), bbox_transform=plt.gcf().transFigure)


    fig.patches.extend([rect1, rect2, rect11, rect12, rect21, rect22])
    # fig.tight_layout()
    plt.subplots_adjust(bottom=0.06, hspace=0.3, wspace=0.2, left=0.1, right=0.95)
    plt.savefig(f"timings/timings.png")
    plt.savefig(f"paper/timings.png")