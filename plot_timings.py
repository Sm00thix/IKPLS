import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
            raise ValueError(f"single_fit_or_loocv must be 'single_fit' or 'loocv'. but got: {single_fit_or_loocv}")
        x = sub_df[x_name].values
        t = sub_df["time"].values
        if model_name == "sk":
            model_name = "scikit-learn NIPALS"
        elif model_name == "np1":
            model_name = "IKPLS #1"
        elif model_name == "np2":
            model_name = "IKPLS #2"
        elif model_name == "jax1":
            model_name = "IKPLS #1 (GPU)"
        elif model_name == "jax2":
            model_name = "IKPLS #2 (GPU)"
        elif model_name == "diffjax1":
            model_name = "IKPLS #1 (GPU, BMD)"
        elif model_name == "diffjax2":
            model_name = "IKPLS #2 (GPU, BMD)"
        name_x_t_dict[model_name] = {"x": x, "t": t}
    return name_x_t_dict

def plot_timings(name_x_t_dict, xlabel, constants_dict, log_scale_x, log_scale_t, single_fit_or_loocv):
    fig, ax = plt.subplots()
    for name, x_t_dict in name_x_t_dict.items():
        x = x_t_dict["x"]
        t = x_t_dict["t"]
        sorted_indices = np.argsort(x)
        x = x[sorted_indices]
        t = t[sorted_indices]
        if log_scale_t:
            if log_scale_x:
                ax.loglog(x, t, 'o-', label=name)
            else:
                ax.semilogy(x, t, 'o-', label=name)
        else:
            if log_scale_x:
                ax.semilogx(x, t, 'o-', label=name)
            else:
                ax.plot(x, t, 'o-', label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Time (s)")
    if xlabel == "n_components":
        xlabel = "a"
    xlabel = xlabel.upper()
    title_str = f"Time vs. {xlabel} for "
    for name, value in constants_dict.items():
        if name == "n_splits":
            continue
        elif name == "n_components":
            name = "a"
        name = name.upper()
        title_str += f"{name}={value}, "
    title_str = title_str[:-2]
    ax.set_title(title_str)
    ax.legend()
    pls1_or_pls2 = "pls1" if constants_dict["m"] == 1 else "pls2"
    plt.savefig(f"timings/{single_fit_or_loocv}_{pls1_or_pls2}_{xlabel}.png")

if __name__ == "__main__":
    df = pd.read_csv("timings/timings.csv")
    df = remove_rows_where_all_values_except_time_are_same(df)

    constants_dict_n_single_fit_pls1 = {"n_components": 30, "k": 500, "m": 1}
    name_x_t_dict_n_single_fit_pls1 = get_name_x_t_dict(df, "n", constants_dict_n_single_fit_pls1, "single_fit")
    plot_timings(name_x_t_dict_n_single_fit_pls1, "N", constants_dict_n_single_fit_pls1, True, False, "single_fit")

    constants_dict_m_single_fit_pls1 = {"n_components": 30, "n": 10000, "m": 1}
    name_x_t_dict_m_single_fit_pls1 = get_name_x_t_dict(df, "k", constants_dict_m_single_fit_pls1, "single_fit")
    plot_timings(name_x_t_dict_m_single_fit_pls1, "K", constants_dict_m_single_fit_pls1, True, False, "single_fit")

    constants_dict_nc_single_fit_pls1 = {"n": 10000, "k": 500, "m": 1}
    name_x_t_dict_nc_single_fit_pls1 = get_name_x_t_dict(df, "n_components", constants_dict_nc_single_fit_pls1, "single_fit")
    plot_timings(name_x_t_dict_nc_single_fit_pls1, "A", constants_dict_nc_single_fit_pls1, False, True, "single_fit")
    
    constants_dict_n_single_fit_pls2 = {"n_components": 30, "k": 500, "m": 10}
    name_x_t_dict_n_single_fit_pls2 = get_name_x_t_dict(df, "n", constants_dict_n_single_fit_pls2, "single_fit")
    plot_timings(name_x_t_dict_n_single_fit_pls2, "N", constants_dict_n_single_fit_pls2, True, True, "single_fit")

    constants_dict_m_single_fit_pls2 = {"n_components": 30, "n": 10000, "m": 10}
    name_x_t_dict_m_single_fit_pls2 = get_name_x_t_dict(df, "k", constants_dict_m_single_fit_pls2, "single_fit")
    plot_timings(name_x_t_dict_m_single_fit_pls2, "K", constants_dict_m_single_fit_pls2, True, True, "single_fit")

    constants_dict_nc_single_fit_pls2 = {"n": 10000, "k": 500, "m": 10}
    name_x_t_dict_nc_single_fit_pls2 = get_name_x_t_dict(df, "n_components", constants_dict_nc_single_fit_pls2, "single_fit")
    plot_timings(name_x_t_dict_nc_single_fit_pls2, "A", constants_dict_nc_single_fit_pls2, False, True, "single_fit")
    # name_x_t_dict_n_loocv = get_name_x_t_dict(df, "n", constants_dict, "loocv")