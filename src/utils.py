"""
Will Buesnel, Jul 26.

Unless specified otherwise, all functions relating to plotting here come from the following source:
https://github.com/MarkBlyth/parameterisation_methodsx/research/scripts/utils.py
This is to keep the style of plots etc inkeeping with his,
and is permissable under the GPU Licence that both repos have.
"""

import time

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple
import csv
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.widgets import Slider
from collections.abc import Sequence


def set_rc_params():
    """
    Set the default rcParams for matplotlib to make plots look nicer.
    """
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 16,
        "figure.figsize": (8, 6),
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
    })

def plot_df(df: pd.DataFrame, title: str = "Data Overview", xlabel: str = "Index", ylabel: str = "Value") -> None:
    # plot every column in the dataframe on a separate subplot, sharing the x-axis
    n_cols = len(df.columns)
    fig, axes = plt.subplots(nrows=n_cols, ncols=1, sharex=True, figsize=(10, 2 * n_cols))
    print(df.columns)
    print(f"Plotting {n_cols} columns from dataframe with shape {df.shape}")
    for i, col in enumerate(df.columns):
        axes[i].plot(df[col], label=col)
        axes[i].set_ylabel(col)
        axes[i].legend()
    axes[-1].set_xlabel("Index")
    plt.show()

def add_zoom_inset(
        ax,
        xs: Sequence[np.ndarray],
        ys: Sequence[np.ndarray],
        colours: Sequence[str] | None = None,
        alphas: Sequence[float] | None = None,
        x_range: tuple[float, float] = (1, 1.05),
        y_range: tuple[float, float] = (4.05, 4.125),
        inset_position: tuple[float, float, float, float] | None = None,
        inset_loc: str = "lower left",
        inset_width: str = "40%",
        inset_height: str = "40%",
):
    """Add a zoomed-in inset view to the main voltage axis."""

    if len(xs) != len(ys):
        raise ValueError("xs and ys must contain the same number of datasets.")

    if colours is None:
        colours = [None] * len(xs)  # use matplotlib defaults

    if len(colours) != len(xs):
        raise ValueError("One colour must be provided for each dataset.")

    if alphas is None:
        alphas = [None] * len(xs)

    if len(alphas) != len(xs):
        raise ValueError("One alpha must be provided for each dataset.")

    bbox_to_anchor = (
        inset_position if inset_position is not None else (0.1, 0.1, 1, 1)
    )

    axins = inset_axes(
        ax,
        width=inset_width,
        height=inset_height,
        loc=inset_loc,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=ax.transAxes,
    )

    for x, y, c, a in zip(xs, ys, colours, alphas):
        axins.plot(x, y, color=c, lw=1, alpha=a)

    axins.set_xlim(*x_range)
    axins.set_ylim(*y_range)
    axins.set_xticks([])

    # set there to be two yticks, one at the top and one at the bottom of the inset.
    axins.set_yticks([y_range[0], y_range[1]])

    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")
    return axins


# -------PATH UTILS--------------------------------------------------------------_----
def get_path_to_data_processed_dir() -> Path:
    """
    Get the path to the data/processed directory, which is assumed to be two levels up from this file.
    """
    current_file_path = Path(__file__).resolve()
    data_processed_dir = current_file_path.parents[1] / "data" / "processed"
    return data_processed_dir

def get_path_to_data_results_dir() -> Path:
    """
    Get the path to the data/results directory, which is assumed to be two levels up from this file.
    """
    current_file_path = Path(__file__).resolve()
    data_results_dir = current_file_path.parents[1] / "data" / "results"
    return data_results_dir

def get_path_to_data_dir() -> Path:
    """
    Get the path to the data directory, which is assumed to be two levels up from this file.
    """
    current_file_path = Path(__file__).resolve()
    data_dir = current_file_path.parents[1] / "data"
    return data_dir

def get_path_to_figures_dir() -> Path:
    """
    Get the path to the figures directory, which is assumed to be two levels up from this file.
    """
    current_file_path = Path(__file__).resolve()
    figures_dir = current_file_path.parents[1] / "figures"
    return figures_dir

#  -----------------------------------------------------------------------------------

# Utils for pyro model:


def plot_mixing(samples: pd.DataFrame, param_names):

    num_params = len(param_names)  
    num_chains = samples["Chain"].nunique()

    fig, axs = plt.subplots(num_params, 1)

    for i, param in enumerate(param_names): 
        print(f"Plotting trace for {param}...")
        for chain in range(1, num_chains+1):
            chain_samples = samples[samples["Chain"] == chain][param]
            axs[i].plot(chain_samples.values, label=f"Chain {chain}", alpha=0.5)
        axs[i].set_title(f"Trace plot for {param}")
        axs[i].set_xlabel("Iteration")
        axs[i].set_ylabel(param)
        axs[i].legend()

    plt.tight_layout()

    plt.show()
        

def save_pred_samples_to_pt(samples, filename, with_time=True):
    # pred_samples is a dict of tensors, convert to pandas dataframe and save to csv
    if with_time:
        filename = filename.replace(".pt", f"_{time.strftime('%Y%m%d_%H%M%S')}.pt")
    torch.save(samples, filename)

def save_diagnostics_to_csv(mcmc, filename, with_time=True):
    if with_time:
        filename = filename.replace(".csv", f"_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    diagnostics = mcmc.diagnostics()
    diagnostics_df = pd.DataFrame(diagnostics)
    diagnostics_df.to_csv(filename, index=False)
    print(f"Diagnostics saved to {filename}")

def open_pred_samples_from_pt(filename):
    return torch.load(filename)

def convert_pred_samples_to_df(samples, drop_index=False):
    # chains are indicated in the individual terms within the samples.
    # i.e.: variance : [[samples from chain 1],...,[samples from chain n]]
    # it is a list of dicts
    # convert to pandas dataframe with columns for each parameter and rows for each sample, with a column for the chain number & iteration within a chain.
    key1 = list(samples.keys())[0]  # get the first key to determine the shape of the samples
    vals1 = samples[key1]  # get the values for the first key
    n_chains, n_samples = vals1.shape[0], vals1.shape[1]  # get the number of chains and samples
    all_samples = []
    for chain in range(1, n_chains + 1):
        for sample in range(1, n_samples + 1):
            sample_dict = {param: samples[param][chain-1, sample-1].detach().numpy() for param in samples.keys()}
            sample_dict["Chain"] = chain
            sample_dict["Iteration"] = sample
            all_samples.append(sample_dict)

    samples_df = pd.DataFrame(all_samples)
    if drop_index:
        samples_df = samples_df.drop(columns=["Iteration"])
    return samples_df


def open_pred_samples_as_df(filename, drop_index=False):
    samples_tensor = open_pred_samples_from_pt(filename)
    # convert to pandas dataframe
    # tensor will be of shape (num_chains, num_samples, num_timesteps)
    return convert_pred_samples_to_df(samples_tensor, drop_index=drop_index)



def df_to_tensor_dict(df, param_cols=None, dtype=torch.float32):
    if param_cols is None:
        param_cols = [c for c in df.columns if c not in ('Iteration', 'Chain')]
    
    chains = sorted(df['Chain'].unique())
    iters = sorted(df['Iteration'].unique())
    n_chains = len(chains)
    n_iters = len(iters)
    
    chain_idx = {c: i for i, c in enumerate(chains)}
    iter_idx = {it: i for i, it in enumerate(iters)}
    
    result = {}
    for param in param_cols:
        sample_len = np.atleast_1d(df[param].iloc[0]).shape[0]
        arr = np.empty((n_chains, n_iters, sample_len), dtype=np.float64)
        
        for chain, iteration, sample in zip(df['Chain'], df['Iteration'], df[param]):
            arr[chain_idx[chain], iter_idx[iteration], :] = sample
        
        result[param] = torch.tensor(arr, dtype=dtype)
    
    return result



def graph_model_outputs(sim, gauss_interps: list[tuple[str, object]] = None, y0: list = [1,0,0,25], obs=None):
        
        res1 = sim.run_simulation(**sim.kwargs, pbar=True, max_step = 40)  # run the simulation once to check it works.
        # reset the gauss interps
        sim.set_gauss_interps(gauss_interps)
        res2 = sim.run_simulation(**sim.kwargs, pbar=True, max_step = 40)  # run the simulation a second time to check it works.
    
        plt.plot(res1["t [s]"], res1["v_oc [V]"], label="Run 1", alpha=0.5)
        plt.plot(res2["t [s]"], res2["v_oc [V]"], label="Run 2", alpha=0.5)
    
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        plt.title("Voltage vs Time for MLP001 Cell")
        plt.legend()
        plt.show()
       
        plt.figure()
        plt.plot(res1["t [s]"], res1["R0 [Ohm]"], label="Run 1", alpha=0.5)
        plt.plot(res2["t [s]"], res2["R0 [Ohm]"], label="Run 2", alpha=0.5)
        plt.xlabel("Time [s]")
        plt.ylabel("R0 [Ohm]")
        plt.title("R0 vs Time for MLP001 Cell")
        plt.legend()
        plt.show()
    


# ------------------------------------------------------------------------------------------


def plot_traces(xs: np.ndarray, Ys: np.ndarray, title: str = "Parameter Traces Over Iterations", xlabel: str = 'x', ylabel: str = 'Parameter Value', multiple_chains: bool = False):
        """
        Plot the parameter values against a given x value for each iteration of the inference process. This function is useful for visualizing how the parameter values evolve over iterations.
        Multiple chains can be plotted if the `multiple_chains` flag is set to True. In this case, the Ys array should have shape (nchains, n_iterations, n_points).
        """
        n_iterations = len(Ys[0] if multiple_chains else Ys)  # number of iterations is the length of the first dimension if multiple chains, otherwise it's the length of Ys
        n_chains = Ys.shape[0] if multiple_chains else 1

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.25)  # leave room for the slider

        lines = []

        # initial plot (iteration 0)
        if multiple_chains:
            for chain in range(n_chains):
                line, = ax.plot(xs, Ys[chain, 0, :], marker="o", label=f"Chain {chain+1} Iteration 0")
                lines.append(line)
        else:
            line, = ax.plot(xs, Ys[0, :], marker="o", label="Iteration 0")
            lines.append(line)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(Ys.min() * 0.95, Ys.max() * 1.05)
        ax.legend()
        ax.grid(True)

        # slider axis
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        slider = Slider(
            ax=ax_slider,
            label="Iteration",
            valmin=0,
            valmax=n_iterations - 1,
            valinit=0,
            valstep=1,
        )

        def update(val):
            print(f"Updating plot for iteration {val}...")
            idx = int(slider.val)
            for i, line in enumerate(lines):
                if multiple_chains:
                    line.set_ydata(Ys[i, idx, :])
                else:
                    line.set_ydata(Ys[idx, :])
            ax.legend()
            fig.canvas.draw_idle()

        slider.on_changed(update)

        plt.show()
        return fig, slider  # return slider so it isn't garbage-collected in some environments


def get_drive_cycle_hours_col(cycler_file: str) -> pd.DataFrame:
    drive_df = pd.read_csv(cycler_file)

    # add an hours column to df:
    # data are split into 'hours:seconds:milliseconds.'
    hours_arr = drive_df["Total Time"].str.split(":") # gives a 2d array of nx3
    arr = np.asarray(hours_arr.tolist(), dtype=float)
    hours = arr @ np.array([1, 1/60, 1/3600])

    # take away the initial time to get elapsed time.
    hours = hours - hours[0]

    drive_df["Elapsed Time[h]"] = hours

    return drive_df


def get_errors_by_resampling(model_ts, experiment_times, experiment_voltages, model_voltages) -> np.ndarray:
    # Interpolate the model voltages at the experiment time points
    resampled_exp_voltages = np.interp(model_ts, experiment_times, experiment_voltages)
    # Calculate the errors
    errors = model_voltages - resampled_exp_voltages
    return errors

import torch
from tqdm import tqdm

def dequantise_data(time: np.ndarray, pbar: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dequantise data assuming resolution error: if N samples share the same
    timestamp, spread them evenly across the interval up to the next distinct
    timestamp, assuming time is monotonically non-decreasing.
    """
    time = time.astype(float).copy()
    n = int(len(time))
    index = 0

    progress = tqdm(total=n, unit="pt", desc="Dequantising") if pbar else None

    try:
        while index < n:
            same_time_points = np.where(time == time[index])[0]
            count = len(same_time_points)
            for i in range(0,count):
                timestep = time[index + i]
                time[index + i] = time[index] + (i / count) * (time[index + count] - timestep) if index + count < n else timestep

            index += count  # always advances, avoids infinite loop
            if progress is not None:
                progress.update(count)
    finally:
        if progress is not None:
            progress.close()

    return time


def incomplete_get_errors_by_dequantisation(model_ts, experiment_times, experiment_voltages, model_voltages) -> np.ndarray:
    """
    The dataset currently has multiple voltage measurements at the same time point due to the resolution not being less than seconds.
    we do however know the data is monotonically increasing in time, therefore we can break down into microseconds
    """

def get_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the mean squared error between two arrays.
    """
    return np.mean((y_true - y_pred) ** 2)


def get_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the mean absolute error between two arrays.
    """
    return np.mean(np.abs(y_true - y_pred))


def convert_datetime_to_hours(df: pd.DataFrame, time_col: str = "Total Time") -> pd.DataFrame:
    hours_arr = df[time_col].str.split(":") # gives a 2d array of nx3
    arr = np.asarray(hours_arr.tolist(), dtype=float)
    hours = arr @ np.array([1, 1/60, 1/3600])

    # take away the initial time to get elapsed time.
    hours = hours - hours[0]
    return hours

def safe_cholesky(K, jitter=1e-6, max_tries=6):
    K = (K + K.T) / 2
    for i in range(max_tries):
        try:
            return torch.linalg.cholesky(K + torch.eye(len(K), dtype=K.dtype, device=K.device) * jitter)
        except RuntimeError:
            jitter *= 10
    raise RuntimeError(f"Cholesky failed even with jitter={jitter:g}")

def plot_matrix(
    ax: plt.Axes,
    fig: plt.Figure,
    matrix: np.ndarray,
    x_labels: list[str] | None = None,
    y_labels: list[str] | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    save_name: str | None = None,
):
    """
    Plot a matrix as a heatmap with optional labels and title.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(matrix, cmap=cmap)
    fig.colorbar(cax)

    if x_labels is not None:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=90)

    if y_labels is not None:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    if save_name is not None:
        plt.savefig(save_name)
    else:
        plt.show()


def plot_2d_matrix(ys, std, soc_grid, temp_grid, train_X_2d, interp_r0, variance, noise):
    import plotly.graph_objects as go
    import numpy as np

    griddims = np.sqrt(ys.shape[0]).astype(int)

    mean_grid = ys.reshape(griddims, griddims).detach().numpy()
    std_grid = std.reshape(griddims, griddims).detach().numpy()
    soc_np = soc_grid.numpy()
    temp_np = temp_grid.numpy()

    fig = go.Figure()

    # std surface (uncertainty)
    fig.add_trace(go.Surface(
        x=soc_np, y=temp_np, z=mean_grid, surfacecolor=std_grid,
        colorscale="Magma", opacity=0.85, name="GP Posterior Uncertainty",
        colorbar=dict(title="Uncertainty [Ohm]")
    ))

    # training points projected at z=interp_r0(train_X_2d[:, 0], train_X_2d[:, 1]).numpy()
    fig.add_trace(go.Scatter3d(
        x=train_X_2d[:, 0].numpy(),
        y=train_X_2d[:, 1].numpy(),
        z=interp_r0(train_X_2d[:, 0], train_X_2d[:, 1]).numpy(),
        mode="markers",
        marker=dict(size=4, color="cyan"),
        name="Training points"
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="SOC", yaxis_title="Temperature (°C)", zaxis_title="R0 [Ohm]",
            # lock aspect ratio so zoom doesn't distort weirdly
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)  # starting zoom/angle
            )
        ),
        title="GP Posterior Uncertainty over Continuous Grid",
        width=900, height=700,
        dragmode="orbit"  # or "turntable" - controls rotation behavior
    )

    # explicit scroll-zoom + toolbar config
    fig.show(config={
        "scrollZoom": True,
        "displayModeBar": True,
        "modeBarButtonsToAdd": ["zoom3d", "pan3d", "resetCameraDefault3d"]
    })
    lengthscale_desc = "ell(soc, temp) = 0.025 + 0.005·sigmoid(5000·(soc−0.3)) + 0.01·temp_norm"

    annotation_text = (
        f"Variance: {variance}<br>"
        f"Noise: {noise}<br>"
        f"Lengthscale fn: {lengthscale_desc}"
    )

    fig.update_layout(
        annotations=[
            dict(
                text=annotation_text,
                xref="paper", yref="paper",
                x=0.01, y=0.01,          # bottom-left corner
                xanchor="left", yanchor="bottom",
                showarrow=False,
                align="left",
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.7)",  # slight background so it's readable over the surface
                bordercolor="black",
                borderwidth=1
            )
        ]
    )



def get_ax(save: bool = False, n_axes: int = 1, **ax_kwargs):
    if save:
        matplotlib.use("pgf")
        matplotlib.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
            }
        )
    ax = subjectively_better_subplots(n_axes, **ax_kwargs)

    def tidy_up(filename: str | None = None):
        if filename is None:
            # make full screen:
            # Get the current figure manager and maximize the window
            mng = plt.get_current_fig_manager()

            mng.full_screen_toggle()
            plt.show()
        else:
            plt.savefig(filename)

    return ax, tidy_up

def make_plot(
        exp_df: pd.DataFrame,
        model_results: Tuple[np.ndarray, np.ndarray, np.ndarray],
        colours: List[str] | None = None,
        alphas: List[float] | None = None,
        x_range: tuple[float, float] = (0.5, 0.55),
        y_range: tuple[float, float] = (4.05, 4.125),
        inset_position: tuple[float, float, float, float] | None = None,
        inset_loc: str = "lower left",
        inset_width: str = "40%",
        inset_height: str = "40%",
        save_name: str | None = None,
):
    """
    Partially taken from Mark's repo, but modified significantly.
    exp_df: DataFrame containing the experimental data with columns "Elapsed Time[h]", "Voltage(V)", "Current(A)"
    model_results: Tuple containing the model results (times, voltages, errors) The time for this should also be in hours to keep consistent.
    save_name: Optional name to save the plot. If None, the plot will be shown
    """
    axes, tidy_up = get_ax(
        bool(save_name),
        n_axes=3,
        bottom_extra=0.35,
        l_margin=1.7,
    )
    for index, ax in enumerate(axes):
        ax.set_xticks([])

    model_ts, model_vs, model_errors = model_results
    # plot the applied current from the experiment
    axes[0].plot(
        exp_df["deq_Elapsed Time[h]"],
        exp_df["Current(A)"],
        label="Applied Current",
        color="tab:green",
    )
    axes[0].set_ylabel("Current (A)")

    # plot experiment data
    axes[1].plot(
            exp_df["deq_Elapsed Time[h]"],
            exp_df["Voltage(V)"],
            label="Experiment",
            color="tab:cyan",
        )
    axes[1].set_ylabel("Voltage (V)")

    # plot model data
    model_ts, model_vs, model_errors = model_results
    axes[1].plot(
            model_ts,
            model_vs,
            label="Model",
            color="k",
        )
    axes[1].legend(frameon=False, loc="upper right")
    
    # plot errors, but in mV to make it more readable.
    axes[2].plot(
            model_ts,  
            model_errors * 1000,  # convert to mV
            label="Errors",
            color="tab:brown",
        )
    axes[2].set_ylabel("Model error (mV)")
    axes[2].set_xlabel("Time (h)")
    
    # plot xticks for the bottom axis only, and set the xlim to the same as the experiment data.
    xmin, xmax = exp_df["deq_Elapsed Time[h]"].min(), exp_df["deq_Elapsed Time[h]"].max()

    x_ticks = np.arange(np.floor(xmin), np.ceil(xmax) + 1, 5)
    for ax in (axes[0], axes[1], axes[2]):
        ax.set_xticks(x_ticks)

    # Then only show tick labels on the bottom axis:
    for ax in (axes[0], axes[1], axes[2])[:-1]:
        ax.tick_params(labelbottom=False)
    
    xmin, xmax = exp_df["deq_Elapsed Time[h]"].min(), exp_df["deq_Elapsed Time[h]"].max()
    xrange = xmax - xmin
    margin = 0.05  # 5%, matplotlib's default

    for ax in (axes[0], axes[1], axes[2]):
        ax.set_xlim(xmin - margin * xrange, xmax + margin * xrange)

    
    axes[2].figure.align_ylabels([axes[0], axes[1], axes[2]])  # align ylabels of all axes

    add_zoom_inset(
        ax=axes[1],
        xs=[model_ts,exp_df["deq_Elapsed Time[h]"].to_numpy()],
        ys=[model_vs,exp_df["Voltage(V)"].to_numpy()],
        colours=["k", "tab:cyan"],
        alphas=[1, 0.7],
        inset_position=[0.05, 0.05, 1, 1],
        x_range=(0.5, 0.55),
        y_range=(4.1, 4.2),
    )

    axes[2].figure.subplots_adjust(left=0.15)
    
    tidy_up(save_name)


def subjectively_better_subplots(
    nrows,
    subheight=10 / 3,
    subwidth=10,
    bottom_extra=0.2,
    vertical_padding=0.25,
    l_margin=1,
    r_margin=0.1,
    **kwargs,
):
    # Source - https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
    # Retrieved 2025-12-08, License - CC BY- 4.0
    lm = l_margin / 2.54
    rm = r_margin / 2.54
    ax_padding = vertical_padding / 2.54
    a = subheight / 2.54
    w = subwidth / 2.54
    width = lm + w + rm
    height = nrows * (2 * ax_padding + a) + bottom_extra
    fig = plt.figure(figsize=(width, height), **kwargs)
    axarr = np.empty(nrows, dtype=object)
    for i in range(nrows):
        axarr[i] = fig.add_axes(
            [
                lm / width,
                (height - (i + 1) * (2 * ax_padding + a) + ax_padding) / height,
                w / width,
                a / height,
            ]
        )
    if nrows == 1:
        return axarr[0]
    return axarr