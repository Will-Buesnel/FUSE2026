"""
Will Buesnel, Jul 26.

Unless specified otherwise, all functions relating to plotting here come from the following source:
https://github.com/MarkBlyth/parameterisation_methodsx/research/scripts/utils.py
This is to keep the style of plots etc inkeeping with his,
and is permissable under the GPU Licence that both repos have.
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Tuple
import csv
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def add_zoom_inset( 
        ax,
        x1s: np.ndarray, # 1st dataset (model) to plot in the inset
        y1s: np.ndarray,
        x2s: np.ndarray, # 2nd dataset (experiment) to plot in the inset
        y2s: np.ndarray,
        x_range: tuple[float, float] = (1, 1.05),
        y_range: tuple[float, float] = (4.05, 4.125),
        inset_position: tuple[float, float, float, float] | None = None,
        inset_loc: str = "lower left",
        inset_width: str = "40%",
        inset_height: str = "40%",
):
    """Add a zoomed-in inset view to the main voltage axis. This does not come from Mark's repo."""
    bbox_to_anchor = inset_position if inset_position is not None else (0.1, 0.1, 1, 1)

    axins = inset_axes(
        ax,
        width=inset_width,
        height=inset_height,
        loc=inset_loc,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=ax.transAxes,
    )

    axins.plot(x1s, y1s, color="k", lw=1)
    axins.plot(x2s, y2s, color="tab:cyan", lw=1)

    axins.set_xlim(*x_range)
    axins.set_ylim(*y_range)
    axins.set_xticks([])

    # set there to be two yticks, one at the top and one at the bottom of the inset.
    axins.set_yticks([y_range[0], y_range[1]])

    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")
    return axins


def get_drive_cycle_df(cycler_file: str) -> pd.DataFrame:
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

def incomplete_get_errors_by_dequantisation(model_ts, experiment_times, experiment_voltages, model_voltages) -> np.ndarray:
    """
    The dataset currently has multiple voltage measurements at the same time point due to the resolution not being less than seconds.
    we do however know the data is monotonically increasing in time, therefore we can break down into microseconds
    """


def convert_datetime_to_hours(df: pd.DataFrame, time_col: str = "Total Time") -> pd.DataFrame:
    hours_arr = df[time_col].str.split(":") # gives a 2d array of nx3
    arr = np.asarray(hours_arr.tolist(), dtype=float)
    hours = arr @ np.array([1, 1/60, 1/3600])

    # take away the initial time to get elapsed time.
    hours = hours - hours[0]
    return hours


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
        save_name: str | None = None,
):
    """
    Partially taken from Mark's repo, but modified significantly.
    exp_df: DataFrame containing the experimental data with columns "Elapsed Time[h]", "Voltage(V)", "Current(A)"
    model_results: Tuple containing the model results (times, voltages, errors) The time for this should also be in hours to keep consistent.
    save_name: Optional name to save the plot. If None, the plot will be shown
    """
    (ax_I, ax_v, ax_e), tidy_up = get_ax(
        bool(save_name),
        n_axes=3,
        bottom_extra=0.35,
        l_margin=1.7,
    )
    

    ax_I.set_xticks([])
    ax_v.set_xticks([])
    ax_e.set_xticks([])

    # plot the applied current from the experiment
    ax_I.plot(
        exp_df["Elapsed Time[h]"],
        exp_df["Current(A)"],
        label="Applied Current",
        color="tab:green",
    )
    ax_I.set_ylabel("Current (A)")

    # plot experiment data
    ax_v.plot(
            exp_df["Elapsed Time[h]"],
            exp_df["Voltage(V)"],
            label="Experiment",
            color="tab:cyan",
        )
    ax_v.set_ylabel("Voltage (V)")

    # plot model data
    model_ts, model_vs, model_errors = model_results
    ax_v.plot(
            model_ts,
            model_vs,
            label="Model",
            color="k",
        )
    ax_v.legend(frameon=False, loc="upper right")
    
    # plot errors, but in mV to make it more readable.
    ax_e.plot(
            model_ts,  
            model_errors * 1000,  # convert to mV
            label="Errors",
            color="tab:brown",
        )
    ax_e.set_ylabel("Model error (mV)")
    ax_e.set_xlabel("Time (h)")
    
    # plot xticks for the bottom axis only, and set the xlim to the same as the experiment data.
    xmin, xmax = exp_df["Elapsed Time[h]"].min(), exp_df["Elapsed Time[h]"].max()

    x_ticks = np.arange(np.floor(xmin), np.ceil(xmax) + 1, 5)
    for ax in (ax_I, ax_v, ax_e):
        ax.set_xticks(x_ticks)

    # Then only show tick labels on the bottom axis:
    for ax in (ax_I, ax_v, ax_e)[:-1]:
        ax.tick_params(labelbottom=False)
    
    xmin, xmax = exp_df["Elapsed Time[h]"].min(), exp_df["Elapsed Time[h]"].max()
    xrange = xmax - xmin
    margin = 0.05  # 5%, matplotlib's default

    for ax in (ax_I, ax_v, ax_e):
        ax.set_xlim(xmin - margin * xrange, xmax + margin * xrange)

    
    ax_e.figure.align_ylabels([ax_I, ax_v, ax_e])  # align ylabels of all axes

    add_zoom_inset(
        ax_v,
        model_ts,
        model_vs,
        exp_df["Elapsed Time[h]"].to_numpy(),
        exp_df["Voltage(V)"].to_numpy(),
        inset_position=[0.05, 0.05, 1, 1],
        x_range=(0.5, 0.55),
        y_range=(4.1, 4.2),
    )

    ax_e.figure.subplots_adjust(left=0.15)
    
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
    # Retrieved 2025-12-08, License - CC BY-SA 4.0
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