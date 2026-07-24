"""Will Buesnel, Jul 26."""

from __future__ import annotations
from typing import Tuple
from utils import get_ax
import pandas as pd
import numpy as np
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


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

def get_errors_by_dequantisation(model_ts, experiment_times, experiment_voltages, model_voltages) -> np.ndarray:
    """
    The dataset currently has multiple voltage measurements at the same time point due to the resolution not being less than seconds.
    we do however know the data is monotonically increasing in time, therefore we can break down into microseconds
    """


def run_simulation_from_exp(
        exp_df: pd.DataFrame,
        ocv_df: pd.DataFrame,
        param_file: str,
        capacity_Ah: float,
        initial_soc: float,
        t_max: float | None = None,
        **kwargs
) -> dict:
    """
    Run a simulation of the electrical model using the given experimental data and parameters.
    The simulation will use the current data from the experiment and the parameters from the parameter file.
    The simulation will run for t_max seconds, or until the end of the experimental data if
    the exp_df will consists of Time, Current and Voltage
    Returns: res, a dictionary containing simulation results. See code below for more details.
    The res["t"] will be in seconds, and the res["soc"] will be a fraction between 0 and 1.
    """
    # check t_max is not None, if it is then set it to the max time in the experiment data.
    if t_max is None:
        t_max = exp_df["Elapsed Time[h]"].max() * 3600  # convert to seconds

    # get various dfs:
    param_df = pd.read_csv(param_file)

    experiment_times = exp_df["Elapsed Time[h]"].to_numpy() * 3600  # convert to seconds.
    experiment_currents = -exp_df["Current(A)"].to_numpy() # flip sign of curret to match equations given.


    # define current funciton that interpolates the current from the experiment data.
    def current_func(t):
        return np.interp(t, experiment_times, experiment_currents)
    
    # build the model and run the simulation.
    from models.electrical import ElectricalModel
    from models.parameters import get_all_parameter_interpolants

    param_interpolants = get_all_parameter_interpolants(param_df, ocv_df) # remember this is a dict of interpolants.

    model = ElectricalModel()
    # set the interpolants as funcs for the model
    for name, interpolant in param_interpolants.items():
        name = name.split(" ")[0].split("[")[0].lower()  # take only the first part of the name, e.g. "R0 [Ohm]" -> "R0"
        setattr(model, f"_{name}_interp", interpolant) 

    
    res = model.simulate(
        y0=[initial_soc, 0, 0],
        current_func= current_func, 
        t_max=t_max,
        max_capacity_Ah=capacity_Ah,
        **kwargs
    )

    return res


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
    """Add a zoomed-in inset view to the main voltage axis."""
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


def make_plot(
        exp_df: pd.DataFrame,
        model_results: Tuple[np.ndarray, np.ndarray, np.ndarray],
        save_name: str | None = None,
):
    """
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
    


def main(cycler_file, ocv_file, param_file):
    exp_df = get_drive_cycle_df(cycler_file)
    ocv_df = pd.read_csv(ocv_file)


    model_results = run_simulation_from_exp(
        exp_df=exp_df,
        ocv_df=ocv_df,
        param_file=param_file,
        capacity_Ah=2.2,  # Ah
        initial_soc=1.0,
        max_step = 10,  # seconds
        pbar=True,
    )
    v_sim = model_results["v_cell"]

    error = get_errors_by_resampling(
        model_results["t"],
        exp_df["Elapsed Time[h]"].to_numpy() * 3600,  # convert to seconds
        exp_df["Voltage(V)"].to_numpy(),
        v_sim,
    )
    make_plot(exp_df, (model_results["t"] * 1/3600, v_sim, error), save_name=None)


    # temporary plot of SoC vs time, for debugging purposes.
    import matplotlib.pyplot as plt
    plt.plot(
        model_results["t"] * 1/3600,
        model_results["soc"],
        label="SoC",
        color="tab:blue",
    )
    plt.xlabel("Time (h)")
    plt.ylabel("SoC")
    plt.legend(frameon=False)
    plt.show()
    print(f"{np.unique(model_results['soc'])}")

if __name__ == "__main__":
    root = Path.cwd().resolve()
    raw_data_dir = root / "data" / "raw"
    processed_data_dir = root / "data" / "processed"
    wltp_file = processed_data_dir / "MLP001_wltp_25degC_record.csv"
    param_file = processed_data_dir / "MLP001_params.csv"
    ocv_file = processed_data_dir / "MLP001_ocv.csv"
    main(cycler_file=wltp_file, ocv_file=ocv_file, param_file=param_file)
