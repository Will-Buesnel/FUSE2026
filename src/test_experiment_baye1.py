"""Will Buesnel, Aug 26."""

from pathlib import Path
import pandas as pd
import numpy as np
import torch

from models.coupled import CoupledModel, ThermalModel
from models.electrical import ElectricalModel
from models.parameters import get_all_parameter_interpolants, format_interpolants, get_parameter_function
from models.local_stats import GibbsKernel
from utils import get_drive_cycle_hours_col, make_plot, get_errors_by_resampling, add_zoom_inset, plot_matrix, plot_2d_matrix
from cells import Cell
import matplotlib.pyplot as plt


def run_simulation_from_exp(
        exp_df: pd.DataFrame,
        ocv_df: pd.DataFrame,
        param_df: pd.DataFrame,
        cell: Cell,
        gauss_interps: list[tuple[str, object]] = None, 
        T_inf_degC: float = 25.0,
        y0: list = [1,0,0,25],# default initial state: soc=1, v_rc1=0, v_rc2=0, T=25 deg C
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
    # look for tspan in kwargs, if it exists then set t_max to the end of the tsapn.
    if "t_span" in kwargs:
        t_max = kwargs["t_span"][1]
    else:
        t_max = exp_df["Elapsed Time[h]"].to_numpy()[-1] * 3600  # convert to seconds.

    experiment_times = exp_df["Elapsed Time[h]"].to_numpy() * 3600  # convert to seconds.
    experiment_currents = -exp_df["Current(A)"].to_numpy() # flip sign of curret to match equations given.


    # define current funciton that interpolates the current from the experiment data.
    def current_func(t):
        return np.interp(t, experiment_times, experiment_currents)
    

    elec_model = ElectricalModel()
    thermal_model = ThermalModel(c=cell.c, h=cell.h, T_inf_degC=T_inf_degC) # hardcoding T_inf for now; could be changed in future.
    coupled_model = CoupledModel(elec_model, thermal_model) # hardcoding this file to take coupled model for now; could be changed in future..

    param_interpolants = get_all_parameter_interpolants(param_df, ocv_df)

    for name, kernel in gauss_interps:
        # create a new ParameterFunction with Gaussian noise for the specified parameter
        param_interpolants[name] = get_parameter_function(param_df, name, kernel, noise=1e-4)

    param_interpolants = format_interpolants(param_interpolants)

    # set the interpolants as funcs for the model
    for name, interpolant in param_interpolants.items():
        setattr(elec_model, f"_{name}_interp", interpolant) 
    elec_model.max_capacity_As = cell.capacity_Ah * 3600 # Ah to As

    # setup entropy coefficient function from entropy_df
    thermal_model.entropy_coeff_func = cell.entropy_coeff_func

    
    return coupled_model.simulate(
        y0=y0,
        t_max=t_max,
        current_func = current_func,
        **kwargs
    )

def main(cycler_file, ocv_file, param_file, entropy_file, gauss_interps=None, **kwargs):
    exp_df = get_drive_cycle_hours_col(cycler_file)
    ocv_df = pd.read_csv(ocv_file)
    param_df = pd.read_csv(param_file)
    entropy_df = pd.read_csv(entropy_file)

    # Define cell properties
    lp_cell = Cell(
        name="MLP001",
        capacity_Ah=2.2,
        c = 42.9, # J K^-1
        h = 3.59, # J K^-1
        c_p = 887, # J kg^-1 K^-1
        rho = 2682, # kg m^-3,
        entropy_coeff_func = lambda soc: np.interp(soc, entropy_df["SOC"].to_numpy(), entropy_df["Entropic_Coefficient"].to_numpy())
    )

    # Run simulation
    results1 = run_simulation_from_exp(exp_df, ocv_df, param_df, lp_cell, gauss_interps=gauss_interps, **kwargs)
    results2 = run_simulation_from_exp(exp_df, ocv_df, param_df, lp_cell, gauss_interps=gauss_interps, **kwargs)
    results3 = run_simulation_from_exp(exp_df, ocv_df, param_df, lp_cell, gauss_interps=gauss_interps, **kwargs)


    # now plot the three lines, should see some stochastic differences.
    def plot_voltage()->plt.axes:

        plt.figure(figsize=(10, 6))
        plt.plot(results1["t"], results1["v_cell"], label="Run 1",color='tab:blue', alpha=0.8)
        plt.plot(results2["t"], results2["v_cell"], label="Run 2", color='tab:orange', alpha=0.8)
        plt.plot(results3["t"], results3["v_cell"], label="Run 3", color='tab:green', alpha=0.8)


        plt.plot(exp_df["Elapsed Time[h]"] * 3600, exp_df["Voltage(V)"], label="Experimental", color='tab:cyan', linestyle='--',alpha=0.4)
        plt.xlabel("Time (s)")
        plt.ylabel("Cell Voltage (V)")
        plt.title("Stochastic Simulations")
        # add text below figure if gauss_interps is not None, to indicate which parameters have stochasticity.
        if gauss_interps is not None:
            plt.figtext(0.5, -0.1, f"Stochastic parameters: {', '.join([name for name, _ in gauss_interps])}", ha="center", fontsize=10)
        plt.legend()
        add_zoom_inset(plt.gca(),
                        xs = [
                            results1["t"], 
                            results2["t"], 
                            results3["t"],
                            exp_df["Elapsed Time[h]"] * 3600],
                        ys = [
                            results1["v_cell"],
                            results2["v_cell"],
                            results3["v_cell"],
                            exp_df["Voltage(V)"],
                        ],
                        colours = [
                            'tab:blue',
                            'tab:orange',
                            'tab:green',
                            'tab:cyan'
                        ],
                        alphas = [
                            .8,
                            .8,
                            .8,
                            0.4
                        ],
                        x_range = (2000, 3000),
                        y_range = (4.1, 4.15)
        )
        plt.legend()
        return plt.gca()

    def plot_resistances():
        plt.figure(figsize=(10, 6))
        plt.plot(results1["t"], results1["r0"], label="Run 1",color='tab:blue',alpha=0.7)
        plt.plot(results2["t"], results2["r0"], label="Run 2", color='tab:orange',alpha=0.7)
        plt.plot(results3["t"], results3["r0"], label="Run 3", color='tab:green',alpha=0.7)
        plt.xlabel("Time (s)")
        plt.ylabel("R0 (Ohm)")
        plt.title("Stochastic Simulations of R0")
        plt.legend()
        return plt.gca()

    ax = plot_voltage()
    plt.savefig(Path.cwd().resolve() / "data" / "processed" / "stochastic_simulation_voltage.pdf", dpi=300)
    ax = plot_resistances()
    plt.savefig(Path.cwd().resolve() / "data" / "processed" / "stochastic_simulation_r0.pdf", dpi=300)


if __name__ == "__main__":

    import matplotlib as mpl
    mpl.rcParams["axes.labelsize"] = 18      # x and y axis labels
    mpl.rcParams["xtick.labelsize"] = 16     # x-axis tick labels
    mpl.rcParams["ytick.labelsize"] = 16     # y-axis tick labels
    mpl.rcParams["legend.fontsize"] = 16   # legend text
    mpl.rcParams["legend.title_fontsize"] = 18
    mpl.rcParams["axes.titlesize"] = 20      # plot title


    root = Path.cwd().resolve()
    raw_data_dir = root / "data" / "raw"
    processed_data_dir = root / "data" / "processed"
    wltp_file = processed_data_dir / "MLP001_wltp_25degC_record.csv"
    param_file = processed_data_dir / "MLP001_params.csv"
    ocv_file = processed_data_dir / "MLP001_ocv.csv"
    entropy_file = processed_data_dir / "entropydata_cell1.csv"


    def lengthscale_func_2d(x):
        soc = x[:, 0]
        temp = x[:, 1]  # raw temperature, e.g. 5-40
        return 0.0025 + 0.0005 * torch.sigmoid(5000 * (soc - 0.3)) + 0.5 * temp
    
    gauss_interps = [("R0 [Ohm]", GibbsKernel(input_dim=2, lengthscale_fn=lengthscale_func_2d,variance=1e-5))]
    
    main(cycler_file=wltp_file, ocv_file=ocv_file, param_file=param_file, entropy_file=entropy_file, gauss_interps=gauss_interps, slow=False, verbose=False, pbar=True, max_step=10, atol=1e-6, rtol=1e-3,t_span=(0, 78000))

    # can I pyro-ify this?
    