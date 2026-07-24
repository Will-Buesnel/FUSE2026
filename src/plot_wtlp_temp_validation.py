"""Will Buesnel, Jul 26."""

from pathlib import Path
import pandas as pd
import numpy as np

from models.coupled import CoupledModel, ThermalModel
from models.electrical import ElectricalModel
from models.parameters import get_all_parameter_interpolants, format_interpolants
from utils import get_drive_cycle_df, make_plot, get_errors_by_resampling
from cells import Cell
import matplotlib.pyplot as plt

def run_simulation_from_exp(
        exp_df: pd.DataFrame,
        ocv_df: pd.DataFrame,
        param_df: pd.DataFrame,
        cell: Cell,
        T_inf_degC: float = 25.0,
        y0: list = [1,0,0,25], # default initial state: soc=1, v_rc1=0, v_rc2=0, T=25 deg C
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

    experiment_times = exp_df["Elapsed Time[h]"].to_numpy() * 3600  # convert to seconds.
    experiment_currents = -exp_df["Current(A)"].to_numpy() # flip sign of curret to match equations given.


    # define current funciton that interpolates the current from the experiment data.
    def current_func(t):
        return np.interp(t, experiment_times, experiment_currents)
    

    elec_model = ElectricalModel()
    thermal_model = ThermalModel(c=cell.c, h=cell.h, T_inf_degC=T_inf_degC) # hardcoding T_inf for now; could be changed in future.
    coupled_model = CoupledModel(elec_model, thermal_model) # hardcoding this file to take coupled model for now; could be changed in future..

    param_interpolants = format_interpolants(get_all_parameter_interpolants(param_df, ocv_df))
    # set the interpolants as funcs for the model
    for name, interpolant in param_interpolants.items():
        setattr(elec_model, f"_{name}_interp", interpolant) 
    elec_model.max_capacity_As = cell.capacity_Ah * 3600 # Ah to As

    
    return coupled_model.simulate(
        y0=y0,
        current_func= current_func, 
        t_max=t_max,
        max_capacity_Ah=cell.capacity_Ah,
        **kwargs
    )


def main1(cycler_file, ocv_file, param_file):
    exp_df = get_drive_cycle_df(cycler_file)
    ocv_df = pd.read_csv(ocv_file)
    param_df = pd.read_csv(param_file)

    mlp_cell = Cell(
        name="MLP001",
        capacity_Ah=2.2,
        c = 42.9, # J K^-1
        h = 3.59, # J K^-1
        c_p = 887, # J kg^-1 K^-1
        rho = 2682 # kg m^-3
    )


    model_results = run_simulation_from_exp(
        exp_df=exp_df,
        ocv_df=ocv_df,
        param_df=param_df,
        cell=mlp_cell,
        T_inf_degC=25,
        y0=[1.0, 0, 0, 25],
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


    # plot temperature vs time, for debugging purposes.
    import matplotlib.pyplot as plt
    plt.plot(
        model_results["t"] * 1/3600,
        model_results["T"],
        label="Temperature (deg C)"
    )
    plt.show()


if __name__ == "__main__":
    root = Path.cwd().resolve()
    raw_data_dir = root / "data" / "raw"
    processed_data_dir = root / "data" / "processed"
    wltp_file = processed_data_dir / "MLP001_wltp_25degC_record.csv"
    param_file = processed_data_dir / "MLP001_params.csv"
    ocv_file = processed_data_dir / "MLP001_ocv.csv"
    main1(cycler_file=wltp_file, ocv_file=ocv_file, param_file=param_file)
    

