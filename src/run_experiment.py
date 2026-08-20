"""
Will Buesnel, Aug 26
Run an experiment given a parameter set, data etc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import get_path_to_data_results_dir, get_path_to_data_dir, make_plot
from models.parameters import get_initial_soc

from simulation import *

def initialise_simulator(exp_df: pd.DataFrame, ocv_df: pd.DataFrame):
    # load in parameter datasets.
    processed_data_dir = get_path_to_data_dir() / "processed"
    param_df = pd.read_csv(processed_data_dir / "MLP001_params.csv")
    entropy_df = pd.read_csv(processed_data_dir / "entropy_data_cell1.csv")

    cell = Cell.get_standard_cell(entropyfunc=EntropyCoeffFunc(entropy_df))

    return Simulator(cycler_df=exp_df, param_df=param_df, ocv_df=ocv_df, cell=cell)


def run_sim(start_idx, stop_idx):
    exp_df = pd.read_csv(get_path_to_data_dir() / "processed" / "MLP001_wltp_25degC_record_deq.csv").iloc[start_idx:stop_idx]
    ocv_df = pd.read_csv(get_path_to_data_dir() / "processed" / "MLP001_ocv.csv")
    
    sim = initialise_simulator(exp_df, ocv_df)

    initial_temp = 25.0  # assume the cell starts at ambient temperature
    initial_soc = get_initial_soc(initial_temp=initial_temp, initial_v_cell=exp_df["Voltage(V)"].iloc[0], ocv_df=ocv_df)
    print(f"Initial SOC: {initial_soc:.4f}")
    sim.y0 = [initial_soc, 0.0, 0.0, initial_temp]  # initial state: [soc, v_rc1, v_rc2]
    times_s = exp_df["deq_Elapsed Time[h]"].to_numpy() * 3600  # convert hours to seconds

    # try using diffrax for the simulation
    sim.set_solver("diffrax", method="Kvaerno3")
    sim_results = sim.run_simulation(t_eval = times_s, pbar=False)

    sim_results["error [V]"] = abs(sim_results["v_cell [V]"] - exp_df["Voltage(V)"].to_numpy())
    model_results = [
        sim_results["t [s]"] / 3600,  # convert seconds to hours for plotting
        sim_results["v_cell [V]"],
        sim_results["error [V]"]
    ]
    make_plot(exp_df = exp_df, model_results = model_results)
    plot_diagnostics(sim_results)

def plot_diagnostics(sim_results: dict):
    """
    Plot diagnostics for the simulation results.
    """
    plt.figure(figsize=(12, 8))

    # Plot SOC
    plt.subplot(2, 2, 1)
    plt.plot(sim_results["t [s]"], sim_results["soc"])
    plt.title("State of Charge (SOC)")
    plt.xlabel("Time [s]")
    plt.ylabel("SOC")

    # Plot Cell Voltage
    plt.subplot(2, 2, 2)
    plt.plot(sim_results["t [s]"], sim_results["v_cell [V]"])
    plt.title("Cell Voltage")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")

    # Plot RC Voltages
    plt.subplot(2, 2, 3)
    plt.plot(sim_results["t [s]"], sim_results["v_rc1 [V]"], label="v_rc1")
    plt.plot(sim_results["t [s]"], sim_results["v_rc2 [V]"], label="v_rc2")
    plt.title("RC Voltages")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.legend()

    # Plot Error
    plt.subplot(2, 2, 4)
    plt.plot(sim_results["t [s]"], sim_results["error [V]"])
    plt.title("Error between Model and Experiment")
    plt.xlabel("Time [s]")
    plt.ylabel("Error [V]")

    plt.tight_layout()
    plt.show()

def main():
    # run the simulation for the first 1000 rows of the experiment data
    run_sim(30000, 80000)

if __name__ == "__main__":
    main()

    

