"""
Will Buesnel, Aug 26

File to numerically solve for the best values of initial Soc and overall max capacity of the battery,
given the values of steady states between pulses. This comes from there being error in the exact initial soc and capacity of a tested cell.

"""
import numpy as np
from scipy.signal import savgol_filter
from pathlib import Path
import pandas as pd
from plot_wtlp_temp_validation import run_simulation_from_exp

def find_steady_states(experiment_data, window_size=100, threshold=1e-3):

    """
    Given the experiment data, find the steady states between pulses.
    This is done by finding the points where the derivative of the voltage is close to zero for a sustained period of time.
    Once found, take them midpoints.
    """
    
    # Smooth the voltage data to reduce noise
    smoothed_voltage = savgol_filter(experiment_data['voltage'], window_length=51, polyorder=3)

    # Calculate the derivative of the smoothed voltage
    voltage_derivative = np.gradient(smoothed_voltage)

    steady_states = []
    for i in range(len(voltage_derivative) - window_size):
        # Check if the derivative is close to zero for a sustained period
        if np.all(np.abs(voltage_derivative[i:i + window_size]) < threshold):
            # If so, take the midpoint of this window as a steady state
            steady_state_index = i + window_size // 2
            steady_states.append((steady_state_index, smoothed_voltage[steady_state_index]))
        i += window_size  # Skip ahead to avoid overlapping windows

    return steady_states

def get_error_at_steady_states(simulation_results: np.ndarray, experiment_data: np.ndarray, steady_states: list):
    """
    Given the simulation results and the experimental data, calculate the error at the steady states.
    The error is defined as the difference between the simulated voltage and the experimental voltage at the steady states.
    The two arrays must be of the exact same length, and be correlated in time.
    """
    errors = []
    for index, _ in steady_states:
        sim_voltage = simulation_results[index]
        exp_voltage = experiment_data[index]
        error = sim_voltage - exp_voltage
        errors.append(error)
    return np.array(errors)


def main():
    # load in data:
    processed_dir = Path.cwd().resolve() / "data" / "processed"
    experiment_data = pd.read_csv(processed_dir / "MLP001_wltp_25degC_record_deq.csv")

    initial_socs = np.linspace(0.8, 1.0, 20)
    cell_capacities = np.linspace(2.0, 2.5, 20)
    ambient_temperatures = np.linspace(22.5, 27.5, 5)

    run_simulation_from_exp(
        exp_df=experiment_data,
        ocv_df=processed_dir / "MLP001_ocv.csv",
        param_df=processed_dir / "MLP001_params.csv",
        entropy_df=processed_dir / "MLP001_entropy.csv",
        cell_name="MLP001",
        T_inf_degC=25,
        y0=[1.0, 0, 0, 25],
        results_dir_filepath=processed_dir,
    )


