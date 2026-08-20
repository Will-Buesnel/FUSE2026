"""
Will Buesnel, Aug 26

File to numerically solve for the best values of initial Soc and overall max capacity of the battery,
given the values of steady states between pulses. This comes from there being error in the exact initial soc and capacity of a tested cell.

"""
import numpy as np
from scipy.signal import savgol_filter
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from utils import set_rc_params, get_path_to_figures_dir, get_path_to_data_dir
from simulation import Simulator
from run_experiment import initialise_simulator


def find_steady_states(experiment_data, column_name, window_size=100, threshold=1e-3):

    """
    Given the experiment data, find the steady states between pulses.
    This is done by finding the points where the derivative of the voltage is close to zero for a sustained period of time.
    Once found, take them midpoints.
    """
    
    # Smooth the voltage data to reduce noise
    smoothed_voltage = savgol_filter(experiment_data[column_name], window_length=51, polyorder=3)

    # Calculate the derivative of the smoothed voltage
    voltage_derivative = np.gradient(smoothed_voltage)

    steady_states = []
    index=window_size//2
    while index < len(voltage_derivative) - window_size // 2:
        # Check if the derivative is close to zero for a sustained period
        if np.all(np.abs(voltage_derivative[index-window_size//2:index + window_size//2]) < threshold):
            # If so, take the midpoint of this window as a steady state
            steady_state_index = index + window_size // 2
            steady_states.append((steady_state_index, smoothed_voltage[steady_state_index]))
            index += window_size  # Skip ahead to avoid overlapping windows
        else:
            index += 1

    # finally, remove any steady states that are of the same value as the previous one, as this is likely just a flat region of the voltage curve
    filtered_steady_states = []
    for i in range(len(steady_states)):
        if i == 0 or steady_states[i][1] != steady_states[i-1][1]:
            filtered_steady_states.append(steady_states[i])
    return filtered_steady_states

def get_error_at_indexes(simulation_results: np.ndarray, experiment_data: np.ndarray, indexes: list):
    """
    Given the simulation results and the experimental data, calculate the error at the specified indexes.
    The error is defined as the difference between the simulated voltage and the experimental voltage at those indexes.
    The two arrays must be of the exact same length, and be correlated in time.
    """
    errors = []
    sim_voltages = simulation_results[indexes]
    exp_voltages = experiment_data[indexes]
    for sim_v, exp_v in zip(sim_voltages, exp_voltages):
        errors.append(sim_v - exp_v)
    return np.array(errors)


def plot_steady_states(experiment_data, steady_states, column_name, ax=None):
    """
    Plot the experiment data and the steady states.
    """
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
    ax.plot(experiment_data[column_name], label='Experiment Voltage')
    ax.scatter([idx for idx, _ in steady_states], [experiment_data[column_name].iloc[i] for i, _ in steady_states], color='red', label='Steady States')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Voltage [V]')
    ax.set_title('Experiment Data with Steady States')
    ax.legend()


def paramsweep_ics(pbar: bool = False):
    set_rc_params()
    # load in data:
    path_to_data = get_path_to_data_dir()
    processed_dir = path_to_data / "processed"
    experiment_data = pd.read_csv(processed_dir / "MLP001_wltp_25degC_record_deq.csv")  # limit to first 100k rows for speed while testing
    ocv_data = pd.read_csv(processed_dir / "MLP001_ocv.csv")

    experiment_data_voltage_column = 'Voltage(V)'
    steady_states = find_steady_states(experiment_data, column_name=experiment_data_voltage_column, window_size=5000)

    #last_steady_state_index = steady_states[-1][0]
    #first_steady_state_index = steady_states[0][0]

    #experiment_data_adj = experiment_data.iloc[first_steady_state_index:last_steady_state_index + 1]  # limit the data to the steady states, as rest of data is not relevant here
    # reset the times column to start at 0, as this is what the simulator expects
    #experiment_data_adj["deq_Elapsed Time[h]"] = experiment_data_adj["deq_Elapsed Time[h]"] - experiment_data_adj["deq_Elapsed Time[h]"].iloc[0]
    # reset the index to start at 0, as this is what the simulator expects
    # find adjusted steady states for this.
    #experiment_data_adj = experiment_data_adj.reset_index(drop=True)


    #adjusted_steady_states = [(idx - first_steady_state_index, val) for idx, val in steady_states if first_steady_state_index <= idx <= last_steady_state_index]
    initial_socs = np.linspace(0.98, 1.0, 4)
    cell_capacities = np.linspace(2.15, 2.21, 4)  # in Ah
    ambient_temperatures = np.linspace(24, 26, 5)

    # parameter sweep over model to find set of initial conditions that minimises error at these steady states.
    sim_count = 0
    total_sims = len(initial_socs) * len(cell_capacities) * len(ambient_temperatures)
    # instantiate simulator
    current_error = np.inf
    sim = initialise_simulator(experiment_data, ocv_data)
    evaluate_times = experiment_data["deq_Elapsed Time[h]"].to_numpy() * 3600  # convert hours to seconds

    # for initial_soc in initial_socs:

    #     for cell_capacity in cell_capacities:

    #         for ambient_temp in ambient_temperatures:

    #             sim.y0 = [initial_soc, 0.0, 0.0, ambient_temp]
    #             sim.set_cell_capacity(cell_capacity)
    #             try:
    #                 res_vcell = sim.run_simulation(pbar=pbar, t_eval=evaluate_times)["v_cell [V]"]
    #             except Exception as e:
    #                 print(f"Error occurred while running simulation: {e}. Skipping this combination of parameters.")
    #                 continue

    #             error_at_steady_states = get_error_at_indexes(res_vcell, experiment_data_adj[experiment_data_voltage_column].to_numpy(), [idx for idx, _ in adjusted_steady_states])

    #             total_error = np.sum(np.abs(error_at_steady_states))

    #             if total_error < current_error:
    #                 current_error = total_error
    #                 best_initial_soc = initial_soc
    #                 best_cell_capacity = cell_capacity
    #                 best_ambient_temp = ambient_temp

    #             print(f"Sim {sim_count}/{total_sims}, Initial SOC: {initial_soc:.4f}, Cell Capacity: {cell_capacity:.4f} Ah, Ambient Temp: {ambient_temp:.2f} degC, Total error at steady states: {total_error:.4f}")
    #             sim_count += 1

    # print(f"Best initial SOC: {best_initial_soc:.4f}, Best cell capacity: {best_cell_capacity:.4f} Ah, Best ambient temp: {best_ambient_temp:.2f} degC, Total error at steady states: {current_error:.4f}")
    sim.y0 = [1, 0.0, 0.0, 25.0]
    res_original = sim.run_simulation(pbar=pbar, t_eval=evaluate_times)["v_cell [V]"]
    sim.set_cell_capacity(2.15)
    sim.y0 = [1, 0.0, 0.0, 26.0]
    res_adjusted = sim.run_simulation(pbar=pbar, t_eval=evaluate_times)["v_cell [V]"]
    steady_st_errors_original = get_error_at_indexes(res_original, experiment_data[experiment_data_voltage_column].to_numpy(), [idx for idx, _ in steady_states])
    steady_st_errors_adjusted = get_error_at_indexes(res_adjusted, experiment_data[experiment_data_voltage_column].to_numpy(), [idx for idx, _ in steady_states])

    percentage_decrease = (np.sum(np.abs(steady_st_errors_original)) - np.sum(np.abs(steady_st_errors_adjusted))) / np.sum(np.abs(steady_st_errors_original)) * 100
    print(f"Percentage decrease in total error at steady states after adjustment: {percentage_decrease:.2f}%")

    # for sense checking, plot the results of the original and adjusted simulations against the experimental data
    # get vector of overall error:
    error_original = abs(res_original - experiment_data[experiment_data_voltage_column].to_numpy())
    error_adjusted = abs(res_adjusted - experiment_data[experiment_data_voltage_column].to_numpy())
    fig, ax = plt.subplots(2,1, figsize=(10, 6))
    ax[0].plot(experiment_data[experiment_data_voltage_column], label='Experiment Voltage', color='black')
    ax[0].plot(res_original, label='Original Simulation', color='tab:blue', alpha=0.7)
    ax[0].plot(res_adjusted, label='Adjusted Simulation', color='tab:orange', alpha=0.7)
    ax[0].scatter([idx for idx, _ in steady_states], [experiment_data [experiment_data_voltage_column].iloc[i] for i, _ in steady_states], color='tab:red', label='Steady States')
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Voltage [V]')
    ax[0].set_title('Experiment Data vs Simulation Results')
    ax[0].legend()

    ax[1].plot(error_original, label='Original Error', color='tab:blue', alpha=0.7)
    ax[1].plot(error_adjusted, label='Adjusted Error', color='tab:orange', alpha=0.7)
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Error [V]')
    ax[1].set_title('Error Analysis')
    ax[1].legend()
    plt.tight_layout()
    fig.savefig(get_path_to_figures_dir() / "steady_state_error_comparison.pdf", dpi=300)
    plt.close(fig)
    
if __name__ == "__main__":
    paramsweep_ics(pbar=True)

    

