"""
Will Buesnel, Aug 26.
File to visualise the default simulation results against the some arbitary new results section.
Idea is that these are taken from MC Simulation results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from utils import safe_cholesky, set_rc_params, convert_pred_samples_to_df, get_path_to_figures_dir, get_path_to_data_dir
from simulation import Simulator
from run_experiment import initialise_simulator
from bayesianModel import generate_standard_simulator
from models.local_stats import GibbsKernel, lengthscale_func_2d
from models.parameters import ParameterFunction

def import_sample_from_results(sample_name: str, method: str ="Last X", n: int = 10)-> np.ndarray:
    """
    Function to take some level of the sampling from the results df, and return the mean ofthem over some delimit so we can add them to the simulator.
    """
    filepath = get_path_to_data_dir() / "results" / "MC_results" / sample_name
    base_df = convert_pred_samples_to_df(torch.load(filepath))
    return_df = pd.DataFrame()
    if method == "All":
        return_df = base_df
    elif method == "Last X":
        for chain in base_df["Chain"].unique():
            chain_df = base_df[base_df["Chain"] == chain]
            return_df = pd.concat([return_df, chain_df.tail(n)])
    return _manually_take_mean_of_df_rows(return_df.drop(columns=["Iteration", "Chain"])) # return the mean of the samples, as a series. therefore we should get one row for each parameter.


def _manually_take_mean_of_df_rows(df: pd.DataFrame) -> pd.Series:
    """
    Function to take the mean of the rows of a df, and return a series with the mean values.
    can't use a built in function because the df has some non-numeric columns, so we need to manually take the mean of the numeric columns.
    This is due to my bad implementation, but imo this way still makes more sense (i.e. rather than storing each r0 value in a separate column)
    """
    # intialise a new df to store the mean values, with the correct columns:
    mean_series = pd.Series(index=df.columns, dtype=object)

    for col in df.columns:
        first_value = df[col].iloc[0]

        if isinstance(first_value, (list, np.ndarray)) and np.ndim(first_value) > 0:  #
            d2_array = np.array(df[col].tolist())
            mean_array = [d2_array.mean(axis=0)]
            mean_series[col] = pd.Series(mean_array, dtype=object)
        else:
            mean_series[col] = df[col].mean()
    return mean_series

def get_r0_eps(param_df: pd.DataFrame = None, mc_filename: str = "test.pt", deg_25_only: bool = True) -> np.ndarray:
    if deg_25_only:
        add_eps_stand = torch.zeros(len(param_df), dtype=torch.float64)
        deg_25_indexes = param_df[param_df["Temperature_degC"] == 25].index.to_numpy()
        add_eps_stand[deg_25_indexes] = torch.tensor(import_sample_from_results(mc_filename, method="All")["eps_R0 [Ohm]_standardised"], dtype=torch.float64)
    else:
        add_eps_stand = import_sample_from_results(mc_filename, method="All")["eps_R0 [Ohm]_standardised"]
    var = import_sample_from_results(mc_filename, method="All")["var_scaled"] * 1e-6
    kernel = GibbsKernel(input_dim=2, lengthscale_fn=lengthscale_func_2d, variance=var)

    # forward the kernel to get the covariance matrix for the parameter function:
    X = np.column_stack([param_df["Temperature_degC"].to_numpy(), param_df["SOC"].to_numpy()])
    K = kernel.forward(torch.tensor(X, dtype=torch.float64))
    L = safe_cholesky(K)
    return L @ torch.tensor(add_eps_stand, dtype=torch.float64).T

def main():
    set_rc_params()
    param_df = pd.read_csv(get_path_to_data_dir() / "processed" / "MLP001_params.csv")
    r0_eps = get_r0_eps(param_df, mc_filename="test6.pt", deg_25_only=True)  # get the epsilons for R0 from the MC results, and apply them to the simulator.
    phys_exp_df = pd.read_csv(get_path_to_data_dir() / "processed" / "MLP001_wltp_25degC_record_shortened.csv")
    phys_exp_df = phys_exp_df.iloc[:int(len(phys_exp_df) * 0.9)]  # shorten physical experiment to remove the last pulse. -i.e. only take the first 8/9s of the rows.
    # shorten physical experiment to remove the last pulse. -i.e. only take the first 8/9s of the rows.
    eval_times = phys_exp_df["deq_Elapsed Time[h]"].to_numpy() * 3600  # convert to seconds

    sim = generate_standard_simulator(stop_idx=len(eval_times)-1, use_deq=False)

    res_std = sim.run_simulation(pbar=True, t_eval=eval_times)

    # add the gaussian-process-simulated epsilons to the simulator, and run the simulation again:
    sim.param_interpolants["R0 [Ohm]"].set_eps_interpolator(temps = param_df["Temperature_degC"].to_numpy(),socs = param_df["SOC"].to_numpy(), eps_sample = r0_eps.detach().numpy())
    sim_results = sim.run_simulation(pbar=True, t_eval=eval_times)


    # compare the errors:

    orig_abs_error = np.abs(res_std["v_cell [V]"] - phys_exp_df["Voltage(V)"].to_numpy())
    new_abs_error = np.abs(sim_results["v_cell [V]"] - phys_exp_df["Voltage(V)"].to_numpy())
    print(f"Original mean absolute error: {orig_abs_error.mean()}")
    print(f"New mean absolute error: {new_abs_error.mean()}")

    print(f"Original overall error: {np.linalg.norm(orig_abs_error)}")
    print(f"New overall error: {np.linalg.norm(new_abs_error)}")

    # plot the simulation results + physicaly experiment. Three subplots; one for the current, one for voltage, & one for the error.



    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(eval_times, phys_exp_df["Current(A)"], label="Physical Experiment", color="green", linewidth=2)
    axs[0].plot(eval_times, res_std["I [A]"], label="Default Simulation", color="blue", linestyle="--")
    #axs[0].plot(eval_times, sim_results["I [A]"], label="Simulation with new interpolants", color="red", linestyle="--")
    axs[0].set_ylabel("Current [A]")
    axs[0].legend()


    axs[1].plot(eval_times, phys_exp_df["Voltage(V)"], label="Physical Experiment", color="black", linewidth=2)
    axs[1].plot(eval_times, res_std["v_cell [V]"], label="Default Simulation", color="blue", linestyle="--")
    #axs[1].plot(eval_times, sim_results["v_cell [V]"], label="Simulation with new interpolants", color="red", linestyle="--")
    axs[1].set_ylabel("Voltage [V]")
    axs[1].legend()

    axs[2].plot(eval_times, orig_abs_error, label="Default Simulation Error", color="brown", linestyle="--")
    axs[2].plot(eval_times, new_abs_error, label="Simulation with new interpolants Error", color="red", linestyle="--")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Absolute Error [V]")
    axs[2].legend()

    plt.tight_layout()
    plt.show()
    





def test_manually_take_mean_of_df_rows():
    # create a test df with some columns that are lists and some that are not:
    test_df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [[1, 2], [3, 4], [5, 6]],
    })
    mean_series = _manually_take_mean_of_df_rows(test_df)
    assert np.isclose(mean_series["A"], 2)
    assert mean_series["B"].iloc[0][0] == 5
    assert mean_series["B"].iloc[0][1] == 4

if __name__ == "__main__":
    main()