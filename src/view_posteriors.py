from bayesianModel import view_post_distributions, generate_standard_simulator
from utils import get_path_to_figures_dir, set_rc_params, get_path_to_data_results_dir
import torch
from utils import add_zoom_inset, convert_fig_size_cm_to_inches
from models.parameters import ParameterFunction, ParameterInterpolator

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def main():
    set_rc_params()
    # plot the traces of the samples
    samples = torch.load(get_path_to_data_results_dir() / "MC_testing/AMH_Gibbs_reducedSS_indexes:0:70000_warmup:700_samples:700_chains:3_adapt_600_thisactualgoodone/post_samples.pt")
    # construct two plots on two diff figures.
    fig, ax = plt.subplots(figsize=convert_fig_size_cm_to_inches((22.47, 9)))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    plot_obs_noise(samples["obss"], ax)
    plt.tight_layout()
    plt.savefig(get_path_to_figures_dir() / "observation_noise_distribution.pdf", dpi=300)
    fig, ax = plt.subplots(figsize=convert_fig_size_cm_to_inches((22.47, 9)))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    plot_r0_eps(samples["eps_R0 [Ohm]_sample"], ax)
    plt.tight_layout()
    plt.savefig(get_path_to_figures_dir() / "observation_noise_and_r0_eps_distribution.pdf", dpi=300)



def plot_obs_noise(obs_noise: torch.Tensor ,ax: plt.Axes):
    """
    Plot the observation noise distribution.
    """

    #generate actual results from the simulator to compare the observation noise distribution to the actual results.
    simulator = generate_standard_simulator(stochastic=False, use_deq=False)
    exp_xs = simulator.exp_df["deq_Elapsed Time[h]"].to_numpy()
    exp_ys = simulator.exp_df["Voltage(V)"].to_numpy()
    ax.plot(exp_xs, exp_ys, label="Physical Experiment", color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2], linestyle="-")

    mean = obs_noise.mean(axis=0)
    lower = np.percentile(obs_noise, 2.5, axis=0)
    upper = np.percentile(obs_noise, 97.5, axis=0)

    x = np.arange(obs_noise.shape[1]) / 3600 # convert to hours
    ax.plot(x, mean, label="Mean")
    ax.fill_between(x, lower, upper, alpha=0.3, label="95% CI")
    ax.set_title("Posterior Observation Noise")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Voltage [V]")
    ax.legend()

    # add a zoomed inset to the plot
    zoom_ax =add_zoom_inset(ax, xs = [x,exp_xs], ys = [mean, exp_ys],
                    colours = [ plt.rcParams['axes.prop_cycle'].by_key()['color'][0] , plt.rcParams['axes.prop_cycle'].by_key()['color'][2]],
                    alphas = [1,1],
                    x_range = (0.925, 0.95),
                    y_range = (4.013, 4.14)
                    )
    zoom_ax.fill_between(x, lower, upper, alpha=0.3, label="95% CI")



    

def plot_r0_eps(r0_eps: torch.Tensor, ax: plt.Axes):
    """
    Plot the R0 and epsilon distributions.
    """

    # generate standard simulator to get the original R0 values for comparison; sett if they fit within the confidence intervals.
    socs = np.linspace(0.1, 1, 100)

    simulator = generate_standard_simulator(stochastic=False)
    r0_interpolator = simulator.param_interpolants["R0 [Ohm]"]
    datapoint_x_vals = simulator.generate_general_x_vals()
    og_interpolated_r0 = r0_interpolator(soc=socs, T = 25) * 1000  # convert to mOhm

    zeros = np.zeros_like(datapoint_x_vals[:, 0])
    param_df_25_deg_indexes = simulator.param_df[simulator.param_df["Temperature_degC"] == 25].index.to_numpy()

    mean = r0_eps.mean(axis=0)
    lower = np.percentile(r0_eps, 2.5, axis=0)
    upper = np.percentile(r0_eps, 97.5, axis=0)

    eps_interp_mean_vals = zeros.copy()
    eps_interpolator_lower_vals = zeros.copy()
    eps_interpolator_upper_vals = zeros.copy()

    eps_interp_mean_vals[param_df_25_deg_indexes] = mean
    eps_interpolator_lower_vals[param_df_25_deg_indexes] = lower
    eps_interpolator_upper_vals[param_df_25_deg_indexes] = upper

    eps_interpolator_mean = ParameterInterpolator(socs = datapoint_x_vals[:, 1], temps = datapoint_x_vals[:, 0], values = eps_interp_mean_vals)
    eps_interpolator_lower = ParameterInterpolator(socs = datapoint_x_vals[:, 1], temps = datapoint_x_vals[:, 0], values = eps_interpolator_lower_vals)
    eps_interpolator_upper = ParameterInterpolator(socs = datapoint_x_vals[:, 1], temps = datapoint_x_vals[:, 0], values = eps_interpolator_upper_vals)

    temps = np.full_like(socs, 25)
    r0_mean = og_interpolated_r0 + eps_interpolator_mean(soc=socs, T = temps) * 1000
    r0_lower = og_interpolated_r0 + eps_interpolator_lower(soc=socs, T = temps) * 1000
    r0_upper = og_interpolated_r0 + eps_interpolator_upper(soc=socs, T = temps) * 1000

    
    ax.plot(socs, r0_mean, label="Mean")
    ax.fill_between(socs, r0_lower, r0_upper, alpha=0.3, label="95% CI")
    ax.plot(socs, og_interpolated_r0, label=r"Original R_0", linestyle="--", color="black")
    ax.set_title("Posterior Resistance Distribution")
    ax.set_xlabel("State of Charge (SOC)")
    ax.set_ylabel(r"Resistance [m$\Omega$]")
    ax.legend()


if __name__ == "__main__":
    main()