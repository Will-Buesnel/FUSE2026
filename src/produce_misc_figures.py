from scipy.fftpack import shift

from bayesianModel import view_post_distributions, generate_standard_simulator, construct_c0_gp_matrix
from utils import plot_mixing, plot_traces, safe_cholesky, set_rc_params, get_path_to_figures_dir
from utils import get_custom_cmap
import torch
from utils import add_zoom_inset, convert_fig_size_cm_to_inches, get_path_to_data_results_dir
from models.parameters import ParameterFunction, ParameterInterpolator
from models.local_stats import GibbsKernel, lengthscale_func_2d, visualise_lengthscale_func_2d
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from utils import get_path_to_data_dir, convert_pred_samples_to_df
from utils import open_pred_samples_as_df
from view_interpolation_error import get_r0_eps

def main():
    set_rc_params()
    # plot the traces of the samples
    # plot lengthscale function
    #_plot_original_voltage_trace_and_error_trace(fig_size_cm=(12, 12))
    #_plot_obs_mixing(fig_size_cm=(9.5, 7.15))
    # _plot_obs_half_normal()
    # _plot_points_and_interpolation_scheme((15, 5))
    _plot_r0_prior_distribution(fig_size_cm=(18.5, 8))
    # _plot_bayesian_diagrams(fig_size_cm=(10, 7))    
    # _plot_new_error_trace(fig_size_cm=(22.5, 27))
    plt.cla()

def plot_inset_cov():

    fig, ax = plt.subplots(figsize=convert_fig_size_cm_to_inches((7, 7.5)))
    visualise_lengthscale_func_2d(ax)

    # --- Inset: covariance matrix in bottom-right corner ---
    sim = generate_standard_simulator()
    kernel = GibbsKernel(input_dim=2, lengthscale_fn=lengthscale_func_2d, variance=1.0)
    X = sim.generate_general_x_vals()
    K = kernel.forward(torch.tensor(X, dtype=torch.float64))

    # [x0, y0, width, height] in axes fraction coordinates (0-1)
    # positioned in bottom-right corner of the main axes
    inset_ax = ax.inset_axes([0.62, 0.08, 0.33, 0.33])

    im = inset_ax.imshow(K.detach().numpy(), cmap=get_custom_cmap())
    inset_ax.set_title('Covariance Matrix', fontsize=9)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])

    # small colorbar for the inset
    cbar = fig.colorbar(im, ax=inset_ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)

    plt.show()
    plt.cla()

def _plot_original_voltage_trace_and_error_trace(fig_size_cm=(3, 10.3)):
    fig, axs = plt.subplots(2, 1, figsize=convert_fig_size_cm_to_inches(fig_size_cm), sharex=True)
    phys_exp_df = pd.read_csv(get_path_to_data_dir() / "processed" / "MLP001_wltp_25degC_record_shortened.csv")
    phys_exp_df = phys_exp_df.iloc[:int(len(phys_exp_df) * 8 / 9)]
    # shorten physical experiment to remove the last pulse. -i.e. only take the first 8/9s of the rows.
    eval_times = phys_exp_df["deq_Elapsed Time[h]"].to_numpy() * 3600  # convert to seconds
    sim = generate_standard_simulator(stop_idx=len(eval_times), use_deq=False)
    
    exp_ys = sim.exp_df["Voltage(V)"].to_numpy()
    results = sim.run_simulation(t_eval=eval_times, pbar=True)
    axs[0].plot(phys_exp_df["deq_Elapsed Time[h]"], exp_ys, label="Experimental Data",alpha=0.7, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
    axs[0].plot(phys_exp_df["deq_Elapsed Time[h]"].to_numpy(), results["v_cell [V]"], label="Original Model", color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
    axs[0].set_ylabel(r"$V_{\text{cell}}$ [V]")
    axs[0].legend()
    #plot the error trace
    error_trace = results["v_cell [V]"] - exp_ys
    axs[1].plot(phys_exp_df["deq_Elapsed Time[h]"], 1000* error_trace, label="Error Trace", color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2], linestyle="-")
    axs[1].legend()
    axs[1].set_xlabel("Time [h]")
    axs[1].set_ylabel("Error [mV]")
    fig.suptitle("Discharge Cycle of Car Battery")
    plt.tight_layout()
    plt.savefig(get_path_to_figures_dir() / "original_voltage_trace_and_error_trace.pdf")
    plt.cla()

def plot_cov_matrix(fig_size_cm=(7, 7.5)):
    # visualise covariance matrix for a given set of parameters
    fig, ax = plt.subplots(figsize=convert_fig_size_cm_to_inches(fig_size_cm))
    # generate a standard simulator to get the parameters for the covariance matrix
    sim = generate_standard_simulator()
    kernel = GibbsKernel(input_dim=2, lengthscale_fn=lengthscale_func_2d, variance=1.0)
    X = sim.generate_general_x_vals()
    K = kernel.forward(torch.tensor(X, dtype=torch.float64))
    im = ax.imshow(K.detach().numpy(), cmap=get_custom_cmap())
    # add a colour bar
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Covariance Matrix')

        # --- Range bracket parallel to the main diagonal pulse ---
    n = K.shape[0]
    i_start, i_end = 23, 46

    # unit vectors: along the diagonal, and perpendicular to it
    diag_dir = np.array([1, 1]) / np.sqrt(2)
    perp_dir = np.array([1, -1]) / np.sqrt(2)  # true perpendicular to the diagonal

    bracket_offset = n * 0.045
    x0, y0 = np.array([i_start, i_start]) + bracket_offset * perp_dir
    x1, y1 = np.array([i_end, i_end]) + bracket_offset * perp_dir

    ax.plot([x0, x1], [y0, y1], color='white', lw=1.2, clip_on=False)

    tick = n * 0.015
    for (x, y) in [(x0, y0), (x1, y1)]:
        tx, ty = np.array([x, y]) + tick * perp_dir
        tx2, ty2 = np.array([x, y]) - tick * perp_dir
        ax.plot([tx, tx2], [ty, ty2], color='white', lw=1.2, clip_on=False)

    # label centered on the bracket midpoint, offset further along the same perpendicular
    mid = (np.array([i_start, i_start]) + np.array([i_end, i_end])) / 2
    label_offset = n * 0.03
    label_pos = mid + (bracket_offset + label_offset) * perp_dir

    ax.text(
        label_pos[0], label_pos[1], "SOC range",
        rotation=45,
        rotation_mode='anchor',
        transform_rotates_text=True,
        ha='center', va='center',   # <-- va='center' keeps it truly centered after rotation
        fontsize=11, color='white', fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(get_path_to_figures_dir() / "covariance_matrix.pdf")
    plt.cla()

def _plot_lengthscale_function(fig_size_cm=(7, 7.5)):
    fig, ax = plt.subplots(figsize=convert_fig_size_cm_to_inches(fig_size_cm))
    visualise_lengthscale_func_2d(ax)
    plt.tight_layout()
    plt.savefig(get_path_to_figures_dir() / "lengthscale_function.pdf")
    plt.cla()

def _plot_obs_mixing(fig_size_cm=(7, 7.5)):
    fig, ax = plt.subplots(figsize=convert_fig_size_cm_to_inches(fig_size_cm))
    samples_df = open_pred_samples_as_df(get_path_to_data_results_dir() / "MC_testing/AMH_Gibbs_reducedSS_indexes:0:70000_warmup:700_samples:700_chains:3_adapt_600_thisactualgoodone/samples.pt")
    num_chains = samples_df["Chain"].nunique()

    for chain in range(1, num_chains+1):
        chain_samples = samples_df[samples_df["Chain"] == chain]["obs_scale"]
        ax.plot(chain_samples.values, label=f"Chain {chain}", alpha=0.8)
    ax.set_title(r"Trace plot for $\sigma_{\text{obs}}$")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\sigma_{\text{obs}}$")
    ax.legend()

    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(get_path_to_figures_dir() / "observation_noise_mixing.pdf")
    plt.cla()

def _plot_obs_half_normal(fig_size_cm=(7, 7.5)):
    # construct a standard half normal distribution
    fig, ax = plt.subplots(figsize=convert_fig_size_cm_to_inches(fig_size_cm))
    x = np.linspace(0, 3, 1000)
    half_normal_pdf = stats.halfnorm.pdf(x, loc=0, scale=1)
    ax.plot(x, half_normal_pdf, label="Half-Normal Distribution", color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
    # fill between line and axis.
    ax.fill_between(x, half_normal_pdf, alpha=0.3)
    ax.set_title(r"$\sigma_{\text{obs}} $ prior")
    ax.set_xlabel(r"$\sigma_{\text{obs}}$")
    ax.set_ylabel("PDF")
    plt.tight_layout()
    plt.savefig(get_path_to_figures_dir() / "half_normal_distribution.pdf")
    plt.cla()

def _plot_r0_prior_distribution(fig_size_cm=(7, 7.5)):
    simulator = generate_standard_simulator(stochastic=False)
    param_df = simulator.param_df
    deg_25_indexes = param_df[param_df["Temperature_degC"] == 25.0].index
    X_orig = param_df.loc[deg_25_indexes, ["Temperature_degC", "SOC"]].values
    # remove first value off X orig, as it is a duplicate of the second value (both are 25.0, 0.05)
    X_orig = X_orig[1:, :]
    
    # Build a finer SOC grid (e.g. 200 points instead of 20) so it resolves ℓ=0.02
    soc_fine = np.linspace(X_orig[:, 1].min(), X_orig[:, 1].max(), 23)  # SOC column
    temp_fine = np.full_like(soc_fine, 25.0)
    X = np.stack([temp_fine, soc_fine], axis=1)  # [Temperature, SOC] order

    kernel = GibbsKernel(input_dim=2, lengthscale_fn=lengthscale_func_2d)
    K = kernel.forward(torch.tensor(X, dtype=torch.float32))
    K = K.detach().numpy() * 1e-5

    L = np.linalg.cholesky(K + 1e-6 * np.eye(K.shape[0]))

    soc_vals = X[:, 1]  # now this is actually SOC, strictly increasing
    mean = np.zeros_like(soc_vals)
    std = np.sqrt(np.diag(K)) * 1000
    r0_interpolated = simulator.param_interpolants["R0 [Ohm]"](soc_vals, T=25.0)

    n_samples = 5
    eps = np.random.randn(L.shape[0], n_samples)
    r0_samples = (L @ eps) * 1000

    mean += r0_interpolated
    mean *= 1000
  
    from scipy.interpolate import PchipInterpolator

    fig, ax = plt.subplots(figsize=convert_fig_size_cm_to_inches(fig_size_cm))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)

    # Denser x grid for smooth interpolation
    soc_smooth = np.linspace(soc_vals.min(), soc_vals.max(), 300)

    # Interpolate mean and std
    mean_smooth = PchipInterpolator(soc_vals, mean)(soc_smooth)
    std_smooth = PchipInterpolator(soc_vals, std)(soc_smooth)

    ax.fill_between(soc_smooth, mean_smooth - 2*std_smooth, mean_smooth + 2*std_smooth,
                    color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], alpha=0.2, label='±2σ')

    for i in range(n_samples):
        sample_smooth = PchipInterpolator(soc_vals, r0_samples[:, i] + mean)(soc_smooth)
        ax.plot(soc_smooth, sample_smooth, alpha=0.5, lw=1, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2], label=r"$R_0$ + Gaussian Process Sample " if i == 0 else None)

    # mean
    ax.plot(soc_smooth, mean_smooth, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], lw=2, label=r"$R_0$(SOC)")
    ax.set_ylabel(r"$R_0$ [m$\Omega$]")
    ax.set_xlabel("SOC")
    ax.set_title(r"$R'_0$ Prior Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(get_path_to_figures_dir() / "r0_prior_distribution.pdf")
    plt.cla()


def _plot_points_and_interpolation_scheme(fig_size_cm=(7, 7.5)):
    # draw a generator object, get param df and r0 interpolator

    simulator = generate_standard_simulator(stochastic=False)
    param_df = simulator.param_df
    r0interpolator = simulator.param_interpolants["R0 [Ohm]"]
    socs = np.linspace(0.05, 1, 100)
    points_25deg = param_df[param_df["Temperature_degC"] == 25.0] 
    r0_values = r0interpolator(socs, T=25.0) * 1000
    datapoints = points_25deg[["SOC", "R0 [Ohm]"]].values 
    datapoints = [[point[0], point[1] * 1000] for point in datapoints]

    fig, ax = plt.subplots(figsize=convert_fig_size_cm_to_inches(fig_size_cm))
    ax.plot(socs, r0_values, color="blue")
    ax.scatter([point[0] for point in datapoints], [point[1] for point in datapoints], label="Known Data", color="red", s=50)
    ax.set_title("Points and Interpolation Scheme")
    ax.set_xlabel("SOC")
    ax.set_ylabel(r"$R_0$ [m$\Omega$]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(get_path_to_figures_dir() / "points_and_interpolation_scheme.pdf")
    plt.cla()


def _plot_bayesian_diagrams(fig_size_cm=(7, 7.5)):
    from scipy.stats import norm
    mu1, sigma1 = 1.5, 0.2 # mean and standard deviation
    mu2, sigma2 = 0, 0.25
    mu3, sigma3 = 1.8, 0.3

    # create a range of x values
    x = np.linspace(-1, 3, 1000)
    # plot via scipy.stats.norm.pdf
    plt.figure(figsize=convert_fig_size_cm_to_inches(fig_size_cm))
    plt.plot(x, norm.pdf(x, mu1, sigma1), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], label=r"Posterior")
    plt.plot(x, norm.pdf(x, mu2, sigma2), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], label=r"Prior")
    plt.plot(x, norm.pdf(x, mu3, sigma3), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2], label=r"True")

    # fill the area under the curves
    plt.fill_between(x, norm.pdf(x, mu1, sigma1), alpha=0.3, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
    plt.fill_between(x, norm.pdf(x, mu2, sigma2), alpha=0.3, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
    plt.fill_between(x, norm.pdf(x, mu3, sigma3), alpha=0.3, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2])
    plt.title(r"Before Inference")
    plt.xlabel("x")
    plt.ylabel("pdf")
    plt.legend()
    plt.tight_layout()
    plt.savefig(get_path_to_figures_dir() / "bayesian_inference_toy_example.pdf")

    # plot after inference, the posterior distribution should be closer to the true distribution

    mu1, sigma1 = 1.7, 0.3 # mean and standard deviation
    mu2, sigma2 = 1.3, 0.25
    mu3, sigma3 = 1.8, 0.3

    plt.figure(figsize=convert_fig_size_cm_to_inches(fig_size_cm))
    plt.plot(x, norm.pdf(x, mu1, sigma1), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], label=r"Posterior")
    plt.plot(x, norm.pdf(x, mu2, sigma2), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], label=r"Prior")
    plt.plot(x, norm.pdf(x, mu3, sigma3), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2], label=r"True")

    # fill the area under the curves
    plt.fill_between(x, norm.pdf(x, mu1, sigma1), alpha=0.3, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
    plt.fill_between(x, norm.pdf(x, mu2, sigma2), alpha=0.3, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
    plt.fill_between(x, norm.pdf(x, mu3, sigma3), alpha=0.3, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2])
    plt.title(r"After Inference")
    plt.xlabel("x")
    plt.ylabel("pdf")
    plt.tight_layout()
    plt.legend()
    plt.savefig(get_path_to_figures_dir() / "bayesian_inference_toy_example_2.pdf", dpi=300)

def _plot_random_walk(fig_size_cm=(7, 7.5)):
    # plot a random walk. This code is all generate by claude (which is pretty obvious from the style of the code).

    np.random.seed(21)        
    n_steps = 10                # number of intermediate points (like 1..10 in the example)
    step_size = 1.0
    
    # random angles for each step -> creates the jagged, non-grid-like path
    angles = np.random.uniform(0, 2 * np.pi, size=n_steps)
    dx = step_size * np.cos(angles)
    dy = step_size * np.sin(angles)
    
    x = np.concatenate([[0], np.cumsum(dx)])
    y = np.concatenate([[0], np.cumsum(dy)])
    
    # ---------------------------------------------------------
    # 2. Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=convert_fig_size_cm_to_inches(fig_size_cm))
    
    # annotate() arrows don't participate in autoscaling, so plot invisible
    # points first to force the axes to the right extent
    ax.scatter(x, y, s=0)
    
    # draw each segment as an arrow (so direction of travel is visible)
    for i in range(len(x) - 1):
        ax.annotate(
            "",
            xy=(x[i + 1], y[i + 1]),
            xytext=(x[i], y[i]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
                lw=2,
                mutation_scale=20,
                shrinkA=8,
                shrinkB=8,
            ),
        )
    
    # number the intermediate points (1 .. n_steps-1), skipping start/end
    # offset each label slightly away from the local path direction to reduce overlap
    for i in range(1, len(x) - 1):
        # direction away from the average of incoming/outgoing segment -> push label outward
        vx = (x[i] - x[i - 1]) + (x[i] - x[i + 1])
        vy = (y[i] - y[i - 1]) + (y[i] - y[i + 1])
        norm = np.hypot(vx, vy)
        if norm < 1e-6:
            vx, vy = 0, 1
            norm = 1
        offset = 0.22
        ax.text(
            x[i] + offset * vx / norm, y[i] + offset * vy / norm, str(i),
            fontsize=13, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], fontweight="bold",
            ha="center", va="center"
        )
    
    # mark initial position
    ax.scatter(x[0], y[0], s=220, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2], zorder=5)
    ax.text(x[0] + 0.2, y[0] - 0.1, "Initial\nPosition",
            fontsize=12, color='black', fontweight="bold", va="center")
    
    # mark final position
    ax.scatter(x[-1], y[-1], s=220, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2], zorder=5)
    ax.text(x[-1] + 0.2, y[-1]-0.05, "Final\nPosition",
            fontsize=12, color='black', fontweight="bold", va="center")

    optimum = (x[-2] - 0.6, y[-2] - 1.0) 
    # mark the optimum with a star + dashed rings to draw the eye to it
    ax.scatter(*optimum, marker="*", s=500, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
           edgecolor="black", linewidth=0.8, zorder=6)
    ax.text(optimum[0] + 0.2, optimum[1], "Optimum",
        fontsize=12, color='black', fontweight="bold", va="center")
    
    ax.set_aspect("equal")
    margin = 0.6
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim(y.min() - margin, y.max() + margin)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(get_path_to_figures_dir() / "random_walk.pdf")


def _plot_new_error_trace(fig_size_cm=(3, 10.3)):
    
    param_df = pd.read_csv(get_path_to_data_dir() / "processed" / "MLP001_params.csv")
    r0_eps = get_r0_eps(param_df, mc_filename="IthinkThisIsTheActualGoodOne.pt", deg_25_only=True)  # get the epsilons for R0 from the MC results, and apply them to the simulator.
    phys_exp_df = pd.read_csv(get_path_to_data_dir() / "processed" / "MLP001_wltp_25degC_record_shortened.csv")
    phys_exp_df = phys_exp_df.iloc[:int(len(phys_exp_df) * 0.9)]  # shorten physical experiment to remove the last pulse. -i.e. only take the first 8/9s of the rows.
    # shorten physical experiment to remove the last pulse. -i.e. only take the first 8/9s of the rows.
    eval_times = phys_exp_df["deq_Elapsed Time[h]"].to_numpy() * 3600  # convert to seconds

    sim_og = generate_standard_simulator(stop_idx=len(eval_times)-1, use_deq=False)
    
    sim_new = generate_standard_simulator(stop_idx=len(eval_times)-1, use_deq=False)  # create a new simulator to add the epsilons to, so we can compare the results.
    sim_new.param_interpolants["R0 [Ohm]"].set_eps_interpolator(temps = param_df["Temperature_degC"].to_numpy(),socs = param_df["SOC"].to_numpy(), eps_sample = r0_eps.detach().numpy())

    sim_og_results = sim_og.run_simulation(pbar=True, t_eval=eval_times)
    
    
    # add the gaussian-process-simulated epsilons to the simulator, and run the simulation again:
    sim_new.param_interpolants["R0 [Ohm]"].set_eps_interpolator(temps = param_df["Temperature_degC"].to_numpy(),socs = param_df["SOC"].to_numpy(), eps_sample = r0_eps.detach().numpy())
    sim_new_results = sim_new.run_simulation(pbar=True, t_eval=eval_times)


    orig_abs_error = sim_og_results["v_cell [V]"] - phys_exp_df["Voltage(V)"].to_numpy()
    new_abs_error = sim_new_results["v_cell [V]"] - phys_exp_df["Voltage(V)"].to_numpy()
    print(f"Original mean absolute error: {orig_abs_error.mean()}")
    print(f"New mean absolute error: {new_abs_error.mean()}")

    print(f"Original overall error: {np.linalg.norm(orig_abs_error)}")
    print(f"New overall error: {np.linalg.norm(new_abs_error)}")

    # plot the simulation results + physicaly experiment. Three subplots; one for the current, one for voltage, & one for the error.

    eval_times_h = eval_times / 3600  # convert to hours for plotting
    fig, axs = plt.subplots(nrows=3,ncols=1,figsize=convert_fig_size_cm_to_inches(fig_size_cm))
   
    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    axs[0].plot(eval_times_h, orig_abs_error * 1000, label="Original Model", color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2])
    axs[0].plot(eval_times_h, new_abs_error * 1000, label="New Model", color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
    axs[0].title.set_text("Error Trace Comparison")
    axs[0].tick_params(axis="x",labelbottom=False)
    axs[0].set_ylabel("Error [mV]")
    axs[0].legend()
    axs[0].grid(True)
    from view_posteriors import plot_obs_noise, plot_r0_eps
    samples = torch.load(get_path_to_data_results_dir() / "MC_testing/AMH_Gibbs_reducedSS_indexes:0:70000_warmup:700_samples:700_chains:3_adapt_600_thisactualgoodone/post_samples.pt")
    plot_obs_noise(samples["obss"], axs[1])
    axs[1].sharex(axs[0])
    axs[1].set_title(r"Posterior Distribution of $\sigma_{\text{obs}}$")
    plot_r0_eps(samples["eps_R0 [Ohm]_sample"], axs[2])
    axs[2].set_title(r"Posterior Distribution of $R'_0$")
    axs[2].set_xlabel("SOC")
    shift = 0.08  # how much to close the gap by, tune this

# Move ax1 down toward ax2
    pos1 = axs[1].get_position()
    axs[1].set_position([pos1.x0, pos1.y0 - shift, pos1.width, pos1.height])
    plt.tight_layout()
    plt.savefig(get_path_to_figures_dir() / "new_error_trace.pdf")

    
if __name__ == "__main__":
    main()