"""Will Buesnel, Aug 26.

Improving the stochastic model by adding inference capability. I dont think the inference needs to be its own class/module as it all already uses pyro modules under the hood.
At a later date, I would like to clean this up a little. For now, I just want to get the distributions.
Relevant links:
    https://pyro.ai/examples/mcmc.html
"""

from pathlib import Path
import pandas as pd
import numpy as np
import torch
from copy import deepcopy
import matplotlib.pyplot as plt

from models.coupled import CoupledModel, ThermalModel
from models.electrical import ElectricalModel
from models.parameters import get_all_parameter_interpolants, format_interpolants, get_parameter_function
from utils import plot_traces
from models.local_stats import GibbsKernel
from bin.cells import Cell

import pyro.distributions as dist
import pyro
from pyro.infer import Predictive


VAR_INITIAL_GUESS = 1e-7  # initial guess for the observation noise
SIM_TIMESTEPS = 30000 # take this number of timesteps for the simulation. While we're setting it up we don't need all the timesteps.
NUM_SAMPLES = 500 # number of samples to draw from the posterior distribution for the parameters.
WARMUP_STEPS = 500 # number of warmup steps for MCMC inference.
OBS_EPS = 1e-5 # observation noise for the likelihood function.
_obs_scale = OBS_EPS**0.5 * torch.ones(SIM_TIMESTEPS)  # observation noise for the likelihood function, as a torch tensor. this is not currently used as obs_noise has become learnable parameter.


class Simulator:
    """
    This will be what I will pass into the pyro inference engine. A wrapper of the coupled model, that allows me to easily propagate uncertainties through it.
    """

    def __init__(self, cycler_df, ocv_df, param_df, entropy_df, cell: Cell, gauss_interps=None, **kwargs):

        self.exp_df = cycler_df
        self.ocv_df = ocv_df
        self.param_df = param_df
        self.entropy_df = entropy_df
        self.cell = cell
        self.kwargs = kwargs

        self.elec_model = ElectricalModel()
        self.thermal_model = ThermalModel(c=cell.c, h=cell.h, T_inf_degC=cell.T_inf_degC)
        self.coupled_model = CoupledModel(self.elec_model, self.thermal_model) #
        self.thermal_model.entropy_coeff_func = cell.entropy_coeff_func
        self.elec_model.max_capacity_As = cell.capacity_Ah * 3600 # Ah to As

        self.set_current_func() # set the current function for the simulation.

        self.param_interpolants = get_all_parameter_interpolants(param_df, ocv_df) # set all parameter interpolants by default.
        # we will use gauss_interps to overwrite any of the default parameter interpolants with a stochastic version.

        if gauss_interps is not None:
            self.set_gauss_interps(gauss_interps)


    def set_gauss_interps(self, gauss_interps: list[tuple[str, dict]]):

        for name, hyperparams in gauss_interps:

            # create a new ParameterFunction with Gaussian noise for the specified parameter
            kernel = GibbsKernel(input_dim=2, lengthscale_fn=hyperparams['lengthscale_func'], variance=hyperparams['variance'])

            X = np.column_stack([self.param_df["Temperature_degC"].to_numpy(), self.param_df["SOC"].to_numpy()])
            cov = kernel.forward(torch.tensor(X,
                                               dtype=torch.float32)) + torch.eye(len(X)) * 1e-6  # add a small jitter for numerical stability

            L = torch.linalg.cholesky(cov)  # Cholesky decomposition of the covariance matrix. This is allowed to have negative & positive values
            eps_standardised = pyro.sample(f"eps_{name}_standardised", dist.Normal(torch.zeros(len(self.param_df)), torch.ones(len(self.param_df))).to_event(1))  # sample standard normal epsilons. Helps the randon walk not blow up when choosing next steps.
            eps_sample = L @ eps_standardised
            # debug: force eps_sample to be zero for now, to check that the model works without noise.
            # eps_sample = torch.zeros(len(self.param_df))  # debug: force eps_sample

            pyro.deterministic(f"eps_{name}_sample", eps_sample)  # record the sampled epsilons for debugging

            self.param_interpolants[name] = get_parameter_function(self.param_df, name, eps_sample.detach().numpy())  # detach to avoid backprop through the sampling process
                                                                                                                    # maybe it is better to take a copy for this?


            # debug; get the parameter function for temp = 25degC
            socs = np.linspace(0.1, 1, 100)
            r0_func = self.param_interpolants[name]
            r0_values = [r0_func(soc, 25) for soc in socs]
            pyro.deterministic(f"{name}_values_at_25degC", torch.tensor(r0_values, dtype=torch.float32))  # record the parameter values at 25degC for debugging


    def set_current_func(self):
        self.set_current_func = lambda t: np.interp(t,
                                                    self.exp_df["Elapsed Time[h]"].to_numpy() * 3600,
                                                    -self.exp_df["Current(A)"].to_numpy()) # flip sign of current to match equations given.


    def run_simulation(self, **kwargs) -> dict:
        
        # set the interpolants as funcs for the model
        for name, interpolant in format_interpolants(self.param_interpolants).items():
            setattr(self.elec_model, f"_{name}_interp", interpolant) 

        return self.coupled_model.simulate(
            t_max=self.exp_df["Elapsed Time[h]"].to_numpy()[-1] * 3600,  # convert to seconds.
            current_func = self.set_current_func,
            **kwargs
        )

def graph_model_outputs(sim: Simulator, gauss_interps: list[tuple[str, object]] = None, y0: list = [1,0,0,25], obs=None):
        
        res1 = sim.run_simulation(**sim.kwargs, pbar=True, max_step = 40)  # run the simulation once to check it works.
        # reset the gauss interps
        sim.set_gauss_interps(gauss_interps)
        res2 = sim.run_simulation(**sim.kwargs, pbar=True, max_step = 40)  # run the simulation a second time to check it works.
    
        plt.plot(res1["t [s]"], res1["v_oc [V]"], label="Run 1", alpha=0.5)
        plt.plot(res2["t [s]"], res2["v_oc [V]"], label="Run 2", alpha=0.5)
    
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        plt.title("Voltage vs Time for MLP001 Cell")
        plt.legend()
        plt.show()
       
        plt.figure()
        plt.plot(res1["t [s]"], res1["R0 [Ohm]"], label="Run 1", alpha=0.5)
        plt.plot(res2["t [s]"], res2["R0 [Ohm]"], label="Run 2", alpha=0.5)
        plt.xlabel("Time [s]")
        plt.ylabel("R0 [Ohm]")
        plt.title("R0 vs Time for MLP001 Cell")
        plt.legend()
        plt.show()

# PYRO INFERENCE MODEL AND GUIDE --------------------------

param_interpolants_debug = [] # these need to be deepcopies so that they don't change over time.
    
def model(simulator: Simulator, obs=None):
    # assume y0 is already set before optimisation.
    var_unconst = pyro.sample("var_unconstrained", dist.Normal(0., 1.))
    var = torch.sigmoid(var_unconst) * 1e-4  # constrain variance to be positive and small
    pyro.deterministic("var", var)  # record the variance for debugging
    obs_scale_unconst = pyro.sample("obs_scale_unconstrained", dist.LogNormal(0., 3))  # learnable observation noise for the likelihood function. This is a vector of length SIM_TIMESTEPS, one for each timestep.
    obs_scale = torch.sigmoid(obs_scale_unconst) * 1e-2  # constrain the observation noise to be positive and small
    pyro.deterministic("obs_scale", obs_scale)  # record the observation noise for debugging


    gauss_interps_q = [("R0 [Ohm]", {"lengthscale_func": lengthscale_func_2d, "variance": var})]  # variational parameters for the Gaussian noise
    simulator.set_gauss_interps(gauss_interps_q)

    output = simulator.run_simulation(y0=simulator.y0, **simulator.kwargs, pbar=True)["v_cell [V]"]  # get the voltage and resistance output from the simulation
    param_interpolants_debug.append(deepcopy(simulator.param_interpolants))  # store the parameter interpolants for debugging
    # convert output to a torch tensor
    output = torch.tensor(output, dtype=torch.float32)

    pyro.sample(
    "obss",
    dist.Normal(output, obs_scale).to_event(1),
    obs=obs
)
    


def run_inference_MCMC(simulator: Simulator, gauss_interps: list[tuple[str, object]] = None, obs=None, num_samples=1000, **kwargs):
    from pyro.infer import MCMC
    from pyro.infer.mcmc import RandomWalkKernel

    kernel = RandomWalkKernel(model=model)  
    mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=WARMUP_STEPS)
    mcmc.run(simulator, obs=obs, **kwargs)
    return mcmc


def check_pyro_params():
    for name, value in pyro.get_param_store().items():
        print(name, value, value.grad if hasattr(value, 'grad') else 'no grad attr')

# ----------------------------------------------------------


def pyro_model_outputs(simulator: Simulator, gauss_interps: list[tuple[str, object]] = None, obs=None, num_samples=1000, **kwargs):
    # run the inference using MCMC
    mcmc = run_inference_MCMC(simulator=simulator, gauss_interps=gauss_interps, obs=obs, num_samples=num_samples, **kwargs)
    samples = mcmc.get_samples()

    print("Generating posterior predictive plots...")
    # on axes 6, plot the v_cell against time uncertainty band
    pred = Predictive(model, posterior_samples=samples)
    pred_values = pred(simulator, obs=None, **kwargs)  # shape (num_samples, num_timesteps)
    print("MCMC inference completed. Samples obtained.")

    def plot_posterior(samples, obs_key="obss", param_name="eps_R0 [Ohm]_sample"):
        param_eps = samples[param_name]  # shape (num_samples, n_dims)

        mean = param_eps.mean(dim=0).numpy()
        lower = param_eps.quantile(0.05, dim=0).numpy()
        upper = param_eps.quantile(0.95, dim=0).numpy()

        temp = simulator.param_df["Temperature_degC"].unique()

        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, len(temp), height_ratios=[1, 1.5])

        top_axes = fig.add_subplot(gs[0, :]) 
        bottom_axes = fig.add_subplot(gs[1, :])
        socs = np.linspace(0.1, 1, len(mean))
        top_axes.plot(socs, mean, label="Mean Prediction")
        top_axes.fill_between(socs, lower, upper, alpha=0.3, label="90% CI")
        top_axes.set_ylabel("R0 [Ohm]")
        top_axes.set_xlabel("SOC")
        top_axes.set_title("Posterior Distribution of R0 [Ohm] vs SOC at 25°C")
        
        mean_pred_v = samples[obs_key].mean(dim=0).numpy()
        lower_pred_v = samples[obs_key].quantile(0.05, dim=0).numpy()
        upper_pred_v = samples[obs_key].quantile(0.95, dim=0).numpy()
        bottom_axes.plot(simulator.exp_df["Elapsed Time[h]"].to_numpy() * 3600, mean_pred_v, label="Mean Prediction")
        bottom_axes.fill_between(simulator.exp_df["Elapsed Time[h]"].to_numpy() * 3600, lower_pred_v, upper_pred_v, alpha=0.3, label="90% CI")
        bottom_axes.set_title("Voltage Prediction vs Time")
        bottom_axes.set_xlabel("Time [s]")
        bottom_axes.set_ylabel("Voltage [V]")

        # paste the actual values from the experiment on the bottom axes
        bottom_axes.plot(simulator.exp_df["Elapsed Time[h]"].to_numpy() * 3600, simulator.exp_df["Voltage(V)"].to_numpy(), label="Experimental", color='tab:cyan', linestyle='--', alpha=0.4)
        bottom_axes.legend()

        # add text in bottom left of figure to indicate initial assumptions
        plt.figtext(0.1, 0.01, f"Initial assumption: var={VAR_INITIAL_GUESS}, obs_eps={OBS_EPS}", ha="left", fontsize=10)
        # add text in bottom right of figure to indicate number of samples and warmup steps
        plt.figtext(0.9, 0.01, f"Number of samples: {num_samples}, Warmup steps: {WARMUP_STEPS}", ha="right", fontsize=10)

        plt.tight_layout()
        plt.show()

    #plot_posterior(pred_values, obs_key="obss", param_name="R0 [Ohm]_values_at_25degC")


    r0s_at_25degC = np.array([[interpolant["R0 [Ohm]"](soc, 25) for soc in np.linspace(0.1, 1, 20)] for interpolant in param_interpolants_debug])

    # observation noises for the likelihood function - want to plot how the distribution changes over time.
 
    plot_traces(xs = np.linspace(0.1,1,20), Ys=r0s_at_25degC, title="R0 vs SOC at 25degC", xlabel="SOC", ylabel="R0 [Ohm]")
    print(samples.keys())
    plot_posterior(pred_values, obs_key="obss", param_name="eps_R0 [Ohm]_sample")

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    const_noises = pred_values["obs_scale"].detach().numpy()
    unconst_noises = samples["obs_scale_unconstrained"].detach().numpy()

    const_variance = pred_values["var"].detach().numpy()
    unconst_variance = samples["var_unconstrained"].detach().numpy()

    for ax, data, title in zip(axs.flatten(),
                                [const_noises, unconst_noises, const_variance, unconst_variance],
                                  ["Learned Observation Noise", "Unconstrained Observation Noise", 
                                   "Learned Variance", "Unconstrained Variance"]):
        ax.plot(data)
        ax.set_title(title)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")

    fig.suptitle("Trace Plots for variational parameters")
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------

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
    wltp_df = pd.read_csv(processed_data_dir / "MLP001_wltp_25degC_record_deq.csv")
    param_df = pd.read_csv(processed_data_dir / "MLP001_params.csv")
    ocv_df = pd.read_csv(processed_data_dir / "MLP001_ocv.csv")
    entropy_df = pd.read_csv(processed_data_dir / "entropydata_cell1.csv")


    # take only the first x records
    wltp_df = wltp_df.iloc[:SIM_TIMESTEPS, :]


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


    def lengthscale_func_2d(x):
        soc = x[:, 0]
        temp = x[:, 1]  # raw temperature, e.g. 5-40
        return 0.0025 + 0.0005 * torch.sigmoid(5000 * (soc - 0.3)) + 0.5 * temp


    gauss_interps = [("R0 [Ohm]", {"lengthscale_func": lengthscale_func_2d, "variance": VAR_INITIAL_GUESS})]  # variational parameters for the Gaussian noise

    observed_values = torch.tensor(wltp_df["Voltage(V)"].to_numpy(), dtype=torch.float32)  # observed voltage values from the experiment
    # set up the simulator.
    sim = Simulator(wltp_df, ocv_df, param_df, entropy_df, lp_cell, gauss_interps=gauss_interps, t_eval = wltp_df["deq_Elapsed Time[h]"].to_numpy() * 3600)
    sim.y0 = [1, 0, 0, 25]  # initial state: soc=1, v_rc1=0, v_rc2=0, T=25 deg 
    sim.kwargs["max_step"] = 1  # set the max step size for the simulation to avoid numerical issues.
    sim.kwargs["dense_output"] = False  # set the output to not be dense, to avoid running out of memory with large simulations.

    # debug: add the first parameter interpolator (without adding any noise) to the param_interpolants_debug list, so we can see how it changes over time.
    param_interpolants_debug.append(get_all_parameter_interpolants(param_df, ocv_df))  # store the initial parameter interpolant for R0

    pyro_model_outputs(simulator=sim, gauss_interps=gauss_interps, obs=observed_values, num_samples=NUM_SAMPLES)  # run the inference and plot the results
    # graph_model_outputs(simulator=sim, gauss_interps=gauss_interps, obs=observed_values)  # run the inference and plot the results