'''
Bayesian model with parallel processing.
'''
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

from models.parameters import get_all_parameter_interpolants
from models.local_stats import AdaptiveMetropolisHastings, MCMCStop
from utils import plot_traces, set_rc_params, get_path_to_data_results_dir, get_path_to_data_processed_dir, plot_df
from simulation import Simulator, Cell, EntropyCoeffFunc
import time

import pyro.distributions as dist

import pyro
from pyro.infer import Predictive
import torch
from pyro.infer import MCMC
from pyro.infer.mcmc.util import diagnostics as pyro_diagnostics
from pyro.ops.stats import effective_sample_size, split_gelman_rubin
from pyro.infer.mcmc import RandomWalkKernel

VAR_INITIAL_GUESS = 1e-7  # initial guess for the observation noise
SIM_TIMESTEPS = 30000 # take this number of timesteps for the simulation. While we're setting it up we don't need all the timesteps.
NUM_SAMPLES = 1 # number of samples to draw from the posterior distribution for the parameters.
WARMUP_STEPS = 1 # number of warmup steps for MCMC inference.
OBS_EPS = 1e-5 # observation noise for the likelihood function.
NUM_CHAINS = 2 # number of chains to run in parallel for MCMC inference.
_obs_scale = OBS_EPS**0.5 * torch.ones(SIM_TIMESTEPS)  # observation noise for the likelihood function, as a torch tensor. this is not currently used as obs_noise has become learnable parameter.

# set random seed for reproducibility
RND_SEED = 42
pyro.set_rng_seed(RND_SEED)

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

    var_scaled = pyro.sample("var_scaled", dist.HalfNormal(1.0))  # order-1 scale
    var = pyro.deterministic("var", var_scaled * 1e-6)
    obs_scale = pyro.sample("obs_scale", dist.Uniform(0., 1e-2))

    gauss_interps_q = [("R0 [Ohm]", {"lengthscale_func": lengthscale_func_2d, "variance": var})]  # variational parameters for the Gaussian noise
    simulator.set_gauss_interps(gauss_interps_q)

    output = simulator.run_simulation(**simulator.kwargs)["v_cell [V]"]  # get the voltage and resistance output from the simulation
    param_interpolants_debug.append(deepcopy(simulator.param_interpolants))  # store the parameter interpolants for debugging
    # convert output to a torch tensor
    output = torch.tensor(output, dtype=torch.float32)

    pyro.sample(
    "obss",
    dist.Normal(output, obs_scale).to_event(1),
    obs=obs
)
    

def run_inference_MCMC(simulator: Simulator, obs=None, warmup_steps=100, num_samples=1000, num_chains=1, **kwargs):

    kernel = RandomWalkKernel(model=model, target_accept_prob=0.234)  # use an adaptive Metropolis-Hastings kernel for MCMC inference

    mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains)  # set up the MCMC inference
    try:
        mcmc.run(simulator, obs=obs, **kwargs)
    except MCMCStop:
        print("Inference stopped due to numerical issues.")
    return mcmc


def check_pyro_params():
    for name, value in pyro.get_param_store().items():
        print(name, value, value.grad if hasattr(value, 'grad') else 'no grad attr')


def compute_diag_stats_on_samples(samples: pd.DataFrame, extra_exclude_cols=None, verbose=False):
    results = {}
    exclude = ("Chain", "Iteration") + (extra_exclude_cols if extra_exclude_cols is not None else ())
    param_cols = [c for c in samples.columns if c not in exclude]

    for param in param_cols:
        pivoted = samples.pivot(index="Iteration", columns="Chain", values=param)
        transpose = pivoted.T.to_numpy(dtype=np.float32)  # transpose to shape (num_chains, num_samples). This means that passing in any vector parameter to this will break it.
        # but in fairness the vector parameters are not designed to have these stats computed for them.

        stacked = torch.tensor(transpose, dtype=torch.float32)

        results[param] = {
            "r_hat": split_gelman_rubin(stacked).item(),
            "n_eff": effective_sample_size(stacked).item(),
        }
    results_df = pd.DataFrame(results).T
    print(results_df)


def pyro_model_sample(simulator: Simulator, obs=None,warmup_steps=100, num_samples=1000, num_chains=1, **kwargs):
    # run the inference using MCMC
    mcmc = run_inference_MCMC(simulator, obs=obs, warmup_steps=warmup_steps, num_samples=num_samples, num_chains=num_chains, **kwargs)
    return mcmc.get_samples(group_by_chain=True), mcmc


def pyro_model_predict(simulator: Simulator, samples, obs=None, **kwargs):
    pred = Predictive(model, posterior_samples=samples)
    pred_values = pred(simulator, obs=obs, **kwargs)  # shape (num_samples, num_timesteps)
    return pred_values


def plot_mixing(samples: pd.DataFrame, param_names):

    num_params = len(param_names)  
    num_chains = samples["Chain"].nunique()

    fig, axs = plt.subplots(num_params, 1)

    for i, param in enumerate(param_names): 
        print(f"Plotting trace for {param}...")
        for chain in range(1, num_chains+1):
            chain_samples = samples[samples["Chain"] == chain][param]
            axs[i].plot(chain_samples.values, label=f"Chain {chain}", alpha=0.5)
        axs[i].set_title(f"Trace plot for {param}")
        axs[i].set_xlabel("Iteration")
        axs[i].set_ylabel(param)
        axs[i].legend()
    # add text to the figure with the randomseed,
    fig.text(0.02, 0.98, f"Random Seed: {RND_SEED}", verticalalignment='top', fontsize=10)
    plt.tight_layout()

    plt.show()
        

def save_pred_samples_to_pt(samples, filename, with_time=True):
    # pred_samples is a dict of tensors, convert to pandas dataframe and save to csv
    if with_time:
        filename = filename.replace(".pt", f"_{time.strftime('%Y%m%d_%H%M%S')}.pt")
    torch.save(samples, filename)

def save_diagnostics_to_csv(mcmc, filename, with_time=True):
    if with_time:
        filename = filename.replace(".csv", f"_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    diagnostics = mcmc.diagnostics()
    diagnostics_df = pd.DataFrame(diagnostics)
    diagnostics_df.to_csv(filename, index=False)
    print(f"Diagnostics saved to {filename}")

def _open_pred_samples_from_pt(filename):
    return torch.load(filename)

def _convert_pred_samples_to_df(samples, drop_index=False):
    # chains are indicated in the individual terms within the samples.
    # i.e.: variance : [[samples from chain 1],...,[samples from chain n]]
    # it is a list of dicts
    # convert to pandas dataframe with columns for each parameter and rows for each sample, with a column for the chain number & iteration within a chain.
    key1 = list(samples.keys())[0]  # get the first key to determine the shape of the samples
    vals1 = samples[key1]  # get the values for the first key
    n_chains, n_samples = vals1.shape[0], vals1.shape[1]  # get the number of chains and samples
    all_samples = []
    for chain in range(1, n_chains + 1):
        for sample in range(1, n_samples + 1):
            sample_dict = {param: samples[param][chain-1, sample-1].detach().numpy() for param in samples.keys()}
            sample_dict["Chain"] = chain
            sample_dict["Iteration"] = sample
            all_samples.append(sample_dict)

    samples_df = pd.DataFrame(all_samples)
    if drop_index:
        samples_df = samples_df.drop(columns=["Iteration"])
    return samples_df


def open_pred_samples_as_df(filename, drop_index=False):
    samples_tensor = _open_pred_samples_from_pt(filename)
    # convert to pandas dataframe
    # tensor will be of shape (num_chains, num_samples, num_timesteps)
    return _convert_pred_samples_to_df(samples_tensor, drop_index=drop_index)
    

def lengthscale_func_2d(x, soc_transition=0.3, soc_steepness=30,
                         l_soc_low=0.02, l_soc_high=0.08,
                         l_temp=0.15, temp_min=5.0, temp_max=40.0):
    temp_raw = x[:, 0]
    soc = x[:, 1]

    # normalize temperature to [0, 1] so it's on a comparable scale to SOC
    temp = (temp_raw - temp_min) / (temp_max - temp_min)

    # smooth transition in SOC: steepness ~10-50 gives a soft knee,
    # not a step. sigmoid input is O(1) in soc-units, not O(1000).
    w = torch.sigmoid(soc_steepness * (soc - soc_transition))
    l_soc = l_soc_low + (l_soc_high - l_soc_low) * w

    # temperature contributes its own smooth, bounded term
    l_temp_contribution = l_temp * temp

    return l_soc + l_temp_contribution

def get_diagnostics_from_csv(filename):
    diagnostics_df = pd.read_csv(filename)
    return diagnostics_df


def print_ESS_per_chain(samples_df, param_names):
    for param in param_names:
        for chain in samples_df["Chain"].unique():
            chain_samples = samples_df[samples_df["Chain"] == chain][param]
            ess = effective_sample_size(torch.tensor(chain_samples.astype(float).values, dtype=torch.float32).unsqueeze(0))  # add a singleton dimension to escape assertion errors.
            print(f"ESS for {param} in Chain {chain}: {ess.item()}")

# ----------------------------------------------------------



def generate_sample_test(num_samples=NUM_SAMPLES, warmup_steps=WARMUP_STEPS, num_chains=NUM_CHAINS):
    root = Path.cwd().resolve()
    processed_data_dir = root / "data" / "processed"
    wltp_df = pd.read_csv(processed_data_dir / "MLP001_wltp_25degC_record_deq.csv")
    param_df = pd.read_csv(processed_data_dir / "MLP001_params.csv")
    ocv_df = pd.read_csv(processed_data_dir / "MLP001_ocv.csv")
    entropy_df = pd.read_csv(processed_data_dir / "entropy_data_cell1.csv")


    # take only the first x records
    wltp_df = wltp_df.iloc[START_IDX:STOP_IDX, :]


    entropyfunc = EntropyCoeffFunc(entropy_df)  # create an instance of the EntropyCoeffFunc class

    # Define cell properties
    lp_cell = Cell(
        name="MLP001",
        capacity_Ah=2.2,
        c = 42.9, # J K^-1
        h = 3.59, # J K^-1
        c_p = 887, # J kg^-1 K^-1
        rho = 2682, # kg m^-3,
        entropy_coeff_func = entropyfunc
    )


    gauss_interps = [("R0 [Ohm]", {"lengthscale_func": lengthscale_func_2d, "variance": VAR_INITIAL_GUESS})]  # variational parameters for the Gaussian noise

    observed_values = torch.tensor(wltp_df["Voltage(V)"].to_numpy(), dtype=torch.float32)  # observed voltage values from the experiment
    # set up the simulator.
    sim = Simulator(wltp_df, ocv_df, param_df, lp_cell, gauss_interps=gauss_interps, t_eval = wltp_df["deq_Elapsed Time[h]"].to_numpy() * 3600)
    sim.y0 = [1, 0, 0, 26.0]  # initial state: soc=1, v_rc1=0, v_rc2=0, T=25 deg 
    sim.set_cell_capacity(2.15) # set the cell capacity to 2.15 Ah, which is a reasonable estimate for this cell. This was found to reduce model error at steady states.
    sim.kwargs["max_step"] = 1  # set the max step size for the simulation to avoid numerical issues.
    sim.kwargs["dense_output"] = False  # set the output to not be dense, to avoid running out of memory with large simulations.
    sim.kwargs["pbar"] = False  # turn off the progress bar for the simulation, as it will be run multiple times.

    # debug: add the first parameter interpolator (without adding any noise) to the param_interpolants_debug list, so we can see how it changes over time.
    param_interpolants_debug.append(get_all_parameter_interpolants(param_df, ocv_df))  # store the initial parameter interpolant for R0

    return pyro_model_sample(simulator=sim, obs=observed_values, warmup_steps=warmup_steps, num_samples=num_samples, num_chains=num_chains)
    
START_IDX = 320000
STOP_IDX = 360000

def save_run(warmup_steps: int = 200, samples: int = 200, chains: int = 2, filename_prefix: str = "pred_samples_test_mulchains"):
    output_dir = get_path_to_data_results_dir() / filename_prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, mcmc = generate_sample_test(warmup_steps=warmup_steps, num_samples=samples, num_chains=chains)  # generate samples from the model using MCMC inference
    save_pred_samples_to_pt(samples, get_path_to_data_results_dir() / f"{filename_prefix}"/ "samples.pt", with_time=False)  # save the samples to a .pt file
    print(mcmc.diagnostics())
    save_diagnostics_to_csv(mcmc, get_path_to_data_results_dir() / f"{filename_prefix}" / "diagnostics.csv", with_time=False)  # save the diagnostics to a .csv file

if __name__ == "__main__":
    
     
    set_rc_params()  # set the rc params for plotting
    filename_pref = "MC_testing/MH_Matern52_point1_5degLS_20000"  # prefix for the output files
    save_run(warmup_steps=10000, samples=10000, chains=1, filename_prefix=filename_pref)  # run the inference and save the samples and diagnostics

    # read the samples back in and convert to pandas dataframe
    samples_df = open_pred_samples_as_df(get_path_to_data_results_dir() / f"{filename_pref}/samples.pt", drop_index=False)

    plot_mixing(samples_df, param_names=["obs_scale", "var_scaled"])  # plot the mixing of the R0 parameter
    diags = pd.read_csv(get_path_to_data_results_dir() / f"{filename_pref}/diagnostics.csv")

    print_ESS_per_chain(samples_df, param_names=["obs_scale", "var_scaled"])  # print the effective sample size for each chain and parameter



    # need to change how samples are saved to and read from a df I think, now I have got parallel computation working correctly...