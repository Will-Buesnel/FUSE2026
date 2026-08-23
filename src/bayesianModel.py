'''
Bayesian model with parallel processing.
'''
"""Will Buesnel, Aug 26.

Improving the stochastic model by adding inference capability. I dont think the inference needs to be its own class/module as it all already uses pyro modules under the hood.
At a later date, I would like to clean this up a little. For now, I just want to get the distributions.
Relevant links:
    https://pyro.ai/examples/mcmc.html


Adding a penalty term to random walk.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.parameters import get_all_parameter_interpolants
from models.local_stats import AdaptiveMetropolisHastings, InvalidProposal, MCMCStop, lengthscale_func_2d, GibbsKernel
from utils import plot_mixing, plot_traces, safe_cholesky, set_rc_params, get_path_to_data_results_dir, get_path_to_data_processed_dir, plot_df, save_pred_samples_to_pt, open_pred_samples_as_df, df_to_tensor_dict
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

# PYRO INFERENCE MODEL AND GUIDE --------------------------

param_interpolants_debug = [] # these need to be deepcopies so that they don't change over time.
temperatures_debug = [] # these need to be deepcopies so that they don't change over time.
    
def model(simulator: Simulator, obs=None):
    # assume y0 is already set before optimisation.
    
    var_scaled = pyro.sample("var_scaled", dist.Uniform(0., 1.0))  # order-1 scale
    var = pyro.deterministic("var", var_scaled * 1e-6)
    obs_scale = pyro.sample("obs_scale", dist.Uniform(0., 1e-2))

    gauss_interps_q = [("R0 [Ohm]", {"lengthscale_func": lengthscale_func_2d, "variance": var})]  # variational parameters for the Gaussian noise

    try:
        simulator.set_gauss_interps(gauss_interps = gauss_interps_q)  # set the Gaussian interpolants for the simulation
    except InvalidProposal:
        return  # if the proposal is invalid, return without running the simulation.

    res = simulator.run_simulation(**simulator.kwargs)  # get the voltage and resistance output from the simulation
    temperatures_debug.append(deepcopy(res["T [°C]"]))  # store the temperature output for debugging

    output = res["v_cell [V]"]
    param_interpolants_debug.append(deepcopy(simulator.param_interpolants))  # store the parameter interpolants for debugging
    # convert output to a torch tensor
    output = torch.tensor(output, dtype=torch.float32)

    pyro.sample(
    "obss",
    dist.Normal(output, obs_scale).to_event(1),
    obs=obs
)

def penalty_function(samples):
    if samples[3:].any() > 1e-3:
        return np.inf   

def run_inference_MCMC(simulator: Simulator, obs=None, warmup_steps=100, num_samples=1000, num_chains=1, adapt_start=800, **kwargs):
    # currently hardcoded the shape of c0; again bad practice by me sorry. Just trying some things out; hopefully I get around to fixing it.
    kernel = AdaptiveMetropolisHastings(model=model, target_accept_prob=0.234, adapt_start = adapt_start, c0=torch.eye(25, dtype=torch.float64)*1e-4, init_step_size=1e-4)  # use an adaptive Metropolis-Hastings kernel for MCMC inference
    #kernel.penalty_function = penalty_function  # set the penalty function for the kernel. This is used to reject proposals that are invalid.
    # currently I am hardcoding the c0 matrix, but ideally its size/shape should be able to be inferred/set adaptively.

    mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains)  # set up the MCMC inference
    try:
        mcmc.run(simulator, obs=obs, **kwargs)
    except MCMCStop:
        print("Inference stopped due to numerical issues.")
    return mcmc


def check_pyro_params():
    for name, value in pyro.get_param_store().items():
        print(name, value, value.grad if hasattr(value, 'grad') else 'no grad attr')


def pyro_model_sample(simulator: Simulator, obs=None, warmup_steps=100, num_samples=1000, num_chains=1, **kwargs):
    # run the inference using MCMC
    mcmc = run_inference_MCMC(simulator, obs=obs, warmup_steps=warmup_steps, num_samples=num_samples, num_chains=num_chains, **kwargs)
    return mcmc.get_samples(group_by_chain=True), mcmc


def pyro_model_predict(simulator: Simulator, samples, obs=None, **kwargs):
    pred = Predictive(model, posterior_samples=samples)
    pred_values = pred(simulator, obs=obs, **kwargs)  # shape (num_samples, num_timesteps)
    return pred_values


def batched_tqdm_predictive(model, posterior_samples, simulator: Simulator, obs=None, batch_size=10, **kwargs):
   
    num_samples = posterior_samples[list(posterior_samples.keys())[0]].shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    all_predictions = []
    for i in tqdm(range(num_batches), desc="Predictive Batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_samples = {k: v[start_idx:end_idx] for k, v in posterior_samples.items()}
        pred = Predictive(model, posterior_samples=batch_samples)
        pred_values = pred(simulator, obs=obs, **kwargs)
        all_predictions.append(pred_values)

    # Concatenate predictions from all batches
    concatenated_predictions = {k: torch.cat([batch[k] for batch in all_predictions], dim=0) for k in all_predictions[0].keys()}
    return concatenated_predictions

# ----------------------------------------------------------------

def print_ESS_per_chain(samples_df, param_names):
    for param in param_names:
        for chain in samples_df["Chain"].unique():
            chain_samples = samples_df[samples_df["Chain"] == chain][param]
            ess = effective_sample_size(torch.tensor(chain_samples.astype(float).values, dtype=torch.float32).unsqueeze(0))  # add a singleton dimension to escape assertion errors.
            print(f"ESS for {param} in Chain {chain}: {ess.item()}")



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

# ----------------------------------------------------------


def generate_standard_simulator(start_idx=0, stop_idx=70000, use_deq=True, stochastic=True):
    root = Path.cwd().resolve()
    processed_data_dir = root / "data" / "processed"
    param_df = pd.read_csv(processed_data_dir / "MLP001_params.csv")
    ocv_df = pd.read_csv(processed_data_dir / "MLP001_ocv.csv")
    entropy_df = pd.read_csv(processed_data_dir / "entropy_data_cell1.csv")

    if use_deq:
        wltp_df = pd.read_csv(processed_data_dir / "MLP001_wltp_25degC_record_deq.csv")
        time_column = "deq_Elapsed Time[h]"
        print("Simulating with dequantised data, assuming uniform spacing between bins")
    else:
        wltp_df = pd.read_csv(processed_data_dir / "MLP001_wltp_25degC_record_shortened.csv")
        time_column = "Elapsed Time[h]"
        print("Simulating with undequantised data, collapsed at the mean per unique time value.")

    # take only the first x records
    wltp_df = wltp_df.iloc[start_idx:stop_idx, :]


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

    if stochastic:
        gauss_interps = [("R0 [Ohm]", {"lengthscale_func": lengthscale_func_2d, "variance": 1e-7})]  # variational parameters for the Gaussian noise
    else:
        gauss_interps = None  # no Gaussian noise interpolants for the deterministic model

    sim = Simulator(wltp_df, ocv_df, param_df, lp_cell, gauss_interps=gauss_interps, t_eval = wltp_df[time_column].to_numpy() * 3600)
    initial_voltage = sim.exp_df["Voltage(V)"].iloc[0]  # get the initial voltage from the experiment data. this is commented out as its only used when we dont simulate from the start.
    # get initial soc via interpolation of the ocv_df
    #initial_soc = np.interp(initial_voltage, sim.ocv_df["OCV[V]"].to_numpy(), sim.ocv_df["SOC"].to_numpy())
    initial_soc = 1
    initial_temp = 26.0  # initial temperature in degrees Celsius. Ideally I would use the one from the experimentdf, but I don't trust it.
    # When we generate comprehensive data, we will start from t0, skipping this problem entirely
    sim.y0 = [initial_soc, 0, 0, initial_temp]  # initial state: soc=
    sim.set_cell_capacity(2.15) # set the cell capacity to 2.15 Ah, which is a reasonable estimate for this cell. This was found to reduce model error at steady states.

    return sim


def generate_sample_test(num_samples=NUM_SAMPLES, warmup_steps=WARMUP_STEPS, num_chains=NUM_CHAINS, adapt_start=800, **kwargs):

    sim = generate_standard_simulator(**kwargs)
    
    sim.kwargs["max_step"] = 1  # set the max step size for the simulation to avoid numerical issues.
    sim.kwargs["dense_output"] = False  # set the output to not be dense, to avoid running out of memory with large simulations.
    sim.kwargs["pbar"] = False  # turn off the progress bar for the simulation, as it will be run multiple times.

    # debug: add the first parameter interpolator (without adding any noise) to the param_interpolants_debug list, so we can see how it changes over time.
    observed_values = torch.tensor(sim.exp_df["Voltage(V)"].to_numpy(), dtype=torch.float32)  # observed voltage values from the experiment
    param_interpolants_debug.append(get_all_parameter_interpolants(sim.param_df, sim.ocv_df))  # store the initial parameter interpolant for R0

    return pyro_model_sample(simulator=sim, obs=observed_values, warmup_steps=warmup_steps, num_samples=num_samples, num_chains=num_chains, adapt_start=adapt_start)  # run the inference using MCMC


def save_run(warmup_steps: int = 200, samples: int = 200, chains: int = 2, filename_prefix: str = "pred_samples_test_mulchains", **kwargs):
    output_dir = get_path_to_data_results_dir() / filename_prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, mcmc = generate_sample_test(warmup_steps=warmup_steps, num_samples=samples, num_chains=chains, **kwargs)  # generate samples from the model using MCMC inference

    save_pred_samples_to_pt(samples, get_path_to_data_results_dir() / f"{filename_prefix}"/ "samples.pt", with_time=False)  # save the samples to a .pt file
    pd.DataFrame(mcmc.diagnostics()).to_csv(get_path_to_data_results_dir() / f"{filename_prefix}" / "diagnostics.csv")  # save the diagnostics to a .csv file
    
def save_and_run_bayesian_mc_infer(warmup_steps: int = 200, samples: int = 200, chains: int = 1,
                                    adapt_start: int = 300, start_idx: int = 334000, stop_idx: int = 350000, use_deq=True):
    
    filename_pref = f"MC_testing/AMH_Gibbs_indexes:{start_idx}:{stop_idx}_warmup:{warmup_steps}_samples:{samples}_chains:{chains}_adapt_{adapt_start}"  # prefix for the output files
    save_run(warmup_steps=warmup_steps, samples=samples, chains=chains, filename_prefix=filename_pref, start_idx=start_idx, stop_idx=stop_idx, adapt_start=adapt_start, use_deq=use_deq)  # run the inference and save the samples and diagnostics
    return filename_pref  # return the filename prefix for later use in plotting and analysis



def view_post_distributions(pred_samples, sim: Simulator):
    post = batched_tqdm_predictive(model, pred_samples, sim)  # pyro expects a dict of tensors to be passed in.
    n_to_save = min(40, post["eps_R0 [Ohm]_sample"].shape[0])
    torch.save(post["eps_R0 [Ohm]_sample"][-n_to_save:], get_path_to_data_results_dir() / f"{filename_pref}/last_{n_to_save}_eps_samples.pt")  # save the last 40 eps samples to a .pt file for seeing if model has improved / plotting etc.
    # save the last 40 eps samples to a .pt file for seeing if model has improved / plotting etc.

    # plot the pred sammples.

    plt.figure()
    plt.plot(sim.exp_df["Elapsed Time[h]"].to_numpy(), post["obss"].mean(0).detach().numpy(), label="Mean Posterior Predictive", color="blue")
    plt.fill_between(sim.exp_df["Elapsed Time[h]"].to_numpy(), post["obss"].mean(0).detach().numpy() - 2 * post["obss"].std(0).detach().numpy(), post["obss"].mean(0).detach().numpy() + 2 * post["obss"].std(0).detach().numpy(), color="blue", alpha=0.2, label="95% Credible Interval")
    plt.plot(sim.exp_df["Elapsed Time[h]"].to_numpy(), sim.exp_df["Voltage(V)"].to_numpy(), label="Observed Voltage", color="orange", alpha=0.5)
    plt.xlabel("Time [h]")
    plt.ylabel("Voltage [V]")
    plt.title("Posterior Predictive Samples vs Observed Voltage")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    start_idx = 0
    stop_idx = 70000
     
    set_rc_params()  # set the rc params for plotting
    filename_pref = save_and_run_bayesian_mc_infer(warmup_steps=30, samples=30, chains=1, adapt_start=40, start_idx=start_idx, stop_idx=stop_idx, use_deq=False)  # run the inference and save the samples and diagnostics
    

    # read the samples back in and convert to pandas dataframe
    samples_df = open_pred_samples_as_df(get_path_to_data_results_dir() / f"{filename_pref}/samples.pt", drop_index=False)
    print(samples_df.head())
    sim = generate_standard_simulator(start_idx=start_idx, stop_idx=stop_idx)  # generate a standard simulator for plotting the model outputs
    sim.kwargs["max_step"] = 1  # set the max step size for the simulation to avoid numerical issues.
    sim.kwargs["dense_output"] = False  # set the output to not be dense, to avoid running out of memory with large simulations.
    sim.kwargs["pbar"] = False  # turn off the progress bar for the simulation, as it will be run multiple times.
  


    samples_tensor_dict = df_to_tensor_dict(samples_df, dtype=torch.float64)  # convert the samples dataframe to a dict of tensors for use in pyro Predictive

    predictive_samples = { # need to remove grouping by chain.
    name: val.reshape(-1, *val.shape[2:])
    for name, val in samples_tensor_dict.items()
}
    #post = batched_tqdm_predictive(model, predictive_samples, sim)  # pyro expects a dict of tensors to be passed in.
    # save to a .pt file
    # save_pred_samples_to_pt(post, get_path_to_data_results_dir() / f"{filename_pref}/post_samples.pt", with_time=False)
    # save the last 40 eps samples to a .pt file for seeing if model has improved / plotting etc.
    

    plot_mixing(samples_df, param_names=["obs_scale", "var_scaled"])  # plot the mixing of the R0 parameter

    # do trace plot of the parameter interpolants for R0, which are stored in param_interpolants_debug. This is a list of lists of ParameterFunction objects, one for each MCMC sample.
    socs = np.linspace(0.1, 1, 100)
    temp = 25
    # get a list of r0 interpolants for each paraminterpolants element
    r0_interpolants = [param_interpolants["R0 [Ohm]"] for param_interpolants in param_interpolants_debug]  # get the first parameter interpolant for R0 from each MCMC sample
    r0_values = np.array([[r0_func(soc, temp) for soc in socs] for r0_func in r0_interpolants])  # get the R0

    regular_interpolant = generate_standard_simulator(start_idx=start_idx, stop_idx=stop_idx, stochastic=False).param_interpolants["R0 [Ohm]"]  # get the regular interpolant for R0 from the standard simulator



    param_df = pd.read_csv(get_path_to_data_processed_dir() / "MLP001_params.csv")  # read in the parameter dataframe
    sampled_socs = param_df["SOC"].to_numpy()  # get the SOC values from the parameter dataframe
    equiv_sampled_r0_values = np.array([regular_interpolant(soc, temp) for soc in sampled_socs])  # get the R0 values from the regular interpolant at the sampled SOC values


    # add eps values to the regular interpolant values to get more context for the eventual trace plot of this.
    # create new column in the samples_df for the r0 values with eps added to the regular interpolant values at the sampled SOC values.
    kernels = [GibbsKernel(input_dim=2, lengthscale_fn=lengthscale_func_2d, variance=torch.tensor(samples_df["var_scaled"].iloc[index] * 1e-6, dtype=torch.float64)) for index in range(len(samples_df))]

    X = np.column_stack([param_df["Temperature_degC"].to_numpy(), param_df["SOC"].to_numpy()])
    Ks = [kernel.forward(torch.tensor(X, dtype=torch.float64)) for kernel in kernels]
    Ls = [safe_cholesky(K) for K in Ks]
    add_eps_stand = samples_df["eps_R0 [Ohm]_standardised"].to_numpy()
    samples_df["eps_R0 [Ohm]_sample"] = [L @ torch.tensor(eps, dtype=torch.float64).T for L, eps in zip(Ls, add_eps_stand)]

    samples_df["r0_with_eps"] = np.nan  # create a new column for the r0 values with eps added to the regular interpolant values at the sampled SOC values.
    samples_df["r0_with_eps"] = samples_df["r0_with_eps"].astype(object)  # set the dtype of the new column to object, so we can store arrays in it.
    r0_array = np.zeros((len(samples_df["Chain"].unique()), len(samples_df["Iteration"].unique()), len(equiv_sampled_r0_values)), dtype=object)  # create an array to store the r0 values with eps added to the regular interpolant values at the sampled SOC values.

    for chain in samples_df["Chain"].unique():
        chain_df = samples_df[samples_df["Chain"] == chain]
        for idx, row in chain_df.iterrows():
            iter_samples = []
            eps_r0 = row["eps_R0 [Ohm]_sample"].detach().numpy()
            # add the eps values to the regular interpolant values at the sampled SOC values
      
            equiv_sampled_r0_values_with_eps = equiv_sampled_r0_values + eps_r0  # add the eps values to the regular interpolant values at the sampled SOC values
            iter_samples.append(equiv_sampled_r0_values_with_eps)
        chain_samples = np.array(iter_samples)

        r0_array[chain-1, :, :] = chain_samples  # store the r0 values with eps added to the regular interpolant values at the sampled SOC values in the r0_array

    plot_traces(xs=range(len(r0_array[0,0])), Ys = r0_array, multiple_chains=True)
    # need to change how samples are saved to and read from a df I think, now I have got parallel computation working correctly...