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

from collections.abc import Callable
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
from utils import plot_mixing, plot_traces, safe_cholesky, get_path_to_data_processed_dir, set_rc_params, get_path_to_data_results_dir, get_path_to_figures_dir, save_pred_samples_to_pt, get_path_to_figures_dir, open_pred_samples_as_df, df_to_tensor_dict, tensor_dict_to_not_grouped_by_chains
from simulation import Simulator, BayesianModelParams, generate_standard_simulator
import time

import pyro.distributions as dist

import pyro
from pyro.infer import Predictive
import torch
from pyro.infer import MCMC
from pyro.infer.mcmc.util import diagnostics as pyro_diagnostics
from pyro.ops.stats import effective_sample_size, split_gelman_rubin
from pyro.infer.mcmc import RandomWalkKernel


VAR_INITIAL_GUESS = 2.5e-3
SIM_TIMESTEPS = 30000 # take this number of timesteps for the simulation. While we're setting it up we don't need all the timesteps.
NUM_SAMPLES = 1 # number of samples to draw from the posterior distribution for the parameters.
WARMUP_STEPS = 1 # number of warmup steps for MCMC inference.
OBS_EPS = 1e-5 # observation noise for the likelihood function.
NUM_CHAINS = 2 # number of chains to run in parallel for MCMC inference.
_obs_scale = OBS_EPS**0.5 * torch.ones(SIM_TIMESTEPS)  # observation noise for the likelihood function, as a torch tensor. this is not currently used as obs_noise has become learnable parameter.

# set random seed for reproducibility
RND_SEED = 42
pyro.set_rng_seed(RND_SEED)


# ---------------------------------------------------------------

# PYRO INFERENCE MODEL AND GUIDE : 

param_interpolants_debug = [] # these need to be deepcopies so that they don't change over time.
temperatures_debug = [] # these need to be deepcopies so that they don't change over time.
    
def model(simulator: Simulator, obs=None):
    # assume y0 is already set before optimisation.
    
    # below could probably be cleaned up a little bit sorry.
    if simulator.bayesian_model_params.is_variational_param("var_scaled"):
        gp_var_standardised = pyro.sample("var_scaled", dist.HalfNormal(1.0))  # order-1 scale
        gp_var = gp_var_standardised*simulator.bayesian_model_params.get_scaling("var_scaled")  # scale the variance by the scaling factor for the var_scaled parameter.
    else:
        gp_var = simulator.bayesian_model_params.var_scaled_det

    if simulator.bayesian_model_params.is_variational_param("obs_scale"):
        obs_standardised_var = pyro.sample("obs_scale", dist.HalfNormal(1.0))  # order-1 scale
        obs_var = obs_standardised_var*simulator.bayesian_model_params.get_scaling("obs_scale")  # scale the observation noise by the scaling factor for the obs_scale parameter.
    else:
        obs_var = simulator.bayesian_model_params.obs_scale_det

    gauss_interps_q = [("R0 [Ohm]", {"lengthscale_func": lengthscale_func_2d, "variance": gp_var})]  # variational parameters for the Gaussian noise

    try:
        simulator.set_gauss_interps(gauss_interps = gauss_interps_q)  # set the Gaussian interpolants for the simulation
    except InvalidProposal:
        return  # if the proposal is invalid, return without running the simulation.
        print("Invalid proposal, returning early.")

    res = simulator.run_simulation(**simulator.kwargs)  # get the voltage and resistance output from the simulation
    temperatures_debug.append(deepcopy(res["T [°C]"]))  # store the temperature output for debugging

    output = res["v_cell [V]"]
    param_interpolants_debug.append(deepcopy(simulator.param_interpolants))  # store the parameter interpolants for debugging
    # convert output to a torch tensor
    output = torch.tensor(output, dtype=torch.float32)

    pyro.sample(
    "obss",
    dist.Normal(output, obs_var).to_event(1),
    obs=obs
)


def construct_c0_gp_matrix(param_df, lengthscale_func, variance):
    kernel = GibbsKernel(input_dim=2, lengthscale_fn=lengthscale_func, variance=variance)
    X = np.column_stack([param_df["Temperature_degC"].to_numpy(), param_df["SOC"].to_numpy()])
    K = kernel.forward(torch.tensor(X, dtype=torch.float64))
    deg_25_indexes = param_df[param_df["Temperature_degC"] == 25].index.to_numpy()
    print(f"{deg_25_indexes.shape[0]} rows at 25 degC, out of {param_df.shape[0]} total rows.")
    c0 = K[deg_25_indexes, :][:, deg_25_indexes].detach() # only take the 25 deg rows as this is what is being sampled.
    return torch.diag(c0)# return the diagonal as I only want to take the scale from this. Also as the baseline assumption is that samples are independent; will allow monte carlo to build correlations between samples without unncessary bias.


def run_inference_MCMC(simulator: Simulator, obs=None, warmup_steps=100, num_samples=1000, num_chains=1, adapt_start=800, **kwargs):
    # simulator must have a bayesian_model_params attribute set before running this function. This is used to set the variational parameters for the model.

    c0 = 1e-5 * simulator.get_bayesian_model_params().generate_c0()  # use the simulator's method to generate the c0 matrix

    kernel = AdaptiveMetropolisHastings(model=model, target_accept_prob=0.234, adapt_start = adapt_start, c0=c0)  # use an adaptive Metropolis-Hastings kernel for MCMC inference

    # kernel.debug = True  # set the debug flag to True to print out the log probs at each site for debugging
    

    mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains)  # set up the MCMC inference
    try:
        mcmc.run(simulator, obs=obs, **kwargs)
    except MCMCStop:
        print("Inference stopped due to numerical issues.")
    return mcmc


def check_pyro_params():
    for name, value in pyro.get_param_store().items():
        print(name, value, value.grad if hasattr(value, 'grad') else 'no grad attr')

# --------------------------------------------------------------------------------
# Using pyro model:

def pyro_model_sample(simulator: Simulator, obs=None, warmup_steps=100, num_samples=1000, num_chains=1, **kwargs):
    # run the inference using MCMC
    mcmc = run_inference_MCMC(simulator, obs=obs, warmup_steps=warmup_steps, num_samples=num_samples, num_chains=num_chains, **kwargs)
    return mcmc.get_samples(group_by_chain=True), mcmc


def pyro_model_predict(simulator: Simulator, samples, obs=None, **kwargs):
    pred = Predictive(model, posterior_samples=samples)
    pred_values = pred(simulator, obs=obs, **kwargs)  # shape (num_samples, num_timesteps)
    return pred_values


def batched_tqdm_predictive(model: Callable, posterior_samples, simulator: Simulator, obs=None, batch_size=10, **kwargs):
   
    num_samples = posterior_samples[list(posterior_samples.keys())[0]].shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    all_predictions = []
    for i in tqdm(range(num_batches), desc="Predictive Batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_samples = {k: v[start_idx:end_idx] for k, v in posterior_samples.items()}
        print(f"{batch_samples=}")
        pred = Predictive(model, posterior_samples=batch_samples)
        pred_values = pred(simulator, obs=obs, **kwargs)
        all_predictions.append(pred_values)

    # Concatenate predictions from all batches
    concatenated_predictions = {k: torch.cat([batch[k] for batch in all_predictions], dim=0) for k in all_predictions[0].keys()}
    return concatenated_predictions

# ----------------------------------------------------------------
# diagnostics

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
# utility functions for running inference and saving results.

def generate_sample_test(num_samples=NUM_SAMPLES, warmup_steps=WARMUP_STEPS, num_chains=NUM_CHAINS, adapt_start=800, **kwargs):

    if "simulator" in kwargs:
        sim = kwargs["simulator"]
    else:
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
    print(f"{samples=}, {type(samples)=}")
    save_pred_samples_to_pt(samples, get_path_to_data_results_dir() / f"{filename_prefix}"/ "samples.pt", with_time=False)  # save the samples to a .pt file
    pd.DataFrame(mcmc.diagnostics()).to_csv(get_path_to_data_results_dir() / f"{filename_prefix}" / "diagnostics.csv")  # save the diagnostics to a .csv file


def create_file_name(warmup_steps: int = 200, samples: int = 200, chains: int = 1,
                                    adapt_start: int = 300, start_idx: int = 334000, stop_idx: int = 350000, kernel_type="AMH", gp_kernel_type="Gibbs", add_txt: str  = ""):
    return f"MC_testing/{add_txt}{kernel_type}_{gp_kernel_type}_reducedSS_indexes:{start_idx}:{stop_idx}_warmup:{warmup_steps}_samples:{samples}_chains:{chains}_adapt_{adapt_start}"  # prefix for the output files


def save_and_run_bayesian_mc_infer(warmup_steps: int = 200, samples: int = 200, chains: int = 1,
                                    adapt_start: int = 300, start_idx: int = 334000, stop_idx: int = 350000, filename: str = None, use_deq=True, bayesian_model_params: BayesianModelParams = None, **kwargs):
    
    if filename is None:
        filename_pref = f"MC_testing/AMH_Gibbs_reducedSS_indexes:{start_idx}:{stop_idx}_warmup:{warmup_steps}_samples:{samples}_chains:{chains}_adapt_{adapt_start}"  # prefix for the output files
    else:
        filename_pref = filename

    save_run(warmup_steps=warmup_steps, samples=samples, chains=chains, filename_prefix=filename_pref, start_idx=start_idx, stop_idx=stop_idx, adapt_start=adapt_start, use_deq=use_deq, bayesian_model_params=bayesian_model_params, **kwargs)  # run the inference and save the samples and diagnostics
    save_metadata = {
        "run_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "warmup_steps": warmup_steps,
        "samples": samples,
        "chains": chains,
        "adapt_start": adapt_start,
        "start_idx": start_idx,
        "stop_idx": stop_idx,
        "use_deq": use_deq,
        "bayesian_model_params": bayesian_model_params.toDict() if bayesian_model_params is not None else None
    }
    pd.DataFrame([save_metadata]).to_csv(get_path_to_data_results_dir() / f"{filename_pref}" / "metadata.csv")  # save the metadata to a .csv file

def generate_and_save_post_distributions_from_samples(samples_df: pd.DataFrame, sim: Simulator, filename_pref: str, **kwargs):
    post = batched_tqdm_predictive(model, tensor_dict_to_not_grouped_by_chains(df_to_tensor_dict(samples_df, dtype=torch.float64)), sim, **kwargs)  # pyro expects a dict of tensors to be passed in.
    torch.save(post["eps_R0 [Ohm]_sample"], get_path_to_data_results_dir() / f"{filename_pref}/eps_samples.pt")  # save the eps samples to a .pt file for seeing if model has improved / plotting etc.

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


def plot_r0_traces(samples_df: pd.DataFrame, param_df: pd.DataFrame, equiv_sampled_r0_values: np.ndarray, bayesianModelParams: BayesianModelParams):

    if not bayesianModelParams.is_variational_param("var_scaled"):
        var_scaled = bayesianModelParams.var_scaled_det * torch.ones(len(samples_df), dtype=torch.float64)  # if var_scaled is not a variational parameter, use the deterministic value for all samples.
        kernels = [GibbsKernel(input_dim=2, lengthscale_fn=lengthscale_func_2d, variance=var_scaled[index]) for index in range(len(samples_df))]
    else:
        kernels = [GibbsKernel(input_dim=2, lengthscale_fn=lengthscale_func_2d, variance=torch.tensor(samples_df["var_scaled"].iloc[index], dtype=torch.float64)) for index in range(len(samples_df))]

    
    X = np.column_stack([param_df["Temperature_degC"].to_numpy(), param_df["SOC"].to_numpy()])
    Ks = [kernel.forward(torch.tensor(X, dtype=torch.float64)) for kernel in kernels]
    Ls = [safe_cholesky(K) for K in Ks]

    add_eps_stand = [torch.zeros(len(param_df), dtype=torch.float64) for _ in range(len(samples_df))]  # initialise a list of tensors to store the eps values for each sample.
    deg_25_indexes = param_df[param_df["Temperature_degC"] == 25].index.to_numpy()

    for eps_row, (index, samplesrow) in zip(add_eps_stand, samples_df.iterrows()):
        eps_row[deg_25_indexes] = torch.tensor(samplesrow["eps_R0 [Ohm]_standardised"], dtype=torch.float64)

    samples_df["eps_R0 [Ohm]_sample"] = [L @ eps.detach().clone()  for L, eps in zip(Ls, add_eps_stand)]

    samples_df["r0_with_eps"] = np.nan  # create a new column for the r0 values with eps added to the regular interpolant values at the sampled SOC values.
    samples_df["r0_with_eps"] = samples_df["r0_with_eps"].astype(object)  # set the dtype of the new column to object, so we can store arrays in it.
    r0_array = np.zeros((len(samples_df["Chain"].unique()), len(samples_df["Iteration"].unique()), len(equiv_sampled_r0_values)), dtype=object)  # create an array to store the r0 values with eps added to the regular interpolant values at the sampled SOC values.

    for chain in samples_df["Chain"].unique():
        chain_df = samples_df[samples_df["Chain"] == chain]
        chain_samples = []
        for idx, row in chain_df.iterrows():
            
            eps_r0 = row["eps_R0 [Ohm]_sample"].detach().numpy()
            # add the eps values to the regular interpolant values at the sampled SOC values
            equiv_sampled_r0_values_with_eps = equiv_sampled_r0_values + eps_r0  # add the eps values to the regular interpolant values at the sampled SOC values
            chain_samples.append(equiv_sampled_r0_values_with_eps)
        chain_samples = np.array(chain_samples)

        r0_array[chain-1, :, :] = chain_samples  # store the r0 values with eps added to the regular interpolant values at the sampled SOC values in the r0_array
    
    plot_traces(xs=range(len(r0_array[0,0])), Ys = r0_array, multiple_chains=True, ylabel=r"$\epsilon_{R_0}$", title="Trace plot of error added to R0 interpolation", xlabel="Sample Index")
    

if __name__ == "__main__":

    start_idx = 0
    stop_idx = 70000


    bayes_model_params = BayesianModelParams(var_scaled_det=10e-6, obs_scale_det=0.005**2)  # create an instance of the BayesianModelParams class
    bayes_model_params.set_variational_params() # set with the default for now.
     
    set_rc_params()  # set the rc params for plotting
    warmup_steps = 100
    samples = 100
    chains = 3
    adapt_start = 100

    # filename_pref = create_file_name(warmup_steps=warmup_steps, samples=samples, chains=chains, adapt_start=adapt_start, start_idx=start_idx, stop_idx=stop_idx, add_txt="testblocks_smallc0")  # create a filename prefix for the output files
    # save_and_run_bayesian_mc_infer(warmup_steps=warmup_steps, samples=samples, chains=chains, adapt_start=adapt_start, start_idx=start_idx, stop_idx=stop_idx, use_deq=False, bayesian_model_params=bayes_model_params,filename=filename_pref)  # run the inference and save the samples and diagnostics
    filename_pref = "MC_testing/AMH_Gibbs_reducedSS_indexes:0:70000_warmup:700_samples:700_chains:3_adapt_600_thisactualgoodone"

    # read the samples back in and convert to pandas dataframe
    samples_df = open_pred_samples_as_df(get_path_to_data_results_dir() / f"{filename_pref}/samples.pt", drop_index=False)

    sim = generate_standard_simulator(start_idx=start_idx, stop_idx=stop_idx, use_deq=False)  # generate a standard simulator for plotting the model outputs
    sim.set_bayesian_model_params(bayes_model_params)  # set the Bayesian model parameters for the simulator
    sim.kwargs["max_step"] = 1  # set the max step size for the simulation to avoid numerical issues.
    sim.kwargs["dense_output"] = False  # set the output to not be dense, to avoid running out of memory with large simulations.
    sim.kwargs["pbar"] = False  # turn off the progress bar for the simulation, as it will be run multiple times.
    samples_tensor_dict = df_to_tensor_dict(samples_df, dtype=torch.float64)  # convert the samples dataframe to a dict of tensors for use in pyro Predictive
    predictive_samples = { # need to remove grouping by chain.
    name: val.reshape(-1, *val.shape[2:])
    for name, val in samples_tensor_dict.items()
}

    # save to a .pt file
    # save_pred_samples_to_pt(post, get_path_to_data_results_dir() / f"{filename_pref}/post_samples.pt", with_time=False)
    # save the last 40 eps samples to a .pt file for seeing if model has improved / plotting etc.

    
    fig, axs = plot_mixing(samples_df, param_names=["obs_scale"])  # plot the mixing of the R0 parameter
    axs[0].set_ylabel(r"$\sigma_{\text{obs}}$ ")  # set the y-axis label for the obs_scale parameter
    plt.savefig(get_path_to_figures_dir() / f"mixing_obs_scale.pdf")
    # do trace plot of the parameter interpolants for R0, which are stored in param_interpolants_debug. This is a list of lists of ParameterFunction objects, one for each MCMC sample.
    # print the ess and r^ for each parameter in the samples_df
    print(samples_df.columns)
   #compute_diag_stats_on_samples(samples_df, extra_exclude_cols=["eps_R0 [Ohm]_standardised", "eps_R0 [Ohm]_sample", "r0_with_eps"])

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
    
    plot_r0_traces(samples_df=samples_df, param_df=param_df, equiv_sampled_r0_values=equiv_sampled_r0_values, bayesianModelParams=bayes_model_params)  # plot the traces of the R0 parameter with eps added to the regular interpolant values at the sampled SOC values
