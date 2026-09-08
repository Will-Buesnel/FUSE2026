"""
Will Buesnel, Aug 26.
generate posteriors from saved samples.
This will take as long if not longer than generating the samples in the first place.
This is because it cannot parallel process the chains.
"""

from simulation import generate_standard_simulator, BayesianModelParams
import torch
from utils import open_pred_samples_as_df, get_path_to_data_results_dir, tensor_dict_to_not_grouped_by_chains, df_to_tensor_dict
from bayesianModel import generate_and_save_post_distributions_from_samples

def main():
    # load the samples
    filename = "MC_testing/AMH_Gibbs_reducedSS_indexes:0:64000_warmup:4_samples:4_chains:2_adapt_4" # change as desired.
    samples_df = open_pred_samples_as_df(get_path_to_data_results_dir() / f"{filename}/samples.pt")
    sim = generate_standard_simulator(stochastic=False, use_deq=False) # stoachastic = false because we have already generated the stochastic samples.
    bayesian_model_params = BayesianModelParams()
    bayesian_model_params.set_variational_params() # set default parameters for the bayesian model.
    sim.bayesian_model_params = bayesian_model_params
    # now all we are effectively doing is runnning the model forward.
    bayesian_model_params.set_variational_params() # set default parameters for the bayesian model.
    sim.bayesian_model_params = bayesian_model_params
    # now all we are effectively doing is runnning the model forward.
    generate_and_save_post_distributions_from_samples(samples_df, sim, filename_pref=filename, batch_size=1) # run the inference and save the results.

if __name__ == "__main__":
    main()