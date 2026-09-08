'''
Will Buesnel, Aug '26
Generate samples. This can take a while, depending on your training window and number of samples.
'''


from bayesianModel import save_and_run_bayesian_mc_infer, generate_standard_simulator, create_file_name
from simulation import BayesianModelParams

def main():
    # specifiy the bayesian Model
    baymodel_params = BayesianModelParams()
    baymodel_params.set_variational_params() # set default parameters for the bayesian model.
    warmup_steps, samples, chains = 4, 4, 2
    adapt_start_idx = 4
    start_idx, stop_idx = 0, 64000 # miss out the end of the last pulse; i.e. where the error spikes.
    filename = create_file_name(warmup_steps=warmup_steps, samples=samples, chains=chains, start_idx=start_idx, stop_idx=stop_idx, adapt_start=adapt_start_idx)
    print(f"Saving results to {filename}")

    sim = generate_standard_simulator() # generate the simulator to use for the inference.
    sim.bayesian_model_params = baymodel_params # set the bayesian model parameters for the simulator.
    save_and_run_bayesian_mc_infer(warmup_steps=warmup_steps, samples=samples, chains=chains,
                                    start_idx=start_idx, stop_idx=stop_idx, adapt_start=adapt_start_idx,
                                      simulator=sim, bayesian_model_params=baymodel_params) # run the inference and save the results.


if __name__ == "__main__":
    main()