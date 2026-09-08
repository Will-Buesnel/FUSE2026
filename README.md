# FUSE2026
Project: "Exploring the coupling between thermal parameters and electrical behaviours."
Faraday summer undergraduate experience project looking into using bayesian inference to decouple & reduce model error.

# Usage
1. Use generate_samples.py to do the actual inference, with bayesian and deterministic parameters of your choosing. By default this will do inference with Adaptive Metroplis-Hastings.
    Running the above (which will take some time just a heads up) will create a directory in data/results/MC_testing. At a high level, the only one currently important is the samples.pt, which contains the inferred samples after warmup.
2. Use the generate_posteriors.py file to use these inferred samples to generate full posteriors on the variational parameters, specifying the filepath corresponding to the previously generated file.

3. Use view_posteriors.py to graph these posteriors, or use view_interpolation_error.py to see if the inferred parameters improved the model results.

# General structure:
1. Inference mechanics & gaussian processes contained in models/local stats.py
2. Forward model structure contained in models/coupled.py
3. Logic for generating interpolation schemes (deterministic and deterministc + gaussian) contained in parameters.py
4. Logic for using both + adding in empirical data and parameters contained in simulation.py
5. Logic for running a bayesian model contained in BayesianModel.py 

# Specifics on directory created after inference.

The files within this will be 
<ol>
  <li>diagnostics.csv: contains information to guide you on how well the inference process went. Within this the following metrics can be seen
    <ol type="a">
      <li>n_eff. The number of effective samples for that parameter over all the inference. In an ideal world, you would like it to be above 1000 for stable estimates. In our world of high dimensionality, 100 is a more realistic aim (but to be honest it is usually far lower than that in these simulations).</li>
      <li>r_hat. The Gelman-Rubin split. Looseley it uses an average over multiple chains to compute how close the solutions are to converging. An r_hat value >> 1 indicates a lack of convergence.</li>
      <li>Mean acceptance rate. This measures how often a proposed solution was accepted by the random walk algorithm. In literature, the ideal acceptance rate for a high-dimensional random walk problem is 0.234.</li>
    </ol>
  </li>
  <li>Metadata.csv</li>
  This contains information on the parameters used in the inference, both deterministic and bayesian. It also includes time of run. If I rewrote some of the code I would make this more comprehensive but, as most deterministic parameters werent changed run-to-run, the ones I have currently listed is sufficient.
  <li>samples.pt</li>
  <li>post_samples.pt</li>
</ol>





















Licence: This work is released under the GNU GPL v3 license. Loosely, this means the software is free and user-modifiable, however any derivatives must be released under the same license. See LICENSE for more details.



## Setup

Create and activate the conda environment, then install dependencies:

```bash
conda create -n myenv python=3.11
conda activate myenv
pip install -r requirements.txt
```