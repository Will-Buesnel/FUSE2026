"""Will Buesnel, Jul 26, 2026."""

"""
Will not keep this file in the final version, but just using it here for exploration of ideas.

At the beginning, will follow the same structure as plot_parameters.py, but just for r_0 for now
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from models.parameters import get_all_parameter_interpolants


def main():
    processed_dir = Path.cwd().resolve() / "data" / "processed"
    param_df = pd.read_csv(processed_dir / "MLP001_params.csv")
    ocv_df = pd.read_csv(processed_dir / "MLP001_ocv.csv")

    # get dict of parameter interpolants.
    param_interpolants = get_all_parameter_interpolants(param_df, ocv_df, method="clough_tocher")



    socs = np.linspace(0.15, 0.9, 100)
    base_temp = 25 * np.ones_like(socs)

    # for now, will just plot at 25 degrees (will assume it is the same throughout the temp range)
    r0_values = param_interpolants["R0 [Ohm]"](socs, base_temp)

    # plot for now, just plot the values of r0 vs soc at 25 degrees
    plt.figure()
    plt.plot(socs, r0_values, label="R0 [Ohm]")
    plt.title("R0 vs SoC at 25°C")
    plt.xlabel("State of Charge (SoC)")
    plt.ylabel("R0 [Ohm]")


    # add a shape uncertainty band around r0 values, that shrink with Ae^(-k*SoC)
    # for now, will just use a simple approach and assume a fixed uncertainty
    uncertainty = r0_values[0] * np.exp(-5 * socs) * np.ones_like(r0_values)
    plt.fill_between(socs, r0_values - uncertainty, r0_values + uncertainty, alpha=0.3, label="Uncertainty")
    # add an outline of the uncertainty band, just to see what it looks like
    plt.plot(socs, r0_values - uncertainty, color='gray', linestyle='--', alpha=0.5)
    plt.plot(socs, r0_values + uncertainty, color='gray', linestyle='--', alpha=0.5)
    plt.legend()

    #plt.savefig(processed_dir / "idealised_r0_vs_soc_with_uncertainty.pdf", dpi=300)
    plt.cla()

    # show a plot for the unstationary covariance function:
    from local_stats import GibbsKernel

    # define a simple lengthscale function (e.g., linear function)
    def lengthscale_fn(X):
        return 0.1 + 0.5 * X  # Example: lengthscale increases with input

    # Create an instance of the GibbsKernel
    kernel = GibbsKernel(input_dim=1, lengthscale_fn=lengthscale_fn)

    # visualise Gaussian process samples using Gibbs kernel on grid of input points, as a time series
    X = torch.linspace(0, 1, 50).view(-1, 1)  # Shape (100, 1)
    K = kernel(X)  # Compute covariance matrix using Gibbs kernel

    # Sample from the multivariate normal distribution with mean 0 and covariance K
    mean = torch.zeros(X.shape[0])
    # check the kernel

    eigvals = torch.linalg.eigvalsh(K)
    is_symmetric = torch.allclose(K, K.t(), atol=1e-6)
    is_psd = torch.all(eigvals >= -1e-6)
    print("Is K positive semidefinite?", is_symmetric and is_psd)

    jitter = 1e-6
    K_stable = K + jitter * torch.eye(K.size(0), dtype=K.dtype)

    samples = torch.distributions.MultivariateNormal(mean, covariance_matrix=K_stable).sample((5,))  # Sample 5 functions  
    # problem is K does not seem to be positive semidefinite.

    # Plot the sampled functions
    for i in range(samples.shape[0]):
        plt.plot(X.numpy(), samples[i].numpy(), label=f'Sample {i+1}')
    plt.title("Samples from Gaussian Process with Gibbs Kernel")
    plt.xlabel("Input X")
    plt.ylabel("Function Value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()