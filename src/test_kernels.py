"""
Will Buesnel, Jul 26.

Workflow to test different kernels for the Gaussian process regression of parameters, and visualise the uncertainty in the parameter predictions.

Generally going to be using the pyro docs tutorial on Gaussian process as a reference: https://pyro.ai/examples/gp.html
"""

import os
import matplotlib.pyplot as plt
import torch
import numpy as np


import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from local_stats import GibbsKernel
import matplotlib.pyplot as plt
import pandas as pd

from models.parameters import get_all_parameter_interpolants

def main():
    # Set random seed for reproducibility
    assert pyro.__version__.startswith('1.9.1')
    pyro.set_rng_seed(0)
    torch.set_default_dtype(torch.float32)

    # gather data from processed directory
    processed_dir = os.path.join(os.getcwd(), "data", "processed")
    params_df = pd.read_csv(os.path.join(processed_dir, "MLP001_params.csv"))

    # get parameter interpolants as a base function.
    interpolants = get_all_parameter_interpolants(params_df, method="clough_tocher")

    # for now, the only interpolant we are interested in is R0, so we will just use that for now.
    r0_interpolant = interpolants["R0 [Ohm]"]

    # the data therefore is only that of R0, and we will just use the SoC as the input variable for now.
    socs = np.linspace(0.15, 0.9, 100)

    r0_values = r0_interpolant(socs, 25 * np.ones_like(socs))  # at 25 degrees





def plot(
    plot_observed_data=False,
    plot_predictions=False,
    n_prior_samples=0,
    model=None,
    kernel=None,
    n_test=500,
    ax=None,
):
    # note that this helper function does three different things:
    # (i) plots the observed data;
    # (ii) plots the predictions from the learned GP after conditioning on data;
    # (iii) plots samples from the GP prior (with no conditioning on observed data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    if plot_observed_data:
        ax.plot(X.numpy(), y.numpy(), "kx")
    if plot_predictions:
        Xtest = torch.linspace(-0.5, 5.5, n_test)  # test inputs
        # compute predictive mean and variance
        with torch.no_grad():
            if type(model) == gp.models.VariationalSparseGP:
                mean, cov = model(Xtest, full_cov=True)
            else:
                mean, cov = model(Xtest, full_cov=True, noiseless=False)
        sd = cov.diag().sqrt()  # standard deviation at each input point x
        ax.plot(Xtest.numpy(), mean.numpy(), "r", lw=2)  # plot the mean
        ax.fill_between(
            Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
            (mean - 2.0 * sd).numpy(),
            (mean + 2.0 * sd).numpy(),
            color="C0",
            alpha=0.3,
        )
    if n_prior_samples > 0:  # plot samples from the GP prior
        Xtest = torch.linspace(-0.5, 5.5, n_test)  # test inputs
        noise = (
            model.noise
            if type(model) != gp.models.VariationalSparseGP
            else model.likelihood.variance
        )
        cov = kernel.forward(Xtest) + noise.expand(n_test).diag()
        samples = dist.MultivariateNormal(
            torch.zeros(n_test), covariance_matrix=cov
        ).sample(sample_shape=(n_prior_samples,))
        ax.plot(Xtest.numpy(), samples.numpy().T, lw=2, alpha=0.4)

    ax.set_xlim(-0.5, 5.5)

if __name__ == "__main__":
    main()