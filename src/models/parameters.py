"""
Will Buesnel, Jul 26.

This module will holdd classes relating to parameters.
This currently includes a ParameterInterpolator class, which wraps a single parameter's SOC/T scattered data as a callable interpolator.
It also includes a function to load parameter data from a CSV file and create interpolants for each parameter.
The CSV file should have columns: 'Temperature_degC', 'SOC', 'R0 [Ohm]', 'R1 [Ohm]', 'R2 [Ohm]', 'tau1 [s]', 'tau2 [s]'.

I've also kept the filter_outliers function here, which is used to filter out local outliers per temperature group, based on deviation from a rolling median across SOC (robust to trends in the parameter surface, unlike a global z-score).
I don't currently use it, but would like to investigate if it is useful in getting better interpolant functions. It outputs a message indicating how many rows were removed.

"""
import numpy as np
import scipy.interpolate
from pathlib import Path
import pandas as pd
import pyro.contrib.gp as gp
import pyro.distributions as dist
import torch

class ParameterInterpolator:
    """Wraps a single parameter's SOC/T scattered data as a callable interpolator."""

    def __init__(self, temps, socs, values, method="clough_tocher"):
        points = np.column_stack([temps, socs])
        if method == "clough_tocher":
            self._interp = scipy.interpolate.CloughTocher2DInterpolator(points, values)
        elif method == 'rbf':
            self._interp = scipy.interpolate.Rbf(temps, socs, values, function='linear')
        elif method == "linear":
            self._interp = scipy.interpolate.LinearNDInterpolator(points, values)
        else:
            raise ValueError(f"Unknown method: {method}")

    def __call__(self, soc, T):
        # vectorized: soc, T can be scalars or arrays
        soc_arr, T_arr = np.broadcast_arrays(soc, T)
        # makes sure the arrays are the same shape, so they can be passed to the interpolator.
        # this would only be a problem in the electric-only or thermal-only case.
        pts = np.stack([T_arr.ravel(), soc_arr.ravel()], axis=-1)
        result = self._interp(pts)
        # check for Nans, which would indicate extrapolation.
        if np.any(np.isnan(result)):
            raise ValueError(f"ParameterInterpolator extrapolated at soc={soc}, T={T}")
        return result.reshape(soc_arr.shape)
    
class ParamFunction:
    """
    Includes a parameter Interpolator + a single realisation of a Gaussian
    Process (unconditioned kernel) added on top, to represent one consistent
    sample of parameter uncertainty across SOC for the whole simulation.
    """

    def __init__(self, socs, temps, values, eps_sample):
        self.interpolator = ParameterInterpolator(temps, socs, values)
        self.eps_interpolator = ParameterInterpolator(temps, socs, eps_sample)

    def set_eps_interpolator(self, temps, socs, eps_sample):
        self.eps_interpolator = ParameterInterpolator(temps, socs, eps_sample)

    def reset_cov(self, temps, socs):
        self.cov = self.prior_kernel.forward(
            torch.tensor(np.column_stack([temps, socs]), dtype=torch.float32)
        )

    def __call__(self, soc, T) -> float:
        inter_value = self.interpolator(soc, T)
        eps = self.eps_interpolator(soc, T)  # single GP realisation, interpolated
        return inter_value + eps
        

def filter_outliers(df, param_cols, temp_col='Temperature_degC', soc_col='SOC',
                     window=5, z_thresh=3.0):
    """
    Filter out local outliers per temperature group, based on deviation from
    a rolling median across SOC (robust to trends in the parameter surface,
    unlike a global z-score).
    Output a message indicating how many rows were removed.
    """
    df = df.sort_values([temp_col, soc_col]).copy()
    keep_mask = pd.Series(True, index=df.index)

    for T, group in df.groupby(temp_col):
        for col in param_cols:
            rolling_med = group[col].rolling(window, center=True, min_periods=1).median()
            resid = (group[col] - rolling_med).abs()
            mad = resid.rolling(window, center=True, min_periods=1).median()
            # scale MAD to be comparable to std dev for a normal distribution
            local_z = resid / (1.4826 * mad + 1e-9)
            keep_mask.loc[group.index] &= (local_z < z_thresh)

    filtered_df = df[keep_mask]
    print(f"Filtered out {len(df) - len(filtered_df)} outliers based on local z-score threshold of {z_thresh}.")
    return filtered_df
    

def get_parameter_interpolant(df: pd.DataFrame, column_name: str, method="clough_tocher") -> dict:
    """
    Load parameter data from a CSV file and create interpolants for each parameter.
    The CSV file should have columns: 'Temperature_degC', 'SOC', 'R0 [Ohm]', 'R1 [Ohm]', 'R2 [Ohm]', 'tau1 [s]', 'tau2 [s]'.
    The function will compute C1 and C2 from tau1/R1 and tau2/R2, respectively, and create interpolants for R0, R1, R2, C1, and C2 as functions of SOC and Temperature.
    Returns a dictionary of ParameterInterpolator objects for each parameter.

    Currently it uses the same interpolation method for all parameters.
    """

    # sort before building interpolants to make sure x is monotonically increasing for the parameter.
    # this stops binary search from failing in the interpolator.
    # the assumption is that the data is already sorted by temperature, then SOC, but this is a safety measure.
    # another assummption is that all records are fully populated, which if not true would throw off the sorting and break the interpolator.
    
    sort_by = ["Temperature_degC", "SOC"]
    df = df.sort_values(by=sort_by)
    temps = df["Temperature_degC"].to_numpy()
    socs = df["SOC"].to_numpy()
    param = df[column_name].to_numpy()

    return {column_name: ParameterInterpolator(temps, socs, param, method=method)}


def get_parameter_function(df: pd.DataFrame, column_name: str, eps_sample) -> ParamFunction:
    """
    Load parameter data from a CSV file and create a ParamFunction for the specified parameter.
    The CSV file should have columns: 'Temperature_degC', 'SOC', 'R0 [Ohm]', 'R1 [Ohm]', 'R2 [Ohm]', 'tau1 [s]', 'tau2 [s]'.
    The function will compute C1 and C2 from tau1/R1 and tau2/R2, respectively, and create ParamFunctions for R0, R1, R2, C1, and C2 as functions of SOC and Temperature.
    Returns a dictionary of ParamFunction objects for each parameter.

    Currently it uses the same interpolation method for all parameters.
    """

    # sort before building interpolants to make sure x is monotonically increasing for the parameter.
    # this stops binary search from failing in the interpolator.
    # the assumption is that the data is already sorted by temperature, then SOC, but this is a safety measure.
    # another assummption is that all records are fully populated, which if not true would throw off the sorting and break the interpolator.
    
    sort_by = ["Temperature_degC", "SOC"]
    df = df.sort_values(by=sort_by)
    temps = df["Temperature_degC"].to_numpy()
    socs = df["SOC"].to_numpy()
    param = df[column_name].to_numpy()

    return ParamFunction(socs, temps, param, eps_sample)


def get_all_parameter_interpolants(paramdf: pd.DataFrame, ocvdf: pd.DataFrame, method="clough_tocher") -> dict:
    # get interpolants for all parameters in the parameter dataframe
    param_cols = ["R0 [Ohm]", "R1 [Ohm]", "R2 [Ohm]", "tau1 [s]", "tau2 [s]"]
    param_interpolants = {}
    for col in param_cols:
        param_interpolants.update(get_parameter_interpolant(paramdf, col, method=method))

    # get interpolants for ocv
    ocv_interpolants = get_parameter_interpolant(ocvdf, "OCV[V]", method=method)
    param_interpolants.update(ocv_interpolants)

    return param_interpolants

def format_interpolants(interpolants: dict) -> dict:
    """
    Format the interpolants dictionary to have keys that are more suitable for setting as attributes in the model.
    For example, "R0 [Ohm]" becomes "r0", "tau1 [s]" becomes "tau1", and "OCV[V]" becomes "ocv".
    """
    formatted_interpolants = {}
    for name, interpolant in interpolants.items():
        # take only the first part of the name, e.g. "R0 [Ohm]" -> "R0"
        name = name.split(" ")[0].split("[")[0].lower()
        formatted_interpolants[name] = interpolant
    return formatted_interpolants

def create_interpolant_with_gibbs_gaussian_noise(x_train, y_train, lengthscale_fn, noise = 1e-4, variance=1e-4):
    from models.local_stats import GibbsKernel, create_posterior_distribution, sample_from_posterior
    # Create a Gibbs kernel with the provided lengthscale function
    kernel = GibbsKernel(input_dim=1, lengthscale_fn=lengthscale_fn, variance=variance)
    # Create the posterior distribution given the training data
    posterior_mean, posterior_cov = create_posterior_distribution(kernel, x_train, y_train, x_train, noise_variance=noise)
    # Sample from the posterior distribution to create a new interpolant
    samples = sample_from_posterior(posterior_mean, posterior_cov, num_samples=1)
    # Return a callable function that interpolates the samples

def get_initial_soc(initial_v_cell: float, initial_temp: float, ocv_df: pd.DataFrame) -> float:
    # construct an interpolater to go backwards to normal. I.e. instead of soc -> ocv, we want ocv -> soc.
    # since this will only be done at steady states (i.e. where current is = dvdt = 0 for an extending period of time), I can assume v_cell == v_ocv.
    ocv_interpolator = ParameterInterpolator(ocv_df["Temperature_degC"].to_numpy(),
                                             ocv_df["OCV[V]"].to_numpy(),
                                                ocv_df["SOC"].to_numpy())

    return ocv_interpolator(initial_v_cell, initial_temp)  # returns the initial SOC corresponding to the initial cell voltage and temperature

def _test_get_all_parameter_interpolants():
    # Load the parameter data from the CSV file
    param_file_path = "data/processed/MLP001_params.csv"
    ocv_file_path = "data/processed/MLP001_ocv.csv"
    paramdf = pd.read_csv(param_file_path)
    ocvdf = pd.read_csv(ocv_file_path)

    # Get all parameter interpolants
    param_interpolants = get_all_parameter_interpolants(paramdf, ocvdf)

    # Test the interpolants by evaluating them at a few points
    test_points = [
        (0.5, 25),  # SoC=0.5, T=25°C
        (0.8, 30),  # SoC=0.8, T=30°C
        (0.2, 20),  # SoC=0.2, T=20°C
    ]

    for soc, T in test_points:
        print(f"At SoC={soc}, T={T}°C:")
        for name, interpolant in param_interpolants.items():
            value = interpolant(soc, T)
            print(f"  {name}: {value}")


def map_param_space_onto_grid(param_df):
    '''Map the entire parameter space onto a grid of SOC and Temperature values.
    This is so we can use a grid interpolator.
    '''

if __name__ == "__main__":
    _test_get_all_parameter_interpolants()


