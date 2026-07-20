"""
Will Buesnel, Jul 26.

This module will hold all classes relating to parameters.
This (inexhaustively) includes:

- Parameter class: a class to hold a single parameter, its value, and its units (bounds?)
    These should be dataclass objects. Not sure about mutability yet.

- ParameterSet class: a class to hold a set of parameters, and to allow for easy access and manipulation of those parameters.
    These should also be dataclass objects. Not sure about mutability yet.

- ParameterTable class: a class to hold a list of parameter sets essentially, 
    above might also include logic for knowing when each parameter set is valid

Haven't totally figured it out yet, but this is the general idea.
"""
import numpy as np
import scipy.interpolate
from pathlib import Path
import pandas as pd

class ParameterInterpolator:
    """Wraps a single parameter's SOC/T scattered data as a callable interpolator."""
    # the data we are working with is not on a regular grid  (hard to do with SoC in fairness), so we need to use  a scattered interpolator
    # I am not sure which is better, but theres relatively low cost to having functionality for both
    # will evaluate which is better later on.

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
        return result.reshape(soc_arr.shape)
    

def load_parameter_interpolants(path: Path, method="clough_tocher") -> dict:
    df = pd.read_csv(path)
    df["C1 [F]"] = df["tau1 [s]"] / df["R1 [Ohm]"]
    df["C2 [F]"] = df["tau2 [s]"] / df["R2 [Ohm]"]

    temps = df["Temperature_degC"].to_numpy()
    socs = df["SOC"].to_numpy()

    param_cols = {
        "r0": "R0 [Ohm]", "r1": "R1 [Ohm]", "r2": "R2 [Ohm]",
        "c1": "C1 [F]", "c2": "C2 [F]",
    }

    return {
        name: ParameterInterpolator(temps, socs, df[col].to_numpy(), method=method)
        for name, col in param_cols.items()
    }