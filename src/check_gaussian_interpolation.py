"""Will Buesnel, Jul 26.

file to see the difference in accuracy between using parameters interpolated by PChip and ones interpolated by Gaussian Process Regression (GPR). The GPR interpolation is done using scipy.stats

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from models.coupled import test_simulation_param_interp_scheme
from models.parameters import get_all_parameter_interpolants
from utils import get_abs_error, get_mse

Y0 = [1.0, 0, 0, 25]  # initial state: soc=1, v_rc1=0, v_rc2=0, T=25 deg C

def main():
    processed_data_dir = Path(__file__).parent.parent / "data" / "processed"
    wltp_file = processed_data_dir / "MLP001_wltp_25degC_record_deq.csv"
    param_file = processed_data_dir / "MLP001_params.csv"
    ocv_file = processed_data_dir / "MLP001_ocv.csv"

    exp_df = pd.read_csv(wltp_file)
    ocv_df = pd.read_csv(ocv_file)
    param_df = pd.read_csv(param_file)

    # construct initial param (PChip) interpolator.

    current_func_interp = lambda t: -np.interp(t, exp_df["Elapsed Time[h]"].to_numpy() * 3600, exp_df["Current(A)"].to_numpy())

    param_interpolants_pchip = get_all_parameter_interpolants(param_df, ocv_df)

    results_pchip = test_simulation_param_interp_scheme(
        y0=Y0,
        current_func=current_func_interp,
        param_interpolants=param_interpolants_pchip,
        verbose=True,
        pbar=True,
        max_step=5.0,  # seconds
        t_eval=exp_df["Elapsed Time[h]"].to_numpy() * 3600,
        t_span=(0, exp_df["Elapsed Time[h]"].max() * 3600),
    )

    param_interpolants_gpr = get_all_parameter_interpolants(param_df, ocv_df, interp_scheme="gpr")
    # plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(results_pchip["t"], results_pchip["v_cell_interp"], label="PChip Interpolation", color="blue")
    plt.plot(results_pchip["t"], results_pchip["v_cell_gpr"], label="GPR Interpolation", color="orange")
    plt.plot(results_pchip["t"], results_pchip["v_cell_exp"], label="Experimental Data", color="green", linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title("Comparison of Interpolation Schemes")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

