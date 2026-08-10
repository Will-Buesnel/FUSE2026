"""Will Buesnel, Jul 26."""

from __future__ import annotations
from utils import get_ax
import pandas as pd
import numpy as np
from pathlib import Path
from models.parameters import get_all_parameter_interpolants
import matplotlib.pyplot as plt
import matplotlib as mpl

def main():
    processed_dir = Path.cwd().resolve() / "data" / "processed"
    param_df = pd.read_csv(processed_dir / "MLP001_params.csv")
    ocv_df = pd.read_csv(processed_dir / "MLP001_ocv.csv")

    # get dict of parameter interpolants.
    param_interpolants = get_all_parameter_interpolants(param_df, ocv_df, method="clough_tocher")



    socs = np.linspace(0.15, 0.9, 100)
    base_temp = 25 * np.ones_like(socs)

    temps = np.linspace(5, 40, 100)
    base_soc = 0.5 * np.ones_like(temps)

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 2)

    # define axes for subplots
    ax_soc_ocv = fig.add_subplot(gs[0, :]) # rectangular plot ie span both columns

    # the rest are square plots, so we can use the same aspect ratio for all of them
    ax_soc_r = fig.add_subplot(gs[1, 0])
    ax_soc_tau = fig.add_subplot(gs[2, 0])

    ax_temp_r = fig.add_subplot(gs[1, 1])
    ax_temp_tau = fig.add_subplot(gs[2, 1])

    r_colnames = ["R0 [Ohm]", "R1 [Ohm]", "R2 [Ohm]"]
    tau_colnames = ["tau1 [s]", "tau2 [s]"]


    # plot values (& evaluate interpolants) for each parameter
    # ocv: for each temperature, plot OCV vs SoC
    for temp in param_df["Temperature_degC"].unique():
        ax_soc_ocv.plot(socs, param_interpolants["OCV[V]"](socs, temp), label=f"T={temp}°C")
    ax_soc_ocv.legend(frameon=False, bbox_to_anchor=[1, 1, 0.30, 0])
    ax_soc_ocv.set_title("OCV vs SoC at 25°C")
    ax_soc_ocv.set_xlabel("State of Charge (SoC)")
    ax_soc_ocv.set_ylabel("Open Circuit Voltage (V)")

    for r in r_colnames:
        ax_soc_r.plot(socs, param_interpolants[r](socs, base_temp), label=r)
        ax_temp_r.plot(temps, param_interpolants[r](base_soc, temps), label=r)

    for tau in tau_colnames:
        ax_soc_tau.plot(socs, param_interpolants[tau](socs, base_temp), label=tau)
        ax_temp_tau.plot(temps, param_interpolants[tau](base_soc, temps), label=tau)

    # add legends, outside and to the right of the plots. the resistance and time constant plots share the same legend, so we can add it to the right of the 2nd column of plots
    ax_temp_r.legend(frameon=False, bbox_to_anchor=[1, 1, 0.60, 0])
    ax_temp_tau.legend(frameon=False, bbox_to_anchor=[1, 1, 0.60, 0])

    plt.savefig(Path.cwd().resolve() / "data" / "processed" / "parameter_plots.pdf", dpi=300)

if __name__ == "__main__":
    main()



    


