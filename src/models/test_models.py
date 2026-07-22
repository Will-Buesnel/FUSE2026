"Will Buesnel, Jul 26."

from os import times

import numpy
from numpy.testing import verbose

from parameters import get_all_parameter_interpolants
from electrical import ElectricalModel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def _test_simulation(**kwargs):
    # load the parameter interpolants from the CSV file
    param_file_path = "data/processed/MLP001_params.csv"
    ocv_file_path = "data/processed/MLP001_ocv.csv"

    param_df, ocv_df = pd.read_csv(param_file_path), pd.read_csv(ocv_file_path)

    param_interpolants = get_all_parameter_interpolants(param_df, ocv_df) # remember this is a dict of interpolants.

    print("Loaded parameter interpolants:")
    
    model = ElectricalModel()

    # set the interpolants as funcs for the model
    for name, interpolant in param_interpolants.items():

        name = name.split(" ")[0].split("[")[0].lower()  # take only the first part of the name, e.g. "R0 [Ohm]" -> "R0"
        print(f"Setting parameter interpolant for {name}")
        setattr(model, f"_{name}_interp", interpolant) 
        
    return model.simulate(**kwargs)


def plot_tests(current: int = -1, t_max: float = 1000.0, pbar: bool = False, verbose: bool = False, slow: bool = False):
    # first test simulation where we add no current, so the SoC should remain constant at 1.0.

    res_1_dict= _test_simulation(y0=[0.5, 0, 0], t_max=t_max) # (passed)

    # test with positive current
    res_2_dict= _test_simulation(y0=[1.0, 0, 0], current_func=lambda t: current, t_max=t_max, pbar=pbar, verbose=verbose, slow=slow) # (passed
    
    # test with positive cosine current
    res_3_dict= _test_simulation(y0=[1.0, 0, 0], current_func=lambda t: np.cos(t)-1, t_max=t_max, pbar=pbar, verbose=verbose, slow=slow) # (passed)

    # graph all results
    # one ax for soc, one for v_rc1, one for v_rc2. 3 rows, 1 column.
    fig, axs = plt.subplots(5, 1, figsize=(8, 6), constrained_layout=True)

    times1 = res_1_dict["t"]
    times2 = res_2_dict["t"]
    times3 = res_3_dict["t"]
    axs[0].plot(times1, res_1_dict["soc"], label="No current")
    axs[0].plot(times2, res_2_dict["soc"], label="Positive current")
    axs[0].plot(times3, res_3_dict["soc"], label="Cosine current")
    axs[0].set_ylabel("SoC")

    axs[1].plot(times1, res_1_dict["v_rc1"], label="No current")
    axs[1].plot(times2, res_2_dict["v_rc1"], label="Positive current")
    axs[1].plot(times3, res_3_dict["v_rc1"], label="Cosine current")
    axs[1].set_ylabel("v_rc1")

    axs[2].plot(times1, res_1_dict["v_rc2"], label="No current")
    axs[2].plot(times2, res_2_dict["v_rc2"], label="Positive current")
    axs[2].plot(times3, res_3_dict["v_rc2"], label="Cosine current")
    axs[2].set_ylabel("v_rc2")


    axs[3].plot(times1, res_1_dict["v_cell"], label="No current")
    axs[3].plot(times2, res_2_dict["v_cell"], label="Positive current")
    axs[3].plot(times3, res_3_dict["v_cell"], label="Cosine current")
    axs[3].set_ylabel("v_cell")


    axs[4].plot(times1, res_1_dict["v_oc"], label="No current")
    axs[4].plot(times2, res_2_dict["v_oc"], label="Positive current")
    axs[4].plot(times3, res_3_dict["v_oc"], label="Cosine current")
    axs[4].set_ylabel("v_oc")
    axs[4].set_xlabel("Time (s)")
   
    # add 0.5 alpha value to all lines so we can see overlapping lines better
    for ax in axs:
        for line in ax.get_lines():
            line.set_alpha(0.5)
     # only add axes for bottom ax, add below the graph,
    axs[-1].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -2))
    plt.show()

# TODO: check sign convention for current,
if __name__ == "__main__":
    plot_tests(t_max = 60**2, pbar=True)  # 1 hour simulation