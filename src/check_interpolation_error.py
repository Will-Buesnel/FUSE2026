"""Will Buesnel, Jul 26.

Run the electrical model with interpolated and de-quantised current data
across multiple solver max_step values, then visualise the errors.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from models.coupled import test_simulation
from utils import get_abs_error, get_mse


def _build_current_functions(exp_df: pd.DataFrame):
    elapsed_time = exp_df["Elapsed Time[h]"].to_numpy() * 3600
    dequantised_time = exp_df["deq_Elapsed Time[h]"].to_numpy() * 3600
    current = exp_df["Current(A)"].to_numpy()

    current_func_interp = lambda t: -np.interp(t, elapsed_time, current)
    dequantised_current_interp = lambda t: -np.interp(t, dequantised_time, current)

    return current_func_interp, dequantised_current_interp, dequantised_time


def _run_single_trial(exp_df: pd.DataFrame, max_step: float, y0, cutoff_fraction: float = 0.75):
    current_func_interp, dequantised_current_interp, dequantised_time = _build_current_functions(exp_df)

    t_span = (dequantised_time[0], dequantised_time[-1])
    t_eval = dequantised_time
    cutoff_index = max(1, int(len(t_eval) * cutoff_fraction))

    res_interp = test_simulation(
        y0=y0,
        current_func=current_func_interp,
        verbose=False,
        pbar=True,
        max_step=max_step,
        t_eval=t_eval,
        t_span=t_span,
    )

    res_dequant = test_simulation(
        y0=y0,
        current_func=dequantised_current_interp,
        verbose=False,
        pbar=True,
        max_step=max_step,
        t_eval=t_eval,
        t_span=t_span,
    )

    y_true = exp_df["Voltage(V)"].to_numpy()[:cutoff_index]
    y_pred_interp = np.asarray(res_interp["v_oc"])[:cutoff_index]
    y_pred_dequant = np.asarray(res_dequant["v_oc"])[:cutoff_index]

    return {
        "max_step": max_step,
        "abs_error_interp": get_abs_error(y_true, y_pred_interp),
        "abs_error_dequant": get_abs_error(y_true, y_pred_dequant),
        "mse_interp": get_mse(y_true, y_pred_interp),
        "mse_dequant": get_mse(y_true, y_pred_dequant),
    }


def run_max_step_sweep(
    max_steps=None,
    y0=None,
    cutoff_fraction: float = 0.75,
    show_plot: bool = True,
):
    if max_steps is None:
        max_steps = [0.25, 0.5, 1.0, 2.0, 5.0]

    root = Path.cwd().resolve()
    processed_data_dir = root / "data" / "processed"
    wltp_file = processed_data_dir / "MLP001_wltp_25degC_record_deq.csv"

    exp_df = pd.read_csv(wltp_file)

    if y0 is None:
        y0 = [1, 0, 0, 25]

    results = []
    for max_step in max_steps:
        print(f"Running trial with max_step={max_step}...")
        metrics = _run_single_trial(
            exp_df=exp_df,
            max_step=max_step,
            y0=y0,
            cutoff_fraction=cutoff_fraction,
        )
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values("max_step").reset_index(drop=True)

    print("\nResults:")
    print(results_df.to_string(index=False))

    if show_plot:
        sns.set_theme(style="whitegrid")

        plot_df = results_df.melt(
            id_vars="max_step",
            value_vars=[
                "abs_error_interp",
                "abs_error_dequant",
                "mse_interp",
                "mse_dequant",
            ],
            var_name="metric",
            value_name="value",
        )

        plot_df["method"] = plot_df["metric"].map(
            {
                "abs_error_interp": "Interpolated",
                "abs_error_dequant": "De-quantised",
                "mse_interp": "Interpolated",
                "mse_dequant": "De-quantised",
            }
        )
        plot_df["metric_type"] = plot_df["metric"].map(
            {
                "abs_error_interp": "Absolute error",
                "abs_error_dequant": "Absolute error",
                "mse_interp": "MSE",
                "mse_dequant": "MSE",
            }
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

        sns.lineplot(
            data=plot_df[plot_df["metric_type"] == "Absolute error"],
            x="max_step",
            y="value",
            hue="method",
            marker="o",
            ax=axes[0],
        )
        axes[0].set_title("Absolute error vs max_step")
        axes[0].set_xlabel("max_step")
        axes[0].set_ylabel("Absolute error (V)")

        sns.lineplot(
            data=plot_df[plot_df["metric_type"] == "MSE"],
            x="max_step",
            y="value",
            hue="method",
            marker="o",
            ax=axes[1],
        )
        axes[1].set_title("MSE vs max_step")
        axes[1].set_xlabel("max_step")
        axes[1].set_ylabel("MSE (V$^2$)")

        plt.tight_layout()
        plt.show()

    return results_df


def main():
    run_max_step_sweep(
        max_steps=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
        cutoff_fraction=0.75,
        show_plot=True,
    )


if __name__ == "__main__":
    main()