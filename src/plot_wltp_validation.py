"""Will Buesnel, Jul 26."""

from __future__ import annotations
from utils import get_ax
import pandas as pd
import numpy as np
from pathlib import Path


def get_drive_cycle_df(cycler_file: str) -> pd.DataFrame:
    drive_df = pd.read_excel(cycler_file, engine="openpyxl", sheet_name="record")

    # add an hours column to df:
    # data are split into 'hours:seconds:milliseconds.'
    hours_arr = drive_df["Total Time"].str.split(":") # gives a 2d array of nx3
    arr = np.asarray(hours_arr.tolist(), dtype=float)
    hours = arr @ np.array([1, 1/60, 1/3600])

    # take away the initial time to get elapsed time.
    hours = hours - hours[0]

    drive_df["Elapsed Time"] = hours

    return drive_df


def make_plot(
        exp_df: pd.DataFrame,
        save_name: str | None = None
):
    "Currently unfinished; just has the functionality to plot experiment data."
    (ax_I, ax_v, ax_e), tidy_up = get_ax(
        bool(save_name),
        n_axes=3,
        bottom_extra=0.35,
        l_margin=1.7,
    )

    ax_I.set_xticks([])
    ax_v.set_xticks([])
   
    "plot experiment data"
    ax_v.plot(
            exp_df["Elapsed Time"],
            exp_df["Voltage(V)"],
            label="Experiment",
            color="tab:cyan",
        )

    tidy_up(save_name)

def main(cycler_file):
    exp_df = get_drive_cycle_df(cycler_file)
    make_plot(exp_df)

if __name__ == "__main__":
    root = Path.cwd().resolve()
    raw_data_dir = root / "data" / "raw"
    wltp_file = raw_data_dir / "MLP001_wltp_25degC.xlsx"
    main(cycler_file=wltp_file)

    

