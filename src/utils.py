"""
Will Buesnel, Jul 26.

Unless specified otherwise, all functions relating to plotting here come from the following source:
https://github.com/MarkBlyth/parameterisation_methodsx/research/scripts/utils.py
This is to keep the style of plots etc inkeeping with his,
and is permissable under the GPU Licence that both repos have.
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import List
import csv


def get_ax(save: bool = False, n_axes: int = 1, **ax_kwargs):
    if save:
        matplotlib.use("pgf")
        matplotlib.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
            }
        )
    ax = subjectively_better_subplots(n_axes, **ax_kwargs)

    def tidy_up(filename: str | None = None):
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

    return ax, tidy_up


def subjectively_better_subplots(
    nrows,
    subheight=10 / 3,
    subwidth=10,
    bottom_extra=0.2,
    vertical_padding=0.25,
    l_margin=1,
    r_margin=0.1,
    **kwargs,
):
    # Source - https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
    # Retrieved 2025-12-08, License - CC BY-SA 4.0
    lm = l_margin / 2.54
    rm = r_margin / 2.54
    ax_padding = vertical_padding / 2.54
    a = subheight / 2.54
    w = subwidth / 2.54
    width = lm + w + rm
    height = nrows * (2 * ax_padding + a) + bottom_extra
    fig = plt.figure(figsize=(width, height), **kwargs)
    axarr = np.empty(nrows, dtype=object)
    for i in range(nrows):
        axarr[i] = fig.add_axes(
            [
                lm / width,
                (height - (i + 1) * (2 * ax_padding + a) + ax_padding) / height,
                w / width,
                a / height,
            ]
        )
    if nrows == 1:
        return axarr[0]
    return axarr