import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xarray import Dataset

from slipper.logger import logger
from slipper.splines.utils import convert_v_to_weights

from .plot_spline_model_and_data import plot_spline_model_and_data
from .utils import plot_xy_binned

LABELS = dict(
    phi=r"$\phi$",
    delta=r"$\delta$",
    tau=r"$\tau$",
    acceptance_rate="Accep %",
    lp="LnP($\theta$|d)",
)


def plot_metadata(
    posterior: pd.DataFrame,
    model_quants: np.ndarray,
    data,
    spline_model: "PSplines",
    weights,
    burn_in,
    fname=None,
    max_it=None,
):
    posterior[posterior == 0] = np.nan
    n_rows = len(posterior.columns) + 2
    max_it = len(posterior) if max_it is None else max_it
    fig = plt.figure(figsize=(8, 3 * n_rows), layout="constrained")
    gs = plt.GridSpec(n_rows, 2, figure=fig)
    row_num = 0
    for i, p in enumerate(posterior.columns):
        param_post = posterior[p]
        if len(np.unique(param_post)) < 2:
            continue

        ax_trace = fig.add_subplot(gs[row_num, 0])
        ax_hist = fig.add_subplot(gs[row_num, 1])
        __plot_trace_and_hist(
            param_post,
            LABELS[p],
            ax_trace,
            ax_hist,
            burn_in,
            max_it,
            color=f"C{i}",
        )
        row_num += 1

    # spline axes
    ax_basis = fig.add_subplot(gs[row_num, 0])
    ax_weights = fig.add_subplot(gs[row_num, 1])
    __plot_spline_data(spline_model, weights, max_it, ax_basis, ax_weights)
    row_num += 1

    # plot the data and the posterior median and 90% CI
    ax = fig.add_subplot(gs[row_num, :])
    plot_spline_model_and_data(
        data,
        model_quants,
        separarte_y_axis=True,
        ax=ax,
        knots=spline_model.knots,
        logged_axes=spline_model.logged,
    )
    if fname:
        basedir = os.path.dirname(fname)
        if basedir:
            os.makedirs(basedir, exist_ok=True)
        fig.savefig(fname)
        plt.close(fig)
    else:
        return fig


def __plot_trace_and_hist(
    data, label, ax_trace, ax_hist, burn_in, max_it, color
):

    samps = data[~np.isnan(data)]
    low, med, high = np.quantile(samps, [0.05, 0.5, 0.95])
    l, h = med - low, high - med
    txt = f"${med:.2f}^{{+{h:.2f}}}_{{-{l:.2f}}}$"

    logger.debug(f"Plotting {label} trace: {txt}")

    ax_trace.tick_params(
        axis="both", which="both", direction="in", pad=-15, zorder=10
    )
    ax_hist.tick_params(
        axis="both", which="both", direction="in", pad=-15, zorder=10
    )

    ax_trace.axvline(burn_in, color="k", linestyle="--", zorder=10)
    ax_trace.axhline(med, color=color, linestyle="--", alpha=0.5)
    ax_trace.axhline(low, color=color, linestyle="--", alpha=0.2)
    ax_trace.axhline(high, color=color, linestyle="--", alpha=0.2)
    ax_trace.plot(data[1:], color=color)
    ax_trace.set_xlim(0, max_it)
    ax_trace.set_yticks([med])
    ax_trace.tick_params(axis="y", direction="in", pad=-15)
    ax_trace.tick_params(axis="x", direction="in")
    ax_trace.set_ylabel(label)
    ax_trace.set_xlabel("Iteration", labelpad=0, fontsize=10)
    # add txtbox with txt at bottom left
    ax_trace.text(
        0.05,
        0.95,
        txt,
        transform=ax_trace.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    if len(samps) < 5:
        return

    if len(samps[burn_in:]) > 0:
        ax_hist.hist(samps[burn_in:], bins=50, color=color, density=True)
    else:
        ax_hist.hist(samps[0:], bins=50, color=color, density=True)
    ax_hist.axvline(med, color=color, linestyle="--", alpha=0.5)
    ax_hist.axvline(low, color=color, linestyle="--", alpha=0.2)
    ax_hist.axvline(high, color=color, linestyle="--", alpha=0.2)
    ax_hist.tick_params(axis="x", direction="in")
    ax_hist.set_xticks([low, high])
    ax_hist.set_yticks([])


def __plot_spline_data(
    spline_model: "PSplines", weights, max_it, ax_basis, ax_weights
):
    w = weights[-1, :]
    if not spline_model.logged:
        w = convert_v_to_weights(w)
    spline_model.plot_basis(ax=ax_basis, weights=w)
    ax_basis.set_xlabel("")
    ax_basis.set_ylabel("Basis")
    ax_basis.tick_params(
        axis="x",
        direction="in",
        pad=-15,
        labelsize=0,
    )
    ax_basis.set_yticks([])
    ax_basis.set_title("")

    # weights
    ax_weights.set_xlabel("Iteration")
    ax_weights.set_ylabel("Weights")
    ax_weights.set_yticks([])
    cbar = ax_weights.pcolor(weights.T, cmap="magma")
    fig = ax_weights.get_figure()
    fig.colorbar(cbar, ax=ax_weights)
    ax_weights.set_xlim(0, max_it)
