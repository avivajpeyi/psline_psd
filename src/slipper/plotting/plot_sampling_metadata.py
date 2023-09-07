import os

import matplotlib.pyplot as plt
import numpy as np

from .plot_spline_model_and_data import plot_spline_model_and_data
from .utils import plot_xy_binned

LATEX_LABELS = dict(
    φ=r"$\phi$",
    δ=r"$\delta$",
    τ=r"$\tau$",
)


def plot_metadata(
    φδτ_samples: np.ndarray,
    frac_accepted: np.array,
    lpost_trace: np.ndarray,
    model_quants: np.ndarray,
    data,
    db_list,
    knots,
    weights,
    burn_in,
    fname=None,
    max_it=None,
    logged_splines=False,
):
    φδτ_samples[φδτ_samples == 0] = np.nan
    frac_accepted[frac_accepted == 0] = np.nan

    fig = plt.figure(figsize=(7, 12), layout="constrained")
    gs = plt.GridSpec(6, 2, figure=fig)
    draw_idx = np.arange(len(φδτ_samples))
    max_it = len(φδτ_samples) if max_it is None else max_it
    for i, p in enumerate(["φ", "δ", "τ"]):
        # TRACE
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(draw_idx[1:], φδτ_samples[1:, i], color=f"C{i}")
        ax.axvline(burn_in, color="k", linestyle="--")
        ax.set_ylabel(LATEX_LABELS[p])
        ax.set_xlabel("Iteration")
        ax.set_xlim(0, max_it)

        # HISTOGRAM
        ax = fig.add_subplot(gs[i, 1])
        samps = φδτ_samples[:, i]
        samps = samps[~np.isnan(samps)]
        if len(samps[burn_in:]) > 0:
            ax.hist(samps[burn_in:], bins=50, color=f"C{i}", density=True)
        else:
            ax.hist(samps[0:], bins=50, color=f"C{i}", density=True)
        ax.set_yticks([])
        ax.set_xlabel(LATEX_LABELS[p])

    # FRAC ACCEPTED TRACE
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(frac_accepted, color="C3")
    ax.axvline(burn_in, color="k", linestyle="--")
    ax.set_ylabel("Accepted %")
    ax.set_xlabel("Iteration")
    ax.set_xlim(0, max_it)

    # LOG POSTERIOR TRACE
    ax = fig.add_subplot(gs[3, 1])
    ax.plot(lpost_trace, color="C4")
    ax.axvline(burn_in, color="k", linestyle="--")
    ax.set_ylabel("LnPost")
    ax.set_yticks([])
    ax.set_xlabel("Iteration")
    ax.set_xlim(0, max_it)

    # splines
    ax = fig.add_subplot(gs[4, 0])
    for i, db in enumerate(db_list.T):
        ax.plot(db, color=f"C{i}", alpha=0.3)
    # max db_val in each row
    spline_ymedian = float(np.median(np.max(db_list, axis=1)))
    ax.set_ylim(0, 1.1 * spline_ymedian)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel("Splines")

    # weights
    ax = fig.add_subplot(gs[4, 1])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Spline")
    ax.set_yticks([])
    # fill with 0s if
    # pcolor plot of weights with colorbar
    cbar = ax.pcolor(weights.T, cmap="magma")
    cbar = fig.colorbar(cbar, ax=ax)



    # plot the data and the posterior median and 90% CI
    ax = fig.add_subplot(gs[5, :])
    plot_spline_model_and_data(
        data, model_quants, separarte_y_axis=True, ax=ax, knots=knots, logged_axes=logged_splines
    )
    if fname:
        basedir = os.path.dirname(fname)
        os.makedirs(basedir, exist_ok=True)
        fig.savefig(fname)
        plt.close(fig)
    else:
        return fig
