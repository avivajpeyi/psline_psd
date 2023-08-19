import matplotlib.pyplot as plt
import numpy as np

from .plot_spline_model_and_data import plot_spline_model_and_data
from .utils import convert_axes_spines_to_arrows, hide_axes_spines

LATEX_LABELS = dict(
    φ=r"$\phi$",
    δ=r"$\delta$",
    τ=r"$\tau$",
)


def plot_metadata(
    post_samples, counts, psd_quants, data, db_list, knots, burn_in, metadata_plotfn
):
    fig = plt.figure(figsize=(5, 8), layout="constrained")
    gs = plt.GridSpec(5, 2, figure=fig)
    draw_idx = np.arange(len(post_samples))
    for i, p in enumerate(["φ", "δ", "τ"]):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(draw_idx, post_samples[:, i], color=f"C{i}")
        ax.axvline(burn_in, color="k", linestyle="--")
        ax.set_ylabel(LATEX_LABELS[p])
        ax.set_xlabel("Iteration")
        ax = fig.add_subplot(gs[i, 1])
        ax.hist(post_samples[burn_in:, i], bins=50, color=f"C{i}")
        ax.set_xlabel(LATEX_LABELS[p])
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(counts, color="C3")
    ax.axvline(burn_in, color="k", linestyle="--")
    ax.set_ylabel("Frac accepted")
    ax.set_xlabel("Iteration")
    ax = fig.add_subplot(gs[3, 1])
    for i, db in enumerate(db_list.T):
        ax.plot(db, color=f"C{i}", alpha=0.3)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel("Splines")
    ax = fig.add_subplot(gs[4, :])

    # plot the data and the posterior median and 90% CI
    plot_spline_model_and_data(
        data, psd_quants, separarte_y_axis=True, ax=ax, knots=knots
    )
    fig.tight_layout()
    if metadata_plotfn:
        fig.savefig(metadata_plotfn)
        plt.close(fig)
    else:
        return fig
