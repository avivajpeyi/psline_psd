import matplotlib.pyplot as plt
import numpy as np

LATEX_LABELS = dict(
    φ=r"$\phi$",
    δ=r"$\delta$",
    τ=r"$\tau$",
)

def plot_metadata(
    post_samples, counts, psd_quants, periodogram, db_list, knots, v, metadata_plotfn
):
    fig = plt.figure(figsize=(5, 8), layout="constrained")
    gs = plt.GridSpec(5, 2, figure=fig)
    for i, p in enumerate(["φ", "δ", "τ"]):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(post_samples[:, i], color=f"C{i}")
        ax.set_ylabel(LATEX_LABELS[p])
        ax.set_xlabel("Iteration")
        ax = fig.add_subplot(gs[i, 1])
        ax.hist(post_samples[:, i], bins=50, color=f"C{i}")
        ax.set_xlabel(LATEX_LABELS[p])
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(counts, color="C3")
    ax.set_ylabel("Frac accepted")
    ax.set_xlabel("Iteration")
    ax = fig.add_subplot(gs[3, 1])
    for i, db in enumerate(db_list.T):
        ax.plot(db, color=f"C{i}", alpha=0.3)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel("Splines")
    ax = fig.add_subplot(gs[4, :])

    # plot the periodogram and the posterior median and 90% CI
    psd_up, psd_low = psd_quants[2, 1:], psd_quants[1, 1:]
    psd_x = np.linspace(0, 1, len(psd_up))
    ax.plot(psd_x, psd_quants[0, 1:], color="tab:orange")
    ax.fill_between(
        psd_x,
        psd_low,
        psd_up,
        color="tab:orange",
        alpha=0.2,
        label="Model",
    )
    ylims = ax.get_ylim()
    xpts = np.linspace(0, 1, len(periodogram))
    ax.scatter([], [], color="k", label="Data", zorder=-10,  s=0.5)
    ax.scatter(xpts, periodogram, color="k", zorder=-10,  s=0.5)
    ax.plot([], [], color="tab:red", label=f"{len(knots)} Knts")
    ax.set_xlim(xpts[2], xpts[-2])
    ax_twin = ax.twinx()
    # turn axes off
    ax_twin.set_yticks([])
    ax_twin.set_ylim(0, 1)
    # plot knots locations
    ax_twin.vlines(knots, 0, 0.1, color="tab:red", alpha=0.5)
    ax.set_ylim(ylims)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_ylabel("PSD")
    fig.tight_layout()
    fig.savefig(metadata_plotfn)
    plt.close(fig)
