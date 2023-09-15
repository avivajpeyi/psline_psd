"""Fit some data with the spline model and log-spline model."""

import os
import numpy as np
import matplotlib.pyplot as plt
from slipper.example_datasets.ar_data import get_ar_periodogram
from slipper.plotting.utils import plot_xy_binned, set_plotting_style

from slipper.sample.spline_model_sampler import fit_data_with_pspline_model
from slipper.sample.spline_model_sampler import fit_data_with_log_spline_model
from scipy.interpolate import UnivariateSpline




set_plotting_style()

OUTDIR = "out_compare_spline_and_log_spline"
os.makedirs(OUTDIR, exist_ok=True)

AR4_PSD = get_ar_periodogram(order=4, n_samples=10000)[1:-1]


def plot_data_and_fits(data, fits={}):
    fig, ax = plt.subplots()
    data_x = np.linspace(0, 1, len(data))
    ax.semilogy(data_x, data, ",k", alpha=0.2)
    plot_xy_binned(data_x, data, ax, bins=30, label="Data", ls='--', ms=0)
    for i, (name, fit) in enumerate(fits.items()):
        ax_t= ax.twinx()
        # color ax_t spine, tick and labeles to match the data
        ax_t.spines["right"].set_color(f"C{i}")
        ax_t.tick_params(axis="y", colors=f"C{i}")

        # space the axes out a bit
        ax_t.spines["right"].set_position(("axes", 1.1 + 0.1 * i))
        # ax_t = ax
        spline_med, spline_p05, spline_p95 = (
            fit[0, :],
            fit[1, :],
            fit[2, :],
        )
        spline_x = np.linspace(0, 1, len(spline_med))
        ax_t.semilogy(spline_x, spline_med, label=name, color=f"C{i}")
        ax_t.fill_between(spline_x, spline_p05, spline_p95, color=f"C{i}", alpha=0.2)
        ax.plot([],[], color=f"C{i}", label=name)
    ax.legend()
    ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power Spectral Density")
    plt.tight_layout()
    return fig, ax


def main():
    plot_data_and_fits(AR4_PSD)
    plt.savefig(f"{OUTDIR}/data_and_fits.png")

    kwargs = dict(data=AR4_PSD, Ntotal=200, degree=3, eqSpaced=False,n_checkpoint_plts=2)
    spline_mcmc = fit_data_with_pspline_model(**kwargs,outdir=OUTDIR+"/spline")
    ln_spline_mcmc = fit_data_with_log_spline_model(**kwargs,outdir=OUTDIR+"/log_spline")

    plot_data_and_fits(AR4_PSD, fits={
        "Linear-Spline": spline_mcmc.psd_quantiles,
        "Log-Spline": ln_spline_mcmc.psd_quantiles
    },
    )
    plt.show()



if __name__ == '__main__':
    main()
