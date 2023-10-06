import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.fft import fft

from slipper.example_datasets.ar_data import (
    generate_ar_timeseries,
    get_ar_periodogram,
    get_periodogram,
)
from slipper.fourier_methods import get_fz
from slipper.plotting.utils import plot_xy_binned
from slipper.sample.post_processing import generate_spline_quantiles

# from slipper.sample.pspline_sampler.bayesian_functions import llike
from slipper.sample.spline_model_sampler import (
    fit_data_with_log_spline_model,
    fit_data_with_pspline_model,
)
from slipper.splines.initialisation import knot_locator

plt.style.use("default")
# import gridspec from matplotlib

try:
    import rpy2
    from rpy2.robjects import default_converter, numpy2ri
    from rpy2.robjects.packages import importr
except ImportError:
    rpy2 = None

np.random.seed(0)


def mkdir(path):
    path = str(path)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


@pytest.mark.skipif(rpy2 is None, reason="rpy2 required for this test")
def test_mcmc_posterior_psd_comparison():
    nsteps = 5000
    nsplines = 40
    eqSpaced = True

    data = generate_ar_timeseries(order=3, n_samples=1000)
    data = data - np.mean(data)
    rescale = np.std(data)
    data = data / rescale
    pdgrm = get_periodogram(timeseries=data)
    # pdgrm = pdgrm[1:]
    psd_x = np.linspace(0, 1, len(pdgrm))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    plt.scatter(
        psd_x[1:], pdgrm[1:] * rescale**2, color="k", label="Data", s=0.1
    )
    plot_xy_binned(
        psd_x[1:],
        pdgrm[1:] * rescale**2,
        ax=ax,
        color="k",
        label="Data",
        ms=2,
        ls="--",
    )
    plt.yscale("log")
    plt.show()

    py_log_mcmc = __log_py_mcmc(pdgrm, nsteps, nsplines, eqSpaced)
    py_mcmc = __py_mcmc(pdgrm, nsteps, nsplines, eqSpaced)
    r_mcmc = __r_mcmc(data, nsteps, nsplines, eqSpaced)

    outdir = mkdir(Path(__file__).parent / "out_compare_spline_and_log_spline")

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    __plot_result(
        pdgrm, [("r", r_mcmc), ("py", py_mcmc), ("log-py", py_log_mcmc)], ax
    )
    fig.tight_layout()
    fig.savefig(f"{outdir}/data_and_fits.png")

    fig, ax = plt.subplots(3, 1, figsize=(5, 10))
    __plot_sampled_params(
        [
            ("r", r_mcmc.samples),
            ("py", py_mcmc.post_samples),
            # ('log-py', py_log_mcmc.post_samples)
        ],
        ax,
    )

    fig.tight_layout()
    fig.savefig(f"{outdir}/sampled_params.png")

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    _plot_psd_samples(
        [
            ("r", r_mcmc.psd.T),
            ("py", py_mcmc.psd_posterior * rescale**2),
            ("log-py", py_log_mcmc.psd_posterior + np.log(rescale * 2)),
        ],
        ax,
    )
    fig.tight_layout()
    fig.savefig(f"{outdir}/psd_samples.png")


def __plot_sampled_params(mcmc_data, axes):
    axlabels = [r"$\phi$", r"$\delta$", r"$\tau$"]
    for param_id, ax in enumerate(axes):
        ax.grid(False)
        ax.set_xlabel(axlabels[param_id])
        for i in range(len(mcmc_data)):
            label, mcmc = mcmc_data[i]
            ax.hist(
                mcmc[:, param_id],
                color=f"C{i}",
                alpha=0.5,
                label=label,
                histtype="step",
                density=True,
                bins=30,
            )
        ax.legend(markerscale=5, frameon=False)
        ax.set_yscale("log")


def _plot_psd_samples(psd_data, ax):
    # plot last 90% of all samples

    for i in range(len(psd_data)):
        label, psds = psd_data[i]
        x = np.linspace(0, 1, len(psds[0]))
        start = int(0.1 * len(psds))
        end = len(psds)
        for j in range(start, end):
            ax.plot(x, psds[j], color=f"C{i}", alpha=0.01)
        ax.plot(x, psds[-1], color=f"C{i}", alpha=1, label=label)

    ax.grid(False)
    ax.set_yscale("log")
    ax.legend(markerscale=5, frameon=False)


def __plot_lnl(lnl_vals, ax):
    for i in range(len(lnl_vals)):
        label, lnl = lnl_vals[i]
        ax.plot(lnl, color=f"C{i}", alpha=0.5, labelTrue=label)
    ax.grid(False)
    ax.legend(markerscale=5, frameon=False)


def __plot_result(pdgrm, mcmc_data, ax):
    psd_x = np.linspace(0, 1, len(pdgrm))
    # ax.scatter(psd_x, pdgrm, color="k", label="Data", s=0.1)
    # plot_xy_binned(psd_x, pdgrm, ax=ax, color="k", label="Data", ms=0, ls='--')
    plt.scatter(
        psd_x[1:],
        pdgrm[1:],
        color="k",
        label="Data",
        marker=",",
        alpha=0.5,
        s=1,
    )
    for i in range(len(mcmc_data)):
        label, mcmc = mcmc_data[i]
        x = np.linspace(0, 1, len(mcmc.psd_quantiles[0]))
        ax.plot(
            x[1:],
            mcmc.psd_quantiles[0][1:],
            color=f"C{i}",
            alpha=0.5,
            label=label,
        )
        ax.fill_between(
            x[1:],
            mcmc.psd_quantiles[1][1:],
            mcmc.psd_quantiles[2][1:],
            color=f"C{i}",
            alpha=0.2,
            linewidth=0.0,
        )

    ax.grid(False)
    ax.legend(markerscale=5, frameon=False)
    ax.set_yscale("log")
    ax.minorticks_off()


def __r_mcmc(data, nsteps, nsplines, eqSpaced):
    r_pspline = importr("psplinePsd")

    np_cv_rules = default_converter + numpy2ri.converter

    burnin = int(0.5 * nsteps)
    with np_cv_rules.context():
        mcmc = r_pspline.gibbs_pspline(
            data,
            burnin=burnin,
            Ntotal=nsteps,
            degree=3,
            eqSpacedKnots=eqSpaced,
            k=nsplines,
        )
    return MCMCdata.from_r(mcmc)


# def __r_mcmc(data, nsteps, nsplines):
#     r_pspline = importr("psplinePsd")
#
#     np_cv_rules = default_converter + numpy2ri.converter
#
#     burnin = int(0.5 * nsteps)
#     with np_cv_rules.context():
#         mcmc = r_pspline.gibbs_pspline(
#             data, burnin=burnin, Ntotal=nsteps, degree=3, eqSpaced=True, k=nsplines
#         )
#     return MCMCdata.from_r(mcmc)


def __py_mcmc(data, nsteps, nsplines, eqSpaced):
    burnin = int(0.5 * nsteps)
    mcmc = fit_data_with_pspline_model(
        data,
        burnin=burnin,
        Ntotal=nsteps,
        degree=3,
        eqSpaced=eqSpaced,
        outdir="py_mcmc",
        k=nsplines,
    )
    return mcmc


def __log_py_mcmc(data, nsteps, nsplines, eqSpaced):
    burnin = int(0.5 * nsteps)
    mcmc = fit_data_with_log_spline_model(
        data,
        burnin=burnin,
        Ntotal=nsteps,
        degree=3,
        eqSpaced=eqSpaced,
        outdir="py_mcmc_log",
        k=nsplines,
        n_checkpoint_plts=10,
    )

    return mcmc


class MCMCdata:
    def __init__(self):
        self.fz = None
        self.v = None
        self.dblist = None
        self.psds = None
        self.psd_quantiles = None
        self.lnl = None
        self.samples = None

    @classmethod
    def from_r(cls, mcmc):
        obj = cls()
        obj.fz = None
        obj.v = mcmc["V"]
        obj.dblist = mcmc["db.list"]
        obj.psd = mcmc["fpsd.sample"]
        obj.psd_quantiles = np.array(
            [
                np.array(mcmc["psd.median"]),
                np.array(mcmc["psd.p05"]),
                np.array(mcmc["psd.p95"]),
            ]
        )
        obj.lnl = mcmc["ll.trace"]
        obj.samples = np.array([mcmc["phi"], mcmc["delta"], mcmc["tau"]]).T
        return obj
