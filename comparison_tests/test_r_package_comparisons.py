import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.fft import fft

from slipper.fourier_methods import get_fz
from slipper.sample.post_processing import generate_spline_quantiles
# from slipper.sample.pspline_sampler.bayesian_functions import llike
from slipper.sample.spline_model_sampler import fit_data_with_pspline_model, fit_data_with_log_spline_model
from slipper.splines.initialisation import knot_locator
from slipper.plotting.utils import plot_xy_binned
import os
from pathlib import Path
from slipper.example_datasets.ar_data import get_ar_periodogram, generate_ar_timeseries, get_periodogram

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
    data = generate_ar_timeseries(order=3, n_samples=2000)
    pdgrm = get_periodogram(timeseries=data)
    pdgrm = pdgrm[1:]
    psd_x = np.linspace(0, 1, len(pdgrm))
    fig,ax = plt.subplots(1,1, figsize=(8, 4))
    plt.scatter(psd_x, pdgrm, color="k", label="Data", s=0.1)
    plot_xy_binned(psd_x, pdgrm, ax=ax, color="k", label="Data", ms=2, ls='--')
    plt.yscale("log")
    plt.show()
    r_mcmc = __r_mcmc(data, nsteps)
    py_mcmc = __py_mcmc(data, nsteps)
    py_log_mcmc = __log_py_mcmc(data, nsteps)


    plt.figure(figsize=(8, 4))
    plt.rcParams["font.family"] = "sans-serif"
    plt.plot(figsize=(8, 4))
    plt.scatter(psd_x, pdgrm, color="k", label="Data", s=0.1)
    # plot_xy_binned(psd_x, pdgrm, ax=plt.gca(),color="k", label="Data", ms=1)

    for i, (label, mcmc) in enumerate(zip(['r', 'py', 'log-py'],[r_mcmc, py_mcmc, py_log_mcmc])):
        x = np.linspace(0, 1, len(mcmc.psd_quantiles[0]))
        plt.plot(
            x[1:],
            mcmc.psd_quantiles[0][1:],
            color=f"C{i}",
            alpha=0.5,
            label=label,
        )
        plt.fill_between(
            x[1:],
            mcmc.psd_quantiles[1][1:],
            mcmc.psd_quantiles[2][1:],
            color=f"C{i}",
            alpha=0.2,
            linewidth=0.0,
        )

    plt.grid(False)
    plt.legend(markerscale=5, frameon=False)
    plt.tight_layout()
    plt.yscale("log")
    # turn off minor ticks
    plt.minorticks_off()
    plt.show()



def __r_mcmc(data, nsteps):
    r_pspline = importr("psplinePsd")

    np_cv_rules = default_converter + numpy2ri.converter

    burnin = int(0.5 * nsteps)
    with np_cv_rules.context():
        mcmc = r_pspline.gibbs_pspline(
            data, burnin=burnin, Ntotal=nsteps, degree=3, eqSpaced=True
        )
    return MCMCdata.from_r(mcmc)


def __py_mcmc(data, nsteps):
    burnin = int(0.5 * nsteps)
    pdgm =np.abs(get_fz(data))
    mcmc = fit_data_with_pspline_model(
        pdgm,
        burnin=burnin,
        Ntotal=nsteps,
        degree=3,
        eqSpaced=True,
        outdir="py_mcmc",
    )

    return mcmc


def __log_py_mcmc(data, nsteps):
    burnin = int(0.5 * nsteps)
    pdgm =np.abs(get_fz(data))
    mcmc = fit_data_with_log_spline_model(
        pdgm,
        burnin=burnin,
        Ntotal=nsteps,
        degree=3,
        eqSpaced=True,
        outdir="py_mcmc_log",
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
                np.array(mcmc["psd.u05"]),
                np.array(mcmc["psd.u95"]),
            ]
        )
        obj.lnl = None
        obj.samples = np.array([mcmc["phi"], mcmc["delta"], mcmc["tau"]]).T
        return obj
