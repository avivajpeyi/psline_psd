import matplotlib.pyplot as plt
import numpy as np
import pytest

from slipper.example_datasets.lisa_data import (
    lisa_noise_periodogram,
    lisa_wd_strain,
)
from slipper.fourier_methods import get_periodogram
from slipper.plotting.plot_spline_model_and_data import (
    plot_spline_model_and_data,
)
from slipper.sample import LogPsplineSampler
from slipper.splines.knot_locator import KnotLocatorType

from .conftest import mkdir


def __plot_res(pdgrm, res, title):
    fig = plot_spline_model_and_data(
        data=pdgrm,
        model_quants=res.psd_quantiles,
        knots=res.knots,
        add_legend=False,
        logged_axes=["x", "y"],
        hide_axes=False,
    )
    fig.suptitle(title)
    ax = fig.axes[0]
    ax.set_xlabel("Scaled frequency")
    ax.set_ylabel("PSD")
    plt.tight_layout()
    return fig


# @pytest.mark.skip("Fails due to -inf qunatiles")
def test_fit_lisa_noise_linear_knots(tmpdir):
    np.random.seed(42)
    pdgrm = lisa_noise_periodogram()
    # keep every 5th point to speed up analysis for testing
    pdgrm = pdgrm[::5]

    outdir = mkdir(f"{tmpdir}/lisa/noise/linear_knots")
    res = LogPsplineSampler.fit(
        data=pdgrm,
        outdir=outdir,
        sampler_kwargs=dict(Ntotal=100, n_checkpoint_plts=2, burnin=10),
        spline_kwargs=dict(
            k=30,
            knot_locator_type=KnotLocatorType.linearly_spaced,
        ),
    )
    fig = __plot_res(pdgrm, res, "LISA noise")
    fig.savefig(f"{outdir}/fit.png")


def test_fit_lisa_noise_binned_knots(tmpdir):
    np.random.seed(42)
    pdgrm = lisa_noise_periodogram()
    # keep every 5th point to speed up analysis for testing
    pdgrm = pdgrm[::5]

    outdir = mkdir(f"{tmpdir}/lisa/noise/binned_knots")
    res = LogPsplineSampler.fit(
        data=pdgrm,
        outdir=outdir,
        sampler_kwargs=dict(
            Ntotal=100, n_checkpoint_plts=2, burnin=10, thin=1
        ),
        spline_kwargs=dict(
            k=30,
            knot_locator_type=KnotLocatorType.binned_knots,
            data_bin_edges=[10**-3, 10**-2.5, 10**-2, 0.1, 0.5],
            data_bin_weights=[0.1, 0.3, 0.4, 0.2, 0.2, 0.1],
            log_data=True,
            n_grid_points=500,
        ),
    )
    fig = __plot_res(pdgrm, res, "LISA noise")
    fig.savefig(f"{outdir}/fit.png")


def test_fit_list_wd_background(tmpdir):
    np.random.seed(42)
    timeseries = lisa_wd_strain()[0:1000]
    pdgrm = get_periodogram(timeseries=timeseries)

    outdir = mkdir(f"{tmpdir}/lisa/wdb/linear_knots")
    res = LogPsplineSampler.fit(
        data=pdgrm,
        outdir=outdir,
        sampler_kwargs=dict(
            Ntotal=100, n_checkpoint_plts=2, burnin=10, thin=1
        ),
        spline_kwargs=dict(
            k=30,
            knot_locator_type=KnotLocatorType.linearly_spaced,
        ),
    )
    fig = __plot_res(pdgrm, res, "White dwarf background")
    fig.savefig(f"{outdir}/fit.png")
