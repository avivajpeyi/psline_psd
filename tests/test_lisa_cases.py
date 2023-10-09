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
        logged_axes=True,
        hide_axes=False,
    )
    fig.suptitle(title)
    ax = fig.axes[0]
    ax.set_xlabel("Scaled frequency")
    ax.set_ylabel("PSD")
    plt.tight_layout()
    return fig


def test_fit_lisa_noise(tmpdir):
    np.random.seed(42)
    pdgrm = lisa_noise_periodogram()
    # keep every 5th point
    pdgrm = pdgrm[::5]

    res = LogPsplineSampler.fit(
        data=pdgrm,
        outdir=mkdir(f"{tmpdir}/lisa/noise"),
        sampler_kwargs=dict(Ntotal=200, n_checkpoint_plts=2),
        spline_kwargs=dict(
            k=30,
            knot_locator_type=KnotLocatorType.linearly_spaced,
            min_val=10**-3,
        ),
    )
    fig = __plot_res(pdgrm, res, "LISA noise")
    fig.savefig(f"{tmpdir}/lisa/noise/fit.png")


# @pytest.mark.skip(reason="Fails -- -inf lnl")
def test_fit_list_wd_background(tmpdir):
    np.random.seed(42)
    timeseries = lisa_wd_strain()[0:1000]
    pdgrm = get_periodogram(timeseries=timeseries)

    res = LogPsplineSampler.fit(
        data=pdgrm,
        outdir=mkdir(f"{tmpdir}/lisa/wdb"),
        sampler_kwargs=dict(Ntotal=200, n_checkpoint_plts=2),
        spline_kwargs=dict(
            k=30,
            knot_locator_type=KnotLocatorType.linearly_spaced,
        ),
    )
    fig = __plot_res(pdgrm, res, "White dwarf background")
    fig.savefig(f"{tmpdir}/lisa/wdb/fit.png")
