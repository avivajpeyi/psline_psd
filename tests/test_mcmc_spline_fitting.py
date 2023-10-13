import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from slipper.logger import logger
from slipper.plotting.plot_spline_model_and_data import (
    plot_spline_model_and_data,
)
from slipper.sample import LogPsplineSampler, PsplineSampler
from slipper.sample.sampling_result import Result

logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])

NTOTAL = 200


MODEL_COL = "tab:orange"
DATA_COL = "tab:blue"


def _plot_spline(mcmc: Result, data, fn):
    fig = plot_spline_model_and_data(
        data=data,
        model_quants=mcmc.psd_quantiles,
        add_legend=True,
        logged_axes="y",
        hide_axes=False,
        metadata_text=mcmc.summary,
    )
    # fig.show()
    fig.savefig(fn)


def test_base_smpler(test_pdgrm: np.ndarray, tmpdir: str):
    np.random.seed(2)
    outdir = f"{tmpdir}/mcmc/linear"
    fn = f"{outdir}/summary.png"
    mcmc = PsplineSampler.fit(
        data=test_pdgrm,
        outdir=outdir,
        sampler_kwargs=dict(Ntotal=NTOTAL, n_checkpoint_plts=2),
        spline_kwargs=dict(
            degree=3,
            k=10,
            knot_locator_type="linearly_spaced",
        ),
    )
    _plot_spline(mcmc, test_pdgrm, fn=f"{outdir}/FIT.png")
    assert os.path.exists(fn)


def test_lnspline_sampler(test_pdgrm: np.ndarray, tmpdir: str):
    np.random.seed(0)
    outdir = f"{tmpdir}/mcmc/log"
    fn = f"{outdir}/summary.png"
    mcmc = LogPsplineSampler.fit(
        data=test_pdgrm,
        outdir=outdir,
        sampler_kwargs=dict(Ntotal=NTOTAL, n_checkpoint_plts=2),
        spline_kwargs=dict(
            degree=3,
            k=10,
            knot_locator_type="data_peak",
        ),
    )
    _plot_spline(mcmc, test_pdgrm, fn=f"{outdir}/FIT.png")

    assert os.path.exists(fn)
