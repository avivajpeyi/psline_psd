import os

import numpy as np

from slipper.sample import LogPsplineSampler, PsplineSampler

NTOTAL = 200


def test_base_smpler(test_pdgrm: np.ndarray, tmpdir: str):
    np.random.seed(0)
    outdir = f"{tmpdir}/mcmc/linear"
    fn = f"{outdir}/summary.png"
    PsplineSampler.fit(
        data=test_pdgrm,
        outdir=outdir,
        sampler_kwargs=dict(Ntotal=NTOTAL, n_checkpoint_plts=2),
        spline_kwargs=dict(
            degree=3,
            k=10,
            knot_locator_type="data_peak",
        ),
    )
    assert os.path.exists(fn)


def test_lnspline_sampler(test_pdgrm: np.ndarray, tmpdir: str):
    np.random.seed(0)
    outdir = f"{tmpdir}/mcmc/log"
    fn = f"{outdir}/summary.png"
    LogPsplineSampler.fit(
        data=test_pdgrm,
        outdir=outdir,
        sampler_kwargs=dict(Ntotal=NTOTAL, n_checkpoint_plts=2),
        spline_kwargs=dict(
            degree=3,
            k=10,
            knot_locator_type="data_peak",
        ),
    )
    assert os.path.exists(fn)
