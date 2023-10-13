import matplotlib.pyplot as plt
import numpy as np
import pytest

from slipper.sample.pspline_sampler import PsplineSampler
from slipper.sample.pspline_sampler.bayesian_functions import (
    LnlArgs,
    _xPx,
    lprior,
    sample_φδτ,
)
from slipper.splines.utils import unroll_list_to_new_length


@pytest.fixture
def lnlargs_for_test(test_pdgrm):
    sampler = PsplineSampler(data=test_pdgrm, spline_kwargs=dict())
    sampler._init_mcmc()
    return LnlArgs(
        w=sampler.samples["V"][0],
        τ=0.1591549431,
        τα=0.001,
        τβ=0.001,
        φ=1,
        φα=1,
        φβ=1,
        δ=1,
        δα=1e-04,
        δβ=1e-04,
        data=test_pdgrm,
        spline_model=sampler.spline_model,
    )


def test_psd_unroll():
    test_args = [
        dict(
            old_list=np.array([1, 2, 3, 4]),
            n=8,
            expected=np.array([1, 1, 2, 2, 3, 3, 4, 4]),
        ),
        dict(
            old_list=np.array([1, 2, 3]),
            n=6,
            expected=np.array([1, 1, 2, 2, 3, 3]),
        ),
        dict(
            old_list=np.array([1, 2, 3]),
            n=5,
            expected=np.array([1, 1, 2, 2, 3]),
        ),
        dict(
            old_list=np.array([1, 2, 3]), n=4, expected=np.array([1, 2, 2, 3])
        ),
    ]

    for test in test_args:
        ar = unroll_list_to_new_length(test["old_list"], n=test["n"])
        assert ar is not None


def test_lprior():
    v = np.array([-68.6346650, 4.4997348, 1.6011013, -0.1020887])
    P = np.array(
        [
            [1e-6, 0.00, 0.0000000000, 0.0000000000],
            [0.00, 1e-6, 0.0000000000, 0.0000000000],
            [0.00, 0.00, 0.6093175700, 0.3906834292],
            [0.00, 0.00, 0.3906834292, 0.3340004330],
        ]
    )
    assert np.isclose(_xPx(v, P), 1.442495205)


def test_llike(test_pdgrm, tmpdir):
    sampler = PsplineSampler(data=test_pdgrm, spline_kwargs=dict())
    sampler._init_mcmc()

    τ, δ, φ = (
        sampler.samples["τ"][0],
        sampler.samples["δ"][0],
        sampler.samples["φ"][0],
    )
    V = sampler.samples["V"][0]
    lnl_val = sampler.spline_model.lnlikelihood(data=test_pdgrm, v=V)
    assert not np.isnan(lnl_val)


def test_sample_prior(lnlargs_for_test, tmpdir):
    pri_samples = sample_φδτ(lnlargs_for_test)
    # assert none of pri_samples are nan
    assert not np.any(np.isnan(pri_samples))
    prior_val = lprior(lnlargs_for_test)
    assert not np.isnan(prior_val)
