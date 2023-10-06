import matplotlib.pyplot as plt
import numpy as np

from slipper.sample.pspline_sampler import PsplineSampler
from slipper.sample.pspline_sampler.bayesian_functions import (
    _vPv,
    llike,
    lprior,
    sample_φδτ,
)
from slipper.splines.utils import unroll_list_to_new_length


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
        assert np.allclose(ar, test["expected"]), f"{ar} != {test['expected']}"


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
    assert np.isclose(_vPv(v, P), 1.442495205)
    val = lprior(
        k=5,
        v=v,
        τ=0.1591549431,
        τα=0.001,
        τβ=0.001,
        φ=1,
        φα=1,
        φβ=1,
        δ=1,
        δα=1e-04,
        δβ=1e-04,
        P=P,
    )
    assert np.isclose(val, 0.1120841558)


def test_llike(test_pdgrm, tmpdir):
    sampler = PsplineSampler(data=test_pdgrm)
    sampler._init_mcmc()

    τ, δ, φ = (
        sampler.samples["τ"][0],
        sampler.samples["δ"][0],
        sampler.samples["φ"][0],
    )
    V = sampler.samples["V"][0]
    llike_val = llike(
        v=V, τ=τ, data=test_pdgrm, spline_model=sampler.spline_model
    )
    assert not np.isnan(llike_val)


def test_sample_prior(test_pdgrm, tmpdir):
    sampler = PsplineSampler(data=test_pdgrm)
    sampler._init_mcmc()

    kwargs = dict(
        k=sampler.n_basis,
        v=sampler.samples["V"][0],
        τ=None,
        τα=0.001,
        τβ=0.001,
        φ=None,
        φα=2,
        φβ=1,
        δ=1,
        δα=1e-4,
        δβ=1e-4,
        data=test_pdgrm,
        spline_model=sampler.spline_model,
    )

    N = 500
    pri_samples = np.zeros((N, 3))
    for i in range(N):
        pri_samples[i, :] = sample_φδτ(**kwargs)

    # plot histogram of pri_samples
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for i in range(3):
        axes[i].hist(pri_samples[:, i], bins=50)
        axes[i].set_xlabel(["φ'", "δ'", "τ'"][i])
    plt.tight_layout()
    plt.savefig(f"{tmpdir}/test_sample_prior.png")
