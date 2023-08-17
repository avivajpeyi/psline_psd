import matplotlib.pyplot as plt
import numpy as np

from pspline_psd.bayesian_utilities import llike, lprior
from pspline_psd.bayesian_utilities.bayesian_functions import _vPv, sample_φδτ
from pspline_psd.sample.gibbs_pspline_simple import (
    _get_initial_spline_data,
    _get_initial_values,
)
from pspline_psd.splines.generator import build_spline_model, unroll_list_to_new_length
from pspline_psd.splines.initialisation import (
    _generate_initial_weights,
    knot_locator,
)
from pspline_psd.utils import get_fz, get_periodogram
from pspline_psd.splines.p_splines import PSplines




def test_psd_unroll():
    test_args = [
        dict(old_list=np.array([1, 2, 3, 4]),n=8,expected=np.array([1, 1, 2, 2, 3, 3, 4, 4])),
        dict(old_list=np.array([1, 2, 3]),n=6,expected=np.array([1, 1, 2, 2, 3, 3])),
        dict(old_list=np.array([1, 2, 3]),n=5,expected=np.array([1, 1, 2, 2, 3])),
        dict(old_list=np.array([1, 2, 3]),n=4,expected=np.array([1, 2, 2, 3])),
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


def test_llike(test_timeseries, tmpdir):
    degree = 3
    k = 32

    τ, δ, φ, fz, periodogram, omega = _get_initial_values(test_timeseries, k)
    V, knots, psplines = _get_initial_spline_data(
        test_timeseries, k, degree, diffMatrixOrder=2, eqSpacedKnots=True
    )
    fz = get_fz(test_timeseries)

    periodogram = get_periodogram(fz)
    knots = knot_locator(test_timeseries, k=k, degree=degree, eqSpaced=True)
    spline_model = PSplines(knots, degree=degree)
    llike_val = llike(v=V, τ=τ, pdgrm=periodogram, spline_model=spline_model)
    assert not np.isnan(llike_val)
    psd = spline_model(v=V)
    assert not np.isnan(psd).any()

    fig = __plot_psd(
        periodogram,
        [psd],
        [f"PSD lnl{llike_val:.2f}"],
        spline_model.basis,
    )
    fig.savefig(f"{tmpdir}/test_llike.png")
    plt.close(fig)


def __plot_psd(periodogram, psds, labels, db_list):
    plt.plot(periodogram / np.sum(periodogram), label="periodogram", color="k")
    for psd, l in zip(psds, labels):
        plt.plot(psd / np.sum(psd), label=l)
    ylims = plt.gca().get_ylim()
    basis = db_list
    net_val = max(periodogram)

    basis = basis / net_val
    for idx, bi in enumerate(basis.T):
        kwgs = dict(color=f"C{idx + 2}", lw=0.1, zorder=-1)
        if idx == 0:
            kwgs["label"] = "basis"
        bi = unroll_list_to_new_length(bi, n=len(periodogram))
        plt.plot(bi / net_val, **kwgs)
    plt.ylim(*ylims)
    plt.ylabel("PSD")
    plt.legend(loc="upper right")
    return plt.gcf()


def test_sample_prior(test_timeseries, tmpdir):
    data = test_timeseries - np.mean(test_timeseries)
    rescale = np.std(data)
    data = data / rescale

    k = 32
    degree = 3
    n = len(data)
    omega = np.linspace(0, 1, n // 2 + 1)
    diffMatrixOrder = 2

    kwargs = {
        "data": data,
        "k": k,
        "degree": degree,
        "omega": omega,
        "diffMatrixOrder": diffMatrixOrder,
    }
    τ0, δ0, φ0, fz, periodogram, omega = _get_initial_values(**kwargs)
    V, knots, psplines = _get_initial_spline_data(
        data, k, degree, diffMatrixOrder, eqSpacedKnots=True
    )
    # create dict with k, v, τ, τα, τβ, φ, φα, φβ, δ, δα, δβ, periodogram, db_list, P

    kwargs = dict(
        k=k,
        v=V,
        τ=None,
        τα=0.001,
        τβ=0.001,
        φ=None,
        φα=2,
        φβ=1,
        δ=1,
        δα=1e-4,
        δβ=1e-4,
        periodogram=periodogram,
        spline_model=psplines,
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
