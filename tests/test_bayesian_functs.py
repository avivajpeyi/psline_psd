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

MAKE_PLOTS = True


def test_psd_unroll():
    ar = unroll_list_to_new_length(np.array([1, 2, 3, 4]), n=8)
    assert np.allclose(ar, np.array([1, 2, 2, 3, 3, 4, 4, 4]))
    ar = unroll_list_to_new_length(np.array([1, 2, 3]), n=6)
    assert np.allclose(ar, np.array([1, 2, 2, 3, 3, 3]))
    ar = unroll_list_to_new_length(np.array([1, 2, 3]), n=5)
    assert np.allclose(ar, np.array([1, 2, 2, 3, 3]))


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


def test_llike(helpers):
    data = helpers.load_raw_data()
    degree = 3
    k = 32
    τ, δ, φ, fz, periodogram, V, omega = _get_initial_values(data, k)
    fz = get_fz(data)

    periodogram = get_periodogram(fz)
    knots = knot_locator(data, k=k, degree=degree, eqSpaced=True)
    spline_model = PSplines(knots, degree=degree)
    llike_val = llike(v=V, τ=τ, pdgrm=periodogram, spline_model=spline_model)
    assert not np.isnan(llike_val)
    psd = spline_model(v=V)
    assert not np.isnan(psd).any()

    ll_vals = helpers.load_ll()
    highest_ll_idx = np.argmax(ll_vals)
    best_V = helpers.load_v()[:, highest_ll_idx]
    best_τ = helpers.load_tau()[highest_ll_idx]
    best_llike_val = llike(v=best_V, τ=best_τ, pdgrm=periodogram, spline_model=spline_model)

    # assert best_llike_val == ll_vals[highest_ll_idx]
    assert np.abs(llike_val - best_llike_val) < 100
    best_psd = spline_model(v=best_V)

    if MAKE_PLOTS:
        fig = __plot_psd(
            periodogram,
            [psd, best_psd],
            [f"PSD lnl{llike_val:.2f}", f"PSD lnl{best_llike_val:.2f}"],
            spline_model.basis,
        )
        fig.savefig(f"{helpers.OUTDIR}/test_llike.png")
        fig.show()


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


def test_sample_prior(helpers):
    data = helpers.load_raw_data()
    data = data - np.mean(data)
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
    τ0, δ0, φ0, fz, periodogram, V0, omega = _get_initial_values(**kwargs)
    db_list, P = _get_initial_spline_data(
        data, k, degree, omega, diffMatrixOrder, eqSpacedKnots=True
    )
    v = _generate_initial_weights(periodogram, k)
    # create dict with k, v, τ, τα, τβ, φ, φα, φβ, δ, δα, δβ, periodogram, db_list, P

    kwargs = dict(
        k=k,
        v=v,
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
        db_list=db_list,
        P=P,
    )

    N = 5000
    pri_samples = np.zeros((N, 3))
    for i in range(N):
        pri_samples[i, :] = sample_φδτ(**kwargs)

    # plot histogram of pri_samples
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for i in range(3):
        axes[i].hist(pri_samples[:, i], bins=50)
        axes[i].set_xlabel(["φ'", "δ'", "τ'"][i])
    plt.tight_layout()
    plt.show()
