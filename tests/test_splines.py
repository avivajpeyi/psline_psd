import matplotlib.pyplot as plt
import numpy as np

from slipper.splines.p_splines import PSplines


def test_spline_creation(tmpdir):
    """Test that the splines can be generated"""
    degree = 2
    knots = sorted(np.random.uniform(0, 1, 5))
    # add 0 and 1
    knots = [0] + knots + [1]
    n_basis = len(knots) + degree - 1
    weights = np.random.uniform(0, 1, n_basis)
    pspline = PSplines(knots=knots, degree=degree, diffMatrixOrder=1)
    fig, ax = pspline.plot(weights=weights)
    fig.savefig(f"{tmpdir}/test_spline_creation.png")
    plt.close()


def test_initial_guess(test_pdgrm, true_psd, tmpdir):
    N = len(test_pdgrm)
    knots = np.linspace(0, 1, 30)
    pspline = PSplines(knots=knots, degree=2, diffMatrixOrder=1, logged=True)
    w0 = pspline.guess_weights(
        test_pdgrm, fname=f"{tmpdir}/test_spline_init_guess.png"
    )
    newx = np.linspace(0, 1, N)
    fig, ax = pspline.plot_basis(weights=w0, basis_kwargs=dict(alpha=0.2))
    plt.plot(newx, test_pdgrm, ",k")
    plt.tight_layout()
    fig.savefig(f"{tmpdir}/test_spline_init_guess.png")

    model = pspline(weights=w0)
    model_x = np.linspace(0, 1, len(model))
    test_pdgrm_x = np.linspace(0, 1, len(test_pdgrm) - 1)
    true_psd_x = np.linspace(0, 1, len(true_psd))
    plt.close("all")
    plt.figure()
    plt.scatter(test_pdgrm_x, test_pdgrm[1:], c="k", s=1)
    plt.plot(true_psd_x, true_psd, "tab:blue", label="True PSD")
    plt.plot(model_x, model, "tab:orange", label="PSpline")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()
