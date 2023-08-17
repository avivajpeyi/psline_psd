import matplotlib.pyplot as plt
import numpy as np

from pspline_psd.sample.gibbs_pspline_simple import _get_initial_values
from pspline_psd.splines.initialisation import  knot_locator
from pspline_psd.splines.p_splines import PSplines

MAKE_PLOTS = True


def test_spline_creation(tmpdir):
    """Test that the splines can be generated"""
    degree = 2
    knots = np.array([0, 1, 2, 3, 4, 5, 6])
    pspline = PSplines(knots=knots, degree=degree)
    assert pspline is not None
    fig, ax = pspline.plot(weights=np.random.randn(pspline.n_basis))
    if MAKE_PLOTS:
        plt.savefig(tmpdir.join("test_spline_creation.png"))
        plt.close()

def test_spline_basis_normalised(helpers):
    """Test that the spline basis are normalised"""
    x = np.linspace(-5, 5, 100)
    knots = np.array([-5, 0, 5])
    degree = 2
    basis = dbspline(x, knots=knots, degree=degree)

    assert np.allclose(np.sum(basis, axis=1), 1)

    if MAKE_PLOTS:
        for i in range(len(basis.T)):
            plt.plot(x, basis[:, i].ravel(), label=f"basis {i}", color=f"C{i}")

        # sum all basis functions
        plt.plot(x, np.sum(basis, axis=1), label="sum of basis functions")
        # plot knots
        plt.plot(np.array([-5, 0, 5]), np.zeros(3), "x", color="k", label="knots")

        plt.legend(loc="upper left")
        plt.savefig(f"{helpers.OUTDIR}/test_b_spline_matrix.png")


def test_basis_same_as_r_package_basis(helpers):
    """Test that the spline basis generated by this package
    are the same as those generated by the r-package"""
    # load raw data and true db_list
    data = helpers.load_raw_data()
    true_db_list = helpers.load_db_list()

    # generate basis functions
    degree = 3
    k = 32
    τ, δ, φ, fz, periodogram, V, omega = _get_initial_values(data, k)
    knots = knot_locator(data, k=k, degree=degree, eqSpaced=True)
    db_list = dbspline(omega, knots, degree=degree).T

    if MAKE_PLOTS:
        for i in range(k):
            if i == 0:
                plt.plot(true_db_list[i], color="gray", label="True")
                plt.plot(db_list[i], color="red", ls="--", label="Estimated")
            else:
                plt.plot(true_db_list[i], color="gray")
                plt.plot(db_list[i], color="red", ls="--")
        plt.xticks([])
        plt.yticks([])
        plt.title("Basis functions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{helpers.OUTDIR}/basis_comparison.png")

    residuals = np.sum(
        np.array([np.sum(np.abs(true_db_list[i] - db_list[i])) for i in range(k)])
    )
    assert residuals < 1e-5
