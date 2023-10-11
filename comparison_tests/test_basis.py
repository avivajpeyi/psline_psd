import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

from slipper.splines.p_splines import PSplines, knot_locator

utils = importr("utils")
base = importr("base")

try:
    bsplinePsd = importr("bsplinePsd")
except Exception as e:
    print(e)
    utils.install_packages("bsplinePsd")
    bsplinePsd = importr("bsplinePsd")


def _norm_basis(basis):
    # return (basis - np.mean(basis, axis=0)) / np.std(basis, axis=0)
    return basis


def r_basismatrix(x, knots, degree=3):
    db_list = bsplinePsd.dbspline(
        x=robjects.FloatVector(x),
        knots=robjects.FloatVector(knots),
        degree=degree,
    )
    db_list = np.array(db_list)
    # call the following: db.list <- apply(db.list, 2, function(x) (x - mean(x))/sd(x)); # Standarization
    # db_list = (db_list - np.mean(db_list, axis=0)) / np.std(db_list, axis=0)
    db_list = _norm_basis(db_list)

    return db_list


def py_basismatrix(x, knots, degree=3):
    basis = PSplines(knots, degree=degree, n_grid_points=len(x)).basis.T
    basis = _norm_basis(basis)
    return basis


def plot_comparison(gridpts, knots, degree=3) -> plt.Figure:

    if isinstance(gridpts, int):
        x = np.linspace(0.000001, 1, gridpts)
    else:
        x = gridpts

    r_matrix = r_basismatrix(x, knots, degree=degree)
    py_matrix = py_basismatrix(x, knots, degree=degree)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    for i in range(r_matrix.shape[0]):
        axes[0, 0].plot(x, r_matrix[i], color=f"C{i}")
        axes[0, 0].plot(x, py_matrix[i], color=f"C{i}", ls="--")
        axes[0, 1].semilogx(x, r_matrix[i], color=f"C{i}")
        axes[0, 1].semilogx(x, py_matrix[i], color=f"C{i}", ls="--")
        axes[1, 0].plot(r_matrix[i], py_matrix[i], color=f"C{i}")
        axes[1, 1].loglog(r_matrix[i], py_matrix[i], color=f"C{i}")

    # for knt in knts:
    #     axes[0, 0].axvline(knt, color="k", ls="--", alpha=0.3)
    #     axes[0, 1].axvline(knt, color="k", ls="--", alpha=0.3)

    for i in range(2):
        axes[0, i].set_xlabel("x-grid")
        axes[0, i].set_ylabel("basis")
        axes[1, i].set_xlabel("R")
        axes[1, i].set_ylabel("Python")

    axes[0, 0].set_title(f"Linear Scale")
    axes[0, 1].set_title(f"Log Scale")
    plt.tight_layout()
    return fig


def test_basic():
    knts = knot_locator(knot_locator_type="linearly_spaced", n_knots=5)
    degree = 3
    fig = plot_comparison(100, knots=knts, degree=degree)
    fig.suptitle("Uniformly spaced knots LOG GRID")
    fig.tight_layout()

    knts = knot_locator(knot_locator_type="linearly_spaced", n_knots=5)
    degree = 3
    fig = plot_comparison(np.linspace(0, 1, 100), knots=knts, degree=degree)
    fig.suptitle("Uniformly spaced knots LINEAR GRID")
    fig.tight_layout()

    from slipper.example_datasets.lisa_data import lisa_noise_periodogram

    pdgrm = lisa_noise_periodogram()[::5]
    knts = knot_locator(
        knot_locator_type="binned_knots",
        n_knots=40,
        data=pdgrm,
        data_bin_edges=[10**-3, 10**-2.5, 10**-2, 0.1, 0.5],
        data_bin_weights=[0.1, 0.3, 0.4, 0.2, 0.2, 0.1],
        log_data=True,
    )
    degree = 3
    fig = plot_comparison(100, knts, degree=degree)
    fig.suptitle("log spaced knots  LOG GRID")
    fig.tight_layout()

    pdgrm = lisa_noise_periodogram()[::5]
    knts = knot_locator(
        knot_locator_type="binned_knots",
        n_knots=40,
        data=pdgrm,
        data_bin_edges=[10**-3, 10**-2.5, 10**-2, 0.1, 0.5],
        data_bin_weights=[0.1, 0.3, 0.4, 0.2, 0.2, 0.1],
        log_data=True,
    )
    degree = 3
    fig = plot_comparison(np.linspace(0, 1, 100), knts, degree=degree)
    fig.suptitle("log spaced knots  LINEAR GRID")
    fig.tight_layout()

    plt.show()


from slipper.example_datasets.lisa_data import lisa_noise_periodogram


def test_patricio_knots():
    data = lisa_noise_periodogram()[::5]
    knts = knot_locator(
        knot_locator_type="binned_knots",
        data_bin_edges=[10**-3, 10**-2.5, 10**-2, 0.1, 0.5],
        data_bin_weights=[0.1, 0.3, 0.4, 0.2, 0.2, 0.1],
        log_data=True,
        n_knots=40,
        data=data,
    )
    degree = 3

    fig = plot_comparison(100, knts, degree=degree)
