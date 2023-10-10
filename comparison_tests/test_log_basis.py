import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

from slipper.splines.p_splines import PSplines

utils = importr("utils")
base = importr("base")

try:
    bsplinePsd = importr("bsplinePsd")
except Exception as e:
    print(e)
    utils.install_packages("bsplinePsd")
    bsplinePsd = importr("bsplinePsd")


def _norm_basis(basis):
    return (basis - np.mean(basis, axis=0)) / np.std(basis, axis=0)


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


def plot_comparison(x, knts, degree=3) -> plt.Figure:
    r_matrix = r_basismatrix(x, knts, degree=degree)
    py_matrix = py_basismatrix(x, knts, degree=degree)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    for i in range(r_matrix.shape[0]):
        axes[0, 0].plot(x, r_matrix[i], color=f"C{i}")
        axes[0, 0].plot(x, py_matrix[i], color=f"C{i}", ls="--")
        axes[0, 1].semilogx(x, r_matrix[i], color=f"C{i}")
        axes[0, 1].semilogx(x, py_matrix[i], color=f"C{i}", ls="--")
        axes[1, 0].plot(r_matrix[i], py_matrix[i], color=f"C{i}")
        axes[1, 1].loglog(r_matrix[i], py_matrix[i], color=f"C{i}")

    for knt in knts:
        axes[0, 0].axvline(knt, color="k", ls="--", alpha=0.3)
        axes[0, 1].axvline(knt, color="k", ls="--", alpha=0.3)

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
    x = np.linspace(0, 1, 100)
    knts = np.linspace(0, 1, 4)
    degree = 3
    fig = plot_comparison(x, knts, degree=degree)
    fig.suptitle("Uniformly spaced knots")
    fig.tight_layout()

    x = np.linspace(0, 1, 100)
    knts = np.geomspace(0.001, 1, 4)
    degree = 3
    fig = plot_comparison(x, knts, degree=degree)
    fig.suptitle("log spaced knots")
    fig.tight_layout()

    plt.show()
