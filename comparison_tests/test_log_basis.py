import os

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
    return basis
    # return (basis - np.mean(basis, axis=0)) / np.std(basis, axis=0)


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


def get_colors(ncols, colorbar="viridis"):
    cmap = plt.get_cmap(colorbar)
    norm = plt.Normalize(0, 1)  # This represents the entire colorbar range
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    return [cbar.to_rgba(i / ncols) for i in range(ncols)]


def plot_comparison(x, knts, degree=3) -> plt.Figure:
    r_matrix = r_basismatrix(x, knts, degree=degree)
    py_matrix = py_basismatrix(x, knts, degree=degree)

    num_basis = r_matrix.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    colors = get_colors(num_basis)
    for i in range(num_basis):
        c = colors[i]
        axes[0, 0].plot(x, r_matrix[i], color=c)
        axes[0, 0].plot(x, py_matrix[i], color=c, ls="--")
        axes[0, 1].semilogx(x, r_matrix[i], color=c)
        axes[0, 1].semilogx(x, py_matrix[i], color=c, ls="--")
        axes[1, 0].scatter(r_matrix[i], py_matrix[i], color=c)
        axes[1, 1].scatter(r_matrix[i], py_matrix[i], color=c)

    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")

    axes[0, 0].vlines(knts, color="k", ymin=-0.1, ymax=0.2, zorder=10)
    axes[0, 1].vlines(knts, color="k", ymin=-0.1, ymax=0.2, zorder=10)

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


def test_different_number_of_knots():
    oudir = "out_basis_with_different_knots"
    os.makedirs(oudir, exist_ok=True)
    for n in range(5, 45, 5):
        x = np.linspace(0, 1, 500)
        knts = np.geomspace(0.0001, 1, n)
        degree = 3
        fig = plot_comparison(x, knts, degree=degree)
        fig.suptitle(f"{n} log spaced knots")
        fig.tight_layout()
        fig.savefig(os.path.join(oudir, f"n_{n}.png"))


def test_py_and_r_basis_low_log_values():
    oudir = "out_basis_at_low_log_values"
    os.makedirs(oudir, exist_ok=True)
    n = 3
    x = np.linspace(0, 1, 500)
    knts = np.geomspace(0.0001, 1, n)
    degree = 3

    r_matrix = r_basismatrix(x, knts, degree=degree)
    py_matrix = py_basismatrix(x, knts, degree=degree)
    num_basis = r_matrix.shape[0]

    colors = get_colors(num_basis)

    for i in range(num_basis):
        fig, axes = plt.subplots(
            1, 2, figsize=(8, 4), sharex=True, sharey=True
        )
        for ax in axes:
            ax.set_xlabel("x-grid")
            ax.set_ylabel("basis")
            # log both axes
            ax.set_xscale("log")
            ax.set_yscale("log")

        axes[0].set_title(f"R Basis")
        axes[1].set_title(f"Python Basis")

        c = colors[i]
        axes[0].scatter(x, r_matrix[i], color=c)
        axes[1].scatter(x, py_matrix[i], color=c, ls="--")

        # add txtbox with num nans in top left corner
        txt = f"nans: {np.isnan(r_matrix[i]).sum()}"
        axes[0].text(
            0.05,
            0.95,
            txt,
            transform=axes[0].transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        txt = f"nans: {np.isnan(py_matrix[i]).sum()}"
        axes[1].text(
            0.05,
            0.95,
            txt,
            transform=axes[1].transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(os.path.join(oudir, f"basis_{i}.png"))

    # for i in range(num_basis):
    #     c = colors[i]
    #     axes[0].scatter(x, r_matrix[i], color=c)
    #     axes[1].scatter(x, py_matrix[i], color=c, ls="--")
    # plt.xscale("log")
    # plt.yscale("log")
