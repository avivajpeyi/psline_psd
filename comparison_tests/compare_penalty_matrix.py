import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects as robjects
import skfda
from matplotlib.colors import TwoSlopeNorm
from rpy2.robjects.packages import importr


def get_r_penalty_matrix(k, degree, diffMatrixOrder, knots):
    """
    library(fda)
    knots = seq(from = 0, to = 1, length = k - degree + 1);
    nknots   = length(knots);
    basisobj = fda::create.bspline.basis(
        c(0, knots[nknots-1]), norder = degree + 1,
        nbasis = k - 1, breaks = knots[-nknots]);
    P = fda::bsplinepen(basisobj, Lfdobj = diffMatrixOrder)
    """
    importr("fda")
    robjects.r(f"k <- {k}")
    robjects.r(f"degree <- {degree}")
    robjects.r(f"diffMatrixOrder <- {diffMatrixOrder}")
    robjects.r(f"knots <- c({','.join(map(str, knots))})")
    # robjects.r("knots <- seq(from = 0, to = 1, length = k - degree + 1)")
    robjects.r("nknots <- length(knots)")
    robjects.r(
        "basisobj <- fda::create.bspline.basis(c(0, knots[nknots-1]), norder = degree + 1, nbasis = k - 1, breaks = knots[-nknots])"
    )
    robjects.r(
        "plot(basisobj, knots=TRUE, las=1, lwd=2, main=sprintf('Num basis = %d' ,basisobj$nbasis))"
    )
    robjects.r("savePlot('r_basisobj.png', type='png')")
    robjects.r("P <- fda::bsplinepen(basisobj, Lfdobj = 1)")
    p = robjects.r["P"]
    return p / np.max(p)


def get_py_penalty_matrix(k, degree, diffMatrixOrder, knots):
    # knots = np.linspace(0, 1, k - degree)
    basis = skfda.representation.basis.BSplineBasis(knots=knots, order=degree + 1)
    basis.plot()
    plt.gcf().suptitle(f"Num basis = {len(basis)}")
    plt.savefig("python_basisobj.png")
    operator = skfda.misc.operators.LinearDifferentialOperator(diffMatrixOrder)
    regularization = skfda.misc.regularization.L2Regularization(operator)
    p = regularization.penalty_matrix(basis)
    return p / np.max(p)


def test_penalty_matrix():
    k = 10
    degree = 3
    diffMatrixOrder = 1
    knots = sorted(np.random.uniform(0, 1, k - degree-1))
    # prepend and postpend boundary knots
    knots = np.r_[0, knots, 1]
    print(knots)
    r_p = get_r_penalty_matrix(k, degree, diffMatrixOrder, knots)
    py_p = get_py_penalty_matrix(k, degree, diffMatrixOrder, knots)
    fig, ax = plt.subplots(1, 2)
    for i, (label, matrix) in enumerate(zip(("r", "python"), (r_p, py_p))):
        matrix = matrix / np.max(matrix)
        ax[i].set_title(label)
        norm = TwoSlopeNorm(vmin=matrix.min(), vcenter=0, vmax=matrix.max())
        im = ax[i].pcolor(
            matrix, ec="tab:gray", lw=0.005, cmap="bwr", norm=norm, antialiased=False
        )
        fig.colorbar(im, ax=ax[i], orientation="horizontal")
    plt.tight_layout()
    plt.savefig("penalty_matrix.png")
    assert np.allclose(r_p, py_p, atol=1e-10)


if __name__ == "__main__":
    test_penalty_matrix()
