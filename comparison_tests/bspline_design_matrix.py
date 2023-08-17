import matplotlib.pyplot as plt
import numpy as np
import skfda
from scipy.interpolate import BSpline

DATA = np.array(
    [
        0.0818115,
        0.02736034,
        0.25879249,
        0.39971164,
        0.11238972,
        0.04159213,
        0.93657936,
        0.56541016,
        0.20229957,
        0.060523,
        0.05961222,
        0.04776608,
        0.08804858,
        0.06760821,
        0.02292355,
        0.01055339,
        0.0251272,
        0.06051161,
        0.13331464,
        0.00454855,
        0.04053032,
        0.10396952,
        0.01447334,
        0.01394002,
        0.00248082,
    ]
)


def generate_bspline_basis(
    x: np.ndarray, knots: np.ndarray, degree=3, normalize=True
) -> np.ndarray:
    """Generate a B-spline basis of any degree

    Parameters:
    -----------
    x : np.ndarray of shape (n,)
    knots : np.ndarray of shape (k,)
    degree : int

    Returns:
    --------
    B : np.ndarray of shape (len(x), len(knots) + degree -1)


    """
    knots_with_boundary = np.r_[[knots[0]] * degree, knots, [knots[-1]] * degree]
    n_knots = len(knots_with_boundary)  # number of knots (including the external knots)
    assert n_knots == degree * 2 + len(knots)

    B = BSpline.design_matrix(x, knots_with_boundary, degree)

    if normalize:
        # normalize the basis functions
        mid_to_end_knots = knots_with_boundary[degree + 1 :]
        start_to_mid_knots = knots_with_boundary[: (n_knots - degree - 1)]
        bs_int = (mid_to_end_knots - start_to_mid_knots) / (degree + 1)
        bs_int[bs_int == 0] = np.inf
        B = B / bs_int

    assert B.shape == (len(x), len(knots) + degree - 1)
    # assert np.allclose(np.sum(B, axis=1), 1), 'Basis functions do not sum to 1'
    return B


def main():
    n_knots = 10
    knots = np.linspace(0, 1, n_knots)
    degree = 3
    xpts = np.linspace(0, 1, len(DATA))
    bspline_matrix = generate_bspline_basis(xpts, knots, degree)

    plt.figure()
    plt.imshow(bspline_matrix)
    plt.show()

    plt.figure()
    for i, db in enumerate(bspline_matrix.T):
        plt.plot(db, color=f"C{i}", alpha=0.3)
    plt.show()

    basis = skfda.representation.basis.BSplineBasis(order=degree + 1, knots=knots)
    plt.figure()
    basis.plot()
    plt.show()

    X = skfda.FDataGrid(data_matrix=DATA, sample_points=xpts)
    X_basis = X.to_basis(basis)
    X_basis.plot()
    plt.show()

    plt.figure()
    X_basis.basis.plot()

    plt.show()


if __name__ == "__main__":
    main()
