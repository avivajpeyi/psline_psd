import numpy as np
from skfda.representation.basis import BSplineBasis


class PSplines:
    """A class for generating Penalised B-spline basis, its weights and its linear combination spline model

    B-splines of order n are basis functions for spline functions of the same order
    defined over the same knots, meaning that all possible spline functions can be
    built from a linear combination of B-splines, and there is only
    one unique combination for each spline function

    The "P" in PSplines stands for "Penalised", which means that the spline model
    is penalised to avoid overfitting.

    """

    def __init__(
        self, knots: np.array, degree: int, diffMatrixOrder: int = 2, n_xpts=50
    ):
        self.n_xpts: int = n_xpts  # number of points to evaluate the basis functions at
        self.__xpts: np.ndarray = np.linspace(0, 1, self.n_xpts)
        self.knots: np.array = knots
        self.degree: int = degree
        self.basis: np.ndarray = self.__generate_basis_matrix(self.knots)
        self.diffMatrixOrder: int = diffMatrixOrder
        self.penalty_matrix: np.ndarray = self.__generate_penalty_matrix()

    @staticmethod
    def __generate_basis_matrix(
        xpts, knots: np.array, degree: int, normalised: bool = True
    ) -> np.ndarray:
        """Generate a B-spline basis of any degree given a set of knots

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
        n_knots = len(
            knots_with_boundary
        )  # number of knots (including the external knots)

        assert n_knots == degree * 2 + len(knots)

        B = BSpline.design_matrix(xpts, knots_with_boundary, degree)

        if normalised:
            # normalize the basis functions
            mid_to_end_knots = knots_with_boundary[degree + 1 :]
            start_to_mid_knots = knots_with_boundary[: (n_knots - degree - 1)]
            bs_int = (mid_to_end_knots - start_to_mid_knots) / (degree + 1)
            bs_int[bs_int == 0] = np.inf
            B = B / bs_int

        if B.shape != (len(xpts), len(knots) + degree - 1):
            raise ValueError(f"Basis matrix has incorrect shape: {B.shape}")

        return B

    @staticmethod
    def __generate_penalty_matrix(self) -> np.ndarray:
        """
        Generate a penalty matrix of any order
        """
        pass

    def __call__(
        self,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Generate a spline model from a vector of spline coefficients and a list of B-spline basis functions
        """
        pass

    def plot_basis(
        self, ax=None, weights=None, basis_kwargs={}, spline_kwargs={}, knots_kwargs={}
    ):
        """Plot the basis + knots.

        If weights are provided, plot the spline model as well.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on. If None, a new figure and axes is created.
            (Useful if you want to plot the basis functions on data)
        weights : np.ndarray
            Vector of spline coefficients (length k)
        basis_kwargs : dict
            Keyword arguments to pass to the plot(basis)
        spline_kwargs : dict
            Keyword arguments to pass to the plot(spline)
        knots_kwargs : dict
            Keyword arguments to pass to the plot(knots)
        """
        pass

    def plot_penalty_matrix(self):
        """Plot the penalty matrix"""
        pass


def test_pslines():
    data = [
        0.0,
        -0.86217009,
        -2.4457478,
        -2.19354839,
        -2.32844575,
        -0.48680352,
        -0.41055718,
        -3.0,
    ]

    knots = np.linspace(0, 1, 5)
    degree = 3
    psline = PSplines(knots, degree)
