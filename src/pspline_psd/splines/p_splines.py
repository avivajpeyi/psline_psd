import numpy as np
from skfda.representation.basis import BSplineBasis
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from .utils import density_mixture
from .generator import convert_v_to_weights, unroll_list_to_new_length


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
            self, knots: np.array, degree: int, diffMatrixOrder: int = 2, n_grid_points=None
    ):
        """Initialise the PSplines class

        Parameters:
        ----------
        knots : np.array
            The knots of the spline basis
        degree : int
            The degree of the spline basis
        diffMatrixOrder : int
            The order of the difference matrix used to calculate the penalty matrix
        n_grid_points : int
            The number of points to evaluate the basis functions at
            If None, then the number of grid points is set to the maximum
            between 501 and 10 times the number of basis elements.
        """

        self.knots: np.array = knots
        self.degree: int = degree

        self.n_grid_points: int = n_grid_points  # number of points to evaluate the basis functions at
        self.diffMatrixOrder: int = diffMatrixOrder
        self.penalty_matrix: np.ndarray = self.__generate_penalty_matrix()
        self.basis: np.ndarray = self.__generate_basis_matrix()

    @property
    def n_grid_points(self) -> int:
        return self._n_grid_points

    @n_grid_points.setter
    def n_grid_points(self, n_grid_points: int):
        if n_grid_points is None:
            n_grid_points = max(501, 10 * self.n_basis)
        self._n_grid_points = n_grid_points

    @property
    def n_knots(self):
        n = len(self.knots)
        # assert n >= self.degree + 2, "k must be at least degree + 2"
        return n

    @property
    def n_basis(self) -> int:
        """Number of basis elements"""
        return self.n_knots + self.degree - 1

    @property
    def order(self) -> int:
        return self.degree + 1

    @property
    def grid_points(self) -> np.array:
        if not hasattr(self, '_grid_points'):
            self._grid_points = np.linspace(self.knots[0], self.knots[-1], self.n_grid_points)
        return self._grid_points

    def __get_fda_bspline_basis(self, knots=None):
        if knots is None:
            knots = self.knots
        return BSplineBasis(order=self.order, knots=knots)

    def __get_knots_with_boundary(self):
        """Add boundary knots to the knots array"""
        knots_with_boundary = np.concatenate(
            [
                np.repeat(self.knots[0], self.degree),
                self.knots,
                np.repeat(self.knots[-1], self.degree),
            ]
        )
        return knots_with_boundary

    def __generate_basis_matrix(self, normalised: bool = True) -> np.ndarray:
        """Generate a B-spline basis matrix of any degree given a set of knots

        Uses:
        grid_points : np.ndarray of shape (n,)
        knots : np.ndarray of shape (k,)
        degree : int

        Returns:
        --------
        basis_matrix : np.ndarray of shape (n_grid_points, n_basis_elements)
        """
        basis = self.__get_fda_bspline_basis().to_basis()
        basis_matrix = basis.to_grid(self.grid_points).data_matrix.squeeze().T

        if normalised:
            # normalize the basis functions
            knots_with_boundary = self.__get_knots_with_boundary()
            n_knots = len(knots_with_boundary)
            mid_to_end_knots = knots_with_boundary[self.degree + 1:]
            start_to_mid_knots = knots_with_boundary[: (n_knots - self.degree - 1)]
            bs_int = (mid_to_end_knots - start_to_mid_knots) / (self.degree + 1)
            bs_int[bs_int == 0] = np.inf
            basis_matrix = basis_matrix / bs_int

        expected_shape = (self.n_grid_points, self.n_basis)
        if basis_matrix.shape != expected_shape:
            raise ValueError(
                "Basis matrix has incorrect shape: "
                f"{basis_matrix.shape} != {expected_shape}"
            )

        return basis_matrix

    def __generate_penalty_matrix(self, epsilon = 1e-6) -> np.ndarray:
        """
        Generate a penalty matrix of any order
        Returns:
        --------
        penalty_matrix : np.ndarray of shape (n_basis_elements, n_basis_elements)
        """
        # exclude the last knot to avoid singular matrix
        basis = self.__get_fda_bspline_basis(knots=self.knots[0:-1])
        regularization = L2Regularization(LinearDifferentialOperator(self.diffMatrixOrder))
        p = regularization.penalty_matrix(basis)
        p / np.max(p)
        p = p + epsilon * np.eye(p.shape[1])  # P^(-1)=Sigma (Covariance matrix)
        return p

    def __call__(
            self,
            weights: np.ndarray = [],
            v: np.ndarray = [],
            n: int = None,
    ) -> np.ndarray:
        """
        Generate a spline model from a vector of spline coefficients and a list of B-spline basis functions
        """
        # check that weights or v is provided
        if len(weights) == 0 and len(v) == 0:
            raise ValueError("Either weights or v must be provided")
        elif len(weights) > 0 and len(v) > 0:
            raise ValueError("Only one of weights or v must be provided")
        elif len(weights) == 0 and len(v) > 0:
            weights = convert_v_to_weights(v)



        spline = density_mixture(weights, self.basis.T)

        if n is None:
            n = self.n_grid_points

        if len(spline) != n:
            spline = unroll_list_to_new_length(spline, n)

        return spline

    def plot_basis(
            self, ax=None, weights=None, basis_kwargs={}, spline_kwargs={}, knots_kwargs={}, plot_weighted_basis=False
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
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(5, 4))
        fig = ax.get_figure()

        plot_weighted_basis = plot_weighted_basis and weights is not None


        for i in range(self.n_basis):
            kwg = basis_kwargs.copy()
            kwg['color'] = kwg.get('color', f'C{i}')
            ax.plot(self.grid_points, self.basis[:, i], **kwg)
            if plot_weighted_basis:
                kwg['ls'] = kwg.get('ls', '--')
                weighted_b = self.basis[:, i] * weights[i]
                ax.plot(self.grid_points, weighted_b, **kwg)

        for i in range(self.n_knots):
            kwg = knots_kwargs.copy()
            kwg['color'] = kwg.get('color', 'tab:gray')
            kwg['marker'] = kwg.get('marker', 'o')
            kwg['ms'] = kwg.get('ms', 15)
            kwg['zorder'] = kwg.get('zorder', 10)
            ax.plot(self.knots[i], 0, **kwg)

        if weights is not None:
            kwg = spline_kwargs.copy()
            kwg['color'] = kwg.get('color', 'k')
            spline_model = self(weights)
            ax.plot(self.grid_points, spline_model, **kwg)

        ax.set_ylim(bottom=0)
        ax.set_xlabel('Grid points')
        ax.set_title('Basis functions')

        return fig, ax

    def plot_penalty_matrix(self, ax=None, **kwargs):
        """Plot the penalty matrix"""
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(4, 4))
        fig = ax.get_figure()

        matrix = self.penalty_matrix
        norm = TwoSlopeNorm(vmin=matrix.min(), vcenter=0, vmax=matrix.max())
        im = ax.pcolor(
            matrix, ec="tab:gray", lw=0.005, cmap="bwr", norm=norm, antialiased=False
        )
        ax.set_aspect("equal")
        ax.set_xlabel("Basis element")
        ax.set_ylabel("Basis element")
        fig.colorbar(im, ax=ax, orientation="vertical", label="Penalty")
        ax.set_title("Penalty matrix")
        return fig, ax

    def plot(self, weights=None):
        """Plot the basis functions and the penalty matrix"""
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        self.plot_basis(ax=ax[0], weights=weights, plot_weighted_basis=True)
        self.plot_penalty_matrix(ax=ax[1])
        plt.tight_layout()
        return fig, ax


def test_pslines():
    knots = np.linspace(0, 1, 5)
    degree = 2
    pspline = PSplines(knots, degree)
    pspline.plot(weights=np.random.randn(pspline.n_basis))
    plt.show()
