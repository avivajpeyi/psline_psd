import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import minimize
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.representation.basis import BSplineBasis

from slipper.plotting.utils import hide_axes_spines

from .utils import convert_v_to_weights, density_mixture, unroll_list_to_new_length


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
            self, knots: np.array, degree: int, diffMatrixOrder: int = 2, n_grid_points=None, logged=False
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
        logged : bool
            If True, the penalty matrix is calculated using all the knots
            If False, the penalty matrix is calculated using all the knots except the last one
        """
        assert degree > diffMatrixOrder
        assert degree in [0, 1, 2, 3, 4, 5]
        assert diffMatrixOrder in [0, 1, 2]
        assert len(knots) >= degree, f"#knots: {len(knots)}, degree: {degree}"

        self.knots: np.array = knots
        self.degree: int = degree

        self.n_grid_points: int = (
            n_grid_points  # number of points to evaluate the basis functions at
        )
        self.diffMatrixOrder: int = diffMatrixOrder
        # basically if log-splines, we use all knots for the penalty matrix, otherwise we use all knots except the last one
        self.penalty_matrix: np.ndarray = self.__generate_penalty_matrix(all_knots=True if logged else False)
        self.basis: np.ndarray = self.__generate_basis_matrix()
        self.logged = logged

    def __str__(self):
        str = f"PSplines(n_basis={self.n_basis}, n_knots={self.n_knots}, degree={self.degree})"
        if self.logged:
            str = f"logged {str}"
        return str

    def __repr__(self):
        return self.__str__()

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
        if not hasattr(self, "_grid_points"):
            self._grid_points = np.linspace(
                self.knots[0], self.knots[-1], self.n_grid_points
            )
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

    def __generate_penalty_matrix(self, epsilon=1e-6, all_knots=False) -> np.ndarray:
        """
        Generate a penalty matrix of any order
        Returns:
        --------
        penalty_matrix : np.ndarray of shape (n_basis_elements, n_basis_elements)
        """

        if self.are_equidistant_knots:
            # diffMatrix
            """
              out = diag(k);

              for(i in 0:diffMatrixOrder-1){
            
                out = diff(out);
            
              }
            """

            if all_knots:
                k = self.n_basis
            else:
                k = self.n_basis - 1

            diffMatrix = np.diag(np.repeat(1, k)).T

            for i in range(self.diffMatrixOrder):
                diffMatrix = np.diff(diffMatrix)

            # penalty matrix
            p = np.matmul(diffMatrix, diffMatrix.T)

        else:
            if all_knots:
                basis = self.__get_fda_bspline_basis(knots=self.knots)
            else:
                # exclude the last knot to avoid singular matrix
                basis = self.__get_fda_bspline_basis(knots=self.knots[0:-1])
            regularization = L2Regularization(
                LinearDifferentialOperator(self.diffMatrixOrder)
            )
            p = regularization.penalty_matrix(basis)
            p / np.max(p)

        return p + epsilon * np.eye(p.shape[1])  # P^(-1)=Sigma (Covariance matrix)

    @property
    def are_equidistant_knots(self):
        return np.allclose(np.diff(self.knots), np.diff(self.knots)[0])

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
            self,
            ax=None,
            weights=None,
            basis_kwargs={},
            spline_kwargs={},
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

        ax.set_xlim(self.grid_points[0], self.grid_points[-1])
        weighed_ax = ax.twinx() if weights is not None else None

        for i in range(self.n_basis):
            kwg = basis_kwargs.copy()
            kwg["color"] = kwg.get("color", f"C{i}")
            ax.plot(self.grid_points, self.basis[:, i], **kwg)
            if weights is not None:
                kwg["ls"] = kwg.get("ls", "--")
                weighted_b = self.basis[:, i] * weights[i]
                weighed_ax.plot(self.grid_points, weighted_b, **kwg)

        if weights is not None:
            kwg = spline_kwargs.copy()
            kwg["color"] = kwg.get("color", "k")
            spline_model = self(weights)
            weighed_ax.plot(self.grid_points, spline_model, **kwg)
            hide_axes_spines(weighed_ax)
            weighed_ax.set_ylim(bottom=0, top=np.max(spline_model) * 1.1)
            weighed_ax.set_xlim(ax.get_xlim())

        median_basis_i = float(np.median(np.max(self.basis, axis=0)))
        ax.set_ylim(0, median_basis_i * 1.1)
        ax.set_xlabel("Grid points")
        ax.set_title("Basis functions")

        # use knots as xticks
        ax.set_xticks(self.knots)
        ax.set_xticklabels(
            [
                f"{self.knots[i]:.1f}"
                if i in [0, int(self.n_knots / 2), self.n_knots - 1]
                else ""
                for i in range(self.n_knots)
            ]
        )

        # add textbox top left
        textstr = f"# basis = {self.n_basis}\n# knots = {self.n_knots}"
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )
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

    def plot(self, weights=None, V=None):
        """Plot the basis functions and the penalty matrix"""
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        if V is not None:
            weights = convert_v_to_weights(V)
        self.plot_basis(ax=ax[0], weights=weights)
        self.plot_penalty_matrix(ax=ax[1])
        plt.tight_layout()
        return fig, ax

    def guess_weights(self, data, n_optimization_steps=10):
        """Guess init 'w' weights for the P-spline model from the data and the knots

        NOTE: length of w is n_basis
        """
        w = np.zeros(self.n_basis)

        # ignore the 1st aand last
        data = data[1:-1]
        n = len(data)

        orig_grid_ln = self.n_grid_points
        self.n_grid_points = n
        res = minimize(
            lambda w: _mse(self(w), data),
            options=dict(
                maxiter=self.n_basis * n_optimization_steps,
                xatol=1e-30,
                disp=False,
            ),
            bounds=[(0, None)] * self.n_basis,
            x0=w,
            method="Nelder-Mead",
        )
        w = res.x
        w[w == 0] = 1e-50  # prevents log(0) errors
        w = w / np.sum(w)
        self.n_grid_points = orig_grid_ln
        return w

    def guess_initial_v(self, data):
        """Guess init 'v' weights for the P-spline model from the data and the knots
        NOTE: length of v is n_basis - 1
        """
        w0 = self.guess_weights(data)[:-1]
        w0[w0 <= 0] = 1e-20
        v = np.log(w0 / (1 - np.sum(w0)))
        # convert nans to very small
        v[np.isnan(v)] = -1e50
        v = v.reshape(self.n_basis - 1, 1)
        return v


def _mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)
