"""Initialisation functions for the penalised B-splines."""
from typing import List, Tuple

import numpy as np
import scipy
from scipy.interpolate import interp1d

from .p_splines import PSplines


def knot_locator(
    data: np.ndarray,
    k: int,
    degree: int,
    eqSpaced: bool = False,
    data_bin_edges: List[float] = None,
    data_bin_weights: List[float] = None,
    **kwargs,
) -> np.array:
    """Determines the knot locations for a B-spline basis of degree `degree` and `k` knots.

    Parameters
    ----------
    data : np.ndarray
        The data to be fitted
    k : int
        The number of basis functions (note, n-knots = k - degree + 1)

    degree : int
        The degree of the B-spline basis

    eqSpaced : bool, optional
        Whether to use equally spaced knots, by default False

    data_bin_edges : List[float], optional
        The frequency bins to use for the knot locations, by default None

    data_bin_weights : List[float], optional
        The weights to use for each frequency bins, by default None
        NOTE: len(wfreqbin) = len(nfreqbin) + 1
        For example, if nfreqbin = [0.1, 0.2, 0.3] and wfreqbin = [0.1, 0.2, 0.6, 0.1],
        then the first knot will be placed at 0.1, the second at 0.2, and the third at 0.3.

    Returns
    -------
    knots : np.ndarray of shape (k - degree + 1,)
    (The x-positions of the knots)

    """
    # This is to be in the function definition

    # Equal spaced knots

    if eqSpaced:
        knots = np.linspace(0, 1, num=k - degree + 1)

    elif data_bin_edges is None:

        aux = np.log(data)

        # aux = np.sqrt(data)
        dens = np.abs(aux - np.mean(aux)) / np.std(aux)
        n = len(data)

        dens = dens / np.sum(dens)
        cumf = np.cumsum(dens)

        df = interp1d(
            np.linspace(0, 1, num=n), cumf, kind="linear", fill_value=(0, 1)
        )

        invDf = interp1d(
            df(np.linspace(0, 1, num=n)),
            np.linspace(0, 1, num=n),
            kind="linear",
            fill_value=(0, 1),
            bounds_error=False,
        )

        # knots based on data peaks
        knots = invDf(np.linspace(0, 1, num=k - degree + 1))
    else:

        nknots = k - degree + 1  # num of internal knots
        # periodogram (By Patricio)
        log_data = np.log(data)
        N = len(log_data)

        if (len(data_bin_edges) + 1) != len(data_bin_weights):
            raise ValueError(
                "length of data_bin_edges is incorrect"
                f"data_bin_edges: {len(data_bin_edges)}, "
                f"data_bin_weights: {len(data_bin_weights)}"
            )

        # Knots placement based on log periodogram (Patricio code) This is when nfreqbin is an array
        if data_bin_weights is None:
            data_bin_weights = np.ones(len(data_bin_edges) + 1)
            data_bin_weights = data_bin_weights / np.sum(data_bin_weights)

        data_bin_weights = data_bin_weights / np.sum(data_bin_weights)
        n_bin_weights = len(data_bin_weights)

        data_bin_edges = np.sort(data_bin_edges)
        # eqval = np.concatenate(([0], nfreqbin / (fs / 2), [1]))
        eqval = np.concatenate(([0], data_bin_edges, [1]))  # Interval [0,1]
        eqval = np.column_stack(
            (eqval[:-1], eqval[1:])
        )  # Each row represents the bin
        j = np.linspace(0, 1, num=N)
        s = np.arange(1, N + 1)
        index = []

        for i in range(n_bin_weights):
            cond = (j >= eqval[i, 0]) & (j <= eqval[i, 1])
            index.append((np.min(s[cond]), np.max(s[cond])))

        Nindex = len(index)

        nknots = nknots - 2  # to include 0 and 1 in the knot vector
        kvec = np.round(nknots * np.array(data_bin_weights))
        kvec = kvec.astype(int)

        while np.sum(kvec) > nknots:
            kvec[np.argmax(kvec)] = np.max(kvec) - 1

        while np.sum(kvec) < nknots:
            kvec[np.argmin(kvec)] = np.min(kvec) + 1

        knots = []

        for i in range(Nindex):
            aux = data[index[i][0] : index[i][1]]

            # aux = np.sqrt(aux) in case using pdgrm
            dens = np.abs(aux - np.mean(aux)) / np.std(aux)

            Naux = len(aux)

            dens = dens / np.sum(dens)
            cumf = np.cumsum(dens)
            x = np.linspace(eqval[i][0], eqval[i][1], num=Naux)

            # Distribution function
            df = interp1d(x, cumf, bounds_error=False, fill_value=(0, 1))
            dfvec = df(x)
            invDf = interp1d(
                dfvec,
                x,
                kind="linear",
                fill_value=(x[0], x[-1]),
                bounds_error=False,
            )
            v = np.linspace(0, 1, num=kvec[i] + 2)
            v = v[1:-1]
            knots = np.concatenate((knots, invDf(v)))

        knots = np.concatenate(([0], knots, [1]))

    # Check that knots are unique
    if len(knots) != len(np.unique(knots)):
        raise ValueError("knots must be unique")

    return knots


def _get_initial_spline_data(
    data: np.ndarray, k: int, degree: int, diffMatrixOrder: int, eqSpaced: bool
) -> Tuple[np.ndarray, np.ndarray, PSplines]:
    knots = knot_locator(data, k, degree, eqSpaced)
    psplines = PSplines(
        knots=knots, degree=degree, diffMatrixOrder=diffMatrixOrder
    )
    V = psplines.guess_initial_v(data)
    return V, knots, psplines
