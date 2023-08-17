"""Initialisation functions for the penalised B-splines."""
import numpy as np
from scipy.interpolate import interp1d
from .p_splines import PSplines

from typing import Tuple, Union


def knot_locator(pdgrm: np.ndarray, k: int, degree: int, eqSpaced: bool = False) -> np.array:
    """Determines the knot locations for a B-spline basis of degree `degree` and `k` knots.

    Returns
    -------
    knots : np.ndarray of shape (k - degree + 1,)
    (The x-positions of the knots)

    """
    if eqSpaced:
        knots = np.linspace(0, 1, num=k - degree + 1)
        return knots

    aux = np.sqrt(pdgrm)
    dens = np.abs(aux - np.mean(aux)) / np.std(aux)
    n = len(pdgrm)

    dens = dens / np.sum(dens)
    cumf = np.cumsum(dens)

    df = interp1d(np.linspace(0, 1, num=n), cumf, kind="linear", fill_value=(0, 1))

    invDf = interp1d(
        df(np.linspace(0, 1, num=n)),
        np.linspace(0, 1, num=n),
        kind="linear",
        fill_value=(0, 1),
        bounds_error=False,
    )

    # knots based on periodogram peaks
    knots = invDf(np.linspace(0, 1, num=k - degree + 1))

    return knots


def _get_initial_spline_data(
        periodogram: np.ndarray, k: int, degree: int, diffMatrixOrder: int, eqSpacedKnots: bool
) -> Tuple[np.ndarray, np.ndarray, PSplines]:
    V = _generate_initial_weights(periodogram, k)
    knots = knot_locator(periodogram, k, degree, eqSpacedKnots)
    psplines = PSplines(knots=knots, degree=degree, diffMatrixOrder=diffMatrixOrder)
    return V, knots, psplines


def _generate_initial_weights(periodogram: np.ndarray, k: int) -> np.ndarray:
    scaled_periodogram = periodogram / np.sum(periodogram)
    idx = np.linspace(0, len(scaled_periodogram) - 1, k)
    idx = np.round(idx).astype(int)
    w = scaled_periodogram[idx]

    assert len(w) == k
    w[w == 0] = 1e-50  # prevents log(0) errors
    w = w / np.sum(w)
    w0 = w[:-1]
    v = np.log(w0 / (1 - np.sum(w0)))
    # convert nans to very small
    v[np.isnan(v)] = -1e50
    v = v.reshape(-1, 1)
    assert v.shape == (k - 1, 1)
    return v
