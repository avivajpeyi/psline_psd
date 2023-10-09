import enum
from typing import List

import numpy as np
from scipy.interpolate import interp1d


class KnotLocatorType(enum.Enum):
    linearly_spaced = "linearly_spaced"
    log_spaced = "log_spaced"
    data_peak = "data_peak"
    binned_knots = "binned_knots"

    def __repr__(self):
        return f"{self.name}"

    @classmethod
    def is_valid_type(cls, val):
        if isinstance(val, cls):
            return True
        elif val in cls.__members__:
            return True
        return False


class KnotLocDict(dict):
    def __setitem__(self, k, v):
        if KnotLocatorType.is_valid_type(k):
            super().__setitem__(KnotLocatorType(k), v)
        else:
            raise KeyError(f"{k} is not valid")

    def __getitem__(self, k):
        if isinstance(k, str):
            k = KnotLocatorType(k)
        return super().__getitem__(k)


def linearly_spaced_knots(n_knots: int) -> np.ndarray:
    return np.linspace(0, 1, n_knots)


def log_spaced_knots(n_knots: int, min_val: float) -> np.ndarray:
    return np.geomspace(min_val, 1, n_knots)


def data_peak_knots(data: np.ndarray, n_knots: int) -> np.ndarray:
    """Returns knots at the peaks of the data"""
    aux = np.sqrt(data)
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
    return invDf(np.linspace(0, 1, num=n_knots))


def binned_knots(
    data: np.ndarray,
    n_knots: int,
    data_bin_edges: List,
    data_bin_weights: List,
    log_data=True,
) -> np.ndarray:
    """Returns the"""

    d = data.copy()
    if log_data:
        d = np.log(data)

    N = len(d)

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

    n_knots = n_knots - 2  # to include 0 and 1 in the knot vector
    kvec = np.round(n_knots * np.array(data_bin_weights))
    kvec = kvec.astype(int)

    while np.sum(kvec) > n_knots:
        kvec[np.argmax(kvec)] = np.max(kvec) - 1

    while np.sum(kvec) < n_knots:
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

    return knots


_KNOT_LOCATOR_FUNC_DICT = KnotLocDict()
_KNOT_LOCATOR_FUNC_DICT[
    KnotLocatorType.linearly_spaced
] = linearly_spaced_knots
_KNOT_LOCATOR_FUNC_DICT[KnotLocatorType.log_spaced] = log_spaced_knots
_KNOT_LOCATOR_FUNC_DICT[KnotLocatorType.data_peak] = data_peak_knots
_KNOT_LOCATOR_FUNC_DICT[KnotLocatorType.binned_knots] = binned_knots
