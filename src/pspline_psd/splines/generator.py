import numpy as np
from .utils import unroll_list_to_new_length, density_mixture


def build_spline_model(v: np.ndarray, db_list: np.ndarray, n: int):
    """Build a spline model from a vector of spline coefficients and a list of B-spline basis functions"""
    unorm_spline = get_unscaled_spline(v, db_list)
    return unroll_list_to_new_length(unorm_spline, n)


def convert_v_to_weights(v: np.ndarray):
    """Convert vector of spline coefficients to weights"""
    v = np.array(v)
    expV = np.exp(v)

    # converting to weights
    # Eq near 4, page 3.1
    if np.any(np.isinf(expV)):
        ls = np.logaddexp(0, v)
        weight = np.exp(v - ls)
    else:
        ls = 1 + np.sum(expV)
        weight = expV / ls

    s = 1 - np.sum(weight)
    # adding last element to weight
    weight = np.append(weight, 0 if s < 0 else s).ravel()
    return weight


def get_unscaled_spline(v: np.ndarray, db_list: np.ndarray):
    """Compute unscaled spline using mixture of B-splines with weights from v

    Parameters
    ----------
    v : np.ndarray
        Vector of spline coefficients (length k)

    db_list : np.ndarray
        Matrix of B-spline basis functions (k x n)

    Returns
    -------
    psd : np.ndarray

    """
    weights = convert_v_to_weights(v)
    psd = density_mixture(densities=db_list.T, weights=weights)
    epsilon = 1e-20
    # element wise maximum value bewteen psd and epsilon
    psd = np.maximum(psd, epsilon)
    return psd
