import numpy as np
from scipy.interpolate import interp1d


def density_mixture(
    weights: np.ndarray, densities: np.ndarray, epsilon=1e-20
) -> np.ndarray:
    """build a density mixture, given mixture weights and densities

    weights:
        mixture weights (k x 1)
    densities:
        densities (k x n)
    """
    if len(weights) != densities.shape[0]:
        raise ValueError(
            f"weights ({weights.shape}) and densities ({densities.shape}) must have the same length",
        )
    res = np.sum(weights[:, None] * densities, axis=0)
    # res = np.maximum(res, epsilon)  # dont allow values below epsilon

    return res


def unroll_list_to_new_length(old_list, n):
    """unroll PSD from qPsd to psd of length n"""
    newx = np.linspace(0, 1, n)
    oldx = np.linspace(0, 1, len(old_list))
    f = interp1d(oldx, old_list, kind="cubic")
    q = f(newx)
    # assert np.all(q >= 0), f"q must be positive, but got {q}"
    return q


def build_spline_model(
    db_list: np.ndarray, n: int, v: np.ndarray = None, weights=None
):
    """Build a spline model from a vector of spline coefficients and a list of B-spline basis functions"""
    unorm_spline = __get_unscaled_spline(db_list, v=v, weights=weights)
    return unroll_list_to_new_length(unorm_spline, n)


def convert_v_to_weights(v: np.ndarray):
    """Convert vector of spline coefficients to weights

    Parameters
    ----------
    v : np.ndarray
        Vector of spline coefficients (length n_basis-1)

    Returns
    -------
    weight : np.ndarray
        Vector of weights (length n_basis)
    """
    v = np.array(v)
    expV = np.exp(v)

    """
     expV   <- exp(v)
      weight <- expV / (1+sum(expV));
      weight <- c(weight, 1-sum(weight));

      psd <- densityMixture(weight, db.list)
      epsilon <- 1e-20
      psd <- pmax(psd, epsilon)
      #psd <- psd[-c(1, length(psd))]
      return(psd)

    """

    # converting to weights
    # Eq near 4, page 3.1
    # if np.any(np.isinf(expV)):
    #     ls = np.logaddexp(0, v)
    #     weight = np.exp(v - ls)
    # else:
    ls = 1 + np.sum(expV)
    weight = expV / ls

    s = 1 - np.sum(weight)
    # adding last element to weight
    weight = np.append(weight, 0 if s < 0 else s).ravel()

    if len(weight) != len(v) + 1:
        raise ValueError(
            "Length of weight vector is not equal to length of v + 1"
        )

    return weight


def __get_unscaled_spline(
    db_list: np.ndarray, epsilon=1e-20, v: np.ndarray = None, weights=None
):
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
    if weights is None:
        weights = convert_v_to_weights(v)
    combined = density_mixture(densities=db_list.T, weights=weights)
    # return np.maximum(combined, epsilon)
    return combined


def _lnlikelihood(data: np.ndarray, model: np.ndarray, **lnl_kwargs) -> float:
    """Whittle log likelihood"""

    # replace any zeros to near-zero values to avoid log(0) = -inf
    data[data == 0] = 1e-50
    model[model == 0] = 1e-50

    lndata = np.log(data)
    lnmodel = np.log(model)

    n = len(data)
    is_even = n % 2 == 0
    if is_even:  # remove first elememt
        lnmodel = lnmodel[1:]
        lndata = lndata[1:]
    else:  # remove last element
        lnmodel = model[1:-1]
        lndata = data[1:-1]

    integrand = lnmodel + np.exp(lndata - lnmodel - np.log(2 * np.pi))
    lnlike = -np.sum(integrand) / 2

    if not np.isfinite(lnlike):
        return np.nan
    return lnlike


def _mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)
