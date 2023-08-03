import numpy as np
from scipy.interpolate import interp1d


def density_mixture(weights: np.ndarray, densities: np.ndarray, epsilon=1e-20) -> np.ndarray:
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
    res = np.maximum(res, epsilon)  # dont allow values below epsilon

    return res


def unroll_list_to_new_length(old_list, n):
    """unroll PSD from qPsd to psd of length n"""
    newx = np.linspace(0, 1, n)
    oldx = np.linspace(0, 1, len(old_list))
    f = interp1d(oldx, old_list, kind="nearest")
    q = f(newx)
    assert np.all(q >= 0), f"q must be positive, but got {q}"
    return q
