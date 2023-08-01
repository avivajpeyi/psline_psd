import numpy as np
from scipy.interpolate import interp1d


def density_mixture(weights: np.ndarray, densities: np.ndarray, epsilon=1e-20) -> np.ndarray:
    """build a density mixture, given mixture weights and densities

    weights:
        mixture weights (k x 1)
    densities:
        densities (k x n)
    """
    assert (
        len(weights) == densities.shape[0],
        f"weights ({weights.shape}) and densities ({densities.shape}) must have the same length",
    )
    res = np.sum(weights[:, None] * densities, axis=0)
    res = np.maximum(res, epsilon) # dont allow values below epsilon
    return res


def unroll_list_to_new_length(qPsd, n):
    """unroll PSD from qPsd to psd of length n"""
    # q = np.zeros(n)
    # q[0] = qPsd[0]
    # N = (n - 1) // 2
    # assert len(qPsd) >= N + 1, f"qPsd ({len(qPsd)}) must have length >= {N + 1}"
    # for i in range(1, N + 1):
    #     j = 2 * i - 1
    #     q[j] = qPsd[i]
    #     q[j + 1] = qPsd[i]
    #
    # q[-1] = qPsd[-1]
    # TODO: is this necessary?

    newx = np.linspace(0, 1, n)
    oldx = np.linspace(0, 1, len(qPsd))
    f = interp1d(oldx, qPsd, kind="nearest")
    q = f(newx)
    assert np.all(q >= 0), f"q must be positive, but got {q}"
    return q
