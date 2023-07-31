import numpy as np
from scipy.fft import fft


def get_fz(x: np.ndarray) -> np.ndarray:
    """
    Function computes FZ (i.e. fast Fourier transformed data)
    Outputs coefficients in correct order and rescaled
    NOTE: x must be mean-centered ( x - np.mean(x) )

    Converted from R code here:
    https://github.com/pmat747/psplinePsd/blob/master/R/internal_gibbs_util.R#L5

    NOTE: This is _not_ the normal FFT function
    See paper Eq XX
    paper:

    # FIXME: the last element has an error

    """

    assert np.allclose(np.mean(x), 0), f"x must be mean-centered (mu(x)={np.mean(x)})"

    n = len(x)
    sqrt2 = np.sqrt(2)
    sqrtn = np.sqrt(n)

    # Cyclically shift so last observation becomes first
    x = np.concatenate(([x[n - 1]], x[:-1]))

    fourier = fft(x)

    FZ = np.empty(n)
    FZ[0] = np.real(fourier[0])  # first coefficient is real

    is_even = n % 2 == 0

    if is_even:
        N = (n - 1) // 2
        FZ[1 : 2 * N + 1 : 2] = sqrt2 * np.real(fourier[1 : N + 1])
        FZ[2 : 2 * N + 2 : 2] = sqrt2 * np.imag(fourier[1 : N + 1])
    else:
        FZ[n - 1] = np.real(fourier[n // 2])
        FZ[1 : n // 2] = sqrt2 * np.real(fourier[1 : n // 2])
        FZ[2 : n // 2 + 1] = sqrt2 * np.imag(fourier[1 : n // 2])

    FZ[-1] = FZ[-2]

    return FZ / sqrtn


def get_periodogram(fz: np.ndarray):
    """
    Function computes the periodogram of fz
    (Assumes fz is already rescaled)
    """
    return np.power(np.abs(fz), 2)
