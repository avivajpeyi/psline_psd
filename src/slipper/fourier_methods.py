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
    # mean-center data
    x = x - np.mean(x)

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


####Change the periodogram thing here
def get_periodogram(
    fz: np.ndarray = None, timeseries: np.ndarray = None, fs: float = None
) -> np.ndarray:
    """
    Function computes the data of fz
    (Assumes fz is already rescaled)
    """
    if timeseries is None and fz is None:
        raise ValueError("Must provide either timeseries or fz")
    elif timeseries is not None and fz is not None:
        raise ValueError("Must provide either timeseries or fz, not both")
    elif timeseries is not None:
        # fz = get_fz(timeseries) #for duplicates
        n = len(timeseries)
        timeseries = timeseries - np.mean(timeseries)  # mean centered
        timeseries = timeseries / np.std(
            timeseries
        )  # Optimal rescaling to prevent numerical issues. The data has SD 1
        fz = fft(timeseries)
    pdgrm = np.power(np.abs(fz), 2) / n
    if fs is not None:
        pdgrm = (
            pdgrm * 2 / (fs)
        )  # multiplication by 2/fs includes the sampling frequency
    pdgrm = pdgrm[: int(n / 2 + 1)]
    return pdgrm
