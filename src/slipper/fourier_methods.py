import numpy as np
from scipy.fft import fft



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
