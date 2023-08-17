import numpy as np

from pspline_psd.utils import get_fz
from scipy.signal import periodogram, welch
from pspline_psd.example_datasets.ar_data import generate_ar_timeseries

import pytest
import matplotlib.pyplot as plt

import numpy as np

from pspline_psd.utils import get_fz, get_periodogram


def test_fft(tmpdir):
    """
    Test that the FFT function works
    """
    test_timeseries = generate_ar_timeseries(order=3, n_samples=5000)

    py_fz = get_fz(test_timeseries)
    py_pdgm = get_periodogram(py_fz)
    f, scipy_per = periodogram(test_timeseries, fs=1)
    psd = welch(test_timeseries, fs=1, nperseg=50)[1]
    plt.plot(np.linspace(0,1,len(py_fz)), py_fz, label="Our Fz", alpha=0.5, marker=",")
    plt.plot(np.linspace(0,1,len(py_pdgm)), py_pdgm, label="Our Periodogram", alpha=0.5, marker=",")
    plt.plot(np.linspace(0,1,len(scipy_per)),scipy_per, label="scipy Periodogram", alpha=0.5, marker=",")
    plt.plot(np.linspace(0, 1, len(psd)), psd, label="scipy PSD", alpha=0.5)
    plt.legend()
    plt.savefig(f"{tmpdir}/test_fft.png")
