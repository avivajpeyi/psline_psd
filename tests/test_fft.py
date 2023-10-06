import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.signal import periodogram, welch

from slipper.example_datasets.ar_data import generate_ar_timeseries
from slipper.fourier_methods import get_fz, get_periodogram


def test_fft(tmpdir):
    """
    Test that the FFT function works
    """
    test_pdgrm = generate_ar_timeseries(order=3, n_samples=5000)

    py_fz = get_fz(test_pdgrm)
    py_pdgm = get_periodogram(py_fz)
    f, scipy_per = periodogram(test_pdgrm, fs=1)
    psd = welch(test_pdgrm, fs=1, nperseg=50)[1]
    plt.plot(
        np.linspace(0, 1, len(py_fz)),
        py_fz,
        label="Our Fz",
        alpha=0.5,
        marker=",",
    )
    plt.plot(
        np.linspace(0, 1, len(py_pdgm)),
        py_pdgm,
        label="Our Periodogram",
        alpha=0.5,
        marker=",",
    )
    plt.plot(
        np.linspace(0, 1, len(scipy_per)),
        scipy_per,
        label="scipy Periodogram",
        alpha=0.5,
        marker=",",
    )
    plt.plot(np.linspace(0, 1, len(psd)), psd, label="scipy PSD", alpha=0.5)
    plt.legend()
    plt.savefig(f"{tmpdir}/test_fft.png")
