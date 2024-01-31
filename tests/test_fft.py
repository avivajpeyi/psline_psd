import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.signal import periodogram, welch

from slipper.example_datasets.ar_data import generate_ar_timeseries
from slipper.fourier_methods import get_periodogram


def test_fft(tmpdir):
    """
    Test that the FFT function works
    """
    pass
