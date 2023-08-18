"""Pytest setup"""
import os.path
from pathlib import Path

import numpy as np
import pytest

DIR = Path(__file__).parent
CLEAN = True


def pytest_configure(config):
    # NB this causes `pspline_psd/__init__.py` to run
    import pspline_psd  # noqa


def mkdir(path):
    path = str(path)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


@pytest.fixture
def tmpdir() -> str:
    return mkdir(DIR / "test_output")


@pytest.fixture
def test_pdgrm(tmpdir) -> str:
    fname = f"{tmpdir}/ar_3.csv"
    regenerate = True if not os.path.exists(fname) else CLEAN
    if regenerate:
        from pspline_psd.example_datasets.ar_data import generate_ar_timeseries
        from pspline_psd.fourier_methods import get_periodogram

        data = generate_ar_timeseries(order=3, n_samples=500)
        pdgm = get_periodogram(timeseries=data)
        np.savetxt(fname, pdgm)
    else:
        pdgm = np.loadtxt(fname)
    return pdgm
