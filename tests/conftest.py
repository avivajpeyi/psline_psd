"""Pytest setup"""
import os.path
from pathlib import Path

import numpy as np
import pytest


DIR = Path(__file__).parent


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
def test_timeseries(tmpdir) -> str:
    datafn = f"{tmpdir}/ar_3.csv"
    if not os.path.exists(datafn):
        from pspline_psd.example_datasets.ar_data import generate_ar_timeseries
        data = generate_ar_timeseries(order=3, n_samples=500)
        np.savetxt(datafn, data)
    else:
        data = np.loadtxt(datafn)
    return data
