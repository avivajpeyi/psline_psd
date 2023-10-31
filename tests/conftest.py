"""Pytest setup"""
import os.path
from pathlib import Path

import numpy as np
import pytest

DIR = Path(__file__).parent
CLEAN = False


def pytest_configure(config):
    # NB this causes `slipper/__init__.py` to run
    import slipper  # noqa


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
    fname = f"{tmpdir}/ar_periodogram_4.csv"
    regenerate = True if not os.path.exists(fname) else CLEAN
    if regenerate:
        from slipper.example_datasets.ar_data import get_ar_periodogram

        pdgm = get_ar_periodogram(order=4, n_samples=5000)
        np.savetxt(fname, pdgm)
    else:
        pdgm = np.loadtxt(fname)
    return pdgm
