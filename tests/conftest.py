"""Pytest setup"""
import os.path
from pathlib import Path

import numpy as np
import pytest

DIR = Path(__file__).parent
CLEAN = False

TEST_AR_ORDER = 5
TEST_AR_N = 500


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
    fname = f"{tmpdir}/ar_periodogram_{TEST_AR_ORDER}.csv"
    regenerate = True if not os.path.exists(fname) else CLEAN
    if regenerate:
        from slipper.example_datasets.ar_data import get_ar_periodogram

        pdgm = get_ar_periodogram(order=TEST_AR_ORDER, n_samples=TEST_AR_N)
        np.savetxt(fname, pdgm)
    else:
        pdgm = np.loadtxt(fname)
    return pdgm


@pytest.fixture
def true_psd(tmpdir) -> str:
    fname = f"{tmpdir}/ar_true_psd_{TEST_AR_ORDER}.csv"
    regenerate = True if not os.path.exists(fname) else CLEAN
    if regenerate:
        from slipper.example_datasets.ar_data import get_true_ar_psd

        _, psd = get_true_ar_psd(order=TEST_AR_ORDER, n_samples=TEST_AR_N)
        np.savetxt(fname, psd)
    else:
        psd = np.loadtxt(fname)
    return psd
