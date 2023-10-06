import os
import shutil
import urllib.request

import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))

BASE = "https://gist.githubusercontent.com/avivajpeyi/"
NOISE_PDGRM = (
    f"{BASE}"
    "f2526ee8d0ee6793858530246c76dda8/raw/"
    "f2b28eaa5ac29b47b9c03bd2ba5a46a7a65e9678/gistfile1.txt"
)
WD_TIME_SERIES = (
    f"{BASE}"
    "67c3129ac75e6e9da1f75209de7ec5fe/raw/"
    "849ee5a2bee139ac0cdcce10720630d7a668a92d/lisa_wd_strain"
)


def __download_file(url, filename):
    """Download a file from a url to a local filename."""
    with urllib.request.urlopen(url) as response:
        with open(filename, "wb") as outfile:
            shutil.copyfileobj(response, outfile)


def __load_data(url, fname):
    if not os.path.exists(fname):
        __download_file(url, fname)
    return np.loadtxt(fname)


def lisa_noise_periodogram():
    """Download the noise periodogram data from the LISA Data Challenge."""
    return __load_data(
        NOISE_PDGRM, os.path.join(DIR, "lisa_noise_periodogram.txt")
    )


def lisa_wd_strain():
    """Download the white dwarf strain data from the LISA Data Challenge."""
    return __load_data(WD_TIME_SERIES, os.path.join(DIR, "lisa_wd_strain.txt"))
