import os

import matplotlib.pyplot as plt

from .plot_psd import plot_psd
from .plot_sampling_metadata import plot_metadata

DIR = os.path.dirname(os.path.abspath(__file__))

plt.style.use(f"{DIR}/style.mplstyle")
