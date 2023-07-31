"""Top-level package for P-spline PSD."""

__author__ = """Avi Vajpeyi"""
__email__ = 'avi.vajpeyi@gmail.com'
__version__ = '0.0.1'

import pathlib
import sys
import matplotlib.pyplot as plt

plt.style.use('plotting/style.mplstyle')

sys.path.append(str(pathlib.Path(__file__).parent))
