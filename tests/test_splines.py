import matplotlib.pyplot as plt
import numpy as np

from pspline_psd.sample.spline_model_sampler import _get_initial_values
from pspline_psd.splines.initialisation import knot_locator
from pspline_psd.splines.p_splines import PSplines


def test_spline_creation(tmpdir):
    """Test that the splines can be generated"""
    degree = 2
    knots = np.array([0, 1, 2, 3, 4, 5, 6])
    pspline = PSplines(knots=knots, degree=degree)
    assert pspline is not None
    fig, ax = pspline.plot(weights=np.random.randn(pspline.n_basis))
    fig.savefig(f"{tmpdir}/test_spline_creation.png")
    plt.close()
