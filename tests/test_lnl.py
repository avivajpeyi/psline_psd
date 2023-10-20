import matplotlib.pyplot as plt
import numpy as np

from slipper.example_datasets.lisa_data import lisa_noise_periodogram
from slipper.plotting.plot_spline_model_and_data import (
    plot_spline_model_and_data,
)
from slipper.splines.knot_locator import KnotLocatorType
from slipper.splines.p_splines import PSplines


def test_lisa_noise_lnl():
    pdgrm = lisa_noise_periodogram()
    pdgrm = pdgrm[::5]

    pspline = PSplines.from_kwarg_dict(
        dict(
            degree=3,
            diffMatrixOrder=2,
            logged=True,
            n_grid_points=5000,
            knot_locator_type=KnotLocatorType.binned_knots,
            data_bin_edges=[10**-3, 10**-2.5, 10**-2, 0.1, 0.5],
            data_bin_weights=[0.1, 0.3, 0.4, 0.2, 0.2, 0.1],
            log_data=True,
            data=pdgrm,
            n_knots=30,
        )
    )
    w = pspline.guess_weights(pdgrm)
    w = np.zeros_like(w)
    w[4] = -1

    model = pspline(weights=w, return_log_value=True)
    model_x = np.linspace(0, 1, len(model))
    fig = plt.scatter(model_x, model, s=1)
    # x = np.linspace(0, 1, len(pdgrm))
    # plt.scatter(x, pdgrm, s=1)
    #
    plt.show()

    assert np.isfinite(pspline.lnlikelihood(pdgrm, weights=w))
