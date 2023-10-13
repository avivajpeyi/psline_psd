import numpy as np

from slipper.example_datasets.lisa_data import lisa_noise_periodogram
from slipper.plotting.plot_spline_model_and_data import (
    plot_spline_model_and_data,
)
from slipper.splines.knot_locator import KnotLocatorType
from slipper.splines.p_splines import PSplines


def test_lisa_noise_lnl():
    pdgrm = lisa_noise_periodogram()
    # keep every 5th point to speed up analysis for testing
    pdgrm = pdgrm[::5]

    pspline = PSplines.from_kwarg_dict(
        dict(
            degree=3,
            diffMatrixOrder=2,
            logged=True,
            n_grid_points=500,
            knot_locator_type=KnotLocatorType.data_peak,
            data=pdgrm,
            n_knots=30,
        )
    )
    assert np.isfinite(
        pspline.lnlikelihood(pdgrm, pspline.guess_weights(pdgrm))
    )
