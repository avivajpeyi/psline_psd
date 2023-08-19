import os

import numpy as np

from slipper.plotting.plot_spline_model_and_data import plot_spline_model_and_data
from slipper.sample.spline_model_sampler import fit_data_with_pspline_model

NTOTAL = 250


def test_simple_example(test_pdgrm: np.ndarray, tmpdir: str):
    np.random.seed(0)
    fn = f"{tmpdir}/sample_metadata.png"
    fit_data_with_pspline_model(
        data=test_pdgrm,
        Ntotal=NTOTAL,
        degree=3,
        eqSpacedKnots=False,
        compute_psds=True,
        metadata_plotfn=fn,
        k=10,
    )
    assert os.path.exists(fn)


def func(x):
    return 1 / (x**2 + 1) * np.cos(np.pi * x)


def test_funct():
    n_obs = 600
    np.random.seed(0)

    # make example data
    x = np.linspace(-3, 3, n_obs)
    y = func(x) + np.random.normal(0, 0.2, len(x))
    # move all values to be positive
    y = y / np.std(y)
    scaling = np.abs(np.min(y))
    y = y + scaling

    mcmc = fit_data_with_pspline_model(
        data=y,
        Ntotal=NTOTAL,
        degree=3,
        eqSpacedKnots=False,
        compute_psds=True,
        metadata_plotfn="test.png",
        k=5,
    )
    fig = plot_spline_model_and_data(
        data=y, model_quants=mcmc.psd_quantiles, knots=mcmc.knots, separarte_y_axis=True
    )
    ax = fig.axes[0]
    true_y = func(x)
    true_y = true_y / np.std(true_y)
    true_y = true_y + scaling
    ax.plot(np.linspace(0, 1, len(true_y)), true_y, color="k", alpha=0.4)
    fig.show()
