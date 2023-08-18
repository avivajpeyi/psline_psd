import os

import numpy as np

from pspline_psd.plotting.plot_spline_model_and_data import plot_spline_model_and_data
from pspline_psd.sample.spline_model_sampler import sample_with_spline_model


def test_simple_example(test_pdgrm: np.ndarray, tmpdir: str):
    np.random.seed(0)
    fn = f"{tmpdir}/sample_metadata.png"
    sample_with_spline_model(
        data=test_pdgrm,
        Ntotal=3000,
        burnin=1000,
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

    mcmc = sample_with_spline_model(
        data=y,
        Ntotal=200,
        burnin=100,
        degree=3,
        eqSpacedKnots=False,
        compute_psds=True,
        metadata_plotfn="test.png",
        k=30,
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
