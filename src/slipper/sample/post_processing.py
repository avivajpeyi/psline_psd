import numpy as np
from scipy.stats import median_abs_deviation
from tqdm.auto import trange

from ..splines import build_spline_model


def generate_spline_posterior(
    spline_len,
    db_list,
    tau_samples,
    weight_samples,
    verbose: bool = False,
):
    n = len(tau_samples)
    splines = np.zeros((n, spline_len))
    kwargs = dict(db_list=db_list, n=spline_len)
    n_basis = db_list.shape[1]
    n_weight_cols = weight_samples.shape[1]
    weight_key = "weights"
    if n_basis-1 == n_weight_cols:
        weight_key = "v"
    for i in trange(n, desc="Generating Spline posterior", disable=not verbose):
        kwargs[weight_key] = weight_samples[i, :]
        splines[i, :] = build_spline_model(**kwargs) * tau_samples[i]
    return splines


def generate_spline_quantiles(
    spline_len,
    db_list,
    tau_samples,
    weight_samples,
    uniform_bands=True,
    verbose: bool = False,
):
    splines = generate_spline_posterior(
        spline_len, db_list, tau_samples, weight_samples, verbose
    )
    splines_median = np.quantile(splines, 0.5, axis=0)
    splines_quants = np.quantile(splines, [0.05, 0.95], axis=0)

    # TBH I don't understand this part -- taken from @patricio's code
    # See internal_gibs_utils and line 395 of gibs-sample-simple
    lnsplines = __logfuller(splines)
    lnsplines_median = np.median(lnsplines, axis=0)
    lnsplines_mad = median_abs_deviation(lnsplines, axis=0)
    lnsplines_uniform_max = __uniformmax(lnsplines)
    lnsplines_c_value = np.quantile(lnsplines_uniform_max, 0.9) * lnsplines_mad

    uniform_psd_quants = np.array(
        [
            np.exp(lnsplines_median - lnsplines_c_value),
            np.exp(lnsplines_median + lnsplines_c_value),
        ]
    )

    if uniform_bands:
        psd_with_unc = np.vstack([splines_median, uniform_psd_quants])
    else:
        psd_with_unc = np.vstack([splines_median, splines_quants])

    assert psd_with_unc.shape == (3, spline_len)
    assert np.all(psd_with_unc > 0)
    return psd_with_unc


def __uniformmax(sample):
    mad = median_abs_deviation(sample, nan_policy="omit", axis=0)
    mad[mad == 0] = 1e-10  # replace 0 with very small number
    return np.max(np.abs(sample - np.median(sample, axis=0)) / mad, axis=0)


def __logfuller(x, xi=0.001):
    return np.log(x + xi) - xi / (x + xi)
