import numpy as np
from scipy.stats import median_abs_deviation
from tqdm.auto import trange

from ..splines import build_spline_model


def generate_spline_posterior(
    spline_len,
    db_list,
    tau_samples,
    v_samples,
):
    n = len(tau_samples)
    splines = np.zeros((n, spline_len))
    kwargs = dict(db_list=db_list, n=spline_len)
    for i in trange(n, desc="Generating Spline posterior"):
        splines[i, :] = build_spline_model(v=v_samples[i, :], **kwargs) * tau_samples[i]
    return splines


def generate_spline_quantiles(
    spline_len, db_list, tau_samples, v_samples, uniform_bands=True
):
    splines = generate_spline_posterior(spline_len, db_list, tau_samples, v_samples)
    splines_median = np.quantile(splines, 0.5, axis=0)
    splines_quants = np.quantile(splines, [0.05, 0.95], axis=0)

    lnsplines = logfuller(splines)
    lnsplines_median = np.median(lnsplines, axis=0)
    lnsplines_mad = median_abs_deviation(lnsplines, axis=0)
    lnsplines_uniform_max = uniformmax(lnsplines)
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


def uniformmax(sample):
    mad = median_abs_deviation(sample, nan_policy="omit", axis=0)
    # replace 0 with very small number
    mad[mad == 0] = 1e-10
    return np.max(np.abs(sample - np.median(sample, axis=0)) / mad, axis=0)


def logfuller(x, xi=0.001):
    return np.log(x + xi) - xi / (x + xi)
