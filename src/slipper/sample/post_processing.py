import numpy as np
from scipy.stats import median_abs_deviation
from tqdm.auto import trange

from slipper.logger import logger

from ..splines import build_spline_model


def generate_spline_posterior(
    spline_len: int,
    db_list: np.ndarray,
    tau_samples: np.ndarray,
    weight_samples: np.ndarray,
    verbose: bool = False,
    logged: bool = False,
):
    n = len(tau_samples)
    splines = np.zeros((n, spline_len))
    kwargs = dict(db_list=db_list, n=spline_len)
    weight_key = "weights" if logged else "v"
    for i in trange(
        n, desc="Generating Spline posterior", disable=not verbose
    ):
        kwargs[weight_key] = weight_samples[i, :]
        splines[i, :] = build_spline_model(**kwargs)

    if logged:
        splines = np.exp(splines)
    else:
        splines *= tau_samples[:, None]
    return splines


def generate_spline_quantiles(
    spline_len,
    db_list,
    tau_samples,
    weight_samples,
    uniform_bands=True,
    verbose: bool = False,
    logged_splines: bool = False,
):
    splines = generate_spline_posterior(
        spline_len,
        db_list,
        tau_samples,
        weight_samples,
        verbose,
        logged=logged_splines,
    )
    splines_median = np.nanquantile(splines, 0.5, axis=0)
    splines_quants = np.nanquantile(splines, [0.05, 0.95], axis=0)

    if logged_splines:
        lnsplines = np.log(splines)
    else:
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

    psd_with_unc = np.vstack([splines_median, splines_quants])
    if uniform_bands:
        psd_with_unif_unc = np.vstack([splines_median, uniform_psd_quants])
        if not np.all(np.isfinite(psd_with_unif_unc)):
            logger.warning(
                "Uniform bands PSD has non-finite values, using quantiles instead."
            )
        else:
            psd_with_unc = psd_with_unif_unc

    assert psd_with_unc.shape == (3, spline_len)

    # assert no nans in psd_with_unc
    if not np.all(np.isfinite(psd_with_unc)):
        num_nans_each_row = np.sum(~np.isfinite(psd_with_unc), axis=1)
        raise ValueError(
            f"PSD quantiles has non-finite values ({num_nans_each_row})."
        )
    return psd_with_unc


def __uniformmax(sample):
    mad = median_abs_deviation(sample, nan_policy="omit", axis=0)
    mad[mad == 0] = 1e-10  # replace 0 with very small number
    return np.max(np.abs(sample - np.median(sample, axis=0)) / mad, axis=0)


def __logfuller(x, xi=0.001):
    return np.log(x + xi) - xi / (x + xi)
