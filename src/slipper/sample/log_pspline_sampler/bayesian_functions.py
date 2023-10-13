from collections import namedtuple
from typing import NamedTuple

import numpy as np
from bilby.core.prior import Gamma
from scipy.stats import gamma

from slipper.splines.p_splines import PSplines

from ..utils import _xPx


class LnlArgs(NamedTuple):
    w: np.ndarray
    φ: float
    φα: float
    φβ: float
    δ: float
    δα: float
    δβ: float
    data: np.ndarray
    spline_model: PSplines


def lprior(args: LnlArgs):
    φα, φβ, δα, δβ = (
        args.φα,
        args.φβ,
        args.δα,
        args.δβ,
    )
    φ, δ = args.φ, args.δ
    P = args.spline_model.penalty_matrix
    k = args.spline_model.n_basis
    w = args.w

    log_prior = k * 0.5 * np.log(φ) - 0.5 * φ * _xPx(w, P)
    log_prior += gamma.logpdf(φ, a=φα, scale=1 / (δ * φβ))
    log_prior += gamma.logpdf(δ, a=δα, scale=1 / δβ)
    return log_prior


def φ_prior(k, w, P, φα, φβ, δ):
    wTPw = _xPx(w, P)
    shape = 0.5 * k + φα
    rate = φβ * δ + 0.5 * wTPw
    return Gamma(k=shape, theta=1 / rate)


def δ_prior(φ, φα, φβ, δα, δβ):
    """Gamma prior for pi(δ|φ)"""
    shape = φα + δα
    rate = φβ * φ + δβ
    return Gamma(k=shape, theta=1 / rate)


def sample_φδ(args: LnlArgs):
    w, φα, φβ, δα, δβ, δ, spline_model = (
        args.w,
        args.φα,
        args.φβ,
        args.δα,
        args.δβ,
        args.δ,
        args.spline_model,
    )
    k = spline_model.n_basis
    φ = φ_prior(k, w, spline_model.penalty_matrix, φα, φβ, δ).sample().flat[0]
    δ = δ_prior(φ, φα, φβ, δα, δβ).sample().flat[0]
    return φ, δ


def lpost(args: LnlArgs):
    w, data, spline_model = args.w, args.data, args.spline_model
    logprior = lprior(args)
    loglike = spline_model.lnlikelihood(data=data, weights=w)
    logpost = logprior + loglike
    if not np.isfinite(logpost):
        raise ValueError(
            f"logpost is not finite:\n"
            f"lnpri: {logprior},\n"
            f"lnlike: {loglike},\n"
            f"lnpost: {logpost}"
        )
    return logpost


import matplotlib.pyplot as plt


def __plot_error_plt(data, spline, knots, integrand):
    # normalize data and spline
    data = data / np.max(data)
    spline = spline / np.max(spline)

    fig, axes = plt.subplots(3, 1, figsize=(5, 8))

    for k in knots:
        axes[0].axvline(k, color="k", linestyle="--", alpha=0.2)
        axes[1].axvline(k, color="k", linestyle="--", alpha=0.2)

    ax = axes[0]
    x_data = np.linspace(0, 1, len(data))
    x_model = np.linspace(0, 1, len(spline))
    ax.loglog(x_data, data, label="data")
    ylm = ax.get_ylim()
    ax.loglog(x_model, spline, label="spline")
    ax.set_ylim(ylm)
    ax.set_ylabel("PSD/PSDmax")
    ax.legend()

    ax = axes[1]
    ax.plot(x_data, data, label="data")
    ylm = ax.get_ylim()
    ax.plot(x_model, spline, label="spline")
    ax.set_ylim(ylm)
    ax.set_ylabel("PSD/PSDmax")
    ax.legend()

    ax = axes[2]

    integrand_x = np.linspace(0, 1, len(integrand))
    ax.semilogx(integrand_x, integrand, label="integrand")
    # scatter red lines whereever nans
    nan_xvals = integrand_x[~np.isfinite(integrand)]
    yl = ax.get_ylim()
    ax.vlines(
        nan_xvals,
        ymin=min(yl),
        ymax=max(yl),
        color="r",
        linestyle="-",
        lw=5,
        alpha=1,
    )
    ax.legend()

    ax.set_ylabel("integrand")

    plt.tight_layout()
    plt.savefig("error.png")

    return fig
