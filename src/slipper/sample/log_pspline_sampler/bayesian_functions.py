import numpy as np
from bilby.core.prior import ConditionalGamma, ConditionalPriorDict, Gamma
from scipy.stats import gamma, norm


def _wPw(w, P):
    return np.dot(np.dot(w.T, P), w)


def lprior(k, w, φ, φα, φβ, δ, δα, δβ, P):
    wTPw = _wPw(w, P)
    log_prior = k * 0.5 * np.log(φ) - 0.5 * φ * wTPw
    log_prior += gamma.logpdf(φ, a=φα, scale=1 / (δ * φβ))
    log_prior += gamma.logpdf(δ, a=δα, scale=1 / δβ)
    return log_prior


def φ_prior(k, w, P, φα, φβ, δ):
    wTPw = _wPw(w, P)
    shape = 0.5 * k + φα
    rate = φβ * δ + 0.5 * wTPw
    return Gamma(k=shape, theta=1 / rate)


def δ_prior(φ, φα, φβ, δα, δβ):
    """Gamma prior for pi(δ|φ)"""
    shape = φα + δα
    rate = φβ * φ + δβ
    return Gamma(k=shape, theta=1 / rate)


def sample_φδ(k, w, φα, φβ, δ, δα, δβ, data, spline_model):
    φ = φ_prior(k, w, spline_model.penalty_matrix, φα, φβ, δ).sample().flat[0]
    δ = δ_prior(φ, φα, φβ, δα, δβ).sample().flat[0]
    return φ, δ


def llike(w, data, spline_model):
    """Whittle log likelihood"""

    n = len(data)
    _lnspline = spline_model(weights=w, n=n)

    is_even = n % 2 == 0
    if is_even:
        _lnspline = _lnspline[1:]
        data = data[1:]
    else:
        _lnspline = _lnspline[1:-1]
        data = data[1:-1]
    _spline = np.exp(_lnspline)

    integrand = _lnspline + np.exp(
        np.log(data) - _lnspline - np.log(2 * np.pi)
    )
    lnlike = -np.sum(integrand) / 2
    if not np.isfinite(lnlike):
        raise ValueError(f"lnlike is not finite: {lnlike}")
    return lnlike


def lpost(k, w, φ, φα, φβ, δ, δα, δβ, data, psline_model):
    logprior = lprior(k, w, φ, φα, φβ, δ, δα, δβ, psline_model.penalty_matrix)
    loglike = llike(w, data, psline_model)
    logpost = logprior + loglike
    if not np.isfinite(logpost):
        raise ValueError(
            f"logpost is not finite: lnpri{logprior}, lnlike{loglike}, lnpost{logpost}"
        )
    return logpost
