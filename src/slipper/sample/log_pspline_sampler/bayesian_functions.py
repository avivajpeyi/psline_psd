import numpy as np
from bilby.core.prior import ConditionalGamma, ConditionalPriorDict, Gamma
from scipy.stats import gamma, norm


def _wPw(w, P):
    return np.dot(np.dot(w.T, P), w)


def lprior(k, w, τ, φ, φα, φβ, δ, δα, δβ, P):
    wTPw = _wPw(w, P)
    log_prior = k * 0.5 * np.log(φ) - 0.5 * φ * wTPw
    log_prior += gamma.logpdf(φ, a=φα, scale=1 / (δ * φβ))
    log_prior += gamma.logpdf(δ, a=δα, scale=1 / δβ)
    log_prior += norm.logpdf(τ, 0, 100)
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


def sample_φδ(k, w, τ, τα, τβ, φ, φα, φβ, δ, δα, δβ, data, spline_model):
    φ = φ_prior(k, w, spline_model.penalty_matrix, φα, φβ, δ).sample().flat[0]
    δ = δ_prior(φ, φα, φβ, δα, δβ).sample().flat[0]
    return φ, δ


def llike(w, τ, data, spline_model):
    """Whittle log likelihood"""
    # TODO: Move to using bilby likelihood
    # TODO: the parameters to this function should
    #  be the sampling parameters, not the matrix itself!
    # todo: V should be computed in here

    n = len(data)
    _lnspline = spline_model(weights=w, n=n) + τ
    # _lnspline = spline_model(weights=w, n=n)

    is_even = n % 2 == 0
    if is_even:
        _lnspline = _lnspline[1:]
        data = data[1:]
    else:
        _lnspline = _lnspline[1:-1]
        data = data[1:-1]
    _spline = np.exp(_lnspline)

    integrand = _lnspline + np.exp(np.log(data) - _lnspline - np.log(2 * np.pi))
    lnlike = -np.sum(integrand) / 2
    if not np.isfinite(lnlike):
        raise ValueError(f"lnlike is not finite: {lnlike}")
    return lnlike


def lpost(k, w, τ, τα, τβ, φ, φα, φβ, δ, δα, δβ, data, psline_model):
    logprior = lprior(k, w, τ, φ, φα, φβ, δ, δα, δβ, psline_model.penalty_matrix)
    loglike = llike(w, τ, data, psline_model)
    logpost = logprior + loglike
    if not np.isfinite(logpost):
        raise ValueError(
            f"logpost is not finite: lnpri{logprior}, lnlike{loglike}, lnpost{logpost}"
        )
    return logpost


#
# #' log-prior
# #' @keywords internal
# lprior = function (k, v, tau, tau.alpha, tau.beta, phi, phi.alpha, phi.beta,
#                    delta, delta.alpha, delta.beta, P)
# {
#   # Sigma^(-1) = P
#
#   logprior <- k * log(phi)/2 - phi * t(v) %*% P %*% v / 2 + #MNormal on weights
#
#     dgamma(phi, phi.alpha, delta * phi.beta, log = TRUE) + # log prior for phi
#
#     dgamma(delta, delta.alpha, delta.beta, log = TRUE) + # log prior for delta
#
#     dnorm(tau, 0, 100, log = TRUE); # prior for tau
#
#   return(logprior)
# }
#
# #' log Whittle likelihood
# #' @importFrom Rcpp evalCpp
# #' @useDynLib psplinePsd, .registration = TRUE
# #' @keywords internal
# llike = function (omega, FZ, k, v, tau, tau.alpha, pdgrm, degree, db.list,
#                   spec_ar)
# {
#   n <- length(FZ);
#
#   # Which boundary frequencies to remove from likelihood computation
#   if (n %% 2) {  # Odd length time series
#     bFreq <- 1  # Remove first
#   }
#   else {  # Even length time series
#     bFreq <- c(1, n)  # Remove first and last
#   }
#
#   # Un-normalised PSD (defined on [0, 1])
#   qq.psd <- qpsd(omega, k, v, degree, db.list);
#
#   q = unrollPsd(qq.psd, n); # Unrolls the unnormalised PSD to length n
#
#   # Normalised PSD (defined on [0, pi])
#   f <- q + tau; # CHANGED --> this is actually log-tau, and log-f
#   f <- f + spec_ar; # Correction spec_ar is the parametric
#
#   # Whittle log-likelihood
#   #llike <- -sum(log(f[-bFreq]) + pdgrm[-bFreq] / (f[-bFreq] * 2 * pi)) / 2; # original
#   #llike <- -sum(f[-bFreq] + pdgrm[-bFreq] / (exp(f[-bFreq]) * 2 * pi)) / 2;
#   llike <- -sum(f[-bFreq] +
#                 exp(log(pdgrm[-bFreq]/(2 * pi)) - f[-bFreq]) ) / 2;
#
#   return(llike)
#
# }
#
# #' Unnormalised log posterior
# #' @keywords internal
# lpost = function (omega, FZ, k, v, tau, tau.alpha, tau.beta,
#                   phi, phi.alpha, phi.beta, delta, delta.alpha, delta.beta,
#                   P, pdgrm, degree, db.list,
#                   spec_ar)
# {
#
#   ll  <- llike(omega, FZ, k, v, tau, tau.alpha, pdgrm, degree, db.list, spec_ar)
#
#   lpr <- lprior(k, v, tau, tau.alpha, tau.beta, phi,
#                 phi.alpha, phi.beta, delta, delta.alpha, delta.beta, P)
#
#   #print(ll)
#   #print(lpr)
#
#   lp <- ll + lpr;
#
#   #return(lp)
#   return(list(ll = ll, lp = lp));
# }
