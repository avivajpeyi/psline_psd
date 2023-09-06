import numpy as np
from bilby.core.prior import ConditionalPriorDict, Gamma


def _wPw(w, P):
    return np.dot(np.dot(w.T, P), w)


def lprior(k, w, τ, τα, τβ, φ, φα, φβ, δ, δα, δβ, P):
    # TODO: Move to using bilby priors

    wTPw = _wPw(w, P)
    logφ = np.log(φ)
    logδ = np.log(δ)
    logτ = np.log(τ)

    lnpri_weights = k * logφ * 0.5 - φ * wTPw * 0.5
    lnpri_φ = φα * logδ + (φα - 1) * logφ - φβ * δ * φ
    lnpri_δ = (δα - 1) * logδ - δβ * δ
    lnpri_τ = -(τα + 1) * logτ - τβ / τ
    log_prior = lnpri_weights + lnpri_φ + lnpri_δ + lnpri_τ
    return log_prior


def φ_prior(k, w, P, φα, φβ, δ):
    wTPw = _wPw(w, P)
    shape = k/2 + φα
    rate = φβ * δ + wTPw / 2
    return Gamma(k=shape, theta=1 / rate)


def δ_prior(φ, φα, φβ, δα, δβ):
    """Gamma prior for pi(δ|φ)"""
    shape = φα + δα
    rate = φβ * φ + δβ
    return Gamma(k=shape, theta=1 / rate)


def inv_τ_prior(w, data, spline_model, τα, τβ):
    """Inverse(?) prior for tau -- tau = 1/inv_tau_sample"""

    # TODO: ask about the even/odd difference, and what 'bFreq' is

    n = len(data)
    _spline = spline_model(weights=w, n=n)
    is_even = n % 2 == 0
    if is_even:
        spline_normed_data = data[1:-1] / _spline[1:-1]
    else:
        spline_normed_data = data[1:] / _spline[1:]

    n = len(spline_normed_data)

    shape = τα + n / 2
    rate = τβ + np.sum(spline_normed_data) / (2 * np.pi) / 2
    return Gamma(k=shape, theta=1 / rate)


def sample_φδτ(k, w, τ, τα, τβ, φ, φα, φβ, δ, δα, δβ, data, spline_model):
    φ = φ_prior(k, w, spline_model.penalty_matrix, φα, φβ, δ).sample().flat[0]
    δ = δ_prior(φ, φα, φβ, δα, δβ).sample().flat[0]
    τ = 1 / inv_τ_prior(w, data, spline_model, τα, τβ).sample()
    return φ, δ, τ


def llike(w, τ, data, spline_model):
    """Whittle log likelihood"""
    # TODO: Move to using bilby likelihood
    # TODO: the parameters to this function should
    #  be the sampling parameters, not the matrix itself!
    # todo: V should be computed in here

    n = len(data)
    _lnspline = spline_model(weights=w, n=n) * τ


    is_even = n % 2 == 0
    if is_even:
        _lnspline = _lnspline[1:]
        data = data[1:]
    else:
        _lnspline = _lnspline[1:-1]
        data = data[1:-1]
    _spline = np.exp(_lnspline)


    integrand = _lnspline + np.exp(np.log(data) - _lnspline * 2 * np.pi)
    lnlike = -np.sum(integrand) / 2
    if not np.isfinite(lnlike):
        raise ValueError(f"lnlike is not finite: {lnlike}")
    return lnlike


def lpost(k, w, τ, τα, τβ, φ, φα, φβ, δ, δα, δβ, data, psline_model):
    logprior = lprior(
        k, w, τ, τα, τβ, φ, φα, φβ, δ, δα, δβ, psline_model.penalty_matrix
    )
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
