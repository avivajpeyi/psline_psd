import matplotlib.pyplot as plt
import numpy as np
from bilby.core.prior import ConditionalPriorDict, Gamma, ConditionalGamma

def _vPv(v, P):
    return np.dot(np.dot(v.T, P), v)


def lprior(k, v, τ, τα, τβ, φ, φα, φβ, δ, δα, δβ, P):
    # TODO: Move to using bilby priors

    vTPv = _vPv(v, P)
    logφ = np.log(φ)
    logδ = np.log(δ)
    logτ = np.log(τ)

    lnpri_weights = (k - 1) * logφ * 0.5 - φ * vTPv * 0.5
    lnpri_φ = φα * logδ + (φα - 1) * logφ - φβ * δ * φ
    lnpri_δ = (δα - 1) * logδ - δβ * δ
    lnpri_τ = -(τα + 1) * logτ - τβ / τ
    log_prior = lnpri_weights + lnpri_φ + lnpri_δ + lnpri_τ
    return log_prior

def lprior2(k, v, τ, τα, τβ, φ, φα, φβ, δ, δα, δβ, P):
    # TODO: Move to using bilby priors

    vTPv = _vPv(v, P)
    logφ = np.log(φ)
    logδ = np.log(δ)
    logτ = np.log(τ)

    lnpri_weights = (k - 1) * logφ * 0.5 - φ * vTPv * 0.5
    lnpri_φ = φα * logδ + (φα - 1) * logφ - φβ * δ * φ
    lnpri_δ = (δα - 1) * logδ - δβ * δ
    lnpri_τ = -(τα + 1) * logτ - τβ / τ
    log_prior = lnpri_weights + lnpri_φ + lnpri_δ + lnpri_τ
    return log_prior


def φ_prior(k, v, P, φα, φβ, δ):
    vTPv = np.dot(np.dot(v.T, P), v)
    shape = (k - 1) / 2 + φα
    rate = φβ * δ + vTPv / 2
    return Gamma(k=shape, theta=1 / rate)


def δ_prior(φ, φα, φβ, δα, δβ):
    """Gamma prior for pi(δ|φ)"""
    shape = φα + δα
    rate = φβ * φ + δβ
    return Gamma(k=shape, theta=1 / rate)


def inv_τ_prior(v, data, spline_model, τα, τβ):
    """Inverse(?) prior for tau -- tau = 1/inv_tau_sample"""

    # TODO: ask about the even/odd difference, and what 'bFreq' is

    n = len(data)
    _spline = spline_model(v=v, n=n)
    is_even = n % 2 == 0
    if is_even:
        spline_normed_data = data[1:-1] / _spline[1:-1]
    else:
        spline_normed_data = data[1:] / _spline[1:]

    n = len(spline_normed_data)

    shape = τα + n / 2
    rate = τβ + np.sum(spline_normed_data) / (2 * np.pi) / 2
    return Gamma(k=shape, theta=1 / rate)



def plot_pri_samples(p):
    plt.hist(p.sample(1000), bins=50)


def sample_φδτ(k, v, τ, τα, τβ, φ, φα, φβ, δ, δα, δβ, data, spline_model):
    φ = φ_prior(k, v, spline_model.penalty_matrix, φα, φβ, δ).sample().flat[0]
    δ = δ_prior(φ, φα, φβ, δα, δβ).sample().flat[0]
    τ = 1 / inv_τ_prior(v, data, spline_model, τα, τβ).sample()
    return φ, δ, τ




def delta_conditional():
    pass


def SplinePrior(k, v, τ, τα, τβ, φ, φα, φβ, δ, δα, δβ, data, spline_model):
    def delta_conditional(reference_params, φ):
        rate = φβ * φ + δβ
        return dict(
            k=reference_params['shape'],
            theta=1/rate
        )

    return ConditionalPriorDict(dict(
        φ=φ_prior(k, v, spline_model.penalty_matrix, φα, φβ, δ),
        δ=ConditionalGamma(k=φα + δα, theta=1/(φβ * φ + δβ), condition_func=delta_conditional),
        τ=ConditionalGamma(),
    ))


from slipper.splines.p_splines import PSplines
from slipper.example_datasets.ar_data import get_ar_periodogram




if __name__ == '__main__':
    np.random.seed(0)
    k = 10
    spline_model = PSplines(knots=np.linspace(0,1,k), degree=3, diffMatrixOrder=2, all_knots_penalty_matrix=False)
    data = get_ar_periodogram(order=4)
    v = spline_model.guess_initial_v(data)
    kwargs = dict(k=k, v=v, τ=1, τα=1, τβ=1, φ=1, φα=1, φβ=1, δ=1, δα=1, δβ=1,  )
    sample = sample_φδτ(**kwargs, data=data, spline_model=spline_model)
    ln_prior = lprior(**kwargs, P=spline_model.penalty_matrix)
    # print(sample)
    print(ln_prior)
