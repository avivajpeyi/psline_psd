from slipper.sample.base_sampler import BaseSampler
import numpy as np
from typing import Callable

from slipper.sample.base_sampler import BaseSampler, _update_weights
from slipper.splines.initialisation import knot_locator
from slipper.splines.p_splines import PSplines

from .bayesian_functions import lpost, sample_φδ

from collections import namedtuple

LnlArgs = namedtuple(
    "LnlArgs",
    [
        "n_basis",
        "w",
        "τ",
        "τα",
        "τβ",
        "φ",
        "φα",
        "φβ",
        "δ",
        "δα",
        "δβ",
        "data",
        "spline_model"
    ])


class LogPsplineSampler(BaseSampler):
    def _init_mcmc(self) -> None:
        """Initialises the self.samples with the itial values of the MCMC"""

        # init spline model
        sk = self.spline_kwargs
        knots = knot_locator(self.data, self.n_basis, **sk)
        self.spline_model = PSplines(
            knots=knots,
            degree=sk["degree"],
            diffMatrixOrder=sk["diffMatrixOrder"],
            logged=True,
        )

        # init samples
        self.samples = dict(
            w=np.zeros((self.n_steps, self.n_basis)),
            φ=np.zeros(self.n_steps),
            δ=np.zeros(self.n_steps),
            τ=np.zeros(self.n_steps),
            # proposal values
            proposal_sigma=np.zeros(self.n_steps),
            acceptance_fraction=np.zeros(self.n_steps),
            sigma_tau=np.zeros(self.n_steps),
            acceptance_fraction_tau=np.zeros(self.n_steps),
        )

        sk = self.sampler_kwargs
        self.samples["τ"][0] = 1 / (2 * np.pi)
        self.samples["δ"][0] = sk["δα"] / sk["δβ"]
        self.samples["φ"][0] = sk["φα"] / (sk["φβ"] * self.samples["δ"][0])
        self.samples["w"][0, :] = self.spline_model.guess_weights(self.data).ravel()
        self.samples["proposal_sigma"][0] = 1
        self.samples["acceptance_fraction"][0] = 0.4
        self.samples["sigma_tau"][0] = 1
        self.samples["acceptance_fraction_tau"][0] = 0.4
        self.samples["lpost_trace"] = np.zeros(self.n_steps)

        self.args = LnlArgs(
            n_basis=self.n_basis,
            w=self.samples["w"][0],
            τ=self.samples["τ"][0],
            τα=self.sampler_kwargs["τα"],
            τβ=self.sampler_kwargs["τβ"],
            φ=self.samples["φ"][0],
            φα=self.sampler_kwargs["φα"],
            φβ=self.sampler_kwargs["φβ"],
            δ=self.samples["δ"][0],
            δα=self.sampler_kwargs["δα"],
            δβ=self.sampler_kwargs["δβ"],
            data=self.data,
            spline_model=self.spline_model,
        )

    def _mcmc_step(self, itr):
        k = self.n_basis
        aux = np.arange(0, k)
        np.random.shuffle(aux)
        accept_frac = self.samples["acceptance_fraction"][itr - 1]
        sigma = self.samples["proposal_sigma"][itr - 1]
        accept_frac_tau = self.samples["acceptance_fraction_tau"][itr - 1]
        sigma_tau = self.samples["sigma_tau"][itr - 1]

        self.args = self.args._replace(
            w=self.samples["w"][itr - 1, :],
            τ=self.samples["τ"][itr - 1],
            φ=self.samples["φ"][itr - 1],
            δ=self.samples["δ"][itr - 1],
        )

        for _ in range(self.thin):
            lpost_store = lpost(*self.args)
            # 1. explore the parameter space for new V
            w, τ, accept_frac, accept_frac_tau, sigma, sigma_tau, lpost_store = _tune_proposal_distribution(
                aux,
                accept_frac, accept_frac_tau,
                sigma, sigma_tau,
                self.args.w,
                self.args.τ,
                lpost_store, self.args, lpost
            )

            # 2. sample new values for φ, δ, τ
            φ, δ = sample_φδ(*self.args)
            self.args = self.args._replace(w=w, τ=τ, φ=φ, δ=δ)

        # 3. store the new values
        self.samples["φ"][itr] = φ
        self.samples["δ"][itr] = δ
        self.samples["τ"][itr] = τ
        self.samples["w"][itr, :] = w
        self.samples["proposal_sigma"][itr] = sigma
        self.samples["acceptance_fraction"][itr] = accept_frac
        self.samples["lpost_trace"][itr] = lpost_store
        self.samples["sigma_tau"][itr] = sigma_tau
        self.samples["acceptance_fraction_tau"][itr] = accept_frac_tau


def _tune_proposal_distribution(
        aux: np.array,
        accept_frac: float,
        accept_frac_tau,
        sigma: float,
        sigma_tau: float,
        weight: np.array,
        τ,
        lpost_store,
        args,
        lnpost_fn: Callable,
):
    n_weight_columns = len(weight)

    # tuning proposal distribution
    if accept_frac < 0.30:  # increasing acceptance pbb
        sigma = sigma * 0.90  # decreasing proposal moves
    elif accept_frac > 0.50:  # decreasing acceptance pbb
        sigma = sigma * 1.1  # increasing proposal moves

    if accept_frac_tau < 0.30:  # increasing acceptance pbb
        sigma_tau = sigma_tau * 0.90  # decreasing proposal moves
    elif accept_frac_tau > 0.50:  # decreasing acceptance pbb
        sigma_tau = sigma_tau * 1.1  # increasing proposal moves

    accept_count = 0  # ACCEPTANCE PROBABILITY
    accept_count_tau = 0

    # Update weights
    for g in range(0, n_weight_columns):
        pos = aux[g]
        weight[pos], lpost_store, accept_count = _update_weights(
            sigma, weight[pos], pos, args, lpost_store, accept_count, lnpost_fn
        )
        τ, lpost_store, accept_count_tau = _update_tau(
            sigma_tau, τ, args, lpost_store, accept_count_tau, lnpost_fn
        )

    accept_frac = accept_count / n_weight_columns
    accept_frac_tau = accept_count_tau / n_weight_columns
    return weight, τ, accept_frac, accept_frac_tau, sigma, sigma_tau, lpost_store  # return updated values


def _update_tau(sigma, τ, lpost_args, lpost_store, accept_count, lnpost_fn):
    Z = np.random.normal()
    U = np.log(np.random.uniform())

    τ_star = τ + sigma * Z
    lpost_args = lpost_args._replace(τ=τ_star)
    lpost_star = lnpost_fn(*lpost_args)

    lnl_diff = (lpost_star - lpost_store).ravel()[0]
    if U < np.min([0, lnl_diff]):
        return τ_star, lpost_star, accept_count + 1
    return τ, lpost_store, accept_count


