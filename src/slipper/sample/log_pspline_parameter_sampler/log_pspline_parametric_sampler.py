from collections import namedtuple
from typing import Callable

import numpy as np

from slipper.example_datasets.ar_data import get_true_ar_psd as arpsd
from slipper.fourier_methods import get_periodogram
from slipper.sample.base_sampler import (
    BaseSampler,
    _update_alpha,
    _update_weights,
)

from .bayesian_functions import LnlArgs, lpost, sample_φδ
from .parametric_models import instru_noise


class LogPsplineSamplerPara(BaseSampler):
    def _init_mcmc(self) -> None:
        """Initialises the self.samples with the itial values of the MCMC"""
        dat = self.data - np.mean(self.data)
        dat = dat / np.std(dat)

        # fs = 1 / 0.75
        sth, param = arpsd(
            data=dat, order=5
        )  # Assuming the data is time series
        self.data = get_periodogram(
            timeseries=self.data
        )  # ,fs=fs)  # converting the data to periodogram Add fs for LISA]

        # f = np.linspace(0, fs / 2, len(self.data))
        # param = instru_noise(f)#Instrumental noise (Include the choice later)

        # For truncating the data. (Include range inputs later)
        # self.data=self.data[(f >= 1e-4) & (f <= 0.1)]
        # param = param[(f >= 1e-4) & (f <= 0.1)]
        self.samples = dict(
            w=np.zeros((self.n_steps, self.n_basis)),
            φ=np.zeros(self.n_steps),
            δ=np.zeros(self.n_steps),
            alph=np.zeros(self.n_steps),
            # proposal values
            proposal_sigma=np.zeros(self.n_steps),
            acceptance_fraction=np.zeros(self.n_steps),
        )

        sk = self.sampler_kwargs
        self.samples["δ"][0] = sk["δα"] / sk["δβ"]
        self.samples["φ"][0] = sk["φα"] / (sk["φβ"] * self.samples["δ"][0])
        self.samples["alph"][0] = 0.5  # taking the mid point
        self.samples["w"][0, :] = self.spline_model.guess_weights(
            self.data, fname=f"{self.outdir}/init_weights.png"
        )
        self.samples["proposal_sigma"][0] = 1
        self.samples["acceptance_fraction"][0] = 0.4
        self.samples["lpost_trace"] = np.zeros(self.n_steps)

        self.args = LnlArgs(
            w=self.samples["w"][0],
            φ=self.samples["φ"][0],
            φα=self.sampler_kwargs["φα"],
            φβ=self.sampler_kwargs["φβ"],
            δ=self.samples["δ"][0],
            δα=self.sampler_kwargs["δα"],
            δβ=self.sampler_kwargs["δβ"],
            data=self.data,
            spline_model=self.spline_model,
            alph=self.samples["alph"][0],
            param=np.log(param),
        )

    def _mcmc_step(self, itr):
        k = self.n_basis
        aux = np.arange(0, k)
        np.random.shuffle(aux)
        accept_frac = self.samples["acceptance_fraction"][itr - 1]
        sigma = self.samples["proposal_sigma"][itr - 1]

        self.args = self.args._replace(
            w=self.samples["w"][itr - 1, :],
            φ=self.samples["φ"][itr - 1],
            δ=self.samples["δ"][itr - 1],
            alph=self.samples["alph"][itr - 1],
        )

        # the values that will be updated
        φ, δ, w, alph, lpost_store = None, None, None, None, None
        for _ in range(self.thin):
            lpost_store = lpost(self.args)
            # 1. explore the parameter space for new V and alph
            (
                w,
                accept_frac,
                sigma,
                lpost_store,
                alph,
            ) = _tune_proposal_distribution(
                aux,
                accept_frac,
                sigma,
                self.args.w,
                lpost_store,
                self.args,
                lpost,
                self.args.alph,
            )

            # 2. sample new values for φ, δ
            φ, δ = sample_φδ(self.args)
            self.args = self.args._replace(w=w, φ=φ, δ=δ, alph=alph)

        # 3. store the new values
        self.samples["φ"][itr] = φ
        self.samples["δ"][itr] = δ
        self.samples["alph"][itr] = alph
        self.samples["w"][itr, :] = w
        self.samples["proposal_sigma"][itr] = sigma
        self.samples["acceptance_fraction"][itr] = accept_frac
        self.samples["lpost_trace"][itr] = lpost_store

    def _default_spline_kwargs(self):
        _kwargs = super()._default_spline_kwargs()
        _kwargs["logged"] = True
        return _kwargs


def _tune_proposal_distribution(
    aux: np.array,
    accept_frac: float,
    sigma: float,
    weight: np.array,
    lpost_store,
    args,
    lnpost_fn: Callable,
    alpha: float,
):
    n_weight_columns = len(weight)

    # tuning proposal distribution
    if accept_frac < 0.30:  # increasing acceptance pbb
        sigma = sigma * 0.90  # decreasing proposal moves
    elif accept_frac > 0.50:  # decreasing acceptance pbb
        sigma = sigma * 1.1  # increasing proposal moves

    accept_count = 0  # ACCEPTANCE PROBABILITY

    # Update weights
    for g in range(0, n_weight_columns):
        pos = aux[g]
        weight[pos], lpost_store, accept_count = _update_weights(
            sigma, weight[pos], pos, args, lpost_store, accept_count, lnpost_fn
        )
    # Update Alpha:
    alpha, lpost_store, accept_count = _update_alpha(
        alpha, args, lpost_store, accept_count, lnpost_fn
    )
    accept_frac = accept_count / n_weight_columns
    print(alpha)
    return (
        weight,
        accept_frac,
        sigma,
        lpost_store,
        alpha,
    )
