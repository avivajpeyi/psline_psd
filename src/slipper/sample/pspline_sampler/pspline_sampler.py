import numpy as np

from slipper.sample.base_sampler import (
    BaseSampler,
    _tune_proposal_distribution,
)

from .bayesian_functions import LnlArgs, lpost, sample_φδτ


class PsplineSampler(BaseSampler):
    def _init_mcmc(self) -> None:
        """Initialises the self.samples with the itial values of the MCMC"""

        self.samples = dict(
            V=np.zeros((self.n_steps, self.n_basis - 1)),
            φ=np.zeros(self.n_steps),
            δ=np.zeros(self.n_steps),
            τ=np.zeros(self.n_steps),
            proposal_sigma=np.zeros(self.n_steps),
            acceptance_fraction=np.zeros(self.n_steps),
        )

        sk = self.sampler_kwargs
        self.samples["τ"][0] = 1.0
        self.samples["δ"][0] = sk["δα"] / sk["δβ"]
        self.samples["φ"][0] = sk["φα"] / (sk["φβ"] * self.samples["δ"][0])
        self.samples["V"][0, :] = self.spline_model.guess_weights(
            self.data, fname=f"{self.outdir}/init_weights.png"
        )
        self.samples["proposal_sigma"][0] = 1
        self.samples["acceptance_fraction"][0] = 0.4
        self.samples["lpost_trace"] = np.zeros(self.n_steps)
        self.args = LnlArgs(
            w=self.samples["V"][0],
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
        aux = np.arange(0, k - 1)
        np.random.shuffle(aux)
        accept_frac = self.samples["acceptance_fraction"][itr - 1]
        sigma = self.samples["proposal_sigma"][itr - 1]

        self.args = self.args._replace(
            w=self.samples["V"][itr - 1, :],
            τ=self.samples["τ"][itr - 1],
            φ=self.samples["φ"][itr - 1],
            δ=self.samples["δ"][itr - 1],
        )

        # the values that will be updated
        τ, φ, δ, V, lpost_store = None, None, None, None, None
        for _ in range(self.thin):
            lpost_store = lpost(self.args)
            # 1. explore the parameter space for new V
            V, accept_frac, sigma, lpost_store = _tune_proposal_distribution(
                aux,
                accept_frac,
                sigma,
                self.args.w,
                lpost_store,
                self.args,
                lpost,
            )

            # 2. sample new values for φ, δ, τ
            φ, δ, τ = sample_φδτ(self.args)
            self.args = self.args._replace(w=V, τ=τ, φ=φ, δ=δ)

        # 3. store the new values
        self.samples["φ"][itr] = φ
        self.samples["δ"][itr] = δ
        self.samples["τ"][itr] = τ
        self.samples["V"][itr, :] = V
        self.samples["proposal_sigma"][itr] = sigma
        self.samples["acceptance_fraction"][itr] = accept_frac
        self.samples["lpost_trace"][itr] = lpost_store
