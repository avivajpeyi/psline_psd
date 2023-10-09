import os
import time
from abc import ABC, abstractmethod
from collections import namedtuple
from pprint import pformat
from typing import Callable, Dict, Optional, Union

import numpy as np
from tqdm.auto import trange

from slipper.plotting.gif_creator import create_gif
from slipper.sample.sampling_result import Result
from slipper.splines.knot_locator import KnotLocatorType
from slipper.splines.p_splines import PSplines

from ..logger import logger

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
        "spline_model",
    ],
)


class BaseSampler(ABC):
    def __init__(
        self,
        data: np.ndarray,
        outdir: str = ".",
        sampler_kwargs: Optional[dict] = {},
        spline_kwargs: Optional[dict] = {},
    ):
        self.data = data
        self.outdir = _mkdir(outdir)
        self.result: Union[Result, None] = None
        self.sampler_kwargs = sampler_kwargs
        self.spline_kwargs = spline_kwargs

        assert (self.n_steps - self.burnin) / self.thin > self.n_basis
        self.samples = None
        self.args: LnlArgs = None

        self.spline_model = PSplines.from_kwarg_dict(self.spline_kwargs)

    @classmethod
    def fit(
        cls, data, outdir=".", sampler_kwargs={}, spline_kwargs={}
    ) -> Result:
        sampler = cls(
            data=data,
            outdir=outdir,
            sampler_kwargs=sampler_kwargs,
            spline_kwargs=spline_kwargs,
        )
        sampler.run()
        return sampler.result

    def __check_to_make_chkpt_plt(self, step_num) -> bool:
        n_plts = self.sampler_kwargs["n_checkpoint_plts"]
        if not hasattr(self, "_checkpoint_plt_idx"):
            self._checkpoint_plt_idx = np.unique(
                np.linspace(1, self.n_steps, n_plts, dtype=int)
            )
        return n_plts > 0 and step_num in self._checkpoint_plt_idx

    def _plot_model_and_data(self, i=None, label="model_and_data"):
        self._compile_sampling_result()
        fig = self.result.plot_model_and_data(i=i)
        fig.savefig(f"{self.outdir}/{label}.png")
        w = self.result.idata.posterior.weight.values
        if self.result.logged_splines:
            kwgs = dict(weights=w[i, :])
        else:
            kwgs = dict(V=w[i, :])
        fig, _ = self.spline_model.plot(**kwgs)
        fig.savefig(f"{self.outdir}/{label}_spline.png")

    def run(self, verbose: bool = True):
        sk = self.sampler_kwargs.copy()
        sk["data"] = f"[{len(self.data)} points]"
        msg = f"Running sampler with the following arguments:\n"
        msg += f"Sampler arguments:\n{pformat(self.sampler_kwargs)}\n"
        msg += f"Spline arguments:\n{pformat(sk)}\n"
        logger.info(msg)

        self.t0 = time.process_time()
        self._init_mcmc()
        self._plot_model_and_data(i=0, label="initial_fit")

        pbar = trange(
            1, self.n_steps, desc="MCMC sampling", disable=not verbose
        )
        for itr in pbar:
            self._mcmc_step(itr)
            if self.__check_to_make_chkpt_plt(itr):
                logger.info("<<Plotting checkpoint>>")
                self.__plot_checkpoint(itr)
            pbar.set_postfix(
                dict(
                    lnP=self.samples["lpost_trace"][itr - 1],
                    accept=self.samples["acceptance_fraction"][itr - 1],
                )
            )
        self._compile_sampling_result()
        self._plot_model_and_data(i=self.n_steps - 1, label="final_fit")
        self.samples = None
        self.save()
        if self.sampler_kwargs["n_checkpoint_plts"]:
            logger.info("<<Creating gif>>")
            create_gif(
                f"{self.outdir}/checkpoint*.png",
                f"{self.outdir}/checkpoint.gif",
            )

    @abstractmethod
    def _init_mcmc(self) -> None:
        """Initialises the self.samples and self.spline_model attributes"""
        raise NotImplementedError

    @abstractmethod
    def _mcmc_step(self, itr: int):
        """Main mcmc step logic.

        Updates self.samples with the new samples from the MCMC
        """
        raise NotImplementedError

    def save(self):
        assert self.result is not None, "No result to save"
        self.result.save(f"{self.outdir}/result.nc")

    def __plot_checkpoint(self, i: int):
        fname = f"{self.outdir}/checkpoint_{i}.png"
        self._compile_sampling_result()
        self.result.make_summary_plot(
            fn=fname, use_cached=False, max_it=self.n_steps
        )

    def _compile_sampling_result(self):
        idx = np.where(self.samples["τ"] != 0)[0]
        if "V" in self.samples:
            weights = self.samples["V"][idx]
        elif "w" in self.samples:
            weights = self.samples["w"][idx]
        else:
            raise ValueError("No weights found")
        self.result = Result.compile_idata_from_sampling_results(
            posterior_samples=np.array(
                [
                    self.samples["φ"][idx],
                    self.samples["δ"][idx],
                    self.samples["τ"][idx],
                ]
            ),
            lpost_trace=self.samples["lpost_trace"][idx],
            frac_accept=self.samples["acceptance_fraction"][idx],
            weight_samples=weights,
            spline_model_kwargs=dict(
                knots=self.spline_model.knots,
                degree=self.spline_model.degree,
                diffMatrixOrder=self.spline_model.diffMatrixOrder,
                logged=self.spline_model.logged,
                basis=self.spline_model.basis,
            ),
            data=self.data,
            runtime=time.process_time() - self.t0,
            burn_in=self.sampler_kwargs["burnin"],
        )

    @property
    def sampler_kwargs(self):
        return self._sampler_kwargs

    @sampler_kwargs.setter
    def sampler_kwargs(self, kwargs):
        kwgs = self._default_sampler_kwargs()
        kwgs.update(kwargs)
        if kwgs["burnin"] == None:
            kwgs["burnin"] = kwgs["Ntotal"] // 3
        self._sampler_kwargs = kwgs
        if self._sampler_kwargs["n_checkpoint_plts"]:
            logger.warning(
                "Checkpoint plotting is enabled. This will slow down the sampling process."
            )

    def _default_sampler_kwargs(self):
        return dict(
            Ntotal=500,
            burnin=100,
            thin=1,
            τα=0.001,
            τβ=0.001,
            φα=1,
            φβ=1,
            δα=1e-04,
            δβ=1e-04,
            n_checkpoint_plts=0,
        )

    @property
    def spline_kwargs(self):
        return self._spline_kwargs

    @spline_kwargs.setter
    def spline_kwargs(self, kwargs: Dict):
        kwgs = self._default_spline_kwargs()
        kwgs.update(kwargs)
        kwgs["n_knots"] = kwgs["k"] - kwgs["degree"] + 1
        kwgs["data"] = self.data
        self._spline_kwargs = kwgs

    def _default_spline_kwargs(self):
        return dict(
            k=min(round(len(self.data) / 4), 40),
            degree=3,
            diffMatrixOrder=2,
            knot_locator_type=KnotLocatorType.linearly_spaced,
            logged=False,
        )

    @property
    def n_steps(self):
        return self.sampler_kwargs["Ntotal"]

    @property
    def thin(self):
        return self.sampler_kwargs["thin"]

    @property
    def n_basis(self):
        return self.spline_kwargs["k"]

    @property
    def burnin(self):
        return self.sampler_kwargs["burnin"]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.sampler_kwargs}, {self.spline_model})"


def _mkdir(d):
    os.makedirs(d, exist_ok=True)
    return d


def _timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def _tune_proposal_distribution(
    aux: np.array,
    accept_frac: float,
    sigma: float,
    weight: np.array,
    lpost_store: float,
    args: LnlArgs,
    lnpost_fn: Callable,
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

    accept_frac = accept_count / n_weight_columns
    return weight, accept_frac, sigma, lpost_store  # return updated values


def _update_weights(
    sigma,
    weight,
    widx,
    lpost_args,
    lpost_store,
    accept_count,
    lpost_fn: Callable,
):
    Z = np.random.normal()
    U = np.log(np.random.uniform())

    # Compute LnL using new value
    weight_star = weight + sigma * Z
    lpost_args[1][widx] = weight_star  # update V_star
    lpost_star = lpost_fn(*lpost_args)

    # Return new value if accepted
    lnl_diff = (lpost_star - lpost_store).ravel()[0]
    if U < np.min([0, lnl_diff]):
        return weight_star, lpost_star, accept_count + 1
    return weight, lpost_store, accept_count
