import os
import time
from abc import ABC, abstractmethod
from pprint import pformat
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm.auto import trange

from slipper.plotting.gif_creator import create_gif
from slipper.sample.sampling_result import Result

from ..logger import logger


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
        self.spline_model = None
        self.samples = None

    def __check_to_make_chkpt_plt(self, step_num) -> bool:
        n_plts = self.sampler_kwargs["n_checkpoint_plts"]
        if not hasattr(self, "_checkpoint_plt_idx"):
            self._checkpoint_plt_idx = np.unique(
                np.linspace(1, self.n_steps, n_plts, dtype=int)
            )
        return n_plts > 0 and step_num in self._checkpoint_plt_idx

    def run(self, verbose: bool = True):
        msg = f"Running sampler with the following arguments:\n"
        msg += f"Sampler arguments:\n{pformat(self.sampler_kwargs)}\n"
        msg += f"Spline arguments:\n{pformat(self.spline_kwargs)}\n"
        logger.info(msg)

        self.t0 = time.process_time()
        self._init_mcmc()
        for itr in trange(1, self.n_steps, desc="MCMC sampling", disable=not verbose):
            self._mcmc_step(itr)
            if self.__check_to_make_chkpt_plt(itr):
                logger.info("<<Plotting checkpoint>>")
                self.__plot_checkpoint(itr)
        self._comile_sampling_result()
        self.samples = None
        self.save()
        if self.sampler_kwargs["n_checkpoint_plts"]:
            logger.info("<<Creating gif>>")
            create_gif(
                f"{self.outdir}/checkpoint*.png", f"{self.outdir}/checkpoint.gif"
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
        self._comile_sampling_result()
        self.result.make_summary_plot(fn=fname, use_cached=False, max_it=self.n_steps)

    def _comile_sampling_result(self):
        idx = np.where(self.samples["τ"] != 0)[0]
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
            v_samples=self.samples["V"][idx],
            basis=self.spline_model.basis,
            knots=self.spline_model.knots,
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

    @property
    def spline_kwargs(self):
        return self._spline_kwargs

    @spline_kwargs.setter
    def spline_kwargs(self, kwargs):
        kwgs = self._default_spline_kwargs()
        kwgs.update(kwargs)
        self._spline_kwargs = kwgs

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

    def _default_spline_kwargs(self):
        return dict(
            k=min(round(len(self.data) / 4), 40),
            eqSpaced=False,
            degree=3,
            diffMatrixOrder=2,
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


def _mkdir(d):
    os.makedirs(d, exist_ok=True)
    return d


def _timestamp():
    return time.strftime("%Y%m%d_%H%M%S")
