import os
from typing import Dict, List, Optional, Tuple, Union

import arviz as az
import numpy as np
import pandas as pd
from arviz import InferenceData
from scipy.fft import fft

from ..plotting import plot_metadata, plot_spline_model_and_data
from ..splines.p_splines import PSplines
from .post_processing import (
    generate_spline_posterior,
    generate_spline_quantiles,
)


class Result:
    def __init__(self, idata):
        self.idata = idata

    @classmethod
    def load(cls, fname: str):
        return cls(az.from_netcdf(fname))

    def save(self, fname: str):
        outdir = os.path.dirname(fname)
        outdir = outdir if outdir else "."
        self.make_summary_plot(os.path.join(outdir, "summary.png"))
        self.idata.to_netcdf(fname)

    @classmethod
    def create_idata(
        cls,
        samples: Dict[str, np.ndarray],
        spline_model,
        data: np.ndarray,
        sampler_stats: Dict,
    ):
        return cls.compile_idata_from_sampling_results(
            posterior_samples=np.array(
                [
                    samples["φ"],
                    samples["δ"],
                    samples["τ"],
                ]
            ),
            lpost_trace=samples["lpost_trace"],
            frac_accept=samples["acceptance_fraction"],
            weight_samples=samples["V"],
            spline_model_kwargs=dict(
                knots=spline_model.knots,
                degree=spline_model.degree,
                diffMatrixOrder=spline_model.diffMatrixOrder,
                logged=spline_model.logged,
                basis=spline_model.basis,
                knot_locator_type=spline_model.knot_locator_type,
            ),
            data=data,
            runtime=sampler_stats["runtime"],
            burn_in=sampler_stats["burnin"],
        )

    @classmethod
    def compile_idata_from_sampling_results(
        cls,
        posterior_samples,
        weight_samples,
        lpost_trace,
        frac_accept,
        spline_model_kwargs,
        data,
        burn_in,
        runtime,
    ) -> "Result":
        nsamp, n_weight_cols = weight_samples.shape

        knots = spline_model_kwargs["knots"]
        basis = spline_model_kwargs["basis"]

        n_knots = len(knots)
        n_gridpoints, n_basis = basis.shape

        draw_idx = np.arange(nsamp)
        knots_idx = np.arange(n_knots)
        weight_idx = np.arange(n_weight_cols)
        basis_idx = np.arange(n_basis)
        grid_point_idx = np.arange(n_gridpoints)

        logged_splines = 1
        if n_weight_cols == n_basis - 1:
            logged_splines = 0

        posterior = az.dict_to_dataset(
            dict(
                phi=posterior_samples[0, :],
                delta=posterior_samples[1, :],
                tau=posterior_samples[2, :],
                weight=weight_samples,
            ),
            coords=dict(weight_idx=weight_idx, draws=draw_idx),
            dims=dict(
                phi=["draws"],
                delta=["draws"],
                tau=["draws"],
                weight=["draws", "weight_idx"],
            ),
            default_dims=[],
            attrs={},
        )
        sample_stats = az.dict_to_dataset(
            dict(
                acceptance_rate=frac_accept[draw_idx],
                lp=lpost_trace[draw_idx],
            ),
            coords=dict(draws=draw_idx),
            attrs=dict(
                burn_in=burn_in,
                runtime=runtime,
            ),
            dims=dict(
                acceptance_rate=["draws"],
            ),
            default_dims=[],
            index_origin=None,
        )
        observed_data = az.dict_to_dataset(
            dict(data=data[0 : len(data)]),
            library=None,
            coords=dict(idx=np.arange(len(data))),
            dims=dict(data=["idx"]),
            default_dims=[],
            index_origin=None,
            attrs={},
        )

        spline_data = az.dict_to_dataset(
            dict(knots=knots, basis=basis),
            library=None,
            coords={
                "location": knots_idx,
                "grid_point": grid_point_idx,
                "basis_idx": basis_idx,
            },
            dims={"knots": ["location"], "basis": ["grid_point", "basis_idx"]},
            default_dims=[],
            attrs=dict(
                logged_splines=logged_splines,
                degree=spline_model_kwargs["degree"],
                diffMatrixOrder=spline_model_kwargs["diffMatrixOrder"],
                knot_locator_type=str(
                    spline_model_kwargs["knot_locator_type"]
                ),
            ),
            index_origin=None,
        )

        idata = InferenceData(
            posterior=posterior,
            sample_stats=sample_stats,
            observed_data=observed_data,
            constant_data=spline_data,
        )
        return cls(idata)

    @property
    def burn_in(self):
        if not hasattr(self, "_burn_in"):
            self._burn_in = self.idata.sample_stats.attrs["burn_in"]
        return self._burn_in

    @property
    def n_steps(self):
        return self.idata.posterior.coords["draws"].values[-1]

    @property
    def __posterior(self):
        # just the samples after 'burn_in' idx
        return self.idata.posterior.sel(draws=slice(self.burn_in, None))

    @property
    def sample_stats(self):
        return self.idata.sample_stats.sel(draws=slice(self.burn_in, None))

    @property
    def post_samples(self):
        return np.array(
            [
                self.__posterior["phi"],
                self.__posterior["delta"],
                self.__posterior["tau"],
            ]
        ).T

    @property
    def weights(self):
        return self.__posterior["weight"].values

    def all_samples(self):
        # samples without burn in cuttoff
        post = self.idata.posterior
        sampling_dat = self.idata.sample_stats
        return pd.DataFrame(
            dict(
                phi=post["phi"].values,
                delta=post["delta"].values,
                tau=post["tau"].values,
                acceptance_rate=sampling_dat["acceptance_rate"].values,
                lp=sampling_dat["lp"].values,
            )
        )

    @property
    def basis(self):
        return self.idata.constant_data["basis"].values

    @property
    def knots(self):
        return self.idata.constant_data["knots"].values

    @property
    def k(self) -> int:
        # umber of basis functions
        return len(self.basis.T)

    @property
    def data(self):
        return self.idata.observed_data["data"]

    @property
    def data_length(self) -> int:
        return len(self.data)

    @property
    def logged_splines(self) -> bool:
        return self.idata.constant_data.attrs["logged_splines"] == 1

    @property
    def spline_model(self):
        if not hasattr(self, "_spline_model"):
            attrs = self.idata.constant_data.attrs
            self._spline_model = PSplines(
                knots=self.knots,
                degree=attrs["degree"],
                diffMatrixOrder=attrs["diffMatrixOrder"],
                logged=self.logged_splines,
            )
        return self._spline_model

    def make_summary_plot(self, fn: str = "", use_cached=True, max_it=None):
        max_it = max_it if max_it else self.n_steps
        data = self.idata.observed_data["data"].values
        if use_cached:
            psd_quants = self.psd_quantiles
        else:
            end = self.n_steps
            if 1.5 * self.burn_in > self.n_steps:
                start = int(self.n_steps * 0.9)
            else:
                start = self.burn_in
            psd_quants = self.get_model_quantiles(start=start, end=end)
        return plot_metadata(
            posterior=self.all_samples(),
            model_quants=psd_quants,
            data=data,
            spline_model=self.spline_model,
            weights=self.idata.posterior["weight"].values,
            burn_in=self.burn_in,
            fname=fn,
            max_it=max_it,
        )

    def get_model_quantiles(self, start=None, end=None):
        """return quantiles of the model"""
        if start is None:
            start = self.burn_in
        if end is None:
            end = self.n_steps
        post = self.idata.posterior.sel(draws=slice(start, end))
        tau_samples = post.tau.values
        weight_samples = post.weight.values
        # get rows where posterior is non-zero
        plot_idx = np.where(post.phi.values != 0)[0]
        tau_samples = tau_samples[plot_idx]
        weight_samples = weight_samples[plot_idx]
        return generate_spline_quantiles(
            self.data_length,
            self.basis,
            tau_samples,
            weight_samples,
            logged_splines=self.logged_splines,
        )

    def plot_model_and_data(self, i=None):
        if i is None:
            start, end = None, None
        elif i == 0:
            start, end = 0, 0
        elif i == -1 or i == self.n_steps - 1:
            start, end = self.n_steps - 1, self.n_steps - 1
        else:
            start, end = i, i + 1
        model_qantiles = self.get_model_quantiles(start, end)
        return plot_spline_model_and_data(
            self.data,
            model_qantiles,
            knots=self.knots,
            logged_axes=self.logged_splines,
        )

    @property
    def psd_quantiles(self):
        """return quants if present, else compute cache and return"""
        # if attribute exists return
        if not hasattr(self, "_psd_quant"):
            self._psd_quant = self.get_model_quantiles(self.burn_in)
        return self._psd_quant

    @property
    def psd_posterior(self):
        if not hasattr(self, "_psds"):
            self._psds = generate_spline_posterior(
                self.data_length,
                self.basis,
                self.post_samples[:, 2],
                weight_samples=self.weights,
                logged=self.logged_splines,
            )
        return self._psds

    # def __str__(self):
    #     retu
    # def __repr__(self):
    #     return "Result(

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"Result("
            f"n_steps={self.n_steps}, "
            f"burn_in={self.burn_in}, "
            f"data_length={self.data_length}, "
            f"n_basis={self.k}, "
            f"logged_splines={self.logged_splines}, "
            f"knot_type={self.idata.constant_data.attrs['knot_locator_type']})"
        )

    @property
    def summary(self) -> str:
        s = self.__str__()
        # remove Result( and )
        s = s[7:-1]
        s = s.replace(", ", "\n")
        return s
