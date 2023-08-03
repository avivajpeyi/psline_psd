import arviz as az
import numpy as np
from arviz import InferenceData
from scipy.fft import fft

from ..plotting import plot_metadata
from .post_processing import generate_psd_posterior, generate_psd_quantiles


class Result:
    def __init__(self, idata):
        self.idata = idata.to_dict()

    @classmethod
    def compile_idata_from_sampling_results(
        cls,
        posterior_samples,
        v_samples,
        lpost_trace,
        frac_accept,
        basis,
        knots,
        periodogram,
        omega,
        raw_data,
    ) -> "Result":
        nsamp, n_basis_minus_1, _ = v_samples.shape

        n_knots = len(knots)
        n_gridpoints, n_basis = basis.shape

        draw_idx = np.arange(nsamp)
        knots_idx = np.arange(n_knots)
        v_idx = np.arange(n_basis_minus_1)
        basis_idx = np.arange(n_basis)
        grid_point_idx = np.arange(n_gridpoints)

        posterior = az.dict_to_dataset(
            dict(
                phi=posterior_samples[:, 0],
                delta=posterior_samples[:, 1],
                tau=posterior_samples[:, 2],
                v=v_samples,
            ),
            coords=dict(v_idx=v_idx, draws=draw_idx),
            dims=dict(
                phi=["draws"],
                delta=[
                    "draws",
                ],
                tau=["draws"],
                v=["draws", "v_idx"],
            ),
            default_dims=[],
            attrs={},
        )
        sample_stats = az.dict_to_dataset(
            dict(acceptance_rate=frac_accept, lp=lpost_trace)
        )
        observed_data = az.dict_to_dataset(
            dict(periodogram=periodogram[0 : len(omega)], raw_data=raw_data),
            library=None,
            coords={"frequency": omega, "idx": np.arange(len(raw_data))},
            dims={"periodogram": ["frequency"], "raw_data": ["idx"]},
            default_dims=[],
            attrs={},
            index_origin=None,
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
            attrs={},
            index_origin=None,
        )

        return cls(
            InferenceData(
                posterior=posterior,
                sample_stats=sample_stats,
                observed_data=observed_data,
                constant_data=spline_data,
            )
        )

    @property
    def post_samples(self):
        post = self.idata["posterior"]
        post_samples = np.array([post["phi"], post["delta"], post["tau"]]).T
        return post_samples

    @property
    def v(self):
        return self.idata["posterior"]["v"]

    @property
    def omega(self):
        return self.idata["coords"]["frequency"]

    @property
    def basis(self):
        return self.idata["constant_data"]["basis"]

    @property
    def sample_stats(self):
        return self.idata["sample_stats"]

    @property
    def knots(self):
        return self.idata["constant_data"]["knots"]

    def make_summary_plot(self, fn: str):
        raw_data = self.idata["observed_data"]["raw_data"]
        data_scale = np.std(raw_data)
        raw_data = raw_data / data_scale

        accept_frac = self.sample_stats["acceptance_rate"].flatten()

        psd_quants = self.psd_quantiles * np.power(data_scale, 2)
        n, newn = len(raw_data), len(self.omega)
        periodogram = np.abs(np.power(fft(raw_data), 2) / (2 * np.pi * n))[0:newn]
        periodogram = periodogram * np.power(data_scale, 2)

        plot_metadata(
            self.post_samples,
            accept_frac,
            psd_quants,
            periodogram,
            self.basis,
            self.knots,
            self.v[-1],
            fn,
        )

    @property
    def psd_quantiles(self):
        """return quants if present, else compute cache and return"""
        # if attribute exists return
        if not hasattr(self, "_psd_quant"):
            self._psd_quant = generate_psd_quantiles(
                self.omega, self.basis, self.post_samples[:, 2], self.v
            )
        return self._psd_quant

    @property
    def psd_posterior(self):
        if not hasattr(self, "_psds"):
            self._psds = generate_psd_posterior(
                self.omega, self.basis, self.post_samples[:, 2], self.v
            )
        return self._psds
