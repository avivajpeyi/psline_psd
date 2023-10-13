import os

import matplotlib.pyplot as plt
import numpy as np

from slipper.example_datasets.lisa_data import lisa_noise_periodogram
from slipper.sample.spline_model_sampler import (
    Result,
    fit_data_with_log_spline_model,
    fit_data_with_pspline_model,
)

outdir = "LISA_test"
mcmc_original_fn = f"{outdir}/linear.netcdf"
mcmc_log_fn = f"{outdir}/log.netcdf"

if not os.path.exists(mcmc_original_fn):
    mcmc = fit_data_with_pspline_model(
        data=lisa_noise_periodogram()[100:],
        Ntotal=5000,
        burnin=2500,
        degree=3,
        eqSpaced=False,
        outdir=outdir,
    )
    mcmc.save(mcmc_original_fn)

if not os.path.exists(mcmc_log_fn):
    mcmc = fit_data_with_log_spline_model(
        data=lisa_noise_periodogram()[100:],
        Ntotal=5000,
        burnin=2500,
        degree=3,
        eqSpaced=False,
        outdir=outdir,
    )
    mcmc.save(mcmc_log_fn)

mcmc_orig = Result.load(mcmc_original_fn)
mcmc_log = Result.load(mcmc_log_fn)

mcmc_orig.make_summary_plot(f"{outdir}/linear.png")
mcmc_log.make_summary_plot(f"{outdir}/log.png")

pdgrm = mcmc_orig.data
pdgrm_x = np.linspace(0, 1, len(pdgrm))

plt.scatter(pdgrm_x[1:], pdgrm[1:], marker=",", s=1, alpha=0.5, label="Data")
print(mcmc_orig.psd_posterior.shape)
x = np.linspace(0, 1, len(mcmc_orig.psd_posterior[0]))
for i in range(10):
    plt.plot(x, mcmc_orig.psd_posterior[-i], color="tab:orange", alpha=0.1)
    plt.plot(x, mcmc_log.psd_posterior[-i], color="tab:blue", alpha=0.1)
plt.xscale("log")
plt.yscale("log")
# plt.show()
