import os

import bilby
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import slipper
from slipper.example_datasets.lisa_data import lisa_wd_strain
from slipper.fourier_methods import get_periodogram

outdir = "outdir_wdb"
os.makedirs(outdir, exist_ok=True)

# LOAD AND PLOT DATA
data = lisa_wd_strain()
data = pd.DataFrame(data, columns=["strain"])
dt = 5.00000079
data["time"] = np.arange(0, len(data) * dt, dt)
sensitivity_range = [9**-4, 11**-1]  # fq
s_in_day = 24 * 60 * 60
fig, (ax0, ax1) = plt.subplots(nrows=2, layout="constrained")
ax0.plot(data.time / s_in_day, data.strain)
ax0.set_xlabel("Time (days)")
ax0.set_ylabel("Normalised Strain")
mpl_psd, freqs = ax1.psd(
    data.strain,
    NFFT=8192,
    Fs=1 / dt,
    scale_by_freq=False,
    window=mlab.window_none,
)
ax1.set_xscale("log")
ax1.set_xlabel("Frequency (Hz)")
ax1.grid(False)
ax1.axvspan(*sensitivity_range, alpha=0.2, color="tab:green")
plt.savefig("lisa_wd_strain.png")

## FIT DATA WITH POWER LAW
mask = (freqs > sensitivity_range[0]) & (freqs < sensitivity_range[1])
# log_f = np.log(freqs[mask])
# log_psd = np.log(mpl_psd[mask])
#
# plt.figure()
# plt.plot(log_f, log_psd)
# plt.show()
#
# model = lambda x, m, c: x * m + c
#
#
#
# priors = bilby.core.prior.PriorDict(dict(
#     m=bilby.core.prior.Normal(mu=-1, sigma=10, name="m"),
#     c=bilby.core.prior.Normal(-100, sigma=10, name="c"),
#     sigma=bilby.core.prior.Normal(0, 5, name="sigma"),
# ))
# likelihood = bilby.likelihood.GaussianLikelihood(log_f, log_psd, model)

# result = bilby.run_sampler(
#         likelihood=likelihood,
#         priors=priors,
#         sampler="dynesty",
#         nlive=1000,
#         sample="unif",
#         outdir=outdir,
#         label="mcmc",
#     )
#
# result = bilby.result.Result.from_json(f"{outdir}/mcmc_result.json")
#
# pred_log_f = np.log(np.geomspace(*sensitivity_range, 1000))
# pred_log_psd = np.zeros((len(result.posterior), len(pred_log_f)))
# m, c = result.posterior["m"].values, result.posterior["c"].values
# for i in range(len(result.posterior)):
#     pred_log_psd[i] = model(pred_log_f, m[i], c[i])
# pred_f = np.exp(pred_log_f)
# pred_psd = np.exp(pred_log_psd)
#
# plt.figure()
# # get the median and 90% credible interval
# median = np.median(pred_psd, axis=0)
# upper = np.percentile(pred_psd, 95, axis=0)
# lower = np.percentile(pred_psd, 5, axis=0)
# plt.plot(pred_f, median, color="C1")
# plt.fill_between(pred_f, lower, upper, color="C1", alpha=0.2)
# plt.plot(freqs, mpl_psd, color="k", alpha=0.5)
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("PSD")
# plt.show()


# LISA is sensitive from 10^-4 to 10^-1 Hz, plot a green box around that


# %% md

# %%


#
from slipper.sample.spline_model_sampler import fit_data_with_log_spline_model
from slipper.splines.initialisation import knot_locator

valid_f_mask = (freqs > sensitivity_range[0]) & (freqs < sensitivity_range[1])

data_to_fit = mpl_psd[valid_f_mask]
valid_freqs = freqs[valid_f_mask]

knots = knot_locator(
    data_to_fit,
    k=30,
    # eqSpaced=True,
    # eqSpaced: bool = False,
    degree=3,
    diffMatrixOrder=2,
    data_bin_edges=[11**-4, 12**-3, 10**-1, 0.2, 0.8],
    # data_bin_weights=[
    #     1,
    #     5,
    #     70,
    #     10,
    #     1,
    #     1
    #     ]
)
x = np.linspace(0, 1, len(data_to_fit))
plt.figure()
plt.plot(x, data_to_fit)
plt.xscale("log")
plt.yscale("log")
plt.scatter(
    knots, np.ones_like(knots) * min(data_to_fit), color="k", marker="x"
)
plt.show()

outdir = "wdb"
mcmc = fit_data_with_log_spline_model(
    data=data_to_fit,
    Ntotal=2000,
    burnin=1000,
    outdir=outdir,
    n_checkpoint_plts=5,
    spline_kwargs=dict(
        k=30,
        eqSpaced=True,
        # eqSpaced: bool = False,
        degree=3,
        diffMatrixOrder=2,
        # nfreqbin=None,
        # wfreqbin=None,
        # eqSpaced=False,
        # data_bin_edges=
        # data_bin_weights=
    ),
)

psd_x = np.linspace(0, 1, len(data_to_fit))
# ax.scatter(psd_x, pdgrm, color="k", label="Data", s=0.1)
# plot_xy_binned(psd_x, pdgrm, ax=ax, color="k", label="Data", ms=0, ls='--')
plt.figure(figsize=(8, 4))
plt.scatter(
    psd_x[1:-1],
    data_to_fit[1:-1],
    color="k",
    label="Data",
    marker=",",
    alpha=0.5,
    s=1,
)
x = np.linspace(0, 1, len(mcmc.psd_quantiles[0]))
plt.plot(
    x[1:-1],
    mcmc.psd_quantiles[0][1:-1],
    color=f"C1",
    alpha=0.5,
)
plt.fill_between(
    x[1:-1],
    mcmc.psd_quantiles[1][1:-1],
    mcmc.psd_quantiles[2][1:-1],
    color=f"C1",
    alpha=0.2,
    linewidth=0.0,
)
plt.xscale("log")
plt.yscale("log")
# plt.savefig(f'{outdir}/lisa_wd_strain_mcmc.png')
plt.show()
