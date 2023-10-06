import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import slipper
from slipper.example_datasets.lisa_data import lisa_wd_strain
from slipper.fourier_methods import get_periodogram

data = lisa_wd_strain()
data = pd.DataFrame(data, columns=["strain"])
dt = 5.00000079
data["time"] = np.arange(0, len(data) * dt, dt)
# mean center and normalize the timeseries
h = data.strain
h_mean = np.mean(h)
h_std = np.std(h)
h = (h - h_mean) / h_std
data["strain"] = h

pdgrm = get_periodogram(timeseries=h[0:1024])


#%%
sensitivity_range = [9**-4, 11**-1]  # fq
s_in_day = 24 * 60 * 60
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, layout="constrained")
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

x = np.linspace(0, 1, len(pdgrm))
# scale x by freqs
x = x * freqs[-1]
# ignore all data outside of sensitivity range
mask = (x > sensitivity_range[0]) & (x < sensitivity_range[1])
x = x[mask]
pdgrm = pdgrm[mask]
ax2.loglog(x, pdgrm)

plt.savefig("lisa_wd_strain.png")


# LISA is sensitive from 10^-4 to 10^-1 Hz, plot a green box around that


#%% md

#%%
from slipper.sample.spline_model_sampler import fit_data_with_log_spline_model

valid_f_mask = (freqs > sensitivity_range[0]) & (freqs < sensitivity_range[1])

data_to_fit = mpl_psd[valid_f_mask]
valid_freqs = freqs[valid_f_mask]

outdir = "wdb"
mcmc = fit_data_with_log_spline_model(
    data=data_to_fit,
    Ntotal=2000,
    burnin=1000,
    outdir=outdir,
    n_checkpoint_plts=5,
    spline_kwargs=dict(
        k=30,
        # eqSpaced: bool = False,
        degree=3,
        diffMatrixOrder=2,
        # nfreqbin=None,
        # wfreqbin=None,
        eqSpaced=False,
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
