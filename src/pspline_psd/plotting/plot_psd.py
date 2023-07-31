import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft


def plot_psd(data, psd_quants):
    psd, psd_p05, psd_p95 = psd_quants[0, :], psd_quants[1, :], psd_quants[2, :]
    n, newn = len(data), len(psd)
    periodogram = np.abs(np.power(fft(data), 2) / (2 * np.pi * n))[0:newn]
    psd_x = np.linspace(0, 3.14, newn)

    plt.plot(figsize=(5, 3))
    plt.scatter(psd_x, periodogram, color="k", label="Data", s=0.75)
    plt.plot(psd_x, psd, color="tab:orange", alpha=0.5, label="Posterior")
    plt.fill_between(
        psd_x, psd_p05, psd_p95, color="tab:orange", alpha=0.2, linewidth=0.0
    )

    plt.grid(False)
    plt.legend(markerscale=5, frameon=False)
    plt.ylabel("PSD")
    plt.xlabel("Freq")
    plt.tight_layout()
    plt.minorticks_off()
    return plt.gcf()
