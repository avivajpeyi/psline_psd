import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft


def plot_psd(data, psd_quants, separarte_y_axis=False):
    psd, psd_p05, psd_p95 = psd_quants[0, :], psd_quants[1, :], psd_quants[2, :]
    n, newn = len(data), len(psd)
    periodogram = np.abs(np.power(fft(data), 2) / (2 * np.pi * n))[0:newn]
    psd_x = np.linspace(0, 3.14, newn)

    fig, ax = plt.subplots(1,1)

    ax.scatter(psd_x, periodogram, color="k", label="Data", s=0.75)

    if separarte_y_axis:
        ax1 = ax.twinx()
    else:
        ax1 = ax
    ax1.plot(psd_x, psd, color="tab:orange", alpha=0.5, label="Posterior")
    ax1.fill_between(
        psd_x, psd_p05, psd_p95, color="tab:orange", alpha=0.2, linewidth=0.0
    )

    ax.grid(False)
    ax.legend(markerscale=5, frameon=False)
    ax.set_ylabel("PSD")
    ax.set_xlabel("Freq")
    plt.tight_layout()
    plt.minorticks_off()
    return fig
