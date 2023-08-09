import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch
from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np


def generate_ar_timeseries(ar_coefs=None, order=None, n_samples=2000, ):
    if ar_coefs is None and order is None:
        raise ValueError("Must specify either order or ar_coefs")
    elif ar_coefs is not None and order is not None:
        raise ValueError("Must specify either order or ar_coefs, not both")
    elif ar_coefs is None and order is not None:
        # using some default AR coefficients
        if order == 1:
            ar_coefs = np.array([0.9])
        elif order == 2:
            ar_coefs = np.array([0.9, -0.8])
        elif order == 3:
            ar_coefs = np.array([0.9, -0.8, 0.7])
        elif order == 4:
            ar_coefs = np.array([0.9, -0.8, 0.7, -0.6])
        elif order == 5:
            ar_coefs = [1, - 2.7607, 3.8106, - 2.6535, 0.9238]
    else:
        # using user-specified AR coefficients
        order = len(ar_coefs)

    ar = np.array(ar_coefs)
    ma = np.array([1])
    ar_simulater = ArmaProcess(ar, ma)
    data = ar_simulater.generate_sample(nsample=n_samples)
    data = data - np.mean(data)
    data = data / np.std(data)

    # assert no nans / infs
    assert np.all(
        np.isfinite(data)), f"AR series contains NaNs or Infs: % invalid {np.sum(~np.isfinite(data)) / len(data) * 100}"

    return data


def plot_ar_spectrogram_psd(timeseries, title=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    ax1.plot(timeseries)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    if title:
        ax1.set_title(title)

    freq, psd = welch(timeseries, fs=1.0, nperseg=256)
    ax2.plot(freq, psd, label='Power Spectral Density', color='orange')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Power/Frequency')

    f, t, Sxx = spectrogram(timeseries, fs=1.0, window='hann', nperseg=64, noverlap=32)
    cbar = ax3.pcolormesh(t, f, 10 * np.log10(Sxx), shading='auto', cmap='viridis')
    ax3.set_ylabel('Frequency [Hz]')
    ax3.set_xlabel('Time [sec]')
    cbar.set_label('Power/Frequency [dB/Hz]')
    fig.colorbar(cbar, ax=ax3)

    fig.tight_layout()
    return fig
