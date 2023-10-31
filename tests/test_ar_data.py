import os

from slipper.example_datasets.ar_data import (
    generate_ar_timeseries,
    plot_ar_spectrogram_psd,
)


def test_ar_datagen(tmpdir):
    # Generate and plot AR timeseries of different orders
    for order in range(1, 6):
        ar_series = generate_ar_timeseries(order=order)
        title = f"AR({order})"
        fig = plot_ar_spectrogram_psd(ar_series, title, order=order)
        fname = f"{tmpdir}/ar_{order}.png"
        fig.savefig(fname)
        assert os.path.exists(fname)
