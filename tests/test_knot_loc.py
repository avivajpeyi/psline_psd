import matplotlib.pyplot as plt
import numpy as np

from slipper.example_datasets.ar_data import get_ar_periodogram
from slipper.splines.initialisation import knot_locator


def test_binned_knots(tmpdir):
    pdgrm = get_ar_periodogram(order=4)
    data_bin_edges = [0.2, 0.4, 0.6]
    data_bin_weights = [0.5, 0.05, 0.5, 0.005]

    knot_locations = knot_locator(
        data=pdgrm,
        k=30,
        degree=3,
        data_bin_edges=data_bin_edges,
        data_bin_weights=data_bin_weights,
    )

    x = np.linspace(0, 1, len(pdgrm))

    plt.plot(x, pdgrm)
    plt.scatter(
        knot_locations, np.zeros(len(knot_locations)), c="r", zorder=10
    )
    # plot vertical lines at the bin edges
    for bin_edge in data_bin_edges:
        plt.axvline(bin_edge, color="k", ls="--", alpha=0.5)
    plt.savefig(f"{tmpdir}/test_binned_knots.png")

    # assert that there are at least 40% of the knots in the 0-0.2 bin
    assert np.sum(knot_locations < 0.2) > 0.4 * len(knot_locations)
