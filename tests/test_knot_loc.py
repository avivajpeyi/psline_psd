import matplotlib.pyplot as plt
import numpy as np

from slipper.example_datasets.ar_data import get_ar_periodogram
from slipper.splines.knot_locator import KnotLocatorType, knot_locator


def test_binned_knots(tmpdir):
    pdgrm = get_ar_periodogram(order=4)
    data_bin_edges = [0.2, 0.4, 0.6]
    data_bin_weights = [0.5, 0.05, 0.5, 0.005]

    kwgs = dict(
        data=pdgrm,
        n_knots=30,
        degree=3,
        data_bin_edges=data_bin_edges,
        data_bin_weights=data_bin_weights,
        log_data=True,
        min_val=0.01,
    )

    # get the knots for each type of knot locator
    knot_loc_types = KnotLocatorType.__members__.keys()

    knots = {
        knot_loc_type: knot_locator(knot_loc_type, **kwgs)
        for knot_loc_type in knot_loc_types
    }

    x = np.linspace(0, 1, len(pdgrm))

    pdgrm = pdgrm / np.max(pdgrm)
    plt.plot(x, pdgrm, color="k", zorder=0, lw=0.3)
    stp = 1 / (len(knots.keys()) + 1)
    for i, (knot_loc_type, knot_loc) in enumerate(knots.items()):
        y = np.zeros(len(knot_loc))
        y = y + i * stp
        plt.scatter(
            knot_loc, y, label=knot_loc_type, color=f"C{i}", zorder=10, s=5
        )

    # plot vertical lines at the bin edges
    for bin_edge in data_bin_edges:
        plt.axvline(bin_edge, color="k", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig(f"{tmpdir}/binned_knots.png")
    plt.xscale("log")
    plt.savefig(f"{tmpdir}/binned_knots_log.png")

    # assert that there are at least 40% of the knots in the 0-0.2 bin
    binned_knots = knots["binned_knots"]
    assert np.sum(binned_knots < 0.2) > 0.4 * len(binned_knots)
