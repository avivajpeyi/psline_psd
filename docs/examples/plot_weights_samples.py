import matplotlib.pyplot as plt

from slipper.sample.sampling_result import Result

# pcolor of weights


def plot_weights(fname):
    plt.close("all")
    r = Result.load(fname=fname)
    weights = r.idata.posterior.weight.values

    cbar = plt.pcolor(weights.T)
    cbar = plt.colorbar(cbar)
    # label
    plt.xlabel("Iteration")
    plt.ylabel("Spline")
    # label cbar
    cbar.set_label("Weight")
    plt.tight_layout()
    plt_fname = fname.replace(".nc", "_weights.png")
    plt.savefig(plt_fname)


plot_weights("out_compare_spline_and_log_spline/log_spline/result.nc")
plot_weights("out_compare_spline_and_log_spline/spline/result.nc")
