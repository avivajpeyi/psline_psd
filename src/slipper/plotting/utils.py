import os

import matplotlib.pyplot as plt
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))


# def set_style():
#     """Set the default style for plotting"""
#     plt.style.use(f"{DIR}/style.mplstyle")


def set_plotting_style():
    plt.style.use("default")
    plt.rcParams["savefig.dpi"] = 100
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["font.size"] = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
    plt.rcParams["font.cursive"] = ["Liberation Sans"]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["image.cmap"] = "inferno"


def hide_axes_spines(ax):
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def convert_axes_spines_to_arrows(ax):
    # Arrow axes spines
    ax.spines[["left", "bottom"]].set_position(("data", 0))
    ax.spines[
        [
            "top",
            "right",
        ]
    ].set_visible(False)
    ax.spines[["left", "bottom"]].set_visible(True)
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)



def plot_xy_binned(
    x,
    y,
    ax,
    bins=30,
    **kwgs,
):
    # set some default kwargs
    defaults = dict(ms=6.0, yerr=[], fmt=".", color="k", zorder=-1000, alpha=0.5)
    for key, value in defaults.items():
        if key not in kwgs:
            kwgs[key] = value

    bins = np.linspace(min(x), max(x), bins)
    denom, _ = np.histogram(x, bins)
    num, _ = np.histogram(x, bins, weights=y)

    yerr = kwgs.pop("yerr")
    if len(yerr) == 0:
        yerr = np.zeros(len(x))


    err, _ = np.histogram(x, bins, weights=yerr)
    # std of y in each x-bin
    std = np.zeros(len(num))
    for i, (n, d) in enumerate(zip(num, denom)):
        if d > 0:
            x_idx = np.logical_and(x >= bins[i], x < bins[i + 1])
            std[i] = np.std(y[x_idx])
        else:
            std[i] = 0




    denom[num == 0] = 1.0
    new_x = 0.5 * (bins[1:] + bins[:-1])
    new_y = num / denom
    new_yerr = err / denom if np.any(err) else std/np.sqrt(denom)

    idx = np.nonzero(new_y)
    ax.errorbar(
        new_x[idx],
        new_y[idx],
        new_yerr[idx],
        **kwgs,
    )