import os

import matplotlib.pyplot as plt

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
