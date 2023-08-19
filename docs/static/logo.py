import os
from urllib.request import urlopen

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, interp1d, splrep
from scipy.signal import savgol_filter
from tqdm.auto import trange

from slipper.plotting import set_plotting_style
from slipper.plotting.plot_spline_model_and_data import plot_spline_model_and_data
from slipper.plotting.utils import convert_axes_spines_to_arrows, hide_axes_spines
from slipper.sample.spline_model_sampler import fit_data_with_pspline_model

BODY_THRESH = 0.38
STRAP_THRESH = 0.38
BODY_DATA = pd.read_csv("body_small.csv", delimiter=", ")
STRAP_DATA = pd.read_csv("strap_small.csv", delimiter=", ")

Ntotal = 2000

FNAME = "logo.png"
data = [0.0]


def download_font():
    github_url = "https://github.com/google/fonts/blob/main/ofl/zeyada/Zeyada.ttf"
    url = github_url + "?raw=true"  # You want the actual file, not some html
    response = urlopen(url)
    with open("Zeyada.ttf", "wb") as f:
        f.write(response.read())


def get_font():
    """Use as fontproperties=prop)"""
    fname = "Zeyada.ttf"
    if not os.path.isfile(fname):
        download_font()
    prop = fm.FontProperties(fname=fname)
    return prop


def generate_base_fig(txt="slipper"):
    """Make a matplotlib figure with no spines, and text that fills the entire background"""
    fig, ax = plt.subplots(1, 1, figsize=(4, 2.2))
    # add text in background
    hide_axes_spines(ax)
    ax.text(
        0.5,
        0.45,
        txt,
        va="center",
        ha="center",
        fontsize=100,
        fontproperties=get_font(),
    )
    ax.set_xlim(0, 1)
    # color points above and below the threshold
    # ax.scatter(BODY_DATA.x, BODY_DATA.y, s=0.1,
    #            color=['tab:red' if y > BODY_THRESH else 'tab:orange' for y in BODY_DATA.y], zorder=10)
    # ax.scatter(STRAP_DATA.x, STRAP_DATA.y, s=0.1,
    #             color = ['tab:red' if y > STRAP_THRESH else 'tab:orange' for y in STRAP_DATA.y]
    #            )
    fig.savefig(FNAME, dpi=300)
    return fig, ax


def add_fits_from_body(ax, color="tab:green"):
    above_region = BODY_DATA[BODY_DATA.y > BODY_THRESH].y.values
    below_region = BODY_DATA[BODY_DATA.y <= BODY_THRESH].y.values
    kgs = dict(
        Ntotal=Ntotal,
        burnin=int(Ntotal * 0.1),
        compute_psds=True,
        k=10,
        eqSpacedKnots=False,
    )
    mcmc1 = fit_data_with_pspline_model(
        above_region, **kgs, metadata_plotfn="body_above.png"
    )
    mcmc2 = fit_data_with_pspline_model(
        below_region, **kgs, metadata_plotfn="body_below.png"
    )
    for mc in [mcmc1, mcmc2]:
        x = np.linspace(0, 1, len(mc.psd_quantiles[0]))
        # twin axes
        ax2 = ax.twinx()
        ax2.fill_between(
            x, mc.psd_quantiles[1], mc.psd_quantiles[2], color=color, alpha=0.5
        )
        ax2.plot(x, mc.psd_quantiles[0], color=color)
        hide_axes_spines(ax2)


def add_fake(
    ax,
    data,
    thresh,
    color="tab:green",
):
    above_region = data[data.y > thresh]
    below_region = data[data.y <= thresh]
    above_qtiles = __gen_qtiles(above_region)
    below_qtiles = __gen_qtiles(below_region)
    for qtiles in [above_qtiles, below_qtiles]:
        x1, y1 = qtiles[0].T
        x2, y2 = qtiles[2].T
        ax.fill(
            np.append(x1, x2[::-1]),
            np.append(y1, y2[::-1]),
            color=color,
            alpha=0.5,
            lw=0,
        )
        ax.plot(*qtiles[1].T, color=color)
        hide_axes_spines(ax)


def __gen_qtiles(df, reps=1000):
    """Generate quantiles for a dataframe, by sampling with replacement"""
    # df drop duplicates x

    lines = []
    xnew = np.linspace(0, 1, 100)
    for _ in trange(reps):
        df1 = df.copy()
        # add some gaussian noise to the x,y values
        df1.y = df1.y + np.random.normal(0, 0.01, len(df1))
        df1.x = df1.x + np.random.normal(0, 0.001, len(df1))
        df1 = df1.drop_duplicates(subset=["x"])
        df1 = df1.sort_values(by="x")
        # random int bw 1 and 5
        # newy = interp1d(
        #     df1.x.values, df1.y.values, kind='cubic', fill_value='extrapolate',
        #     bounds_error=False, assume_sorted=True
        #     )(xnew)

        # Linear length along the line:
        points = df1.values
        distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
        distance = np.insert(distance, 0, 0) / distance[-1]
        interpolator = interp1d(distance, points, kind="cubic", axis=0)
        alpha = np.linspace(0, 1, 75)
        newxy = interpolator(alpha)
        lines.append(newxy)

    qtles = np.quantile(lines, [0.025, 0.5, 0.975], axis=0)
    return qtles


def fig_with_spline_fits():
    fig, ax = generate_base_fig()
    # add spline fits
    # add_fits_from_body(ax)
    # plt.figure()
    add_fake(ax, BODY_DATA, BODY_THRESH, color="tab:orange")
    add_fake(ax, STRAP_DATA, STRAP_THRESH, color="tab:blue")

    # fig= plot_spline_model_and_data(mcmc1.data,  mcmc1.psd_quantiles + 0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(FNAME, dpi=300, transparent=True)
    # plot_spline_model_and_data(mcmc2, ax=ax)
    # fig.savefig(FNAME, dpi=300)


if __name__ == "__main__":
    generate_base_fig()
    fig_with_spline_fits()

fig, ax = plt.subplots()
ax.plot([1, 2, 3])
