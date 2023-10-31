from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from .utils import (
    convert_axes_spines_to_arrows,
    hide_axes_spines,
    plot_xy_binned,
)


def plot_spline_model_and_data(
    data,
    model_quants: Optional[np.ndarray] = None,
    model=None,
    knots=[],
    separarte_y_axis=False,
    x=None,
    ax=None,
    colors={},
    add_legend=False,
    logged_axes: Union[bool, List[str], str] = False,
    hide_axes=True,
    metadata_text="",
    focus_on_data=True,
    bin_data=False,
) -> plt.Figure:
    # prepare axes + figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.grid(False)
    ax_model = ax.twinx() if separarte_y_axis else ax

    default_colors = dict(Data="black", Splines="tab:orange", Knots="tab:red")
    # update default colors with user provided colors
    colors = {**default_colors, **colors}

    if hide_axes:
        hide_axes_spines(ax_model)

        if x is None:
            ax.set_yticks([])
            ax.set_xticks([])
            convert_axes_spines_to_arrows(ax)

    ax_knots = ax.twinx() if len(knots) > 0 else None

    d = data[1:]
    # unpack data
    if model_quants is not None:
        model_med, model_p05, model_p95 = (
            model_quants[0, 1:],
            model_quants[1, 1:],
            model_quants[2, 1:],
        )
    else:
        model_med = model[1:]
        model_p05 = model_med
        model_p95 = model_med

    if x is None:
        x = np.linspace(0, 1, len(d))

    model_x = np.linspace(min(x), max(x), len(model_med))

    # plot data
    if bin_data:
        data_bins = min(30, len(d) // 10)
        if data_bins > 10:
            plot_xy_binned(
                x, d, ax, bins=data_bins, label="Data", ls="--", ms=0.5
            )
    ax.scatter(x, d, color=colors["Data"], marker=".", alpha=0.1, s=1)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax_model.plot(model_x, model_med, color=colors["Splines"], alpha=0.5)
    ax_model.fill_between(
        model_x,
        model_p05,
        model_p95,
        color=colors["Splines"],
        alpha=0.2,
        linewidth=0.0,
    )
    # Transforming [0,1] to [1e-4,1e-1]
    m = max(x) - min(x)
    c = min(x)
    knots = m * knots + c
    if len(knots) > 0:
        ax_knots.vlines(knots, 0, 0.1, color="tab:red", alpha=0.5)
        ax_knots.set_ylim(0, 1)
        hide_axes_spines(ax_knots)

    if isinstance(logged_axes, bool):
        logged_axes = ["x", "y"] if logged_axes else []

    if "x" in logged_axes:
        ax.set_xscale("log")
    if "y" in logged_axes:
        ax_model.set_yscale("log")
        ax.set_yscale("log")

        # turn off y axes for log scale
        if hide_axes:
            ax.get_yaxis().set_visible(False)
            ax_model.get_yaxis().set_visible(False)

    if add_legend:
        for label, color in colors.items():
            if label == "Knots" and len(knots) == 0:
                continue
            if label == "Splines" and model_quants is not None:
                label = r"Splines (median $\pm 90\%$ CI)"

            ax.plot([], [], color=color, label=label)
        ax.legend(markerscale=5, frameon=False)

    if metadata_text:
        ax.text(
            1.05,
            1.0,
            metadata_text,
            bbox=dict(facecolor="none", edgecolor="none", pad=10),
            transform=ax.transAxes,
            verticalalignment="top",
        )

    if focus_on_data:
        ax_model.set_xlim(xlim)
        ax_model.set_ylim(ylim)

    fig = ax.get_figure()
    return fig
