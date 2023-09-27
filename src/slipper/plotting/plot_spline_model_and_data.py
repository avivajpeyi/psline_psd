import matplotlib.pyplot as plt
import numpy as np

from .utils import convert_axes_spines_to_arrows, hide_axes_spines, plot_xy_binned


def plot_spline_model_and_data(
    data,
    model_quants,
    knots=[],
    separarte_y_axis=False,
    x=None,
    ax=None,
    colors=dict(Data="black", Splines="tab:orange", Knots="tab:red"),
    add_legend=False,
    logged_axes=False,
) -> plt.Axes:
    # prepare axes + figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.grid(False)
    ax_model = ax.twinx() if separarte_y_axis else ax
    hide_axes_spines(ax_model)
    if x is None:
        ax.set_yticks([])
        ax.set_xticks([])
        convert_axes_spines_to_arrows(ax)

    ax_knots = ax.twinx() if len(knots) > 0 else None

    # unpack data
    model_med, model_p05, model_p95 = (
        model_quants[0, :],
        model_quants[1, :],
        model_quants[2, :],
    )

    if x is None:
        x = np.linspace(0, 1, len(data))

    model_x = np.linspace(0, 1, len(model_med))

    # plot data

    data_bins = min(30, len(data) // 10)
    if data_bins > 10:
        plot_xy_binned(x, data, ax, bins=data_bins, label="Data", ls='--', ms=0.5)
    ax.scatter(x, data, color=colors["Data"], marker='.', alpha=0.1, s=0.6)
    ax_model.plot(model_x, model_med, color=colors["Splines"], alpha=0.5)
    ax_model.fill_between(
        model_x, model_p05, model_p95, color=colors["Splines"], alpha=0.2, linewidth=0.0
    )
    if len(knots) > 0:
        ax_knots.vlines(knots, 0, 0.1, color="tab:red", alpha=0.5)
        ax_knots.set_ylim(0, 1)
        hide_axes_spines(ax_knots)

    if logged_axes:
        ax_model.set_yscale("log")
        ax.set_yscale("log")
        # turn off y axes for log scale
        ax.get_yaxis().set_visible(False)
        ax_model.get_yaxis().set_visible(False)

    if add_legend:
        for label, color in colors.items():
            if label == "Knots" and len(knots) == 0:
                continue
            ax.plot([], [], color=color, label=label)
        ax.legend(markerscale=5, frameon=False, loc="upper right")

    fig = ax.get_figure()
    return fig
