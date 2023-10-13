import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from slipper.logger import logger
from slipper.plotting.plot_spline_model_and_data import (
    plot_spline_model_and_data,
)


def __optimize_ith_x(pspline, data, x, xkey, i, xi):
    x[i] = xi
    return pspline.lnlikelihood(data, **{xkey: x})


def __optimization_loop(pspline, data, x, xkey, kwgs, fast_optim=False):
    optim_all = lambda _x: -pspline.lnlikelihood(data, **{xkey: _x})
    x = minimize(optim_all, x0=x, **kwgs).x
    current_lnl = pspline.lnlikelihood(data, **{xkey: x})
    logger.debug(f"LnL={current_lnl:.3E}")

    if fast_optim:
        return x  # optimized enough
    else:  # optimise each x_i individually
        for i in range(len(x)):
            optim_ith = lambda _xi: -__optimize_ith_x(
                pspline, data, x, xkey, i, _xi
            )
            x[i] = minimize(optim_ith, x0=x[i], **kwgs).x
            current_lnl = pspline.lnlikelihood(data, **{xkey: x})
            logger.debug(f"LnL={current_lnl:.3E}")

        x = minimize(optim_all, x0=x, **kwgs).x
        current_lnl = pspline.lnlikelihood(data, **{xkey: x})
        logger.debug(f"LnL={current_lnl:.3E}")

        return x


def __optimize_starting_X(
    pspline,
    data,
    init_x,
    x_key="weights",
    n_optimization_steps=100,
    bounds=None,
    fname="",
):
    if bounds is not None:
        bounds = [(bounds[0], bounds[1]) * len(init_x)]

    # ignore the 1st and last datapoints (there are often issues with the start/end points)
    data = data[1:-1]
    n = len(data)

    orig_grid_n = pspline.n_grid_points
    pspline.n_grid_points = n

    init_lnl = pspline.lnlikelihood(data, **{x_key: init_x})
    kwgs = dict(
        options=dict(
            maxiter=pspline.n_basis * n_optimization_steps,
            xatol=1e-30,
            gtol=1e-30,
            disp=False,
            adaptive=True,
            return_all=False,
        ),
        # bounds=bounds,
        tol=1e-50,
        method="L-BFGS-B",
    )

    logger.debug(f"=====STARTING OPTIMISATION ({x_key}) =====")
    logger.debug(f"LnL={init_lnl:.3E}")
    x = __optimization_loop(
        pspline, data, init_x, x_key, kwgs, fast_optim=True
    )
    current_lnl = pspline.lnlikelihood(data, **{x_key: x})
    logger.info(
        f"Init weights optimization Lnl: {init_lnl:.3E}-->{current_lnl:.3E}"
    )
    logger.debug(f"{x_key}={x}")
    logger.debug(f"=====COMPLETED OPTIMISATION ({x_key}) =====")

    pspline.n_grid_points = orig_grid_n

    if fname:
        init_model = pspline(**{x_key: init_x})
        final_model = pspline(**{x_key: x})
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        plt_kwgs = dict(
            data=data,
            knots=pspline.knots,
            hide_axes=False,
            logged_axes="xy",
            ax=ax,
        )
        plot_spline_model_and_data(
            model=init_model, **plt_kwgs, colors=dict(Splines="tab:blue")
        )
        plot_spline_model_and_data(
            model=final_model, **plt_kwgs, colors=dict(Splines="tab:orange")
        )
        ax.set_title(f"Init Lnl: {init_lnl:.3E}-->{current_lnl:.3E}")
        plt.tight_layout()
        plt.show()
        plt.savefig(fname)

    return x.ravel()


def optimise_starting_weights(
    pspline, data, n_optimization_steps=100, fname=""
):
    """Guess init 'w' weights for the P-spline model from the data and the knots"""
    if pspline.logged:
        n = pspline.n_basis
        xKey = "weights"
    else:
        n = pspline.n_basis - 1
        xKey = "v"
    return __optimize_starting_X(
        pspline=pspline,
        data=data,
        init_x=np.zeros(n),
        x_key=xKey,
        n_optimization_steps=n_optimization_steps,
        # bounds=[(0,None) for i in range(pspline.n_basis-1)],
        fname=fname,
    )
