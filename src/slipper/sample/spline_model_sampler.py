import numpy as np

from .log_pspline_sampler import LogPsplineSampler
from .pspline_sampler import PsplineSampler
from .sampling_result import Result


def fit_data_with_pspline_model(
    data: np.ndarray,
    Ntotal: int = 1000,
    burnin: int = None,
    thin: int = 1,
    τα: float = 0.001,
    τβ: float = 0.001,
    φα: float = 1,
    φβ: float = 1,
    δα: float = 1e-04,
    δβ: float = 1e-04,
    k: int = 30,
    eqSpaced: bool = False,
    degree: int = 3,
    diffMatrixOrder: int = 2,
    outdir: str = ".",
    n_checkpoint_plts: int = 0,
) -> Result:
    sampler = PsplineSampler(
        data=data,
        outdir=outdir,
        sampler_kwargs=dict(
            Ntotal=Ntotal,
            thin=thin,
            burnin=burnin,
            τα=τα,
            τβ=τβ,
            φα=φα,
            φβ=φβ,
            δα=δα,
            δβ=δβ,
            n_checkpoint_plts=n_checkpoint_plts,
        ),
        spline_kwargs=dict(
            k=k,
            eqSpaced=eqSpaced,
            degree=degree,
            diffMatrixOrder=diffMatrixOrder,
        ),
    )
    sampler.run()
    return sampler.result


def fit_data_with_log_spline_model(
    data: np.ndarray,
    Ntotal: int = 1000,
    burnin: int = None,
    thin: int = 1,
    τα: float = 0.001,
    τβ: float = 0.001,
    φα: float = 1,
    φβ: float = 1,
    δα: float = 1e-04,
    δβ: float = 1e-04,
    # k: int = 30,
    # eqSpaced: bool = False,
    # degree: int = 3,
    # diffMatrixOrder: int = 2,
    outdir: str = ".",
    n_checkpoint_plts: int = 0,
    spline_kwargs: dict = None,
) -> Result:

    sampler = LogPsplineSampler(
        data=data,
        outdir=outdir,
        sampler_kwargs=dict(
            Ntotal=Ntotal,
            thin=thin,
            burnin=burnin,
            τα=τα,
            τβ=τβ,
            φα=φα,
            φβ=φβ,
            δα=δα,
            δβ=δβ,
            n_checkpoint_plts=n_checkpoint_plts,
        ),
        # spline_kwargs=dict(
        #     k=k,
        #     eqSpaced=eqSpaced,
        #     degree=degree,
        #     diffMatrixOrder=diffMatrixOrder,
        # ),
        spline_kwargs=spline_kwargs,
    )
    sampler.run()
    return sampler.result
