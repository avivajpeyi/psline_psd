from typing import Tuple

import numpy as np


def _get_initial_values(
    data: np.ndarray,
    φα: float = 1,
    φβ: float = 1,
    δα: float = 1e-04,
    δβ: float = 1e-04,
) -> Tuple[float, float, float]:
    """
    Return the initial values for the Gibbs sampler

    Parameters
    ----------
    data : np.ndarray
        The TIMESERIES data to be analysed

    φα : float, optional
        The alpha parameter for the prior on φ, by default 1

    φβ : float, optional
        The beta parameter for the prior on φ, by default 1

    δα : float, optional
        The alpha parameter for the prior on δ, by default 1e-04

    δβ : float, optional
        The beta parameter for the prior on δ, by default 1e-04

    Returns
    -------
    Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]
    The initial values for τ, δ, φ, fz, data, omega
    """
    τ = np.var(data) / (2 * np.pi)
    δ = δα / δβ
    φ = φα / (φβ * δ)
    return τ, δ, φ


def _argument_preconditions(
    data: np.ndarray,
    Ntotal: int,
    burnin: int,
    thin: int = 1,
    τα: float = 0.001,
    τβ: float = 0.001,
    φα: float = 1,
    φβ: float = 1,
    δα: float = 1e-04,
    δβ: float = 1e-04,
    k: int = None,
    eqSpacedKnots: bool = False,
    degree: int = 3,
    diffMatrixOrder: int = 2,
    metadata_plotfn: str = None,
    **kwargs,
):
    assert data.shape[0] > 2, "data must be a non-empty np.array"
    assert burnin < Ntotal, "burnin must be less than Ntotal"
    pos_ints = np.array([thin, Ntotal, burnin])
    assert np.all(pos_ints >= 0) and np.all(
        pos_ints % 1 == 0
    ), "thin, Ntotal, burnin must be +ive ints"
    assert Ntotal > 0, "Ntotal must be a positive integer"
    pos_flts = np.array([τα, τβ, φα, φβ, δα, δβ])
    assert np.all(pos_flts > 0), "τ.α, τ.β, φ.α, φ.β, δ.α, δ.β must be +ive"
    assert isinstance(eqSpacedKnots, bool), "eqSpacedKnots must be a boolean"
    assert degree in [0, 1, 2, 3, 4, 5], "degree must be between 0 and 5"
    assert diffMatrixOrder in [0, 1, 2], "diffMatrixOrder must be either 0, 1, or 2"
    assert (
        degree > diffMatrixOrder
    ), "penalty order must be lower than the bspline density degree"
    assert isinstance(metadata_plotfn, str), "metadata_plotdir must be a string"
    assert k >= degree + 2, f"k must be at least degree + 2, (k={k}, degree={degree})"
    assert (
        Ntotal - burnin
    ) / thin > k, (
        f"Must have (Ntotal-burnin)/thin > k, atm:({Ntotal} - {burnin}) / {thin} < {k}"
    )
    assert (
        k - 2 >= diffMatrixOrder
    ), f"diffMatrixOrder ({diffMatrixOrder}) must be lower than or equal to k-2 (k={k})"
