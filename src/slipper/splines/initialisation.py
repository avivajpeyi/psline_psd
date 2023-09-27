"""Initialisation functions for the penalised B-splines."""
from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d

from .p_splines import PSplines
import scipy

def knot_locator(
    data: np.ndarray, k: int,degree: int, eqSpaced: bool = False,
    nfreqbin = [5e-4, 1e-3, 1e-2, 1e-1],wfreqbin = [2.5, 10, 25, 60, 2.5],
    fs =1/0.75
) -> np.array:
    """Determines the knot locations for a B-spline basis of degree `degree` and `k` knots.

    Returns
    -------
    knots : np.ndarray of shape (k - degree + 1,)
    (The x-positions of the knots)

    """
    #This is to be in the function definition
    
    
    
    #Equal spaced knots

    K = k - degree + 1 #num of internal knots
    if eqSpaced:
        knots = np.linspace(0, 1, num=k - degree + 1)
        return knots

    #periodogram (By Patricio)
    pdgrm=np.log(data)
    N=len(pdgrm)
    #data = data - np.mean(data)
    #FZ = np.fft.fft(data)  # FFT data to frequency domain. NOTE: Must be mean-centered.
    #pdgrm = np.log(np.abs(FZ) ** 2)# Periodogram: NOTE: the length is n here.
    #N=len(pdgrm)

    #Knots placement based on log periodogram (Patricio code) This is when nfreqbin is an array
    if wfreqbin is None:
        n_wfreqbin = len(nfreqbin) + 1
        wfreqbin = np.ones(n_wfreqbin) / n_wfreqbin
    else:
        if (len(nfreqbin) + 1) != len(wfreqbin):
            return('length of nfreqbin is incorrect')
        wfreqbin = wfreqbin / np.sum(wfreqbin)
        n_wfreqbin = len(wfreqbin)

    nfreqbin = np.sort(nfreqbin)
    eqval = np.concatenate(([0], nfreqbin / (fs/2), [1]))#Interval [0,1]
    eqval = np.column_stack((eqval[:-1], eqval[1:])) # Each row represents the bin
    j = np.linspace(0, 1, num=N)
    s = np.arange(1, N + 1)
    index = []

    for i in range(n_wfreqbin):
        cond = (j >= eqval[i, 0]) & (j <= eqval[i, 1])
        index.append((np.min(s[cond]), np.max(s[cond])))

    Nindex = len(index)

    K = K - 2# to include 0 and 1 in the knot vector
    kvec = np.round(K * np.array(wfreqbin))
    kvec=kvec.astype(int)

    while np.sum(kvec) > K:
        kvec[np.argmax(kvec)] = np.max(kvec) - 1

    while np.sum(kvec) < K:
        kvec[np.argmin(kvec)] = np.min(kvec) + 1

    knots = []

    for i in range(Nindex):
        aux = pdgrm[index[i][0]:index[i][1]]

        # aux = np.sqrt(aux) in case using pdgrm
        dens = np.abs(aux - np.mean(aux)) / np.std(aux)

        Naux = len(aux)

        dens = dens / np.sum(dens)
        cumf = np.cumsum(dens)
        x = np.linspace(eqval[i][0], eqval[i][1], num=Naux)

        # Distribution function
        df =interp1d(x, cumf, bounds_error=False, fill_value=(0, 1))
        dfvec = df(x)

        invDf = interp1d(dfvec, x, kind='linear', fill_value=(x[0], x[-1]), bounds_error=False)

        v = np.linspace(0, 1, num=kvec[i] + 2)
        v = v[1:-1]

        knots = np.concatenate((knots, invDf(v)))

        # Inverse distribution
        #v = np.linspace(0, 1, num=int(kvec[i] + 2))[1:-1]
        
        #invDf = np.interp(v, np.linspace(eqval[i, 0], eqval[i, 1], num=Naux), df)

        #knots.extend(invDf)

    knots = np.concatenate(([0], knots, [1]))
    return knots


def _get_initial_spline_data(
    data: np.ndarray, k: int, degree: int, diffMatrixOrder: int, eqSpaced: bool
) -> Tuple[np.ndarray, np.ndarray, PSplines]:

    knots = knot_locator(data, k, degree, eqSpaced)
    psplines = PSplines(knots=knots, degree=degree, diffMatrixOrder=diffMatrixOrder)
    V = psplines.guess_initial_v(data)
    return V, knots, psplines
