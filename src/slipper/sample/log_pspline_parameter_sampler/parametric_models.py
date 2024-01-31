import numpy as np


# Model 1: PSD of sum of test mass noise and Readout noise
def instru_noise(f):
    c = 299792458  # ms^-1
    L = 2.5e9  # m
    s1 = (
        ((3) ** 2)
        * (1 + (4e-4 / f) ** 2)
        * (1 + (f / 8e-3) ** 4)
        * ((2 * np.pi * f / c) ** 2)
        / (2 * np.pi * f) ** 4
    )
    s2 = ((15) ** 2) * (1 + (2e-3 / f) ** 4) * ((2 * np.pi * f / c) ** 2)
    noi = (
        16
        * ((np.sin(2 * np.pi * f * L / c)) ** 2)
        * (s2 + (3 + np.cos(4 * np.pi * f * L / c)) * s1)
    )
    # res=0.3*((2*np.pi*f*L/c)**2)/(1+0.6*(2*np.pi*f*L/c))# residuals: for estimating sensitivity curve
    return noi
