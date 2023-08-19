import enum

import numpy as np


class KnotLocatorType(enum.Enum):
    linear = 1
    log = 2
    data_peak = 3


def get_knots(knot_locator_type, knots_kwargs):
    """Returns the knots for the given knot locator type and kwargs"""
    return __get_knot_locator(knot_locator_type)(**knots_kwargs)


def __get_knot_locator(knot_locator_type):
    if knot_locator_type == KnotLocatorType.linear:
        return linearly_spaced_knots
    elif knot_locator_type == KnotLocatorType.log:
        return log_spaced_knots
    elif knot_locator_type == KnotLocatorType.data_peak:
        return data_peak_knots
    else:
        raise ValueError("Unrecognised knot locator type")


def linearly_spaced_knots(n):
    return np.linspace(0, 1, n)


def log_spaced_knots(n):
    return np.logspace(0, 1, n)


def data_peak_knots(data, n):
    """Returns knots at the peaks of the data"""
    return np.sort(np.argpartition(data, -n)[-n:])
