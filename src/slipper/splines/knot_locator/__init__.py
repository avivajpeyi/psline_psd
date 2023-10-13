import inspect
from typing import Dict, Union

import numpy as np

from slipper.logger import logger

from .methods import _KNOT_LOCATOR_FUNC_DICT, KnotLocatorType


def knot_locator(
    knot_locator_type: Union[KnotLocatorType, str], **knots_kwargs
) -> np.ndarray:
    """Returns the knots for the given knot locator type and kwargs"""
    knot_loc_func = _KNOT_LOCATOR_FUNC_DICT[knot_locator_type]
    expected_args = inspect.getfullargspec(knot_loc_func).args
    kwargs = {k: knots_kwargs.get(k, None) for k in expected_args}

    # if any of the args are None, raise an error
    if any([v is None for v in kwargs.values()]):
        raise ValueError(
            f"Missing arguments for {knot_locator_type}:\n{kwargs}"
        )

    knots = knot_loc_func(**kwargs)

    if knots[0] != 0 or knots[-1] != 1:
        logger.warning(
            "Knots must start at 0 and end at 1. "
            "Manually changing the first and last knots to 0 and 1 respectively."
            "Original knots:\n"
            f"{knots}"
        )
        knots[0] = 0
        knots[-1] = 1
    return knots
