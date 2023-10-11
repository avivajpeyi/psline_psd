import inspect
from typing import Dict, Union

import numpy as np

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

    return knot_loc_func(**kwargs)
