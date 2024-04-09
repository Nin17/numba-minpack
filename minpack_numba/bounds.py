"""Functions to map from unbounded internal variables to bounded external
variables and vice versa.

Follows the implementation in Minuit/Minuit2 and in lmfit.
https://lmfit.github.io/lmfit-py/bounds.html
"""
# pylint: disable=invalid-name, unused-argument

from __future__ import annotations

from numba import njit
from numba.types import none
from numpy import arcsin, float64, sin, sqrt

from .utils import generated_jit

# from numpy.typing import NDArray


@generated_jit(nopython=True, fastmath=True)
def _ext2in(
    x: float64,
    lb: None | float64 = None,
    ub: None | float64 = None,
) -> float64:
    _lb = lb is not none
    _ub = ub is not none

    # TODO allow iterables for lb and ub

    if _lb and not _ub:
        return lambda x, lb, ub: sqrt((x - lb + 1) ** 2 - 1)
    if not _lb and _ub:
        return lambda x, lb, ub: sqrt((ub - x + 1) ** 2 - 1)
    if _lb and _ub:
        return lambda x, lb, ub: arcsin(2 * (x - lb) / (ub - lb) - 1)
    return lambda x, lb, ub: x


@njit(fastmath=True)
def ext2in(
    x: float64,
    lb: None | float64 = None,
    ub: None | float64 = None,
) -> float64:
    # TODO docstring
    """Maps from a bounded external variable to an unbounded internal variable.
    If only the lower bound is given, the mapping is:
    ...
    If only the upper bound is given, the mapping is:
    ...
    If both bounds are given, the mapping is:
    ...
    If no bounds are given, the input x is unchanged.

    Parameters
    ----------
    x : float64
        _description_
    lb : None | float64, optional
        _description_, by default None
    ub : None | float64, optional
        _description_, by default None

    Returns
    -------
    float64
        _description_

    """
    return _ext2in(x, lb, ub)


@generated_jit(nopython=True, fastmath=True)
def _in2ext(
    x: float64,
    lb: None | float64 = None,
    ub: None | float64 = None,
) -> float64:
    _lb = lb is not none
    _ub = ub is not none

    # TODO allow iterables for lb and ub

    if _lb and not _ub:
        return lambda x, lb, ub: lb - 1.0 + sqrt(x * x + 1.0)
    if not _lb and _ub:
        return lambda x, lb, ub: ub + 1 - sqrt(x * x + 1)
    if _lb and _ub:
        return lambda x, lb, ub: lb + (ub - lb) / 2 * (sin(x) + 1)
    return lambda x, lb, ub: x


@njit(fastmath=True)
def in2ext(
    x: float64,
    lb: None | float64 = None,
    ub: None | float64 = None,
) -> float64:
    # TODO docstring
    r"""Maps from an unbounded internal variable to a bounded external variable.
    If only the lower bound is given, the mapping is:
    .. math::
        P_{bounded} = {minimum} - 1 + \sqrt{P_{internal}^2 + 1}
    If only the upper bound is given, the mapping is:
    .. math::
        P_{bounded}  = {maximum} + 1 - \sqrt{P_{internal}^2 + 1}
    If both bounds are given, the mapping is:
    .. math::
        P_{bounded} = {minimum} + \left(\sin\left(P_{internal}\right) +
        1\right) \frac{\left( {maximum} - {minimum} \right)}{2}
    If no bounds are given, the input x is unchanged.

    Parameters
    ----------
    x : float64
        _description_
    lb : None | float64, optional
        _description_, by default None
    ub : None | float64, optional
        _description_, by default None

    Returns
    -------
    float64
        _description_

    """
    return _in2ext(x, lb, ub)


import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.njit(fastmath=True)
def ext2in_lb(x: NDArray[np.float64], lb: np.float64) -> NDArray[np.float64]:
    r"""Maps from a bounded external variable to an unbounded internal variable
    given a lower bound with the mapping:

    .. math::
        P_{internal} = \sqrt{\left(P_{bounded} - {minimum} + 1 \right)^2 - 1}

    Parameters
    ----------
    x : NDArray[np.float64]
        external bounded variable
    lb : np.float64
        lower bound

    Returns
    -------
    NDArray[np.float64]
        internal unbounded variable

    """
    return np.sqrt((x - lb + 1) ** 2 - 1)


@nb.njit(fastmath=True)
def in2ext_lb(x: NDArray[np.float64], lb: np.float64) -> NDArray[np.float64]:
    r"""Maps from an unbounded internal variable to a bounded external variable
    given a lower bound with the mapping:

    .. math::
        P_{bounded} = {minimum} - 1 + \sqrt{P_{internal}^2 + 1}

    Parameters
    ----------
    x : NDArray[np.float64]
        internal unbounded variable
    lb : np.float64
        lower bound

    Returns
    -------
    NDArray[np.float64]
        external bounded variable

    """
    return lb - 1.0 + np.sqrt(x * x + 1.0)


@nb.njit(fastmath=True)
def ext2in_ub(x: NDArray[np.float64], ub: np.float64) -> NDArray[np.float64]:
    r"""Maps from a bounded external variable to an unbounded internal variable
    given an upper bound with the mapping:

    .. math::
        P_{internal} = \sqrt{\left({maximum} - P_{bounded} + 1 \right)^2 - 1}

    Parameters
    ----------
    x : NDArray[np.float64]
        external bounded variable
    ub : np.float64
        upper bound

    Returns
    -------
    NDArray[np.float64]
        internal unbounded variable

    """
    return np.sqrt((ub - x + 1) ** 2 - 1)


@nb.njit(fastmath=True)
def in2ext_ub(x: NDArray[np.float64], ub: np.float64) -> NDArray[np.float64]:
    r"""Maps from an unbounded internal variable to a bounded external variable
    given an upper bound with the mapping:

    .. math::
        P_{bounded}  = {maximum} + 1 - \sqrt{P_{internal}^2 + 1}

    Parameters
    ----------
    x : NDArray[np.float64]
        internal unbounded variable
    ub : np.float64
        upper bound

    Returns
    -------
    NDArray[np.float64]
        external bounded variable

    """
    return ub + 1 - np.sqrt(x * x + 1)


@nb.njit(fastmath=True)
def ext2in_lbub(
    x: NDArray[np.float64],
    lb: np.float64,
    ub: np.float64,
) -> NDArray[np.float64]:
    r"""Maps from a bounded external variable to an unbounded internal variable
    given a lower and upper bound with the mapping:

    .. math::
        P_{internal} = \arcsin \left(\frac{2 \left(P_{bounded} -
        {mininum}\right)}{\left({maximum} - {minimum}\right)} - 1 \right)


    Parameters
    ----------
    x : NDArray[np.float64]
        bounded external variable
    lb : np.float64
        lower bound
    ub : np.float64
        upper bound

    Returns
    -------
    NDArray[np.float64]
        unbounded internal variable

    """
    return np.arcsin(2 * (x - lb) / (ub - lb) - 1)


@nb.njit(fastmath=True)
def in2ext_lbub(
    x: NDArray[np.float64],
    lb: np.float64,
    ub: np.float64,
) -> NDArray[np.float64]:
    r"""Maps from an unbounded internal variable to a bounded external variable
    given a lower and upper bound with the mapping:

    .. math::
        P_{bounded} = {minimum} + \left(\sin\left(P_{internal}\right) +
        1\right) \frac{\left( {maximum} - {minimum} \right)}{2}

    Parameters
    ----------
    x : NDArray[np.float64]
        unbounded internal variable
    lb : np.float64
        lower bound
    ub : np.float64
        upper bound

    Returns
    -------
    NDArray[np.float64]
        bounded external variable

    """
    return lb + (ub - lb) / 2 * (np.sin(x) + 1)
