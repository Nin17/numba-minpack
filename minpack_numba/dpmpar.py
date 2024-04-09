"""Wrapper of the `dpmpar` function."""

from __future__ import annotations

from numba import extending, njit
from numpy import float64, int32

from ._minpack import Minpack

# -------------------------------------- dpmpar -------------------------------------- #


def _dpmpar(i):
    raise NotImplementedError


@extending.overload(_dpmpar)
def _dpmpar_overload(i):
    dpmpar_external = Minpack.dpmpar()

    def impl(i):
        return dpmpar_external(int32(i))

    return impl


@njit
def dpmpar(i: int32) -> float64:
    """Double precision machine parameters.

    Parameters
    ----------
    i : int32
        1: the machine precision
        2: the smallest magnitude
        3: the largest magnitude

    Returns
    -------
    float64
        machine parameter

    """
    return _dpmpar(i)
