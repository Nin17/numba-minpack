"""_summary_
"""
# pylint: disable=invalid-name, too-many-arguments

# from __future__ import annotations
import ctypes as ct
import typing
import numba as nb
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .utils import aavp

minpack = ct.CDLL(ct.util.find_library("minpack"))

# --------------------------------------------------------------------------- #
#                                    hybrd                                    #
# --------------------------------------------------------------------------- #
# TODO: hybrd

# --------------------------------------------------------------------------- #
#                                    hybrd1                                   #
# --------------------------------------------------------------------------- #
# TODO: hybrd1
_hybrd1 = minpack.minpack_hybrd1
_hybrd1.argtypes = []
_hybrd1.restype = None

# --------------------------------------------------------------------------- #
#                                    hybrj                                    #
# --------------------------------------------------------------------------- #
# TODO: hybrj


# --------------------------------------------------------------------------- #
#                                    hybrj1                                   #
# --------------------------------------------------------------------------- #
# TODO hybrj1
_hybrj1 = minpack.minpack_hybrj1
_hybrj1.argtypes = []
_hybrj1.restype = None


# --------------------------------------------------------------------------- #
#                                    lmdif                                    #
# --------------------------------------------------------------------------- #
# TODO: lmdif

# --------------------------------------------------------------------------- #
#                                    lmdif1                                   #
# --------------------------------------------------------------------------- #

_lmdif1 = minpack.minpack_lmdif1
_lmdif1.argtypes = [
    ct.c_int,  # func  #??? c_void_p
    ct.c_int,  # m
    ct.c_int,  # n
    ct.c_void_p,  # x
    ct.c_void_p,  # fvec
    ct.c_double,  # tol
    ct.c_int,  # info  #??? c_void_p
    ct.c_int,  # iwa   #??? c_void_p
    ct.c_int,  # wa    #??? c_void_p
    ct.c_int,  # lwa
    ct.c_void_p,  # udata
]
_lmdif1.restype = None


@nb.njit
def lmdif1(
    func,
    m: int,
    n: int,
    x: NDArray[np.float64],
    fvec: NDArray[np.float64],
    tol: float,
    info: NDArray[np.int32],
    iwa: NDArray[np.int32],
    wa: NDArray[np.float64],
    lwa: int,
    udata: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """_summary_

    Parameters
    ----------
    func : _type_
        _description_
    m : int
        _description_
    n : int
        _description_
    x : NDArray[np.float64]
        _description_
    fvec : NDArray[np.float64]
        _description_
    tol : float
        _description_
    info : NDArray[np.int32]
        _description_
    iwa : NDArray[np.int32]
        _description_
    wa : NDArray[np.float64]
        _description_
    lwa : int
        _description_
    udata : NDArray[np.float64]
        _description_

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], int]
        _description_
    """

    _lmdif1(
        func,
        m,
        n,
        aavp(x.ctypes.data),
        aavp(fvec.ctypes.data),
        tol,
        info.ctypes.data,
        iwa.ctypes.data,
        wa.ctypes.data,
        lwa,
        aavp(udata.ctypes.data),
    )
    return x, fvec, info[0]


@nb.njit
def lmdif1_wrapper(
    func: typing.Callable,
    x0: ArrayLike,
    m: np.int32,
    n: None | np.int32 = None,
    tol: np.float64 = 1.49012e-8,
    udata: None | ArrayLike = None,
):
    """_summary_

    Parameters
    ----------
    func : typing.Callable
        _description_
    x0 : npt.ArrayLike
        _description_
    m : np.int32
        _description_
    tol : np.float64, optional
        _description_, by default 1.49012e-8
    udata : None | npt.ArrayLike, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    tol = np.float64(tol)

    m = np.int32(m)
    n = np.int32(x0.size)

    fvec = np.zeros(m, dtype=np.float64)

    info = np.zeros(1, dtype=np.int32)
    iwa = np.zeros(n, dtype=np.int32)
    lwa = np.int32((m * n + 5 * n + m))  # m*n+5*n+m
    wa = np.zeros(lwa, dtype=np.float64)

    if udata is None:
        udata = np.zeros(1, dtype=np.float64)

    _lmdif1(
        func,
        m,
        n,
        aavp(x0.ctypes.data),
        aavp(fvec.ctypes.data),
        tol,
        info.ctypes.data,
        iwa.ctypes.data,
        wa.ctypes.data,
        lwa,
        aavp(udata.ctypes.data),
    )
    return x0, fvec, (1 <= info <= 4)[0], info[0]


# --------------------------------------------------------------------------- #
#                                    lmder                                    #
# --------------------------------------------------------------------------- #
# TODO: lmder

# --------------------------------------------------------------------------- #
#                                    lmder1                                   #
# --------------------------------------------------------------------------- #
# TODO: lmder1

_lmder1 = minpack.minpack_lmder1
_lmder1.argtypes = [
    ct.c_void_p,  # fcn
    ct.c_int,  # m
    ct.c_int,  # n
    ct.c_void_p,  # x
    ct.c_void_p,  # fvec
    ct.c_void_p,  # fjac
    ct.c_int,  # ldfjac
    ct.c_double,  # tol
    ct.c_int,  # info
    ct.c_int,  # ipvt # ???
    ct.c_void_p,  # wa
    ct.c_int,  # lwa
    ct.c_void_p,  # udata
]
_lmder1.restype = None


# @nb.njit
# def lmder1_wrapper(func, x, )

# --------------------------------------------------------------------------- #
#                                    lmstr                                    #
# --------------------------------------------------------------------------- #
# TODO: lmstr


# --------------------------------------------------------------------------- #
#                                    lmstr1                                   #
# --------------------------------------------------------------------------- #
# TODO: lmstr1
