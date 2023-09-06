"""_summary_
"""
# from __future__ import annotations
import ctypes as ct
import typing

from numba import extending, njit, types
from numba.core import cgutils
import numpy as np
from numpy import typing as npt

minpack = ct.CDLL(ct.util.find_library("minpack"))

# minpack = ct.CDLL(
#     "/Users/chris/Documents/PhD/minpack_numba/minpack/libminpack_numba.so"
# )


sig_minpack_func = types.void(
    types.int32,  # n
    types.CPointer(types.double),  # x
    types.CPointer(types.double),  # fvec
    types.CPointer(types.int32),  # iflag
    types.CPointer(types.double),  # udata
)

sig_minpack_func2 = types.void(
    types.int32,  # m
    types.int32,  # n
    types.CPointer(types.double),  # x
    types.CPointer(types.double),  # fvec
    types.CPointer(types.int32),  # iflag
    types.CPointer(types.double),  # udata
)

# TODO check this
sig_minpack_fcn_hybrj = types.void(
    types.int32,  # n
    types.CPointer(types.double),  # x
    types.int32,  # ldfjac
    types.CPointer(types.double),  # fvec
    types.CPointer(types.double),  # fjac
    types.CPointer(types.int32),  # iflag
    types.CPointer(types.double),  # udata
)

sig_minpack_fcn_lmder = types.void(
    types.int32,  # m
    types.int32,  # n
    types.CPointer(types.double),  # x
    types.CPointer(types.double),  # fvec
    types.CPointer(types.double),  # fjac
    types.int32,  # ldfjac
    types.CPointer(types.int32),  # iflag
    types.CPointer(types.double),  # udata
)

# TODO check this
sig_minpack_fcn_lmstr = types.void(
    types.int32,  # m
    types.int32,  # n
    types.CPointer(types.double),  # x
    types.CPointer(types.double),  # fvec
    types.CPointer(types.double),  # fjrow
    types.CPointer(types.int32),  # iflag
    types.CPointer(types.double),  # udata
)


@extending.intrinsic
def _address_as_void_pointer(typingctx, src=None):
    """
    Copied from: https://stackoverflow.com/a/61550054/15456681

    returns a void pointer from a given memory address
    """
    sig = types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)

    return sig, codegen


# TODO hybrd1
_hybrd1 = minpack.minpack_hybrd1
_hybrd1.argtypes = []
_hybrd1.restype = None

# TODO hybrj1
_hybrj1 = minpack.minpack_hybrj1
_hybrj1.argtypes = []
_hybrj1.restype = None


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


@njit
def lmdif1(
    func: typing.Callable,
    x0: npt.ArrayLike,
    m: np.int32,
    n: None | np.int32 = None,
    tol: np.float64 = 1.49012e-8,
    udata: None | npt.ArrayLike = None,
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
    # x = np.atleast_1d(np.asarray(x0)).copy()
    # assert x0.ndim == 1

    tol = np.float64(tol)

    m = np.int32(m)
    if n is None:
        n = np.int32(x0.size)
    else:
        n = np.int32(n)

    fvec = np.zeros(m, dtype=np.float64)

    info = np.zeros(1, dtype=np.int32)
    iwa = np.zeros(n, dtype=np.int32)
    # (m * n + 5 * n + m)+2
    lwa = np.int32((m * n + 5 * n + m))  # m*n+5*n+m

    wa = np.zeros(lwa, dtype=np.float64)

    # if udata is None:
    #     udata = np.zeros(1, dtype=np.float64)
    # else:
    #     udata = np.asarray(udata)

    _lmdif1(
        func,
        m,
        n,
        _address_as_void_pointer(x0.ctypes.data),
        _address_as_void_pointer(fvec.ctypes.data),
        tol,
        info.ctypes.data,
        iwa.ctypes.data,
        wa.ctypes.data,
        lwa,
        _address_as_void_pointer(udata.ctypes.data),
    )
    return x0, fvec, (1 <= info <= 4)[0], info[0]
