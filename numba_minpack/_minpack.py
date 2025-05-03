"""Signatures for the minpack functions."""

from __future__ import annotations

from ctypes.util import find_library

from llvmlite import binding
from numba import types

from .utils import get_extension_path

__all__ = [
    "Minpack",
]


_minpack_path = get_extension_path("libminpack") or find_library("minpack")
if not _minpack_path:
    msg = "minpack library not found"
    raise ImportError(msg)
binding.load_library_permanently(_minpack_path)


class Minpack:
    @staticmethod
    def chkder() -> types.ExternalFunction:
        sig = types.void(
            types.int32,  # m
            types.int32,  # n
            types.CPointer(types.float64),  # *x
            types.CPointer(types.float64),  # *fvec
            types.CPointer(types.float64),  # *fjac
            types.int32,  # ldfjac
            types.CPointer(types.float64),  # *xp
            types.CPointer(types.float64),  # *fvecp
            types.int32,  # mode
            types.CPointer(types.float64),  # *err
        )
        return types.ExternalFunction("minpack_chkder", sig)

    @staticmethod
    def dpmpar() -> types.ExternalFunction:
        sig = types.float64(types.int32)  # n
        return types.ExternalFunction("minpack_dpmpar", sig)

    @staticmethod
    def hybrd() -> types.ExternalFunction:
        sig = types.void(
            types.voidptr,  # fcn
            types.int32,  # n
            types.CPointer(types.float64),  # *x
            types.CPointer(types.float64),  # *fvec
            types.float64,  # xtol
            types.int32,  # maxfev
            types.int32,  # ml
            types.int32,  # mu
            types.float64,  # epsfcn
            types.CPointer(types.float64),  # *diag
            types.int32,  # mode
            types.float64,  # factor
            types.int32,  # nprint
            types.CPointer(types.int32),  # *info
            types.CPointer(types.int32),  # *nfev
            types.CPointer(types.float64),  # *fjac
            types.int32,  # ldfjac
            types.CPointer(types.float64),  # *r
            types.int32,  # lr
            types.CPointer(types.float64),  # *qtf
            types.CPointer(types.float64),  # *wa1
            types.CPointer(types.float64),  # *wa2
            types.CPointer(types.float64),  # *wa3
            types.CPointer(types.float64),  # *wa4
            types.voidptr,  # *udata
        )
        return types.ExternalFunction("minpack_hybrd", sig)

    @staticmethod
    def hybrd1() -> types.ExternalFunction:
        sig = types.void(
            types.voidptr,  # fcn
            types.int32,  # n
            types.CPointer(types.float64),  # *x
            types.CPointer(types.float64),  # *fvec
            types.float64,  # ftol
            types.CPointer(types.int32),  # *info
            types.CPointer(types.float64),  # *wa
            types.int32,  # lwa
            types.voidptr,  # *udata
        )
        return types.ExternalFunction("minpack_hybrd1", sig)

    @staticmethod
    def hybrj() -> types.ExternalFunction:
        sig = types.void(
            types.voidptr,  # fcn
            types.int32,  # n
            types.CPointer(types.float64),  # *x
            types.CPointer(types.float64),  # *fvec
            types.CPointer(types.float64),  # *fjac
            types.int32,  # ldfjac
            types.float64,  # xtol
            types.int32,  # maxfev
            types.CPointer(types.float64),  # *diag
            types.int32,  # mode
            types.float64,  # factor
            types.int32,  # nprint
            types.CPointer(types.int32),  # *info
            types.CPointer(types.int32),  # *nfev
            types.CPointer(types.int32),  # *njev
            types.CPointer(types.float64),  # *r
            types.int32,  # lr
            types.CPointer(types.float64),  # *qtf
            types.CPointer(types.float64),  # *wa1
            types.CPointer(types.float64),  # *wa2
            types.CPointer(types.float64),  # *wa3
            types.CPointer(types.float64),  # *wa4
            types.voidptr,  # *udata
        )
        return types.ExternalFunction("minpack_hybrj", sig)

    @staticmethod
    def hybrj1() -> types.ExternalFunction:
        sig = types.void(
            types.voidptr,  # fcn
            types.int32,  # n
            types.CPointer(types.float64),  # *x
            types.CPointer(types.float64),  # *fvec
            types.CPointer(types.float64),  # *fjac
            types.int32,  # ldfjac
            types.float64,  # ftol
            types.CPointer(types.int32),  # *info
            types.CPointer(types.float64),  # *wa
            types.int32,  # lwa
            types.voidptr,  # *udata
        )
        return types.ExternalFunction("minpack_hybrj1", sig)

    @staticmethod
    def lmdif() -> types.ExternalFunction:
        sig = types.void(
            types.voidptr,  # fcn
            types.int32,  # m
            types.int32,  # n
            types.CPointer(types.float64),  # *x
            types.CPointer(types.float64),  # *fvec
            types.float64,  # ftol
            types.float64,  # xtol
            types.float64,  # gtol
            types.int32,  # maxfev
            types.float64,  # epsfcn
            types.CPointer(types.float64),  # *diag
            types.int32,  # mode
            types.float64,  # factor
            types.int32,  # nprint
            types.CPointer(types.int32),  # *info
            types.CPointer(types.int32),  # *nfev
            types.CPointer(types.float64),  # *fjac
            types.int32,  # ldfjac
            types.CPointer(types.int32),  # *ipvt
            types.CPointer(types.float64),  # *qtf
            types.CPointer(types.float64),  # *wa1
            types.CPointer(types.float64),  # *wa2
            types.CPointer(types.float64),  # *wa3
            types.CPointer(types.float64),  # *wa4
            types.voidptr,  # *udata
        )
        return types.ExternalFunction("minpack_lmdif", sig)

    @staticmethod
    def lmdif1() -> types.ExternalFunction:
        sig = types.void(
            types.voidptr,  # fcn
            types.int32,  # m
            types.int32,  # n
            types.CPointer(types.float64),  # *x
            types.CPointer(types.float64),  # *fvec
            types.float64,  # ftol
            types.CPointer(types.int32),  # *info
            types.CPointer(types.int32),  # *iwa
            types.CPointer(types.float64),  # *wa
            types.int32,  # lwa
            types.voidptr,  # *udata
        )
        return types.ExternalFunction("minpack_lmdif1", sig)

    @staticmethod
    def lmder() -> types.ExternalFunction:
        sig = types.void(
            types.voidptr,  # fcn
            types.int32,  # m
            types.int32,  # n
            types.CPointer(types.float64),  # *x
            types.CPointer(types.float64),  # *fvec
            types.CPointer(types.float64),  # *fjac
            types.int32,  # ldfjac
            types.float64,  # ftol
            types.float64,  # xtol
            types.float64,  # gtol
            types.int32,  # maxfev
            types.CPointer(types.float64),  # *diag
            types.int32,  # mode
            types.float64,  # factor
            types.int32,  # nprint
            types.CPointer(types.int32),  # *info
            types.CPointer(types.int32),  # *nfev
            types.CPointer(types.int32),  # *njev
            types.CPointer(types.int32),  # *ipvt
            types.CPointer(types.float64),  # *qtf
            types.CPointer(types.float64),  # *wa1
            types.CPointer(types.float64),  # *wa2
            types.CPointer(types.float64),  # *wa3
            types.CPointer(types.float64),  # *wa4
            types.voidptr,  # *udata
        )
        return types.ExternalFunction("minpack_lmder", sig)

    @staticmethod
    def lmder1() -> types.ExternalFunction:
        sig = types.void(
            types.voidptr,  # fcn
            types.int32,  # m
            types.int32,  # n
            types.CPointer(types.float64),  # *x
            types.CPointer(types.float64),  # *fvec
            types.CPointer(types.float64),  # *fjac
            types.int32,  # ldfjac
            types.float64,  # ftol
            types.CPointer(types.int32),  # *info
            types.CPointer(types.int32),  # *ipvt
            types.CPointer(types.float64),  # *wa
            types.int32,  # lwa
            types.voidptr,  # *udata
        )
        return types.ExternalFunction("minpack_lmder1", sig)

    @staticmethod
    def lmstr() -> types.ExternalFunction:
        sig = types.void(
            types.voidptr,  # fcn
            types.int32,  # m
            types.int32,  # n
            types.CPointer(types.float64),  # *x
            types.CPointer(types.float64),  # *fvec
            types.CPointer(types.float64),  # *fjac
            types.int32,  # ldfjac
            types.float64,  # ftol
            types.float64,  # xtol
            types.float64,  # gtol
            types.int32,  # maxfev
            types.CPointer(types.float64),  # *diag
            types.int32,  # mode
            types.float64,  # factor
            types.int32,  # nprint
            types.CPointer(types.int32),  # *info
            types.CPointer(types.int32),  # *nfev
            types.CPointer(types.int32),  # *njev
            types.CPointer(types.int32),  # *ipvt
            types.CPointer(types.float64),  # *qtf
            types.CPointer(types.float64),  # *wa1
            types.CPointer(types.float64),  # *wa2
            types.CPointer(types.float64),  # *wa3
            types.CPointer(types.float64),  # *wa4
            types.voidptr,  # *udata
        )
        return types.ExternalFunction("minpack_lmstr", sig)

    @staticmethod
    def lmstr1() -> types.ExternalFunction:
        sig = types.void(
            types.voidptr,  # fcn
            types.int32,  # m
            types.int32,  # n
            types.CPointer(types.float64),  # *x
            types.CPointer(types.float64),  # *fvec
            types.CPointer(types.float64),  # *fjac
            types.int32,  # ldfjac
            types.float64,  # ftol
            types.CPointer(types.int32),  # *info
            types.CPointer(types.int32),  # *ipvt
            types.CPointer(types.float64),  # *wa
            types.int32,  # lwa
            types.voidptr,  # *udata
        )
        return types.ExternalFunction("minpack_lmstr1", sig)
