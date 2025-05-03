"""Signatures for the functions passed to the minpack functions."""

from __future__ import annotations

__all__ = [
    "hybrd_sig",
    "hybrj_sig",
    "lmdif_sig",
    "lmder_sig",
    "lmstr_sig",
    "MinpackSignature",
]

from typing import TYPE_CHECKING

from numba import types

if TYPE_CHECKING:
    from numba.core.typing import Signature

_ptr_double = types.CPointer(types.float64)
_ptr_int = types.CPointer(types.int32)


class MinpackSignature:
    """Signatures for the functions passed to the minpack functions."""

    @staticmethod
    def hybrd(udata_type: types.Type = types.voidptr) -> Signature:
        """Signature for the function passed to the `hybrd` function.

        Parameters
        ----------
        udata_type : types.Type, optional
            The type of the user data, by default types.voidptr

        Returns
        -------
        Signature
            The signature of the function

        """
        return types.void(
            types.int32,  # n
            _ptr_double,  # *x
            _ptr_double,  # *fvec
            _ptr_int,  # *iflag
            udata_type,  # udata
        )

    @staticmethod
    def hybrj(udata_type: types.Type = types.voidptr) -> Signature:
        """Signature for the function passed to the `hybrj` function.

        Parameters
        ----------
        udata_type : types.Type, optional
            The type of the user data, by default types.voidptr

        Returns
        -------
        Signature
            The signature of the function

        """
        return types.void(
            types.int32,  # n
            _ptr_double,  # *x
            _ptr_double,  # *fvec
            _ptr_double,  # *fjac
            types.int32,  # ldfjac
            _ptr_int,  # *iflag
            udata_type,  # udata
        )

    @staticmethod
    def lmdif(udata_type: types.Type = types.voidptr) -> Signature:
        """Signature for the function passed to the `lmdif` function.

        Parameters
        ----------
        udata_type : types.Type, optional
            The type of the user data, by default types.voidptr

        Returns
        -------
        Signature
            The signature of the function

        """
        return types.void(
            types.int32,  # m
            types.int32,  # n
            _ptr_double,  # *x
            _ptr_double,  # *fvec
            _ptr_int,  # *iflag
            udata_type,  # udata
        )

    @staticmethod
    def lmder(udata_type: types.Type = types.voidptr) -> Signature:
        """Signature for the function passed to the `lmder` function.

        Parameters
        ----------
        udata_type : types.Type, optional
            The type of the user data, by default types.voidptr

        Returns
        -------
        Signature
            The signature of the function

        """
        return types.void(
            types.int32,  # m
            types.int32,  # n
            _ptr_double,  # *x
            _ptr_double,  # *fvec
            _ptr_double,  # *fjac
            types.int32,  # ldfjac
            _ptr_int,  # *iflag
            udata_type,  # udata
        )

    @staticmethod
    def lmstr(udata_type: types.Type = types.voidptr) -> Signature:
        """Signature for the function passed to the `lmstr` function.

        Parameters
        ----------
        udata_type : types.Type, optional
            The type of the user data, by default types.voidptr

        Returns
        -------
        Signature
            The signature of the function

        """
        return types.void(
            types.int32,  # m
            types.int32,  # n
            _ptr_double,  # *x
            _ptr_double,  # *fvec
            _ptr_double,  # *fjrow
            _ptr_int,  # *iflag
            udata_type,  # udata
        )


# func
hybrd_sig = MinpackSignature.hybrd()
"""
(n: int32, x: float64*, fvec: float64*, iflag: int32*, udata: void*) -> none
"""

# fcn_hybrj
hybrj_sig = MinpackSignature.hybrj()
"""
(n: int32, x: float64*, fvec: float64*, fjac: float64*, ldfjac: int32, iflag: int32*,
    udata: void*) -> none
"""

# func2
lmdif_sig = MinpackSignature.lmdif()
"""
(m: int32, n: int32, x: float64*, fvec: float64*, iflag: int32*, udata: void*)
    -> none
"""

# fcn_lmder
lmder_sig = MinpackSignature.lmder()
"""
(m: int32, n: int32, x: float64*, fvec: float64*, fjac: float64*, ldfjac: int32,
    iflag: int32*, udata: void*) -> none
"""

# fcn_lmstr
lmstr_sig = MinpackSignature.lmstr()
"""
(m: int32, n: int32, x: float64*, fvec: float64*, fjrow: float64*, iflag: int32*,
    udata: void*) -> none
"""
