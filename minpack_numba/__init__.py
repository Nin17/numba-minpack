"""A numba wrapper for the minpack library."""

__version__ = "0.1.0"

from .chkder import chkder
from .dpmpar import dpmpar
from .hybrd import hybrd, hybrd1, hybrd1_, hybrd_
from .hybrj import hybrj, hybrj1, hybrj1_, hybrj_
from .lmder import lmder, lmder1, lmder1_, lmder_
from .lmdif import lmdif, lmdif1, lmdif1_, lmdif_
from .lmstr import lmstr, lmstr1, lmstr1_, lmstr_
from .signatures import (
    MinpackSignature,
    hybrd_sig,
    hybrj_sig,
    lmder_sig,
    lmdif_sig,
    lmstr_sig,
)
from .utils import address_as_void_pointer, check_cfunc, ptr_from_val, val_from_ptr

__all__ = [
    # Utils
    "address_as_void_pointer",
    "check_cfunc",
    "ptr_from_val",
    "val_from_ptr",
    # Signature
    "hybrd_sig",
    "hybrj_sig",
    "lmdif_sig",
    "lmder_sig",
    "lmstr_sig",
    "MinpackSignature",
    # Functions
    "chkder",
    "dpmpar",
    "hybrd",
    "hybrd_",
    "hybrd1",
    "hybrd1_",
    "hybrj",
    "hybrj_",
    "hybrj1",
    "hybrj1_",
    "lmder",
    "lmder_",
    "lmder1",
    "lmder1_",
    "lmdif",
    "lmdif_",
    "lmdif1",
    "lmdif1_",
    "lmstr1",
    "lmstr1_",
    "lmstr",
    "lmstr_",
]
