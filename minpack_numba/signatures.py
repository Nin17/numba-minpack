"""_summary_
"""

import numba as nb

sig_minpack_func = nb.types.void(
    nb.types.int32,  # n
    nb.types.CPointer(nb.types.double),  # x
    nb.types.CPointer(nb.types.double),  # fvec
    nb.types.CPointer(nb.types.int32),  # iflag
    nb.types.CPointer(nb.types.double),  # udata
)

sig_minpack_func2 = nb.types.void(
    nb.types.int32,  # m
    nb.types.int32,  # n
    nb.types.CPointer(nb.types.double),  # x
    nb.types.CPointer(nb.types.double),  # fvec
    nb.types.CPointer(nb.types.int32),  # iflag
    nb.types.CPointer(nb.types.double),  # udata
)

# TODO check this
sig_minpack_fcn_hybrj = nb.types.void(
    nb.types.int32,  # n
    nb.types.CPointer(nb.types.double),  # x
    nb.types.int32,  # ldfjac
    nb.types.CPointer(nb.types.double),  # fvec
    nb.types.CPointer(nb.types.double),  # fjac
    nb.types.CPointer(nb.types.int32),  # iflag
    nb.types.CPointer(nb.types.double),  # udata
)

sig_minpack_fcn_lmder = nb.types.void(
    nb.types.int32,  # m
    nb.types.int32,  # n
    nb.types.CPointer(nb.types.double),  # x
    nb.types.CPointer(nb.types.double),  # fvec
    nb.types.CPointer(nb.types.double),  # fjac
    nb.types.int32,  # ldfjac
    nb.types.CPointer(nb.types.int32),  # iflag
    nb.types.CPointer(nb.types.double),  # udata
)

# TODO check this
sig_minpack_fcn_lmstr = nb.types.void(
    nb.types.int32,  # m
    nb.types.int32,  # n
    nb.types.CPointer(nb.types.double),  # x
    nb.types.CPointer(nb.types.double),  # fvec
    nb.types.CPointer(nb.types.double),  # fjrow
    nb.types.CPointer(nb.types.int32),  # iflag
    nb.types.CPointer(nb.types.double),  # udata
)
