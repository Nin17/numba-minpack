"""_summary_
"""
import numba as nb


@nb.extending.intrinsic
def address_as_void_pointer(typingctx, src=None):
    """
    Copied from: https://stackoverflow.com/a/61550054/15456681

    returns a void pointer from a given memory address
    """
    sig = nb.types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], nb.core.cgutils.voidptr_t)

    return sig, codegen


aavp = address_as_void_pointer
