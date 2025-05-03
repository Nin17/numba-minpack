"""Wrappers of the `hybrj` and `hybrj1` functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import extending, njit, types
from numpy import empty, int32, ones

from ._minpack import Minpack
from .utils import _check_dtype, ptr_from_val, val_from_ptr

if TYPE_CHECKING:
    from numpy import float64, int64
    from numpy.typing import NDArray

# TODO(nin17): check_dtype & type hints

# -------------------------------------- hybrj1 -------------------------------------- #


def _hybrj1(fcn, n, x, fvec, fjac, ldfjac, tol, info, wa, lwa, udata):
    raise NotImplementedError


@extending.overload(_hybrj1)
def _hybrj1_overload(fcn, n, x, fvec, fjac, ldfjac, tol, info, wa, lwa, udata):
    _check_dtype((fvec, fjac, wa), x.dtype)
    hybrj1_external = Minpack.hybrj1()

    @extending.register_jitable
    def impl(fcn, n, x, fvec, fjac, ldfjac, tol, info, wa, lwa, udata):
        hybrj1_external(
            fcn,
            n,
            x.ctypes,
            fvec.ctypes,
            fjac.ctypes,
            ldfjac,
            tol,
            info,
            wa.ctypes,
            lwa,
            udata,
        )
        return x, fvec, fjac, val_from_ptr(info)

    if isinstance(udata, types.Array):
        return lambda fcn, n, x, fvec, fjac, ldfjac, tol, info, wa, lwa, udata: impl(
            fcn,
            n,
            x,
            fvec,
            fjac,
            ldfjac,
            tol,
            info,
            wa,
            lwa,
            udata.ctypes,
        )
    if udata is not types.none:
        return impl
    return lambda fcn, n, x, fvec, fjac, ldfjac, tol, info, wa, lwa, udata: impl(
        fcn,
        n,
        x,
        fvec,
        fjac,
        ldfjac,
        tol,
        info,
        wa,
        lwa,
        0,
    )


@njit
def hybrj1_(fcn, n, x, fvec, fjac, ldfjac, tol, info, wa, lwa, udata=None):
    return _hybrj1(fcn, n, x, fvec, fjac, ldfjac, tol, info, wa, lwa, udata)


@njit
def hybrj1(fcn, x, tol=None, udata=None):
    tol = tol or 1.49012e-8
    n = int32(x.size)
    lwa = int32((n * (3 * n + 13)) // 2)
    fvec = empty(n, dtype=x.dtype)
    fjac = empty((n, n), dtype=x.dtype)
    wa = empty(lwa, dtype=x.dtype)
    infoptr = ptr_from_val(int32(0))
    return _hybrj1(fcn, n, x.copy(), fvec, fjac, n, tol, infoptr, wa, lwa, udata)


# --------------------------------------- hybrj -------------------------------------- #


def _hybrj(
    fcn,
    n,
    x,
    fvec,
    fjac,
    ldfjac,
    xtol,
    maxfev,
    diag,
    mode,
    factor,
    nprint,
    info,
    nfev,
    njev,
    r,
    lr,
    qtf,
    wa1,
    wa2,
    wa3,
    wa4,
    udata,
):
    raise NotImplementedError


@extending.overload(_hybrj)
def _hybrj_overload(
    fcn,
    n,
    x,
    fvec,
    fjac,
    ldfjac,
    xtol,
    maxfev,
    diag,
    mode,
    factor,
    nprint,
    info,
    nfev,
    njev,
    r,
    lr,
    qtf,
    wa1,
    wa2,
    wa3,
    wa4,
    udata,
):
    _check_dtype((fvec, fjac, diag, r, qtf, wa1, wa2, wa3, wa4), x.dtype)
    hybrj_external = Minpack.hybrj()

    @extending.register_jitable
    def impl(
        fcn,
        n,
        x,
        fvec,
        fjac,
        ldfjac,
        xtol,
        maxfev,
        diag,
        mode,
        factor,
        nprint,
        info,
        nfev,
        njev,
        r,
        lr,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata,
    ):
        hybrj_external(
            fcn,
            n,
            x.ctypes,
            fvec.ctypes,
            fjac.ctypes,
            ldfjac,
            xtol,
            maxfev,
            diag.ctypes,
            mode,
            factor,
            nprint,
            info,
            nfev,
            njev,
            r.ctypes,
            lr,
            qtf.ctypes,
            wa1.ctypes,
            wa2.ctypes,
            wa3.ctypes,
            wa4.ctypes,
            udata,
        )

        return (
            x,
            fvec,
            fjac,
            r,
            qtf,
            val_from_ptr(nfev),
            val_from_ptr(njev),
            val_from_ptr(info),
        )

    if isinstance(udata, types.Array):
        return (
            lambda fcn,
            n,
            x,
            fvec,
            fjac,
            ldfjac,
            xtol,
            maxfev,
            diag,
            mode,
            factor,
            nprint,
            info,
            nfev,
            njev,
            r,
            lr,
            qtf,
            wa1,
            wa2,
            wa3,
            wa4,
            udata: impl(
                fcn,
                n,
                x,
                fvec,
                fjac,
                ldfjac,
                xtol,
                maxfev,
                diag,
                mode,
                factor,
                nprint,
                info,
                nfev,
                njev,
                r,
                lr,
                qtf,
                wa1,
                wa2,
                wa3,
                wa4,
                udata.ctypes,
            )
        )
    if udata is not types.none:
        return impl

    return (
        lambda fcn,
        n,
        x,
        fvec,
        fjac,
        ldfjac,
        xtol,
        maxfev,
        diag,
        mode,
        factor,
        nprint,
        info,
        nfev,
        njev,
        r,
        lr,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata: impl(
            fcn,
            n,
            x,
            fvec,
            fjac,
            ldfjac,
            xtol,
            maxfev,
            diag,
            mode,
            factor,
            nprint,
            info,
            nfev,
            njev,
            r,
            lr,
            qtf,
            wa1,
            wa2,
            wa3,
            wa4,
            0,
        )
    )


@njit
def hybrj_(
    fcn,
    n,
    x,
    fvec,
    fjac,
    ldfjac,
    xtol,
    maxfev,
    diag,
    mode,
    factor,
    nprint,
    info,
    nfev,
    njev,
    r,
    lr,
    qtf,
    wa1,
    wa2,
    wa3,
    wa4,
    udata=None,
):
    return _hybrj(
        fcn,
        n,
        x,
        fvec,
        fjac,
        ldfjac,
        xtol,
        maxfev,
        diag,
        mode,
        factor,
        nprint,
        info,
        nfev,
        njev,
        r,
        lr,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata,
    )


@njit
def hybrj(
    fcn: int64,
    x: NDArray[float64],
    xtol: float64 | None = None,
    maxfev: int32 | None = None,
    diag: NDArray[float64] | None = None,
    mode: int32 | None = None,
    factor: float64 | None = None,
    nprint: int32 | None = None,
    udata: NDArray | None = None,
) -> tuple[
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    int,
    int,
    int,
]:
    n = int32(x.size)
    fvec = empty(n, dtype=x.dtype)
    infoptr = ptr_from_val(int32(0))
    nfevptr = ptr_from_val(int32(0))
    njevptr = ptr_from_val(int32(0))
    fjac = empty((n, n), dtype=x.dtype)
    ldfjac = n
    lr = (n * (n + 1)) // 2
    r = empty(lr, dtype=x.dtype)
    qtf = empty(n, dtype=x.dtype)
    wa = empty(4 * n, dtype=x.dtype)
    wa1 = wa[:n]
    wa2 = wa[n : 2 * n]
    wa3 = wa[2 * n : 3 * n]
    wa4 = wa[3 * n :]

    xtol = xtol or 1.49012e-8
    maxfev = maxfev or 200 * (n + 1)
    diag = ones(n, dtype=x.dtype) if diag is None else diag
    mode = mode or 1
    factor = factor or 100.0
    nprint = nprint or 0

    return _hybrj(
        fcn,
        n,
        x.copy(),
        fvec,
        fjac,
        ldfjac,
        xtol,
        maxfev,
        diag,
        mode,
        factor,
        nprint,
        infoptr,
        nfevptr,
        njevptr,
        r,
        lr,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata,
    )
