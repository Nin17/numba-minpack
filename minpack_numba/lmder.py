"""Wrappers of the `lmder` and `lmder1` functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import extending, njit, types
from numpy import empty, int32, ones

from ._minpack import Minpack
from .utils import _check_dtype, ptr_from_val, val_from_ptr

if TYPE_CHECKING:
    from numpy import float64, int64
    from numpy.typing import NDArray

# TODO(nin17): type hints
# -------------------------------------- lmder1 -------------------------------------- #


def _lmder1(fcn, m, n, x, fvec, fjac, ldfjac, tol, info, ipvt, wa, lwa, udata):
    raise NotImplementedError


@extending.overload(_lmder1)
def _lmder1_overload(fcn, m, n, x, fvec, fjac, ldfjac, tol, info, ipvt, wa, lwa, udata):
    _check_dtype((fvec, fjac, wa), x.dtype)
    lmder1_external = Minpack.lmder1()

    @extending.register_jitable
    def impl(fcn, m, n, x, fvec, fjac, ldfjac, tol, info, ipvt, wa, lwa, udata):
        lmder1_external(
            fcn,
            m,
            n,
            x.ctypes,
            fvec.ctypes,
            fjac.ctypes,
            ldfjac,
            tol,
            info,
            ipvt.ctypes,
            wa.ctypes,
            lwa,
            udata,
        )
        return x, fvec, fjac, ipvt, val_from_ptr(info)

    if isinstance(udata, types.Array):
        return (
            lambda fcn,
            m,
            n,
            x,
            fvec,
            fjac,
            ldfjac,
            tol,
            info,
            ipvt,
            wa,
            lwa,
            udata: impl(
                fcn,
                m,
                n,
                x,
                fvec,
                fjac,
                ldfjac,
                tol,
                info,
                ipvt,
                wa,
                lwa,
                udata.ctypes,
            )
        )
    if udata is not types.none:
        return impl
    return (
        lambda fcn, m, n, x, fvec, fjac, ldfjac, tol, info, ipvt, wa, lwa, udata: impl(
            fcn,
            m,
            n,
            x,
            fvec,
            fjac,
            ldfjac,
            tol,
            info,
            ipvt,
            wa,
            lwa,
            0,
        )
    )


@njit
def lmder1_(fcn, m, n, x, fvec, fjac, ldfjac, tol, info, ipvt, wa, lwa, udata=None):
    return _lmder1(fcn, m, n, x, fvec, fjac, ldfjac, tol, info, ipvt, wa, lwa, udata)


@njit
def lmder1(fcn, m, x, tol=None, udata=None):
    tol = tol or 1.49012e-8
    n = int32(x.size)
    lwa = int32(5 * n + m)
    fvec = empty(m, dtype=x.dtype)
    fjac = empty((m, n), dtype=x.dtype)
    wa = empty(lwa, dtype=x.dtype)
    ipvt = empty(n, dtype=int32)
    infoptr = ptr_from_val(int32(0))
    return _lmder1(
        fcn,
        m,
        n,
        x.copy(),
        fvec,
        fjac,
        m,
        tol,
        infoptr,
        ipvt,
        wa,
        lwa,
        udata,
    )


# --------------------------------------- lmder -------------------------------------- #


def _lmder(
    fcn,
    m,
    n,
    x,
    fvec,
    fjac,
    ldfjac,
    ftol,
    xtol,
    gtol,
    maxfev,
    diag,
    mode,
    factor,
    nprint,
    info,
    nfev,
    njev,
    ipvt,
    qtf,
    wa1,
    wa2,
    wa3,
    wa4,
    udata,
):
    raise NotImplementedError


@extending.overload(_lmder)
def _lmder_overload(
    fcn,
    m,
    n,
    x,
    fvec,
    fjac,
    ldfjac,
    ftol,
    xtol,
    gtol,
    maxfev,
    diag,
    mode,
    factor,
    nprint,
    info,
    nfev,
    njev,
    ipvt,
    qtf,
    wa1,
    wa2,
    wa3,
    wa4,
    udata,
):
    _check_dtype((fvec, fjac, wa1, wa2, wa3, wa4), x.dtype)
    lmder_external = Minpack.lmder()

    @extending.register_jitable
    def impl(
        fcn,
        m,
        n,
        x,
        fvec,
        fjac,
        ldfjac,
        ftol,
        xtol,
        gtol,
        maxfev,
        diag,
        mode,
        factor,
        nprint,
        info,
        nfev,
        njev,
        ipvt,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata,
    ):
        lmder_external(
            fcn,
            m,
            n,
            x.ctypes,
            fvec.ctypes,
            fjac.ctypes,
            ldfjac,
            ftol,
            xtol,
            gtol,
            maxfev,
            diag.ctypes,
            mode,
            factor,
            nprint,
            info,
            nfev,
            njev,
            ipvt.ctypes,
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
            ipvt,
            qtf,
            val_from_ptr(nfev),
            val_from_ptr(njev),
            val_from_ptr(info),
        )

    if isinstance(udata, types.Array):
        return (
            lambda fcn,
            m,
            n,
            x,
            fvec,
            fjac,
            ldfjac,
            ftol,
            xtol,
            gtol,
            maxfev,
            diag,
            mode,
            factor,
            nprint,
            info,
            nfev,
            njev,
            ipvt,
            qtf,
            wa1,
            wa2,
            wa3,
            wa4,
            udata: impl(
                fcn,
                m,
                n,
                x,
                fvec,
                fjac,
                ldfjac,
                ftol,
                xtol,
                gtol,
                maxfev,
                diag,
                mode,
                factor,
                nprint,
                info,
                nfev,
                njev,
                ipvt,
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
        m,
        n,
        x,
        fvec,
        fjac,
        ldfjac,
        ftol,
        xtol,
        gtol,
        maxfev,
        diag,
        mode,
        factor,
        nprint,
        info,
        nfev,
        njev,
        ipvt,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata: impl(
            fcn,
            m,
            n,
            x,
            fvec,
            fjac,
            ldfjac,
            ftol,
            xtol,
            gtol,
            maxfev,
            diag,
            mode,
            factor,
            nprint,
            info,
            nfev,
            njev,
            ipvt,
            qtf,
            wa1,
            wa2,
            wa3,
            wa4,
            0,
        )
    )


@njit
def lmder_(
    fcn,
    m,
    n,
    x,
    fvec,
    fjac,
    ldfjac,
    ftol,
    xtol,
    gtol,
    maxfev,
    diag,
    mode,
    factor,
    nprint,
    info,
    nfev,
    njev,
    ipvt,
    qtf,
    wa1,
    wa2,
    wa3,
    wa4,
    udata=None,
):
    return _lmder(
        fcn,
        m,
        n,
        x,
        fvec,
        fjac,
        ldfjac,
        ftol,
        xtol,
        gtol,
        maxfev,
        diag,
        mode,
        factor,
        nprint,
        info,
        nfev,
        njev,
        ipvt,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata,
    )


@njit
def lmder(
    fcn: int64,
    m: int32,
    x: NDArray[float64],
    ftol: float64 | None = None,
    xtol: float64 | None = None,
    gtol: float64 | None = None,
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
    NDArray[int32],
    NDArray[float64],
    int,
    int,
    int,
]:
    n = int32(x.size)
    fvec = empty(m, dtype=x.dtype)
    fjac = empty((m, n), dtype=x.dtype)
    ldfjac = m
    infoptr = ptr_from_val(int32(0))
    nfevptr = ptr_from_val(int32(0))
    njevptr = ptr_from_val(int32(0))
    ipvt = empty(n, dtype=int32)
    qtf = empty(n, dtype=x.dtype)
    wa = empty(3 * n + m, dtype=x.dtype)
    wa1 = wa[:n]
    wa2 = wa[n : 2 * n]
    wa3 = wa[2 * n : 3 * n]
    wa4 = wa[3 * n :]

    ftol = ftol or 1.49012e-8
    xtol = xtol or 1.49012e-8
    gtol = gtol or 0.0
    mode = mode or 1
    factor = factor or 100.0
    nprint = nprint or 0
    diag = ones(n, dtype=x.dtype) if diag is None else diag
    maxfev = maxfev or 200 * (n + 1)

    return _lmder(
        fcn,
        m,
        n,
        x.copy(),
        fvec,
        fjac,
        ldfjac,
        ftol,
        xtol,
        gtol,
        maxfev,
        diag,
        mode,
        factor,
        nprint,
        infoptr,
        nfevptr,
        njevptr,
        ipvt,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata,
    )
