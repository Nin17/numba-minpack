"""Wrappers of the `hybrd` and `hybrd1` functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import extending, njit, types
from numpy import empty, finfo, float64, floating, int32, int64, ones

from ._minpack import Minpack
from .utils import _check_dtype, ptr_from_val, ptr_int32, val_from_ptr

if TYPE_CHECKING:
    from numpy.typing import NDArray

# TODO(nin17): check_dtype & type hints

# -------------------------------------- hybrd1 -------------------------------------- #


def _hybrd1(fcn, n, x, fvec, tol, info, wa, lwa, udata):
    raise NotImplementedError


@extending.overload(_hybrd1)
def _hybrd1_overload(fcn, n, x, fvec, tol, info, wa, lwa, udata):
    _check_dtype((fvec, wa), x.dtype)
    hybrd1_external = Minpack.hybrd1()

    @extending.register_jitable
    def impl(fcn, n, x, fvec, tol, info, wa, lwa, udata):
        hybrd1_external(
            fcn,
            n,
            x.ctypes,
            fvec.ctypes,
            tol,
            info,
            wa.ctypes,
            lwa,
            udata,
        )
        return x, fvec, val_from_ptr(info)

    if isinstance(udata, types.Array):
        return lambda fcn, n, x, fvec, tol, info, wa, lwa, udata: impl(
            fcn,
            n,
            x,
            fvec,
            tol,
            info,
            wa,
            lwa,
            udata.ctypes,
        )

    if udata is not types.none:
        return impl
    return lambda fcn, n, x, fvec, tol, info, wa, lwa, udata: impl(
        fcn,
        n,
        x,
        fvec,
        tol,
        info,
        wa,
        lwa,
        0,
    )


@njit
def hybrd1_(
    fcn: int64,
    n: int32,
    x: NDArray[float64],
    fvec: NDArray[float64],
    tol: float64,
    info: ptr_int32,
    wa: NDArray[float64],
    lwa: int32,
    udata: NDArray | None = None,
) -> tuple[NDArray[float64], NDArray[float64], int]:
    return _hybrd1(fcn, n, x, fvec, tol, info, wa, lwa, udata)


@njit
def hybrd1(
    fcn: int64,
    x: int32,
    tol: float64 | None = None,
    udata: NDArray | None = None,
) -> tuple[NDArray[float64], NDArray[float64], int]:
    tol = tol or 1.49012e-8
    n = int32(x.size)
    lwa = int32((n * (3 * n + 13)) // 2)
    fvec = empty(n, dtype=float64)
    wa = empty(lwa, dtype=float64)
    infoptr = ptr_from_val(int32(0))
    return _hybrd1(fcn, n, x.copy(), fvec, tol, infoptr, wa, lwa, udata)


# --------------------------------------- hybrd -------------------------------------- #


def _hybrd(
    fcn,
    n,
    x,
    fvec,
    xtol,
    maxfev,
    ml,
    mu,
    epsfcn,
    diag,
    mode,
    factor,
    nprint,
    info,
    nfev,
    fjac,
    ldfjac,
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


@extending.overload(_hybrd)
def _hybrd_overload(
    fcn,
    n,
    x,
    fvec,
    xtol,
    maxfev,
    ml,
    mu,
    epsfcn,
    diag,
    mode,
    factor,
    nprint,
    info,
    nfev,
    fjac,
    ldfjac,
    r,
    lr,
    qtf,
    wa1,
    wa2,
    wa3,
    wa4,
    udata,
):
    _check_dtype((fvec, fjac, qtf, wa1, wa2, wa3, wa4), x.dtype)
    hybrd_external = Minpack.hybrd()

    @extending.register_jitable
    def impl(
        fcn,
        n,
        x,
        fvec,
        xtol,
        maxfev,
        ml,
        mu,
        epsfcn,
        diag,
        mode,
        factor,
        nprint,
        info,
        nfev,
        fjac,
        ldfjac,
        r,
        lr,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata,
    ):
        hybrd_external(
            fcn,
            n,
            x.ctypes,
            fvec.ctypes,
            xtol,
            maxfev,
            ml,
            mu,
            epsfcn,
            diag.ctypes,
            mode,
            factor,
            nprint,
            info,
            nfev,
            fjac.ctypes,
            ldfjac,
            r.ctypes,
            lr,
            qtf.ctypes,
            wa1.ctypes,
            wa2.ctypes,
            wa3.ctypes,
            wa4.ctypes,
            udata,
        )
        return x, fvec, fjac, r, qtf, val_from_ptr(nfev), val_from_ptr(info)

    if isinstance(udata, types.Array):
        return (
            lambda fcn,
            n,
            x,
            fvec,
            xtol,
            maxfev,
            ml,
            mu,
            epsfcn,
            diag,
            mode,
            factor,
            nprint,
            info,
            nfev,
            fjac,
            ldfjac,
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
                xtol,
                maxfev,
                ml,
                mu,
                epsfcn,
                diag,
                mode,
                factor,
                nprint,
                info,
                nfev,
                fjac,
                ldfjac,
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
        xtol,
        maxfev,
        ml,
        mu,
        epsfcn,
        diag,
        mode,
        factor,
        nprint,
        info,
        nfev,
        fjac,
        ldfjac,
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
            xtol,
            maxfev,
            ml,
            mu,
            epsfcn,
            diag,
            mode,
            factor,
            nprint,
            info,
            nfev,
            fjac,
            ldfjac,
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
def hybrd_(
    fcn,
    n,
    x,
    fvec,
    xtol,
    maxfev,
    ml,
    mu,
    epsfcn,
    diag,
    mode,
    factor,
    nprint,
    info,
    nfev,
    fjac,
    ldfjac,
    r,
    lr,
    qtf,
    wa1,
    wa2,
    wa3,
    wa4,
    udata=None,
):
    return _hybrd(
        fcn,
        n,
        x,
        fvec,
        xtol,
        maxfev,
        ml,
        mu,
        epsfcn,
        diag,
        mode,
        factor,
        nprint,
        info,
        nfev,
        fjac,
        ldfjac,
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
def hybrd(
    fcn: int64,
    x: NDArray[floating],
    xtol: floating | None = None,
    maxfev: int32 | None = None,
    ml: int32 | None = None,
    mu: int32 | None = None,
    epsfcn: floating | None = None,
    diag: NDArray[floating] | None = None,
    mode: int32 | None = None,
    factor: floating | None = None,
    nprint: int32 | None = None,
    udata: NDArray | None = None,
) -> tuple[
    NDArray[floating],
    NDArray[floating],
    NDArray[floating],
    NDArray[floating],
    NDArray[floating],
    int,
    int,
]:
    # TODO(nin17): docstring.
    """.

    Parameters
    ----------
    fcn : int64
        _description_
    x : NDArray[floating]
        _description_
    xtol : floating | None, optional
        _description_, by default None
    maxfev : int32 | None, optional
        _description_, by default None
    ml : int32 | None, optional
        _description_, by default None
    mu : int32 | None, optional
        _description_, by default None
    epsfcn : floating | None, optional
        _description_, by default None
    diag : NDArray[floating] | None, optional
        _description_, by default None
    mode : int32 | None, optional
        _description_, by default None
    factor : floating | None, optional
        _description_, by default None
    nprint : int32 | None, optional
        _description_, by default None
    udata : NDArray | None, optional
        _description_, by default None

    Returns
    -------
    tuple[ NDArray[floating], NDArray[floating], NDArray[floating], NDArray[floating],
           NDArray[floating], int, int, ]
        _description_

    """
    n = int32(x.size)
    fvec = empty(n, dtype=x.dtype)
    infoptr = ptr_from_val(int32(0))
    nfevptr = ptr_from_val(int32(0))
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
    ml = ml or n
    mu = mu or n
    epsfcn = epsfcn or finfo(x.dtype).eps
    diag = ones(n, dtype=x.dtype) if diag is None else diag
    mode = mode or 1
    factor = factor or 100.0
    nprint = nprint or 0

    return _hybrd(
        fcn,
        n,
        x.copy(),
        fvec,
        xtol,
        maxfev,
        ml,
        mu,
        epsfcn,
        diag,
        mode,
        factor,
        nprint,
        infoptr,
        nfevptr,
        fjac,
        ldfjac,
        r,
        lr,
        qtf,
        wa1,
        wa2,
        wa3,
        wa4,
        udata,
    )
