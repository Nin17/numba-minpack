"""Test eps, max and tiny given by numpy against dpmpar."""

from numpy import finfo, float64
from numpy.testing import assert_equal

from minpack_numba import dpmpar


def test_machine_precision_double() -> None:
    """Test that the machine precision is equal to the numpy float64 precision."""
    double_eps = dpmpar(1)
    assert_equal(double_eps, finfo(float64).eps)


def test_smallest_magnitude_double() -> None:
    """Test that the smallest magnitude is equal to the numpy float64 tiny."""
    double_tiny = dpmpar(2)
    assert_equal(double_tiny, finfo(float64).tiny)


def test_largest_magnitude_double() -> None:
    """Test that the largest magnitude is equal to the numpy float64 max."""
    double_max = dpmpar(3)
    assert_equal(double_max, finfo(float64).max)
