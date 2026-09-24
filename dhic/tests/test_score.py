import pytest

from score import bd_rate


def test_bd_rate_for_uniform_rate_increase():
    reference = ([0.1, 0.2, 0.4, 0.8], [30, 32, 34, 36])
    candidate = ([0.11, 0.22, 0.44, 0.88], [30, 32, 34, 36])

    value, interval = bd_rate(reference, candidate)

    assert value == pytest.approx(10.0)
    assert interval == [30.0, 36.0]


def test_bd_rate_requires_overlapping_psnr_ranges():
    reference = ([0.1, 0.2], [30, 32])
    candidate = ([0.3, 0.4], [34, 36])

    with pytest.raises(ValueError, match="do not overlap"):
        bd_rate(reference, candidate)
