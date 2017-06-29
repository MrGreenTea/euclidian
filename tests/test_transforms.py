import numpy as np
import pytest
from hypothesis import given, assume, note

import euclidian.transforms
from tests.strategies import points, values, arrays
from tests.utils import PRECISION


@given(ps=points(), tr=arrays(shape=(1, 2)))
def test_translate(ps: np.array, tr: np.array):
    transformed = euclidian.transforms.translate(ps, tr)
    assert transformed.shape == ps.shape
    backwards = euclidian.transforms.translate(transformed, -tr)
    np.testing.assert_allclose(backwards, ps, atol=PRECISION)


@given(ps=points())
def test_scale_to_zero(ps):
    scaled = euclidian.transforms.scale(ps, 0, center=np.matrix([0, 0]))
    assert not scaled.any()


@pytest.mark.skip('test scale not yet designed')
@given(ps=points(), s=values())
def test_scale(ps, s):  # TODO design correctly
    assume(s != 0)
    rs = 1 / s

    scaled = euclidian.transforms.scale(ps, s)
    assert scaled.shape == ps.shape

    reverted = euclidian.transforms.scale(ps, rs)
    note(reverted)
    assert reverted.shape == ps.shape
    np.testing.assert_allclose(reverted, ps, atol=PRECISION)


@pytest.mark.skip('test rotation not yet designed')
@given(ps=points(), a=values(), center=arrays(shape=(1, 2)))
def test_rotation(ps, a, center):  # TODO design correctly
    rotated = euclidian.transforms.rotate(points, a, center=center)
    assert rotated.shape == ps.shape

    reverted = euclidian.transforms.rotate(rotated, -a, center=center)
    note(reverted)
    np.testing.assert_allclose(reverted, ps, atol=PRECISION)