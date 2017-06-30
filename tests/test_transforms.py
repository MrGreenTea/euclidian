import numpy as np
from hypothesis import given

import euclidian.transforms
import tests.utils
from tests.strategies import points, values, arrays


@given(ps=points(), tr=arrays(shape=(1, 2)))
def test_translate(ps: np.array, tr: np.array):
    transformed = euclidian.transforms.translate(ps, tr)
    assert transformed.shape == ps.shape
    backwards = euclidian.transforms.translate(transformed, -tr)
    np.testing.assert_allclose(backwards, ps, atol=tests.utils.PRECISION)


@given(ps=points())
def test_scale_to_zero(ps):
    scaled = euclidian.transforms.scale(ps, 0, center=np.matrix([0, 0]))
    assert not scaled.any()


@given(ps=points(), s=values())
def test_scale_retains_shape(ps, s):
    scaled = euclidian.transforms.scale(ps, s)
    assert scaled.shape == ps.shape


@given(ps=points(), s=values())
def test_scale(ps, s):
    scaled = euclidian.transforms.scale(ps, s, center=np.matrix([0, 0]))
    for p, sp in zip(ps.flat, scaled.flat):
        assert tests.utils.isclose(p*s, sp)


@given(ps=points(), a=values(), center=arrays(shape=(1, 2)))
def test_rotation(ps, a, center):  # TODO design correctly
    rotated = euclidian.transforms.rotate(ps, a, center=center)
    assert rotated.shape == ps.shape
