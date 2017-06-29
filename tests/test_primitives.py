from hypothesis import given, strategies
import numpy as np

import primitives
import utils

import tests.strategies as my_st


@given(n=strategies.integers(min_value=2, max_value=1e4))
def test_polygon_vertices(n):
    polygon = primitives.regular_polygon(n)
    assert len(polygon) == n
    np.testing.assert_almost_equal(utils.calc_center(polygon), 0)


@given(ps=my_st.points(shape=(2, 2)), endpoint=strategies.booleans(),
       segments=strategies.integers(min_value=2, max_value=1e4))
def test_segmented_line(ps, segments, endpoint):
    start, end = ps[0], ps[1]
    line = primitives.segmented_line(start, end, segments=segments, endpoint=endpoint)
    assert len(line) == segments
    np.testing.assert_almost_equal(line[0], start)

    if endpoint:
        np.testing.assert_almost_equal(line[-1], end)

    a, b = utils.bounding_box(line)
    assert (a <= line).all()
    assert (b >= line).all()
