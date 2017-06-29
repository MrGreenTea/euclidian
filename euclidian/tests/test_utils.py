import math

from hypothesis import given, strategies, note, assume

import tests.strategies as my_st
import utils


@given(start=my_st.floats(), stop=my_st.floats(), base=strategies.just(1/3),
       num=strategies.integers(min_value=2, max_value=1e4))
def test_linlogspace_correct_range(start, stop, num, base):
    space = utils.linlogspace(start, stop, num, base=base)
    note(space)

    assert math.isclose(space[0], start, abs_tol=my_st.PRECISION)
    assert math.isclose(space[-1], stop, abs_tol=my_st.PRECISION)
    assert math.isclose(space.max(), max(start, stop), abs_tol=my_st.PRECISION)
    assert math.isclose(space.min(), min(start, stop), abs_tol=my_st.PRECISION)
    assert len(space) == num
