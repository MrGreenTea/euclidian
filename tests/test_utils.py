from hypothesis import given, strategies, note

import tests.strategies as my_st
from euclidian import utils
from tests.utils import isclose


@given(start=my_st.values(), stop=my_st.values(), base=my_st.values(min_value=1e-10),
       num=strategies.integers(min_value=2, max_value=1e4))
def test_linlogspace_correct_range(start, stop, num, base):
    space = utils.linlogspace(start=start, stop=stop, num=num, base=base)
    note(space)

    assert isclose(space[0], start)
    assert isclose(space[-1], stop)
    assert isclose(space.max(), max(start, stop))
    assert isclose(space.min(), min(start, stop))
    assert len(space) == num


@given(ps=my_st.points())
def test_close(ps):
    closed = utils.close(ps)
    assert len(closed) == len(ps) + 1
    assert (ps[0] == closed[-1]).all()
