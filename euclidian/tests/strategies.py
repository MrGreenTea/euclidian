from functools import partial

import hypothesis.extra.numpy as np_st
import numpy as np
from hypothesis import strategies as st

PRECISION = 1e-6


# Strategies
floats = partial(st.floats, allow_nan=False, allow_infinity=False)


values = partial(floats, min_value=-1e10, max_value=1e10)


def shapes(max_len=10):
    return st.tuples(st.integers(min_value=1, max_value=max_len), st.just(2))


arrays = partial(np_st.arrays, dtype=np.float, elements=values())  # general arrays


def points(*args, **kwargs):
    kwargs.setdefault('shape', shapes(kwargs.pop('max_len', 10)))
    return arrays(*args, **kwargs).map(np.matrix)
