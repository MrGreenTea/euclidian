from functools import partial

import hypothesis.extra.numpy as np_st
import numpy as np
from hypothesis import strategies as st

# Strategies
floats = partial(st.floats, allow_nan=False, allow_infinity=False)
# sane float values to minimize problems due to rounding errors.
values = partial(floats, min_value=-1e10, max_value=1e10)


def shapes(max_len):
    return st.tuples(st.integers(min_value=1, max_value=max_len), st.just(2))


arrays = partial(np_st.arrays, dtype=np.float, elements=values())  # general arrays


def points(*args, max_len=50, **kwargs):
    kwargs.setdefault('shape', shapes(max_len))
    return arrays(*args, **kwargs)
