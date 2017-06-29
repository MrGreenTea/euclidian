from functools import partial

import math

PRECISION = 1e-5

isclose = partial(math.isclose, rel_tol=PRECISION, abs_tol=PRECISION)