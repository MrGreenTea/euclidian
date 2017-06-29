from functools import partial

import numpy as np


def calc_center(points: np.matrix) -> np.matrix:
    return np.average(points, axis=0)


def bounding_box(points: np.matrix):
    """Return a, b so that all points lay in bounding box spanned by a and b."""
    return points.min(axis=0), points.max(axis=0)


def linlogspace(start, stop, num=50, *args, base=0.5, **kwargs) -> np.ndarray:
    """A log space with later values being closer together."""
    p = (1 - np.logspace(0, 1, num=num, *args, base=base, **kwargs)) * (1 / base)
    return (1 - p) * start + p * stop


append = partial(np.append, axis=0)


def close(points: np.matrix):
    """Closes points by appending the first point to the end."""
    return append(points, points[0])
