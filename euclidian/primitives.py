import typing

import numpy as np

import euclidian.utils
import euclidian.transforms


def regular_polygon(n) -> np.ndarray:
    """unit regular n-polygon."""
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack((np.cos(phi), np.sin(phi))).T


def polygon_with_side(n, s):
    """
    n-polygon with side-length of s.
    
    uses :py:func:`regular_polygon()` and scales accordingly
    """
    r = s / (2 * np.sin(np.pi/n))
    return euclidian.transforms.scale(regular_polygon(n), r)


def segmented_line(start: np.ndarray, end: np.ndarray, segments: int, endpoint=True) -> np.ndarray:
    """
    Similar to np.arange for points.
    """
    x_values = np.linspace(start[0], end[0], num=segments, endpoint=endpoint)
    y_values = np.linspace(start[1], end[1], num=segments, endpoint=endpoint)
    return np.stack([x_values, y_values]).T


def identity(phi: np.ndarray) -> np.ndarray:
    return np.array([np.cos(phi), np.sin(phi)]) / (2 * np.pi)


def polar(start=0, stop=2, num=8, *, space=euclidian.utils.linlogspace,
          f: typing.Callable[[np.ndarray], np.ndarray] = identity, **kwargs):
    """
    polar plot of f(phi)=(x,y).

    start and stop will be multiplied by PI.
    f defaults to f(phi)=phi/2PI.
    """
    phi = np.pi * space(start=start, stop=stop, num=num, **kwargs)
    m = f(phi)
    return m.T
