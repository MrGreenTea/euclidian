import typing

import numpy as np

import utils


def regular_polygon(n, r=1):
    """Regular polygon with n sides and radius of circumcircle r."""
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = np.matrix(np.stack((np.cos(phi), np.sin(phi))))
    return p.T * r


def segmented_line(start: np.matrix, end: np.matrix, segments: int, endpoint=True) -> np.matrix:
    """
    Similar to np.arange for points.
    """
    start, end = start.A[0], end.A[0]
    X = np.linspace(start[0], end[0], num=segments, endpoint=endpoint)
    Y = np.linspace(start[1], end[1], num=segments, endpoint=endpoint)
    return np.matrix(np.stack([X, Y])).T


def _identity(phi):
    return np.array([np.cos(phi), np.sin(phi)]) / (2 * np.pi)


def polar(start=0, stop=2, points=8, *, f: typing.Callable[[np.ndarray], np.ndarray]=_identity):
    """
    polar plot of f(phi)=(x,y).

    start and stop will be multiplied by PI.
    f defaults to f(phi)=phi/2PI.
    """
    phi = np.pi * utils.linlogspace(start, stop, points)
    m = f(phi)
    return m.T
