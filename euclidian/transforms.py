import operator

import numpy as np

from utils import calc_center

translate = np.vectorize(operator.add)


def rotate(points, angle, center=None):
    """Rotate points around center by angle."""
    if center is None:
        center = calc_center(points)
    sin, cos = np.sin(angle), np.cos(angle)
    m = np.array([[cos, -sin],  # rotation matrix
                  [sin, cos]])
    new_p = np.dot(m, translate(points, -center).T)
    return translate(new_p.T, center)


def scale(points, s, center=None):
    """Scale from center by s"""
    if center is None:
        center = calc_center(points)
    return translate(translate(points, -center) * s, center)
