from functools import partial

import numpy as np


def calc_center(points):
    return np.average(points, axis=0)


append = partial(np.append, axis=0)