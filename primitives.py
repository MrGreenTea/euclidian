import numpy as np


def regular_polygon(n, r=1):
    """Regular polygon with n sides and radius of circumcircle r."""
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = np.matrix(np.stack((np.cos(phi), np.sin(phi))))
    return p.T * r


def segmented_line(start: np.matrix, end: np.matrix, segments: int, endpoint=True) -> np.matrix:
    """
    Similar to np.arange for points.
    
    """
    segments = segments - endpoint
    step = (end - start) / segments
    ps = [list((start + step * i).flat) for i in range(segments)]
    if endpoint:
        ps.append(list(end.flat))
    return np.matrix(ps)


def _identity(phi):
    return phi


def spiral(points=8, start=0, stop=2, f=_identity):
    """
    polar plot of f(phi)=r.

    start and stop will be multiplied by PI.
    f defaults to identity f(phi)=phi.
    """
    stop = np.log2(stop+1)
    phi = np.pi*(np.logspace(start, stop, points, base=2)-1)
    m = np.array([np.cos(phi), np.sin(phi)]) * f(phi)
    return m.T
