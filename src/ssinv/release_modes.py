"""Four nominal traction amplitudes on the engineering reference contour."""
import numpy as np

NAMES = ("Qx", "Qy", "Gv", "Gh")


def traction_modes(points, normal, length, origin=(0., 0.)):
    xy = (np.asarray(points) - origin) / length
    normal = np.asarray(normal)
    out = np.zeros((*xy.shape[:-1], 2, len(NAMES)))
    out[..., 0, 0] = normal[..., 0]
    out[..., 1, 1] = normal[..., 1]
    out[..., 0, 2] = xy[..., 1] * normal[..., 0]
    out[..., 1, 3] = xy[..., 0] * normal[..., 1]
    return out
