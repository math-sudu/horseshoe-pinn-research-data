"""Point ordering and coordinate indices shared by the survey operators."""
import numpy as np


def baseline_points(layout):
    """Crown, then the ordered endpoint pair of each of the three chords."""
    crown = layout["baseline"][0]
    points = [dict(crown)]
    for chord in layout["baseline"][1:]:
        for name, point, xy in zip(chord["id"].split("-"), chord["endpoints"], chord["xy_m"], strict=True):
            points.append(dict(point, id=name, xy_m=xy))
    return points


def coordinate_rows(points):
    """Point indices in a flattened x,y coordinate vector."""
    return np.array([[2*i, 2*i+1] for i in points], dtype=int).ravel()
