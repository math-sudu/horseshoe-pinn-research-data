"""Integration weights for physical support-depth readouts."""
import numpy as np


def thickness_weights(depth):
    """Trapezoidal weights on the contact-to-intrados depth coordinates."""
    spacing = np.diff(depth)
    return np.r_[spacing[0], spacing[:-1]+spacing[1:], spacing[-1]]/2
