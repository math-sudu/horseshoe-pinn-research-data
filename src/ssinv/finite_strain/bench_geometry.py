"""Remaining invert rock, with a horizontal excavation floor at the arch toes."""
from __future__ import annotations

import numpy as np
from ..patch_basis import LocalNetwork


class GroundBenchNetwork(LocalNetwork):
    material_zone = 'ground'

    def __init__(self, section, fraction, width, seed, amplitude):
        super().__init__(section, 'invert', fraction, (0., 1.), 'liner', width, seed, amplitude)
        arc = section.arc('initial_support_outer', 'invert')
        # Match the inner end of the existing arch cut. This leaves real rock
        # beneath both toe faces, rather than prescribing a toe spring.
        self.floor_y = (self.center[1]+(self.radius-self.thickness)*np.sin(arc['start_rad']))

    def mapping(self, local):
        a, b = local.T
        half_phi = (self.phi1-self.phi0)/2
        phi = (self.phi0+self.phi1)/2+b*half_phi
        normal = np.column_stack((np.cos(phi), np.sin(phi)))
        tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
        inner = (self.floor_y-self.center[1])/np.sin(phi)
        inner_derivative = -inner*np.cos(phi)/np.sin(phi)
        radius = inner+(self.radius-inner)*(a+1)/2
        point = self.center+radius[:, None]*normal
        da = ((self.radius-inner)/2)[:, None]*normal
        db = ((1-a)*inner_derivative*half_phi/2)[:, None]*normal+(radius*half_phi)[:, None]*tangent
        return point, np.stack((da, db), 1), normal, np.ones(len(local))
