"""Displacement and Cauchy-stress readout of coupled finite-strain fields."""
from __future__ import annotations

import numpy as np


class FieldReadout:
    def __init__(self, field):
        self.field = field
        self.model = field.model
        self.operators = field.operators

    def values(self, points, *, ground=False):
        """Evaluate material points and their deformed local frames in batches."""
        groups = {}
        for row, item in enumerate(points):
            segment = next(i for i, (arc, bounds) in enumerate(self.model.segments)
                if arc == item['arc'] and bounds[0]-1e-12 <= item['fraction'] <= bounds[1]+1e-12)
            index = segment if ground else self.operators.support_index[segment]
            patch = self.operators.patches[index]
            b = 2*(item['fraction']-patch.fraction[0])/(patch.fraction[1]-patch.fraction[0])-1
            a = -1. if ground else 1-2*item.get('depth', 0.)/self.model.section.thickness
            groups.setdefault(index, []).append((row, a, b))
        displacement = np.empty((len(points), 2))
        deformation = np.empty((len(points), 2, 2))
        stress = np.empty((len(points), 2, 2))
        for index, entries in groups.items():
            rows = np.array([r for r, _, _ in entries])
            local = np.array([[a, b] for _, a, b in entries])
            patch = self.operators.patches[index]
            u, s, gu, _, _, _ = patch.basis(local)
            coefficients = self.operators.coefficients(self.field.x, index)
            displacement[rows] = u @ coefficients[:2].T*self.model.length/self.model.physical['e_ground']
            f = np.eye(2)+np.einsum('qjf,if->qij', gu, coefficients[:2])/self.model.physical['e_ground']
            nominal = (s @ coefficients[2:].T).reshape(-1, 2, 2)
            if ground:
                nominal += self.field.materials.initial_stress[:2, :2].cpu().numpy()
            sigma = nominal @ f.transpose(0, 2, 1)/np.linalg.det(f)[:, None, None]
            deformation[rows] = f
            stress[rows] = .5*(sigma+sigma.transpose(0, 2, 1))
        normal, tangent = np.array([self.model.section.local_frame(p['arc'], p['fraction'])
                                    for p in points]).transpose(1, 0, 2)
        current_normal = np.linalg.solve(deformation.transpose(0, 2, 1), normal[..., None])[..., 0]
        current_normal /= np.linalg.norm(current_normal, axis=1)[:, None]
        current_tangent = np.einsum('qij,qj->qi', deformation, tangent)
        current_tangent /= np.linalg.norm(current_tangent, axis=1)[:, None]
        pressure = -np.einsum('qi,qij,qj->q', current_normal, stress, current_normal)
        compression = -np.einsum('qi,qij,qj->q', current_tangent, stress, current_tangent)
        shear = np.einsum('qi,qij,qj->q', current_tangent, stress, current_normal)
        return dict(displacement_m=displacement, deformation=deformation,
                    stress_mpa=stress, normal=current_normal, tangent=current_tangent,
                    pressure_mpa=pressure, compression_mpa=compression, shear_mpa=shear)

    def survey_and_targets(self, layout, depth):
        from ..measurements import baseline_points
        points = baseline_points(layout)+layout['candidates']
        coordinates = self.values(points)['displacement_m']
        outer = self.values([dict(p, depth=0.) for p in layout['outer']])
        inner = self.values([dict(p, depth=self.model.section.thickness) for p in layout['inner']])
        sections = [dict(arc=arc, fraction=.5, depth=float(d))
                    for arc in ('crown', 'left_knee', 'right_knee', 'invert') for d in depth]
        profiles = self.values(sections)
        support_depth = np.linspace(0., self.model.section.thickness, 11)
        support = self.values([dict(p, depth=float(d)) for p in layout['outer']
                               for d in support_depth])
        return dict(baseline_xy_m=coordinates[:7], candidate_xy_m=coordinates[7:],
                    pressure_mpa=outer['pressure_mpa'], inner_u_m=inner['displacement_m'],
                    stress_mpa=np.stack((profiles['compression_mpa'], profiles['pressure_mpa'],
                                        profiles['shear_mpa']), axis=-1).reshape(4, len(depth), 3),
                    support_stress_mpa=np.stack((support['compression_mpa'], support['pressure_mpa'],
                                                support['shear_mpa']), axis=-1).reshape(
                                                    len(layout['outer']), len(support_depth), 3))

    def ground_display(self, count=49):
        """Read the near-ground patches on the existing figure's display grid."""
        a, b = np.meshgrid(np.linspace(-1., 1., count), np.linspace(-1., 1., count), indexing='ij')
        local = np.column_stack((a.ravel(), b.ravel()))
        coordinates, fields = [], []
        for index, patch in enumerate(self.model.ground):
            if patch.radial[0] > self.model.radial_edges[1]+1e-12:
                continue
            u, s, gu, _, _, _ = patch.basis(local)
            coefficients = self.operators.coefficients(self.field.x, index)
            displacement = u @ coefficients[:2].T*self.model.length/self.model.physical['e_ground']
            f = np.eye(2)+np.einsum('qjf,if->qij', gu, coefficients[:2])/self.model.physical['e_ground']
            nominal = (s @ coefficients[2:].T).reshape(-1, 2, 2)
            nominal += self.field.materials.initial_stress[:2, :2].cpu().numpy()
            sigma = nominal @ f.transpose(0, 2, 1)/np.linalg.det(f)[:, None, None]
            sigma = .5*(sigma+sigma.transpose(0, 2, 1))
            coordinates.append((patch.mapping(local)[0]*self.model.length).reshape(count, count, 2))
            fields.append(np.column_stack((displacement, sigma[:, 0, 0], sigma[:, 1, 1],
                                            sigma[:, 0, 1])).reshape(count, count, 5))
        return dict(ground_xy_m=np.asarray(coordinates), ground_fields=np.asarray(fields))
