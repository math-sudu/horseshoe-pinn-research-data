"""Existing section/patch geometry with finite-strain Piola field operators.

The contour maps and fixed tanh features are shared through patch_basis.
Nominal tractions use reference normals; readout uses current_traction's
Nanson map. Inactive invert support leaves a ground boundary with the same
released initial traction. Arch cuts are free unless the retained-floor
option couples them to the remaining ground.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from ..patch_basis import System, grid
from ..release_modes import traction_modes


def trace_matrix(patch, local, normal):
    u, s, _, _, _, q = patch.basis(local)
    n, f = u.shape
    normal = np.broadcast_to(normal, (n, 2))
    a = np.zeros((n, 4, 6, f))
    a[:, 0, 0], a[:, 1, 1] = u, u
    a[:, 2, 2], a[:, 2, 3] = s*normal[:, 0:1], s*normal[:, 1:2]
    a[:, 3, 4], a[:, 3, 5] = s*normal[:, 0:1], s*normal[:, 1:2]
    return a.reshape(n, 4, 6*f), q


class PatchOperators:
    def __init__(self, model, n_volume, n_edge, boundary_weight, n_loads=3):
        self.model = model
        self.boundary_weight = boundary_weight
        self.patches = model.ground+model.liner
        for patch in self.patches:
            patch.n_weight = 6*patch.n_feature
        self.offset = np.r_[0, np.cumsum([p.n_weight for p in self.patches])]
        self.local = grid(n_volume)
        self.nq = len(self.local)
        self.bases = [p.basis(self.local) for p in self.patches]
        self.coordinates = np.concatenate([p.mapping(self.local)[0]*model.length
                                           for p in self.patches])
        self.ground_patch_indices = [i for i, p in enumerate(self.patches)
                                     if getattr(p, 'material_zone', p.zone) == 'ground']
        self.ground_rows = np.concatenate([np.arange(i*self.nq, (i+1)*self.nq)
                                           for i in self.ground_patch_indices])
        self.support_rows = np.setdiff1d(np.arange(len(self.patches)*self.nq), self.ground_rows)
        system = System(self.patches)
        for i, (patch, (_, s, gu, gs, _, q)) in enumerate(zip(self.patches, self.bases)):
            a = np.zeros((self.nq, 2, 6, patch.n_feature))
            a[:, 0, 2], a[:, 0, 3] = gs[:, 0], gs[:, 1]
            a[:, 1, 4], a[:, 1, 5] = gs[:, 0], gs[:, 1]
            a *= (1/q**3 if patch.zone == 'ground' else
                  np.full_like(q, patch.thickness))[:, None, None, None]
            system.add([(i, a.reshape(self.nq*2, -1))], np.zeros(self.nq*2),
                       'piola_equilibrium', 1/np.sqrt(self.nq))
        edge = np.polynomial.legendre.leggauss(n_edge)[0]
        n, ng = len(model.segments), len(model.ground)
        segment_number = {key: i for i, key in enumerate(model.segments)}
        self.support_index = {segment_number[(p.arc_id, p.fraction)]: ng+i
                              for i, p in enumerate(model.liner)}
        weight, far_weight = model.displacement_trace_weight, model.far_displacement_trace_weight
        self.interface_points = []
        release_rows = []

        def couple(i, j, a, b, normal, label):
            ma, q = trace_matrix(self.patches[i], a, normal)
            mb, _ = trace_matrix(self.patches[j], b, normal)
            scale = np.column_stack((weight/q, weight/q, 1/q**2, 1/q**2))
            gi, gj = i in self.ground_patch_indices, j in self.ground_patch_indices
            first = system.n_row
            system.add([(i, (ma*scale[:, :, None]).reshape(-1, self.patches[i].n_weight)),
                        (j, -(mb*scale[:, :, None]).reshape(-1, self.patches[j].n_weight))],
                       np.zeros(len(a)*4), 'toe_ground_contact' if gi != gj else label,
                       boundary_weight/np.sqrt(len(a)))
            if gi != gj:
                release_rows.append((first, system.n_row, True, 1. if gi else -1.))
                self.interface_points.append((self.patches[i].mapping(a)[0], normal))

        for layer in range(len(model.radial_edges)-1):
            for segment in range(n):
                i, j = layer*n+segment, layer*n+(segment+1) % n
                a, b = np.column_stack((edge, np.ones(n_edge))), np.column_stack((edge, -np.ones(n_edge)))
                direction = self.patches[i].mapping(a)[1][:, 0]
                normal = np.column_stack((-direction[:, 1], direction[:, 0]))
                normal /= np.linalg.norm(normal, axis=1)[:, None]
                couple(i, j, a, b, normal, 'angular_continuity')
        for segment in range(n):
            i = self.support_index.get(segment)
            j = self.support_index.get((segment+1) % n)
            if i is None and j is None:
                continue
            a, b = np.column_stack((edge, np.ones(n_edge))), np.column_stack((edge, -np.ones(n_edge)))
            end_index, end_local = (i, a) if i is not None else (j, b)
            direction = self.patches[end_index].mapping(end_local)[1][:, 0]
            normal = np.column_stack((-direction[:, 1], direction[:, 0]))
            normal /= np.linalg.norm(normal, axis=1)[:, None]
            if i is not None and j is not None:
                couple(i, j, a, b, normal, 'angular_continuity')
            else:
                trace, _ = trace_matrix(self.patches[end_index], end_local, normal)
                system.add([(end_index, trace[:, 2:].reshape(-1, self.patches[end_index].n_weight))],
                           np.zeros(n_edge*2), 'free_support_end', boundary_weight/np.sqrt(n_edge))
        for layer in range(len(model.radial_edges)-2):
            for segment in range(n):
                i, j = layer*n+segment, (layer+1)*n+segment
                a, b = np.column_stack((np.ones(n_edge), edge)), np.column_stack((-np.ones(n_edge), edge))
                couple(i, j, a, b, self.patches[i].mapping(a)[2], 'radial_continuity')
        for segment in range(n):
            inner, outer = np.column_stack((-np.ones(n_edge), edge)), np.column_stack((np.ones(n_edge), edge))
            normal = self.patches[segment].mapping(inner)[2]
            ma, _ = trace_matrix(self.patches[segment], inner, normal)
            support_index = self.support_index.get(segment)
            is_floor = support_index in self.ground_patch_indices if support_index is not None else False
            first = system.n_row
            if support_index is not None:
                mb, _ = trace_matrix(self.patches[support_index], outer, normal)
                ma[:, :2] *= weight
                mb[:, :2] *= weight
                system.add([(segment, ma.reshape(-1, self.patches[segment].n_weight)),
                            (support_index, -mb.reshape(-1, self.patches[support_index].n_weight))],
                           np.zeros(n_edge*4), 'ground_floor_continuity' if is_floor else 'release_interface',
                           boundary_weight/np.sqrt(n_edge))
            else:
                system.add([(segment, ma[:, 2:].reshape(-1, self.patches[segment].n_weight))],
                           np.zeros(n_edge*2), 'unsupported_release', boundary_weight/np.sqrt(n_edge))
            if not is_floor:
                release_rows.append((first, system.n_row, support_index is not None, 1.))
                self.interface_points.append((self.patches[segment].mapping(inner)[0], normal))
            if support_index is not None:
                floor_normal = normal
                if is_floor:
                    tangent = self.patches[support_index].mapping(inner)[1][:, 1]
                    floor_normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))
                    floor_normal /= np.linalg.norm(floor_normal, axis=1)[:, None]
                mi, _ = trace_matrix(self.patches[support_index], inner, floor_normal)
                first = system.n_row
                system.add([(support_index, mi[:, 2:].reshape(-1, self.patches[support_index].n_weight))],
                           np.zeros(n_edge*2), 'excavation_floor' if is_floor else 'free_intrados',
                           boundary_weight/np.sqrt(n_edge))
                if is_floor:
                    release_rows.append((first, system.n_row, False, 1.))
                    self.interface_points.append((self.patches[support_index].mapping(inner)[0], floor_normal))
            far_index = ng-n+segment
            far, q = trace_matrix(self.patches[far_index], outer, normal)
            system.add([(far_index, (far[:, :2]/q[:, None, None]).reshape(-1, self.patches[far_index].n_weight))],
                       np.zeros(n_edge*2), 'far_displacement', boundary_weight*far_weight/np.sqrt(n_edge))
        self.linear, _ = system.arrays()
        self.groups = system.groups
        self.loads = np.zeros((system.n_row, n_loads))
        for (xy, normal), (first, last, supported, sign) in zip(self.interface_points, release_rows, strict=True):
            rhs = np.zeros((n_edge, 4 if supported else 2, n_loads))
            rhs[:, -2:] = sign*traction_modes(xy, normal, 1.)[:, :, :n_loads]
            self.loads[first:last] = rhs.reshape(-1, n_loads)*boundary_weight/np.sqrt(n_edge)

    def coefficients(self, x, index):
        return x[self.offset[index]:self.offset[index+1]].reshape(6, -1)

    def volume_fields(self, x):
        grad, piola = [], []
        for i, (u, s, gu, gs, normal, q) in enumerate(self.bases):
            c = self.coefficients(x, i)
            grad.append(np.einsum('qjf,if->qij', gu, c[:2])/self.model.physical['e_ground'])
            piola.append((s @ c[2:].T).reshape(-1, 2, 2))
        return np.concatenate(grad), np.concatenate(piola)

    def point(self, item, *, ground=False, derivative=False):
        index = next(i for i, (name, bounds) in enumerate(self.model.segments)
                     if name == item['arc'] and bounds[0]-1e-12 <= item['fraction'] <= bounds[1]+1e-12)
        if not ground:
            if index not in self.support_index:
                raise ValueError('Requested observation lies on support not yet installed')
            index = self.support_index[index]
        p = self.patches[index]
        b = 2*(item['fraction']-p.fraction[0])/(p.fraction[1]-p.fraction[0])-1
        a = -1. if ground else 1-2*item.get('depth', 0.)/self.model.section.thickness
        u, s, gu, _, _, _ = p.basis(np.array([[a, b]]))
        m = np.zeros((6, p.n_weight))
        for i in range(6):
            m[i, i*p.n_feature:(i+1)*p.n_feature] = (u if i < 2 else s)[0]
        m[:2] *= self.model.length/self.model.physical['e_ground']
        padded = sparse.hstack((sparse.csr_matrix((6, self.offset[index])), sparse.csr_matrix(m),
                               sparse.csr_matrix((6, self.offset[-1]-self.offset[index+1]))), format='csr')
        if derivative:
            return padded, index, gu[0]
        return padded

    def observations(self, x, layout, *, ground=False):
        values, jacobian = [], []
        for o in layout['baseline']:
            if o['kind'] == 'component':
                row = self.point(o, ground=ground)[o['component']]
                value = float((row @ x)[0])
            elif o['kind'] == 'settlement':
                row = -self.point(o, ground=ground)[1]
                value = float((row @ x)[0])
            else:
                left, right = [self.point(p, ground=ground)[:2] for p in o['endpoints']]
                delta = np.asarray(o['xy_m'][1])-o['xy_m'][0]+(right-left) @ x
                length = np.linalg.norm(delta)
                value = o['original_distance_m']-length
                row = sparse.csr_matrix([delta/length]) @ (left-right)
            values.append(value)
            jacobian.append(row)
        return np.array(values), sparse.vstack(jacobian, format='csr')
