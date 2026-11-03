"""Displacement finite elements for the closed ground--support reference problem.

The material update uses principal elastic stretches and an inverse plastic
metric. Geometry, weak assembly, initialization and material history are
independent of the mixed-field solver. Units are metres and MPa.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.special import lambertw
from pypardiso import PyPardisoSolver


def plane(a):
    out = np.zeros(a.shape[:-2] + (3, 3))
    out[..., :2, :2] = a
    out[..., 2, 2] = 1.
    return out


def principal_tensor(values, vectors):
    return (vectors * values[..., None, :]) @ vectors.swapaxes(-1, -2)


class SpectralMaterial:
    """Hencky elasticity and zero-dilatancy, perfectly plastic Cauchy DP.

    For the specified smooth cone, the principal deviatoric Kirchhoff stress
    is shortened to alpha*p_tau + J*k. The unchanged volumetric elastic
    strain enforces zero plastic volume change. No material routine from the
    mixed-field implementation is called.
    """
    def __init__(self, config):
        physical, poisson, ground = config['physical'], config['poisson'], config['ground']
        if ground['dilatancy_deg'] != 0 or ground['hardening_mpa'] != 0:
            raise ValueError('This reference implements the prescribed zero-dilatancy perfect plasticity')
        self.gg = physical['e_ground'] / (2 * (1 + poisson['ground']))
        self.kg = physical['e_ground'] / (3 * (1 - 2 * poisson['ground']))
        self.gs = physical['e_support'] / (2 * (1 + poisson['support']))
        self.ks = physical['e_support'] / (3 * (1 - 2 * poisson['support']))
        phi = np.deg2rad(ground['friction_deg'])
        self.alpha = 6 * np.sin(phi) / (3 - np.sin(phi))
        self.strength = 6 * ground['cohesion_mpa'] * np.cos(phi) / (3 - np.sin(phi))
        sy = ground['initial_vertical_compression_mpa']
        sx = sy * ground['initial_horizontal_ratio']
        self.sigma0 = -np.diag([sx, sy, poisson['ground'] * (sx + sy)])
        lam = self.kg - 2 * self.gg / 3
        v = np.linalg.solve([[2*self.gg+lam, lam], [lam, 2*self.gg+lam]], [-sx, -sy])
        # t = tr(log F0), t*exp(-t)=sum(v), hence t=-W(-sum(v)).
        total = -float(lambertw(-v.sum()).real)
        self.f0 = np.diag([*np.exp(np.exp(total) * v), 1.])
        self.j0 = np.linalg.det(self.f0)
        d = np.diag(self.sigma0) - np.trace(self.sigma0)/3
        if np.sqrt(1.5 * (d*d).sum()) + self.alpha*np.trace(self.sigma0)/3 >= self.strength:
            raise ValueError('The prescribed initial state must be strictly elastic')

    def ground(self, f, inverse_cp, *, commit=False):
        ft = f @ self.f0
        jg = np.linalg.det(ft)
        if np.any(jg <= 0):
            raise ValueError('Nonpositive deformation determinant')
        b = ft @ inverse_cp @ ft.swapaxes(-1, -2)
        stretches2, directions = np.linalg.eigh(b)
        if np.any(stretches2 <= 0):
            raise ValueError('Nonpositive elastic metric')
        strain = .5 * np.log(stretches2)
        volume = strain.sum(axis=-1)
        dev = strain - volume[..., None]/3
        qt = self.gg * np.sqrt(6 * (dev*dev).sum(axis=-1))
        limit = -self.alpha * self.kg * volume + jg*self.strength
        if np.any(limit <= 0):
            raise ValueError('Tensile apex is outside the reference material domain')
        plastic = qt > limit
        ratio = np.ones_like(qt)
        np.divide(limit, qt, out=ratio, where=plastic)
        corrected = dev*ratio[..., None] + volume[..., None]/3
        tau = principal_tensor(2*self.gg*dev*ratio[..., None] + self.kg*volume[..., None], directions)
        sigma = tau / jg[..., None, None]
        # Direct engineering-area formula, independent of the natural-P pullback.
        piola = np.linalg.det(f)[..., None, None] * sigma @ np.linalg.inv(f).swapaxes(-1, -2)
        piola -= self.sigma0
        if not commit:
            return piola[..., :2, :2]
        fi = np.linalg.inv(ft)
        next_inverse_cp = fi @ principal_tensor(np.exp(2*corrected), directions) @ fi.swapaxes(-1, -2)
        next_inverse_cp = np.where(plastic[..., None, None], next_inverse_cp, inverse_cp)
        dl = np.maximum(qt-limit, 0)/(3*self.gg)
        return piola[..., :2, :2], sigma, next_inverse_cp, dl

    def support(self, f, *, cauchy=False):
        j = np.linalg.det(f)
        if np.any(j <= 0):
            raise ValueError('Nonpositive deformation determinant')
        eigenvalues, directions = np.linalg.eigh(f @ f.swapaxes(-1, -2))
        strain = .5*np.log(eigenvalues)
        tr = strain.sum(axis=-1)
        tau = principal_tensor(2*self.gs*(strain-tr[..., None]/3) + self.ks*tr[..., None], directions)
        if cauchy:
            return tau/j[..., None, None]
        return (tau @ np.linalg.inv(f).swapaxes(-1, -2))[..., :2, :2]


def shape(local, degree=2):
    """Tensor-product Lagrange displacement shape functions, radial first."""
    a, b = np.asarray(local).T
    def basis(x):
        nodes = np.linspace(-1., 1., degree+1)
        polynomials = []
        for i, node in enumerate(nodes):
            polynomial = np.polynomial.Polynomial.fromroots(np.delete(nodes, i))
            polynomials.append(polynomial/polynomial(node))
        return (np.stack([p(x) for p in polynomials], -1),
                np.stack([p.deriv()(x) for p in polynomials], -1))
    la, da = basis(a)
    lb, db = basis(b)
    count = (degree+1)**2
    value = (la[:, :, None]*lb[:, None, :]).reshape(-1, count)
    grad = np.stack(((da[:, :, None]*lb[:, None, :]).reshape(-1, count),
                     (la[:, :, None]*db[:, None, :]).reshape(-1, count)), -1)
    return value, grad


class ReferenceMesh:
    """Conforming displacement grid with exact circular contour maps."""
    def __init__(self, geometry, config, spacing=.5, radial_first=.015,
                 radial_growth=1.3, support_layers=2, degree=2):
        if geometry['scale'] != 1 or any(geometry['translation_m']):
            raise ValueError('Reference mesh currently uses the original untranslated metre geometry')
        if config['support']['closure'] != 'closed':
            raise ValueError('This reference problem requires the closed support')
        boundaries = {b['role']: b for b in geometry['boundaries']}
        self.arcs = boundaries['initial_support_outer']['circular_arcs']
        self.thickness = (self.arcs[0]['radius'] -
                          boundaries['initial_support_inner']['circular_arcs'][0]['radius'])
        self.length = boundaries['initial_support_outer']['width']/2
        self.support_layers = support_layers
        if degree not in (2, 3, 4):
            raise ValueError('Use polynomial degree 2, 3 or 4')
        self.degree = degree
        self.local_dofs = 2*(degree+1)**2
        self.segments = []
        for arc in self.arcs:
            count = max(2, 2*int(np.ceil(arc['radius']*arc['sweep_rad']/(2*spacing))))
            self.segments.extend((arc['arc_id'], i/count, (i+1)/count) for i in range(count))
        self.nt = len(self.segments)
        distance, increment = [1.], radial_first
        far = config['training']['far_factor']
        while distance[-1] < far:
            distance.append(min(far, distance[-1]+increment))
            increment *= radial_growth
        self.ranges = [('support', a, b) for a, b in zip(
            np.linspace(-self.thickness, 0., support_layers+1)[:-1],
            np.linspace(-self.thickness, 0., support_layers+1)[1:])]
        self.ranges += [('ground', a, b) for a, b in zip(np.log(distance[:-1]), np.log(distance[1:]))]
        self.nr = len(self.ranges)
        self.elements = [(layer, segment) for layer in range(self.nr) for segment in range(self.nt)]
        conn = []
        for layer, segment in self.elements:
            conn.append([(degree*layer+i)*degree*self.nt+(degree*segment+j) % (degree*self.nt)
                         for i in range(degree+1) for j in range(degree+1)])
        self.conn = np.array(conn)
        self.dofs = (2*self.conn[..., None]+np.arange(2)).reshape(-1, self.local_dofs)
        self.n_nodes = (degree*self.nr+1)*degree*self.nt
        self.free = np.arange(2*(self.n_nodes-degree*self.nt))
        self.support = np.arange(support_layers*self.nt)
        self.ground = np.arange(support_layers*self.nt, len(self.elements))
        z, w = np.polynomial.legendre.leggauss(degree+1)
        self.local = np.array([(a, b) for a in z for b in z])
        self.qweights = (w[:, None]*w[None, :]).ravel()
        self.basis, dlocal = shape(self.local, degree)
        coords, gradients, weights = [], [], []
        for e in range(len(self.elements)):
            xy, jac = self.mapping(e, self.local)
            determinant = np.linalg.det(jac)
            if np.any(determinant <= 0):
                raise ValueError('Inverted mesh map')
            coords.append(xy)
            gradients.append(dlocal @ np.linalg.inv(jac))
            weights.append(determinant*self.qweights)
        self.coordinates = np.array(coords)
        self.gradient = np.array(gradients)
        self.weights = np.array(weights)
        self.bmatrix = np.zeros((len(conn), len(self.local), 4, self.local_dofs))
        for component in range(2):
            for derivative in range(2):
                self.bmatrix[:, :, 2*component+derivative, component::2] = self.gradient[:, :, :, derivative]
        self.btranspose = self.bmatrix.transpose(0, 3, 1, 2).reshape(len(conn), self.local_dofs, -1)
        self.rows = np.broadcast_to(self.dofs[:, :, None], (len(conn), self.local_dofs, self.local_dofs)).ravel()
        self.cols = np.broadcast_to(self.dofs[:, None, :], (len(conn), self.local_dofs, self.local_dofs)).ravel()
        self.load_modes = self.interface_loads()

    def mapping(self, element, local):
        layer, segment = self.elements[element]
        name, f0, f1 = self.segments[segment]
        arc = next(a for a in self.arcs if a['arc_id'] == name)
        a, b = np.asarray(local).T
        angle = arc['start_rad'] + arc['sweep_rad']*(f0+(b+1)*(f1-f0)/2)
        angle_derivative = arc['sweep_rad']*(f1-f0)/2
        normal = np.column_stack((np.cos(angle), np.sin(angle)))
        tangent = np.column_stack((-np.sin(angle), np.cos(angle)))
        zone, r0, r1 = self.ranges[layer]
        param = r0+(a+1)*(r1-r0)/2
        center, radius = np.array(arc['center']), arc['radius']
        if zone == 'support':
            xy = center + (radius+param)[:, None]*normal
            da = (r1-r0)/2*normal
            db = (radius+param)[:, None]*angle_derivative*tangent
        else:
            factor = np.exp(param)
            xy = factor[:, None]*(center+radius*normal)
            da = (r1-r0)/2*xy
            db = (factor*radius*angle_derivative)[:, None]*tangent
        return xy, np.stack((da, db), -1)

    def interface_loads(self):
        z, w = np.polynomial.legendre.leggauss(self.degree+2)
        local = np.column_stack((np.ones(len(z)), z))
        n, _ = shape(local, self.degree)
        load = np.zeros((2*self.n_nodes, 4))
        for segment in range(self.nt):
            e = (self.support_layers-1)*self.nt+segment
            xy, jac = self.mapping(e, local)
            tangent = jac[:, :, 1]
            measure = np.linalg.norm(tangent, axis=-1)
            normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))/measure[:, None]
            t = np.zeros((len(z), 2, 4))
            t[:, 0, 0], t[:, 1, 1] = normal[:, 0], normal[:, 1]
            t[:, 0, 2] = xy[:, 1]/self.length*normal[:, 0]
            t[:, 1, 3] = xy[:, 0]/self.length*normal[:, 1]
            # The summed interface traction is -(Delta Pg - Ps) N.
            local_load = -np.einsum('qa,qic,q->aic', n, t, measure*w).reshape(self.local_dofs, 4)
            np.add.at(load, self.dofs[e], local_load)
        return load


class ReferenceFEM:
    def __init__(self, mesh, config):
        self.mesh = mesh
        self.material = SpectralMaterial(config)
        self.u = np.zeros((mesh.n_nodes, 2))
        self.inverse_cp = np.broadcast_to(np.eye(3), (len(mesh.ground), len(mesh.local), 3, 3)).copy()
        self.kappa = np.zeros((len(mesh.ground), len(mesh.local)))
        self.accepted = []
        self.metrics = []
        self.solver = PyPardisoSolver(mtype=11)

    def stresses(self, f, inverse_cp=None):
        m = self.mesh
        p = np.empty(f.shape[:-2]+(2, 2))
        p[m.ground] = self.material.ground(f[m.ground], self.inverse_cp if inverse_cp is None else inverse_cp)
        p[m.support] = self.material.support(f[m.support])
        return p

    def residual(self, u, loads, *, tangent=False):
        m = self.mesh
        f = plane(np.eye(2)+np.einsum('eai,eqaj->eqij', u[m.conn], m.gradient))
        p = self.stresses(f)
        internal = np.einsum('eqij,eqaj,eq->eai', p, m.gradient, m.weights).reshape(-1, m.local_dofs)
        r = -m.load_modes @ loads
        np.add.at(r, m.dofs.ravel(), internal.ravel())
        if not tangent:
            return r, f
        h = 2e-7
        c = np.empty(f.shape[:-2]+(2, 2, 2, 2))
        for k in range(2):
            for l in range(2):
                df = np.zeros((3, 3))
                df[k, l] = h
                c[..., k, l] = (self.stresses(f+df)-self.stresses(f-df))/(2*h)
        material_matrix = c.reshape(len(m.elements), len(m.local), 4, 4)
        product = (material_matrix @ m.bmatrix) * m.weights[:, :, None, None]
        stiffness = m.btranspose @ product.reshape(len(m.elements), -1, m.local_dofs)
        matrix = sparse.coo_matrix((stiffness.ravel(), (m.rows, m.cols)),
                                   shape=(2*m.n_nodes, 2*m.n_nodes)).tocsr()
        return r, f, matrix

    def solve(self, final_loads, steps, emit=print, tolerance=2e-8):
        m = self.mesh
        for step in range(1, steps+1):
            loads = np.asarray(final_loads)*step/steps
            norm_load = np.linalg.norm((m.load_modes @ loads)[m.free])
            for iteration in range(35):
                r, f, matrix = self.residual(self.u, loads, tangent=True)
                residual = np.linalg.norm(r[m.free])/norm_load
                if residual < tolerance:
                    break
                delta = np.zeros(2*m.n_nodes)
                delta[m.free] = self.solver.solve(matrix[m.free][:, m.free], -r[m.free])
                line = 1.
                while line >= 1/1024:
                    candidate = self.u+line*delta.reshape(-1, 2)
                    try:
                        new_r, _ = self.residual(candidate, loads)
                        new_norm = np.linalg.norm(new_r[m.free])/norm_load
                    except ValueError:
                        new_norm = np.inf
                    if new_norm < residual:
                        self.u = candidate
                        break
                    line /= 2
                else:
                    raise RuntimeError(f'Newton line search failed at step {step}, residual {residual}')
            else:
                raise RuntimeError(f'Newton failed at step {step}, residual {residual}')
            _, _, self.inverse_cp, dl = self.material.ground(f[m.ground], self.inverse_cp, commit=True)
            self.kappa += dl
            self.accepted.append(self.u.copy())
            record = dict(step=step, iterations=iteration, residual_relative=float(residual),
                          min_j=float(np.linalg.det(f).min()),
                          plastic_points=int((self.kappa > 1e-7).sum()),
                          max_kappa=float(self.kappa.max()))
            self.metrics.append(record)
            emit(record)

    def values(self, points):
        """Support stresses computed from displacement gradients, with trace averaging."""
        m = self.mesh
        result = dict(displacement_m=[], pressure_mpa=[], compression_mpa=[], shear_mpa=[])
        for point in points:
            name, fraction = point['arc'], point['fraction']
            depth = point.get('depth', 0.)
            entries = []
            for segment, (arc, f0, f1) in enumerate(m.segments):
                if arc != name or not f0-1e-12 <= fraction <= f1+1e-12:
                    continue
                for layer, (_, r0, r1) in enumerate(m.ranges[:m.support_layers]):
                    if not r0-1e-12 <= -depth <= r1+1e-12:
                        continue
                    e = layer*m.nt+segment
                    local = np.array([[2*(-depth-r0)/(r1-r0)-1, 2*(fraction-f0)/(f1-f0)-1]])
                    n, grad = shape(local, m.degree)
                    _, jac = m.mapping(e, local)
                    grad = grad @ np.linalg.inv(jac)
                    displacement = (n @ self.u[m.conn[e]])[0]
                    f = np.eye(2)+self.u[m.conn[e]].T @ grad[0]
                    sigma = self.material.support(plane(f), cauchy=True)[:2, :2]
                    arc_data = next(a for a in m.arcs if a['arc_id'] == name)
                    angle = arc_data['start_rad']+fraction*arc_data['sweep_rad']
                    normal = np.array([np.cos(angle), np.sin(angle)])
                    tangent = np.array([-normal[1], normal[0]])
                    normal = np.linalg.solve(f.T, normal)
                    normal /= np.linalg.norm(normal)
                    tangent = f @ tangent
                    tangent /= np.linalg.norm(tangent)
                    entries.append([*displacement, -normal @ sigma @ normal,
                                    -tangent @ sigma @ tangent, tangent @ sigma @ normal])
            if not entries:
                raise ValueError(f'Point outside the support: {point}')
            row = np.mean(entries, axis=0)
            result['displacement_m'].append(row[:2])
            for key, item in zip(('pressure_mpa', 'compression_mpa', 'shear_mpa'), row[2:]):
                result[key].append(item)
        return {key: np.array(value) for key, value in result.items()}
