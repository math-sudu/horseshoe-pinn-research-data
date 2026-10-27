"""Contour-following neural features and sparse residual assembly."""
from __future__ import annotations
import numpy as np
from scipy import linalg, sparse


def grid(n):
    """Gauss points exclude patch edges, whose one-sided traces are separate."""
    x = np.polynomial.legendre.leggauss(n)[0]
    a, b = np.meshgrid(x, x, indexing="ij")
    return np.column_stack((a.ravel(), b.ravel()))


class LocalNetwork:
    def __init__(self, section, arc_id, fraction, radial, zone, width, seed, amplitude=1.):
        self.arc_id, self.fraction, self.radial, self.zone = arc_id, tuple(fraction), tuple(radial), zone
        self.length = section.scale*section.outer["width"]/2
        arc = section.arc("initial_support_outer", arc_id)
        self.center = section.scale*np.asarray(arc["center"])/self.length
        self.radius = section.scale*arc["radius"]/self.length
        self.thickness = section.thickness/self.length
        self.phi0 = arc["start_rad"]+fraction[0]*arc["sweep_rad"]
        self.phi1 = arc["start_rad"]+fraction[1]*arc["sweep_rad"]
        rng = np.random.default_rng(seed)
        self.weight = rng.uniform(-amplitude, amplitude, (2, width))
        self.bias = rng.uniform(-amplitude, amplitude, width)
        # Invertible change of output coordinates, dropping numerically null
        # hidden features. It uses coordinate values only, never solution data.
        points = grid(max(18, int(np.ceil(np.sqrt(width)*2))))
        value, _ = self.hidden(points)
        _, singular, vt = linalg.svd(value, full_matrices=False)
        retain = singular > 1e-10*singular[0]
        self.transform = vt[retain].T/singular[retain]*np.sqrt(len(points))
        self.n_feature = self.transform.shape[1]
        self.n_weight = self.n_feature*6

    def hidden(self, local):
        y = np.tanh(local @ self.weight+self.bias)
        value = np.column_stack((np.ones(len(local)), local, y))
        derivative = np.zeros((len(local), 2, value.shape[1]))
        derivative[:, 0, 1], derivative[:, 1, 2] = 1., 1.
        derivative[:, :, 3:] = (1-y*y)[:, None, :]*self.weight[None, :, :]
        return value, derivative

    def mapping(self, local):
        a, b = local.T
        half_phi = (self.phi1-self.phi0)/2
        phi = (self.phi0+self.phi1)/2+b*half_phi
        normal = np.column_stack((np.cos(phi), np.sin(phi)))
        tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
        if self.zone == "liner":
            radius = self.radius-self.thickness*(1-a)/2
            point = self.center+radius[:, None]*normal
            da = self.thickness/2*normal
            db = (radius*half_phi)[:, None]*tangent
            q = np.ones(len(local))
        else:
            half_log = (self.radial[1]-self.radial[0])/2
            distance = np.exp((self.radial[0]+self.radial[1])/2+a*half_log)
            point = distance[:, None]*(self.center+self.radius*normal)
            da = point*half_log
            db = (distance*self.radius*half_phi)[:, None]*tangent
            q = 1/distance
        return point, np.stack((da, db), 1), normal, q

    def basis(self, local):
        h, dh = self.hidden(local)
        h, dh = h @ self.transform, dh @ self.transform
        _, jac, normal, q = self.mapping(local)
        if self.zone == "ground":
            half_log = (self.radial[1]-self.radial[0])/2
            u, s = h*q[:, None], h*q[:, None]**2
            du, ds = dh*q[:, None, None], dh*q[:, None, None]**2
            du[:, 0] -= half_log*u
            ds[:, 0] -= 2*half_log*s
        else:
            u, s, du, ds = h, h, dh, dh
        inv = np.linalg.inv(jac)
        gu = np.einsum("nij,njf->nif", inv, du)
        gs = np.einsum("nij,njf->nif", inv, ds)
        return u, s, gu, gs, normal, q


class System:
    def __init__(self, patches):
        self.offset = np.cumsum([0]+[p.n_weight for p in patches])
        self.rows, self.cols, self.values, self.rhs, self.groups = [], [], [], [], []
        self.n_row = 0

    def add(self, terms, rhs, label, weight=1.):
        count = len(rhs)
        for index, matrix in terms:
            row, col = np.nonzero(matrix)
            self.rows.append(row+self.n_row)
            self.cols.append(col+self.offset[index])
            self.values.append(matrix[row, col]*weight)
        self.rhs.append(np.asarray(rhs)*weight)
        self.groups.append((label, self.n_row, self.n_row+count))
        self.n_row += count

    def arrays(self):
        a = sparse.coo_matrix((np.concatenate(self.values),
                               (np.concatenate(self.rows), np.concatenate(self.cols))),
                              shape=(self.n_row, self.offset[-1])).tocsr()
        return a, np.concatenate(self.rhs)


class PatchModel:
    def __init__(self, section, physical, width=64, patch_length=5., radial_bands=4,
                 far_factor=100., seed=71, amplitude=1., nu_ground=.255, nu_support=.2,
                 displacement_trace_weight=1., far_displacement_trace_weight=None):
        self.section, self.physical = section, dict(physical)
        self.length = section.scale*section.outer["width"]/2
        self.nu_ground, self.nu_support = nu_ground, nu_support
        self.far_factor = far_factor
        self.displacement_trace_weight = float(displacement_trace_weight)
        self.far_displacement_trace_weight = float(displacement_trace_weight if far_displacement_trace_weight is None
                                                  else far_displacement_trace_weight)
        self.segments = []
        for arc in section.outer["circular_arcs"]:
            n = max(1, int(np.ceil(section.scale*arc["radius"]*arc["sweep_rad"]/patch_length)))
            for i in range(n):
                self.segments.append((arc["arc_id"], (i/n, (i+1)/n)))
        self.radial_edges = np.linspace(0., np.log(far_factor), radial_bands+1)
        self.ground, self.liner = [], []
        for layer in range(radial_bands):
            for arc, fraction in self.segments:
                self.ground.append(LocalNetwork(section, arc, fraction,
                    self.radial_edges[layer:layer+2], "ground", width, seed+len(self.ground), amplitude))
        for arc, fraction in self.segments:
            self.liner.append(LocalNetwork(section, arc, fraction, (0., 1.), "liner", width,
                                            seed+len(self.ground)+len(self.liner), amplitude))



def make_model(section, config, seed, pilot=False):
    t = config['training']
    physical = dict(config['physical'],sigma_v=1.,k0=1.)
    return PatchModel(section,physical,width=32 if pilot else t['width'],
          patch_length=t['patch_length'],radial_bands=t['radial_bands'],
          far_factor=t['far_factor'],seed=seed,amplitude=t['amplitude'],
          nu_ground=config['poisson']['ground'],nu_support=config['poisson']['support'],
          displacement_trace_weight=t['displacement_trace_weight'],
          far_displacement_trace_weight=t['far_displacement_trace_weight'])
