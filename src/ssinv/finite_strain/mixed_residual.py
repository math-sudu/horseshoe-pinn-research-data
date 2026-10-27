"""Material consistency in the mixed Piola PINN.

Equilibrium and returned material stress use the same immutable parent
history within each load increment.
"""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
from scipy.optimize import brentq
import torch

from . import dp_material as dp
from ..patch_basis import System


class Materials:
    def __init__(self, physical, config):
        self.device = config['device']
        self.e_ground = physical['physical']['e_ground']
        phi = np.deg2rad(config['ground']['friction_deg'])
        psi = np.deg2rad(config['ground']['dilatancy_deg'])
        alpha, beta = 6*np.sin(phi)/(3-np.sin(phi)), 6*np.sin(psi)/(3-np.sin(psi))
        k0 = 6*config['ground']['cohesion_mpa']*np.cos(phi)/(3-np.sin(phi))
        self.ground = dp.DPParameters.from_young_poisson(self.e_ground,
            physical['poisson']['ground'], alpha, beta, k0, config['ground']['hardening_mpa'])
        e, nu = physical['physical']['e_support'], physical['poisson']['support']
        self.support = SimpleNamespace(shear=e/(2*(1+nu)), bulk=e/(3*(1-2*nu)))
        # Hencky precompression for the prescribed initial stress ratio.
        vertical = config['ground']['initial_vertical_compression_mpa']
        sigma = -vertical*np.array([config['ground']['initial_horizontal_ratio'], 1.])
        g, lam = self.ground.shear, self.ground.bulk-2*self.ground.shear/3
        elastic_log = np.linalg.solve(np.array([[2*g+lam, lam], [lam, 2*g+lam]]), sigma)
        total = brentq(lambda t: t*np.exp(-t)-elastic_log.sum(), -1., 0.)
        e0 = np.exp(total)*elastic_log
        self.f0 = self.tensor(np.diag([*np.exp(e0), 1.]))
        self.j0 = torch.linalg.det(self.f0)
        initial = dp.evaluate(self.f0, dp.MaterialState.virgin((), device=self.device), self.ground)
        if int(initial.branch) != 0:
            raise ValueError('Engineering initial stress is outside the elastic initial-state domain')
        self.initial_stress = initial.sigma.detach()

    def tensor(self, array, *, grad=False):
        return torch.tensor(array, device=self.device, dtype=torch.float64, requires_grad=grad)

    def elastic(self, f, birth=None):
        elastic_f = f if birth is None else f @ torch.linalg.inv(birth)
        strain = .5*dp.log_spd_plane(elastic_f @ elastic_f.transpose(-1, -2))
        tau, _ = dp.hencky_response(strain, self.support)
        sigma = tau/torch.linalg.det(elastic_f)[..., None, None]
        # Piola is per original engineering area, even when support is born
        # in a later configuration. Stress depends only on F * inverse(Fbirth).
        return dp.cauchy_to_piola(f, sigma), sigma

    def returned(self, gradients, history, ng, *, jacobian=True, support_birth=None, ground_indices=None):
        g = self.tensor(gradients, grad=jacobian)
        f = dp.plane_tensor(g+torch.eye(2, device=self.device, dtype=torch.float64))
        dp.check_deformation(f)
        gi = np.arange(ng) if ground_indices is None else np.asarray(ground_indices)
        si = np.setdiff1d(np.arange(len(gradients)), gi)
        total_ground_f = f[gi] @ self.f0
        trial = history.trial(total_ground_f, history.x, history.ids, history.state.step, self.ground)
        # Source P is per natural area. Convert to the fixed engineering
        # reference used by the existing support geometry before coupling.
        ground_p = trial.piola @ self.f0.T/self.j0-self.initial_stress
        birth = None if support_birth is None else dp.plane_tensor(self.tensor(support_birth))
        support_p, support_sigma = self.elastic(f[si], birth)
        piola = torch.empty_like(f).index_copy(0, torch.as_tensor(gi, device=self.device), ground_p)
        piola = piola.index_copy(0, torch.as_tensor(si, device=self.device), support_p)[:, :2, :2]
        tangent = None
        if jacobian:
            tangent = torch.stack([torch.autograd.grad(piola[:, i, j].sum(), g,
                                   retain_graph=True)[0].flatten(1)
                                   for i in range(2) for j in range(2)], 1).detach().cpu().numpy()
        info = dict(j_min=float(torch.linalg.det(f).min().detach()),
                    ground_kappa_max=float(trial.kappa.max().detach()),
                    ground_plastic_points=int((trial.kappa > 1e-7).sum()),
                    ground_points=ng,
                    ground_yield_max_mpa=float(trial.yield_value.max().detach()))
        return piola.detach().cpu().numpy(), tangent, trial, info


def assemble_material(operators, x, returned, tangent=None):
    system = System(operators.patches)
    _, predicted = operators.volume_fields(x)
    residuals = []
    for i, (patch, basis) in enumerate(zip(operators.patches, operators.bases)):
        _, s, gu, _, _, q = basis
        at = slice(i*operators.nq, (i+1)*operators.nq)
        scale = (1/q**2 if patch.zone == 'ground' else np.ones_like(q))/np.sqrt(operators.nq)
        residuals.append(((predicted[at]-returned[at]).reshape(-1, 4)*scale[:, None]).ravel())
        if tangent is None:
            continue
        a = np.zeros((operators.nq, 4, 6, patch.n_feature))
        for c in range(4):
            a[:, c, c+2] = s
            for component in range(2):
                a[:, c, component] = -np.einsum('qj,qjf->qf',
                    tangent[at, c, component*2:component*2+2], gu)/operators.model.physical['e_ground']
        a *= scale[:, None, None, None]
        system.add([(i, a.reshape(operators.nq*4, -1))], np.zeros(operators.nq*4), 'material_consistency')
    matrix = system.arrays()[0] if tangent is not None else None
    return np.concatenate(residuals), matrix
