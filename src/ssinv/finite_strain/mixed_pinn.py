"""Supported-section mixed PINN with accepted finite-strain material history.

Contour patches with fixed tanh features represent displacement and first
Piola stress. Output weights and release loads are solved at each nonlinear
iterate; material histories are committed after an increment is accepted.
"""
from __future__ import annotations

import json
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize
import torch

from ..patch_basis import make_model, LocalNetwork, grid
from . import dp_material as dp
from .current_traction import current_frame
from .mixed_geometry import PatchOperators
from .mixed_history import PointHistory, accept_all, content_identity
from .mixed_residual import Materials, assemble_material
from .bench_geometry import GroundBenchNetwork


class FieldPINN:
    def __init__(self, section, physical, layout, config):
        self.config, self.layout = config, layout
        settings = dict(physical, training=dict(physical['training']))
        settings['training'].update(config['training'])
        self.model = make_model(section, settings, physical['feature_seed'])
        toe_width = settings['training'].get('toe_width')
        if toe_width is not None:
            for zone in ('ground', 'liner'):
                patches = getattr(self.model, zone)
                seed_offset = 0 if zone == 'ground' else len(self.model.ground)
                for i, p in enumerate(patches):
                    if p.arc_id in ('left_knee', 'right_knee', 'invert') and (zone == 'liner' or p.radial[0] == 0.):
                        patches[i] = LocalNetwork(section, p.arc_id, p.fraction, p.radial, p.zone,
                            toe_width, physical['feature_seed']+seed_offset+i, settings['training']['amplitude'])
        floor = config.get('support', {}).get('floor') == 'remaining_ground_bench'
        if floor:
            self.model.liner = [GroundBenchNetwork(section, p.fraction, toe_width or settings['training']['width'],
                physical['feature_seed']+len(self.model.ground)+i, settings['training']['amplitude'])
                if p.arc_id == 'invert' else p for i, p in enumerate(self.model.liner)]
        if config.get('support', {}).get('closure') == 'open_invert':
            self.model.liner = [p for p in self.model.liner if p.arc_id != 'invert' or floor]
        elif config.get('support', {}).get('closure') == 'uninstalled':
            self.model.liner = [p for p in self.model.liner if floor and p.arc_id == 'invert']
        t = config['training']
        n_loads = config.get('release', {}).get('n_loads', 3)
        if n_loads not in (3, 4):
            raise ValueError('Release uses Qx, Qy, Gv and optionally Gh')
        self.operators = PatchOperators(self.model, t['n_volume'], t['n_edge'],
                                        settings['training']['boundary_weight'], n_loads)
        self.materials = Materials(physical, config)
        self.material_id = content_identity(dict(parameters=vars(self.materials.ground),
                                                initial_f=self.materials.f0))
        self.ng = len(self.operators.ground_rows)
        edge = np.polynomial.legendre.leggauss(t['n_edge']+4)[0]
        domain_local = np.vstack((self.operators.local, grid(t['n_volume']+2),
            np.column_stack((-np.ones_like(edge), edge)), np.column_stack((np.ones_like(edge), edge)),
            np.column_stack((edge, -np.ones_like(edge))), np.column_stack((edge, np.ones_like(edge)))))
        self.domain_patch_indices = self.operators.ground_patch_indices
        self.domain_bases = [self.operators.patches[i].basis(domain_local)[2] for i in self.domain_patch_indices]
        self.history = PointHistory('ground_volume',
            self.materials.tensor(self.operators.coordinates[self.operators.ground_rows]), 0.)
        self.x = np.zeros(self.operators.offset[-1])
        self.birth_coefficients = None
        self.prescribed_loads = None
        self.damping = 0.
        self.loads = np.zeros(n_loads)
        self.load_basis = np.eye(n_loads)
        if config.get('release', {}).get('parameterization') == 'fixed_initial_stress_ratio':
            self.load_basis = np.array([[-self.materials.initial_stress[0, 0].item()],
                                       [-self.materials.initial_stress[1, 1].item()],
                                       *[[0.]]*(n_loads-2)])
        selected = config.get('observations', {}).get('fit_indices', list(range(len(layout['baseline']))))
        self.fit_indices = np.asarray(selected, dtype=int)
        self.release_matrix = None
        if config.get('release', {}).get('law') == 'bounded_initial_stress_release':
            # The affine horizontal mode reaches its extrema at the contour's
            # top/bottom. These bounds cover the entire interface, not just
            # collocation points. The residual virtual confinement cannot pull.
            y0, y1 = np.asarray([section.outer['bounds_min'][1],
                                section.outer['bounds_max'][1]])/self.model.length
            self.release_matrix = np.array([[1., 0., y0], [1., 0., y1], [0., 1., 0.]])
            self.release_capacity = -np.array([self.materials.initial_stress[0, 0].item(),
                self.materials.initial_stress[0, 0].item(), self.materials.initial_stress[1, 1].item()])
            if n_loads == 4:
                x0, x1 = np.asarray([section.outer['bounds_min'][0],
                                    section.outer['bounds_max'][0]])/self.model.length
                self.release_matrix = np.array([[1., 0., y0, 0.], [1., 0., y1, 0.],
                                                [0., 1., 0., x0], [0., 1., 0., x1]])
                self.release_capacity = -np.repeat(
                    self.materials.initial_stress[:2, :2].diagonal().cpu().numpy(), 2)
        self.accepted = []
        self.metrics = []
        self.load_rhs = np.vstack((self.operators.loads,
            np.zeros((len(self.operators.patches)*self.operators.nq*4, n_loads)))) @ self.load_basis

    def fit_observations(self, x):
        values, jacobian = self.operators.observations(x, self.layout,
            ground=not bool(len(self.operators.support_rows)))
        return values[self.fit_indices], jacobian[self.fit_indices]

    def set_support_birth(self, coefficients):
        self.birth_coefficients = np.asarray(coefficients).copy()
        self.material_id = content_identity(dict(parameters=vars(self.materials.ground),
            initial_f=self.materials.f0, support_birth=self.materials.tensor(self.birth_coefficients)))

    def support_birth(self, operators=None):
        if self.birth_coefficients is None:
            return None
        op = self.operators if operators is None else operators
        return np.eye(2)+op.volume_fields(self.birth_coefficients)[0][op.support_rows]

    def scalar_release_at_capacity(self):
        return (self.load_basis.shape[1] == 1 and self.release_matrix is not None and
                np.allclose(self.release_matrix @ self.loads, self.release_capacity, rtol=0., atol=1e-8))

    def domain_row(self, index, cofactor, j0):
        patch_index, point_index = divmod(index, len(self.domain_bases[0]))
        basis = self.domain_bases[patch_index][point_index]
        row = np.zeros_like(self.x)
        actual_patch = self.domain_patch_indices[patch_index]
        start = self.operators.offset[actual_patch]
        width = self.operators.patches[actual_patch].n_feature
        row[start:start+2*width] = (j0*cofactor[index] @ basis/self.model.physical['e_ground']).ravel()
        return row

    def enlarged_scalar_direction(self, directions, matrix, residual, observed,
                                  factor, scale, cap, emit):
        """Resolve an infeasible reduced inverse step with active field modes."""
        f = np.eye(2)+self.domain_gradients(self.x)
        j0 = float(self.materials.j0)
        cofactor = np.stack((np.stack((f[:, 1, 1], -f[:, 1, 0]), -1),
                             np.stack((-f[:, 0, 1], f[:, 0, 0]), -1)), -2)
        margin = self.domain_margin(self.x)
        nq = len(self.domain_bases[0])
        indices = [i*nq+int(np.argmin(values))
                   for i, values in enumerate(margin.reshape(-1, nq)) if values.min() < 1e-4]
        if not indices:
            raise RuntimeError('Reduced scalar step failed away from an active material-domain boundary')
        extra = []
        norm = max(float(np.linalg.norm(directions[:, 0])), 1.)
        for index in indices:
            row = self.domain_row(index, cofactor, j0)
            response = scale*factor.solve(scale*row)
            extra.append(response*norm/np.linalg.norm(response))
        basis = np.column_stack((directions, *extra))
        gradients = np.stack([self.domain_gradients(d) for d in basis.T], 1)
        prediction, observation = self.fit_observations(self.x)
        physics = matrix @ basis
        physics[:, 1:2] -= self.load_rhs
        weight = np.sqrt(self.config['training']['observation_weight'])
        displacement_scale = self.model.length/self.model.physical['e_ground']
        a = np.vstack((physics, weight*(observation @ basis)/displacement_scale))
        b = np.r_[-residual, weight*(observed-prediction)/displacement_scale]
        gram, rhs = a.T @ a, a.T @ b
        objective_scale = max(float(np.linalg.norm(gram, 2)), 1.)
        load_map = np.zeros((len(self.release_capacity), basis.shape[1]))
        load_map[:, 1:2] = self.release_matrix @ self.load_basis
        now = self.release_matrix @ self.loads

        def determinant(v):
            return j0*np.linalg.det(f+np.einsum('qkij,k->qij', gradients, v))

        def jacobian(v):
            value = f+np.einsum('qkij,k->qij', gradients, v)
            cof = np.stack((np.stack((value[:, 1, 1], -value[:, 1, 0]), -1),
                            np.stack((-value[:, 0, 1], value[:, 0, 0]), -1)), -2)
            return j0*np.einsum('qij,qkij->qk', cof, gradients)

        result = minimize(lambda v: (.5*v @ gram @ v-rhs @ v)/objective_scale,
            np.zeros(basis.shape[1]), jac=lambda v: (gram @ v-rhs)/objective_scale,
            method='SLSQP', constraints=[
                LinearConstraint(load_map, self.released_floor()-now, self.release_capacity-now),
                NonlinearConstraint(determinant, 1e-8, cap, jac=jacobian)],
            options=dict(ftol=1e-12, maxiter=100))
        if not result.success:
            raise RuntimeError('Expanded scalar material-domain step failed: '+result.message)
        emit(dict(event='expanded_scalar_domain_direction', additional_field_directions=len(extra),
                  maximum_determinant_violation=float(np.max(determinant(result.x)-cap))))
        return basis @ result.x, self.load_basis[:, 0]*result.x[1]

    def forward_domain_direction(self, direction, factor, scale, matrix, residual, emit):
        """Project the forward Gauss--Newton step onto active volume bounds.

        Once release is fixed, scaling a single Newton vector cannot turn
        around an active material-domain boundary. The normal-matrix metric
        supplies full-field constraint corrections without a new load mode.
        """
        m = self.materials.ground
        f = np.eye(2)+self.domain_gradients(self.x)
        j0 = float(self.materials.j0)
        j = j0*np.linalg.det(f)
        cofactor = np.stack((np.stack((f[:, 1, 1], -f[:, 1, 0]), -1),
                             np.stack((-f[:, 0, 1], f[:, 0, 0]), -1)), -2)
        cap = float(np.exp(m.k0/(m.alpha*m.bulk)))
        for _ in range(6):
            cap -= (np.log(cap)-m.k0*cap/(m.alpha*m.bulk)+1e-5/(m.alpha*m.bulk))/(1/cap-m.k0/(m.alpha*m.bulk))
        bound = cap-j

        def change(d):
            return j0*np.einsum('qij,qij->q', cofactor, self.domain_gradients(d))

        if np.max(change(direction)-bound) <= 1e-10:
            return direction
        active, rows, responses = [], [], []
        step = direction.copy()
        for _ in range(128):
            violation = change(step)-bound
            index = int(np.argmax(violation))
            if violation[index] <= 1e-10:
                # Enforce the exact determinant in this enlarged field space.
                # Linearized volume bounds alone give tiny line-search steps
                # when the current state is already on the curved boundary.
                sizes = np.array([np.linalg.norm(z) for z in responses])/np.linalg.norm(direction)
                basis = np.column_stack((direction, np.column_stack(responses)/sizes))
                gradients = np.stack([self.domain_gradients(d) for d in basis.T], 1)
                a = matrix @ basis
                gram, rhs = a.T @ a, -a.T @ residual
                objective_scale = max(float(np.linalg.norm(gram, 2)), 1.)

                def determinant(v):
                    return j0*np.linalg.det(f+np.einsum('qkij,k->qij', gradients, v))

                def jacobian(v):
                    value = f+np.einsum('qkij,k->qij', gradients, v)
                    cof = np.stack((np.stack((value[:, 1, 1], -value[:, 1, 0]), -1),
                                    np.stack((-value[:, 0, 1], value[:, 0, 0]), -1)), -2)
                    return j0*np.einsum('qij,qkij->qk', cof, gradients)

                result = minimize(lambda v: (.5*v @ gram @ v-rhs @ v)/objective_scale,
                    np.r_[1., -multipliers*sizes], jac=lambda v: (gram @ v-rhs)/objective_scale,
                    method='SLSQP', constraints=[NonlinearConstraint(determinant, 1e-8, cap, jac=jacobian)],
                    options=dict(ftol=1e-12, maxiter=100))
                if not result.success:
                    raise RuntimeError('Forward material-domain projection failed: '+result.message)
                emit(dict(event='forward_active_domain_projection', active_constraints=len(active),
                          maximum_determinant_violation=float(np.max(determinant(result.x)-cap))))
                return basis @ result.x
            if index in active:
                raise RuntimeError('Active material-domain projection lost numerical rank')
            row = self.domain_row(index, cofactor, j0)
            response = scale*factor.solve(scale*row)
            active.append(index)
            rows.append(row)
            responses.append(response)
            while active:
                g, z = np.stack(rows), np.column_stack(responses)
                multipliers = np.linalg.lstsq(g @ z, g @ direction-bound[active], rcond=1e-12)[0]
                if np.min(multipliers) >= -1e-10:
                    step = direction-z @ multipliers
                    break
                remove = int(np.argmin(multipliers))
                active.pop(remove)
                rows.pop(remove)
                responses.pop(remove)
            else:
                step = direction.copy()
        raise RuntimeError('Forward material-domain active set did not converge')

    def domain_gradients(self, x):
        return np.concatenate([np.einsum('qjf,if->qij', gu,
            self.operators.coefficients(x, i)[:2])/self.model.physical['e_ground']
            for i, gu in zip(self.domain_patch_indices, self.domain_bases, strict=True)])

    def domain_margin(self, x):
        m = self.materials.ground
        j = np.linalg.det(np.eye(2)+self.domain_gradients(x))*float(self.materials.j0)
        if np.any(j <= 0):
            raise ValueError('A material-domain sample reverses orientation')
        # For this fixed beta=0 case det(Cp)=1 along every accepted path.
        return j*m.k0-m.alpha*m.bulk*np.log(j)

    def released_floor(self):
        """Already removed virtual confinement cannot be restored by excavation."""
        return (self.release_matrix @ np.asarray(self.metrics[-1]['loads_mpa'])
                if self.metrics else np.zeros_like(self.release_capacity))

    def evaluate(self, x, loads, *, jacobian):
        if self.release_matrix is not None:
            released = self.release_matrix @ loads
            if (np.any(released < self.released_floor()-1e-8) or
                    np.any(released > self.release_capacity+1e-8)):
                raise ValueError('Released traction exceeds the available initial compression')
        domain_margin = float(self.domain_margin(x).min())
        if domain_margin <= 0:
            raise ValueError('A volume/edge material-domain sample reaches the unsupported tensile apex')
        gradients, _ = self.operators.volume_fields(x)
        returned, tangent, trial, info = self.materials.returned(gradients, self.history, self.ng,
            jacobian=jacobian, support_birth=self.support_birth(), ground_indices=self.operators.ground_rows)
        material_residual, cmatrix = assemble_material(self.operators, x, returned, tangent)
        linear = self.operators.linear @ x-self.operators.loads @ loads
        residual = np.r_[linear, material_residual]
        matrix = sparse.vstack((self.operators.linear, cmatrix), format='csr') if jacobian else None
        info['physics_weighted_rms_mpa'] = float(np.sqrt(np.mean(residual**2)))
        info['material_weighted_rms_mpa'] = float(np.sqrt(np.mean(material_residual**2)))
        info['linear_weighted_rms_mpa'] = float(np.sqrt(np.mean(linear**2)))
        info['domain_margin_min_mpa'] = domain_margin
        return residual, matrix, trial, info

    def newton_direction(self, matrix, residual, observed, emit):
        scale = 1/np.sqrt(np.asarray(matrix.power(2).sum(0)).ravel())
        scaled = matrix @ sparse.diags(scale)
        normal = (scaled.T @ scaled+(self.config['training']['ridge']+self.damping)*
                  sparse.eye(matrix.shape[1])).tocsc()
        rhs = scaled.T @ np.column_stack((-residual, self.load_rhs))
        emit(dict(event='factorization', unknowns=matrix.shape[1], equations=matrix.shape[0], nnz=normal.nnz))
        if self.config['training'].get('linear_solver', 'superlu') == 'pardiso':
            from ..sparse_spd import factor_spd
            with factor_spd(normal, self.config['training']['threads']) as factor:
                return self.direction_from_factor(matrix, residual, observed, emit, scale, normal, rhs, factor)
        # The regularized normal matrix is positive definite. Preserve its
        # symmetric ordering instead of introducing fill through row pivots.
        factor = splu(normal, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.,
                      options={'SymmetricMode': True})
        return self.direction_from_factor(matrix, residual, observed, emit, scale, normal, rhs, factor)

    def direction_from_factor(self, matrix, residual, observed, emit, scale, normal, rhs, factor):
        predicted, observation = self.fit_observations(self.x)
        z = factor.solve(rhs)
        for _ in range(2):
            z += factor.solve(rhs-normal @ z)
        directions = scale[:, None]*z
        if self.prescribed_loads is not None:
            return self.forward_domain_direction(directions[:, 0], factor, scale, matrix, residual, emit), np.zeros_like(self.loads)
        if self.scalar_release_at_capacity() and np.all(predicted <= observed):
            # The target still exceeds the response at full release. Recover
            # forward equilibrium before claiming it is unreachable. A later
            # iterate can leave this bound if its response crosses the target.
            emit(dict(event='full_release_forward_equilibrium'))
            return self.forward_domain_direction(directions[:, 0], factor, scale, matrix, residual, emit), np.zeros_like(self.loads)
        observation_response = observation @ directions
        count = self.load_basis.shape[1]
        if np.linalg.matrix_rank(observation_response[:, 1:]) != count:
            raise ValueError('Displacement-only load tangent has lost rank')
        # Use the same residual merit in the reduced step and line search.
        # The first coordinate scales the physical correction; the other
        # remaining coordinates are load increments. Zero is admissible.
        physics_response = matrix @ directions
        physics_response[:, 1:] -= self.load_rhs
        scale_obs = self.model.length/self.model.physical['e_ground']
        obs_weight = np.sqrt(self.config['training']['observation_weight'])
        a = np.vstack((physics_response, obs_weight*observation_response/scale_obs))
        b = np.r_[-residual, obs_weight*(observed-predicted)/scale_obs]
        correction = np.linalg.lstsq(a, b, rcond=None)[0]
        gram, reduced_rhs = a.T @ a, a.T @ b
        constraints = []
        need_constrained_step = False
        if self.release_matrix is not None:
            now = self.release_matrix @ self.loads
            load_map = np.column_stack((np.zeros(len(self.release_capacity)), self.release_matrix @ self.load_basis))
            floor = self.released_floor()
            constraints.append(LinearConstraint(load_map, floor-now, self.release_capacity-now))
            proposed_release = now+load_map @ correction
            need_constrained_step = bool(np.any(proposed_release < floor) or
                                         np.any(proposed_release > self.release_capacity))
        # The reused zero-dilatancy, perfectly plastic kernel has no tensile
        # apex return. Its smooth-cone domain is an actual material constraint,
        # not a reason to shrink an infeasible inverse direction indefinitely.
        # For beta=H=0: margin = J*k0 - alpha*K*log(Je) > 0.
        m = self.materials.ground
        if m.beta == 0 and m.hardening == 0:
            gradients = self.domain_gradients(self.x)
            f = np.eye(2)+gradients
            j = np.linalg.det(f)*float(self.materials.j0)
            cp_logdet = np.zeros(len(gradients))
            margin = j*m.k0-m.alpha*m.bulk*(np.log(j)-.5*cp_logdet)
            inverse = np.linalg.inv(f)
            changes = []
            for k in range(count+1):
                dg = self.domain_gradients(directions[:, k])
                changes.append((j*m.k0-m.alpha*m.bulk)*np.einsum('qij,qji->q', inverse, dg))
            changes = np.column_stack(changes)
            safety = 1e-5  # MPa: keep trial arithmetic strictly inside the existing cone domain.
            proposed = margin+changes @ correction
            if proposed.min() < safety or need_constrained_step:
                dg = np.stack([self.domain_gradients(directions[:, k])
                               for k in range(count+1)], 1)
                base_f, response_f = f, dg
                j0 = float(self.materials.j0)
                # Exact upper determinant at the existing tensile-domain
                # boundary. Newton solves this monotone local scalar equation.
                cap = np.exp(.5*cp_logdet+m.k0/(m.alpha*m.bulk))
                for _ in range(6):
                    cap -= (np.log(cap)-.5*cp_logdet-m.k0*cap/(m.alpha*m.bulk)
                            +safety/(m.alpha*m.bulk))/(1/cap-m.k0/(m.alpha*m.bulk))

                def determinant(v):
                    fv = base_f+np.einsum('qkij,k->qij', response_f, v)
                    return j0*np.linalg.det(fv)

                def determinant_jacobian(v):
                    fv = base_f+np.einsum('qkij,k->qij', response_f, v)
                    cofactor = np.stack((np.stack((fv[:, 1, 1], -fv[:, 1, 0]), -1),
                                         np.stack((-fv[:, 0, 1], fv[:, 0, 0]), -1)), -2)
                    return j0*np.einsum('qij,qkij->qk', cofactor, response_f)

                constraints.append(NonlinearConstraint(determinant, 1e-8, cap, jac=determinant_jacobian))
                objective_scale = (max(float(np.linalg.norm(gram, 2)), 1.)
                                   if self.release_matrix is not None else 1.)
                result = minimize(lambda v: (.5*v @ gram @ v-reduced_rhs @ v)/objective_scale,
                    np.zeros(count+1) if need_constrained_step else correction,
                    jac=lambda v: (gram @ v-reduced_rhs)/objective_scale, method='SLSQP',
                    constraints=constraints,
                    options=dict(ftol=1e-12, maxiter=100))
                if not result.success:
                    if count == 1:
                        return self.enlarged_scalar_direction(directions, matrix, residual, observed,
                                                              factor, scale, cap, emit)
                    raise RuntimeError('Physically constrained load step failed: '+result.message)
                correction = result.x
                new_j = determinant(correction)
                emit(dict(event=('release_constrained_direction' if self.release_matrix is not None
                                 else 'material_domain_constrained_direction'),
                    unconstrained_margin_mpa=float(proposed.min()),
                    constrained_margin_mpa=float((new_j*m.k0-m.alpha*m.bulk*(np.log(new_j)-.5*cp_logdet)).min()),
                    release_bounds_enforced=self.release_matrix is not None))
        return directions @ correction, self.load_basis @ correction[1:]

    def solve_step(self, fraction, observed, emit):
        observed = np.asarray(observed)[self.fit_indices]
        t = self.config['training']
        start = time.perf_counter()
        last_step = np.inf
        stable_iterations = 0
        # A changed load/displacement target needs a fresh tangent step.
        # Damping learned near the previous optimum can otherwise suppress
        # the new load response and stall at the monotone release floor.
        self.damping = 0.
        # The same physical/displacement residual merit defines the reduced
        # correction and line search. Pressure is absent from both.
        displacement_scale = self.model.length/self.model.physical['e_ground']

        def merit(r, x):
            if self.prescribed_loads is not None or (self.scalar_release_at_capacity() and
                    np.all(self.fit_observations(self.x)[0] <= observed)):
                return float(r @ r)
            y = self.fit_observations(x)[0]
            return float(r @ r+t['observation_weight']*np.sum(((y-observed)/displacement_scale)**2))

        for iteration in range(t['maximum_iterations']):
            residual, matrix, trial, info = self.evaluate(self.x, self.loads, jacobian=True)
            prediction = self.fit_observations(self.x)[0]
            emit(dict(event='iteration', fraction=fraction, iteration=iteration,
                displacement_rmse_mm=float(1000*np.sqrt(np.mean((prediction-observed)**2))),
                loads_mpa=self.loads.tolist(), last_relative_step=last_step if np.isfinite(last_step) else None,
                **info))
            if (((last_step < t['newton_step_tolerance'] and self.damping < 1e-7) or stable_iterations >= 3) and
                    info['physics_weighted_rms_mpa'] <= t['physics_rms_tolerance_mpa']):
                info.update(accepted=True, fraction=fraction, iterations=iteration,
                            seconds=time.perf_counter()-start,
                            convergence=('stable_physical_outputs' if stable_iterations >= 3 else 'coefficient_step'),
                            displacement_mm=(1000*prediction).tolist(), loads_mpa=self.loads.tolist())
                accept_all([self.history], [trial], fraction, info)
                self.history.bind(content_identity(self.materials.tensor(self.x)), self.material_id,
                                  self.history.history_chain_id)
                self.accepted.append(self.x.copy())
                self.metrics.append(info)
                return info
            dx, dq = self.newton_direction(matrix, residual, observed, emit)
            relative_direction = float(np.linalg.norm(dx)/max(np.linalg.norm(self.x), 1.))
            emit(dict(event='newton_direction', fraction=fraction, iteration=iteration,
                      relative_step=relative_direction, load_step_mpa=dq.tolist()))
            if (relative_direction < t['newton_step_tolerance'] and self.damping < 1e-7 and
                    info['physics_weighted_rms_mpa'] <= t['physics_rms_tolerance_mpa']):
                info.update(accepted=True, fraction=fraction, iterations=iteration,
                            seconds=time.perf_counter()-start,
                            displacement_mm=(1000*prediction).tolist(), loads_mpa=self.loads.tolist())
                accept_all([self.history], [trial], fraction, info)
                self.history.bind(content_identity(self.materials.tensor(self.x)), self.material_id,
                                  self.history.history_chain_id)
                self.accepted.append(self.x.copy())
                self.metrics.append(info)
                return info
            previous = merit(residual, self.x)
            previous_pressure = np.array([p['pressure_kpa'] for p in self.pressure_readout()])
            alpha = 1.
            rejection = None
            while alpha >= t['line_search_minimum']:
                candidate, loads = self.x+alpha*dx, self.loads+alpha*dq
                try:
                    next_residual, _, _, next_info = self.evaluate(candidate, loads, jacobian=False)
                    value = merit(next_residual, candidate)
                    if value <= previous+1e-12:
                        break
                    rejection = 'merit did not decrease'
                except ValueError as error:
                    rejection = str(error)
                alpha *= .5
            if alpha < t['line_search_minimum']:
                # A reduced inverse Newton direction can stagnate at the
                # least-squares observation optimum. Do not commit a state
                # whose physical residual still exceeds the declared bound.
                raise RuntimeError(f'Load step {fraction}: line search failed: {rejection}; {info}')
            # Test the full proposed step, not its line-search-truncated size.
            last_step = relative_direction
            self.x, self.loads = candidate, loads
            next_prediction = self.fit_observations(self.x)[0]
            next_pressure = np.array([p['pressure_kpa'] for p in self.pressure_readout()])
            merit_change = abs(previous-value)/max(previous, 1e-12)
            displacement_change = float(1000*np.max(np.abs(next_prediction-prediction)))
            pressure_change = (float(np.max(np.abs(next_pressure-previous_pressure))/max(np.max(np.abs(next_pressure)), 1.))
                               if len(next_pressure) else 0.)
            stable = (alpha >= .125 and
                      displacement_change < t['displacement_step_tolerance_mm'] and
                      pressure_change < t['pressure_step_relative_tolerance'])
            stable_iterations = stable_iterations+1 if stable else 0
            if t.get('adaptive_damping', False):
                # Damping changes the numerical step, not the equilibrium
                # equations. Repeated short searches signal an unreliable
                # undamped local model near plastic branch changes.
                if alpha < .25:
                    self.damping = max(1e-6, self.damping*10)
                elif alpha == 1.:
                    self.damping *= .1
            emit(dict(event='accepted_iterate', fraction=fraction, iteration=iteration,
                      line_search=alpha, relative_step=last_step, merit=value,
                      merit_relative_change=merit_change, displacement_step_mm=displacement_change,
                      pressure_step_relative=pressure_change, stable_iterations=stable_iterations,
                      numerical_damping=self.damping))
        raise RuntimeError(f'Load step {fraction} did not converge in {t["maximum_iterations"]} iterations')

    def pressure_readout(self):
        rows = []
        if not len(self.operators.support_rows):
            return rows
        for o in self.layout['outer']:
            matrix, index, gu = self.operators.point(o, derivative=True)
            value = matrix @ self.x
            coefficients = self.operators.coefficients(self.x, index)
            grad = np.einsum('jf,if->ij', gu, coefficients[:2])/self.model.physical['e_ground']
            f = dp.plane_tensor(self.materials.tensor((np.eye(2)+grad)[None]))
            nominal = self.materials.tensor(value[2:].reshape(1, 2, 2))
            sigma2 = (nominal @ f[:, :2, :2].transpose(-1, -2))/torch.linalg.det(f)[:, None, None]
            normal = self.materials.tensor([[*o['normal'], 0.]])
            current, _, area = current_frame(f, normal)
            n = current[0, :2]
            pressure = -n @ sigma2[0] @ n
            birth = None
            if self.birth_coefficients is not None:
                bc = self.operators.coefficients(self.birth_coefficients, index)
                bg = np.einsum('jf,if->ij', gu, bc[:2])/self.model.physical['e_ground']
                birth = dp.plane_tensor(self.materials.tensor((np.eye(2)+bg)[None]))
            _, material_sigma = self.materials.elastic(f, birth)
            material_pressure = -current[0] @ material_sigma[0] @ current[0]
            rows.append(dict(x_m=o['xy_m'][0], pressure_kpa=float(pressure)*1000,
                material_pressure_kpa=float(material_pressure)*1000,
                current_normal=current[0].tolist(), area_factor=float(area[0]),
                displacement_m=value[:2].tolist()))
        return rows

    def save(self, directory):
        np.savez_compressed(directory/'accepted_fields.npz', coefficients=np.asarray(self.accepted),
                            final_loads_mpa=self.loads,
                            support_birth=np.array([]) if self.birth_coefficients is None else self.birth_coefficients)
        torch.save(self.history.checkpoint(), directory/'ground_history.pt')
        (directory/'history.json').write_text(json.dumps(self.metrics, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    def restore(self, directory, through_step=None):
        with np.load(directory/'accepted_fields.npz') as data:
            if 'support_birth' in data and data['support_birth'].size:
                self.set_support_birth(data['support_birth'])
            self.accepted = list(data['coefficients'])
            self.x = self.accepted[-1].copy()
            self.loads = data['final_loads_mpa'].copy()
        self.metrics = json.loads((directory/'history.json').read_text(encoding='utf-8'))
        if through_step is not None:
            if not 1 <= through_step <= len(self.accepted):
                raise ValueError('Requested accepted prefix is unavailable')
            self.accepted = self.accepted[:through_step]
            self.metrics = self.metrics[:through_step]
            self.x = self.accepted[-1].copy()
            self.loads = np.asarray(self.metrics[-1]['loads_mpa'])
            for coefficients, metric in zip(self.accepted, self.metrics, strict=True):
                gradients = self.operators.volume_fields(coefficients)[0]
                _, _, trial, _ = self.materials.returned(gradients, self.history, self.ng,
                    jacobian=False, support_birth=self.support_birth(), ground_indices=self.operators.ground_rows)
                accept_all([self.history], [trial], metric['fraction'], {'accepted': True})
                self.history.bind(content_identity(self.materials.tensor(coefficients)), self.material_id,
                                  self.history.history_chain_id)
            return
        saved = torch.load(directory/'ground_history.pt', map_location=self.materials.device, weights_only=True)
        self.history.restore(saved, content_identity(self.materials.tensor(self.x)), self.material_id)
        if self.history.state.step != len(self.accepted):
            raise ValueError('Saved field sequence and material history have different lengths')

    def check_interpolation(self):
        """Check the same learned solution at new points with full history replay."""
        t = self.config['training']
        operators = PatchOperators(self.model, t['n_volume']+2, t['n_edge']+4,
                                   self.operators.boundary_weight, len(self.loads))
        ng = len(operators.ground_rows)
        history = PointHistory('validation_volume', self.materials.tensor(operators.coordinates[operators.ground_rows]), 0.)
        for x, metric in zip(self.accepted, self.metrics, strict=True):
            gradients, _ = operators.volume_fields(x)
            returned, _, trial, info = self.materials.returned(gradients, history, ng,
                jacobian=False, support_birth=self.support_birth(operators), ground_indices=operators.ground_rows)
            history.state = dp.commit_state(history.state, trial)
        residual, _ = assemble_material(operators, self.x, returned)
        linear = operators.linear @ self.x-operators.loads @ self.loads
        pressure = self.pressure_readout()
        check = dict(**info,
            material_weighted_rms_mpa=float(np.sqrt(np.mean(residual**2))),
            linear_weighted_rms_mpa=float(np.sqrt(np.mean(linear**2))),
            pressure_network_minus_material_kpa=[p['pressure_kpa']-p['material_pressure_kpa'] for p in pressure],
            scope='New collocation and edge points; the entire accepted material history is replayed, with no fitting.')
        ends = [linear[i:j] for label, i, j in operators.groups if label == 'free_support_end']
        if ends:
            tractions = np.concatenate(ends)*np.sqrt(t['n_edge']+4)/operators.boundary_weight
            check['free_support_end_traction_rms_mpa'] = float(np.sqrt(np.mean(tractions**2)))
        return check
