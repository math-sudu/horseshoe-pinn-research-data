"""Nonlinear conditional load fits and stiffness-marginalized field moments."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from .nonlinear_response import monomials, monomial_derivatives, monomial_hessians, gaussian_moments


def box_quadratic_step(z, gradient, precision, lower=-4., upper=4.):
    """Solve each small bounded quadratic subproblem."""
    step = np.linalg.solve(precision, -gradient[..., None])[..., 0]
    outside = np.any((z+step < lower) | (z+step > upper), axis=-1)
    if not np.any(outside):
        return step
    value, g, h = z[outside], gradient[outside], precision[outside]
    best = np.full(len(value), np.inf)
    solution = np.zeros_like(value)
    for status in product((-1, 0, 1), repeat=z.shape[-1]):
        status = np.asarray(status)
        free, fixed = np.flatnonzero(status == 0), np.flatnonzero(status != 0)
        trial = np.zeros_like(value)
        if len(fixed):
            trial[:, fixed] = np.where(status[fixed] < 0, lower, upper)-value[:, fixed]
        if len(free):
            rhs = -g[:, free]
            if len(fixed):
                rhs -= np.einsum('nij,nj->ni', h[:, free[:, None], fixed], trial[:, fixed])
            trial[:, free] = np.linalg.solve(h[:, free[:, None], free], rhs[..., None])[..., 0]
        feasible = np.all((value+trial >= lower-1e-10) & (value+trial <= upper+1e-10), axis=-1)
        objective = np.einsum('ni,ni->n', g, trial)+.5*np.einsum('ni,nij,nj->n', trial, h, trial)
        choose = feasible & (objective < best)
        solution[choose], best[choose] = trial[choose], objective[choose]
    if not np.all(np.isfinite(best)):
        raise RuntimeError('Bounded conditional quadratic has no feasible solution')
    step[outside] = solution
    return step


class ChordSummaries:
    """Crown settlement and exact changes of the three chord lengths."""
    def __init__(self, points):
        self.coordinates = np.asarray([point['xy_m'] for point in points])
        if self.coordinates.shape != (7, 2):
            raise ValueError('Summary observations require the seven installed reflectors')
        self.chords = self.coordinates[[2, 4, 6]]-self.coordinates[[1, 3, 5]]
        self.lengths = np.linalg.norm(self.chords, axis=-1)

    def values_and_jacobian(self, readings):
        displacement = np.asarray(readings).reshape(readings.shape[:-1]+(7, 2))
        change = displacement[..., [2, 4, 6], :]-displacement[..., [1, 3, 5], :]
        chords = self.chords+change
        lengths = np.linalg.norm(chords, axis=-1)
        shortening = -np.sum(change*(2*self.chords+change), axis=-1)/(self.lengths+lengths)
        values = np.concatenate((-displacement[..., 0, 1:2], shortening), axis=-1)
        jacobian = np.zeros(readings.shape[:-1]+(4, 14))
        jacobian[..., 0, 1] = -1.
        directions = chords/lengths[..., None]
        for i in range(3):
            jacobian[..., i+1, 2+4*i:4+4*i] = directions[..., i, :]
            jacobian[..., i+1, 4+4*i:6+4*i] = -directions[..., i, :]
        return values, jacobian

    def parameter_hessian(self, readings, jacobian, hessian):
        displacement = readings.reshape(readings.shape[:-1]+(7, 2))
        chords = self.chords+displacement[..., [2, 4, 6], :]-displacement[..., [1, 3, 5], :]
        length = np.linalg.norm(chords, axis=-1)
        direction = chords/length[..., None]
        result = np.empty(readings.shape[:-1]+(4, 4, 4))
        result[..., 0, :, :] = -hessian[..., 1, :, :]
        for i in range(3):
            left, right = slice(2+4*i, 4+4*i), slice(4+4*i, 6+4*i)
            first = jacobian[..., right, :]-jacobian[..., left, :]
            second = hessian[..., right, :, :]-hessian[..., left, :, :]
            normal = direction[..., i, :]
            curvature = (np.eye(2)-normal[..., :, None]*normal[..., None, :])/length[..., i, None, None]
            result[..., i+1, :, :] = -np.einsum('...a,...aij->...ij', normal, second)-np.einsum(
                '...ai,...ab,...bj->...ij', first, curvature, first)
        return result


@dataclass
class NonlinearPosterior:
    surface: object
    stiffness_coordinates: np.ndarray
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    iterations: int
    maximum_stationarity: float
    log_evidence: np.ndarray

    def moments(self):
        if not hasattr(self, '_moments'):
            shape = self.means.shape[:-1]
            mean = np.concatenate((self.means, np.broadcast_to(
                self.stiffness_coordinates, shape)[..., None]), axis=-1)
            covariance = np.zeros(shape+(5, 5))
            covariance[..., :4, :4] = self.covariances
            self._moments = gaussian_moments(mean, covariance, self.surface.moment_powers)
        return self._moments

    def feature_means(self):
        return self.moments()[..., self.surface.feature_moment_indices]

    def mean(self, coefficient):
        feature_mean = np.einsum('nk,nkf->nf', self.weights, self.feature_means())
        return feature_mean @ coefficient

    def conditional_mean(self, coefficient):
        return self.feature_means() @ coefficient

    def target_variance(self, projection, squared_coefficients):
        second = np.einsum('nk,nkf,f->n', self.weights, self.moments(), squared_coefficients)
        mean = self.mean(projection)
        variance = second-np.sum(mean**2, axis=-1)
        if np.any(variance < -1e-9*np.maximum(second, 1.)):
            raise FloatingPointError('Negative nonlinear target variance')
        return np.maximum(variance, 0.)

    def field_mean(self, key):
        mean = self.mean(self.surface.coefficients[key])
        return mean.reshape((len(mean),)+tuple(self.surface.shapes[key]))

    def field_variance(self, key):
        moments = np.einsum('nk,nkf->nf', self.weights, self.moments())
        index = np.array([[self.surface.moment_index[tuple(first+second)]
                           for second in self.surface.powers] for first in self.surface.powers])
        feature_mean = moments[:, self.surface.feature_moment_indices]
        covariance = moments[:, index]-feature_mean[:, :, None]*feature_mean[:, None, :]
        coefficient = self.surface.coefficients[key]
        variance = np.einsum('fo,nfg,go->no', coefficient, covariance, coefficient, optimize=True)
        return np.maximum(variance, 0.).reshape((len(variance),)+tuple(self.surface.shapes[key]))


class NonlinearInference:
    """Conditional Gauss--Newton Gaussian approximation over a nonlinear response.

    Each stiffness node solves the nonlinear load fit. Its local Gaussian
    integral supplies the node evidence; polynomial target moments retain
    within-node curvature and between-node variation.
    """
    def __init__(self, surface, rows, covariance, *, order=21, fixed_modulus=None,
                 summaries=None, moduli_mpa=None):
        self.surface = surface
        self.rows = np.asarray(rows, dtype=int)
        self.summaries = summaries
        if moduli_mpa is not None:
            self.nodes = (np.log(moduli_mpa)-surface.centers[-1])/surface.scales[-1]
            self.prior_weights = np.ones(len(self.nodes))/len(self.nodes)
        elif fixed_modulus is None:
            self.nodes, self.prior_weights = np.polynomial.legendre.leggauss(order)
            self.prior_weights /= 2
        else:
            self.nodes = np.array([(np.log(fixed_modulus)-surface.centers[-1])/surface.scales[-1]])
            self.prior_weights = np.ones(1)
        self.load_dimensions = 4
        self.fixed_asymmetry = None
        if summaries is not None:
            # A chord-length survey is invariant under Gh -> -Gh. Integrate
            # that coordinate instead of choosing one of its reflected modes.
            nodes, weights = np.polynomial.hermite.hermgauss(7)
            self.fixed_asymmetry = np.tile(np.sqrt(2)*nodes, len(self.nodes))
            self.nodes = np.repeat(self.nodes, len(nodes))
            self.prior_weights = np.repeat(self.prior_weights, len(nodes))*np.tile(
                weights/np.sqrt(np.pi), len(self.prior_weights))
            self.load_dimensions = 3
        coefficient = surface.observation_coefficients()[:, self.rows]
        noise = covariance[np.ix_(self.rows, self.rows)]
        if summaries is not None:
            _, jacobian = summaries.values_and_jacobian(np.zeros(len(self.rows)))
            noise = jacobian @ noise @ jacobian.T
        self.factor = np.linalg.cholesky(noise)
        self.whitening = np.linalg.inv(self.factor)
        self.coefficient = (coefficient if summaries is not None else
                            solve_triangular(self.factor, coefficient.T, lower=True).T)

    def fit(self, readings, *, initial=None):
        readings = np.atleast_2d(readings)[:, self.rows]
        if self.summaries is not None:
            readings = self.summaries.values_and_jacobian(readings)[0]
        y = solve_triangular(self.factor, readings.T, lower=True).T
        n, k = len(y), len(self.nodes)
        z = np.zeros((n, k, 4)) if initial is None else np.array(initial, copy=True)
        if z.shape != (n, k, 4):
            raise ValueError('Initial conditional loads have the wrong shape')
        if self.fixed_asymmetry is not None:
            z[..., 3] = self.fixed_asymmetry
        stiffness = np.broadcast_to(self.nodes, (n, k))[..., None]

        def evaluate(value, derivative=False, indices=None):
            local_stiffness = stiffness if indices is None else stiffness[indices][:, None, :]
            local_y = y if indices is None else y[indices[0]]
            parameters = np.concatenate((value, local_stiffness), axis=-1)
            prediction = monomials(parameters, self.surface.powers) @ self.coefficient
            if self.summaries is not None:
                coordinate_prediction = prediction
                prediction, summary_jacobian = self.summaries.values_and_jacobian(prediction)
                prediction = np.linalg.solve(self.factor, prediction[..., None])[..., 0]
            residual = local_y[:, None, :]-prediction
            objective = np.sum(residual**2, axis=-1)+np.sum(value[..., :self.load_dimensions]**2, axis=-1)
            if not derivative:
                return objective
            features = monomial_derivatives(parameters, self.surface.powers)
            jacobian = np.einsum('nkfd,fo->nkod', features, self.coefficient)
            second = monomial_hessians(parameters, self.surface.powers)
            if self.summaries is not None:
                raw_hessian = np.einsum('nkfij,fo->nkoij', second, self.coefficient)
                observation_hessian = self.summaries.parameter_hessian(coordinate_prediction, jacobian, raw_hessian)
                observation_hessian = np.einsum('so,nkoij->nksij', self.whitening, observation_hessian)
                jacobian = np.einsum('nksr,nkrd->nksd', summary_jacobian, jacobian)
                jacobian = np.einsum('so,nkod->nksd', self.whitening, jacobian)
            precision = np.eye(4)+np.einsum('nkoi,nkoj->nkij', jacobian, jacobian)
            gradient = value-np.einsum('nkoi,nko->nki', jacobian, residual)
            if self.summaries is not None:
                hessian = precision-np.einsum('nko,nkoij->nkij', residual, observation_hessian)
            else:
                hessian = precision-np.einsum('nkf,nkfij->nkij',
                    residual @ self.coefficient.T, second)
            smallest = np.linalg.eigvalsh(hessian)[..., 0]
            hessian = hessian+np.maximum(.1-smallest, 0.)[..., None, None]*np.eye(4)
            d = self.load_dimensions
            return objective, gradient[..., :d], precision[..., :d, :d], hessian[..., :d, :d]

        stationarity = np.inf
        active = np.ones((n, k), dtype=bool)
        for iteration in range(100):
            indices = np.where(active)
            current = z[indices][:, None, :]
            objective, gradient, precision, hessian = evaluate(current, True, indices)
            step = np.zeros_like(current)
            d = self.load_dimensions
            step[..., :d] = box_quadratic_step(current[..., :d], gradient, hessian)
            proposed = np.clip(current+step, -4., 4.)
            projected = proposed-current
            stationarity = float(np.max(np.abs(projected)))
            active[indices] = np.max(np.abs(projected), axis=(1, 2)) >= 1e-6
            if stationarity < 1e-6:
                break
            alpha = np.ones(current.shape[:-1])
            candidate = proposed
            value = evaluate(candidate, indices=indices)
            for _ in range(24):
                rejected = value > objective+1e-11
                if not np.any(rejected):
                    break
                alpha[rejected] *= .5
                candidate = np.clip(current+alpha[..., None]*step, -4., 4.)
                value = evaluate(candidate, indices=indices)
            if np.any(value > objective+1e-9):
                raise RuntimeError('Nonlinear conditional load fit failed its descent check')
            displacement = np.max(np.abs(candidate-current))
            z[indices] = candidate[:, 0]
            if displacement < 1e-8 and stationarity > 1e-4:
                raise RuntimeError('Nonlinear conditional load fit stagnated')
        else:
            raise RuntimeError(f'Nonlinear conditional fit did not converge: {stationarity:g}')
        objective, gradient, precision, _ = evaluate(z, True)
        sign, logdet = np.linalg.slogdet(precision)
        if not np.all(sign > 0):
            raise FloatingPointError('Conditional precision is not positive definite')
        log_evidence = -.5*(objective+logdet)
        log_weights = np.log(self.prior_weights)+log_evidence
        weights = np.exp(log_weights-logsumexp(log_weights, axis=1, keepdims=True))
        covariance = np.zeros((n, k, 4, 4))
        covariance[..., :self.load_dimensions, :self.load_dimensions] = np.linalg.inv(precision)
        return NonlinearPosterior(self.surface, self.nodes, weights, z, covariance,
                                  iteration+1, stationarity, log_evidence)
