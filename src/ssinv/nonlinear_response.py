"""Common nonlinear response interpolation from finite-strain PINN histories."""
from __future__ import annotations

from itertools import combinations_with_replacement
import json
import numpy as np


def monomial_powers(dimension, degree):
    powers = [np.zeros(dimension, dtype=int)]
    for order in range(1, degree+1):
        for indices in combinations_with_replacement(range(dimension), order):
            powers.append(np.bincount(indices, minlength=dimension))
    return np.array(powers)


def monomials(x, powers):
    return np.prod(np.asarray(x)[..., None, :]**powers, axis=-1)


def monomial_derivatives(x, powers, dimensions=4):
    derivatives = []
    for j in range(dimensions):
        reduced = powers.copy()
        reduced[:, j] = np.maximum(reduced[:, j]-1, 0)
        derivatives.append(powers[:, j]*monomials(x, reduced))
    return np.stack(derivatives, axis=-1)


def monomial_hessians(x, powers, dimensions=4):
    result = np.empty(np.asarray(x).shape[:-1]+(len(powers), dimensions, dimensions))
    for i in range(dimensions):
        for j in range(i, dimensions):
            reduced = powers.copy()
            first = reduced[:, i].copy()
            reduced[:, i] = np.maximum(reduced[:, i]-1, 0)
            second = reduced[:, j].copy()
            reduced[:, j] = np.maximum(reduced[:, j]-1, 0)
            result[..., i, j] = first*second*monomials(x, reduced)
            result[..., j, i] = result[..., i, j]
    return result


def gaussian_moments(mean, covariance, powers):
    """Raw multivariate Gaussian moments, including zero-variance coordinates."""
    mean, covariance = np.asarray(mean), np.asarray(covariance)
    index = {tuple(p): i for i, p in enumerate(powers)}
    moments = np.empty(mean.shape[:-1]+(len(powers),))
    moments[..., 0] = 1.
    for k, alpha in enumerate(powers[1:], 1):
        i = int(np.flatnonzero(alpha)[0])
        beta = alpha.copy()
        beta[i] -= 1
        value = mean[..., i]*moments[..., index[tuple(beta)]]
        for j in np.flatnonzero(beta):
            lower = beta.copy()
            lower[j] -= 1
            value = value+beta[j]*covariance[..., i, j]*moments[..., index[tuple(lower)]]
        moments[..., k] = value
    return moments


class ResponseSurface:
    def __init__(self, powers, coefficients, shapes, centers, scales):
        self.powers = np.asarray(powers, dtype=int)
        self.coefficients = {key: np.asarray(value) for key, value in coefficients.items()}
        self.shapes = shapes
        self.centers, self.scales = np.asarray(centers), np.asarray(scales)
        self.degree = int(np.max(self.powers.sum(axis=1)))
        self.moment_powers = monomial_powers(5, 2*self.degree)
        self.moment_index = {tuple(p): i for i, p in enumerate(self.moment_powers)}
        self.feature_moment_indices = np.array([self.moment_index[tuple(p)] for p in self.powers])

    @classmethod
    def fit(cls, parameters, fields, inference, degree=3):
        low, high = np.log(inference['stiffness_bounds_mpa'])
        centers = np.r_[inference['load_mean_mpa'], .5*(low+high)]
        scales = np.r_[inference['load_std_mpa'], .5*(high-low)]
        x = (parameters-centers)/scales
        powers = monomial_powers(5, degree)
        features = monomials(x, powers)
        normalization = np.linalg.norm(features, axis=0)
        shapes = {key: list(values.shape[1:]) for key, values in fields.items()}
        widths = {key: int(np.prod(shape)) for key, shape in shapes.items()}
        values = np.concatenate([fields[key].reshape(len(x), -1) for key in shapes], axis=1)
        solution, _, rank, singular = np.linalg.lstsq(features/normalization, values, rcond=None)
        if rank != len(powers):
            raise ValueError(f'Nonlinear response interpolation has rank {rank}/{len(powers)}')
        solution /= normalization[:, None]
        coefficients, start = {}, 0
        for key, width in widths.items():
            coefficients[key] = solution[:, start:start+width]
            start += width
        surface = cls(powers, coefficients, shapes, centers, scales)
        report = dict(degree=degree, cases=len(x), terms=len(powers),
                      feature_condition=float(singular[0]/singular[-1]))
        return surface, report

    def standardized(self, loads, modulus):
        loads = np.asarray(loads)
        log_e = np.broadcast_to(np.log(modulus), loads.shape[:-1])
        return (np.concatenate((loads, log_e[..., None]), axis=-1)-self.centers)/self.scales

    def evaluate_standardized(self, x, keys=None):
        features = monomials(x, self.powers)
        return {key: (features @ self.coefficients[key]).reshape(features.shape[:-1]+tuple(self.shapes[key]))
                for key in (self.coefficients if keys is None else keys)}

    def observation_coefficients(self):
        return np.concatenate((self.coefficients['baseline_xy_m'],
                               self.coefficients['candidate_xy_m']), axis=1)

    def norm_polynomial(self, coefficients):
        """Coefficients of the squared norm of a projected target polynomial."""
        gram = coefficients @ coefficients.T
        result = np.zeros(len(self.moment_powers))
        for i, first in enumerate(self.powers):
            for j, second in enumerate(self.powers):
                result[self.moment_index[tuple(first+second)]] += gram[i, j]
        return result

    def target_projection(self, key, weights, selection=None):
        coefficient = self.coefficients[key]
        if selection is not None:
            coefficient = coefficient[:, selection]
        coefficient = coefficient*np.sqrt(weights)[None, :]
        left, singular, _ = np.linalg.svd(coefficient, full_matrices=False)
        keep = singular > np.finfo(float).eps*max(coefficient.shape)*singular[0]
        projected = left[:, keep]*singular[keep]
        return projected, self.norm_polynomial(projected)

    def save(self, path):
        np.savez_compressed(path, powers=self.powers, centers=self.centers, scales=self.scales,
            shapes=json.dumps(self.shapes), **{'coefficient_'+k: v for k, v in self.coefficients.items()})

    @classmethod
    def load(cls, path):
        with np.load(path) as saved:
            shapes = json.loads(str(saved['shapes']))
            return cls(saved['powers'], {k: saved['coefficient_'+k] for k in shapes},
                       shapes, saved['centers'], saved['scales'])
