"""Bounded posterior reference integration for the nonlinear response surface."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp, ndtri
from scipy.stats import chi2, qmc

from .nonlinear_response import monomials, gaussian_moments


@dataclass
class BoundedPosterior:
    stiffness_coordinates: np.ndarray
    weights: np.ndarray
    feature_means: np.ndarray
    feature_covariances: np.ndarray
    log_evidence: np.ndarray
    conditional_ess: np.ndarray
    maximum_conditional_weight: np.ndarray
    controlled_nodes: np.ndarray

    def mean(self, coefficient):
        return (self.weights @ self.feature_means) @ coefficient

    def variance_components(self, projection):
        """Return total, within-stiffness and between-stiffness target variance."""
        conditional_means = self.feature_means @ projection
        mean = self.weights @ conditional_means
        between = np.sum(self.weights[:, None]*(conditional_means-mean)**2)
        within = np.einsum('k,kfg,fo,go->', self.weights,
                           self.feature_covariances, projection, projection,
                           optimize=True)
        return np.array([within+between, within, between])


def integrate_bounded(model, readings, *, posterior=None, power=11, seed=7301,
                      tail_scale=1.5, bounds=(-4., 4.), control_variate=True):
    """Integrate the exact bounded load posterior at each stiffness node.

    The fitted Gauss--Newton normal is only a proposal. Equal-size scrambled
    Sobol samples from that normal and an inflated multivariate t(5) are
    combined using their equal-weight mixture density. Every evidence and
    moment integral uses exp[-(z'z + r'r)/2] inside the original load box.
    Stiffness integration uses the model's prior quadrature weights. Constants
    common to all nodes, including the truncated-normal prior normalizer,
    cancel on posterior normalization.

    Independent seeds estimate integration variation; power, stiffness order
    and tail_scale can be varied without changing the target distribution.
    With control_variate, subtract the normal proposal's sampled polynomial
    integrals and add their analytic values. The remaining integral still
    contains the exact likelihood and the box indicator, including its tails.
    At a box-constrained mode use the positive importance integral directly:
    the normal control mass need not be close to the one-sided target mass.
    """
    if model.summaries is not None:
        raise ValueError('This reference integral requires coordinate observations')
    readings = np.asarray(readings)
    if readings.ndim != 1:
        raise ValueError('Integrate one survey at a time')
    posterior = model.fit(readings) if posterior is None else posterior
    y = np.linalg.solve(model.factor, readings[model.rows])
    d, df = 4, 5.
    uniform = qmc.Sobol(d+1, scramble=True, seed=seed).random_base2(power)
    normal = ndtri(uniform[:, :d])
    student = tail_scale*normal/np.sqrt(chi2.ppf(uniform[:, d], df)/df)[:, None]
    standardized = np.vstack((normal, student))
    squared_radius = np.sum(standardized**2, axis=1)
    log_normal = -.5*(d*np.log(2*np.pi)+squared_radius)
    log_student = (gammaln((df+d)/2)-gammaln(df/2)
                   -.5*d*np.log(df*np.pi)-d*np.log(tail_scale)
                   -.5*(df+d)*np.log1p(squared_radius/(df*tail_scale**2)))
    log_mixture = np.logaddexp(log_normal, log_student)-np.log(2.)
    gaussian_weight = np.exp(log_normal-log_mixture)
    controlled = (control_variate & np.all((posterior.means[0] > bounds[0])
                                          & (posterior.means[0] < bounds[1]), axis=1))
    if control_variate:
        feature_indices = model.surface.feature_moment_indices
        product_indices = np.array([[model.surface.moment_index[tuple(a+b)]
                                     for b in model.surface.powers] for a in model.surface.powers])
        gaussian_mean = np.column_stack((posterior.means[0], model.nodes))
        gaussian_covariance = np.zeros((len(model.nodes), 5, 5))
        gaussian_covariance[:, :4, :4] = posterior.covariances[0]
        moments = gaussian_moments(gaussian_mean, gaussian_covariance, model.surface.moment_powers)
    means, covariances, evidence, ess, largest = [], [], [], [], []
    for k, stiffness in enumerate(model.nodes):
        factor = np.linalg.cholesky(posterior.covariances[0, k])
        z = posterior.means[0, k]+standardized @ factor.T
        inside = np.all((z >= bounds[0]) & (z <= bounds[1]), axis=1)
        parameters = np.column_stack((z, np.full(len(z), stiffness)))
        features = monomials(parameters, model.surface.powers)
        residual = y-features[inside] @ model.coefficient
        objective = np.sum(z[inside]**2, axis=1)+np.sum(residual**2, axis=1)
        log_weight = (-.5*objective-log_mixture[inside]
                      +np.log(np.diag(factor)).sum())
        normalizer = logsumexp(log_weight)
        if not np.isfinite(normalizer):
            raise FloatingPointError('No finite posterior mass in the load box')
        weight = np.exp(log_weight-normalizer)
        if controlled[k]:
            map_features = monomials(np.r_[posterior.means[0, k], stiffness], model.surface.powers)
            map_objective = np.sum(posterior.means[0, k]**2)+np.sum((y-map_features @ model.coefficient)**2)
            normal_mass = -.5*map_objective+.5*d*np.log(2*np.pi)+np.log(np.diag(factor)).sum()
            correction = -gaussian_weight.copy()
            correction[inside] += np.exp(log_weight-normal_mass)
            mass_ratio = 1+np.mean(correction)
            if mass_ratio <= 0:
                raise FloatingPointError(f'Control integral has nonpositive evidence at stiffness node {k}; refine the proposal')
            normal_mean = moments[k, feature_indices]
            normal_covariance = moments[k, product_indices]-np.outer(normal_mean, normal_mean)
            centered = features-normal_mean
            mean_shift = correction @ centered/(len(z)*mass_ratio)
            mean = normal_mean+mean_shift
            covariance = ((normal_covariance+(centered*correction[:, None]).T @ centered/len(z))
                          /mass_ratio-np.outer(mean_shift, mean_shift))
            log_evidence = normal_mass+np.log(mass_ratio)
        else:
            mean = weight @ features[inside]
            centered = (features[inside]-mean)*np.sqrt(weight[:, None])
            covariance = centered.T @ centered
            log_evidence = normalizer-np.log(len(standardized))
        means.append(mean)
        covariances.append(covariance)
        evidence.append(log_evidence)
        ess.append(1/np.sum(weight**2))
        largest.append(np.max(weight))
    evidence = np.array(evidence)
    log_weight = np.log(model.prior_weights)+evidence
    weights = np.exp(log_weight-logsumexp(log_weight))
    return BoundedPosterior(model.nodes, weights, np.array(means),
                            np.array(covariances), evidence, np.array(ess),
                            np.array(largest), controlled)
