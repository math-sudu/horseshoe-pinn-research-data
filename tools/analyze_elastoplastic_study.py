"""Nonlinear survey inference, prospective placement and paired recovery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import ndtri, ndtr, logsumexp
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ssinv.nonlinear_response import ResponseSurface
from ssinv.nonlinear_posterior import NonlinearInference, ChordSummaries
from ssinv.measurements import baseline_points
from ssinv.quadrature import thickness_weights
from build_elastoplastic_response import reflection_indices

OUT = ROOT/'results/elastoplastic'
TARGETS = ('pressure', 'transition_stress', 'axial_force', 'bending_moment',
           'axial_profile', 'bending_profile', 'remaining_profile')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def inputs():
    config = json.loads((ROOT/'config/elastoplastic_study.json').read_text(encoding='utf-8'))
    layout = json.loads((OUT/'baseline/layout.json').read_text(encoding='utf-8'))
    surface = ResponseSurface.load(OUT/'response.npz')
    with np.load(OUT/'baseline/readout.npz') as saved:
        depth = saved['depth_m']
    return dict(config=config, layout=layout['layout'], cases=layout['cases'], surface=surface, depth=depth)


def target_maps(data):
    surface, depth = data['surface'], data['depth']
    weights = thickness_weights(depth)/(depth[-1]-depth[0])
    indices = np.arange(4*len(depth)*3).reshape(4, len(depth), 3)[1:3, :, 0].ravel()
    maps = dict(pressure=surface.target_projection('pressure_mpa', np.ones(600)/600),
                transition_stress=surface.target_projection('stress_mpa', np.tile(weights/2, 2), indices))
    h = depth[-1]-depth[0]
    z = depth-(depth[0]+depth[-1])/2
    j = np.sum(weights*z*z)
    stress = surface.coefficients['stress_mpa'][:, indices].reshape(len(surface.powers), 2, len(depth))
    force = h*np.einsum('fsd,d->fs', stress, weights)
    moment = h*np.einsum('fsd,d->fs', stress, weights*z)
    remainder = stress-force[:, :, None]/h-moment[:, :, None]*z/(h*j)
    for name, coefficient in (
        ('axial_force', force/np.sqrt(2)), ('bending_moment', moment/np.sqrt(2)),
        ('axial_profile', force/(h*np.sqrt(2))),
        ('bending_profile', moment/(h*np.sqrt(2*j))),
        ('remaining_profile', (remainder*np.sqrt(weights/2)).reshape(len(surface.powers), -1))):
        left, singular, _ = np.linalg.svd(coefficient, full_matrices=False)
        projection = left*singular
        maps[name] = projection, surface.norm_polynomial(projection)
    return maps


def covariance(data, scenario):
    n = data['surface'].observation_coefficients().shape[1]
    sigma = data['config']['inference']['noise_std_m']
    base = sigma*sigma*np.eye(n)
    if scenario == 'independent_02':
        return 4*base
    if scenario in ('datum_full', 'datum_diagonal'):
        datum = np.tile(np.eye(2), (n//2, 1))
        full = base+sigma*sigma*(datum @ datum.T)
        return np.diag(np.diag(full)) if scenario == 'datum_diagonal' else full
    if scenario != 'independent_01':
        raise ValueError(scenario)
    return base


def predictive_draws(data, noise, scramble, power):
    surface = data['surface']
    n = noise.shape[0]
    uniform = qmc.Sobol(5+n, scramble=True, seed=64500+scramble).random_base2(power)
    x = np.column_stack((ndtri(ndtr(-4)+(ndtr(4)-ndtr(-4))*uniform[:, :4]), 2*uniform[:, 4]-1))
    fields = surface.evaluate_standardized(x, ('baseline_xy_m', 'candidate_xy_m'))
    readings = np.concatenate((fields['baseline_xy_m'].reshape(len(x), -1),
                               fields['candidate_xy_m'].reshape(len(x), -1)), axis=1)
    readings += ndtri(uniform[:, 5:]) @ np.linalg.cholesky(noise).T
    points = baseline_points(data['layout'])+data['layout']['candidates']
    mirror = reflection_indices(points)
    rows = (2*mirror[:, None]+np.arange(2)).ravel()
    signs = np.tile([-1., 1.], len(points))
    return np.vstack((readings, readings[:, rows]*signs))


def posterior_variances(posterior, maps):
    return np.column_stack([posterior.target_variance(*maps[key]) for key in TARGETS])


def selection(data, case, scores, risks, target):
    allowed = np.array(case['allowed_indices'])
    j = TARGETS.index(target)
    means = np.mean(scores[:, :, j], axis=0)
    maximum = np.max(means[allowed])
    equivalent = allowed[np.isclose(means[allowed], maximum, rtol=2e-6, atol=1e-12)]
    chosen = int(equivalent[0])
    point = data['layout']['candidates'][chosen]
    return dict(layout=case['name'], baseline=case['baseline'], target=target,
        selected_index=chosen, selected_id=point['id'], location=point,
        equivalent_indices=equivalent.tolist(), allowed_indices=allowed.tolist(),
        scores_mpa2=means.tolist(),
        score_scramble_se_mpa2=(np.std(scores[:, :, j], axis=0, ddof=1)/np.sqrt(len(scores))).tolist(),
        expected_baseline_variance_mpa2=float(np.mean(risks[:, j])))


def design(args):
    data, started = inputs(), time.perf_counter()
    maps = target_maps(data)
    noise = covariance(data, args.scenario)
    baselines = {case['baseline']: case['baseline_rows'] for case in data['cases']}
    scores, risks = {}, {}
    for baseline, rows in baselines.items():
        scores[baseline], risks[baseline] = [], []
        for scramble in range(args.scrambles):
            readings = predictive_draws(data, noise, scramble, args.power)
            base = NonlinearInference(data['surface'], rows, noise).fit(readings)
            base_variance = posterior_variances(base, maps)
            risks[baseline].append(base_variance.mean(axis=0))
            batch = np.zeros((48, len(TARGETS)))
            for i in range(48):
                added = [14+2*i, 15+2*i]
                if set(rows).intersection(added):
                    continue
                after = NonlinearInference(data['surface'], rows+added, noise).fit(readings, initial=base.means)
                batch[i] = (base_variance-posterior_variances(after, maps)).mean(axis=0)
            scores[baseline].append(batch)
            print(json.dumps(dict(stage='design', scenario=args.scenario, baseline=baseline,
                scramble=scramble, elapsed_seconds=time.perf_counter()-started)), flush=True)
        scores[baseline], risks[baseline] = np.asarray(scores[baseline]), np.asarray(risks[baseline])
    report = dict(scenario=args.scenario, draws_per_scramble=2**(args.power+1),
        scrambles=args.scrambles, targets=list(TARGETS),
        selections=[selection(data, case, scores[case['baseline']], risks[case['baseline']], target)
                    for target in ('pressure', 'transition_stress') for case in data['cases']],
        scores={name: values.tolist() for name, values in scores.items()},
        risks={name: values.tolist() for name, values in risks.items()})
    write_json(OUT/f'design_{args.scenario}.json', report)


def reference_cases():
    manifest = json.loads((OUT/'sampling.json').read_text(encoding='utf-8'))
    return [row for row in manifest['cases'] if row['role'] == 'reference']


def choice(report, layout, target):
    return next(row for row in report['selections'] if row['layout'] == layout and row['target'] == target)


def uncertainty_interval(baseline, augmented, rng):
    # Every independently solved physical state retains the same noise sample count.
    count = baseline.shape[-1]
    indices = rng.integers(0, count, (2000, baseline.shape[0], count))
    b = np.take_along_axis(baseline[None], indices, axis=2).mean(axis=(1, 2))
    a = np.take_along_axis(augmented[None], indices, axis=2).mean(axis=(1, 2))
    return np.quantile(100*(1-a/b), [.025, .975]).tolist()


def recovery(args):
    data, started = inputs(), time.perf_counter()
    report = json.loads((OUT/f'design_{args.scenario}.json').read_text(encoding='utf-8'))
    noise = covariance(data, args.scenario)
    actual_noise = covariance(data, 'datum_full') if args.scenario == 'datum_diagonal' else noise
    references = reference_cases()
    if args.scenario != 'independent_01':
        references = [row for row in references if row['modulus_mpa'] == 1000.]
    rng = np.random.default_rng(42022)
    errors = rng.normal(size=(len(references), args.noise_samples, len(noise))) @ np.linalg.cholesky(actual_noise).T
    methods = {}
    for case in data['cases']:
        rows = case['baseline_rows']
        methods.setdefault(case['baseline']+'_baseline', (rows, {}))
        for target in ('pressure', 'transition_stress'):
            selected = choice(report, case['name'], target)['selected_index']
            methods[case['name']+'_'+target] = (rows+[14+2*selected, 15+2*selected], {})
    if args.scenario == 'independent_01':
        methods['fixed_modulus'] = (list(range(14)), dict(fixed_modulus=1000.))
        methods['summaries'] = (list(range(14)), dict(summaries=ChordSummaries(baseline_points(data['layout']))))
    surface, depth = data['surface'], data['depth']
    dw = thickness_weights(depth)/(depth[-1]-depth[0])/2
    arrays = dict(noise_m=errors, reference_names=np.array([row['name'] for row in references]),
                  moduli_mpa=np.array([row['modulus_mpa'] for row in references]), depth_m=depth)
    partial = OUT/f'recovery_partial_{args.scenario}.npz'
    if partial.exists():
        with np.load(partial) as saved:
            np.testing.assert_array_equal(saved['noise_m'], errors)
            np.testing.assert_array_equal(saved['reference_names'], arrays['reference_names'])
            arrays.update(dict(saved))
    summaries = []
    for name, (rows, options) in methods.items():
        if name+'_mse' in arrays:
            continue
        all_mse, pressure_errors, stress_errors, boundary = [], [], [], []
        for i, row in enumerate(references):
            with np.load(OUT/row['name']/'readout.npz') as saved:
                truth = dict(saved)
            readings = np.r_[truth['baseline_xy_m'].ravel(), truth['candidate_xy_m'].ravel()]+errors[i]
            posterior = NonlinearInference(surface, rows, noise, **options).fit(readings)
            dp = posterior.field_mean('pressure_mpa')-truth['pressure_mpa']
            ds = posterior.field_mean('stress_mpa')[:, 1:3, :, 0]-truth['stress_mpa'][1:3, :, 0]
            all_mse.append(np.column_stack((np.mean(dp*dp, axis=1), np.einsum('nsd,d->n', ds*ds, dw))))
            pressure_errors.append(np.mean(dp*dp, axis=0))
            stress_errors.append(np.mean(ds*ds, axis=0))
            boundary.append(float(np.max(np.sum(posterior.weights*np.any(np.abs(posterior.means)>3.99, axis=-1), axis=-1))))
        arrays[name+'_mse'] = np.asarray(all_mse)
        arrays[name+'_pressure_pointwise_mse'] = np.asarray(pressure_errors)
        arrays[name+'_stress_pointwise_mse'] = np.asarray(stress_errors)
        np.savez_compressed(partial, **arrays)
        print(json.dumps(dict(stage='recovery', scenario=args.scenario, method=name,
            maximum_boundary_node_mass=max(boundary), elapsed_seconds=time.perf_counter()-started)), flush=True)
    for name in methods:
        for modulus in np.unique(arrays['moduli_mpa']):
            at = arrays['moduli_mpa'] == modulus
            summaries.append(dict(method=name, modulus_mpa=float(modulus),
                pressure_rmse_mpa=float(np.sqrt(arrays[name+'_mse'][at, :, 0].mean())),
                stress_rmse_mpa=float(np.sqrt(arrays[name+'_mse'][at, :, 1].mean()))))
    gains = []
    bootstrap = np.random.default_rng(42023)
    for case in data['cases']:
        baseline = arrays[case['baseline']+'_baseline_mse']
        for target, column in (('pressure', 0), ('transition_stress', 1)):
            augmented = arrays[case['name']+'_'+target+'_mse']
            for modulus in np.unique(arrays['moduli_mpa']):
                at = arrays['moduli_mpa'] == modulus
                b, a = baseline[at, :, column], augmented[at, :, column]
                gains.append(dict(layout=case['name'], target=target, modulus_mpa=float(modulus),
                    mse_reduction_percent=float(100*(1-a.mean()/b.mean())),
                    interval95_percent=uncertainty_interval(b, a, bootstrap)))
    np.savez_compressed(OUT/f'recovery_{args.scenario}.npz', **arrays)
    write_json(OUT/f'recovery_{args.scenario}.json', dict(scenario=args.scenario,
        noise_samples_per_state=args.noise_samples, physical_states=references, results=summaries, gains=gains))
    partial.unlink()


def single_case(args):
    data = inputs()
    surface, noise = data['surface'], covariance(data, 'independent_01')
    design = json.loads((OUT/'design_independent_01.json').read_text(encoding='utf-8'))
    references = reference_cases()
    index = next(i for i, row in enumerate(references) if row['name'] == 'reference_1000_1')
    rng = np.random.default_rng(42022)
    errors = (rng.normal(size=(len(references), args.noise_samples, len(noise))) @
              np.linalg.cholesky(noise).T)[index, 0]
    with np.load(OUT/'reference_1000_1/readout.npz') as saved:
        truth = dict(saved)
    readings = np.r_[truth['baseline_xy_m'].ravel(), truth['candidate_xy_m'].ravel()]+errors
    baseline_rows = list(range(14))
    base = NonlinearInference(surface, baseline_rows, noise).fit(readings)
    pressure = choice(design, 'symmetric_full', 'pressure')['selected_index']
    stress = choice(design, 'symmetric_full', 'transition_stress')['selected_index']
    arrays = dict(readings_m=readings, pressure_candidate=pressure, stress_candidate=stress,
        stiffness_mpa=np.exp(surface.centers[-1]+surface.scales[-1]*base.stiffness_coordinates),
        quadrature_weights=np.polynomial.legendre.leggauss(21)[1]/2,
        pressure_reference_mpa=truth['pressure_mpa'],
        stress_reference_mpa=truth['stress_mpa'], support_reference_stress_mpa=truth['support_stress_mpa'],
        depth_m=data['depth'], pressure_std_mpa=np.sqrt(base.field_variance('pressure_mpa')[0]))
    means = {key: [] for key in ('pressure_mpa', 'stress_mpa', 'support_stress_mpa', 'inner_u_m')}
    for i in range(-1, 48):
        posterior = base if i == -1 else NonlinearInference(surface,
            baseline_rows+[14+2*i, 15+2*i], noise).fit(readings, initial=base.means)
        for key in means:
            means[key].append(posterior.field_mean(key)[0])
        if i in (-1, pressure, stress):
            prefix = 'baseline' if i == -1 else f'candidate_{i}'
            arrays[prefix+'_weights'] = posterior.weights[0]
            arrays[prefix+'_feature_means'] = posterior.feature_means()[0]
            arrays[prefix+'_load_means_mpa'] = surface.centers[:4]+posterior.means[0]*surface.scales[:4]
            arrays[prefix+'_log_evidence'] = posterior.log_evidence[0]
    arrays.update({key: np.asarray(values) for key, values in means.items()})
    moduli = np.array([700., 750., 800., 850., 900., 950., 1000., 1050., 1100., 1200., 1300., 1400.])
    dense = np.linspace(700., 1400., 241)
    conditional, densities = [], []
    for name, rows in (('baseline', baseline_rows),
                       (f'candidate_{stress}', baseline_rows+[14+2*stress, 15+2*stress])):
        posterior = NonlinearInference(surface, rows, noise, moduli_mpa=np.r_[moduli, dense]).fit(readings)
        conditional.append(posterior.feature_means()[0, :len(moduli)])
        normalizer = logsumexp(np.log(arrays['quadrature_weights'])+arrays[name+'_log_evidence'])
        densities.append(np.exp(posterior.log_evidence[0, len(moduli):]-normalizer)*1000/(dense*np.log(2)))
    change = conditional[1]-conditional[0]
    arrays['conditional_moduli_mpa'] = moduli
    arrays['density_moduli_gpa'] = dense/1000
    arrays['stiffness_density'] = np.asarray(densities)
    arrays['conditional_stress_change_mpa'] = (change @ surface.coefficients['support_stress_mpa']).reshape(
        (len(moduli),)+tuple(surface.shapes['support_stress_mpa']))[..., 0]
    arrays['conditional_pressure_change_mpa'] = change @ surface.coefficients['pressure_mpa']
    arrays['conditional_ground_change_m'] = (change[[0, 2, 4, 6, 9, 11]] @
        surface.coefficients['ground_displacement_m']).reshape((6,)+tuple(surface.shapes['ground_displacement_m']))
    np.savez_compressed(OUT/'single_case.npz', **arrays)
    print(json.dumps(dict(stage='single_case', pressure_candidate=pressure, stress_candidate=stress)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['design', 'recovery', 'case'])
    parser.add_argument('--scenario', choices=['independent_01', 'independent_02', 'datum_full', 'datum_diagonal'],
                        default='independent_01')
    parser.add_argument('--power', type=int, default=6)
    parser.add_argument('--scrambles', type=int, default=4)
    parser.add_argument('--noise-samples', type=int, default=256)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {'design': design, 'recovery': recovery, 'case': single_case}[args.stage](args)
