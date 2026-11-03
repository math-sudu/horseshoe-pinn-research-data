"""Compare bounded target variances with the local Gaussian approximation."""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ssinv.bounded_posterior import integrate_bounded
from ssinv.nonlinear_posterior import NonlinearInference
from ssinv.nonlinear_response import monomials
from analyze_elastoplastic_study import (inputs, target_maps, covariance,
                                        reference_cases, TARGETS, OUT, write_json)


def surveys(data, names):
    """Reuse the first saved recovery noise realization for each refined state."""
    noise = covariance(data, 'independent_01')
    references = reference_cases()
    rng = np.random.default_rng(42022)
    errors = rng.normal(size=(len(references), 256, len(noise))) @ np.linalg.cholesky(noise).T
    records = {}
    for name in names:
        i = next(i for i, row in enumerate(references) if row['name'] == name)
        with np.load(OUT/name/'readout.npz') as saved:
            readings = np.r_[saved['baseline_xy_m'].ravel(), saved['candidate_xy_m'].ravel()]+errors[i, 0]
        if name == 'reference_1000_1':
            with np.load(OUT/'single_case.npz') as saved:
                np.testing.assert_array_equal(readings, saved['readings_m'])
        records[name] = readings
    return records


def approximate_components(posterior, maps):
    output = []
    for key in TARGETS:
        projection, squared = maps[key]
        total = posterior.target_variance(projection, squared)[0]
        conditional = posterior.conditional_mean(projection)[0]
        mean = posterior.weights[0] @ conditional
        between = np.sum(posterior.weights[0, :, None]*(conditional-mean)**2)
        output.append([total, total-between, between])
    return np.array(output)


def stiffness_summary(surface, nodes, weights):
    moduli = np.exp(surface.centers[-1]+surface.scales[-1]*nodes)
    mean = weights @ moduli
    return dict(mean_mpa=float(mean), std_mpa=float(np.sqrt(weights @ (moduli-mean)**2)),
                standardized_mean=float(weights @ nodes))


def multistart_check(model, readings, posterior, count):
    rng = np.random.default_rng(57301)
    y = np.linalg.solve(model.factor, readings[model.rows])

    def objective(fit):
        z = fit.means[0]
        residual = y-monomials(np.column_stack((z, model.nodes)), model.surface.powers) @ model.coefficient
        return np.sum(z*z, axis=1)+np.sum(residual*residual, axis=1)

    original = objective(posterior)
    largest_improvement, largest_shift = 0., 0.
    for _ in range(count):
        initial = rng.uniform(-3., 3., posterior.means.shape)
        other = model.fit(readings, initial=initial)
        relevant = posterior.weights[0] > 1e-5
        largest_improvement = max(largest_improvement, float(np.max(original-objective(other))))
        largest_shift = max(largest_shift, float(np.max(np.abs(other.means[0, relevant]-posterior.means[0, relevant]))))
    return dict(starts=count, largest_objective_improvement=largest_improvement,
                largest_relevant_load_shift=largest_shift)


def run(args):
    started = time.perf_counter()
    data = inputs()
    surface = data['surface']
    maps = target_maps(data)
    noise = covariance(data, 'independent_01')
    if args.grid_orders:
        records = []
        for name, readings in surveys(data, args.surveys).items():
            for selected in args.candidates:
                rows = list(range(14)) if selected == -1 else list(range(14))+[14+2*selected, 15+2*selected]
                estimates = []
                for order in args.grid_orders:
                    model = NonlinearInference(surface, rows, noise, order=order)
                    posterior = model.fit(readings)
                    estimates.append(dict(order=order, components=approximate_components(posterior, maps).tolist(),
                        stiffness=stiffness_summary(surface, model.nodes, posterior.weights[0])))
                records.append(dict(survey=name, candidate=selected, estimates=estimates))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, dict(targets=list(TARGETS), records=records))
        return
    records, arrays = [], {}
    for name, readings in surveys(data, args.surveys).items():
        for selected in args.candidates:
            rows = list(range(14)) if selected == -1 else list(range(14))+[14+2*selected, 15+2*selected]
            model = NonlinearInference(surface, rows, noise, order=args.order)
            posterior = model.fit(readings)
            approximation = approximate_components(posterior, maps)
            reference, stiffness, ess, weights, feature_means = [], [], [], [], []
            for replicate in range(args.replicates):
                bounded = integrate_bounded(model, readings, posterior=posterior,
                    power=args.power, seed=7301+replicate, tail_scale=args.tail_scale,
                    bounds=data['config']['inference']['load_standardized_bounds'],
                    control_variate=not args.raw_importance)
                reference.append([bounded.variance_components(maps[key][0]) for key in TARGETS])
                stiffness.append(stiffness_summary(surface, model.nodes, bounded.weights))
                relevant = bounded.weights > 1e-4
                ess.append(dict(minimum_relevant_ess=float(np.min(bounded.conditional_ess[relevant])),
                    maximum_relevant_weight=float(np.max(bounded.maximum_conditional_weight[relevant])),
                    direct_integral_stiffness_mass=float(np.sum(bounded.weights[~bounded.controlled_nodes]))))
                weights.append(bounded.weights)
                feature_means.append(bounded.weights @ bounded.feature_means)
            prefix = f'{name}_{selected}'
            arrays[prefix+'_components'] = np.array(reference)
            arrays[prefix+'_stiffness_weights'] = np.array(weights)
            arrays[prefix+'_feature_means'] = np.array(feature_means)
            arrays[prefix+'_approximate_stiffness_weights'] = posterior.weights[0]
            arrays[prefix+'_readings_m'] = readings
            point = data['layout']['candidates'][selected] if selected >= 0 else None
            records.append(dict(survey=name, candidate=selected,
                arc_length_m=None if point is None else point['s_m'],
                approximate_components=approximation.tolist(),
                reference_components=np.mean(reference, axis=0).tolist(),
                reference_scramble_se=(np.std(reference, axis=0, ddof=1)/np.sqrt(args.replicates)).tolist(),
                relative_variance_change=(np.mean(reference, axis=0)[:, 0]/approximation[:, 0]-1).tolist(),
                weighted_mean_change=[float(np.linalg.norm(np.mean(feature_means, axis=0) @ maps[key][0]
                    -posterior.mean(maps[key][0])[0])) for key in TARGETS],
                approximate_stiffness=stiffness_summary(surface, model.nodes, posterior.weights[0]),
                reference_stiffness=stiffness,
                stiffness_weight_total_variation=float(.5*np.sum(np.abs(np.mean(weights, axis=0)-posterior.weights[0]))),
                importance_diagnostics=ess,
                multistart=multistart_check(model, readings, posterior, args.multistart)))
            print(json.dumps(dict(survey=name, candidate=selected,
                variance_change_percent=(100*np.array(records[-1]['relative_variance_change'])[:2]).tolist(),
                elapsed_seconds=round(time.perf_counter()-started, 1))), flush=True)
    report = dict(response='results/elastoplastic/response.npz',
        inference='Original truncated-normal load and uniform-log stiffness priors; independent coordinate noise',
        targets=list(TARGETS), component_order=['total', 'within_stiffness', 'between_stiffness'],
        settings=vars(args).copy(), records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report['settings']['output'] = str(args.output)
    write_json(args.output, report)
    arrays['stiffness_coordinates'] = model.nodes
    np.savez_compressed(args.output.with_suffix('.npz'), **arrays)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--power', type=int, default=11,
                        help='Each mixture component uses 2**power scrambled Sobol points per stiffness node')
    parser.add_argument('--order', type=int, default=41)
    parser.add_argument('--replicates', type=int, default=4)
    parser.add_argument('--tail-scale', type=float, default=1.5)
    parser.add_argument('--raw-importance', action='store_true', help='Disable analytic normal control variates')
    parser.add_argument('--grid-orders', nargs='+', type=int, help='Only compare conditional Gaussian stiffness quadrature orders')
    parser.add_argument('--multistart', type=int, default=0)
    parser.add_argument('--surveys', nargs='+', default=['reference_800_1', 'reference_1000_1', 'reference_1200_1'])
    parser.add_argument('--candidates', nargs='+', type=int, default=[-1, 34, 35, 36, 11])
    parser.add_argument('--output', type=Path, default=ROOT/'results/verification/bounded_posterior.json')
    args = parser.parse_args()
    if args.replicates < 2:
        parser.error('At least two independent scrambles are needed to estimate integration variation')
    with threadpool_limits(limits=4):
        run(args)
