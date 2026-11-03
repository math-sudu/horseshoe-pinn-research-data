"""Paired prior-predictive checks of stiffness quadrature and bounded utilities."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ssinv.bounded_posterior import integrate_bounded
from ssinv.nonlinear_posterior import NonlinearInference
from analyze_elastoplastic_study import (
    inputs, target_maps, covariance, predictive_draws, TARGETS, write_json)


def initialize(scenario, baseline):
    global DATA, MAPS, NOISE, ROWS, LIMITS
    LIMITS = threadpool_limits(limits=1)
    DATA = inputs()
    MAPS = target_maps(DATA)
    NOISE = covariance(DATA, scenario)
    ROWS = next(case['baseline_rows'] for case in DATA['cases']
                if case['baseline'] == baseline)


def components(posterior):
    result = []
    for name in TARGETS:
        projection, squared = MAPS[name]
        total = posterior.target_variance(projection, squared)
        conditional = posterior.conditional_mean(projection)
        mean = np.einsum('nk,nko->no', posterior.weights, conditional)
        between = np.sum(posterior.weights[..., None]*(conditional-mean[:, None])**2,
                         axis=(1, 2))
        result.append(np.stack((total, total-between, between), axis=-1))
    return np.stack(result, axis=1)


def evaluate(task):
    scramble, candidate, settings = task
    readings = predictive_draws(DATA, NOISE, scramble, settings['power'])
    rows = ROWS+([] if candidate < 0 else [14+2*candidate, 15+2*candidate])
    if len(set(rows)) != len(rows):
        raise ValueError('Candidate is already part of the installed survey')
    gaussian = []
    for order in settings['orders']:
        model = NonlinearInference(DATA['surface'], rows, NOISE, order=order)
        posterior = model.fit(readings)
        gaussian.append(components(posterior))
    result = dict(gaussian=np.asarray(gaussian))
    if settings['bounded']:
        reference, diagnostics = [], []
        for replicate in range(settings['replicates']):
            values, diagnostic = [], []
            for i, survey in enumerate(readings):
                single = replace(posterior, weights=posterior.weights[i:i+1],
                    means=posterior.means[i:i+1], covariances=posterior.covariances[i:i+1],
                    log_evidence=posterior.log_evidence[i:i+1])
                bounded = integrate_bounded(model, survey, posterior=single,
                    power=settings['inner_power'], seed=7301+replicate,
                    tail_scale=settings['tail_scale'],
                    bounds=DATA['config']['inference']['load_standardized_bounds'])
                values.append([bounded.variance_components(MAPS[name][0]) for name in TARGETS])
                relevant = bounded.weights > 1e-4
                diagnostic.append([
                    np.min(bounded.conditional_ess[relevant]),
                    np.max(bounded.maximum_conditional_weight[relevant]),
                    .5*np.sum(np.abs(bounded.weights-single.weights[0]))])
            reference.append(values)
            diagnostics.append(diagnostic)
        result['bounded'] = np.asarray(reference)
        result['diagnostics'] = np.asarray(diagnostics)
    return scramble, candidate, result


def summarize(arrays, settings):
    candidates = settings['candidates']
    layout = inputs()['layout']
    records = []
    # Shapes are outer scramble, candidate, integration setting, survey, target, component.
    for method in ('gaussian', 'bounded'):
        if method not in arrays:
            continue
        values = arrays[method]
        for index in range(values.shape[2]):
            risk = values[:, 0, index, :, :, 0].mean(axis=1)
            variance = values[:, :, index, :, :, 0].mean(axis=2)
            utility = 100*(risk[:, None, :]-variance)/risk.mean(axis=0)
            records.append(dict(method=method,
                order=settings['orders'][index] if method == 'gaussian' else settings['orders'][-1],
                replicate=index if method == 'bounded' else None,
                baseline_components=values[:, 0, index].mean(axis=(0, 1)).tolist(),
                utility_percent=utility.mean(axis=0).tolist(),
                outer_paired_se_percent=(utility.std(axis=0, ddof=1)/np.sqrt(len(risk))).tolist()))
    comparison = {}
    for method in ('gaussian', 'bounded'):
        if method not in arrays:
            continue
        values = arrays[method]
        # Use the final grid for the Gaussian estimate and the mean of the
        # independently randomized integrals for the bounded estimate.
        averaged = values[:, :, -1] if method == 'gaussian' else values.mean(axis=2)
        expected = averaged.mean(axis=(0, 2))
        variance = averaged[..., 0].mean(axis=2)
        risk = variance[:, 0].mean(axis=0)
        utility = 100*(variance[:, :1]-variance)/risk
        pairs = []
        for first in range(1, len(candidates)):
            for second in range(first+1, len(candidates)):
                paired = utility[:, first]-utility[:, second]
                row = dict(first=candidates[first], second=candidates[second],
                    difference_percentage_points=paired.mean(axis=0).tolist(),
                    outer_standard_error=(paired.std(axis=0, ddof=1)/np.sqrt(len(paired))).tolist())
                if method == 'bounded':
                    replicate_variance = values[..., 0].mean(axis=(0, 3))
                    inner_pair = 100*(replicate_variance[second]-replicate_variance[first])/replicate_variance[0]
                    row['integration_standard_error'] = (
                        inner_pair.std(axis=0, ddof=1)/np.sqrt(len(inner_pair))).tolist()
                pairs.append(row)
        comparison[method] = dict(expected_components=expected.tolist(),
            utility_percent=utility.mean(axis=0).tolist(),
            bending_share_of_baseline=expected[0, 5, 0]/expected[0, 1, 0],
            bending_share_of_stress_reduction=[None]+[
                float((expected[0, 5, 0]-row[5, 0])/(expected[0, 1, 0]-row[1, 0]))
                for row in expected[1:]], contrasts=pairs)
    return dict(settings=settings, targets=list(TARGETS),
        component_order=['total', 'within_stiffness', 'between_stiffness'],
        candidates=[dict(index=i, arc_length_m=None if i < 0 else
                         layout['candidates'][i]['s_m']) for i in candidates],
        records=records, comparison=comparison)


def run(args):
    started = time.perf_counter()
    settings = vars(args).copy()
    settings['output'] = str(args.output)
    tasks = [(scramble, candidate, settings) for scramble in range(args.scrambles)
             for candidate in args.candidates]
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(args.scenario, args.baseline)) as pool:
        futures = [pool.submit(evaluate, task) for task in tasks]
        for future in as_completed(futures):
            scramble, candidate, result = future.result()
            results[scramble, candidate] = result
            print(json.dumps(dict(scramble=scramble, candidate=candidate,
                completed=len(results), total=len(tasks),
                elapsed_seconds=round(time.perf_counter()-started, 1))), flush=True)
    arrays = {name: np.array([[results[scramble, candidate][name]
                              for candidate in args.candidates]
                             for scramble in range(args.scrambles)])
              for name in results[0, -1]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), **arrays)
    write_json(args.output, summarize(arrays, settings))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', default='independent_01',
                        choices=['independent_01', 'independent_02', 'datum_full', 'datum_diagonal'])
    parser.add_argument('--baseline', choices=['symmetric', 'asymmetric'], default='symmetric')
    parser.add_argument('--candidates', type=int, nargs='+', default=[-1, 33, 34, 35, 36, 10, 11])
    parser.add_argument('--orders', type=int, nargs='+', default=[41, 61, 81])
    parser.add_argument('--power', type=int, default=6)
    parser.add_argument('--scrambles', type=int, default=4)
    parser.add_argument('--bounded', action='store_true')
    parser.add_argument('--inner-power', type=int, default=8)
    parser.add_argument('--replicates', type=int, default=4)
    parser.add_argument('--tail-scale', type=float, default=1.5)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.candidates[0] != -1:
        parser.error('The first candidate must be -1 (the installed baseline)')
    run(args)
