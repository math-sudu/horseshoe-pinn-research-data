"""Sample, solve and interpolate the nonlinear load--stiffness response."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ssinv.nonlinear_response import ResponseSurface
from ssinv.measurements import baseline_points

OUT = ROOT/'results/elastoplastic'
CONFIG = ROOT/'config/elastoplastic_study.json'
KEYS = ('baseline_xy_m', 'candidate_xy_m', 'pressure_mpa', 'inner_u_m',
        'stress_mpa', 'support_stress_mpa')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sample_cases():
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    mean = np.asarray(config['inference']['load_mean_mpa'])
    scale = np.asarray(config['inference']['load_std_mpa'])
    low, high = np.log(config['inference']['stiffness_bounds_mpa'])
    rows = []
    for role, power, seed in (('training', 6, 2601), ('validation', 3, 8913)):
        uniform = qmc.Sobol(5, scramble=True, seed=seed).random_base2(power)
        z = 8*uniform[:, :4]-4
        modulus = np.exp(low+(high-low)*uniform[:, 4])
        for i, (value, e) in enumerate(zip(z, modulus, strict=True)):
            rows.append(dict(name=f'{role}_{i:03d}', role=role, modulus_mpa=float(e),
                             loads_mpa=(mean+scale*value).tolist()))
    for i in range(4):
        loads = mean.copy()
        loads[i] += scale[i]
        rows.append(dict(name=f'load_change_{i}', role='load_change', modulus_mpa=1000.,
                         loads_mpa=loads.tolist()))
    for modulus in (800., 1000., 1200.):
        for i, z in enumerate(([0., 0., 0., 0.], [.65, -.4, .8, .5], [-.75, .5, -.6, -.9])):
            name = 'reference_baseline' if modulus == 1000 and i == 0 else f'reference_{int(modulus)}_{i}'
            rows.append(dict(name=name, role='reference', modulus_mpa=modulus,
                             loads_mpa=(mean+scale*z).tolist(),
                             width=96, seed=7107, steps=16, n_volume=18, n_edge=29))
    manifest = dict(config=str(CONFIG.relative_to(ROOT)), training_seed=2601,
                    validation_seed=8913, reflection_augmentation=True, cases=rows)
    path = OUT/'sampling.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Existing sampling specification differs')
    write_json(path, manifest)
    return manifest


def solve_case(row):
    folder = OUT/row['name']
    if (folder/'summary.json').exists():
        return dict(case=row['name'], state='complete')
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT/'tools/run_elastoplastic_study.py'), 'forward',
               '--case', row['name'], '--modulus', str(row['modulus_mpa']),
               '--loads', *map(str, row['loads_mpa'])]
    for key in ('width', 'seed', 'steps', 'n_volume', 'n_edge'):
        if key in row:
            command += ['--'+key.replace('_', '-'), str(row[key])]
    if (folder/'accepted_fields.npz').exists():
        command.append('--resume')
    started = time.perf_counter()
    with (folder/'solver.log').open('a', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if result.returncode:
        raise RuntimeError(f"Nonlinear solve failed: {row['name']}; see {folder/'solver.log'}")
    return dict(case=row['name'], state='solved', elapsed_seconds=time.perf_counter()-started)


def campaign(args):
    manifest = sample_cases()
    rows = [row for row in manifest['cases'] if row['role'] in args.roles]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(solve_case, row): row for row in rows}
        for task in as_completed(tasks):
            print(json.dumps(task.result()), flush=True)


def reflection_indices(points):
    coordinates = np.array([p['xy_m'] for p in points])
    distances = np.linalg.norm(coordinates[:, None]-coordinates[None]*[-1., 1.], axis=-1)
    mirror = distances.argmin(axis=1)
    if np.max(distances[np.arange(len(points)), mirror]) > 1e-8:
        raise ValueError('Readout grid is not reflection closed')
    return mirror


def reflected(fields, layout, ground_xy=None):
    result = {}
    for key, points in (('baseline_xy_m', baseline_points(layout)),
                        ('candidate_xy_m', layout['candidates']), ('inner_u_m', layout['inner'])):
        result[key] = fields[key][reflection_indices(points)]*[-1., 1.]
    mirror = reflection_indices(layout['outer'])
    result['pressure_mpa'] = fields['pressure_mpa'][mirror]
    result['support_stress_mpa'] = fields['support_stress_mpa'][mirror]*[1., 1., -1.]
    result['stress_mpa'] = fields['stress_mpa'][[0, 2, 1, 3]]*[1., 1., -1.]
    if 'ground_displacement_m' in fields:
        centers = ground_xy.mean(axis=(1, 2))
        distances = np.linalg.norm(centers[:, None]-centers[None]*[-1., 1.], axis=-1)
        mirror = distances.argmin(axis=1)
        np.testing.assert_allclose(ground_xy[mirror, :, ::-1]*[-1., 1.], ground_xy, atol=2e-12)
        result['ground_displacement_m'] = fields['ground_displacement_m'][mirror, :, ::-1]*[-1., 1.]
    return result


def ground_readout(name):
    folder = OUT/name
    if (folder/'ground_display.npz').exists():
        return
    command = [sys.executable, str(ROOT/'tools/run_elastoplastic_study.py'), 'readout',
               '--case', name, '--ground-display', '--ground-only']
    with (folder/'readout.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if result.returncode:
        raise RuntimeError(f'Ground readout failed: {name}')
    print(json.dumps(dict(stage='ground_readout', case=name)), flush=True)


def ground(args):
    manifest = json.loads((OUT/'sampling.json').read_text(encoding='utf-8'))
    names = [row['name'] for row in manifest['cases'] if row['role'] != 'reference']
    names += ['baseline', 'edge_low', 'edge_high']
    names = [name for name in names if (OUT/name/'summary.json').exists()]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(ground_readout, names):
            pass


def interpolate(args):
    manifest = json.loads((OUT/'sampling.json').read_text(encoding='utf-8'))
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    layout = json.loads((OUT/'baseline/layout.json').read_text(encoding='utf-8'))['layout']
    training = [row for row in manifest['cases'] if row['role'] in ('training', 'load_change')]
    for name, modulus in (('baseline', 1000.), ('edge_low', 700.), ('edge_high', 1400.)):
        training.append(dict(name=name, modulus_mpa=modulus, loads_mpa=config['nominal_loads_mpa']))
    keys = (*KEYS, 'ground_displacement_m')
    parameters, fields = [], {key: [] for key in keys}
    for row in training:
        with np.load(OUT/row['name']/'readout.npz') as data:
            direct = {key: data[key] for key in KEYS}
        with np.load(OUT/row['name']/'ground_display.npz') as data:
            ground_xy = data['ground_xy_m']
            direct['ground_displacement_m'] = data['ground_fields'][..., :2]
        mirror = reflected(direct, layout, ground_xy)
        for sign, values in ((1., direct), (-1., mirror)):
            loads = np.asarray(row['loads_mpa']).copy()
            loads[3] *= sign
            parameters.append(np.r_[loads, np.log(row['modulus_mpa'])])
            for key in keys:
                fields[key].append(values[key])
    fields = {key: np.asarray(values) for key, values in fields.items()}
    surface, report = ResponseSurface.fit(np.asarray(parameters), fields, config['inference'], args.degree)
    report['validation'] = []
    for row in manifest['cases']:
        if row['role'] != 'validation':
            continue
        x = surface.standardized(np.asarray(row['loads_mpa']), row['modulus_mpa'])
        predicted = surface.evaluate_standardized(x)
        with np.load(OUT/row['name']/'readout.npz') as data:
            errors = {key: dict(rmse=float(np.sqrt(np.mean((predicted[key]-data[key])**2))),
                               relative_l2=float(np.linalg.norm(predicted[key]-data[key])/np.linalg.norm(data[key])))
                      for key in KEYS}
        with np.load(OUT/row['name']/'ground_display.npz') as data:
            expected = data['ground_fields'][..., :2]
            difference = predicted['ground_displacement_m']-expected
            errors['ground_displacement_m'] = dict(rmse=float(np.sqrt(np.mean(difference**2))),
                                                   relative_l2=float(np.linalg.norm(difference)/np.linalg.norm(expected)))
        report['validation'].append(dict(case=row['name'], errors=errors))
    surface.save(OUT/'response.npz')
    np.savez_compressed(OUT/'ground_coordinates.npz', ground_xy_m=ground_xy)
    write_json(OUT/'response.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['sample', 'campaign', 'ground', 'interpolate'])
    parser.add_argument('--roles', nargs='+', default=['training', 'validation', 'load_change', 'reference'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--degree', type=int, default=3)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.stage == 'sample':
            sample_cases()
        elif args.stage == 'campaign':
            campaign(args)
        elif args.stage == 'ground':
            ground(args)
        else:
            interpolate(args)
