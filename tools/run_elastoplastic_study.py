"""Run nonlinear load-history cases for the elastoplastic monitoring study."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ssinv.cad_geometry import SupportSection
from ssinv.finite_strain.mixed_pinn import FieldPINN
from ssinv.finite_strain.readout import FieldReadout
from ssinv.measurements import baseline_points
from ssinv.observation_design import build_design

CONFIG = ROOT/'config/elastoplastic_study.json'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def setup(config, modulus, *, width=None, seed=None, steps=None, solver=None, threads=None,
          n_volume=None, n_edge=None):
    config = copy.deepcopy(config)
    config['physical']['e_ground'] = float(modulus)
    if width is not None:
        config['training']['width'] = width
    if seed is not None:
        config['feature_seed'] = seed
    if steps is not None:
        config['increment']['steps'] = steps
    if solver is not None:
        config['training']['linear_solver'] = solver
    if threads is not None:
        config['training']['threads'] = threads
    if n_volume is not None:
        config['training']['n_volume'] = n_volume
    if n_edge is not None:
        config['training']['n_edge'] = n_edge
    section = SupportSection(ROOT/config['geometry'])
    design = json.loads((ROOT/config['design']).read_text(encoding='utf-8'))
    layout, cases = build_design(section, design, config['evaluation']['contour_points'])
    coordinates = [dict(point, kind='component', component=component)
                   for point in baseline_points(layout) for component in (0, 1)]
    # These four support positions track convergence of the nonlinear solve.
    control = [dict(arc=arc, fraction=.5, depth=0., xy_m=section.point(
                   'initial_support_outer', arc, .5).tolist(),
                   normal=section.local_frame(arc, .5)[0].tolist())
               for arc in ('crown', 'left_knee', 'right_knee', 'invert')]
    field = FieldPINN(section, config, dict(baseline=coordinates, outer=control), config)
    return config, field, layout, cases


def forward(args):
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    modulus = args.modulus or config['physical']['e_ground']
    config, field, layout, cases = setup(config, modulus, width=args.width,
                                        seed=args.seed, steps=args.steps,
                                        solver=args.solver, threads=args.threads,
                                        n_volume=args.n_volume, n_edge=args.n_edge)
    loads = np.asarray(args.loads if args.loads is not None else config['nominal_loads_mpa'])
    out = ROOT/config['output']/args.case
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(config=config, loads_mpa=loads.tolist())
    if (out/'config.json').exists():
        if json.loads((out/'config.json').read_text(encoding='utf-8')) != settings:
            raise ValueError('Saved case has different settings; choose a new case name')
    write_json(out/'config.json', settings)
    write_json(out/'layout.json', dict(layout=layout, cases=cases))
    if args.resume:
        field.restore(out)
    elif (out/'accepted_fields.npz').exists():
        raise FileExistsError('Saved case exists; use --resume')
    started = time.perf_counter()
    with (out/'run.jsonl').open('a', encoding='utf-8') as stream:
        def emit(value):
            value['elapsed_seconds'] = time.perf_counter()-started
            stream.write(json.dumps(value, allow_nan=False)+'\n')
            stream.flush()
            if value['event'] in ('iteration', 'step_complete'):
                print(json.dumps(value, allow_nan=False), flush=True)
        for step in range(len(field.accepted)+1, config['increment']['steps']+1):
            fraction = step/config['increment']['steps']
            field.loads = fraction*loads
            field.prescribed_loads = field.loads.copy()
            field.solve_step(fraction, np.zeros(14), emit)
            field.save(out)
            emit(dict(event='step_complete', step=step, total=config['increment']['steps']))
    depth = np.linspace(0, field.model.section.thickness, config['evaluation']['depth_points'])
    readout = FieldReadout(field).survey_and_targets(layout, depth)
    np.savez_compressed(out/'readout.npz', **readout, depth_m=depth)
    check = field.check_interpolation()
    write_json(out/'check.json', check)
    write_json(out/'summary.json', dict(modulus_mpa=modulus, loads_mpa=loads.tolist(),
        plastic_points=field.metrics[-1]['ground_plastic_points'],
        ground_points=field.metrics[-1]['ground_points'],
        maximum_plastic_multiplier=field.metrics[-1]['ground_kappa_max'],
        pressure_range_mpa=[float(readout['pressure_mpa'].min()), float(readout['pressure_mpa'].max())],
        stress_range_mpa=[float(readout['stress_mpa'][:, :, 0].min()), float(readout['stress_mpa'][:, :, 0].max())],
        maximum_displacement_mm=float(1000*np.max(np.linalg.norm(readout['inner_u_m'], axis=1))),
        elapsed_seconds=time.perf_counter()-started))
    print((out/'summary.json').read_text(encoding='utf-8'), flush=True)


def readout(args):
    out = ROOT/'results/elastoplastic'/args.case
    settings = json.loads((out/'config.json').read_text(encoding='utf-8'))
    config, field, layout, _ = setup(settings['config'], settings['config']['physical']['e_ground'])
    field.restore(out)
    if len(field.accepted) != config['increment']['steps']:
        raise ValueError('The load history is incomplete')
    reader = FieldReadout(field)
    depth = np.linspace(0., field.model.section.thickness, config['evaluation']['depth_points'])
    if not args.ground_only:
        np.savez_compressed(out/'readout.npz', **reader.survey_and_targets(layout, depth), depth_m=depth)
    if args.ground_display:
        np.savez_compressed(out/'ground_display.npz', **reader.ground_display())
    print(f'Updated readout: {out}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['forward', 'readout'])
    parser.add_argument('--case', default='baseline')
    parser.add_argument('--modulus', type=float)
    parser.add_argument('--loads', type=float, nargs=4)
    parser.add_argument('--width', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--solver', choices=['superlu', 'pardiso'])
    parser.add_argument('--threads', type=int)
    parser.add_argument('--n-volume', type=int)
    parser.add_argument('--n-edge', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--ground-display', action='store_true')
    parser.add_argument('--ground-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    threads = args.threads or config['training']['threads']
    torch.set_num_threads(threads)
    with threadpool_limits(limits=threads):
        (forward if args.stage == 'forward' else readout)(args)
