"""Apply the common finite-strain PINN to the measured shoulder convergence."""
from __future__ import annotations

import argparse
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


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def solve(resume):
    config = json.loads((ROOT/'config/elastoplastic_field.json').read_text(encoding='utf-8'))
    base, out = ROOT/config['baseline'], ROOT/config['output']
    physical = json.loads((base/'physical_config.json').read_text(encoding='utf-8'))
    layout = json.loads((base/'layout.json').read_text(encoding='utf-8'))
    observed = json.loads((base/'displacement_input.json').read_text(encoding='utf-8'))
    field = FieldPINN(SupportSection(base/'geometry.json'), physical, layout, config)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text(encoding='utf-8')) != config:
        raise ValueError('Saved field-application settings differ')
    write_json(out/'config.json', config)
    if resume:
        field.restore(out)
    elif (out/'accepted_fields.npz').exists():
        raise FileExistsError('Saved field history exists; use --resume')
    started = time.perf_counter()
    with (out/'run.jsonl').open('a', encoding='utf-8') as stream:
        def emit(value):
            value['elapsed_seconds'] = time.perf_counter()-started
            stream.write(json.dumps(value, allow_nan=False)+'\n')
            stream.flush()
            if value['event'] in ('iteration', 'committed_material_history'):
                print(json.dumps(value, allow_nan=False), flush=True)
        for step in range(len(field.accepted)+1, config['increment']['steps']+1):
            fraction = step/config['increment']['steps']
            field.solve_step(fraction, fraction*np.asarray(observed['increment_m']), emit)
            field.save(out)
            emit(dict(event='committed_material_history', step=step, fraction=fraction))
    prediction = field.operators.observations(field.x, layout)[0]
    index = config['observations']['fit_indices'][0]
    write_json(out/'result.json', dict(observed_shoulder_mm=1000*observed['increment_m'][index],
        predicted_shoulder_mm=float(1000*prediction[index]),
        residual_mm=float(1000*(prediction[index]-observed['increment_m'][index])),
        load_fraction=float(field.loads[1]/config['ground']['initial_vertical_compression_mpa']),
        loads_mpa=field.loads.tolist(), plastic_points=field.metrics[-1]['ground_plastic_points'],
        maximum_plastic_multiplier=field.metrics[-1]['ground_kappa_max'],
        scope='The shoulder-convergence observation fits the scalar release magnitude; material properties and initial stress ratio remain prescribed.',
        elapsed_seconds=time.perf_counter()-started))
    write_json(out/'check.json', field.check_interpolation())
    print((out/'result.json').read_text(encoding='utf-8'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        solve(args.resume)
