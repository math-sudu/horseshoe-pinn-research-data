"""Read conditional arch pressure and derive regional stress-ratio inputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ssinv.finite_strain.readout import FieldReadout


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def regional_stress():
    source = ROOT/'data/field_convergence/borehole_stresses.csv'
    with source.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    depth = 290.
    fits = {}
    for component in ('SH', 'Sh', 'Sv'):
        data = np.array([[float(r['depth_m']), float(r['measured_principal_stress_MPa'])]
                         for r in rows if r['series'] == component])
        slope, intercept = np.polyfit(data[:, 0], data[:, 1], 1)
        fits[component] = dict(count=len(data), slope_mpa_per_m=float(slope),
            intercept_mpa=float(intercept), stress_at_depth_mpa=float(slope*depth+intercept))
    vertical = fits['Sv']['stress_at_depth_mpa']
    lower = fits['Sh']['stress_at_depth_mpa']/vertical
    upper = fits['SH']['stress_at_depth_mpa']/vertical
    result = dict(source=source.relative_to(ROOT).as_posix(), depth_m=depth, fits=fits,
        horizontal_ratio_from_regional_trends=[lower, upper],
        nominal_design_ratio=1.1, alternative_ratio=round(upper, 1),
        transfer='The horizontal normal stress lies between Sh and SH as the horizontal direction varies. Transfer their fitted ratios to Sv at the representative depth, retaining the local design vertical stress of 5.8 MPa. The two conditions represent different equivalent transverse stress assignments; neither reconstructs the local three-dimensional stress tensor.',
        interpretation='These are directional endpoints of regional fitted trends, not confidence limits or locally measured stresses. The nominal design ratio is close to the lower trend; the alternative is the upper trend rounded to one decimal place.')
    out = ROOT/'results/field_pressure'
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'regional_stress.json', result)
    return result


def write_pressure(field, out):
    """Extract support-side Cauchy normal pressure on the supported open arch."""
    section = field.model.section
    nodes, weights = np.polynomial.legendre.leggauss(16)
    points, lengths = [], []
    for patch in field.model.liner:
        arc = section.arc('initial_support_outer', patch.arc_id)
        start, end = patch.fraction
        fractions = start+(nodes+1)*(end-start)/2
        patch_length = section.scale*arc['radius']*arc['sweep_rad']*(end-start)
        points.extend(dict(arc=patch.arc_id, fraction=float(f), depth=0.) for f in fractions)
        lengths.extend(weights*patch_length/2)
    lengths = np.asarray(lengths)
    readout = FieldReadout(field)
    pressure = readout.values(points)['pressure_mpa']
    coordinates = np.array([section.point('initial_support_outer', p['arc'], p['fraction'])
                            for p in points])
    with (out/'pressure.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['arc', 'fraction', 'x_m', 'y_m', 'reference_length_weight_m', 'pressure_mpa'])
        for point, xy, length, value in zip(points, coordinates, lengths, pressure, strict=True):
            writer.writerow([point['arc'], point['fraction'], *xy, length, value])
    landmarks = [dict(arc='crown', fraction=.5, depth=0.)]
    observation = field.config['observations']['fit_indices'][0]
    landmarks.extend(dict(p, depth=0.) for p in field.layout['baseline'][observation]['endpoints'])
    landmark_values = readout.values(landmarks)['pressure_mpa']
    arc_means = {}
    for name in dict.fromkeys(p['arc'] for p in points):
        mask = np.array([p['arc'] == name for p in points])
        arc_means[name] = float(np.average(pressure[mask], weights=lengths[mask]))
    result = dict(ground_modulus_mpa=field.model.physical['e_ground'],
        initial_vertical_compression_mpa=field.config['ground']['initial_vertical_compression_mpa'],
        initial_horizontal_ratio=field.config['ground']['initial_horizontal_ratio'],
        release_fraction=float(field.loads[1]/field.config['ground']['initial_vertical_compression_mpa']),
        arch_reference_length_m=float(lengths.sum()), quadrature_points=len(points),
        arch_mean_pressure_mpa=float(np.average(pressure, weights=lengths)),
        arc_mean_pressure_mpa=arc_means,
        landmark_pressure_mpa=dict(zip(('crown', 'left_shoulder', 'right_shoulder'),
                                       landmark_values.tolist(), strict=True)),
        pressure_definition='Minus the support-side Cauchy stress contracted with the deformed interface normal, positive in compression. Means use reference arc length over the supported arch; the open invert is excluded.',
        scope='Conditional mechanical demand of the specified equivalent initially stress-free arch and its calibrated release history, not measured field pressure or the actual stress increment between survey dates.')
    write_json(out/'pressure_summary.json', result)
    return result


def saved_pressure(case):
    import torch
    from threadpoolctl import threadpool_limits
    from ssinv.cad_geometry import SupportSection
    from ssinv.finite_strain.mixed_pinn import FieldPINN

    config = json.loads((case/'config.json').read_text(encoding='utf-8'))
    base = ROOT/config['baseline']
    physical = json.loads((base/'physical_config.json').read_text(encoding='utf-8'))
    layout = json.loads((base/'layout.json').read_text(encoding='utf-8'))
    torch.set_num_threads(config['training']['threads'])
    with threadpool_limits(limits=config['training']['threads']):
        field = FieldPINN(SupportSection(base/'geometry.json'), physical, layout, config)
        field.restore(case)
        # The saved application's two existing pressure stations anchor this readout.
        existing = np.array([r['pressure_kpa'] for r in field.pressure_readout()])/1000
        np.testing.assert_allclose(FieldReadout(field).values(layout['outer'])['pressure_mpa'],
                                   existing, rtol=1e-10, atol=1e-10)
        return write_pressure(field, case)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('regional')
    pressure_parser = commands.add_parser('pressure')
    pressure_parser.add_argument('--case', type=Path, required=True)
    args = parser.parse_args()
    result = regional_stress() if args.command == 'regional' else saved_pressure(ROOT/args.case)
    print(json.dumps(result, indent=2, allow_nan=False))
