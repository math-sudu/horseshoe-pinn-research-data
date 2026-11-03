"""Run an independent weak-form reference and compare saved mechanical targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ssinv.reference_fem import ReferenceFEM, ReferenceMesh, SpectralMaterial, plane
from ssinv.measurements import baseline_points


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def material_checks(config):
    """Closed principal-strain paths and objectivity, without the production kernel."""
    m = SpectralMaterial(config)
    eye = np.eye(3)
    p, sigma, _, dl = m.ground(eye, eye, commit=True)
    np.testing.assert_allclose(sigma, m.sigma0, atol=5e-11)
    np.testing.assert_allclose(p, 0., atol=5e-11)
    # A diagonal plane-strain compression with a nonzero shear invariant.
    ft = np.diag(np.exp([-.012, .002, 0.]))
    f = ft @ np.linalg.inv(m.f0)
    p, sigma, cp, dl = m.ground(f, eye, commit=True)
    ev = np.array([-.012, .002, 0.])
    dev = ev-ev.mean()
    mean = m.kg*ev.sum()
    qt = 2*m.gg*np.sqrt(1.5*np.sum(dev**2))
    cap = -m.alpha*mean+np.linalg.det(ft)*m.strength
    expected = np.diag(mean+2*m.gg*dev*min(1., cap/qt))/np.linalg.det(ft)
    np.testing.assert_allclose(sigma, expected, atol=5e-10)
    if not dl > 0:
        raise AssertionError('The test must activate ground yielding')
    np.testing.assert_allclose(np.linalg.det(cp), 1., atol=2e-12)
    angle = .41
    rotation = plane(np.array([[np.cos(angle), -np.sin(angle)],
                              [np.sin(angle), np.cos(angle)]]))
    _, rotated, _, _ = m.ground(rotation@f, eye, commit=True)
    np.testing.assert_allclose(rotated, rotation@sigma@rotation.T, atol=5e-10)
    # Keeping a converged deformation fixed must not add plastic strain.
    _, repeated, _, repeat_dl = m.ground(f, cp, commit=True)
    np.testing.assert_allclose(repeated, sigma, atol=5e-10)
    np.testing.assert_allclose(repeat_dl, 0., atol=2e-12)
    support_f = np.diag(np.exp([-.004, .001, 0.]))
    ev = np.array([-.004, .001, 0.])
    expected = np.diag(2*m.gs*(ev-ev.mean())+m.ks*ev.sum())/np.linalg.det(support_f)
    np.testing.assert_allclose(m.support(support_f, cauchy=True), expected, atol=1e-9)
    return dict(initial_stress_mpa=m.sigma0.tolist(), initial_f=m.f0.tolist(),
                diagonal_return=True, zero_plastic_volume=True, objectivity=True,
                fixed_state=True, support_hencky=True)


def readout(field, layout, depth):
    coordinates = field.values(baseline_points(layout)+layout['candidates'])['displacement_m']
    outer = field.values([dict(p, depth=0.) for p in layout['outer']])
    inner = field.values([dict(p, depth=field.mesh.thickness) for p in layout['inner']])
    profiles = field.values([dict(arc=arc, fraction=.5, depth=float(d))
        for arc in ('crown', 'left_knee', 'right_knee', 'invert') for d in depth])
    return dict(baseline_xy_m=coordinates[:7], candidate_xy_m=coordinates[7:],
                pressure_mpa=outer['pressure_mpa'], inner_u_m=inner['displacement_m'],
                stress_mpa=np.stack((profiles['compression_mpa'], profiles['pressure_mpa'],
                                    profiles['shear_mpa']), -1).reshape(4, len(depth), 3),
                depth_m=depth)


def assembly_check(geometry, config):
    """Check the assembled Newton derivative in a coupled displacement mesh."""
    mesh = ReferenceMesh(geometry, config, spacing=2., radial_first=.04,
                         radial_growth=1.7, support_layers=1, degree=4)
    field = ReferenceFEM(mesh, config)
    rng = np.random.default_rng(12)
    u = rng.normal(size=field.u.shape)*1e-6
    direction = rng.normal(size=field.u.shape)*1e-5
    u.ravel()[len(mesh.free):] = 0.
    direction.ravel()[len(mesh.free):] = 0.
    loads = np.array(config['nominal_loads_mpa'])
    _, _, matrix = field.residual(u, loads, tangent=True)
    h = 1e-3
    plus, _ = field.residual(u+h*direction, loads)
    minus, _ = field.residual(u-h*direction, loads)
    numerical = (plus-minus)/(2*h)
    assembled = matrix @ direction.ravel()
    error = float(np.linalg.norm(numerical-assembled)/np.linalg.norm(assembled))
    if error > 1e-5:
        raise AssertionError(f'Assembled material/geometry derivative differs: {error}')
    return dict(assembled_tangent_relative_error=error)


def target_summary(readings):
    depth = readings['depth_m']
    delta = np.diff(depth)
    w = np.r_[delta[0], delta[:-1]+delta[1:], delta[-1]]/2
    z = depth-(depth[0]+depth[-1])/2
    compression = readings['stress_mpa'][1:3, :, 0]
    return dict(pressure_range_mpa=[float(readings['pressure_mpa'].min()),
                                    float(readings['pressure_mpa'].max())],
        transition_stress_range_mpa=[float(compression.min()), float(compression.max())],
        axial_force_mn_per_m=(compression @ w).tolist(),
        bending_moment_mnm_per_m=(compression @ (w*z)).tolist(),
        maximum_displacement_mm=float(1000*np.linalg.norm(readings['inner_u_m'], axis=1).max()),
        baseline_displacement_mm=(1000*readings['baseline_xy_m']).tolist())


def comparison(a, b):
    depth = a['depth_m']
    delta = np.diff(depth)
    weights = np.r_[delta[0], delta[:-1]+delta[1:], delta[-1]]/(2*np.ptp(depth))
    x, y = a['stress_mpa'][1:3, :, 0], b['stress_mpa'][1:3, :, 0]
    stress_rmse = float(np.sqrt(np.sum((x-y)**2*weights/2)))
    pressure_rmse = float(np.sqrt(np.mean((a['pressure_mpa']-b['pressure_mpa'])**2)))
    sa, sb = target_summary(a), target_summary(b)
    return dict(baseline_coordinate_rmse_mm=float(1000*np.sqrt(np.mean(
                    (a['baseline_xy_m']-b['baseline_xy_m'])**2))),
                pressure_rmse_mpa=pressure_rmse,
                pressure_relative_l2=pressure_rmse/float(np.sqrt(np.mean(b['pressure_mpa']**2))),
                transition_stress_rmse_mpa=stress_rmse,
                transition_stress_relative_l2=stress_rmse/float(np.sqrt(np.sum(y*y*weights/2))),
                axial_force_difference_mn_per_m=(np.array(sa['axial_force_mn_per_m'])-
                                                sb['axial_force_mn_per_m']).tolist(),
                bending_moment_difference_mnm_per_m=(np.array(sa['bending_moment_mnm_per_m'])-
                                                     sb['bending_moment_mnm_per_m']).tolist())


def main(args):
    if args.compare:
        entries, readings = [], []
        settings = None
        for name in args.compare:
            folder = ROOT/name
            cfg = json.loads((folder/'config.json').read_text())
            if settings is not None and cfg['physical_settings'] != settings:
                raise ValueError('Mesh/path comparisons require the same physical problem')
            settings = cfg['physical_settings']
            summary = json.loads((folder/'summary.json').read_text())
            readings.append(np.load(folder/'readout.npz'))
            entries.append(dict(path=name, elements=summary['elements'], nodes=summary['nodes'],
                steps=cfg['steps'], degree=cfg.get('degree', 2), targets=summary['targets']))
        report = dict(physical_case=cfg['case'], references=entries,
            successive_differences=[dict(first=args.compare[i-1], second=args.compare[i],
                differences=comparison(readings[i-1], readings[i])) for i in range(1, len(readings))],
            refinement_to_final={name:comparison(value, readings[-1])
                                 for name, value in zip(args.compare[:-1], readings[:-1])},
            comparison_to_final={name:comparison(np.load(ROOT/'results/elastoplastic'/name/'readout.npz'),
                                  readings[-1]) for name in dict.fromkeys(['baseline', 'reference_baseline', cfg['case']])})
        write_json(ROOT/args.report, report)
        print(json.dumps(report, indent=2))
        return
    settings = json.loads((ROOT/'results/elastoplastic'/args.case/'config.json').read_text())
    config = settings['config']
    checks = material_checks(config)
    if args.material_only:
        geometry = json.loads((ROOT/config['geometry']).read_text())
        checks.update(assembly_check(geometry, config))
        print(json.dumps(checks, indent=2))
        return
    output = ROOT/args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output/'solution.npz').exists():
        raise FileExistsError(f'Reference already exists: {output}')
    geometry = json.loads((ROOT/config['geometry']).read_text())
    layout = json.loads((ROOT/'results/elastoplastic'/args.case/'layout.json').read_text())['layout']
    mesh = ReferenceMesh(geometry, config, args.spacing, args.radial_first,
                         args.radial_growth, args.support_layers, args.degree)
    field = ReferenceFEM(mesh, config)
    write_json(output/'config.json', dict(case=args.case, physical_settings=settings,
        spacing_m=args.spacing, radial_first=args.radial_first, radial_growth=args.radial_growth,
        support_layers=args.support_layers, degree=args.degree, steps=args.steps))
    print(json.dumps(dict(elements=len(mesh.elements), nodes=mesh.n_nodes,
                         free_dofs=len(mesh.free), radial_layers=mesh.nr, segments=mesh.nt)), flush=True)
    started = time.perf_counter()
    with (output/'progress.jsonl').open('w') as stream:
        def emit(record):
            record['elapsed_seconds'] = time.perf_counter()-started
            text = json.dumps(record)
            stream.write(text+'\n')
            stream.flush()
            print(text, flush=True)
        field.solve(settings['loads_mpa'], args.steps, emit)
    depth = np.linspace(0., mesh.thickness, config['evaluation']['depth_points'])
    readings = readout(field, layout, depth)
    np.savez_compressed(output/'readout.npz', **readings)
    np.savez_compressed(output/'solution.npz', displacement_m=field.u,
        accepted_displacements_m=np.array(field.accepted), inverse_cp=field.inverse_cp,
        kappa=field.kappa, connectivity=mesh.conn)
    compare = {}
    for case in dict.fromkeys(['baseline', 'reference_baseline', args.case]):
        saved = np.load(ROOT/'results/elastoplastic'/case/'readout.npz')
        compare[case] = comparison(saved, readings)
    summary = dict(elements=len(mesh.elements), nodes=mesh.n_nodes, steps=args.steps,
        elapsed_seconds=time.perf_counter()-started, material_checks=checks,
        metrics=field.metrics, targets=target_summary(readings), comparison_to_fem=compare)
    write_json(output/'summary.json', summary)
    (output/'progress.jsonl').unlink()
    print(json.dumps(dict(targets=summary['targets'], comparison_to_fem=compare), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='baseline')
    parser.add_argument('--spacing', type=float, default=.5)
    parser.add_argument('--radial-first', type=float, default=.015)
    parser.add_argument('--radial-growth', type=float, default=1.3)
    parser.add_argument('--support-layers', type=int, default=2)
    parser.add_argument('--degree', type=int, default=2)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--material-only', action='store_true')
    parser.add_argument('--compare', nargs='+')
    parser.add_argument('--report', default='results/verification/mechanics_comparison.json')
    parser.add_argument('--output', default='results/verification/mechanics_coarse')
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        main(args)
