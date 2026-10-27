"""Physical contour stations and installed survey geometry."""
import numpy as np


def stations(section, role, distances):
    arcs = section.boundaries[role]['circular_arcs']
    lengths = np.array([section.scale*a['radius']*a['sweep_rad'] for a in arcs])
    edges = np.r_[0., np.cumsum(lengths)]
    rows = []
    for s in np.asarray(distances):
        s = float(s % edges[-1])
        i = min(np.searchsorted(edges, s, side='right')-1, len(arcs)-1)
        fraction = (s-edges[i])/lengths[i]
        arc = arcs[i]['arc_id']
        point = section.point(role, arc, fraction)
        normal, tangent = section.local_frame(arc, fraction)
        rows.append(dict(s_m=s, arc=arc, fraction=float(fraction), xy_m=point.tolist(),
                         normal=normal.tolist(), tangent=tangent.tolist(),
                         curvature_per_m=1/(section.scale*arcs[i]['radius'])))
    return rows


def layouts(section, n_candidate=24, n_eval=600):
    perimeter_i = section.scale*section.inner['perimeter']
    perimeter_o = section.scale*section.outer['perimeter']
    candidates = stations(section, 'initial_support_inner', (np.arange(n_candidate)+.5)*perimeter_i/n_candidate)
    for i, row in enumerate(candidates):
        row.update(id=f'N{i+1:02d}', kind='normal', depth=section.thickness)
    points = section.monitoring_points()
    baseline = [dict(id='GD', kind='settlement', arc='crown', fraction=.5, depth=section.thickness,
                     xy_m=points['GD'].tolist())]
    for pair, depth in section.raw['monitoring']['chord_depth_m'].items():
        names = pair.split('-')
        ends = [points[n] for n in names]
        locations = [section.locate('initial_support_inner', p) for p in ends]
        direction = (ends[1]-ends[0])/np.linalg.norm(ends[1]-ends[0])
        baseline.append(dict(id=pair,kind='chord', endpoints=[dict(arc=a,fraction=float(f),depth=section.thickness)
                        for a,f in locations], xy_m=[p.tolist() for p in ends], direction=direction.tolist(),
                        original_distance_m=float(np.linalg.norm(ends[1]-ends[0])), crown_depth_m=depth))
    outer = stations(section, 'initial_support_outer', (np.arange(n_eval)+.5)*perimeter_o/n_eval)
    inner = stations(section, 'initial_support_inner', (np.arange(n_eval)+.5)*perimeter_i/n_eval)
    return dict(baseline=baseline,candidates=candidates,outer=outer,inner=inner,
                inner_perimeter_m=perimeter_i,outer_perimeter_m=perimeter_o)
