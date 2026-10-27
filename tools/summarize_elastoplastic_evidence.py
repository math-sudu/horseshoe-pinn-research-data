"""Summarize target accuracy and paired design scores from saved results."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ssinv.nonlinear_response import ResponseSurface
from ssinv.quadrature import thickness_weights

OUT = ROOT / 'results/elastoplastic'


def read_json(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n',
                           encoding='utf-8')


def main():
    surface = ResponseSurface.load(OUT / 'response.npz')
    cases = {row['name']: row for row in read_json('sampling.json')['cases']}
    accuracy = read_json('response.json')
    for row in accuracy['validation']:
        case = cases[row['case']]
        x = surface.standardized(case['loads_mpa'], case['modulus_mpa'])
        predicted = surface.evaluate_standardized(x, ('stress_mpa',))['stress_mpa'][1:3, :, 0]
        with np.load(OUT / row['case'] / 'readout.npz') as saved:
            reference = saved['stress_mpa'][1:3, :, 0]
            weights = thickness_weights(saved['depth_m']) / (2 * np.ptp(saved['depth_m']))
            error = predicted - reference
            row['errors']['transition_stress_mpa'] = {
                'rmse': float(np.sqrt(np.sum(error**2 * weights))),
                'relative_l2': float(np.sqrt(np.sum(error**2 * weights) /
                                             np.sum(reference**2 * weights))),
            }
    write_json('response.json', accuracy)

    design = read_json('design_independent_01.json')
    selected = {row['target']: row for row in design['selections']
                if row['layout'] == 'symmetric_full'}
    scores = np.asarray(design['scores']['symmetric'])
    risks = np.asarray(design['risks']['symmetric'])
    contrasts = []
    for target, other in [('pressure', 'transition_stress'), ('transition_stress', 'pressure')]:
        j = design['targets'].index(target)
        chosen, alternative = selected[target]['selected_index'], selected[other]['selected_index']
        difference = 100 * (scores[:, chosen, j] - scores[:, alternative, j]) / risks[:, j].mean()
        contrasts.append(dict(target=target, selected_index=chosen, alternative_index=alternative,
            paired_difference_percentage_points=difference.tolist(),
            mean_percentage_points=float(difference.mean()),
            standard_error_percentage_points=float(difference.std(ddof=1) / np.sqrt(len(difference)))))
    design['selected_location_contrasts'] = dict(layout='symmetric_full',
        normalization='Common mean baseline variance across the saved scrambles',
        targets=contrasts)
    write_json('design_independent_01.json', design)
    print(json.dumps(dict(transition_stress_rmse_mpa=[
        row['errors']['transition_stress_mpa']['rmse'] for row in accuracy['validation']],
        selected_location_contrasts=contrasts), indent=2))


if __name__ == '__main__':
    main()
