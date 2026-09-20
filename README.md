# Horseshoe PINN research data

Research inputs and compact numerical response operators by **Pengcheng Zhu
and Tielin Chen** for:

*Placement of an additional displacement observation for contact pressure
inversion in horseshoe tunnels using a mixed physics informed neural network*

The study uses synthetic displacement observations on a full horseshoe
engineering section with a 0.30 m support. The archive contains the inputs,
response operators and Python code used to reconstruct pressure inference.

## Contents

- `config/`: section geometry, observation locations, material and training
  settings, load priors, random seeds, monitoring layouts and noise models.
- `data/uniform_reference.npz`: the synthetic reference readings and target
  fields for the uniform-release study.
- `data/uniform_seed*.npz`: compact response operators for the three frozen
  uniform-release estimators.
- `data/neural_responses.npz` and `data/reference_responses.npz`: four unit-load
  response maps, ordered Qx, Qy, Gv, Gh.
- `data/reference_stiffness_*.npz`: unit-load references for the prescribed
  stiffness cases.
- `data/stress_responses.npz`: neural and independent reference stress
  responses along four through-thickness profiles.
- `data_dictionary.json`: array shapes, units, ordering and reconstruction
  conventions.
- `reproduce.py` and `src/ssinv/pressure_posterior.py`: result reconstruction.

The compact uniform operators map four baseline readings to predicted
quantities and apply a rank-one update for each additional reading.
They preserve the frozen study's reported numerical precision. The four-mode
operators map load amplitudes to readings, pressure and displacement.
Sampled loads and noise are regenerated from the declared random seeds.

## Reproduce the two principal studies

Extract `horseshoe_pinn_research_data.zip`, open a terminal in the extracted
directory, and run:

```text
python -m pip install -r requirements.txt
python reproduce.py four-mode
python reproduce.py uniform
```

The four-mode command reconstructs the independently calibrated pressure
bands and the 2000-case test. The uniform command reconstructs 825 cases
across feature seeds 71, 1071 and 2071. To process one feature seed:

```text
python reproduce.py uniform --feature-seed 71
```

Outputs are written to `results/`; `--out PATH` changes the output directory.
The commands use the supplied response operators without retraining.

## Array conventions

NPZ files contain numeric NumPy arrays and can be opened with
`numpy.load(path, allow_pickle=False)`. Displacements are in metres, loads and
stresses in MPa. Unit-load maps therefore use m/MPa or MPa/MPa.
Pressure is compression-positive.

For four-mode reconstruction, multiply a `(cases, 4)` load matrix by
`pressure_mpa.T`. Observations use
`vstack([baseline_m, candidates_m])` and the declared noise model.
Set the neural baseline Gh column to zero when constructing the inference
map, as required by reflection symmetry. The reference observations retain
their independently computed response.

Calibration and test sampling use separate seeds. Within each split, draw
all four load amplitudes first and then the 28 reading errors with
`numpy.random.default_rng`; the first four readings are the installed
baseline and the remaining 24 are N01-N24. Layout and noise studies use the
separate split in `config/layouts.json`. Shared-datum errors use
`config/noise.json`: settlement receives `-dy`, chords cancel translation,
and candidate readings receive `-normal dot [dx, dy]`.

To reconstruct local stress, contract the final mode axis of a stress
response array with the load amplitudes. Station order is crown, left
transition, right transition, invert; stress-component order is tangential
compression, normal compression and signed shear. The two independent
reference meshes and their sampling locations are in `config/stress.json`.
