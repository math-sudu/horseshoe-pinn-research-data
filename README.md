# Tunnel support identification and reflector placement

Code and input data for “Monitoring-driven identification of tunnel support
mechanical states under uncertain ground stiffness using an interface-aware PINN”
by Pengcheng Zhu and Tielin Chen.

The method couples Drucker–Prager elastoplastic ground with finite-thickness
Hencky elastic support. Mixed displacement and first-Piola stress fields
follow incremental material histories. A nonlinear response surface supports
joint load–stiffness inference and the selection of an additional reflector
for contact pressure or support stress.

## Environment

Run the commands below from this directory in a Python environment containing
the packages in `requirements.txt`. The reported calculations used Windows,
PyTorch 2.8, NumPy 2.4.4, SciPy 1.16.2 and pypardiso 0.4.7, with double
precision and CUDA device `cuda:0`. Install a CUDA-enabled PyTorch build
compatible with the available driver before installing the requirements.
The model device is specified in the two `config/elastoplastic_*.json` files.
The saved configuration uses the PARDISO sparse solver.

```powershell
python -m pip install -r requirements.txt
```

## Numerical study

`config/horseshoe_geometry.json` defines the five-centered section and 0.30 m
support. `config/elastoplastic_study.json` specifies material properties,
four release modes, priors, random-feature seed and solution settings.
`config/observation_design.json` defines baseline reflector positions and
candidate access. These are prescribed numerical scenarios.

Starting with an empty result directory, solve the three nominal-load anchors
and then generate the sampled histories and response surface:

```powershell
python tools/run_elastoplastic_study.py forward --case baseline --modulus 1000
python tools/run_elastoplastic_study.py forward --case edge_low --modulus 700
python tools/run_elastoplastic_study.py forward --case edge_high --modulus 1400
python tools/build_elastoplastic_response.py campaign --workers 4
python tools/build_elastoplastic_response.py ground --workers 2
python tools/build_elastoplastic_response.py interpolate
```

The campaign fixes 64 training states, eight held-out states, four isolated
load changes and nine independently refined reference histories. It resumes
incomplete campaign histories and reuses completed ones. An explicitly named
forward run requires `--resume` to continue an existing accepted history.
Reduce `--workers` if parallel histories exceed the available memory.

Run prospective design before empirical recovery for each measurement-error
scenario. The following PowerShell loop reproduces all four scenarios:

```powershell
foreach ($scenario in @('independent_01', 'independent_02', 'datum_full', 'datum_diagonal')) {
    python tools/analyze_elastoplastic_study.py design --scenario $scenario
    python tools/analyze_elastoplastic_study.py recovery --scenario $scenario
}
python tools/summarize_elastoplastic_evidence.py
python tools/analyze_elastoplastic_study.py case
```

The output directory is `results/elastoplastic/`. `response.json` summarizes
held-out response accuracy. `design_*.json` reports target-specific expected
variance reduction, and `recovery_*.json` reports empirical errors against the
refined histories. Corresponding NPZ files retain the response fields, paired
errors and reference identities. `single_case.npz` contains the conditional
pressure, support and ground fields used for mechanical interpretation.

## Field shoulder-convergence calibration

`data/field_convergence/shoulder_history.csv` contains all 26 digitized shoulder
readings from monitoring collected by the authors' team. The deformation values
are unchanged. `source_elapsed_time_day` preserves the original digitized
abscissa; `relative_day` subtracts six days to use the comparison interval's
time origin. Fractional days are digitized plot coordinates rather than recovered
instrument timestamps. Connecting the readings uses linear interpolation.

`interval.json` gives the selected endpoints at 4.1666666667 and 24.8888888889
relative days. Their interpolated cumulative readings are 96.2747935871 and
158.9872581763 mm, giving the fitted increment of 62.7124645892 mm.
`displacement_input.json` supplies this increment in metres. The layout retains
only this fitted shoulder chord; its endpoints and full left-to-right closure
definition are unchanged. Geometric and prescribed material inputs are in
`geometry.json` and `physical_config.json`.

```powershell
python tools/run_elastoplastic_field.py
```

Use `--resume` only to continue an existing accepted field history. The field
configuration prescribes material properties and the initial-stress ratio and
uses an open invert with traction-free arch ends. Eight displacement-controlled
continuation steps fit an effective release magnitude. They are solution steps,
not dated stages of construction. The measured increment is a calibration
target. The reported fitted increment is 62.708004656 mm and the effective
release fraction is 0.979441690; the run writes its results to
`results/elastoplastic/field_convergence/`.

## Code organization

- `src/ssinv/finite_strain/`: material return, coupled field equations,
  accepted history and physical readouts.
- `src/ssinv/nonlinear_response.py`: polynomial response and Gaussian moments.
- `src/ssinv/nonlinear_posterior.py`: conditional approximate inference and
  stiffness integration.
- `src/ssinv/observation_design.py`: reflector geometry and access constraints.
- `tools/`: computation and analysis entry points used in the commands above.

The package contains source code, configuration and non-regenerable field
inputs. The commands generate the numerical outputs locally.
