# Tunnel support identification and reflector placement

Code and input data for “Monitoring-driven identification of tunnel support
mechanical states under uncertain ground stiffness using an interface-aware PINN”
by Pengcheng Zhu and Tielin Chen.

The method couples Drucker–Prager elastoplastic ground with finite-thickness
Hencky elastic support. Mixed displacement and first-Piola stress fields
follow incremental material histories. A nonlinear response surface supports
joint load–stiffness inference with 41 stiffness nodes and the selection of an additional reflector
for contact pressure or support stress.

## Environment

Run the commands below from this directory in a Python environment containing
the packages in `requirements.txt`. The reported calculations used Windows,
PyTorch 2.8, NumPy 2.4.4, SciPy 1.16.2 and pypardiso 0.4.7, with double
precision and CUDA device `cuda:0`. Install a CUDA-enabled PyTorch build
compatible with the available driver before installing the requirements.
The model device is specified in the `config/elastoplastic_*.json` files.
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

Run prospective design before empirical recovery for the independent-error
surveys used in the manuscript:

```powershell
python tools/analyze_elastoplastic_study.py design --scenario independent_01
python tools/analyze_elastoplastic_study.py recovery --scenario independent_01
python tools/summarize_elastoplastic_evidence.py
python tools/analyze_elastoplastic_study.py case
```

The output directory is `results/elastoplastic/`. `response.json` summarizes
held-out response accuracy. `design_*.json` reports target-specific expected
variance reduction, and `recovery_*.json` reports empirical errors against the
refined histories. Corresponding NPZ files retain the response fields, paired
errors and reference identities. `single_case.npz` contains the conditional
pressure, support and ground fields used for mechanical interpretation.

## Mechanical and posterior references

After generating the numerical histories above, the independent displacement
finite-element reference can be reproduced with the same material laws,
initial state and proportional release. It uses its own spectral material
update and complete loading history:

```powershell
python tools/check_elastoplastic_mechanics.py --degree 4 --spacing 0.6 --steps 16 --output results/verification/mechanics_quartic
python tools/check_elastoplastic_mechanics.py --degree 4 --spacing 0.4 --radial-first 0.0075 --radial-growth 1.25 --support-layers 3 --steps 16 --output results/verification/mechanics_quartic_refined
python tools/check_elastoplastic_mechanics.py --degree 4 --spacing 0.6 --steps 32 --output results/verification/mechanics_quartic_path
python tools/check_elastoplastic_mechanics.py --compare results/verification/mechanics_quartic results/verification/mechanics_quartic_path results/verification/mechanics_quartic_refined
```

The bounded-posterior calculation reweights a Gaussian/Student-t proposal
with the original nonlinear likelihood and bounded prior. It uses the same
paired future readings and target norms as the approximate design:

```powershell
python tools/check_elastoplastic_design.py --output results/verification/design_stiffness_grid.json
python tools/check_elastoplastic_design.py --orders 41 --bounded --output results/verification/design_bounded_initial.json
python tools/check_elastoplastic_design.py --orders 81 --bounded --inner-power 9 --replicates 2 --tail-scale 3 --output results/verification/design_bounded_control.json
python tools/check_elastoplastic_design.py --baseline asymmetric --candidates -1 32 33 36 37 15 16 --orders 41 --bounded --output results/verification/design_asymmetric_bounded.json
```

These calculations support the informative invert region and its bending
information. The maximum-score candidates are used as representative
selections in recovery tests; small neighboring pressure and crown score
differences do not establish a unique installation coordinate.

## Field convergence and conditional pressure

`data/field_convergence/shoulder_history.csv` contains the 26 shoulder-convergence
readings from monitoring conducted by the authors' team. The columns report
elapsed time in days, time relative to the analysis origin, and cumulative
deformation in millimetres. The relative time subtracts six days from the
elapsed-time coordinate. Linear interpolation between adjacent readings
defines the convergence at the selected interval endpoints.

`interval.json` gives the selected endpoints at 4.1666666667 and 24.8888888889
relative days. Their interpolated cumulative readings are 96.2747935871 and
158.9872581763 mm, giving the fitted increment of 62.7124645892 mm.
`displacement_input.json` supplies this increment in metres. The layout retains
only this fitted shoulder chord; its endpoints and full left-to-right closure
definition are unchanged. Geometric and prescribed material inputs are in
`geometry.json` and `physical_config.json`.

Ground modulus 560 MPa follows the reported results for 47 groups of rock
specimens (reported mean 560.15 MPa). The design inputs prescribe 5.8 MPa
initial vertical compression for 290 m overburden, horizontal ratio 1.1,
Poisson ratio 0.30, cohesion 0.20 MPa and friction angle 25.6 degrees.
The support schedule specifies 0.27 m C30 shotcrete and HW200 ribs at 0.8 m
spacing. Nominal concrete/steel moduli of 30/200 GPa and rib area 63.53 cm²
give the area-weighted support modulus of 35.0 GPa. These values are fixed
independently of the selected convergence fit. The open invert, homogeneous
composite support, zero dilation and perfect plasticity are model assumptions.

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
The stress-free equivalent arch origin does not reconstruct the actual
stresses at the first field endpoint, where cumulative closure is already
96.2748 mm. The fitted pressure therefore describes demand on the specified
equivalent arch. `pressure.csv` retains support-side Cauchy normal pressure
along that arch, and `pressure_summary.json` gives its reference-arclength mean
and the crown and shoulder values. The unsupported invert is excluded.

`data/field_convergence/borehole_stresses.csv` contains the 58 regional
principal-stress readings used to define a second initial-stress condition.
Separate ordinary least-squares trends for SH, Sh and Sv give 12.47, 8.36
and 7.42 MPa at 290 m depth. Their horizontal-to-vertical ratios range from
1.13 to 1.68 as the horizontal direction varies. The alternative uses the
upper trend rounded to 1.7, while retaining the local design vertical
stress of 5.8 MPa and the same modulus. These are alternative equivalent
transverse-stress assignments, not statistical confidence limits.

```powershell
python tools/analyze_field_pressure.py regional
python tools/run_elastoplastic_field.py --config config/elastoplastic_field_regional_ratio.json
```

The second command recalibrates the same observed convergence and writes
`results/elastoplastic/field_convergence_regional_ratio/`. The fitted release
fraction is 0.566025643 and convergence is 62.711998407 mm. Mean arch pressure
is 2.850681878 MPa, compared with 2.935162221 MPa at the design ratio 1.1.
Pressure can also be extracted directly from either saved history:

```powershell
python tools/analyze_field_pressure.py pressure --case results/elastoplastic/field_convergence
python tools/analyze_field_pressure.py pressure --case results/elastoplastic/field_convergence_regional_ratio
```

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
