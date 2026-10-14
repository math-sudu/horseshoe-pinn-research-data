# Horseshoe PINN research data

Research inputs and numerical response operators by **Pengcheng Zhu and
Tielin Chen** for:

*Contact-pressure inversion and monitoring augmentation in horseshoe tunnels
using a mixed physics-informed neural network*

The study uses synthetic displacement observations on a full horseshoe
engineering section with a 0.30 m support. The principal analysis jointly
infers four released-load amplitudes and effective ground stiffness, then
evaluates pressure and support stress from the same mechanical state.
Homogeneous 28-day concrete has compressive strength 60 MPa and modulus
39068.09 MPa. The stage transfers the remaining load after 90% pre-activation
release. Effective load means are (0.4, 0.5, 0, 0) MPa and standard deviations
(0.04, 0.05, 0.04, 0.04) MPa.

The common stiffness prior is uniform in log modulus from 0.7 to 1.4 GPa.
Physics-trained responses at 0.7, 1.0 and 1.4 GPa define the interpolation;
the independently trained 0.9 GPa response is held out. Reference moduli
0.8, 1.0 and 1.2 GPa share the same 2000 load and coordinate-error draws.
The separate uniform-release example follows the spatial propagation of a
scalar-reading update through the fitted ground and support fields. It retains
its 825 fits, equivalent elastic support modulus 34821 MPa and prescribed
full-release loads (4, 5) MPa.

## Contents

- `config/joint.json`: stiffness/load priors, training and evaluation inputs,
  exact C01--C48 candidate and contour coordinates, installation constraints,
  baseline row mappings and noise scenarios.
- `config/released_load.json`: material, release stage, common survey interval
  and physical-training settings.
- `data/joint_neural_family.npz`: stiffness-dependent PINN reading, pressure,
  displacement and internal-stress responses.
- `data/joint_reference.npz`: independent reference fields at the three test
  moduli, evaluated on the same reading and target locations.
- `data/joint_samples.npz`: separate calibration and test load/error draws,
  including common-datum vectors.
- `data/joint_stress_reference.npz`: finer nominal-stiffness support stresses
  at four depth profiles and throughout the section.
- `data/joint_report.json` and `data/joint_mechanics.json`: prospective scores,
  selections, pressure bands, paired recovery and mechanical interpretation.
- `data/uniform_reference.npz` and `data/uniform_seed*.npz`: compact operators
  reproducing the original uniform-release study.
- `data_dictionary.json`: array shapes, units and ordering.
- `reproduce.py` and `src/ssinv/`: inference and reconstruction code.

## Reproduce

Extract `horseshoe_pinn_research_data.zip` and run in the extracted directory:

```text
python -m pip install -r requirements.txt
python reproduce.py joint
python reproduce.py design
python reproduce.py design --target transition_stress
python reproduce.py uniform
```

The `joint` command reconstructs pressure recovery, independent pressure
bands and pressure-/stress-directed profile, face and section-force errors from the saved response
operators and paired observations. It evaluates the fixed and joint PINN
responses and the six additional noise/likelihood conditions.
Selections are those persisted before reference evaluation. The four-summary
baseline uses the same fine nominal stress reference as the complete-coordinate
and augmented baselines.

In `data/joint_report.json`, `selections` contains pressure-directed choices
and `target_designs.transition_stress.selections` contains stress-directed
choices. `data/joint_mechanics.json` collects `target_selections` and the twelve
paired `recovery` conditions. Within each condition, `baseline`, `selected`
and `stress_selected` identify the complete-coordinate baseline, pressure
addition and stress addition. The historical `fixed_design` state retains
the point selected by the fixed-stiffness pressure response.

The `design` command independently recomputes prospective scores for its target
from neural responses, priors and error models, without reading reference
fields or test outcomes. It integrates future baseline and candidate readings
using the declared scrambled and reflection-paired Sobol draws. Apply the
saved feasible index sets before selection; equivalent symmetric choices are
resolved by increasing candidate index. The default target is mean squared
contact pressure. `--target transition_stress` averages squared tangential
compression over the two lower-transition midpoints and their full thickness,
with equal section weights and normalized trapezoidal depth weights. Both
targets use the same mixed load/stiffness posterior and installation constraints.

The `uniform` command reconstructs all 825 original cases. To process one
feature map, use `python reproduce.py uniform --feature-seed 71`.
Outputs go to `results/`; `--out PATH` changes the destination.
These commands use supplied response operators without retraining.

## Array and observation conventions

Displacements are in metres; loads and stresses are in MPa. All four-mode
responses have final axis Qx, Qy, Gv, Gh. The neural-family first axis orders
700, 1000, 1400 and 900 MPa; only the first three responses define interpolation.

The raw survey has 110 coordinates: x/y at seven baseline reflectors,
then x/y at C01--C48. The asymmetric baseline relocates the deepest left
reflector to C08 through its saved row mapping. The 48-point pool combines
the original 24 locations and their exact reflections. The declared synthetic
installation clearance is 0.30 m along the closed intrados. Accessibility and
visibility exclusions are applied before selection. All acquired reflectors
share the before-release reference and after-release survey.

For each symmetric left/right baseline pair, the coordinate channels are
`u+ = (uxR-uxL, uyR+uyL)/sqrt(2)` and
`u- = (uxR+uxL, uyR-uyL)/sqrt(2)`. The Qx, Qy and Gv responses occupy `u+`
and crown uy; Gh occupies `u-` and crown ux. Keeping both crown components
and every pair channel is an orthogonal transformation of the fourteen
coordinates. Apply the same transformation to the covariance, including datum
correlations. Settlement and the three horizontal chords retain crown uy and
the horizontal part of `u+`; they omit the vertical pair sums and every
antisymmetric channel. The complete baseline arrays retain the nonzero Gh
response. The coordinate--summary recovery comparison measures the combined
value of all retained information.

The nominal instrument SD is 0.1 mm per coordinate. Doubled-noise cases rescale
the same errors. Datum cases add the same independent dx/dy draw to every
point; full and diagonal likelihoods receive identical raw errors.
Calibration uses 199 separate cases at 1.0 GPa. Its baseline and selected-action
radii transfer unchanged to the other truth moduli.

A field mean is the sum over stiffness of posterior weight times the
node-specific response times its conditional load mean. Averaging responses
and loads separately loses their dependence. Field covariance includes both
within-node covariance and between-node mean variation.

Stress profiles order crown, left transition, right transition and invert.
Depth runs from contact to intrados; components are compression-positive
tangential/normal stress and signed shear. Integrating tangential stress over
depth gives force per axial metre, in MN/m. Integrating with lever arm
`d-h/2` gives moment in MN m/m, positive toward the intrados.
Nominal section recovery uses the finer reference. The full-section reference
also permits principal-stress and elastic-strain evaluation under plane strain.

The uniform study has its own four-scalar baseline, N01--N24 normal readings
and 28-component noise vectors. Its target grids and operators retain their
original conventions and are not interchangeable with the joint study.
