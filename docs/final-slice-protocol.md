# Experiment 024 final-slice protocol

## Rule

The ModelNet40 final slice is evaluated once, after every method, checkpoint,
rate, metric, comparison, and statistical decision is frozen. No result from
`final_validation` or `final_ood` may be used to add, remove, tune, select, or
retrain a method. This is a final test, not another validation pass.

The run must use `configs/experiment_024_final_slice.json` without a cloud
limit. The runner rejects `--max-clouds-per-split` and any underlying official
configuration that sets `max_clouds_per_split`. It writes results below
`artifacts/local/final_slice/<manifest_sha256>/`. Before evaluating a cloud it
writes `FINAL_SLICE_LOCK`, containing the exact manifest SHA-256, Git commit,
and UTC timestamp. Once `official_metrics.json` exists, every later invocation
for those exact manifest bytes fails before changing a file.

An interrupted run may resume only while `official_metrics.json` is absent and
the lock still names the same manifest and Git commit. The original lock
timestamp is preserved. Any protocol or code change requires abandoning the
claim that the slice was evaluated only once; it must not be hidden by deleting
the lock or completed metrics.

The run-once guard does not make the result a general compression claim. The
learned message remains only the serialized, unordered, exact-quantized
`K x 3` coordinate set. Model size remains separate from per-cloud rate, and
nearest-sample metrics remain finite-cloud proxies.

## Manifest construction

The manifest builder scans the official ModelNet40 test split and excludes any
test mesh named by the pilot, scale, or stability manifest. It records the full
excluded-mesh records and the path and SHA-256 of every input manifest. Within
the 32 decoder-training categories, remaining meshes are ranked
deterministically and capped per category. All remaining meshes from the eight
held-out categories enter `final_ood`.

For the declared 16-cloud validation cap, the three input manifests have these
hashes:

- `modelnet40_pilot.local.json`:
  `c5ce18c64f98f26410c4f248e31ff576dfa1de277ebdf1ca373f5d2f520f4297`
- `modelnet40_scale.local.json`:
  `b08bbe80713ae748e50c8b91308a14abd6240854fe1452af6027d90b35cf4a6f`
- `modelnet40_stability.local.json`:
  `d44014dd2313b8815562cde9df2ba1927e1110fcfbc218428a7db39ef6b829ac`

Regenerate the ignored local manifest only after the three input manifests are
available with those exact hashes:

```bash
.venv-train/bin/python -m pointconstellation.mesh_manifest \
  --dataset modelnet40-final-slice \
  --root data/modelnet40_official/ModelNet40 \
  --output configs/manifests/modelnet40_final_slice.local.json \
  --validation-cap-per-category 16 \
  --minimum-ood-clouds 200 \
  --seed 1517
```

Against the declared archive and input manifests, this produces 512
`final_validation` clouds, exactly 16 per training category, and 528
`final_ood` clouds. The held-out counts are 96 each for bed, bottle, monitor,
range hood, and table, and 16 each for bowl, stairs, and stool. Thus all eight
held-out categories are represented and the OOD count exceeds 200.

The evaluator maps these two manifest partitions to the existing official
stability evaluator's `validation` and `ood` analysis labels. It first verifies
the Experiment 019 artifact configuration, scientific-contract checks, sealed
selection, decoder/refiner hashes, and original data identity. The final-slice
manifest and its partition identities are then recorded separately in the
Experiment 024 run manifest.

## Method freeze

`configs/experiment_024_final_slice.json` references the frozen Experiment 019
decoder/refiner artifacts and records the intended Experiment 021 selection
baselines and Experiment 022 equal-protocol feature-codec configuration paths.
The latter paths are integration placeholders until those experiments merge.
Do not invoke Experiment 024 until those methods are present, their final-slice
adapters are registered, and the complete method set is frozen. This pull
request does not run the command below:

```bash
.venv-train/bin/python -m pointconstellation.final_slice \
  --config configs/experiment_024_final_slice.json \
  --device mps
```

## Previously used official-test clouds

The following 160 clouds informed Experiments 016--020 and are excluded. The
pilot's 40 clouds are contained in this list.

| Category | Excluded model IDs |
|---|---|
| airplane | `airplane_0641`, `airplane_0648`, `airplane_0662`, `airplane_0711` |
| bathtub | `bathtub_0129`, `bathtub_0140`, `bathtub_0143`, `bathtub_0155` |
| bed | `bed_0525`, `bed_0549`, `bed_0555`, `bed_0599` |
| bench | `bench_0175`, `bench_0183`, `bench_0186`, `bench_0191` |
| bookshelf | `bookshelf_0608`, `bookshelf_0622`, `bookshelf_0638`, `bookshelf_0671` |
| bottle | `bottle_0347`, `bottle_0418`, `bottle_0419`, `bottle_0422` |
| bowl | `bowl_0065`, `bowl_0067`, `bowl_0076`, `bowl_0079` |
| car | `car_0243`, `car_0264`, `car_0290`, `car_0291` |
| chair | `chair_0895`, `chair_0901`, `chair_0942`, `chair_0973` |
| cone | `cone_0170`, `cone_0172`, `cone_0184`, `cone_0185` |
| cup | `cup_0082`, `cup_0084`, `cup_0091`, `cup_0099` |
| curtain | `curtain_0139`, `curtain_0146`, `curtain_0153`, `curtain_0154` |
| desk | `desk_0225`, `desk_0233`, `desk_0245`, `desk_0268` |
| door | `door_0115`, `door_0122`, `door_0123`, `door_0129` |
| dresser | `dresser_0214`, `dresser_0227`, `dresser_0228`, `dresser_0286` |
| flower_pot | `flower_pot_0161`, `flower_pot_0162`, `flower_pot_0164`, `flower_pot_0167` |
| glass_box | `glass_box_0214`, `glass_box_0262`, `glass_box_0266`, `glass_box_0270` |
| guitar | `guitar_0187`, `guitar_0241`, `guitar_0242`, `guitar_0247` |
| keyboard | `keyboard_0151`, `keyboard_0152`, `keyboard_0163`, `keyboard_0164` |
| lamp | `lamp_0129`, `lamp_0137`, `lamp_0141`, `lamp_0143` |
| laptop | `laptop_0151`, `laptop_0158`, `laptop_0161`, `laptop_0162` |
| mantel | `mantel_0286`, `mantel_0339`, `mantel_0350`, `mantel_0382` |
| monitor | `monitor_0469`, `monitor_0479`, `monitor_0506`, `monitor_0534` |
| night_stand | `night_stand_0203`, `night_stand_0217`, `night_stand_0220`, `night_stand_0253` |
| person | `person_0091`, `person_0096`, `person_0098`, `person_0108` |
| piano | `piano_0252`, `piano_0296`, `piano_0310`, `piano_0312` |
| plant | `plant_0250`, `plant_0291`, `plant_0300`, `plant_0323` |
| radio | `radio_0105`, `radio_0111`, `radio_0115`, `radio_0120` |
| range_hood | `range_hood_0131`, `range_hood_0137`, `range_hood_0141`, `range_hood_0196` |
| sink | `sink_0129`, `sink_0136`, `sink_0137`, `sink_0148` |
| sofa | `sofa_0707`, `sofa_0725`, `sofa_0761`, `sofa_0763` |
| stairs | `stairs_0130`, `stairs_0132`, `stairs_0138`, `stairs_0142` |
| stool | `stool_0091`, `stool_0105`, `stool_0108`, `stool_0110` |
| table | `table_0430`, `table_0450`, `table_0471`, `table_0489` |
| tent | `tent_0165`, `tent_0167`, `tent_0179`, `tent_0183` |
| toilet | `toilet_0352`, `toilet_0379`, `toilet_0417`, `toilet_0437` |
| tv_stand | `tv_stand_0315`, `tv_stand_0319`, `tv_stand_0337`, `tv_stand_0350` |
| vase | `vase_0522`, `vase_0539`, `vase_0565`, `vase_0572` |
| wardrobe | `wardrobe_0094`, `wardrobe_0102`, `wardrobe_0105`, `wardrobe_0107` |
| xbox | `xbox_0105`, `xbox_0111`, `xbox_0116`, `xbox_0118` |
