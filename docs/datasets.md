# External dataset preparation

Downloaded archives, extracted geometry, and generated `*.local.json` manifests
are local research inputs. They must not be committed or redistributed. Both
builders record a SHA-256 for every referenced file and create disjoint
`train`, `calibration`, `validation`, `ood`, and `final` roles. The final role
is reserved and is not consumed by Experiments 028 or 029.

Install the optional HDF5 adapter when preparing ScanObjectNN:

```bash
.venv-train/bin/python -m pip install -e '.[datasets,train]'
```

## Thingi10K

### Access and licensing

Use the [Thingi10K project](https://github.com/Thingi10K/Thingi10K) or one of
the mirrors linked by its maintainers to obtain the raw variant. The organizer
code is Apache-2.0, but that license does not replace the license of an
individual Thingiverse upload. Every selected manifest record therefore keeps
the source `license` field. A missing source value is recorded as `unknown`;
it is not inferred from the organizer repository. Do not redistribute the
meshes.

The builder accepts CSV, JSON, or JSONL metadata. The most direct route is the
maintained `thingi10k` package and a narrow JSONL export:

```python
import json
from pathlib import Path

import thingi10k

cache = Path("data/thingi10k_raw")
thingi10k.init(cache_dir=cache, variant="raw")
fields = (
    "file_id",
    "category",
    "license",
    "num_components",
    "closed",
    "euler",
)
with Path("data/thingi10k_metadata.jsonl").open("w") as output:
    for entry in thingi10k.dataset():
        output.write(json.dumps({field: entry.get(field) for field in fields}) + "\n")
```

Point `--root` at the common ancestor of the raw mesh paths reported by that
package. The loader supports binary STL, ASCII STL, OBJ, and OFF. The official
errata include corrupt and empty files; parse failures are excluded by the
declared validity rule.

### Exact manifest procedure

```bash
.venv-train/bin/python -m pointconstellation.mesh_manifest \
  --dataset thingi10k \
  --root data/thingi10k_raw \
  --metadata data/thingi10k_metadata.jsonl \
  --output configs/manifests/thingi10k_stability.local.json \
  --train-count 1200 \
  --calibration-count 200 \
  --validation-count 200 \
  --ood-count 200 \
  --final-count 200 \
  --minimum-faces 500 \
  --seed 2028
```

The procedure is deterministic:

1. Recursively discover supported files whose stem matches a metadata
   `file_id`; parse each mesh, drop triangles with doubled area at or below
   `1e-12`, and retain meshes with at least 500 remaining faces.
2. Record the relative path, file SHA-256, source license, valid face count,
   category, and selection proxies. Face-count bins are 500--1,999,
   2,000--7,999, and 8,000 or more. For a closed mesh with component and Euler
   metadata, the genus proxy is `(2 * components - Euler) / 2`, coarsened to
   0, 1, or 2 or more. Open meshes and incomplete topology metadata form a
   separate `open_or_unknown` stratum. This is a selection proxy, not a new
   topology measurement.
3. Rank categories by SHA-256 of the seed and category, holding out categories
   until at least 200 OOD records are available while leaving enough ID
   records. An explicit comma-separated `--heldout-categories` overrides this
   deterministic choice.
4. Place one deterministic anchor from every ID category in training, then
   allocate the remaining exact split counts proportionally across the joint
   size/genus strata using largest remainders. Within a stratum, rank model IDs
   by a seeded SHA-256, then sort selected records by category and model ID.

The resulting manifest contains exactly 2,000 records and 200 category-held-out
OOD meshes. The manifest itself is ignored because it includes source-specific
metadata and paths; regenerate it from the retained metadata export.

## ScanObjectNN

### Access and licensing

Request and download the processed HDF5 files through the links and terms on
the [official ScanObjectNN repository](https://github.com/hkust-vgd/scanobjectnn).
Access is manual and form-gated; no downloader or CI network path is provided.
The official repository states an MIT license for the release. Retain the
download terms with the local copy and do not redistribute the HDF5/NPZ data.

Experiment 029 uses the unaugmented main-split OBJ_BG containers described by
the maintainers:

```text
data/scanobjectnn/main_split/training_objectdataset.h5
data/scanobjectnn/main_split/test_objectdataset.h5
```

The adapter also reads equivalent NPZ containers. The point array key is
`data` (or `points` for NPZ), and the optional label key is `label` (or
`labels`). HDF5 support is imported lazily so the NumPy-only core package still
imports without `h5py`.

### Exact manifest procedure

```bash
.venv-train/bin/python -m pointconstellation.mesh_manifest \
  --dataset scanobjectnn \
  --root data/scanobjectnn/main_split \
  --output configs/manifests/scanobjectnn_stability.local.json \
  --train-files training_objectdataset.h5 \
  --test-files test_objectdataset.h5 \
  --train-count 1000 \
  --calibration-count 150 \
  --validation-count 100 \
  --ood-count 200 \
  --final-count 100 \
  --seed 2029
```

The builder expands each container row into a record and stores its container
SHA-256, row index, numeric category label, official train/test membership, and
`normals_estimated=true`. Categories are named `label_00`, `label_01`, and so
on unless a comma-separated `--category-names` list is supplied. Categories
are deterministically held out from both training roles; their official-test
records supply at least 200 OOD clouds. Train/calibration use only official
training rows. Validation, OOD, and final use only official-test rows. Each
role is category-stratified and every row identity is disjoint.

For a loaded row, the adapter applies bounding-box centering and unit-radius
scaling, then deterministically partitions the finite scan indices into
disjoint source and fresh halves. Each half is sampled to 2,048 points, with
replacement because the official processed rows contain 2,048 points total.
Thus repeated points can occur within a role, but no original point index can
occur in both roles. Normals are estimated by 16-neighbor PCA on the normalized
finite scan. D2 results must retain the `normals_estimated=true` label; these
are not analytic or sensor-provided normals.

Category labels, normals, and sampling indices are metadata. The stability
runner forwards only `source_points` to the coordinate encoder/refiner and the
frozen decoder. The numeric label remains available in dataset items for the
separate Track B classification tasks.

## Stability runs on EmpireAI

After the local datasets and ignored manifests have been staged in the same
paths on the remote checkout, use the guarded dispatchers:

```bash
scripts/launch_empire_experiment_028.sh
scripts/launch_empire_experiment_029.sh
```

They dispatch the six-decoder by three-refiner Experiment 019 protocol through
the existing EmpireAI Jupyter allocation registry. They do not download data,
obtain an allocation, cancel a job, or evaluate the reserved final split.
