# EmpireAI GPU workflow

This tooling is adapted from the working HalluLens EmpireAI workflow. Point
Constellation reserves ports `8600` through `8699`; HalluLens uses a separate
port namespace. Port separation makes workstream ownership machine-checkable,
even when multiple logical allocations share one physical GPU host. It keeps
credentials out of the repository and deliberately has no job-cancellation
command.

## What was ported

- a guarded SLURM Jupyter launcher with a six-allocation cap;
- live discovery of `jupyter_*_<port>` allocations through `squeue`;
- authenticated Jupyter REST/WebSocket execution;
- GPU health and free-VRAM inspection;
- background dispatch with logs and a locked local job manifest.

The old HalluLens RIT connector was not copied because it embeds an SSH key path
and passphrase. The persistent Empire shell/reaper was also excluded because it
can terminate remote processes and is unnecessary for obtaining a GPU.

## Prerequisites

The workstation SSH configuration must already provide the `empire-ai` alias:

```bash
ssh empire-ai 'hostname && squeue --me'
```

Do not put private keys, tokens, or Jupyter passwords in this repository.

## 1. Install the remote checkout

Run on the EmpireAI login node:

```bash
git clone https://github.com/hyang0129/pointconstellation.git ~/pointconstellation
cd ~/pointconstellation
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[cluster,train]'
cp configs/empire.nodes.example.json configs/empire.nodes.json
export POINTCONSTELLATION_JUPYTER_PASSWORD='<cluster-jupyter-password>'
```

The real node registry is ignored by Git because it contains ephemeral host and
allocation metadata.

## 2. Obtain a guarded Jupyter allocation

Preview only:

```bash
ssh empire-ai \
  'cd ~/pointconstellation && .venv/bin/python scripts/launch_empire_jupyter.py 8600'
```

The preview permits at most six Point Constellation allocations, counting only
jobs in the reserved `86xx` namespace. Other workstreams do not consume this
project cap. It also rejects namespace violations and port collisions.

After explicitly deciding to allocate the GPU, submit with:

```bash
ssh empire-ai \
  'cd ~/pointconstellation && .venv/bin/python scripts/launch_empire_jupyter.py 8600 --submit'
```

The launcher invokes `~/rit_rc_scripts/empire_jupyter_lab.sh` by default. Set
`EMPIRE_JUPYTER_SCRIPT` if the batch script lives elsewhere.

## 3. Discover and inspect allocated GPUs

Run these on the login node, where internal `alphagpu*` URLs are reachable:

```bash
cd ~/pointconstellation
export POINTCONSTELLATION_JUPYTER_PASSWORD='<cluster-jupyter-password>'
.venv/bin/python scripts/empire_gpu.py sync
.venv/bin/python scripts/empire_gpu.py status
```

`sync` rewrites only the ignored `configs/empire.nodes.json`, using live SLURM
state as the source of truth.

## 4. Dispatch the ML smoke experiment

Once the Experiment 1 training entry point exists, the intended command is:

```bash
.venv/bin/python scripts/empire_gpu.py run \
  --min-vram 20 \
  --desc 'experiment-001 K16 q12 smoke' \
  -- .venv/bin/python -m pointconstellation.train \
     --config configs/experiment_001.json \
     --k 16 --bits 12 --smoke
```

The dispatcher selects a reachable allocation with sufficient free VRAM,
launches the command in a new session, writes its output under
`artifacts/empire/logs/`, and records it in
`artifacts/empire/gpu_jobs.json`.

Inspect recorded jobs with:

```bash
.venv/bin/python scripts/empire_gpu.py jobs --all
```

After staging the licensed local data and ignored manifests, dispatch the two
external-dataset stability protocols with:

```bash
scripts/launch_empire_experiment_028.sh
scripts/launch_empire_experiment_029.sh
```

Their dataset preparation and split contracts are in
[`datasets.md`](datasets.md).

The manifest is observational. A disappeared allocation can leave a job marked
running; compare with `squeue --me` before assuming work is active.

## 5. Interactive access from the workstation

GPU nodes are reachable through the login node. After `sync` reports a node and
port, open a local tunnel:

```bash
ssh -N -L 18600:alphagpuXX:8600 empire-ai
```

In another shell:

```bash
export POINTCONSTELLATION_JUPYTER_URL=http://localhost:18600
export POINTCONSTELLATION_JUPYTER_PASSWORD='<cluster-jupyter-password>'
python -m pointconstellation.cluster.jupyter \
  'import torch; print(torch.cuda.get_device_name(0))'
```

## Safety rules

- The login node is for Git, SLURM inspection, and dispatch only—never training.
- Never launch compute by directly SSHing to an `alphagpu*` host. Submit through
  SLURM or execute through the selected `hostname-port` Jupyter endpoint so the
  process inherits that logical allocation's GPU cgroup and visibility.
- Treat `hostname-port`, not hostname alone, as the logical node identity. Two
  jobs on the same physical host are distinct allocations.
- Preview allocation commands before using `--submit`.
- Never work around an allocation-cap or port-collision refusal.
- Confirm remote job state before retrying a timed-out submission.
- Deploy reproducible code through Git, not ad hoc remote file edits.
- Job cancellation remains a manual, explicitly authorized cluster action.
