# Experiments 008-012: co-adaptation fan-out

## Outcome

All five additional co-adaptation hypotheses now have runnable, CPU-safe
prototypes. None of their one-epoch smoke tests is a scale result, but the
batch narrows the next decision:

- balanced allocation and homotopy did not beat their matched direct controls;
- decoder-population training was effectively tied with single-decoder
  training;
- frozen-decoder gradients decisively beat two gradient-free searches at a
  tiny matched query budget; and
- an autoregressive pointer beat a matched scalar ranker at every corrected
  operating point, but remained worse than FPS everywhere.

The strongest current direction remains Experiment 005's competitive recurrent
refiner with legal encoder-side decoder gradients. The fan-out does not yet
support diffusion, gradient-free search, homotopy, or population training as a
replacement.

## Shared contract

Every experiment transmits only an exactly quantized `K x 3` coordinate set.
Decoder parameters are frozen during encoder/search comparisons and verified by
state hashes. Surface distance is the nearest distance to finite observed
samples, not to the analytic generating surface.

These configurations are deliberately tiny integration/scientific-direction
smokes. A negative result here rejects neither the mechanism nor its tuned,
scaled form.

## Experiment 008: balanced transport and density-aware objectives

Hypothesis H9 asks whether explicit mass balance prevents several slots from
claiming the same region. Three matched arms share initialization, batches, and
one frozen decoder:

- ordinary Chamfer with competitive responsibilities;
- density-aware Chamfer with competitive responsibilities; and
- balanced reconstruction transport plus balanced Sinkhorn `K x N` internal
  slot responsibilities.

| Arm | Validation step 0 | Validation final | OOD step 0 | OOD final |
|---|---:|---:|---:|---:|
| Chamfer | 0.440095 | 0.441363 | 0.371390 | 0.372699 |
| Density-aware | 0.440095 | 0.441311 | 0.371390 | 0.371529 |
| Balanced transport | 0.440095 | **0.440831** | 0.371390 | 0.372372 |

No arm improves reconstruction in one epoch. Balanced transport is least bad
on validation and density-aware is least bad on OOD. The objectives have very
different numerical scales under one learning rate, so a scaled comparison
must normalize or tune each arm before judging H9.

```bash
.venv-train/bin/python -m pointconstellation.transport_experiment \
  --config configs/experiment_008_transport_smoke.json --device cpu
```

## Experiment 009: compression homotopy

Hypothesis H10 asks whether gradual set merging provides a continuous path out
of a bad direct target-rate basin. A shared conditional merge transition follows
`K=16 -> 8 -> 4`; the direct arm receives the same optimizer-step budget and
initial parameter state. The decoder is trained at every stage size, and FPS is
reported at every rate.

| Split, `K=4` | Homotopy | Direct | FPS |
|---|---:|---:|---:|
| Validation | 0.399087 | **0.396282** | 0.428694 |
| Parameter OOD | 0.392330 | **0.384739** | 0.404286 |

Homotopy is 0.71% worse than direct on validation and 1.97% worse on OOD,
although both learned transitions beat FPS in this smoke. This is not an
isolated rejection of reachability: direct gets target-`K` supervision every
epoch, while homotopy reaches `K=4` only in its final curriculum phase. A
follow-up must match target-rate exposure, not merely optimizer steps.

```bash
.venv-train/bin/python -m pointconstellation.homotopy_experiment \
  --config configs/experiment_009_homotopy_smoke.json --device cpu
```

## Experiment 010: decoder populations and cross-play

Hypothesis H11 asks whether training against several independently seeded
decoders discourages a brittle private coordinate language. The population arm
optimizes mean plus weighted-worst reconstruction across three decoders; its
matched control optimizes decoder 0. Evaluation records the full
message-row-by-decoder matrix, paired lattice perturbations, and population
deltas normalized per decoder.

Across eight validation/OOD operating points, population training is better at
two and worse at six. The mean normalized delta is `+0.0436%` (positive is
worse), with a range from `-0.0560%` to `+0.1192%`: effectively a smoke-scale
tie.

This tests robust multi-decoder optimization, not yet decisive private-language
cross-play. There is no separately trained paired encoder for every decoder, so
decoder quality and interoperability cannot be fully separated. A decisive
version needs one encoder per decoder plus a population encoder and normalized
off-diagonal regret.

```bash
.venv-train/bin/python -m pointconstellation.crossplay_experiment \
  --config configs/experiment_010_crossplay_smoke.json --device cpu
```

## Experiment 011: gradient-free constellation search

Hypothesis H12 asks whether non-local, gradient-free search crosses decoder
basins that local gradient descent cannot. Coordinate CEM, unique subset
mutation, and Adam/STE start from the exact same FPS or random constellation and
receive nine decoder-score queries per cloud.

| Split/start | Adam/STE | Coordinate CEM | Subset mutation |
|---|---:|---:|---:|
| Validation/FPS | **0.30795** | 0.34240 | 0.32373 |
| Validation/random | **0.31532** | 0.36438 | 0.33870 |
| OOD/FPS | **0.25044** | 0.29244 | 0.28518 |
| OOD/random | **0.25167** | 0.31547 | 0.28439 |

Adam improves every cloud and wins all four comparisons. Subset mutation is
usually useful; CEM makes only small gains with two generations of population
four. The fair interpretation is query-budget, not compute-budget, equality:
an Adam query includes a backward pass, while CEM and mutation use forward
scores only. The subset arm is also representation-constrained whereas the two
coordinate arms are free. This smoke supports useful decoder gradients; it does
not rule out larger or better-tuned global searches.

```bash
.venv-train/bin/python -m pointconstellation.gradient_free_experiment \
  --config configs/experiment_011_gradient_free_smoke.json --device cpu
```

## Experiment 012: autoregressive pointer subset selection

Hypothesis H13 asks whether each selection must condition on already selected
points rather than use one global scalar ranking. The corrected comparison:

- trains every Cartesian `(N,K)` point rather than only diagonal pairs;
- matches pointer/scalar Gumbel stochasticity, temperature, quantizer jitter,
  coverage loss, shared initialization, batches, and update count; and
- disables the pointer-only entropy bonus in the checked-in smoke.

| Split | `N` | `K` | Pointer | Scalar | FPS | Target-oracle beam |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 16 | 4 | **0.47316** | 0.54898 | 0.42824 | 0.46080 |
| Validation | 16 | 8 | **0.32479** | 0.40097 | 0.30643 | 0.31973 |
| Validation | 32 | 4 | **0.49266** | 0.58916 | 0.43301 | 0.49178 |
| Validation | 32 | 8 | **0.40633** | 0.50083 | 0.29849 | 0.39903 |
| OOD | 16 | 4 | **0.46830** | 0.58080 | 0.39738 | 0.46064 |
| OOD | 16 | 8 | **0.36275** | 0.40978 | 0.32819 | 0.34124 |
| OOD | 32 | 4 | **0.47789** | 0.60316 | 0.41037 | 0.47417 |
| OOD | 32 | 8 | **0.41977** | 0.53763 | 0.32744 | 0.41720 |

The pointer beats scalar ranking at all eight corrected points, supporting
selection-conditioned allocation. FPS still wins all eight. Evaluation entropy
is `0.9984-0.9991`, so the pointer logits remain almost uniform after one smoke
epoch. Target-scored beam improves all eight by `0.18-5.93%` using roughly
`7.3-8.0` decoder evaluations per cloud, but it is an explicitly non-deployable
oracle because it scores against the complete evaluation target.

```bash
.venv-train/bin/python -m pointconstellation.pointer_experiment \
  --config configs/experiment_012_pointer_smoke.json --device cpu
```

## Decision after fan-out

The evidence currently ranks the mechanisms as follows:

1. Scale the legal Experiment 005 recurrent competitive refiner. Its
   no-gradient arm improves the primary point by 5.07%, and input-only decoder
   feedback raises the improvement to 22.94%.
2. Fold H13's sequential selection state into a strict-subset refiner or train
   it substantially longer; it consistently beats scalar ranking but has not
   learned FPS-level coverage.
3. Retain balanced responsibilities as an ablation inside that scaled run,
   with objective-specific normalization/tuning.
4. Keep homotopy, decoder populations, and gradient-free search as second-line
   controls until stronger evidence warrants their extra complexity.

Before a compression claim, run multiple seeds, analytic-surface metrics,
fresh resampling, perturbation tests, target-rate-exposure controls, and actual
rate/runtime accounting.
