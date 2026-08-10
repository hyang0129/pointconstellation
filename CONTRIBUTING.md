# Contributing

Point Constellation is an early research repository. Contributions should make
the core hypothesis easier to falsify, reproduce, or compare fairly.

## Set up the project

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
ruff check .
```

## Research contributions

- State the hypothesis and baseline before presenting a result.
- Report distortion against actual coded bits, not only point-count or tensor
  size.
- Count all per-cloud metadata needed by the decoder.
- Report decoder model size and explain whether it is shared or amortized.
- Fix random seeds and record dataset splits and preprocessing.
- Preserve the strict coordinate-only bottleneck for results labeled
  "geometry-only." Variants with features are welcome but must be labeled.
- Include a small test or reproduction command with code changes.

Please open an issue before adding a large dependency, dataset, or training
framework. Do not commit downloaded datasets or experiment artifacts.
