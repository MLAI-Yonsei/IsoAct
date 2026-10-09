# Controlled Synthetic Benchmark

This directory contains the code for the controlled synthetic/toy experiments on:

- `sphere/`
- `hyperbolic/`
- `se3/`

The `build_*_unified_metrics.py` scripts regenerate metric tables and figures. They depend on minimal legacy source code under `legacy/` to generate the original baselines and CMF comparison caches.

## Run

```bash
bash run_all.sh
```

`run_all.sh` first regenerates CMF caches locally via `legacy/cmf_comparison/run_*_cmf.py`, then runs the three unified-metric builders. Generated CSV/JSON/MD/PNG/NPZ outputs are written next to each builder and are intentionally not pre-shipped in this release.
