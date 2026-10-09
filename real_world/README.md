# IsoAct-FLD: real-world representations

IsoAct-FLD fits protected directions from training representations and applies
manifold-native edits to CLIP sphere embeddings, MERU Lorentz embeddings, and
SE(3) poses. Inputs are precomputed representations; datasets and encoders are
not included.

Use Python 3.10+ and install `pip install -r requirements.txt` in your environment.
Tested with NumPy 1.26.4, SciPy 1.15.3, and scikit-learn 1.7.2.

## Input and output

Prepare an NPZ with `X_train` (N,D) and `y_train` (N,) protected labels. Optional
`X` (M,D) and `X_utility` (U,D) are additional arrays to transform; neither requires
labels. Partition training and evaluation arrays before saving the file. Fit uses
only `X_train` and `y_train`. Labels must be integer or Unicode arrays, not object arrays.

CLIP inputs are unit sphere embeddings, normally float32 and D=512. MERU inputs
are float32 spatial coordinates, normally D=512: the time coordinate is reconstructed
as `sqrt(1/curvature + sum(spatial**2))`, with time-first Lorentz metric. Curvature
must match the supplied representation. SE3 inputs are float64 (N,12): nine row-major
rotation entries followed by three translation entries in their existing native
units. No rotation/translation rescaling is performed. SE3 training requires valid
rotations, relative rotation angles away from pi, and six nonzero discriminant
directions (at least seven protected classes). Face fitting requires at least
two classes with two training samples each.

The output NPZ contains transformed `X_train` and whichever optional arrays were
supplied. The state NPZ stores fitted reference, neutral, ordered axes, and action
options without data or labels. Reuse it with the `transform` subcommand, whose
input needs only `X`; see `python run.py transform --help`. No refitting occurs.

From this directory, replace the input filenames below with your prepared NPZs:

```sh
python run.py fit-transform --backend clip --input clip_age.npz --output clip_age_transformed.npz --state-out clip_age_state.npz
python run.py fit-transform --backend se3 --input poses.npz --output poses_transformed.npz --state-out poses_state.npz
```

## Settings

- CLIP defaults: fit rank 8, ridge .1; apply the first 7 axes with alpha 1,
  forward order, one pass. Face ranks are capped by class count and dimension.
- MERU is available through `--backend meru` or `face.fit(..., backend="meru")`.
  Defaults: rank 8, ridge .01, curvature .10000000149011612, angular gain .75,
  translation gain 1, forward order, one pass. `--angular` and `--translation`
  are independent effective gains. `--composition` sets their order **within each
  axis** (default: angular then translation). `--alpha` is for CLIP/SE3;
  MERU uses its two gains.
- SE3 fits six tangent FLD axes with fixed ridge 1e-6. Defaults: first 5 axes,
  alpha 2, translation-only action. Rotation-only and joint actions are also
  supported via `--action rotation` and `--action joint`. A comma-separated
  `--schedule` can assign rotation, translation, joint, or skip to each selected
  axis. The action choice does not change the reference or FLD fitting method.

`--rank`, `--alpha`, `--max-rank`, `--ridge`, `--order`, and `--passes` expose the
applicable settings above; run `python run.py fit-transform --help` for syntax.
Alpha is a correction gain and may exceed 1. Repeated Face passes reuse frozen
axes. `--max-rank` controls Face fitting; `--rank` selects the fitted axis prefix.

The Python API is `face.fit` / `face.transform` and `se3.fit` / `se3.transform`.
Run the tests with `python -m unittest discover -p 'test_*.py'`.
