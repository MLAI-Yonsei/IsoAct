"""Fit IsoAct-FLD from training arrays or reuse a saved state."""
import argparse
import json
from pathlib import Path
import numpy as np
import face
import se3


def save_state(path, state, options):
    arrays = {key: np.asarray(value) for key, value in state.items()}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError('State must contain numeric or string arrays only')
    np.savez_compressed(path, **arrays, transform_options=np.asarray(json.dumps(options)))


def load_state(path):
    with np.load(path, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files if key != 'transform_options'}
        options = json.loads(str(archive['transform_options'].item()))
    for key, value in state.items():
        if value.ndim == 0:
            state[key] = value.item()
    return state, options


def apply(X, state, options):
    if state['backend'] == 'se3':
        return se3.transform(X, state, **options)
    return face.transform(X, state, **options)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit-transform', help='Fit on X_train/y_train and transform supplied arrays')
    fit.add_argument('--backend', choices=['clip', 'meru', 'se3'], required=True)
    fit.add_argument('--input', type=Path, required=True, help='NPZ: X_train, y_train; optional X, X_utility')
    fit.add_argument('--output', type=Path, required=True, help='Output NPZ of transformed arrays')
    fit.add_argument('--state-out', type=Path, required=True, help='Fitted state and action options, NPZ')
    fit.add_argument('--rank', type=int, help='Action prefix: CLIP 7, MERU 8, SE3 5')
    fit.add_argument('--max-rank', type=int, default=8, help='Face fit rank before action prefix (default: 8)')
    fit.add_argument('--ridge', type=float, help='Face ridge: CLIP .1, MERU .01; SE3 fixed 1e-6')
    fit.add_argument('--alpha', type=float, help='CLIP correction (default 1), SE3 correction (default 2)')
    fit.add_argument('--angular', type=float, default=.75, help='MERU effective angular gain')
    fit.add_argument('--translation', type=float, default=1., help='MERU effective translation gain')
    fit.add_argument('--curvature', type=float, default=.10000000149011612, help='MERU positive curvature magnitude')
    fit.add_argument('--order', choices=['forward', 'reverse'], default='forward', help='Face axis order')
    fit.add_argument('--passes', type=int, default=1, help='Face frozen-axis passes')
    fit.add_argument('--composition', choices=['angular_then_translation', 'translation_then_angular'], default='angular_then_translation', help='MERU composition within each axis')
    fit.add_argument('--action', choices=['rotation', 'translation', 'joint'], default='translation', help='SE3 common action')
    fit.add_argument('--schedule', help='SE3 comma-separated per-axis rotation,translation,joint,skip; length=rank')
    reuse = commands.add_parser('transform', help='Apply the saved fit and action options without labels')
    reuse.add_argument('--state', type=Path, required=True)
    reuse.add_argument('--input', type=Path, required=True, help='NPZ containing X')
    reuse.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    # Reject output aliases before opening inputs or writing fitted state.
    inputs = [args.input] + ([args.state] if args.command == 'transform' else [])
    outputs = [args.output] + ([args.state_out] if args.command == 'fit-transform' else [])
    resolved = [p.resolve() for p in inputs + outputs]
    if len(set(resolved)) != len(resolved):
        parser.error('Input, output and state paths must be distinct')
    if any(p.suffix != '.npz' for p in inputs + outputs):
        parser.error('Input, output and state paths must end in .npz')
    with np.load(args.input, allow_pickle=False) as archive:
        if args.command == 'transform':
            state, options = load_state(args.state)
            result = {'X': apply(archive['X'], state, options)}
        else:
            X, y = archive['X_train'], archive['y_train']
            if args.backend == 'se3':
                if args.ridge is not None and args.ridge != 1e-6:
                    parser.error('SE3 ridge is fixed at 1e-6')
                state = se3.fit(X, y)
                options = dict(rank=5 if args.rank is None else args.rank,
                               alpha=2. if args.alpha is None else args.alpha, action=args.action,
                               schedule=args.schedule.split(',') if args.schedule else None)
            else:
                state = face.fit(X, y, backend=args.backend, max_rank=args.max_rank,
                                 ridge=args.ridge, curvature=args.curvature)
                options = dict(rank=args.rank, alpha=1. if args.alpha is None else args.alpha,
                               angular=args.angular, translation=args.translation, order=args.order,
                               passes=args.passes, composition=args.composition)
            result = {'X_train': apply(X, state, options)}
            for key in ('X', 'X_utility'):
                if key in archive:
                    result[key] = apply(archive[key], state, options)
            save_state(args.state_out, state, options)
    np.savez_compressed(args.output, **result)
    print(f"Saved {', '.join(result)} to {args.output}")


if __name__ == '__main__':
    main()
