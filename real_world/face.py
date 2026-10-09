"""Train-only Face IsoAct-FLD fitting and manifold-native transforms.

Inputs are precomputed CLIP unit embeddings or MERU spatial coordinates.  The
module intentionally contains no encoder, dataset, evaluation, or cache code.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_CURVATURE = 0.10000000149011612
_DEFAULT_RIDGE = {"clip": 0.1, "meru": 0.01}
_DEFAULT_RANK = {"clip": 7, "meru": 8}


def _validate_training(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(X)
    labels = np.asarray(y)
    if values.ndim != 2 or labels.ndim != 1 or len(values) != len(labels):
        raise ValueError("X_train must be N x D and y_train must have length N")
    if values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("X_train must have at least two finite columns")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or np.any(counts < 2):
        raise ValueError("FLD requires at least two classes with two rows each")
    return values, labels


def _orthonormal_rows(rows: np.ndarray, rank: int) -> np.ndarray:
    matrix = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    _, singular, right = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * (
        float(singular[0]) if singular.size else 0.0
    )
    numerical_rank = int(np.sum(singular > tolerance))
    result = right[: min(rank, numerical_rank)].copy()
    for axis in result:
        pivot = int(np.argmax(np.abs(axis)))
        if axis[pivot] < 0:
            axis *= -1.0
    return result


def _learn_fld(
    coordinates: np.ndarray, labels: np.ndarray, ridge: float, requested_rank: int
) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    classes, encoded = np.unique(labels, return_inverse=True)
    dimension = coordinates.shape[1]
    target_rank = min(int(requested_rank), len(classes) - 1, dimension)
    within = np.zeros((dimension, dimension), dtype=np.float64)
    between = np.zeros_like(within)
    global_mean = coordinates.mean(axis=0)
    means = []
    for index in range(len(classes)):
        group = coordinates[encoded == index]
        mean = group.mean(axis=0)
        means.append(mean)
        centered = group - mean
        within += centered.T @ centered
        delta = (mean - global_mean)[:, None]
        between += len(group) * (delta @ delta.T)
    within /= max(len(coordinates) - len(classes), 1)
    regularized = within + float(ridge) * np.eye(dimension)
    if len(classes) == 2:
        directions = np.linalg.solve(regularized, means[1] - means[0])
    else:
        between /= len(classes) - 1
        eigenvalues, eigenvectors = np.linalg.eigh(regularized)
        floor = max(float(eigenvalues.max()) * np.finfo(np.float64).eps, 1e-15)
        inv_sqrt = (
            eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, floor)))
        ) @ eigenvectors.T
        whitened = inv_sqrt @ between @ inv_sqrt
        values, vectors = np.linalg.eigh(0.5 * (whitened + whitened.T))
        order = np.argsort(values)[::-1]
        directions = (inv_sqrt @ vectors[:, order[:target_rank]]).T
    axes = _orthonormal_rows(directions, target_rank)
    if axes.shape[0] == 0 or not np.isfinite(axes).all():
        raise ValueError("FLD produced an empty or non-finite subspace")
    return axes


def _fit_neutral(X: np.ndarray, y: np.ndarray, *, normalize: bool) -> tuple[np.ndarray, int, bool]:
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"),
    )
    probe.fit(X, y)
    low_confidence = probe.predict_proba(X).max(axis=1) < 0.7
    count = int(low_confidence.sum())
    fallback = count < 10
    neutral = (X[low_confidence] if not fallback else X).mean(axis=0).astype(np.float32)
    if normalize:
        norm = float(np.linalg.norm(neutral))
        if norm <= 0.0:
            raise ValueError("CLIP neutral has zero norm")
        neutral = (neutral / norm).astype(np.float32)
    return neutral, count, fallback


def _sphere_coordinates(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("CLIP embeddings must be nonzero")
    values = values / norms
    reference = values.mean(axis=0)
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm <= 1e-12:
        raise ValueError("CLIP train mean is degenerate")
    reference /= reference_norm
    cosine = np.clip(values @ reference, -1.0 + 1e-7, 1.0 - 1e-7)
    angle = np.arccos(cosine)
    scale = angle / np.maximum(np.sin(angle), 1e-8)
    tangent = (values - cosine[:, None] * reference[None, :]) * scale[:, None]
    tangent -= (tangent @ reference)[:, None] * reference[None, :]
    return tangent, reference


def _clip_axes_at_reference(axes: np.ndarray, reference: np.ndarray) -> np.ndarray:
    projected = np.asarray(axes, dtype=np.float64)
    projected -= (projected @ reference)[:, None] * reference[None, :]
    q, _ = np.linalg.qr(projected.T)
    result = q[:, : projected.shape[0]].T
    for axis in result:
        pivot = int(np.argmax(np.abs(axis)))
        if axis[pivot] < 0:
            axis *= -1.0
    return result


def _lorentz_inner(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -(x[..., 0] * y[..., 0]) + np.sum(x[..., 1:] * y[..., 1:], axis=-1)


def _space_to_lorentz(X: np.ndarray, curvature: float) -> np.ndarray:
    time = np.sqrt(1.0 / curvature + np.sum(X * X, axis=-1, keepdims=True))
    return np.concatenate([time, X], axis=-1)


def _parallel_transport(
    source: np.ndarray, target: np.ndarray, tangent: np.ndarray, curvature: float
) -> np.ndarray:
    denominator = (1.0 / curvature) - float(_lorentz_inner(source, target))
    if not np.isfinite(denominator) or denominator <= 1e-14:
        raise ValueError("invalid Lorentz parallel-transport denominator")
    coefficient = _lorentz_inner(target, tangent) / denominator
    return tangent + coefficient[..., None] * (source + target)


def _meru_coordinates(X: np.ndarray, curvature: float) -> tuple[np.ndarray, np.ndarray]:
    full = _space_to_lorentz(np.asarray(X, dtype=np.float64), curvature)
    mean_full = full.mean(axis=0)
    reference = _space_to_lorentz(mean_full[None, 1:], curvature)[0]
    neg_dot = np.maximum(-_lorentz_inner(reference, full), 1.0 / curvature + 1e-15)
    scaled = np.maximum(curvature * neg_dot, 1.0 + 1e-15)
    distance = np.arccosh(scaled) / np.sqrt(curvature)
    direction = full - neg_dot[:, None] * reference[None, :]
    sinh_distance = np.sinh(np.sqrt(curvature) * distance)
    safe_sinh = np.where(sinh_distance < 1e-12, 1.0, sinh_distance)
    coefficient = np.sqrt(curvature) * distance / safe_sinh
    coefficient = np.where(np.abs(distance) < 1e-12, 1.0, coefficient)
    raw_log = coefficient[:, None] * direction
    tangent = raw_log + curvature * _lorentz_inner(raw_log, reference)[..., None] * reference
    origin = np.zeros(full.shape[1], dtype=np.float64)
    origin[0] = 1.0 / np.sqrt(curvature)
    transported = _parallel_transport(reference, origin, tangent, curvature)
    round_trip = _parallel_transport(origin, reference, transported, curvature)
    sample_n = min(128, len(tangent))
    before_gram = np.asarray(
        [[_lorentz_inner(a, b) for b in tangent[:sample_n]] for a in tangent[:sample_n]]
    )
    after_gram = np.asarray(
        [
            [_lorentz_inner(a, b) for b in transported[:sample_n]]
            for a in transported[:sample_n]
        ]
    )
    residual = max(
        float(np.max(np.abs(_lorentz_inner(reference, tangent)))),
        float(np.max(np.abs(_lorentz_inner(origin, transported)))),
        float(np.max(np.abs(transported[:, 0]))),
        float(np.max(np.abs(round_trip - tangent))),
        float(np.max(np.abs(after_gram - before_gram))),
    )
    if residual > 1e-8:
        raise RuntimeError(f"MERU Lorentz mapping residual exceeded 1e-8: {residual:.3e}")
    return transported[:, 1:], reference


def _canonicalize_meru_axes(
    axes: np.ndarray, X_train: np.ndarray, y_train: np.ndarray
) -> np.ndarray:
    result = np.asarray(axes, dtype=np.float64).copy()
    classes = np.array(sorted(np.unique(y_train).tolist()))
    for index, axis in enumerate(result):
        projection = X_train @ axis
        if len(classes) == 2:
            displacement = float(
                projection[y_train == classes[-1]].mean()
                - projection[y_train == classes[0]].mean()
            )
        else:
            overall = float(projection.mean())
            displacements = [float(projection[y_train == cls].mean() - overall) for cls in classes]
            displacement = displacements[int(np.argmax(np.abs(displacements)))]
        if displacement < 0:
            result[index] *= -1.0
    return result.astype(np.float32)


def fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    backend: str = "clip",
    max_rank: int = 8,
    ridge: float | None = None,
    curvature: float = DEFAULT_CURVATURE,
) -> dict[str, Any]:
    """Fit a train-only Face IsoAct-FLD state reusable on unlabeled inputs."""

    X, y = _validate_training(X_train, y_train)
    backend = str(backend).lower()
    if backend not in _DEFAULT_RIDGE:
        raise ValueError("backend must be 'clip' or 'meru'")
    if int(max_rank) < 1:
        raise ValueError("max_rank must be positive")
    ridge_value = _DEFAULT_RIDGE[backend] if ridge is None else float(ridge)
    if ridge_value <= 0 or not np.isfinite(ridge_value):
        raise ValueError("ridge must be positive and finite")
    if curvature <= 0 or not np.isfinite(curvature):
        raise ValueError("curvature must be positive and finite")

    if backend == "clip":
        coordinates, reference = _sphere_coordinates(X)
        rank_cap = min(int(max_rank), len(np.unique(y)) - 1, X.shape[1] - 1)
        axes = _learn_fld(coordinates, y, ridge_value, rank_cap)
        axes = _clip_axes_at_reference(axes, reference)
        neutral, low_count, fallback = _fit_neutral(X, y, normalize=True)
    else:
        coordinates, reference = _meru_coordinates(X, float(curvature))
        rank_cap = min(int(max_rank), len(np.unique(y)) - 1, X.shape[1])
        axes = _learn_fld(coordinates, y, ridge_value, rank_cap)
        axes = _canonicalize_meru_axes(axes, X, y)
        neutral, low_count, fallback = _fit_neutral(X, y, normalize=False)

    return {
        "backend": backend,
        "reference": reference,
        "neutral": neutral,
        "axes": axes,
        "ridge": ridge_value,
        "curvature": float(curvature),
        "max_rank_requested": int(max_rank),
        "fitted_rank": int(axes.shape[0]),
        "feature_dim": int(X.shape[1]),
        "class_count": int(len(np.unique(y))),
        "neutral_low_confidence_count": low_count,
        "neutral_used_train_mean_fallback": fallback,
    }


def _selected_axes(state: dict[str, Any], rank: int | None, order: str) -> np.ndarray:
    backend = str(state["backend"])
    axes = np.asarray(state["axes"])
    requested = _DEFAULT_RANK[backend] if rank is None else int(rank)
    if requested < 0:
        raise ValueError("rank must be nonnegative")
    if requested > len(axes):
        warnings.warn(
            f"requested rank {requested} exceeds fitted rank {len(axes)}; using {len(axes)}",
            RuntimeWarning,
            stacklevel=3,
        )
        requested = len(axes)
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be 'forward' or 'reverse'")
    chosen = axes[:requested]
    return chosen if order == "forward" else chosen[::-1]


def _clip_step(X: np.ndarray, axis: np.ndarray, neutral: np.ndarray, alpha: float) -> np.ndarray:
    Y = np.asarray(X, dtype=np.float32)
    u = np.asarray(axis, dtype=np.float32)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    projection = Y @ u
    radius = np.sqrt(np.maximum(1.0 - projection * projection, 0.0))
    valid = radius > 1e-7
    target = np.clip(
        (1.0 - float(alpha)) * projection + float(alpha) * float(neutral @ u),
        -1.0 + 1e-7,
        1.0 - 1e-7,
    )
    direction = np.zeros_like(Y, dtype=np.float32)
    direction[valid] = (
        Y[valid] - projection[valid, None] * u[None, :]
    ) / radius[valid, None]
    new_radius = np.sqrt(np.maximum(1.0 - target * target, 0.0)).astype(np.float32)
    result = Y.copy()
    result[valid] = (
        target[valid, None].astype(np.float32) * u[None, :]
        + new_radius[valid, None] * direction[valid]
    )
    norms = np.linalg.norm(result, axis=1)
    renormalize = np.abs(norms - 1.0) > 1e-5
    if np.any(renormalize):
        result[renormalize] /= np.maximum(norms[renormalize, None], 1e-12)
    return result


def _meru_angular_step(
    X: np.ndarray, axis: np.ndarray, neutral: np.ndarray, strength: float
) -> np.ndarray:
    Y = np.asarray(X, dtype=np.float32)
    u = np.asarray(axis).astype(np.float32)
    projection = Y @ u
    perpendicular = Y - projection[:, None] * u[None, :]
    perpendicular_norm = np.linalg.norm(perpendicular, axis=1)
    valid = perpendicular_norm > 1e-9
    direction = np.zeros_like(perpendicular)
    direction[valid] = perpendicular[valid] / perpendicular_norm[valid, None]
    target = (1.0 - float(strength)) * projection + float(strength) * float(neutral @ u)
    radius = np.sqrt(np.maximum(projection**2 + perpendicular_norm**2, 0.0))
    target = np.clip(target, -radius, radius)
    new_perpendicular = np.sqrt(np.maximum(radius**2 - target**2, 0.0))
    return (target[:, None] * u[None, :] + new_perpendicular[:, None] * direction).astype(
        np.float32
    )


def _translation_step(
    X: np.ndarray,
    axis: np.ndarray,
    neutral: np.ndarray,
    strength: float,
    curvature: float,
) -> np.ndarray:
    values = np.asarray(X, dtype=np.float64)
    u = np.asarray(axis, dtype=np.float64)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    projection = values @ u
    target = (1.0 - float(strength)) * projection + float(strength) * float(neutral @ u)
    time = np.sqrt(np.sum(values * values, axis=1) + 1.0 / curvature)
    positive = time + projection
    negative = time - projection
    discriminant = np.maximum(target * target + positive * negative, 0.0)
    root = np.sqrt(discriminant)
    tau = np.zeros_like(projection)
    solved = np.zeros_like(projection, dtype=bool)
    best = np.full_like(projection, np.inf)
    for sign in (1.0, -1.0):
        w = (target + sign * root) / np.maximum(positive, 1e-300)
        valid = np.isfinite(w) & (w > 1e-300)
        candidate = np.log(np.maximum(w, 1e-300))
        take = valid & (np.abs(candidate) < best)
        tau[take] = candidate[take]
        best[take] = np.abs(candidate[take])
        solved[take] = True
    tau = np.clip(np.where(solved, tau, 0.0), -8.0, 8.0)
    perpendicular = values - projection[:, None] * u[None, :]
    new_projection = np.sinh(tau) * time + np.cosh(tau) * projection
    return (perpendicular + new_projection[:, None] * u[None, :]).astype(np.float32)


def transform(
    X: np.ndarray,
    state: dict[str, Any],
    rank: int | None = None,
    alpha: float = 1.0,
    angular: float = 0.75,
    translation: float = 1.0,
    order: str = "forward",
    passes: int = 1,
    composition: str = "angular_then_translation",
) -> np.ndarray:
    """Apply a fitted state without using labels from the transformed split."""

    values = np.asarray(X)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("X must be a finite N x D matrix")
    backend = str(state.get("backend", ""))
    if backend not in _DEFAULT_RANK:
        raise ValueError("state backend must be 'clip' or 'meru'")
    if values.shape[1] != int(state["feature_dim"]):
        raise ValueError("X feature dimension does not match fitted state")
    if int(passes) < 1:
        raise ValueError("passes must be positive")
    axes = _selected_axes(state, rank, order)
    neutral = np.asarray(state["neutral"])
    result = values.astype(np.float32, copy=True)
    for _ in range(int(passes)):
        for axis in axes:
            if backend == "clip":
                result = _clip_step(result, axis, neutral, float(alpha))
            elif composition == "angular_then_translation":
                result = _meru_angular_step(result, axis, neutral, float(angular))
                result = _translation_step(
                    result, axis, neutral, float(translation), float(state["curvature"])
                )
            elif composition == "translation_then_angular":
                result = _translation_step(
                    result, axis, neutral, float(translation), float(state["curvature"])
                )
                result = _meru_angular_step(result, axis, neutral, float(angular))
            else:
                raise ValueError(
                    "composition must be 'angular_then_translation' or "
                    "'translation_then_angular'"
                )
    if not np.isfinite(result).all():
        raise RuntimeError("transform produced non-finite output")
    return result
