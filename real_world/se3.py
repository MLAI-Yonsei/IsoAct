"""Train-only SE(3) embedded tangent FLD and sequential pose actions."""
from __future__ import annotations
import hashlib
import json
from typing import Any
import numpy as np
from scipy import linalg
from scipy.spatial.transform import Rotation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

RIDGE_LAMBDA = 1e-6
RELATIVE_EIGEN_TOLERANCE = 1e-10
REPEATED_EIGEN_TOLERANCE = 1e-9
LOG_BRANCH_MARGIN = 1e-6
ROUNDTRIP_TOLERANCE = 2e-10
PREIMAGE_TOLERANCE = 2e-12
SO3_ORTHOGONALITY_TOLERANCE = 1e-8
SO3_DETERMINANT_TOLERANCE = 1e-8
ROTATION_DEGENERACY_TOLERANCE = 1e-10
REQUIRED_DISC_RANK = 6
_THRESHOLDS = {"basis_orthonormality": 2e-14, "basis_projector": 2e-13}

class TangentFLDError(RuntimeError):
    """Fail-closed error carrying a stable machine-readable reason code."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(reason if not detail else f"{reason}: {detail}")


def _f64(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if shape is not None and out.shape != shape:
        raise ValueError(f"expected shape {shape}, got {out.shape}")
    if not np.isfinite(out).all():
        raise TangentFLDError("NON_FINITE_POSE")
    return out


def _array_hash(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    payload = json.dumps(
        {"dtype": "float64", "shape": list(arr.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload + b"\0" + arr.tobytes()).hexdigest()


def skew(omega: np.ndarray) -> np.ndarray:
    x, y, z = _f64(omega, (3,))
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def project_to_so3(matrix: np.ndarray) -> np.ndarray:
    """The audited SVD-polar SO(3) projection used by choose_mu_pose."""
    u, _, vt = np.linalg.svd(_f64(matrix, (3, 3)))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u = u.copy()
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def validate_rotation(rotation: np.ndarray) -> dict[str, float]:
    """Reject an original/non-reference rotation instead of repairing it."""
    rotation = _f64(rotation, (3, 3))
    orthogonality_residual = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    determinant_residual = abs(float(np.linalg.det(rotation)) - 1.0)
    if orthogonality_residual > SO3_ORTHOGONALITY_TOLERANCE:
        raise TangentFLDError(
            "SO3_ORTHOGONALITY_FAILURE", f"residual={orthogonality_residual:.17g}"
        )
    if determinant_residual > SO3_DETERMINANT_TOLERANCE:
        raise TangentFLDError(
            "SO3_DETERMINANT_FAILURE", f"residual={determinant_residual:.17g}"
        )
    return {
        "orthogonality_residual": orthogonality_residual,
        "determinant_residual": determinant_residual,
    }


def so3_exp(omega: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(_f64(omega, (3,))).as_matrix().astype(np.float64)


def so3_log(rotation: np.ndarray) -> np.ndarray:
    rotation = _f64(rotation, (3, 3))
    validate_rotation(rotation)
    return Rotation.from_matrix(rotation).as_rotvec().astype(np.float64)


def left_jacobian_so3(omega: np.ndarray) -> np.ndarray:
    omega = _f64(omega, (3,))
    theta = float(np.linalg.norm(omega))
    w_hat = skew(omega)
    if theta < 1.0e-8:
        return np.eye(3) + 0.5 * w_hat + (1.0 / 6.0) * (w_hat @ w_hat)
    a = (1.0 - np.cos(theta)) / theta**2
    b = (theta - np.sin(theta)) / theta**3
    return np.eye(3) + a * w_hat + b * (w_hat @ w_hat)


def left_jacobian_so3_inverse(omega: np.ndarray) -> np.ndarray:
    omega = _f64(omega, (3,))
    theta = float(np.linalg.norm(omega))
    w_hat = skew(omega)
    if theta < 1.0e-8:
        return np.eye(3) - 0.5 * w_hat + (1.0 / 12.0) * (w_hat @ w_hat)
    # Half-angle form avoids the cancellation in (1+cos(theta))/sin(theta).
    coefficient = (1.0 - 0.5 * theta / np.tan(0.5 * theta)) / theta**2
    return np.eye(3) - 0.5 * w_hat + coefficient * (w_hat @ w_hat)


def se3_exp(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xi = _f64(xi, (6,))
    omega, velocity = xi[:3], xi[3:]
    return so3_exp(omega), left_jacobian_so3(omega) @ velocity


def se3_log(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    omega = so3_log(rotation)
    velocity = left_jacobian_so3_inverse(omega) @ _f64(translation, (3,))
    return np.concatenate((omega, velocity))


def compose(
    rotation_a: np.ndarray,
    translation_a: np.ndarray,
    rotation_b: np.ndarray,
    translation_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_a = _f64(rotation_a, (3, 3))
    translation_a = _f64(translation_a, (3,))
    rotation_b = _f64(rotation_b, (3, 3))
    translation_b = _f64(translation_b, (3,))
    return rotation_a @ rotation_b, rotation_a @ translation_b + translation_a


def inverse(rotation: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation_inverse = _f64(rotation, (3, 3)).T
    return rotation_inverse, -rotation_inverse @ _f64(translation, (3,))


def right_relative_coordinates(
    reference_rotation: np.ndarray,
    reference_translation: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    inv_rotation, inv_translation = inverse(reference_rotation, reference_translation)
    relative_rotation, relative_translation = compose(
        inv_rotation, inv_translation, rotation, translation
    )
    return se3_log(relative_rotation, relative_translation)


def from_right_relative_coordinates(
    reference_rotation: np.ndarray,
    reference_translation: np.ndarray,
    xi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    relative_rotation, relative_translation = se3_exp(xi)
    return compose(
        reference_rotation,
        reference_translation,
        relative_rotation,
        relative_translation,
    )


def choose_mu_pose(
    train_rotations: np.ndarray, train_translations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen audited routine: SVD-polar mean rotation and arithmetic mean t."""
    rotations = _f64(train_rotations)
    translations = _f64(train_translations)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("train_rotations must have shape (n, 3, 3)")
    if translations.shape != (len(rotations), 3) or len(rotations) == 0:
        raise ValueError("train_translations must have matching nonempty shape (n, 3)")
    return project_to_so3(rotations.mean(axis=0)), translations.mean(axis=0)


def pack_rt(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Row-major vec(R), followed by native-unit t."""
    return np.concatenate((_f64(rotation, (3, 3)).reshape(9), _f64(translation, (3,))))


def analytic_j_mu(reference_rotation: np.ndarray) -> np.ndarray:
    rotation = _f64(reference_rotation, (3, 3))
    jacobian: np.ndarray = np.zeros((12, 6), dtype=np.float64)
    for column in range(3):
        basis: np.ndarray = np.zeros(3, dtype=np.float64)
        basis[column] = 1.0
        jacobian[:9, column] = (rotation @ skew(basis)).reshape(9)
        jacobian[9:, column + 3] = rotation @ basis
    return jacobian


def deterministic_embedded_basis(jacobian: np.ndarray) -> np.ndarray:
    """Fixed body-coordinate order/sign basis for im(J_mu)."""
    jacobian = _f64(jacobian, (12, 6))
    if np.linalg.matrix_rank(jacobian) != 6:
        raise TangentFLDError("NON_USABLE_JACOBIAN_RANK")
    basis = np.empty_like(jacobian)
    accepted: list[np.ndarray] = []
    for index in range(6):
        vector = jacobian[:, index].copy()
        for prior in accepted:
            vector -= prior * float(prior @ vector)
        norm = float(np.linalg.norm(vector))
        if norm <= np.finfo(np.float64).eps:
            raise TangentFLDError("NON_USABLE_JACOBIAN_RANK")
        vector /= norm
        # Sign is tied to the corresponding ordered analytic J column.
        if float(vector @ jacobian[:, index]) < 0.0:
            vector *= -1.0
        accepted.append(vector)
        basis[:, index] = vector
    if not np.allclose(
        basis.T @ basis, np.eye(6), atol=_THRESHOLDS["basis_orthonormality"], rtol=0.0
    ):
        raise TangentFLDError("EMBEDDED_BASIS_ORTHONORMALITY_FAILURE")
    if not np.allclose(
        basis @ basis.T @ jacobian, jacobian, atol=_THRESHOLDS["basis_projector"], rtol=0.0
    ):
        raise TangentFLDError("EMBEDDED_BASIS_PROJECTOR_FAILURE")
    return basis


def embedded_coordinates(xi: np.ndarray, jacobian: np.ndarray, basis: np.ndarray) -> np.ndarray:
    coordinates = _f64(xi)
    if coordinates.ndim == 1:
        coordinates = coordinates[None, :]
    if coordinates.ndim != 2 or coordinates.shape[1] != 6:
        raise ValueError("xi must have shape (n, 6) or (6,)")
    return (coordinates @ _f64(jacobian, (12, 6)).T) @ _f64(basis, (12, 6))


def _canonical_basis_from_projector(projector: np.ndarray, rank: int) -> np.ndarray:
    """Canonical range basis from e_i anchors, MGS, QR, and pivot signs."""
    projector = _f64(projector)
    dimension = projector.shape[0]
    accepted: list[np.ndarray] = []
    threshold = 64.0 * np.finfo(np.float64).eps
    for anchor_index in range(dimension):
        vector = projector[:, anchor_index].copy()
        for prior in accepted:
            vector -= prior * float(prior @ vector)
        norm = float(np.linalg.norm(vector))
        if norm > threshold:
            vector /= norm
            accepted.append(vector)
        if len(accepted) == rank:
            break
    if len(accepted) != rank:
        raise TangentFLDError("EIGENSPACE_CANONICALIZATION_FAILURE")
    q, _ = np.linalg.qr(np.column_stack(accepted), mode="reduced")
    for column in range(rank):
        pivot = int(np.argmax(np.abs(q[:, column])))
        if q[pivot, column] < 0.0:
            q[:, column] *= -1.0
    return q


def canonicalize_repeated_eigenspaces(
    eigenvalues_descending: np.ndarray,
    eigenvectors_descending: np.ndarray,
    relative_tolerance: float = REPEATED_EIGEN_TOLERANCE,
) -> np.ndarray:
    """Discard arbitrary solver bases and reconstruct each tied eigenspace."""
    values = _f64(eigenvalues_descending)
    vectors = _f64(eigenvectors_descending)
    if vectors.shape != (len(values), len(values)):
        raise ValueError("eigenvectors must be a square column-basis matrix")
    output = np.empty_like(vectors)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values):
            scale = max(1.0, abs(float(values[start])), abs(float(values[stop])))
            if abs(float(values[start] - values[stop])) > relative_tolerance * scale:
                break
            stop += 1
        block = vectors[:, start:stop]
        projector = block @ block.T
        output[:, start:stop] = _canonical_basis_from_projector(projector, stop - start)
        start = stop
    return output


def fit_regularized_multiclass_fld(
    z_train: np.ndarray,
    labels_train: np.ndarray,
    *,
    ridge_lambda: float = RIDGE_LAMBDA,
    relative_eigen_tolerance: float = RELATIVE_EIGEN_TOLERANCE,
) -> dict[str, Any]:
    """Fit the frozen FLD contract to train coordinates only."""
    z = _f64(z_train)
    labels = np.asarray(labels_train)
    if z.ndim != 2 or z.shape[1] != 6 or len(z) != len(labels) or len(z) == 0:
        raise ValueError("z_train/labels_train must be matching nonempty train arrays")
    if not np.isfinite(ridge_lambda) or ridge_lambda != RIDGE_LAMBDA:
        raise TangentFLDError("RIDGE_CONTRACT_MISMATCH")
    classes = np.unique(labels)
    global_mean = z.mean(axis=0)
    within: np.ndarray = np.zeros((6, 6), dtype=np.float64)
    between: np.ndarray = np.zeros((6, 6), dtype=np.float64)
    for label in classes:
        samples = z[labels == label]
        class_mean = samples.mean(axis=0)
        centered = samples - class_mean
        within += centered.T @ centered
        delta = class_mean - global_mean
        between += len(samples) * np.outer(delta, delta)
    # Both scatter matrices use the explicit population normalization 1/n.
    within /= len(z)
    between /= len(z)
    regularized_within = within + ridge_lambda * np.eye(6)

    # Use SciPy's symmetric generalized eigensolver explicitly.  Mapping its
    # Sw-orthonormal vectors through L.T exposes the corresponding Euclidean
    # whitened eigenspace, whose projector is invariant to arbitrary solver
    # rotations in repeated eigenvalue blocks.
    chol = linalg.cholesky(regularized_within, lower=True, check_finite=True)
    eigenvalues, generalized_vectors = linalg.eigh(
        between, regularized_within, check_finite=True
    )
    order = np.argsort(eigenvalues, kind="stable")[::-1]
    eigenvalues = eigenvalues[order]
    whitened_vectors = chol.T @ generalized_vectors[:, order]
    whitened_vectors = canonicalize_repeated_eigenspaces(eigenvalues, whitened_vectors)
    coefficient_axes = linalg.solve_triangular(
        chol.T, whitened_vectors, lower=False, check_finite=True
    )
    coefficient_axes /= np.linalg.norm(coefficient_axes, axis=0, keepdims=True)
    for column in range(6):
        pivot = int(np.argmax(np.abs(coefficient_axes[:, column])))
        if coefficient_axes[pivot, column] < 0.0:
            coefficient_axes[:, column] *= -1.0

    scale = max(1.0, abs(float(eigenvalues[0])))
    discriminant_rank = int(np.count_nonzero(eigenvalues > relative_eigen_tolerance * scale))
    if discriminant_rank < REQUIRED_DISC_RANK:
        raise TangentFLDError(
            "NON_USABLE_DISC_RANK", f"r_disc={discriminant_rank}, required=6"
        )
    return {
        "axes": coefficient_axes[:, :REQUIRED_DISC_RANK],
        "eigenvalues": eigenvalues,
        "r_disc": discriminant_rank,
        "between_scatter": between,
        "within_scatter": within,
        "regularized_within_scatter": regularized_within,
        "scatter_normalization": "population_1_over_n_train",
        "solver": "scipy.linalg.eigh_symmetric_generalized",
    }


def lift_axes_and_preimages(
    coefficient_axes: np.ndarray,
    basis: np.ndarray,
    jacobian: np.ndarray,
) -> dict[str, Any]:
    axes = _f64(coefficient_axes, (6, 6))
    basis = _f64(basis, (12, 6))
    jacobian = _f64(jacobian, (12, 6))
    lifted = basis @ axes
    lifted /= np.linalg.norm(lifted, axis=0, keepdims=True)
    pseudoinverse = np.linalg.pinv(jacobian)
    preimages = pseudoinverse @ lifted
    residuals = np.linalg.norm(jacobian @ preimages - lifted, axis=0)
    if np.any(residuals > PREIMAGE_TOLERANCE):
        raise TangentFLDError("BODY_TWIST_PREIMAGE_FAILURE", str(residuals.tolist()))
    rotation_directions: np.ndarray = np.zeros((3, 6), dtype=np.float64)
    rotation_reason_codes: list[str] = []
    for column in range(6):
        omega = preimages[:3, column]
        norm = float(np.linalg.norm(omega))
        if norm <= ROTATION_DEGENERACY_TOLERANCE:
            rotation_reason_codes.append("ROTATION_DEGENERATE_TRANSLATION_ONLY")
        else:
            rotation_directions[:, column] = omega / norm
            rotation_reason_codes.append("OK")
    return {
        "lifted_axes": lifted,
        "body_twist_preimages": preimages,
        "rotation_directions": rotation_directions,
        "rotation_reason_codes": tuple(rotation_reason_codes),
        "preimage_residuals": residuals,
        "subspace_hash": _array_hash(lifted @ lifted.T),
        "axes_hash": _array_hash(lifted),
    }


def preflight_pose_splits(
    train: tuple[np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray] | None = None,
    test: tuple[np.ndarray, np.ndarray] | None = None,
    *,
    branch_margin: float = LOG_BRANCH_MARGIN,
    roundtrip_tolerance: float = ROUNDTRIP_TOLERANCE,
) -> dict[str, Any]:
    """Validate every supplied split, fitting the reference from train only."""
    reference_rotation, reference_translation = choose_mu_pose(*train)
    split_inputs = {"train": train, "validation": validation, "test": test}
    coordinates: dict[str, np.ndarray] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for name, split in split_inputs.items():
        if split is None:
            continue
        rotations = _f64(split[0])
        translations = _f64(split[1])
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
            raise ValueError(f"{name} rotations must have shape (n,3,3)")
        if translations.shape != (len(rotations), 3):
            raise ValueError(f"{name} translations must have shape (n,3)")
        xis = []
        margins = []
        residuals = []
        for rotation, translation in zip(rotations, translations, strict=True):
            validate_rotation(rotation)
            xi = right_relative_coordinates(
                reference_rotation, reference_translation, rotation, translation
            )
            margin = float(np.pi - np.linalg.norm(xi[:3]))
            if margin <= branch_margin:
                raise TangentFLDError("LOG_BRANCH_MARGIN_FAILURE", f"split={name}")
            reconstructed_rotation, reconstructed_translation = from_right_relative_coordinates(
                reference_rotation, reference_translation, xi
            )
            residual = max(
                float(np.linalg.norm(reconstructed_rotation - rotation)),
                float(np.linalg.norm(reconstructed_translation - translation)),
            )
            if residual > roundtrip_tolerance:
                raise TangentFLDError("LOG_EXP_ROUNDTRIP_FAILURE", f"split={name}")
            xis.append(xi)
            margins.append(margin)
            residuals.append(residual)
        coordinates[name] = np.asarray(xis, dtype=np.float64).reshape(-1, 6)
        receipts[name] = {
            "count": len(rotations),
            "minimum_branch_margin": min(margins, default=float("inf")),
            "maximum_roundtrip_residual": max(residuals, default=0.0),
        }
    return {
        "reference_rotation": reference_rotation,
        "reference_translation": reference_translation,
        "coordinates": coordinates,
        "receipts": receipts,
    }


def project_to_SO3(M):
    """Project 3x3 matrix to nearest SO(3) via SVD polar decomposition."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U2 = U.copy()
        U2[:, -1] *= -1
        R = U2 @ Vt
    return R


def project_all_SO3(R_arr):
    """Project N x 3 x 3 array to SO(3)."""
    out = np.zeros_like(R_arr)
    for i in range(len(R_arr)):
        out[i] = project_to_SO3(R_arr[i])
    return out


def pack_Rt(R_arr, t_arr):
    """N x 3 x 3 + N x 3 -> N x 12 (R.flatten() then t)."""
    return np.concatenate([R_arr.reshape(-1, 9), t_arr], axis=1)


def unpack_Rt(X):
    """N x 12 -> R (N, 3, 3), t (N, 3)."""
    return X[:, :9].reshape(-1, 3, 3), X[:, 9:12].copy()


def isorot_step(X, axis_unit, neutral, alpha):
    """Single-axis IsoRot step: R → R · R_axis(α·Δφ), t preserved.

    Δφ_i = projection of Log(R_i^T R_n) onto axis_unit (signed scalar).
    """
    R_arr, t_arr = unpack_Rt(X)
    R_n = project_to_SO3(neutral[:9].reshape(3, 3))

    Delta = R_arr.transpose(0, 2, 1) @ R_n[None, :, :]
    rotvecs = Rotation.from_matrix(Delta).as_rotvec()  # (N,3)
    delta = rotvecs @ axis_unit  # (N,)

    rotvec_corr = (alpha * delta)[:, None] * axis_unit[None, :]  # (N,3)
    R_corr = Rotation.from_rotvec(rotvec_corr).as_matrix()  # (N,3,3)
    R_new = R_arr @ R_corr
    R_new = project_all_SO3(R_new)
    return pack_Rt(R_new, t_arr)


def trans_isorot_step(X, u, neutral, alpha):
    """Apply one sample-dependent pose correction."""
    R_arr, t_arr = unpack_Rt(X)
    u_t = u[9:12]
    u_t_norm = np.linalg.norm(u_t)
    if u_t_norm < 1e-8:
        return X.copy()  # axis has no t component
    u_t_unit = u_t / u_t_norm
    cur = X @ u
    s_n = float(u @ neutral)
    shift = alpha * (s_n - cur)
    t_new = t_arr + shift[:, None] * u_t_unit[None, :]
    return pack_Rt(R_arr, t_new)


def se3_isorot_step(X, u, axis_unit_R, neutral, alpha):
    """Apply one sample-dependent pose correction."""
    R_arr, t_arr = unpack_Rt(X)
    R_n = project_to_SO3(neutral[:9].reshape(3, 3))

    # Rotation correction.
    Delta = R_arr.transpose(0, 2, 1) @ R_n[None, :, :]
    rotvecs = Rotation.from_matrix(Delta).as_rotvec()
    delta_R = rotvecs @ axis_unit_R
    rotvec_corr = (alpha * delta_R)[:, None] * axis_unit_R[None, :]
    R_corr = Rotation.from_rotvec(rotvec_corr).as_matrix()
    R_new = R_arr @ R_corr
    R_new = project_all_SO3(R_new)

    # Translation correction.
    u_t = u[9:12]
    u_t_norm = np.linalg.norm(u_t)
    if u_t_norm > 1e-8:
        u_t_unit = u_t / u_t_norm
        cur = X @ u
        s_n = float(u @ neutral)
        shift = alpha * (s_n - cur)
        t_new = t_arr + shift[:, None] * u_t_unit[None, :]
    else:
        t_new = t_arr.copy()

    return pack_Rt(R_new, t_new)


def fit(X_train, y_train):
    """Fit the six-axis bridge and neutral using training poses and labels only.

    Poses are float64 row-major rotation entries followed by native-unit
    translation. Six nonzero discriminants are required (at least 7 classes).
    """
    X = _f64(X_train)
    y = np.asarray(y_train)
    if X.ndim != 2 or X.shape[1] != 12 or y.shape != (len(X),) or not len(X):
        raise ValueError("Expected nonempty X_train (N,12) and y_train (N,)")
    preflight = preflight_pose_splits((X[:, :9].reshape(-1, 3, 3), X[:, 9:]))
    R = preflight['reference_rotation']
    t = preflight['reference_translation']
    jacobian = analytic_j_mu(R)
    basis = deterministic_embedded_basis(jacobian)
    coordinates = embedded_coordinates(preflight['coordinates']['train'], jacobian, basis)
    fld = fit_regularized_multiclass_fld(coordinates, y)
    bridge = lift_axes_and_preimages(fld['axes'], basis, jacobian)
    encoded = LabelEncoder().fit_transform(y)
    scaled = StandardScaler().fit_transform(X)
    probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0, n_jobs=1)
    probe.fit(scaled, encoded)
    low = probe.predict_proba(scaled).max(axis=1) < 0.7
    neutral = X[low].mean(axis=0) if int(low.sum()) >= 10 else X.mean(axis=0)
    neutral = np.concatenate([project_to_SO3(neutral[:9].reshape(3, 3)).reshape(-1), neutral[9:]])
    return dict(backend='se3', reference=pack_rt(R, t), neutral=neutral,
                axes=bridge['lifted_axes'].T, body_twists=bridge['body_twist_preimages'].T,
                jacobian=jacobian, basis=basis, eigenvalues=fld['eigenvalues'],
                neutral_count=int(low.sum()))


def transform(X, state, rank=5, alpha=2.0, action='translation', schedule=None):
    """Apply ordered axes with rotation, translation, joint, or skip actions.

    An optional sequence of action names overrides the common action. It must
    contain one entry per selected axis. Rotation directions come from each
    lifted axis's body-twist preimage, never the first three packed components.
    """
    Y = _f64(X).copy()
    if Y.ndim != 2 or Y.shape[1] != 12:
        raise ValueError('Expected poses with shape (N,12)')
    if int(rank) != rank or not 1 <= rank <= 6 or not np.isfinite(alpha):
        raise ValueError('rank must be 1..6 and alpha must be finite')
    rank = int(rank)
    axes = _f64(state['axes'], (6, 12))
    twists = _f64(state['body_twists'], (6, 6))
    neutral = _f64(state['neutral'], (12,))
    actions = [action] * rank if schedule is None else list(schedule)
    if len(actions) != rank or any(a not in ('rotation','translation','joint','skip') for a in actions):
        raise ValueError('Schedule must have rank entries: rotation, translation, joint, or skip')
    for i, selected in enumerate(actions):
        if selected == 'skip':
            continue
        if selected == 'translation':
            Y = trans_isorot_step(Y, axes[i], neutral, float(alpha))
            continue
        omega = twists[i, :3]
        norm = np.linalg.norm(omega)
        if norm <= ROTATION_DEGENERACY_TOLERANCE:
            raise ValueError(f'Rotation preimage is degenerate at axis {i + 1}')
        omega = omega / norm
        if selected == 'rotation':
            Y = isorot_step(Y, omega, neutral, float(alpha))
        else:
            Y = se3_isorot_step(Y, axes[i], omega, neutral, float(alpha))
    if not np.isfinite(Y).all():
        raise ValueError('Nonfinite transformed poses')
    return Y
