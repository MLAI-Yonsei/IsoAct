import warnings
import unittest

import numpy as np

import face


def _training(backend: str, *, seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(9), 5)
    centers = rng.normal(size=(9, 12))
    values = centers[labels] + 0.35 * rng.normal(size=(len(labels), 12))
    if backend == "clip":
        values /= np.linalg.norm(values, axis=1, keepdims=True)
    else:
        values *= 0.2
    return values.astype(np.float32), labels


class FaceTest(unittest.TestCase):
    def test_clip_fit_maxrank_then_transform_uses_prefix_and_preserves_sphere(self):
        X, y = _training("clip")
        state = face.fit(X, y, backend="clip", max_rank=8)

        self.assertEqual(state["axes"].shape, (8, X.shape[1]))
        self.assertLess(np.max(np.abs(state["axes"] @ state["reference"])), 1e-10)
        self.assertLess(
            np.max(np.abs(state["axes"] @ state["axes"].T - np.eye(8))), 1e-10
        )

        actual = face.transform(X[:7], state, rank=7, alpha=1.0)
        expected = X[:7].copy()
        for axis in state["axes"][:7]:
            expected = face._clip_step(expected, axis, state["neutral"], 1.0)
        np.testing.assert_array_equal(actual, expected)
        self.assertLess(np.max(np.abs(np.linalg.norm(actual, axis=1) - 1.0)), 1e-6)


    def test_default_rank_caps_to_fitted_rank_with_warning(self):
        rng = np.random.default_rng(3)
        y = np.repeat([0, 1, 2], 6)
        X = rng.normal(size=(len(y), 7))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        state = face.fit(X.astype(np.float32), y, backend="clip")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transformed = face.transform(X[:3], state)
        self.assertTrue(any("exceeds fitted rank 2" in str(item.message) for item in caught))
        self.assertEqual(transformed.shape, (3, 7))


    def test_fit_is_train_only_and_state_reuses_on_unlabeled_rows(self):
        X, y = _training("clip")
        state = face.fit(X[:35], y[:35], backend="clip", max_rank=8)
        before = {
            key: np.array(value, copy=True)
            for key, value in state.items()
            if key in {"axes", "reference", "neutral"}
        }

        first = face.transform(X[35:], state, rank=4)
        second = face.transform(X[35:], state, rank=4)
        np.testing.assert_array_equal(first, second)
        for key, value in before.items():
            np.testing.assert_array_equal(state[key], value)


    def test_meru_fit_uses_lorentz_log_transport_and_ordered_per_axis_composition(self):
        X, y = _training("meru")
        state = face.fit(X, y, backend="meru", max_rank=8)
        self.assertEqual(state["reference"].shape, (X.shape[1] + 1,))
        self.assertEqual(state["axes"].shape, (8, X.shape[1]))
        self.assertEqual(state["axes"].dtype, np.float32)

        actual = face.transform(
            X[:6],
            state,
            rank=3,
            angular=0.75,
            translation=1.0,
            composition="angular_then_translation",
        )
        expected = X[:6].copy()
        for axis in state["axes"][:3]:
            expected = face._meru_angular_step(expected, axis, state["neutral"], 0.75)
            expected = face._translation_step(
                expected, axis, state["neutral"], 1.0, state["curvature"]
            )
        np.testing.assert_array_equal(actual, expected)


    def test_meru_composition_order_and_independent_strengths_are_effective(self):
        X, y = _training("meru", seed=22)
        state = face.fit(X, y, backend="meru")
        frozen_axes = state["axes"].copy()
        angular_then_translation = face.transform(
            X[:8], state, rank=2, angular=0.5, translation=1.0,
            composition="angular_then_translation",
        )
        translation_then_angular = face.transform(
            X[:8], state, rank=2, angular=0.5, translation=1.0,
            composition="translation_then_angular",
        )
        translation_only = face.transform(
            X[:8], state, rank=2, angular=0.0, translation=1.0,
            composition="angular_then_translation",
        )
        self.assertFalse(np.array_equal(angular_then_translation, translation_then_angular))
        self.assertFalse(np.array_equal(angular_then_translation, translation_only))
        self.assertTrue(np.isfinite(angular_then_translation).all())
        np.testing.assert_array_equal(state["axes"], frozen_axes)


    def test_reverse_and_passes_repeat_the_frozen_axis_sequence(self):
        X, y = _training("clip", seed=31)
        state = face.fit(X, y, backend="clip")
        actual = face.transform(X[:4], state, rank=3, order="reverse", passes=2, alpha=0.8)
        expected = X[:4].copy()
        for _ in range(2):
            for axis in state["axes"][:3][::-1]:
                expected = face._clip_step(expected, axis, state["neutral"], 0.8)
        np.testing.assert_array_equal(actual, expected)


    def test_invalid_meru_composition_is_rejected_when_action_runs(self):
        X, y = _training("meru", seed=42)
        state = face.fit(X, y, backend="meru")
        with self.assertRaisesRegex(ValueError, "composition"):
            face.transform(X[:2], state, rank=1, composition="all_angular_first")

    def test_neutral_fallback_uses_train_mean(self):
        rng = np.random.default_rng(51)
        y = np.repeat([0, 1], 4)
        X = rng.normal(size=(8, 6)).astype(np.float32)
        state = face.fit(X, y, backend="meru", max_rank=1)
        self.assertTrue(state["neutral_used_train_mean_fallback"])
        np.testing.assert_array_equal(state["neutral"], X.mean(axis=0).astype(np.float32))

    def test_clip_degenerate_zero_rows_fail_closed(self):
        y = np.repeat([0, 1], 2)
        with self.assertRaisesRegex(ValueError, "nonzero"):
            face.fit(np.zeros((4, 5), dtype=np.float32), y, backend="clip")

    def test_meru_translation_tau_is_clipped_and_finite(self):
        X = np.zeros((2, 3), dtype=np.float32)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        neutral = np.array([1e10, 0.0, 0.0], dtype=np.float32)
        transformed = face._translation_step(
            X, axis, neutral, strength=1.0, curvature=face.DEFAULT_CURVATURE
        )
        expected_projection = np.sinh(8.0) / np.sqrt(face.DEFAULT_CURVATURE)
        self.assertTrue(np.isfinite(transformed).all())
        np.testing.assert_allclose(transformed[:, 0], expected_projection, rtol=2e-7)
        np.testing.assert_array_equal(transformed[:, 1:], np.zeros((2, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
