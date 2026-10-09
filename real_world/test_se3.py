"""Small synthetic checks; these are not benchmark data."""
import unittest
import numpy as np
import se3


def fixture():
    rng = np.random.default_rng(61)
    labels = np.repeat(np.arange(9), 15)
    centers = rng.normal(size=(9, 6)) * .35
    twist = centers[labels] + rng.normal(size=(len(labels), 6)) * .12
    poses = np.array([se3.pack_rt(*se3.se3_exp(v)) for v in twist])
    return poses, labels


class PoseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = fixture()
        cls.state = se3.fit(cls.X, cls.y)

    def test_bridge_and_prefix(self):
        s = self.state
        np.testing.assert_allclose(s['body_twists'] @ s['jacobian'].T, s['axes'], atol=2e-12)
        self.assertEqual(s['axes'].shape, (6, 12))
        expected = self.X.copy()
        for u in s['axes'][:5]:
            expected = se3.trans_isorot_step(expected, u, s['neutral'], 2.)
        np.testing.assert_array_equal(se3.transform(self.X, s), expected)
        np.testing.assert_array_equal(expected[:, :9], self.X[:, :9])

    def test_rotation_joint_and_schedule(self):
        s = self.state
        for action in ('rotation', 'joint'):
            out = se3.transform(self.X[:7], s, action=action)
            R = out[:, :9].reshape(-1, 3, 3)
            np.testing.assert_allclose(R.transpose(0, 2, 1) @ R, np.broadcast_to(np.eye(3), R.shape), atol=2e-14)
            if action == 'rotation':
                np.testing.assert_array_equal(out[:, 9:], self.X[:7, 9:])
        np.testing.assert_array_equal(se3.transform(self.X, s, schedule=['skip']*5), self.X)
        schedule = ['rotation', 'translation', 'joint', 'skip', 'translation']
        out = se3.transform(self.X, s, schedule=schedule)
        self.assertTrue(np.isfinite(out).all())
        with self.assertRaises(ValueError):
            se3.transform(self.X, s, schedule=['skip'])

    def test_fit_independent_of_transform_input(self):
        before = {k: np.asarray(v).copy() for k, v in self.state.items()}
        se3.transform(self.X[::-1][:11], self.state)
        for key in before:
            np.testing.assert_array_equal(before[key], self.state[key])
        state = se3.fit(self.X, self.y)
        for key in before:
            np.testing.assert_array_equal(before[key], state[key])

    def test_rank_and_degenerate_rotation(self):
        with self.assertRaises(se3.TangentFLDError):
            se3.fit(self.X, self.y % 2)
        state = dict(self.state, body_twists=self.state['body_twists'].copy())
        state['body_twists'][0, :3] = 0
        with self.assertRaisesRegex(ValueError, 'degenerate'):
            se3.transform(self.X, state, action='rotation')
        se3.transform(self.X, state, action='translation')

    def test_small_angle_and_joint_original_input(self):
        xi = np.array([1e-10, -2e-10, 3e-10, 1., -2., 3.])
        np.testing.assert_allclose(se3.se3_log(*se3.se3_exp(xi)), xi, atol=1e-14)
        s = self.state
        w = s['body_twists'][0, :3]
        w = w / np.linalg.norm(w)
        joint = se3.se3_isorot_step(self.X, s['axes'][0], w, s['neutral'], .8)
        trans = se3.trans_isorot_step(self.X, s['axes'][0], s['neutral'], .8)
        np.testing.assert_array_equal(joint[:, 9:], trans[:, 9:])


if __name__ == '__main__':
    unittest.main()
