import numpy as np

from qvio import metrics, so3


def test_alignment_recovers_yaw_and_translation():
    rng = np.random.default_rng(0)
    p_gt = rng.normal(size=(200, 3)) * [3, 3, 1]
    Rz = so3.exp(np.array([0, 0, 0.7]))
    t = np.array([1.0, -2.0, 0.5])
    p_est = (p_gt - t) @ Rz        # p_gt = Rz p_est + t
    R_hat, t_hat = metrics.align_yaw_translation(p_est, p_gt)
    np.testing.assert_allclose(R_hat, Rz, atol=1e-10)
    np.testing.assert_allclose(t_hat, t, atol=1e-10)


def test_tilt_ignores_yaw():
    R = so3.exp(np.array([0.1, -0.2, 0.3]))
    yawed = so3.exp(np.array([0, 0, 1.2])) @ R
    assert metrics.tilt_error_deg(R[None], yawed[None])[0] < 1e-9
    tilted = so3.exp(np.radians([2.0, 0, 0])) @ R
    np.testing.assert_allclose(metrics.tilt_error_deg(R[None], tilted[None])[0], 2.0, atol=1e-9)


def test_first_pose_alignment():
    R_est0 = so3.exp(np.array([0.05, 0.02, 0.4]))
    Rz = so3.exp(np.array([0, 0, -1.1]))
    R_gt0 = Rz @ R_est0
    Rf, tf = metrics.align_first_pose(R_est0, np.zeros(3), R_gt0, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(Rf, Rz, atol=1e-10)
    np.testing.assert_allclose(tf, [1, 2, 3], atol=1e-12)
