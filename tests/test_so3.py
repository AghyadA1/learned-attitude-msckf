import numpy as np
from scipy.spatial.transform import Rotation

from qvio import so3


def test_exp_log_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(500):
        phi = rng.normal(size=3) * rng.uniform(0, 3.1)
        R = so3.exp(phi)
        np.testing.assert_allclose(R, Rotation.from_rotvec(phi).as_matrix(), atol=1e-10)
        np.testing.assert_allclose(so3.exp(so3.log(R)), R, atol=1e-8)


def test_log_near_pi():
    axis = np.array([1.0, -2.0, 0.5]) / np.linalg.norm([1.0, -2.0, 0.5])
    for angle in (np.pi - 1e-3, np.pi - 1e-6, np.pi):
        R = so3.exp(angle * axis)
        np.testing.assert_allclose(so3.exp(so3.log(R)), R, atol=1e-6)


def test_quaternions_match_scipy():
    rng = np.random.default_rng(1)
    phi = rng.normal(size=(100, 3))
    q = so3.rotvec_to_quat(phi)
    np.testing.assert_allclose(np.abs(np.sum(q * Rotation.from_rotvec(phi).as_quat(), axis=1)), 1.0, atol=1e-12)
    for p, qq in zip(phi, q):
        np.testing.assert_allclose(so3.quat_to_rot(qq), so3.exp(p), atol=1e-12)
        np.testing.assert_allclose(so3.quat_to_rot(so3.rot_to_quat(so3.exp(p))), so3.exp(p), atol=1e-12)
    a, b = q[:50], q[50:]
    product = so3.quat_mul(a, b)
    expected = (Rotation.from_quat(a) * Rotation.from_quat(b)).as_quat()
    np.testing.assert_allclose(np.abs(np.sum(product * expected, axis=1)), 1.0, atol=1e-12)


def test_gravity_alignment_points_up():
    accel = np.array([9.7, 0.4, -1.2])       # EuRoC-like: IMU x roughly up
    R = so3.gravity_aligned_rotation(accel)
    np.testing.assert_allclose(R @ accel / np.linalg.norm(accel), [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
