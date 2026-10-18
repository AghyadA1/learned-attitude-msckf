"""Small SO(3) toolbox.

Quaternions are Hamilton, scalar-last ``[x, y, z, w]`` (the SciPy order).
``R_WB`` maps body vectors into the world frame.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-10


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp(phi: np.ndarray) -> np.ndarray:
    """Rotation vector to rotation matrix (Rodrigues)."""
    angle = float(np.linalg.norm(phi))
    K = skew(phi)
    if angle < _EPS:
        return np.eye(3) + K + 0.5 * K @ K
    a = np.sin(angle) / angle
    b = (1.0 - np.cos(angle)) / angle**2
    return np.eye(3) + a * K + b * K @ K


def log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to rotation vector, valid up to (and near) pi."""
    cos_angle = np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1e-6:
        return 0.5 * w
    if np.pi - angle < 1e-4:
        # sin(angle) ~ 0: read the axis from the symmetric part instead.
        B = 0.5 * (R + np.eye(3))
        axis = np.sqrt(np.clip(np.diag(B), 0.0, None))
        k = int(np.argmax(axis))
        axis[np.arange(3) != k] = B[k, np.arange(3) != k] / axis[k]
        axis /= np.linalg.norm(axis)
        if axis @ w < 0:
            axis = -axis
        return angle * axis
    return angle / (2.0 * np.sin(angle)) * w


def right_jacobian(phi: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(phi))
    K = skew(phi)
    if angle < 1e-6:
        return np.eye(3) - 0.5 * K
    return (
        np.eye(3)
        - (1.0 - np.cos(angle)) / angle**2 * K
        + (angle - np.sin(angle)) / angle**3 * K @ K
    )


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to unit quaternion with w >= 0."""
    m = R
    trace = np.trace(m)
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, works on [..., 4] arrays."""
    x1, y1, z1, w1 = np.moveaxis(np.asarray(q1, float), -1, 0)
    x2, y2, z2, w2 = np.moveaxis(np.asarray(q2, float), -1, 0)
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.array(q, float)
    q[..., :3] *= -1.0
    return q


def rotvec_to_quat(phi: np.ndarray) -> np.ndarray:
    """Batched rotation vector to quaternion, [..., 3] -> [..., 4]."""
    phi = np.asarray(phi, float)
    angle = np.linalg.norm(phi, axis=-1, keepdims=True)
    half = 0.5 * angle
    # sin(x/2)/x with a series fallback near zero
    scale = np.where(angle > 1e-8, np.sin(half) / np.maximum(angle, 1e-12), 0.5 - angle**2 / 48.0)
    return np.concatenate([phi * scale, np.cos(half)], axis=-1)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Batched quaternion to rotation vector, shortest rotation."""
    q = np.asarray(q, float)
    q = q * np.where(q[..., 3:4] < 0, -1.0, 1.0)
    v = q[..., :3]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, q[..., 3:4])
    scale = np.where(s > 1e-12, angle / np.maximum(s, 1e-12), 2.0)
    return v * scale


def angle_between_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic distance between two rotations, in degrees."""
    return float(np.degrees(np.linalg.norm(log(R1.T @ R2))))


def gravity_aligned_rotation(accel_mean: np.ndarray) -> np.ndarray:
    """R_WB with zero yaw from the mean specific force of a static IMU.

    A static accelerometer measures +g along the world up axis, expressed in
    the body frame, so we look for R_WB with R_WB @ a/|a| = e_z.
    """
    up_body = np.asarray(accel_mean, float) / np.linalg.norm(accel_mean)
    z = up_body
    # Pick the body axis least aligned with up as the seed for world x.
    seed = np.eye(3)[int(np.argmin(np.abs(z)))]
    x = seed - (seed @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_BW = np.column_stack([x, y, z])  # columns: world axes in body coordinates
    return R_BW.T
