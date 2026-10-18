"""Velocity from the camera and IMU alone, for recovering after an outage.

After a 5 s gap the filter can come back with a velocity error of a metre
per second and a covariance that claims a tenth of that. Its chi-square gate
then rejects nearly every feature, since they disagree with its state, so
the camera cannot pull the velocity back. On V2_03 the filter stays wrong
for the whole 5 s of vision before the next gap, and the error carries over.

The fix solves for the velocity again from the first second of images after
the gap, using only the feature bearings, the IMU, and the filter's attitude
and biases (which stay accurate through a gap, unlike the velocity). With
the camera centre at time tau written as

    c(tau) = p0 + v0 tau + d(tau),

where d is the IMU position change from a standing start, every bearing of a
feature at f gives two linear equations in f and v0. Eliminating the
features leaves a 3x3 system for v0 (the closed-form visual-inertial
solution of Martinelli, 2014, with the attitude known). That linear
solution minimises distances in metres, which pulls it towards zero
velocity as the pixel noise grows (0.70 m/s came out as 0.38 m/s in a
simulation with 3 mrad of noise), so it is refined by Gauss-Newton on the
reprojection error, which removes the bias.

The filter only takes the result when it has accepted few features over
that second, which is the sign that its own state no longer fits the
images. Then its velocity covariance is widened to cover the disagreement
before the update, so the update all but replaces the velocity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import so3
from .euroc import Camera
from .msckf import GRAVITY


@dataclass(frozen=True)
class RecoveryConfig:
    frames: int = 20                 # 1 s of images at 20 Hz
    max_accepted: int = 40           # features accepted over those frames
    sigma_m_s: float = 0.2           # velocity standard deviation per axis
    min_observations: int = 3
    min_features: int = 8
    huber: float = 3e-3              # in normalised image units


def preintegrate(t: np.ndarray, gyro: np.ndarray, accel: np.ndarray, R: np.ndarray,
                 bg: np.ndarray, ba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World-frame velocity and position change over the samples, from rest."""
    dv = np.zeros(3)
    dp = np.zeros(3)
    for i in range(len(t) - 1):
        dt = (t[i + 1] - t[i]) * 1e-9
        R1 = R @ so3.exp((0.5 * (gyro[i] + gyro[i + 1]) - bg) * dt)
        a = 0.5 * (R @ (accel[i] - ba) + R1 @ (accel[i + 1] - ba)) + GRAVITY
        dp = dp + dv * dt + 0.5 * a * dt * dt
        dv = dv + a * dt
        R = R1
    return dv, dp


class RecoveryWindow:
    """Images and IMU from the moment the camera comes back."""

    def __init__(self, t0: int):
        self.t0 = t0
        self.dp = np.zeros(3)        # IMU position change since t0, from rest
        self.dv = np.zeros(3)
        self.tau, self.R, self.q, self.ids, self.xy = [], [], [], [], []
        self.accepted = 0

    def add_imu(self, t: np.ndarray, gyro: np.ndarray, accel: np.ndarray, R: np.ndarray,
                bg: np.ndarray, ba: np.ndarray) -> None:
        dv, dp = preintegrate(t, gyro, accel, R, bg, ba)
        self.dp = self.dp + self.dv * (t[-1] - t[0]) * 1e-9 + dp
        self.dv = self.dv + dv

    def add_frame(self, t: int, R: np.ndarray, camera: Camera, ids: np.ndarray, xy: np.ndarray) -> None:
        self.tau.append((t - self.t0) * 1e-9)
        self.R.append(R @ camera.R_BC)
        self.q.append(self.dp + R @ camera.p_BC)
        self.ids.append(ids)
        self.xy.append(xy)

    def __len__(self) -> int:
        return len(self.tau)

    def velocity(self, config: RecoveryConfig = RecoveryConfig()) -> np.ndarray | None:
        """Velocity at the latest frame, or None if the images do not fix it."""
        obs = self._observations(config.min_observations)
        if obs is None or obs[0].max() + 1 < config.min_features:
            return None
        v0 = _linear_velocity(*obs)
        if v0 is None:
            return None
        v0 = _refine_velocity(*obs, v0, config.huber)
        if v0 is None:
            return None
        return v0 + self.dv

    def _observations(self, min_obs: int):
        frame = np.concatenate([np.full(len(i), k) for k, i in enumerate(self.ids)])
        ids = np.concatenate(self.ids)
        xy = np.vstack(self.xy)
        _, feature, counts = np.unique(ids, return_inverse=True, return_counts=True)
        keep = counts[feature] >= min_obs
        if not keep.any():
            return None
        _, feature = np.unique(feature[keep], return_inverse=True)
        tau = np.asarray(self.tau)[frame[keep]]
        return feature, xy[keep], np.asarray(self.R)[frame[keep]], np.asarray(self.q)[frame[keep]], tau


def _linear_velocity(feature, xy, R, q, tau):
    """v0 minimising the distances of the features from their bearing rays."""
    d = np.einsum("nij,nj->ni", R, np.column_stack([xy, np.ones(len(xy))]))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    M = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
    n = feature.max() + 1
    A = np.zeros((n, 3, 3)); P = np.zeros((n, 3, 3)); a = np.zeros((n, 3)); b = np.zeros((n, 3))
    np.add.at(A, feature, M)
    np.add.at(P, feature, tau[:, None, None] * M)
    np.add.at(a, feature, np.einsum("nij,nj->ni", M, q))
    np.add.at(b, feature, tau[:, None] * np.einsum("nij,nj->ni", M, q))
    Q = np.einsum("n,nij->ij", tau**2, M)
    Ai = np.linalg.inv(A + 1e-9 * np.eye(3))
    H = Q - np.einsum("nij,njk,nkl->il", P, Ai, P)
    g = np.einsum("nij,njk,nk->i", P, Ai, a) - b.sum(axis=0)
    try:
        return np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        return None


def _refine_velocity(feature, xy, R, q, tau, v0, huber, iterations=5):
    """Gauss-Newton on the reprojection error over v0 and the feature points."""
    n = feature.max() + 1
    # Start the points from the rays given v0.
    d = np.einsum("nij,nj->ni", R, np.column_stack([xy, np.ones(len(xy))]))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    M = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
    A = np.zeros((n, 3, 3)); b = np.zeros((n, 3))
    np.add.at(A, feature, M)
    np.add.at(b, feature, np.einsum("nij,nj->ni", M, q + tau[:, None] * v0))
    f = np.linalg.solve(A + 1e-9 * np.eye(3), b[..., None])[..., 0]
    v = np.array(v0, float)
    for _ in range(iterations):
        x = np.einsum("nji,nj->ni", R, f[feature] - q - tau[:, None] * v)    # camera frame
        front = x[:, 2] > 0.05
        iz = 1.0 / np.where(front, x[:, 2], 1.0)
        r = xy - x[:, :2] * iz[:, None]
        e = np.linalg.norm(r, axis=1)
        w = np.where(e < huber, 1.0, huber / np.maximum(e, 1e-12)) * front
        Jp = np.zeros((len(xy), 2, 3))
        Jp[:, 0, 0] = iz
        Jp[:, 1, 1] = iz
        Jp[:, :, 2] = -x[:, :2] * iz[:, None] ** 2
        Jf = np.einsum("nij,nkj->nik", Jp, R)
        Jv = -tau[:, None, None] * Jf
        Wf = Jf * w[:, None, None]
        Aff = np.zeros((n, 3, 3)); Bfv = np.zeros((n, 3, 3)); gf = np.zeros((n, 3))
        np.add.at(Aff, feature, np.einsum("nki,nkj->nij", Wf, Jf))
        np.add.at(Bfv, feature, np.einsum("nki,nkj->nij", Wf, Jv))
        np.add.at(gf, feature, np.einsum("nki,nk->ni", Wf, r))
        Ai = np.linalg.inv(Aff + 1e-9 * np.eye(3))
        AiB = Ai @ Bfv
        Aig = np.einsum("nij,nj->ni", Ai, gf)
        H = np.einsum("nki,nkj->ij", Jv * w[:, None, None], Jv) - np.einsum("nji,njk->ik", Bfv, AiB)
        g = np.einsum("nki,nk->i", Jv * w[:, None, None], r) - np.einsum("nji,nj->i", Bfv, Aig)
        try:
            dv = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return None
        v = v + dv
        f = f + Aig - np.einsum("nij,j->ni", AiB, dv)
        if np.linalg.norm(dv) < 1e-4:
            break
    return v
