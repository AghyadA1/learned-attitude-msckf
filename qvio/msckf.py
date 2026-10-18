"""Monocular MSCKF (Mourikis and Roumeliotis, 2007) with an error-state EKF.

State: R_WB, p_WB, v_WB, gyro bias, accel bias, plus a sliding window of IMU
pose clones taken at camera times. Attitude errors are right-multiplicative,
R = R_hat Exp(dtheta), so the attitude error lives in the body frame. The
world frame is gravity aligned with z up.

Error-state layout: [dtheta(3) dp(3) dv(3) dbg(3) dba(3) | clone_1(6) ...],
each clone being [dtheta(3) dp(3)].

The changes from the original project's filter that matter on real data:
  * every feature gets its own chi-square test, so one bad track no longer
    throws away the whole update;
  * a track that has been used (or rejected) keeps being tracked and simply
    starts a fresh measurement history;
  * the filter starts from a detected stationary interval, not ground truth;
  * the gyro can be replaced by a learned gyro model (see ``qvio.gru``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.stats import chi2

from . import so3
from .euroc import Camera, ImuNoise

GRAVITY = np.array([0.0, 0.0, -9.81])
TH, POS, VEL, BG, BA = (slice(i, i + 3) for i in range(0, 15, 3))
N_IMU = 15


@dataclass(frozen=True)
class FilterConfig:
    max_clones: int = 11
    min_track_length: int = 4
    max_features_per_update: int = 50
    pixel_sigma: float = 1.0
    chi2_probability: float = 0.95
    min_depth_m: float = 0.2
    max_depth_m: float = 40.0
    max_reprojection_px: float = 3.0
    zupt_sigma_m_s: float = 0.01
    fej: bool = True
    # Scale factors on the sensor.yaml noise densities.
    gyro_noise_scale: float = 1.0
    accel_noise_scale: float = 1.0
    walk_noise_scale: float = 1.0
    # Initial standard deviations at the end of the stationary interval.
    init_tilt_sigma_deg: float = 1.0
    init_yaw_sigma_deg: float = 0.1
    init_velocity_sigma: float = 0.02
    init_gyro_bias_sigma: float = 0.002
    init_accel_bias_sigma: float = 0.1


@dataclass
class Clone:
    t: int
    R: np.ndarray
    p: np.ndarray
    R_fej: np.ndarray       # value when the clone was taken, used in Jacobians
    p_fej: np.ndarray


@dataclass
class UpdateStats:
    t: int
    candidates: int = 0
    triangulated: int = 0
    accepted: int = 0
    rows: int = 0


@dataclass
class _Track:
    obs: list = field(default_factory=list)   # (clone time, xy)


class MSCKF:
    def __init__(self, camera: Camera, noise: ImuNoise, config: FilterConfig = FilterConfig(),
                 gyro_model=None):
        self.camera = camera
        self.config = config
        self.gyro_model = gyro_model
        c = config
        self.q_gyro = (noise.gyro_noise * c.gyro_noise_scale) ** 2
        self.q_accel = (noise.accel_noise * c.accel_noise_scale) ** 2
        self.q_gyro_walk = (noise.gyro_walk * c.walk_noise_scale) ** 2
        self.q_accel_walk = (noise.accel_walk * c.walk_noise_scale) ** 2
        self.sigma_xy = c.pixel_sigma / camera.focal
        self.tracks: dict[int, _Track] = {}
        self.clones: list[Clone] = []
        self.stats: list[UpdateStats] = []
        self.t: int | None = None

    # ------------------------------------------------------------------ setup

    def initialize(self, t: int, R: np.ndarray, bg: np.ndarray, ba: np.ndarray | None = None) -> None:
        c = self.config
        self.t = int(t)
        self.R = np.asarray(R, float).copy()
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.bg = np.asarray(bg, float).copy()
        self.ba = np.zeros(3) if ba is None else np.asarray(ba, float).copy()
        # Tilt is uncertain, yaw is arbitrary (we chose it), express both in
        # the world frame and rotate into the body-frame error.
        world = np.diag(np.radians([c.init_tilt_sigma_deg, c.init_tilt_sigma_deg, c.init_yaw_sigma_deg]) ** 2)
        P = np.zeros((N_IMU, N_IMU))
        P[TH, TH] = self.R.T @ world @ self.R
        P[POS, POS] = np.eye(3) * 1e-6
        P[VEL, VEL] = np.eye(3) * c.init_velocity_sigma**2
        P[BG, BG] = np.eye(3) * c.init_gyro_bias_sigma**2
        P[BA, BA] = np.eye(3) * c.init_accel_bias_sigma**2
        self.P = P
        self._fej = (self.R.copy(), self.p.copy(), self.v.copy())

    @property
    def dim(self) -> int:
        return N_IMU + 6 * len(self.clones)

    # ------------------------------------------------------------ propagation

    def propagate(self, t: np.ndarray, gyro: np.ndarray, accel: np.ndarray) -> None:
        """Integrate IMU samples whose first timestamp is the current time.

        ``t[-1]`` becomes the new filter time. When a learned gyro model is
        attached it supplies a constant rate correction and its own noise for
        the whole interval.
        """
        t = np.asarray(t, np.int64)
        if t[0] != self.t:
            raise ValueError("IMU samples must start at the current filter time")
        r = np.zeros(3)
        q_gyro = self.q_gyro
        if self.gyro_model is not None:
            out = self.gyro_model.correction(t, gyro, accel, self.bg)
            if out is not None:
                r, window_var = out
                # A per-window error of variance V spread evenly over the
                # window behaves like white noise of density V / T.
                q_gyro = window_var / ((t[-1] - t[0]) * 1e-9)

        Phi_total = np.eye(N_IMU)
        Qc = np.zeros((12, 12))
        Qc[0:3, 0:3] = np.diag(np.broadcast_to(q_gyro, 3))
        Qc[3:6, 3:6] = np.eye(3) * self.q_accel
        Qc[6:9, 6:9] = np.eye(3) * self.q_gyro_walk
        Qc[9:12, 9:12] = np.eye(3) * self.q_accel_walk
        P = self.P[:N_IMU, :N_IMU]
        # First-estimate Jacobians: the transition out of the last update is
        # linearised at the state before that update, which keeps yaw and
        # position unobservable (Huang, Mourikis and Roumeliotis, 2010).
        R0, p0, v0 = self._fej if self.config.fej else (self.R, self.p, self.v)
        I3 = np.eye(3)
        for i in range(len(t) - 1):
            dt = (t[i + 1] - t[i]) * 1e-9
            w = 0.5 * (gyro[i] + gyro[i + 1]) - self.bg + r
            a0 = accel[i] - self.ba
            a1 = accel[i + 1] - self.ba
            dR = so3.exp(w * dt)
            R1 = self.R @ dR
            acc = 0.5 * (self.R @ a0 + R1 @ a1) + GRAVITY
            p1 = self.p + self.v * dt + 0.5 * acc * dt * dt
            v1 = self.v + acc * dt

            R_mid = self.R @ so3.exp(0.5 * w * dt)
            Ra = R_mid @ so3.skew(0.5 * (a0 + a1))
            Phi = np.eye(N_IMU)
            Phi[TH, TH] = R1.T @ R0
            Phi[TH, BG] = -dt * I3
            Phi[POS, TH] = -so3.skew(p1 - p0 - v0 * dt - 0.5 * GRAVITY * dt * dt) @ R0
            Phi[POS, VEL] = dt * I3
            Phi[POS, BG] = Ra * dt**3 / 6.0
            Phi[POS, BA] = -0.5 * R_mid * dt * dt
            Phi[VEL, TH] = -so3.skew(v1 - v0 - GRAVITY * dt) @ R0
            Phi[VEL, BG] = 0.5 * Ra * dt * dt
            Phi[VEL, BA] = -R_mid * dt
            G = np.zeros((N_IMU, 12))
            G[TH, 0:3] = -I3
            G[VEL, 3:6] = -R_mid
            G[BG, 6:9] = I3
            G[BA, 9:12] = I3
            Qd = G @ Qc @ G.T * dt
            P = Phi @ P @ Phi.T + Qd
            Phi_total = Phi @ Phi_total

            self.R, self.p, self.v = R1, p1, v1
            R0, p0, v0 = R1, p1, v1

        self.P[:N_IMU, :N_IMU] = P
        if self.clones:
            self.P[:N_IMU, N_IMU:] = Phi_total @ self.P[:N_IMU, N_IMU:]
            self.P[N_IMU:, :N_IMU] = self.P[:N_IMU, N_IMU:].T
        self.R = _orthonormalize(self.R)
        self.t = int(t[-1])
        self._fej = (self.R.copy(), self.p.copy(), self.v.copy())

    # ------------------------------------------------------- clone management

    def augment(self) -> None:
        n = self.dim
        J = np.zeros((6, n))
        J[0:3, TH] = np.eye(3)
        J[3:6, POS] = np.eye(3)
        P = np.zeros((n + 6, n + 6))
        P[:n, :n] = self.P
        P[n:, :n] = J @ self.P
        P[:n, n:] = P[n:, :n].T
        P[n:, n:] = J @ self.P @ J.T
        self.P = P
        self.clones.append(Clone(self.t, self.R.copy(), self.p.copy(), self.R.copy(), self.p.copy()))

    def _marginalize_oldest(self) -> None:
        keep = np.r_[np.arange(N_IMU), np.arange(N_IMU + 6, self.dim)]
        self.P = self.P[np.ix_(keep, keep)]
        old = self.clones.pop(0)
        for track in self.tracks.values():
            track.obs = [o for o in track.obs if o[0] != old.t]

    # ---------------------------------------------------------------- vision

    def process_frame(self, t: int, ids: np.ndarray, xy: np.ndarray) -> UpdateStats:
        """Clone the pose at this camera time and run the MSCKF update."""
        if int(t) != self.t:
            raise ValueError("propagate to the frame time before processing it")
        self.augment()
        for i, track_id in enumerate(ids):
            self.tracks.setdefault(int(track_id), _Track()).obs.append((self.t, xy[i]))

        current = {int(i) for i in ids}
        lost = [k for k in self.tracks if k not in current]
        ready = [k for k in lost if len(self.tracks[k].obs) >= self.config.min_track_length]
        if len(self.clones) > self.config.max_clones:
            oldest = self.clones[0].t
            ready += [
                k for k, tr in self.tracks.items()
                if k in current and tr.obs and tr.obs[0][0] == oldest
                and len(tr.obs) >= self.config.min_track_length
            ]
        stats = self._update_features(ready)
        for k in lost:
            self.tracks.pop(k, None)
        for k in ready:
            if k in self.tracks:
                # Measurements are used once; the track keeps going with a
                # fresh history from the next frame on.
                self.tracks[k].obs = []
        if len(self.clones) > self.config.max_clones:
            self._marginalize_oldest()
        self.stats.append(stats)
        return stats

    def drop_tracks(self) -> None:
        """Forget all feature histories (camera outage)."""
        self.tracks.clear()

    def _update_features(self, track_ids: list[int]) -> UpdateStats:
        c = self.config
        stats = UpdateStats(self.t, candidates=len(track_ids))
        if not track_ids:
            return stats
        track_ids = sorted(track_ids, key=lambda k: len(self.tracks[k].obs), reverse=True)
        track_ids = track_ids[: c.max_features_per_update]

        index = {clone.t: i for i, clone in enumerate(self.clones)}
        R_WC = np.asarray([cl.R @ self.camera.R_BC for cl in self.clones])
        p_WC = np.asarray([cl.p + cl.R @ self.camera.p_BC for cl in self.clones])
        poses = (R_WC, p_WC)
        rows_H, rows_r = [], []
        for k in track_ids:
            obs = [(index[t], z) for t, z in self.tracks[k].obs if t in index]
            if len(obs) < c.min_track_length:
                continue
            f = self._triangulate(obs, poses)
            if f is None:
                continue
            stats.triangulated += 1
            H, r = self._feature_jacobian(f, obs)
            if H is None:
                continue
            S = H @ self.P @ H.T + self.sigma_xy**2 * np.eye(len(r))
            gamma = float(r @ np.linalg.solve(S, r))
            if gamma > _chi2_threshold(c.chi2_probability, len(r)):
                continue
            rows_H.append(H)
            rows_r.append(r)
            stats.accepted += 1
        if not rows_H:
            return stats
        H = np.vstack(rows_H)
        r = np.concatenate(rows_r)
        if H.shape[0] > H.shape[1]:
            Q, T = np.linalg.qr(H)
            H, r = T, Q.T @ r
        stats.rows = len(r)
        self._ekf_update(H, r, self.sigma_xy**2 * np.eye(len(r)))
        return stats

    def _triangulate(self, obs, poses) -> np.ndarray | None:
        c = self.config
        idx = [j for j, _ in obs]
        z = np.asarray([zj for _, zj in obs])
        R_CW = np.transpose(poses[0][idx], (0, 2, 1))     # [M, 3, 3]
        p_WC = poses[1][idx]                              # [M, 3]

        # Linear estimate: x * row3 - row1 = 0 and y * row3 - row2 = 0 per view.
        P34 = np.concatenate([R_CW, -np.einsum("mij,mj->mi", R_CW, p_WC)[:, :, None]], axis=2)
        A = np.concatenate([z[:, :1] * P34[:, 2] - P34[:, 0], z[:, 1:] * P34[:, 2] - P34[:, 1]])
        X = np.linalg.svd(A)[2][-1]
        if abs(X[3]) < 1e-12:
            return None
        f = X[:3] / X[3]

        # Gauss-Newton on the reprojection error.
        for _ in range(5):
            pc = np.einsum("mij,mj->mi", R_CW, f - p_WC)
            if np.any(pc[:, 2] <= 1e-3):
                return None
            J = np.einsum("mab,mbc->mac", _projection_jacobians(pc), R_CW).reshape(-1, 3)
            e = (z - pc[:, :2] / pc[:, 2:]).ravel()
            try:
                step = np.linalg.solve(J.T @ J, J.T @ e)
            except np.linalg.LinAlgError:
                return None
            f = f + step
            if np.linalg.norm(step) < 1e-6 * max(1.0, np.linalg.norm(f)):
                break

        pc = np.einsum("mij,mj->mi", R_CW, f - p_WC)
        if np.any(pc[:, 2] <= c.min_depth_m) or np.any(pc[:, 2] >= c.max_depth_m):
            return None
        reprojection = np.linalg.norm(z - pc[:, :2] / pc[:, 2:], axis=1) * self.camera.focal
        # No minimum parallax. While the drone hovers every track has almost
        # none, and those tracks are what keeps the velocity from drifting
        # (MH_04 drifted 7 m in a 10 s hover with a 1 deg threshold). Bad
        # triangulations are caught by the depth and reprojection checks
        # above and by the chi-squared test.
        if np.any(reprojection > c.max_reprojection_px):
            return None
        return f

    def _feature_jacobian(self, f, obs):
        R_BC, p_BC = self.camera.R_BC, self.camera.p_BC
        idx = [j for j, _ in obs]
        z = np.asarray([zj for _, zj in obs])
        m = len(obs)
        clones = [self.clones[j] for j in idx]
        R = np.asarray([cl.R for cl in clones])
        p = np.asarray([cl.p for cl in clones])
        # Residual at the current estimate, Jacobians at the first estimates.
        pc = (np.einsum("mji,mj->mi", R, f - p) - p_BC) @ R_BC
        if self.config.fej:
            R = np.asarray([cl.R_fej for cl in clones])
            p = np.asarray([cl.p_fej for cl in clones])
        v = np.einsum("mji,mj->mi", R, f - p)               # feature in the body frames
        Jpi = _projection_jacobians((v - p_BC) @ R_BC) @ R_BC.T   # [m, 2, 3], d(pi)/d(body point)
        RT = np.transpose(R, (0, 2, 1))
        H_x = np.zeros((2 * m, self.dim))
        for row, j in enumerate(idx):
            col = N_IMU + 6 * j
            H_x[2 * row:2 * row + 2, col:col + 3] = Jpi[row] @ so3.skew(v[row])
            H_x[2 * row:2 * row + 2, col + 3:col + 6] = -Jpi[row] @ RT[row]
        H_f = np.einsum("mab,mbc->mac", Jpi, RT).reshape(-1, 3)
        r = (z - pc[:, :2] / pc[:, 2:]).ravel()
        # Project onto the left null space of H_f to remove the feature.
        Q, _ = np.linalg.qr(H_f, mode="complete")
        N = Q[:, 3:]
        return N.T @ H_x, N.T @ r

    # ------------------------------------------------------- other updates

    def zero_velocity_update(self) -> bool:
        H = np.zeros((3, self.dim))
        H[:, VEL] = np.eye(3)
        Rn = np.eye(3) * self.config.zupt_sigma_m_s**2
        r = -self.v
        S = H @ self.P @ H.T + Rn
        if float(r @ np.linalg.solve(S, r)) > _chi2_threshold(0.999, 3):
            return False
        self._ekf_update(H, r, Rn)
        return True

    def _ekf_update(self, H: np.ndarray, r: np.ndarray, Rn: np.ndarray) -> None:
        PHt = self.P @ H.T
        S = H @ PHt + Rn
        K = np.linalg.solve(S, PHt.T).T
        dx = K @ r
        IKH = np.eye(self.dim) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rn @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        self.R = _orthonormalize(self.R @ so3.exp(dx[TH]))
        self.p = self.p + dx[POS]
        self.v = self.v + dx[VEL]
        self.bg = self.bg + dx[BG]
        self.ba = self.ba + dx[BA]
        for j, clone in enumerate(self.clones):
            s = N_IMU + 6 * j
            clone.R = _orthonormalize(clone.R @ so3.exp(dx[s:s + 3]))
            clone.p = clone.p + dx[s + 3:s + 6]

    # ---------------------------------------------------------------- output

    def snapshot(self) -> dict:
        return {
            "t": self.t,
            "p": self.p.copy(),
            "q": so3.rot_to_quat(self.R),
            "v": self.v.copy(),
            "bg": self.bg.copy(),
            "ba": self.ba.copy(),
            "P": self.P[:N_IMU, :N_IMU].copy(),
        }


@lru_cache(maxsize=None)
def _chi2_threshold(probability: float, dof: int) -> float:
    return float(chi2.ppf(probability, dof))


def _projection_jacobians(pc: np.ndarray) -> np.ndarray:
    """d(x/z, y/z)/d(point) for [M, 3] camera points, shape [M, 2, 3]."""
    x, y, z = pc.T
    J = np.zeros((len(pc), 2, 3))
    J[:, 0, 0] = J[:, 1, 1] = 1.0 / z
    J[:, 0, 2] = -x / z**2
    J[:, 1, 2] = -y / z**2
    return J


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt
