"""A small synthetic EuRoC-like world for checking the filter.

The camera model and IMU-camera extrinsics are EuRoC's, the trajectory is a
smooth figure-of-eight inside a box of random landmarks, and the IMU noise
uses the sensor.yaml densities. Everything is seeded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qvio import so3
from qvio.euroc import Camera, ImuNoise

R_BC = np.array([
    [0.0148655429818, -0.999880929698, 0.00414029679422],
    [0.999557249008, 0.0149672133247, 0.025715529948],
    [-0.0257744366974, 0.00375618835797, 0.999660727178],
])
CAMERA = Camera(
    K=np.array([[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]]),
    dist=np.zeros(4),
    size=(752, 480),
    R_BC=R_BC,
    p_BC=np.array([-0.0216401454975, -0.064676986768, 0.00981073058949]),
)
NOISE = ImuNoise(gyro_noise=1.6968e-4, accel_noise=2.0e-3, gyro_walk=1.9393e-5, accel_walk=3.0e-3)
# EuRoC's IMU has x pointing up; this base attitude makes the camera look
# horizontally along world +x.
R_BASE = np.column_stack([[0, 0, 1], [0, -1, 0], [1, 0, 0]]).astype(float)
G = np.array([0.0, 0.0, -9.81])


def pose(t: float, still_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    # Optional stationary start: the motion clock tau ramps up smoothly
    # (velocity and acceleration both start at zero).
    u = max(t - still_s, 0.0)
    t = u**4 / (u**3 + 1.0)
    p = np.array([1.5 * np.sin(0.4 * t), 1.0 * np.sin(0.8 * t), 1.2 + 0.3 * np.sin(0.6 * t)])
    yaw = 0.6 * np.sin(0.3 * t)
    tilt = np.array([0.08 * np.sin(1.1 * t), 0.06 * np.cos(0.9 * t), 0.0])
    R = so3.exp(np.array([0.0, 0.0, yaw])) @ so3.exp(tilt) @ R_BASE
    return R, p


@dataclass
class SimData:
    imu_t: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray
    cam_t: np.ndarray
    frames: list            # (ids, noisy xy, clean pixels) per camera frame
    gt_R: np.ndarray        # at imu_t
    gt_p: np.ndarray
    gt_v: np.ndarray
    bg: np.ndarray
    ba: np.ndarray


def simulate(duration_s: float = 40.0, seed: int = 0, n_landmarks: int = 1500,
             track_drop_rate: float = 0.04, pixel_sigma: float = 1.0, still_s: float = 0.0) -> SimData:
    rng = np.random.default_rng(seed)
    rate = 200
    n = int(duration_s * rate) + 1
    imu_t = (np.arange(n) * 5_000_000).astype(np.int64)
    ts = imu_t * 1e-9
    h = 1e-4

    Rs, ps, vs, gyro, accel = [], [], [], [], []
    for t in ts:
        R0, p0 = pose(t, still_s)
        Rm, pm = pose(t - h, still_s)
        Rp, pp = pose(t + h, still_s)
        w = so3.log(Rm.T @ Rp) / (2 * h)
        v = (pp - pm) / (2 * h)
        a = (pp - 2 * p0 + pm) / h**2
        Rs.append(R0); ps.append(p0); vs.append(v)
        gyro.append(w)
        accel.append(R0.T @ (a - G))
    gyro, accel = np.asarray(gyro), np.asarray(accel)

    dt = 1.0 / rate
    bg0 = np.radians([0.3, -0.2, 0.4])       # deg/s-sized biases, like EuRoC
    ba0 = np.array([0.05, -0.03, 0.08])
    bg = bg0 + np.cumsum(rng.normal(0, NOISE.gyro_walk * np.sqrt(dt), (n, 3)), axis=0)
    ba = ba0 + np.cumsum(rng.normal(0, NOISE.accel_walk * np.sqrt(dt), (n, 3)), axis=0)
    gyro_meas = gyro + bg + rng.normal(0, NOISE.gyro_noise / np.sqrt(dt), (n, 3))
    accel_meas = accel + ba + rng.normal(0, NOISE.accel_noise / np.sqrt(dt), (n, 3))

    # Landmarks on the walls of a 10 m x 8 m x 4 m room.
    L = rng.uniform([-5, -4, 0], [5, 4, 4], (n_landmarks, 3))
    wall = rng.integers(0, 4, n_landmarks)
    L[wall == 0, 0] = 5.0
    L[wall == 1, 0] = -5.0
    L[wall == 2, 1] = 4.0
    L[wall == 3, 1] = -4.0

    cam_idx = np.arange(0, n, 10)
    frames = []
    track_of = {}           # landmark -> current track id
    next_id = 0
    K = CAMERA.K
    w_px, h_px = CAMERA.size
    for k in cam_idx:
        R_WC = Rs[k] @ CAMERA.R_BC
        p_WC = ps[k] + Rs[k] @ CAMERA.p_BC
        pc = (L - p_WC) @ R_WC
        visible = pc[:, 2] > 0.3
        uv = np.full((len(L), 2), -1.0)
        uv[visible] = (pc[visible, :2] / pc[visible, 2:3]) @ K[:2, :2].T + K[:2, 2]
        visible &= (uv[:, 0] > 10) & (uv[:, 0] < w_px - 10) & (uv[:, 1] > 10) & (uv[:, 1] < h_px - 10)
        ids, xy, px = [], [], []
        for j in np.flatnonzero(visible):
            if j in track_of and rng.random() < track_drop_rate:
                del track_of[j]
            if j not in track_of:
                track_of[j] = next_id
                next_id += 1
            noisy = uv[j] + rng.normal(0, pixel_sigma, 2)
            ids.append(track_of[j])
            xy.append([(noisy[0] - K[0, 2]) / K[0, 0], (noisy[1] - K[1, 2]) / K[1, 1]])
            px.append(uv[j])
        for j in [j for j in track_of if not visible[j]]:
            del track_of[j]
        # Keep it KLT-sized: at most ~150 tracks, oldest first.
        order = np.argsort(ids)[:150]
        frames.append((np.asarray(ids, np.int64)[order], np.asarray(xy).reshape(-1, 2)[order],
                       np.asarray(px).reshape(-1, 2)[order]))

    return SimData(imu_t, gyro_meas, accel_meas, imu_t[cam_idx], frames,
                   np.asarray(Rs), np.asarray(ps), np.asarray(vs), bg, ba)


def as_sequence(d: SimData):
    """Wrap simulated data in the same types the EuRoC loader returns."""
    from pathlib import Path

    from qvio.euroc import GroundTruth, Sequence
    from qvio.frontend import Frame

    seq = Sequence("sim", Path("."), d.imu_t, d.gyro, d.accel, d.cam_t, [], CAMERA, NOISE)
    # Pixel noise is redrawn every frame, which a real tracker does not do
    # on a still camera, so the stillness check gets the clean pixels.
    frames = [Frame(int(t), ids, xy, px) for t, (ids, xy, px) in zip(d.cam_t, d.frames)]
    gt = GroundTruth(d.imu_t, d.gt_p, np.asarray([so3.rot_to_quat(R) for R in d.gt_R]), d.gt_v, d.bg, d.ba)
    return seq, frames, gt


def write_euroc(d: SimData, folder) -> None:
    """Write simulated data in the EuRoC ASL layout (no images)."""
    from pathlib import Path

    import yaml

    root = Path(folder)
    for sub in ("imu0", "cam0", "state_groundtruth_estimate0"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    T_BC = np.eye(4)
    T_BC[:3, :3], T_BC[:3, 3] = CAMERA.R_BC, CAMERA.p_BC
    (root / "imu0" / "sensor.yaml").write_text(yaml.safe_dump({
        "T_BS": {"rows": 4, "cols": 4, "data": np.eye(4).ravel().tolist()},
        "gyroscope_noise_density": NOISE.gyro_noise, "accelerometer_noise_density": NOISE.accel_noise,
        "gyroscope_random_walk": NOISE.gyro_walk, "accelerometer_random_walk": NOISE.accel_walk,
    }))
    K = CAMERA.K
    (root / "cam0" / "sensor.yaml").write_text(yaml.safe_dump({
        "T_BS": {"rows": 4, "cols": 4, "data": T_BC.ravel().tolist()},
        "intrinsics": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0], "resolution": list(CAMERA.size),
    }))
    with open(root / "imu0" / "data.csv", "w") as f:
        f.write("#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n")
        for t, w, a in zip(d.imu_t, d.gyro, d.accel):
            f.write(f"{t},{w[0]:.12g},{w[1]:.12g},{w[2]:.12g},{a[0]:.12g},{a[1]:.12g},{a[2]:.12g}\n")
    with open(root / "cam0" / "data.csv", "w") as f:
        f.write("#timestamp [ns],filename\n")
        for t in d.cam_t:
            f.write(f"{t},{t}.png\n")
    with open(root / "state_groundtruth_estimate0" / "data.csv", "w") as f:
        f.write("#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,bw_x,bw_y,bw_z,ba_x,ba_y,ba_z\n")
        for t, R, p, v, bg, ba in zip(d.imu_t, d.gt_R, d.gt_p, d.gt_v, d.bg, d.ba):
            x, y, z, w = so3.rot_to_quat(R)
            vals = [*p, w, x, y, z, *v, *bg, *ba]
            f.write(f"{t}," + ",".join(f"{x:.12g}" for x in vals) + "\n")
