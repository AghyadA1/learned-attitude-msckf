"""Running the filter over a EuRoC sequence, from a cold start.

The filter never sees ground truth: it waits for the first stretch of
sensor data where the drone is still, takes gravity and the gyro bias from
that stretch, and starts there. Until the drone first moves it also applies
zero-velocity updates, which is what OpenVINS does with ``zupt_only_at_beginning``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import so3
from .euroc import Sequence
from .frontend import Frame, KLTTracker, TrackerConfig
from .msckf import MSCKF, FilterConfig
from .recovery import RecoveryConfig, RecoveryWindow


@dataclass(frozen=True)
class StaticConfig:
    window_s: float = 1.0
    lowpass_samples: int = 20          # 0.1 s moving average, removes motor vibration
    # Thresholds from the takeoff detector in the original project.
    max_gyro_std: float = 0.025        # rad/s, low-passed
    max_accel_std: float = 0.18        # m/s^2, low-passed
    max_gravity_error: float = 0.45    # m/s^2
    max_flow_px: float = 0.5           # median feature motion between frames


@dataclass
class StaticStart:
    frame: int        # first camera frame the filter processes
    t: int
    R: np.ndarray
    bg: np.ndarray


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    out = np.empty_like(x)
    out[n - 1:] = (c[n:] - c[:-n]) / n
    out[: n - 1] = c[1:n] / np.arange(1, n)[:, None]   # causal warm-up
    return out


def median_flow(a: Frame, b: Frame) -> float:
    common, ia, ib = np.intersect1d(a.ids, b.ids, return_indices=True)
    if len(common) < 10:
        return np.inf
    return float(np.median(np.linalg.norm(a.px[ia] - b.px[ib], axis=1)))


def find_static_start(seq: Sequence, frames: list[Frame], cfg: StaticConfig = StaticConfig()) -> StaticStart:
    """First camera frame that closes a full window of stillness."""
    gyro_lp = _moving_average(seq.gyro, cfg.lowpass_samples)
    accel_lp = _moving_average(seq.accel, cfg.lowpass_samples)
    flows = np.r_[np.inf, [median_flow(frames[k - 1], frames[k]) for k in range(1, len(frames))]]
    window_ns = int(cfg.window_s * 1e9)
    for k, frame in enumerate(frames):
        t1 = frame.t
        t0 = t1 - window_ns
        if t0 < seq.imu_t[0]:
            continue
        i0, i1 = np.searchsorted(seq.imu_t, [t0, t1])
        gyro_std = np.sqrt(np.mean(np.var(gyro_lp[i0:i1], axis=0)))
        accel_std = np.sqrt(np.mean(np.var(accel_lp[i0:i1], axis=0)))
        accel_mean = seq.accel[i0:i1].mean(axis=0)
        gravity_error = abs(np.linalg.norm(accel_mean) - 9.81)
        in_window = [j for j in range(k + 1) if frames[j].t > t0]
        flow = np.max(flows[in_window[1:]]) if len(in_window) > 1 else np.inf
        if (gyro_std < cfg.max_gyro_std and accel_std < cfg.max_accel_std
                and gravity_error < cfg.max_gravity_error and flow < cfg.max_flow_px):
            return StaticStart(k, int(t1), so3.gravity_aligned_rotation(accel_mean),
                               seq.gyro[i0:i1].mean(axis=0))
    raise RuntimeError(f"{seq.name}: no stationary interval found to initialise from")


def _imu_samples(seq: Sequence, t0: int, t1: int):
    """IMU samples covering [t0, t1], interpolated at both ends if needed."""
    i0 = int(np.searchsorted(seq.imu_t, t0, side="right"))
    i1 = int(np.searchsorted(seq.imu_t, t1, side="left"))
    t = [t0, *seq.imu_t[i0:i1].tolist(), t1]
    gyro = np.vstack([_interp(seq.imu_t, seq.gyro, t0), seq.gyro[i0:i1], _interp(seq.imu_t, seq.gyro, t1)])
    accel = np.vstack([_interp(seq.imu_t, seq.accel, t0), seq.accel[i0:i1], _interp(seq.imu_t, seq.accel, t1)])
    return np.asarray(t, np.int64), gyro, accel


def _interp(ts: np.ndarray, x: np.ndarray, t: int) -> np.ndarray:
    j = int(np.searchsorted(ts, t))
    if j < len(ts) and ts[j] == t:
        return x[j]
    j = min(max(j, 1), len(ts) - 1)
    a = (t - ts[j - 1]) / (ts[j] - ts[j - 1])
    return (1 - a) * x[j - 1] + a * x[j]


def cached_frames(seq: Sequence, cache_path=None, config: TrackerConfig = TrackerConfig(), progress=None):
    from pathlib import Path

    from .frontend import load_tracks, save_tracks, track_sequence

    if cache_path is not None and Path(cache_path).is_file():
        return load_tracks(cache_path)
    frames = track_sequence(seq, config, progress)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        save_tracks(cache_path, frames)
    return frames


def outage_frames(seq: Sequence, outages: list[tuple[int, int]],
                  config: TrackerConfig = TrackerConfig(), progress=None) -> list[Frame]:
    """Re-run the tracker on the images with the camera switched off.

    Hidden frames come back empty and the tracker restarts after each gap,
    so no track ever crosses an outage.
    """
    hidden = hidden_frames(seq, outages)
    tracker = KLTTracker(seq.camera, config)
    frames = []
    for k, t in enumerate(seq.cam_t):
        if hidden[k]:
            if k == 0 or not hidden[k - 1]:
                tracker.reset()
            frames.append(Frame(int(t), np.empty(0, np.int64), np.empty((0, 2)), np.empty((0, 2))))
        else:
            frames.append(tracker.track(seq.image(k), int(t)))
        if progress is not None:
            progress(k, len(seq.cam_t))
    return frames


def _velocity_prediction(f: MSCKF, seq: Sequence, t: int, model):
    i1 = int(np.searchsorted(seq.imu_t, t, side="right"))
    i0 = i1 - model.window
    if i0 < 0:
        return None, None
    return model.predict(seq.imu_t[i0:i1], seq.gyro[i0:i1], seq.accel[i0:i1], f.R, f.bg, f.ba)


def hidden_frames(seq: Sequence, outages: list[tuple[int, int]]) -> np.ndarray:
    hidden = np.zeros(len(seq.cam_t), bool)
    for a, b in outages:
        hidden |= (seq.cam_t >= a) & (seq.cam_t < b)
    return hidden


def run(seq: Sequence, frames: list[Frame], config: FilterConfig = FilterConfig(),
        gyro_model=None, outages: list[tuple[int, int]] | None = None,
        start: StaticStart | None = None, velocity_model=None, velocity_every: int = 20,
        recovery: RecoveryConfig | None = RecoveryConfig(), progress=None) -> dict:
    """Run the filter and return the estimated states at every camera time.

    With ``outages`` the frames must come from :func:`outage_frames`, and
    ``start`` from the uninterrupted tracks, since the camera is hidden only
    after the filter has started. A ``velocity_model`` (see
    ``qvio.velocity``) is applied every ``velocity_every`` frames while the
    camera is off; 20 frames is the network's 1 s window, so no two
    measurements share IMU samples and their errors are close to
    independent, as the update assumes (every 10 frames counted each error
    twice and made the filter too sure of its velocity). After each outage, ``recovery`` (see ``qvio.recovery``)
    re-estimates the velocity from the first images if the filter rejects
    most of their features; None turns that off.
    """
    start = start or find_static_start(seq, frames)
    f = MSCKF(seq.camera, seq.noise, config, gyro_model)
    f.initialize(start.t, start.R, start.bg)
    hidden = hidden_frames(seq, outages or [])

    still = True
    previous = frames[start.frame]
    out = {k: [] for k in ("t", "p", "q", "v", "bg", "ba", "P", "hidden", "accepted")}
    recovered = []
    n = len(seq.cam_t)
    since_velocity = 0
    window = None
    for k in range(start.frame, n):
        t = int(seq.cam_t[k])
        if t > f.t:
            samples = _imu_samples(seq, f.t, t)
            if window is not None:
                window.add_imu(*samples, f.R, f.bg, f.ba)
            f.propagate(*samples)
        if hidden[k] and not hidden[k - 1]:
            f.drop_tracks()
            since_velocity = 0
            window = None
        if recovery is not None and k > 0 and hidden[k - 1] and not hidden[k]:
            window = RecoveryWindow(t)
        if hidden[k] and velocity_model is not None:
            since_velocity += 1
            if since_velocity == velocity_every:
                since_velocity = 0
                v, var = _velocity_prediction(f, seq, t, velocity_model)
                if v is not None:
                    f.body_velocity_update(v, var)
        if window is not None:
            window.add_frame(t, f.R, seq.camera, frames[k].ids, frames[k].xy)
            if len(window) == recovery.frames:
                if window.accepted <= recovery.max_accepted:
                    v = window.velocity(recovery)
                    if v is not None:
                        f.reset_velocity(v, recovery.sigma_m_s)
                        recovered.append(t)
                window = None
        if hidden[k]:
            accepted = 0
        else:
            stats = f.process_frame(t, frames[k].ids, frames[k].xy)
            accepted = stats.accepted
            if window is not None:
                window.accepted += accepted
            if still:
                still = median_flow(previous, frames[k]) < 0.5
                if still:
                    f.zero_velocity_update()
            previous = frames[k]
        snap = f.snapshot()
        for key in ("t", "p", "q", "v", "bg", "ba", "P"):
            out[key].append(snap[key])
        out["hidden"].append(bool(hidden[k]))
        out["accepted"].append(accepted)
        if progress is not None:
            progress(k, n)
    result = {k: np.asarray(v) for k, v in out.items()}
    result["start_t"] = start.t
    result["recovered_t"] = np.asarray(recovered, np.int64)
    return result
