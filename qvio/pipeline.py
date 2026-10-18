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


def run(seq: Sequence, frames: list[Frame], config: FilterConfig = FilterConfig(),
        gyro_model=None, outages: list[tuple[int, int]] | None = None,
        tracker_config: TrackerConfig = TrackerConfig(), progress=None) -> dict:
    """Run the filter and return the estimated states at every camera time.

    ``frames`` are the cached tracks. With ``outages`` the tracker is re-run
    on the images instead, skipping the hidden frames and restarting after
    each gap, so no track ever crosses an outage.
    """
    start = find_static_start(seq, frames)
    f = MSCKF(seq.camera, seq.noise, config, gyro_model)
    f.initialize(start.t, start.R, start.bg)

    outages = outages or []
    hidden = np.zeros(len(seq.cam_t), bool)
    for a, b in outages:
        hidden |= (seq.cam_t >= a) & (seq.cam_t < b)
    tracker = KLTTracker(seq.camera, tracker_config) if outages else None

    still = True
    previous = frames[start.frame]
    out = {k: [] for k in ("t", "p", "q", "v", "bg", "ba", "P", "hidden", "accepted")}
    n = len(seq.cam_t)
    for k in range(start.frame, n):
        t = int(seq.cam_t[k])
        if t > f.t:
            f.propagate(*_imu_samples(seq, f.t, t))
        if hidden[k]:
            if tracker is not None and not hidden[k - 1]:
                tracker.reset()
                f.drop_tracks()
            accepted = 0
        else:
            frame = tracker.track(seq.image(k), t) if tracker is not None else frames[k]
            stats = f.process_frame(t, frame.ids, frame.xy)
            accepted = stats.accepted
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
    return result
