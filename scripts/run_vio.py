"""Run the MSCKF on EuRoC sequences, with or without the learned gyro model.

Variants
  msckf            plain gyro, noise densities from sensor.yaml
  msckf-gru-noise  plain gyro, but with the per-window noise measured for the
                   fine-tuned GRU on MH_05: the control for the GRU variants
  gru-preflight    the original network (trained with the preflight bias)
  gru-filter       the network fine-tuned with ground-truth bias + variance head
  gru-velocity     plain gyro, and the velocity GRU while the camera is off
  gru-both         gru-filter and the velocity GRU together
  msckf-speed-prior  control for the velocity GRU: a zero-velocity measurement
                   with the spread of the training velocities, while the camera is off

With --outage D the camera is switched off for D seconds every max(10, 2D) s,
starting 15 s after initialisation; the tracker restarts after each gap. The tracks
for each outage length are cached too, so only the first run needs images. After
each gap the velocity is re-estimated from the first second of images if the
filter rejects most of their features (qvio/recovery.py); --no-recovery turns
that off.

  python scripts/run_vio.py --euroc DATA --variant msckf gru-filter --jobs 4
  python scripts/run_vio.py --euroc DATA --variant msckf-gru-noise gru-filter --outage 2 --sequences V2_01_easy

--velocity-model swaps in another velocity GRU checkpoint, for example one
trained on the machine hall only and run on the Vicon room.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# One BLAS thread per worker; with several jobs the default oversubscribes
# the CPU and every run gets slower.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qvio import euroc, pipeline  # noqa: E402
from qvio.gru import WINDOW  # noqa: E402
from qvio.recovery import RecoveryConfig  # noqa: E402

MODELS = ROOT / "models"
VARIANTS = {
    "msckf": None,
    "msckf-gru-noise": MODELS / "gru_filter_cal.pt",
    "gru-preflight": MODELS / "gru_preflight_cal.pt",
    "gru-filter": MODELS / "gru_filter_cal.pt",
    "msckf-speed-prior": None,
    "gru-velocity": None,
    "gru-both": MODELS / "gru_filter_cal.pt",
}
VELOCITY_MODEL = MODELS / "velocity_gru.pt"


class GRUNoiseOnly:
    """Plain gyro with the noise measured for a GRU checkpoint; no rate correction.

    It separates the two things a GRU variant changes: the rotation over each
    interval and the gyro noise the filter assumes.
    """

    def __init__(self, checkpoint: Path):
        import torch

        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.window_variance = np.asarray(ck["window_variance"], float)

    def correction(self, t, gyro, accel, bias):
        # Same intervals as the GRU uses; the others keep the sensor.yaml noise.
        if len(t) != WINDOW + 1 or abs((t[-1] - t[0]) * 1e-9 - 0.05) > 0.002:
            return None
        return np.zeros(3), self.window_variance


def gyro_model(variant: str):
    path = VARIANTS[variant]
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run the calibration step first (see README)")
    if variant == "msckf-gru-noise":
        return GRUNoiseOnly(path)
    import torch

    torch.set_num_threads(1)
    from qvio.gru import GRUGyroModel

    return GRUGyroModel(path)


def velocity_model(variant: str, path: Path = VELOCITY_MODEL):
    """What the filter uses for the velocity while the camera is off, if anything."""
    if variant not in ("gru-velocity", "gru-both", "msckf-speed-prior"):
        return None
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run scripts/train_velocity.py train first")
    import torch

    torch.set_num_threads(1)
    from qvio.velocity import SpeedPrior, VelocityModel

    return SpeedPrior(path) if variant == "msckf-speed-prior" else VelocityModel(path)


def outage_schedule(seq, start_t: int, duration_s: float, first_s=15.0, period_s=None, margin_s=2.0):
    """Outages every 10 s, or every 2D seconds for gaps longer than 5 s, so
    the camera is always back for at least as long as it was off."""
    period_s = period_s or max(10.0, 2.0 * duration_s)
    end = int(seq.cam_t[-1])
    outages = []
    a = start_t + int(first_s * 1e9)
    while a + int((duration_s + margin_s) * 1e9) < end:
        outages.append((a, a + int(duration_s * 1e9)))
        a += int(period_s * 1e9)
    return outages


def run_one(root: str, name: str, variant: str, outage_s: float, out_dir: str,
            velocity_checkpoint: str = str(VELOCITY_MODEL), recovery: bool = True,
            velocity_every: int = 20) -> str:
    import cv2

    cv2.setNumThreads(1)
    seq = euroc.load_sequence(root, name)
    frames = pipeline.cached_frames(seq, track_cache(name))
    start = pipeline.find_static_start(seq, frames)
    outages = []
    if outage_s > 0:
        outages = outage_schedule(seq, start.t, outage_s)
        frames = _outage_tracks(seq, outages, outage_s)
    model = gyro_model(variant)
    tic = time.time()
    result = pipeline.run(seq, frames, gyro_model=model, outages=outages, start=start,
                          velocity_model=velocity_model(variant, velocity_checkpoint),
                          velocity_every=velocity_every,
                          recovery=RecoveryConfig() if recovery else None)
    result["runtime_s"] = time.time() - tic
    result["outages"] = np.asarray(outages, np.int64).reshape(-1, 2)
    if model is not None and hasattr(model, "used"):
        result["gru_windows_used"] = model.used
        result["gru_windows_skipped"] = model.skipped
    path = Path(out_dir) / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    return f"{variant:>13s} {name:<16s} outage {outage_s:g}s  {result['runtime_s']:6.1f}s  -> {path}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--euroc", required=True)
    parser.add_argument("--variant", nargs="+", default=["msckf"], choices=list(VARIANTS))
    parser.add_argument("--sequences", nargs="+", default=list(euroc.SEQUENCES))
    parser.add_argument("--outage", type=float, nargs="+", default=[0.0], help="outage length(s) in seconds")
    parser.add_argument("--out", default=str(ROOT / "runs"))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--velocity-model", default=str(VELOCITY_MODEL), help="velocity GRU checkpoint")
    parser.add_argument("--velocity-every", type=int, default=20,
                        help="frames between velocity measurements while the camera is off (20 = once a second)")
    parser.add_argument("--no-recovery", action="store_true",
                        help="skip the velocity reset after each outage (qvio/recovery.py)")
    parser.add_argument("--force", action="store_true", help="re-run even if the output file exists")
    args = parser.parse_args()

    # Build the track caches first, one process per sequence and outage length.
    jobs = [(args.euroc, name, d) for name in args.sequences for d in sorted({0.0, *args.outage})]
    with ProcessPoolExecutor(args.jobs) as pool:
        list(pool.map(_cache_tracks, *zip(*jobs)))

    tasks = [
        (args.euroc, name, variant, d, str(Path(args.out) / (variant if d == 0 else f"{variant}_outage{d:g}s")))
        for variant in args.variant for d in args.outage for name in args.sequences
    ]
    if not args.force:
        tasks = [t for t in tasks if not (Path(t[4]) / f"{t[1]}.npz").is_file()]
    failed = []
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(run_one, *task, args.velocity_model, not args.no_recovery, args.velocity_every): task
                   for task in tasks}
        for future in as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception:
                # One bad run should not throw away the others.
                _, name, variant, outage, _ = futures[future]
                failed.append(f"{variant} {name} outage {outage:g}s")
                print(f"FAILED {failed[-1]}\n{traceback.format_exc()}", flush=True)
    if failed:
        print(f"{len(failed)} run(s) failed: " + "; ".join(failed))


def track_cache(name: str, outage_s: float = 0.0) -> Path:
    suffix = "" if outage_s == 0 else f"_outage{outage_s:g}s"
    return ROOT / "cache" / "tracks" / f"{name}{suffix}.npz"


def _outage_tracks(seq, outages, outage_s: float):
    from qvio.frontend import load_tracks, save_tracks

    path = track_cache(seq.name, outage_s)
    if path.is_file():
        return load_tracks(path)
    frames = pipeline.outage_frames(seq, outages)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_tracks(path, frames)
    return frames


def _cache_tracks(root: str, name: str, outage_s: float) -> None:
    import cv2

    cv2.setNumThreads(1)
    if track_cache(name, outage_s).is_file():
        return
    seq = euroc.load_sequence(root, name)
    tic = time.time()
    frames = pipeline.cached_frames(seq, track_cache(name))
    if outage_s > 0:
        start = pipeline.find_static_start(seq, frames)
        _outage_tracks(seq, outage_schedule(seq, start.t, outage_s), outage_s)
    label = name if outage_s == 0 else f"{name} with {outage_s:g} s outages"
    print(f"tracked {label} in {time.time() - tic:.0f}s", flush=True)


if __name__ == "__main__":
    main()
