"""Run the MSCKF on EuRoC sequences, with or without the learned gyro model.

Variants
  msckf          plain gyro, noise densities from sensor.yaml
  msckf-cal      plain gyro, per-window noise measured on MH_05 (fair baseline
                 for the GRU variants, which also get MH_05-measured noise)
  gru-preflight  the original network (trained with the preflight bias)
  gru-filter     the network fine-tuned with ground-truth bias + variance head

With --outage D the camera is switched off for D seconds every 10 s, starting
15 s after initialisation; the tracker restarts after each gap.

  python scripts/run_vio.py --euroc DATA --variant msckf gru-filter --jobs 4
  python scripts/run_vio.py --euroc DATA --variant msckf-cal gru-filter --outage 2 --sequences V2_01_easy
"""

from __future__ import annotations

import argparse
import json
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

MODELS = ROOT / "models"
VARIANTS = {
    "msckf": None,
    "msckf-cal": MODELS / "gyro_calibrated.json",
    "gru-preflight": MODELS / "gru_preflight_cal.pt",
    "gru-filter": MODELS / "gru_filter_cal.pt",
}


class CalibratedGyro:
    """Plain gyro with a measured per-window noise; no rate correction."""

    def __init__(self, path: Path):
        self.window_variance = np.asarray(json.loads(Path(path).read_text())["window_variance_rad2"])

    def correction(self, t, gyro, accel, bias):
        return np.zeros(3), self.window_variance


def gyro_model(variant: str):
    path = VARIANTS[variant]
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run the calibration step first (see README)")
    if path.suffix == ".json":
        return CalibratedGyro(path)
    import torch

    torch.set_num_threads(1)
    from qvio.gru import GRUGyroModel

    return GRUGyroModel(path)


def outage_schedule(seq, start_t: int, duration_s: float, first_s=15.0, period_s=10.0, margin_s=2.0):
    end = int(seq.cam_t[-1])
    outages = []
    a = start_t + int(first_s * 1e9)
    while a + int((duration_s + margin_s) * 1e9) < end:
        outages.append((a, a + int(duration_s * 1e9)))
        a += int(period_s * 1e9)
    return outages


def run_one(root: str, name: str, variant: str, outage_s: float, out_dir: str) -> str:
    import cv2

    cv2.setNumThreads(1)
    seq = euroc.load_sequence(root, name)
    frames = pipeline.cached_frames(seq, ROOT / "cache" / "tracks" / f"{name}.npz")
    outages = []
    if outage_s > 0:
        start = pipeline.find_static_start(seq, frames)
        outages = outage_schedule(seq, start.t, outage_s)
    model = gyro_model(variant)
    tic = time.time()
    result = pipeline.run(seq, frames, gyro_model=model, outages=outages)
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
    parser.add_argument("--force", action="store_true", help="re-run even if the output file exists")
    args = parser.parse_args()

    # Build the track caches first, one process per sequence.
    with ProcessPoolExecutor(args.jobs) as pool:
        list(pool.map(_cache_tracks, [args.euroc] * len(args.sequences), args.sequences))

    tasks = [
        (args.euroc, name, variant, d, str(Path(args.out) / (variant if d == 0 else f"{variant}_outage{d:g}s")))
        for variant in args.variant for d in args.outage for name in args.sequences
    ]
    if not args.force:
        tasks = [t for t in tasks if not (Path(t[4]) / f"{t[1]}.npz").is_file()]
    failed = []
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(run_one, *task): task for task in tasks}
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


def _cache_tracks(root: str, name: str) -> None:
    path = ROOT / "cache" / "tracks" / f"{name}.npz"
    if not path.is_file():
        seq = euroc.load_sequence(root, name)
        tic = time.time()
        pipeline.cached_frames(seq, path)
        print(f"tracked {name} in {time.time() - tic:.0f}s", flush=True)


if __name__ == "__main__":
    main()
