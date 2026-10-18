"""Find the still interval, motor start and takeoff in each EuRoC sequence.

Writes one row per sequence with the selected calibration interval and the
gyro bias estimated from it, and optionally an annotated video.

  python scripts/detect_takeoff.py --euroc DATA --out results/takeoff
  python scripts/detect_takeoff.py --euroc DATA --sequences V2_01_easy --video
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qvio import euroc, takeoff  # noqa: E402

COLORS = {  # BGR
    "INITIAL_STATIC": (200, 200, 200), "MOTORS_ON_GROUNDED": (0, 200, 255), "TAKEOFF_CANDIDATE": (0, 120, 255),
    "AIRBORNE": (80, 220, 80), "AIRBORNE_MOTION": (80, 220, 80), "LATER_STATIC": (220, 160, 60),
    "FREE_FALL_OR_LOW_SPECIFIC_FORCE": (60, 60, 230),
}


def write_video(seq, cal, table: pd.DataFrame, path: Path, max_seconds: float = 25.0) -> None:
    import cv2

    t0 = seq.imu_t[0]
    t_win = table["timestamp_ns"].to_numpy(np.int64)
    end = cal.takeoff_time_s + 8.0 if cal.takeoff_detected else max_seconds
    frames = [k for k, t in enumerate(seq.cam_t) if (t - t0) * 1e-9 <= min(end, max_seconds)]
    w, h = seq.camera.size
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h + 90))
    score = table["static_score"].to_numpy(float)
    vib = table["vibration_score"].to_numpy(float)
    last = max(int(np.searchsorted(t_win, seq.cam_t[frames[-1]])), 2)

    def px(j: int) -> int:
        return int(j / last * (w - 20)) + 10

    for k in frames:
        ts = (seq.cam_t[k] - t0) * 1e-9
        i = int(np.clip(np.searchsorted(t_win, seq.cam_t[k]), 0, len(t_win) - 1))
        state = table["state"].iloc[i]
        canvas = np.zeros((h + 90, w, 3), np.uint8)
        canvas[:h] = cv2.cvtColor(seq.image(k), cv2.COLOR_GRAY2BGR)
        label = state.replace("_", " ").lower()
        if cal.start_time_s <= ts <= cal.end_time_s:
            label = "calibrating gyro bias"
            cv2.rectangle(canvas, (2, 2), (w - 3, h - 3), (0, 200, 255), 3)
        if cal.takeoff_detected and 0 <= ts - cal.takeoff_time_s <= 0.75:
            label = "takeoff"
            cv2.rectangle(canvas, (2, 2), (w - 3, h - 3), (80, 220, 80), 5)
        cv2.putText(canvas, f"{seq.name}  t = {ts:5.2f} s", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, label, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS.get(state, (255, 255, 255)), 2)
        # Score strip under the image: stillness (white) and motor vibration (orange).
        for j in range(1, last):
            xa, xb = px(j - 1), px(j)
            cv2.line(canvas, (xa, h + 80 - int(70 * score[j - 1])), (xb, h + 80 - int(70 * score[j])), (255, 255, 255), 1)
            cv2.line(canvas, (xa, h + 80 - int(70 * vib[j - 1])), (xb, h + 80 - int(70 * vib[j])), (0, 160, 255), 1)
        cv2.line(canvas, (px(i), h + 5), (px(i), h + 85), (80, 220, 80), 1)
        out.write(canvas)
    out.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--euroc", required=True)
    parser.add_argument("--sequences", nargs="+", default=list(euroc.SEQUENCES))
    parser.add_argument("--out", default=str(ROOT / "results" / "takeoff"))
    parser.add_argument("--video", action="store_true", help="also write an annotated mp4 per sequence")
    parser.add_argument("--windows", action="store_true", help="also write the per-window scores")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in args.sequences:
        seq = euroc.load_sequence(args.euroc, name)
        cal, table, _ = takeoff.detect(seq)
        if args.windows:
            table.to_csv(out / f"{name}_windows.csv", index=False)
        # Ground truth only for scoring, after the estimate is made: median
        # distance to the EuRoC bias estimate over the whole flight.
        gt = euroc.load_groundtruth(args.euroc, name)
        bias_error = np.degrees(np.median(np.linalg.norm(gt.bg - cal.gyro_bias, axis=1)))
        rows.append({
            "sequence": name,
            "takeoff_detected": cal.takeoff_detected,
            "takeoff_s": round(cal.takeoff_time_s, 2),
            "motor_start_s": None if cal.motor_start_time_s is None else round(cal.motor_start_time_s, 2),
            "interval": cal.interval_type,
            "causal": cal.causal,
            "start_s": round(cal.start_time_s, 2),
            "end_s": round(cal.end_time_s, 2),
            **{f"bias_{a}_deg_s": np.degrees(cal.gyro_bias[i]) for i, a in enumerate("xyz")},
            "bias_error_deg_s": float(bias_error),
        })
        print(f"{name:<16s} takeoff {cal.takeoff_time_s:6.2f} s  {cal.interval_type:<26s} "
              f"{cal.start_time_s:5.1f}-{cal.end_time_s:5.1f} s  bias error {rows[-1]['bias_error_deg_s']:.3f} deg/s")
        if args.video:
            write_video(seq, cal, table, out / f"{name}_takeoff.mp4")
    pd.DataFrame(rows).to_csv(out / "takeoff_and_bias.csv", index=False)


if __name__ == "__main__":
    main()
