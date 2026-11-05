"""Position error over time with repeated camera outages, for the README.

  python scripts/plot_outage_error.py --euroc DATA      # -> results/vio/outage_error_over_time.png

Two flights, each comparing the plain filter with one of the two fixes:
V2_03 with 5 s outages, without and with the velocity reset after each gap
(runs/no_recovery/msckf_outage5s against runs/msckf_outage5s), and V1_03 with
10 s outages, with and without the velocity GRU trained on the machine hall
only (runs/cross_room). The estimate is aligned to ground truth in yaw and
position on the camera-on stretch before the first outage, so the error
starts near zero and its growth is what the outages cause.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qvio import euroc, metrics  # noqa: E402

PANELS = (
    ("V2_03_difficult", "Test flight V2_03, camera off 5 s out of every 10 s",
     (("no_recovery/msckf_outage5s", "standard filter"),
      ("msckf_outage5s", "with velocity recovery"))),
    ("V1_03_difficult", "Flight V1_03, camera off 10 s out of every 20 s, network trained in another room",
     (("cross_room/msckf_outage10s", "standard filter"),
      ("cross_room/gru-velocity_outage10s", "with velocity GRU"))),
)
COLORS = ("#4C72B0", "#DD5144")


def error_over_time(run: dict, gt: euroc.GroundTruth):
    keep, p_gt, _, _ = metrics.interpolate_groundtruth(gt, run["t"])
    t, p = run["t"][keep], run["p"][keep]
    before = t < run["outages"][0, 0]
    Rz, tz = metrics.align_yaw_translation(p[before], p_gt[before])
    err = np.linalg.norm(p @ Rz.T + tz - p_gt, axis=1)
    return (t - t[0]) * 1e-9, err, (run["outages"] - t[0]) * 1e-9


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--euroc", required=True)
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=str(ROOT / "results/vio/outage_error_over_time.png"))
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(10, 6.6), sharey=True)
    for ax, (name, title, variants) in zip(axes, PANELS):
        gt = euroc.load_groundtruth(args.euroc, name)
        for k, ((folder, label), color) in enumerate(zip(variants, COLORS)):
            run = dict(np.load(Path(args.runs) / folder / f"{name}.npz"))
            t, err, outages = error_over_time(run, gt)
            if k == 0:
                for a, b in outages:
                    ax.axvspan(a, b, color="0.88", lw=0)
            ax.plot(t, err, color=color, lw=1.8, label=f"{label} (off by {err[-1]:.1f} m at the end)")
        ax.set_yscale("log")
        ax.set_ylim(0.01, 300)
        ax.set_xlim(0, t[-1])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} m"))
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel("position error")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", fontsize=9.5, frameon=False)
    axes[-1].set_xlabel("time [s]   (grey bands: camera off)")
    fig.suptitle("Position error when the camera keeps dropping out", fontsize=13, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
