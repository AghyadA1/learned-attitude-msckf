"""Score every run in runs/ against ground truth and write the result tables.

  python scripts/evaluate.py --euroc DATA            # -> results/vio/*.csv, *.md, figures

With --openvins DIR the OpenVINS baseline exports (DIR/<sequence>/state_trajectory.csv,
see baselines/openvins) are scored by the same code, so all rows share one
alignment and one set of metric definitions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qvio import euroc, metrics  # noqa: E402

ORDER = ["openvins", "msckf", "msckf-cal", "gru-preflight", "gru-filter"]


def markdown(df: pd.DataFrame, index: bool = False, digits: int = 4) -> str:
    """Plain markdown table, so we do not need tabulate."""
    if index:
        df = df.reset_index()
    cells = [[f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in row] for row in df.itertuples(index=False)]
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in cells]
    return "\n".join(lines) + "\n"


def load_runs(runs: Path):
    if not runs.is_dir():
        return
    for folder in sorted(p for p in runs.iterdir() if p.is_dir()):
        variant, _, outage = folder.name.partition("_outage")
        for path in sorted(folder.glob("*.npz")):
            data = dict(np.load(path, allow_pickle=False))
            yield variant, float(outage.rstrip("s") or 0), path.stem, data


def load_openvins(folder: Path):
    for name in euroc.SEQUENCES:
        path = folder / name / "state_trajectory.csv"
        if not path.is_file():
            continue
        d = pd.read_csv(path)
        # OpenVINS keeps the JPL quaternion q_GtoI. Read as a Hamilton
        # quaternion, the same four numbers are world-from-body.
        q = d[["qx", "qy", "qz", "qw"]].to_numpy()
        t = d["timestamp_ns"].to_numpy(np.int64)
        yield "openvins", 0.0, name, {
            "t": t, "q": q, "p": d[["p_x_m", "p_y_m", "p_z_m"]].to_numpy(),
            "v": d[["v_x_m_s", "v_y_m_s", "v_z_m_s"]].to_numpy(),
            "start_t": t[0], "runtime_s": np.nan,
        }


def split_of(name: str) -> str:
    return "test" if name in euroc.TEST else "val" if name in euroc.VALIDATION else "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--euroc", required=True)
    parser.add_argument("--runs", default=str(ROOT / "runs"))
    parser.add_argument("--out", default=str(ROOT / "results" / "vio"))
    parser.add_argument("--openvins", help="folder with OpenVINS exports, one sub-folder per sequence")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = list(load_runs(Path(args.runs)))
    if args.openvins:
        runs += list(load_openvins(Path(args.openvins).expanduser()))
    gts, full, outages = {}, [], []
    for variant, outage, name, run in runs:
        gt = gts.setdefault(name, euroc.load_groundtruth(args.euroc, name))
        if outage == 0:
            row = metrics.evaluate_run(run, gt)
            row.update(variant=variant, sequence=name, split=split_of(name),
                       start_s=float((int(run["start_t"]) - int(gt.t[0])) * 1e-9),
                       runtime_s=float(run["runtime_s"]))
            full.append(row)
        else:
            for row in metrics.evaluate_outages(run, gt):
                row.update(variant=variant, sequence=name, split=split_of(name), outage_s=outage)
                outages.append(row)

    if full:
        df = pd.DataFrame(full)
        df.to_csv(out / "full_sequences.csv", index=False)
        cols = ["ate_rmse_m", "attitude_rmse_deg", "tilt_rmse_deg", "velocity_rmse_m_s",
                "rpe_1s_deg", "rpe_10s_m", "nees_attitude", "nees_position"]
        summary = df.groupby(["split", "variant"])[cols].mean().reset_index()
        summary["variant"] = pd.Categorical(summary["variant"], [v for v in ORDER if v in set(summary["variant"])])
        summary = summary.sort_values(["split", "variant"])
        summary.to_csv(out / "full_sequences_summary.csv", index=False)
        (out / "full_sequences_summary.md").write_text(markdown(summary))
        per_seq = df.pivot_table(index="sequence", columns="variant", values="ate_rmse_m")
        per_seq = per_seq[[v for v in ORDER if v in per_seq.columns]]
        per_seq = per_seq.reindex([s for s in euroc.SEQUENCES if s in per_seq.index])
        (out / "ate_per_sequence.md").write_text(markdown(per_seq, index=True, digits=3))
        print(summary.to_string(index=False))
        print(per_seq.to_string())
        plot_trajectories(args.euroc, Path(args.runs), out)

    if outages:
        df = pd.DataFrame(outages)
        df.to_csv(out / "outages.csv", index=False)
        stats = (df.groupby(["split", "outage_s", "variant"])
                 .agg(n=("translation_drift_m", "size"),
                      translation_median_m=("translation_drift_m", "median"),
                      translation_p90_m=("translation_drift_m", lambda x: np.percentile(x, 90)),
                      rotation_median_deg=("rotation_drift_deg", "median"),
                      tilt_after_median_deg=("tilt_after_deg", "median"))
                 .reset_index())
        stats.to_csv(out / "outages_summary.csv", index=False)
        (out / "outages_summary.md").write_text(markdown(stats))
        print(stats.to_string(index=False))
        plot_outages(df, out)


def plot_trajectories(root: str, runs: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in euroc.TEST if (runs / "msckf" / f"{n}.npz").is_file()]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.2))
    for ax, name in zip(np.atleast_1d(axes), names):
        gt = euroc.load_groundtruth(root, name)
        ax.plot(gt.p[:, 0], gt.p[:, 1], color="0.6", lw=2.5, label="ground truth")
        for variant, color in (("msckf", "tab:blue"), ("gru-filter", "tab:red")):
            path = runs / variant / f"{name}.npz"
            if not path.is_file():
                continue
            run = dict(np.load(path))
            keep, p_gt, _, _ = metrics.interpolate_groundtruth(gt, run["t"])
            Rz, tz = metrics.align_yaw_translation(run["p"][keep], p_gt)
            p = run["p"][keep] @ Rz.T + tz
            ax.plot(p[:, 0], p[:, 1], color=color, lw=1, label=variant)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
    np.atleast_1d(axes)[0].set_ylabel("y [m]")
    np.atleast_1d(axes)[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "trajectories_v2.png", dpi=130)
    plt.close(fig)


def plot_outages(df: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test = df[df["split"] == "test"]
    if test.empty:
        return
    variants = [v for v in ORDER if v in set(test["variant"])]
    durations = sorted(test["outage_s"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, column, label in ((axes[0], "translation_drift_m", "translation drift [m]"),
                              (axes[1], "rotation_drift_deg", "rotation drift [deg]")):
        width = 0.8 / len(variants)
        for i, variant in enumerate(variants):
            data = [test[(test["variant"] == variant) & (test["outage_s"] == d)][column] for d in durations]
            pos = np.arange(len(durations)) + (i - (len(variants) - 1) / 2) * width
            ax.boxplot(data, positions=pos, widths=width * 0.9, showfliers=False,
                       patch_artist=True, boxprops=dict(facecolor=f"C{i}", alpha=0.6))
            ax.plot([], [], color=f"C{i}", lw=6, alpha=0.6, label=variant)
        ax.set_xticks(range(len(durations)), [f"{d:g} s" for d in durations])
        ax.set_xlabel("camera outage")
        ax.set_ylabel(label)
    axes[0].legend(fontsize=8)
    fig.suptitle("Drift during camera outages, V2 sequences")
    fig.tight_layout()
    fig.savefig(out / "outages_v2.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
