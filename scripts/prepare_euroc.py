"""Unpack the EuRoC MAV archives into the layout the scripts expect, or check a copy.

EuRoC is hosted on the ETH Research Collection (https://doi.org/10.3929/ethz-b-000690084).
Download the machine hall and Vicon room archives into one folder by hand,
then:

  python scripts/prepare_euroc.py unpack --zips ~/Downloads/euroc --out data/euroc
  python scripts/prepare_euroc.py check data/euroc

The result is DATA/<sequence>/mav0/{cam0,imu0,state_groundtruth_estimate0}.
Archives inside archives (one zip per sequence) are unpacked as well. If you
already have the data in the ASL format, just point --euroc at it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qvio.euroc import SEQUENCES, sequence_dir  # noqa: E402

NEEDED = ("cam0/data.csv", "cam0/sensor.yaml", "imu0/data.csv", "imu0/sensor.yaml",
          "state_groundtruth_estimate0/data.csv")


def unpack(zips: Path, out: Path) -> None:
    work = out / "_unpacked"
    pending = sorted(zips.expanduser().glob("*.zip"))
    if not pending:
        sys.exit(f"no .zip files in {zips}")
    while pending:
        archive = pending.pop()
        target = work / archive.stem
        print(f"unpacking {archive.name}")
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
        pending += sorted(target.rglob("*.zip"))

    # Whatever folder holds imu0/ is the ASL "mav0" folder, named or not.
    for imu in list(work.rglob("imu0/data.csv")):
        mav0 = imu.parent.parent
        name = next((s for s in SEQUENCES if s in str(mav0)), None)
        if name is None or (out / name / "mav0").exists():
            continue
        (out / name).mkdir(parents=True, exist_ok=True)
        shutil.move(str(mav0), out / name / "mav0")
        print(f"  {name}")
    shutil.rmtree(work)
    check(out)


def check(root: Path) -> None:
    missing = []
    for name in SEQUENCES:
        base = sequence_dir(root, name)
        gaps = [f for f in NEEDED if not (base / f).is_file()]
        images = len(list((base / "cam0" / "data").glob("*.png"))) if (base / "cam0" / "data").is_dir() else 0
        status = "ok" if not gaps and images else "missing " + ", ".join(gaps or ["cam0 images"])
        print(f"{name:<16s} {images:5d} images  {status}")
        if status != "ok":
            missing.append(name)
    if missing:
        sys.exit(f"{len(missing)} sequence(s) incomplete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="job", required=True)
    u = sub.add_parser("unpack")
    u.add_argument("--zips", required=True, type=Path)
    u.add_argument("--out", required=True, type=Path)
    c = sub.add_parser("check")
    c.add_argument("root", type=Path)
    args = parser.parse_args()
    unpack(args.zips, args.out) if args.job == "unpack" else check(args.root.expanduser())


if __name__ == "__main__":
    main()
