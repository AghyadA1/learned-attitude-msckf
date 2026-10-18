"""Reading EuRoC MAV sequences in the ASL folder format.

Sensor data and ground truth are loaded by separate functions so that the
estimator code never touches ground truth by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

SEQUENCES = (
    "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult",
    "V1_01_easy", "V1_02_medium", "V1_03_difficult",
    "V2_01_easy", "V2_02_medium", "V2_03_difficult",
)
TRAIN = ("MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult",
         "V1_01_easy", "V1_02_medium", "V1_03_difficult")
VALIDATION = ("MH_05_difficult",)
TEST = ("V2_01_easy", "V2_02_medium", "V2_03_difficult")


@dataclass(frozen=True)
class Camera:
    K: np.ndarray            # 3x3 pinhole intrinsics
    dist: np.ndarray         # radial-tangential k1 k2 p1 p2
    size: tuple[int, int]    # (width, height)
    R_BC: np.ndarray         # camera -> body (IMU) rotation
    p_BC: np.ndarray         # camera origin in the body frame

    @property
    def focal(self) -> float:
        return float(np.sqrt(self.K[0, 0] * self.K[1, 1]))


@dataclass(frozen=True)
class ImuNoise:
    gyro_noise: float        # rad/s/sqrt(Hz)
    accel_noise: float       # m/s^2/sqrt(Hz)
    gyro_walk: float         # rad/s^2/sqrt(Hz)
    accel_walk: float        # m/s^3/sqrt(Hz)


@dataclass
class Sequence:
    name: str
    path: Path
    imu_t: np.ndarray        # int64 ns
    gyro: np.ndarray         # rad/s, IMU frame
    accel: np.ndarray        # m/s^2, IMU frame
    cam_t: np.ndarray        # int64 ns
    cam_files: list[Path]
    camera: Camera
    noise: ImuNoise

    def image(self, index: int) -> np.ndarray:
        import cv2

        image = cv2.imread(str(self.cam_files[index]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(self.cam_files[index])
        return image


@dataclass
class GroundTruth:
    t: np.ndarray            # int64 ns
    p: np.ndarray            # world position of the IMU
    q: np.ndarray            # R_WB as xyzw
    v: np.ndarray
    bg: np.ndarray
    ba: np.ndarray


def sequence_dir(root: str | Path, name: str) -> Path:
    """Accept both ``root/NAME/mav0`` (official zip) and ``root/NAME``."""
    base = Path(root).expanduser() / name
    return base / "mav0" if (base / "mav0").is_dir() else base


def _read_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.genfromtxt(path, delimiter=",", comments="#", dtype=str)
    raw = np.atleast_2d(raw)
    t = raw[:, 0].astype(np.int64)
    return t, raw[:, 1:]


def _transform(path: Path) -> np.ndarray:
    with path.open() as f:
        data = yaml.safe_load(f)
    return np.asarray(data["T_BS"]["data"], float).reshape(4, 4)


def load_sequence(root: str | Path, name: str) -> Sequence:
    path = sequence_dir(root, name)
    if not (path / "imu0" / "data.csv").is_file():
        raise FileNotFoundError(f"No EuRoC sequence at {path}")

    imu_t, imu = _read_csv(path / "imu0" / "data.csv")
    imu = imu.astype(float)
    cam_t, files = _read_csv(path / "cam0" / "data.csv")

    with (path / "cam0" / "sensor.yaml").open() as f:
        cam_yaml = yaml.safe_load(f)
    with (path / "imu0" / "sensor.yaml").open() as f:
        imu_yaml = yaml.safe_load(f)

    T_BI = _transform(path / "imu0" / "sensor.yaml")
    if not np.allclose(T_BI, np.eye(4)):
        raise ValueError("Expected imu0 to define the body frame (T_BS = I)")
    T_BC = _transform(path / "cam0" / "sensor.yaml")

    fu, fv, cu, cv = cam_yaml["intrinsics"]
    camera = Camera(
        K=np.array([[fu, 0.0, cu], [0.0, fv, cv], [0.0, 0.0, 1.0]]),
        dist=np.asarray(cam_yaml["distortion_coefficients"], float),
        size=tuple(int(v) for v in cam_yaml["resolution"]),
        R_BC=T_BC[:3, :3],
        p_BC=T_BC[:3, 3],
    )
    noise = ImuNoise(
        gyro_noise=float(imu_yaml["gyroscope_noise_density"]),
        accel_noise=float(imu_yaml["accelerometer_noise_density"]),
        gyro_walk=float(imu_yaml["gyroscope_random_walk"]),
        accel_walk=float(imu_yaml["accelerometer_random_walk"]),
    )
    return Sequence(
        name=name,
        path=path,
        imu_t=imu_t,
        gyro=imu[:, 0:3],
        accel=imu[:, 3:6],
        cam_t=cam_t,
        cam_files=[path / "cam0" / "data" / str(f[0]).strip() for f in files],
        camera=camera,
        noise=noise,
    )


def load_groundtruth(root: str | Path, name: str) -> GroundTruth:
    path = sequence_dir(root, name) / "state_groundtruth_estimate0" / "data.csv"
    t, values = _read_csv(path)
    values = values.astype(float)
    q_wxyz = values[:, 3:7]
    return GroundTruth(
        t=t,
        p=values[:, 0:3],
        q=np.column_stack([q_wxyz[:, 1:4], q_wxyz[:, 0]]),
        v=values[:, 7:10],
        bg=values[:, 10:13],
        ba=values[:, 13:16],
    )


def imu_between(seq: Sequence, t0: int, t1: int) -> slice:
    """Index range of IMU samples with t0 <= t < t1."""
    i0 = int(np.searchsorted(seq.imu_t, t0, side="left"))
    i1 = int(np.searchsorted(seq.imu_t, t1, side="left"))
    return slice(i0, i1)
