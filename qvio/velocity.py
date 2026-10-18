"""A GRU that estimates the drone's velocity from the IMU alone.

The attitude GRU in ``qvio.gru`` can only make the gyro a little better, and
the gyro is not what makes the position drift when the camera drops out.
Over a few seconds without vision the drift comes from the velocity: small
tilt and accelerometer bias errors integrate into it, and nothing in the
inertial equations pulls it back.

This network follows AirIO (Qiu et al., 2025): it reads the last second or
two of bias-corrected IMU samples in the body frame, together with the
direction of gravity in that frame, and predicts the body-frame velocity and
its variance. On a multirotor that is learnable because rotor drag shows up
in the accelerometer as a force against the velocity (Leishman et al.,
2014), and because the network sees how the drone has been accelerating.
The prediction does not depend on the filter's velocity, so it bounds the
velocity error instead of accumulating it. The filter uses it as a
measurement of R^T v while the camera is off.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import so3

UP = np.array([0.0, 0.0, 1.0])


def build_model(hidden: int = 64, pool: int = 4, gravity_per_sample: bool = False):
    import torch
    from torch import nn

    class VelocityGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool1d(pool) if pool > 1 else None
            self.gru = nn.GRU(9 if gravity_per_sample else 6, hidden, num_layers=2,
                              batch_first=True, dropout=0.1)
            self.head = nn.Sequential(nn.Linear(hidden + 3, 64), nn.GELU(), nn.Linear(64, 6))

        def forward(self, x, gravity):
            if self.pool is not None:
                x = self.pool(x.transpose(1, 2)).transpose(1, 2)
            _, h = self.gru(x)
            out = self.head(torch.cat([h[-1], gravity], dim=-1))
            return out[:, :3], out[:, 3:].clamp(-12.0, 4.0)

    return VelocityGRU()


def gravity_in_body(R: np.ndarray, gyro: np.ndarray, imu_t: np.ndarray) -> np.ndarray:
    """Up direction in the body frame at every sample of a window.

    ``R`` is the attitude at the last sample; earlier attitudes are found by
    integrating the bias-corrected gyro backwards from there.
    """
    out = np.empty((len(gyro), 3))
    Ri = np.asarray(R, float)
    out[-1] = Ri.T @ UP
    for i in range(len(gyro) - 1, 0, -1):
        dt = (imu_t[i] - imu_t[i - 1]) * 1e-9
        Ri = Ri @ so3.exp(-0.5 * (gyro[i] + gyro[i - 1]) * dt)
        out[i - 1] = Ri.T @ UP
    return out


class VelocityModel:
    """Wraps a checkpoint for use by ``pipeline.run(velocity_model=...)``."""

    def __init__(self, checkpoint: str | Path):
        import torch

        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.window = int(ck["window"])
        self.gravity_per_sample = bool(ck.get("gravity_per_sample", False))
        # A checkpoint holds one network or several trained from different
        # seeds; an ensemble averages them (Lakshminarayanan et al., 2017).
        self.models = []
        for state in ck.get("members") or [ck["model_state_dict"]]:
            model = build_model(int(ck["hidden"]), int(ck["pool"]), self.gravity_per_sample)
            model.load_state_dict(state)
            self.models.append(model.eval())
        self.mean = np.asarray(ck["input_mean"], np.float32)
        self.std = np.asarray(ck["input_std"], np.float32)
        # Measured on the validation sequence by train_velocity.py calibrate.
        self.variance_scale = np.asarray(ck.get("variance_scale", np.ones(3)), float)
        self.torch = torch

    def predict(self, imu_t, gyro, accel, R, bg, ba):
        """Body-frame velocity and its variance from one window of raw IMU samples."""
        torch = self.torch
        g = np.asarray(gyro, float) - bg
        x = [g, np.asarray(accel, float) - ba]
        if self.gravity_per_sample:
            x.append(gravity_in_body(R, g, imu_t))
        x = ((np.concatenate(x, axis=-1) - self.mean) / self.std).astype(np.float32)
        up = (np.asarray(R, float).T @ UP).astype(np.float32)
        with torch.inference_mode():
            out = [m(torch.from_numpy(x[None]), torch.from_numpy(up[None])) for m in self.models]
        v = np.array([o[0].numpy()[0] for o in out], float)
        var = np.exp(np.array([o[1].numpy()[0] for o in out], float))
        # Mixture of the members: mean of their variances plus their spread.
        return v.mean(axis=0), (var.mean(axis=0) + v.var(axis=0)) * self.variance_scale


class SpeedPrior:
    """Control for the velocity GRU: "the drone is not moving", with the spread
    of the training velocities as its uncertainty. It tells the filter as
    much about the speed as a network that learned nothing but the typical
    speed would."""

    window = 1

    def __init__(self, checkpoint: str | Path):
        import torch

        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.variance = np.asarray(ck["training_velocity_rms"], float) ** 2

    def predict(self, imu_t, gyro, accel, R, bg, ba):
        return np.zeros(3), self.variance
