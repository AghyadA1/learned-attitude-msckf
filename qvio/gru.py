"""The attitude GRU from the original project, used as a gyro model.

The network looks at the ten IMU samples of one camera interval (50 ms),
with the gyro already bias-corrected, and predicts a small rotation that is
applied after plain gyro integration:

    q_net = q_gyro(omega - b) * Exp(dtheta),   |dtheta| <= 0.15 rad per axis

Inside the filter the bias ``b`` is the filter's own current estimate, and
the predicted rotation is turned into a constant rate correction for the
interval. That keeps the gyro bias in the state, where vision can still
correct it. Earlier versions fed the network a fixed preflight bias and then
solved the rate against the filter bias, which cancelled the filter bias
out of the attitude entirely.

Checkpoints from the original project (``EstimatedBiasPriorResidualGRU``)
load directly. Retrained checkpoints add a second head with a per-axis log
variance, which the filter uses as the gyro noise for that interval.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import so3

WINDOW = 10


def integrate_rectangle(gyro: np.ndarray, duration_s: np.ndarray) -> np.ndarray:
    """Left-rectangle integration of [N, 10, 3] rates into quaternions.

    This is the prior the network was trained against, so it must stay a
    left-rectangle rule with dt = duration / 10 even though the filter uses
    the midpoint rule.
    """
    gyro = np.asarray(gyro, float)
    dt = np.asarray(duration_s, float).reshape(-1, 1) / gyro.shape[1]
    q = np.tile([0.0, 0.0, 0.0, 1.0], (len(gyro), 1))
    for i in range(gyro.shape[1]):
        q = so3.quat_mul(q, so3.rotvec_to_quat(gyro[:, i] * dt))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


def build_model(hidden: int = 96, max_residual: float = 0.15, with_variance: bool = False):
    import torch
    from torch import nn

    class AttitudeGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.max_residual = max_residual
            self.gru = nn.GRU(6, hidden, batch_first=True)
            self.fc = nn.Linear(hidden + 4, 3)
            self.fc_logvar = nn.Linear(hidden + 4, 3) if with_variance else None

        def forward(self, x, q_prior):
            _, h = self.gru(x)
            z = torch.cat([h[-1], q_prior], dim=-1)
            dtheta = self.max_residual * torch.tanh(self.fc(z))
            logvar = self.fc_logvar(z) if self.fc_logvar is not None else None
            return dtheta, logvar

    return AttitudeGRU()


class GRUGyroModel:
    """Wraps a checkpoint for use as ``MSCKF(gyro_model=...)``."""

    def __init__(self, checkpoint: str | Path, window_variance: np.ndarray | None = None):
        import torch

        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ck["model_state_dict"]
        arch = ck.get("architecture_config", {})
        self.model = build_model(
            hidden=int(arch.get("hidden_size", 96)),
            max_residual=float(arch.get("max_residual_rad", 0.15)),
            with_variance="fc_logvar.weight" in state,
        )
        self.model.load_state_dict(state)
        self.model.eval()
        self.mean = np.asarray(ck["input_mean"], np.float32)
        self.std = np.asarray(ck["input_std"], np.float32)
        if window_variance is None and "window_variance" in ck:
            window_variance = np.asarray(ck["window_variance"], float)
        self.window_variance = window_variance
        # Per-axis factor measured on the validation sequence by
        # ``train_gru.py calibrate``; 1 for an uncalibrated variance head.
        self.variance_scale = np.asarray(ck.get("variance_scale", np.ones(3)), float)
        self.torch = torch
        self.used = 0
        self.skipped = 0

    def predict(self, gyro: np.ndarray, accel: np.ndarray, bias: np.ndarray, duration_s: np.ndarray):
        """Batched prediction. gyro/accel are raw [N, 10, 3], bias is [N, 3]."""
        torch = self.torch
        corrected = np.asarray(gyro, float) - np.asarray(bias, float)[:, None, :]
        prior = integrate_rectangle(corrected, duration_s)
        x = np.concatenate([corrected, accel], axis=-1).astype(np.float32)
        x = (x - self.mean) / self.std
        with torch.inference_mode():
            dtheta, logvar = self.model(torch.from_numpy(x), torch.from_numpy(prior.astype(np.float32)))
        q = so3.quat_mul(prior, so3.rotvec_to_quat(dtheta.numpy().astype(float)))
        var = np.exp(logvar.numpy().astype(float)) * self.variance_scale if logvar is not None else None
        return q / np.linalg.norm(q, axis=1, keepdims=True), var

    def correction(self, t: np.ndarray, gyro: np.ndarray, accel: np.ndarray, bias: np.ndarray):
        """Rate correction and window variance for one camera interval.

        ``t``, ``gyro``, ``accel`` hold the samples from one frame time to the
        next, both ends included. Returns ``None`` when the interval does not
        look like a training window (dropped IMU samples, odd length).
        """
        duration = (t[-1] - t[0]) * 1e-9
        if len(t) != WINDOW + 1 or abs(duration - 0.05) > 0.002:
            self.skipped += 1
            return None
        q_net, var = self.predict(gyro[None, :-1], accel[None, :-1], bias[None], np.array([duration]))
        # What the filter's midpoint integration gives without correction.
        R_mid = np.eye(3)
        for i in range(WINDOW):
            dt = (t[i + 1] - t[i]) * 1e-9
            R_mid = R_mid @ so3.exp((0.5 * (gyro[i] + gyro[i + 1]) - bias) * dt)
        delta = so3.log(R_mid.T @ so3.quat_to_rot(q_net[0]))
        variance = var[0] if var is not None else self.window_variance
        if variance is None:
            raise ValueError("checkpoint has no variance head; pass window_variance")
        self.used += 1
        return delta / duration, np.asarray(variance, float)
