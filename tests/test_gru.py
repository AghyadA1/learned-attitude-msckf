from pathlib import Path

import numpy as np
import pytest

from qvio import so3
from qvio.gru import WINDOW, integrate_rectangle

torch = pytest.importorskip("torch")
CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "gru_preflight.pt"


def test_rectangle_prior_matches_direct_product():
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.5, (4, WINDOW, 3))
    q = integrate_rectangle(w, np.full(4, 0.05))
    for k in range(4):
        R = np.eye(3)
        for i in range(WINDOW):
            R = R @ so3.exp(w[k, i] * 0.005)
        np.testing.assert_allclose(so3.quat_to_rot(q[k]), R, atol=1e-12)


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="original checkpoint not present")
def test_correction_is_small_and_finite():
    from qvio.gru import GRUGyroModel

    model = GRUGyroModel(CHECKPOINT, window_variance=np.full(3, 1e-8))
    rng = np.random.default_rng(1)
    t = np.arange(WINDOW + 1, dtype=np.int64) * 5_000_000
    gyro = rng.normal(0, 0.2, (WINDOW + 1, 3))
    accel = np.tile([9.6, 0.1, -1.5], (WINDOW + 1, 1)) + rng.normal(0, 0.3, (WINDOW + 1, 3))
    rate, var = model.correction(t, gyro, accel, np.zeros(3))
    assert np.all(np.isfinite(rate)) and np.linalg.norm(rate) < 0.15 / 0.05
    np.testing.assert_allclose(var, 1e-8)
    # A window with a missing sample is left to the plain gyro.
    assert model.correction(t[:-1], gyro[:-1], accel[:-1], np.zeros(3)) is None


def test_untrained_variance_head_is_used():
    from qvio.gru import build_model

    model = build_model(with_variance=True)
    x = torch.zeros(2, WINDOW, 6)
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2)
    dtheta, logvar = model(x, q)
    assert dtheta.shape == (2, 3) and logvar.shape == (2, 3)
