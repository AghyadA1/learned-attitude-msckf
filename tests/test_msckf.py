import numpy as np
import pytest

import simulation as sim
from qvio import metrics, pipeline
from qvio.msckf import FilterConfig

# The simulated IMU has exactly the sensor.yaml noise.
SIM = FilterConfig(accel_noise_scale=1.0)


@pytest.fixture(scope="module")
def flight():
    data = sim.simulate(25.0, seed=2, still_s=3.0)
    return sim.as_sequence(data)


@pytest.fixture(scope="module")
def result(flight):
    seq, frames, gt = flight
    run = pipeline.run(seq, frames, SIM)
    return run, metrics.evaluate_run(run, gt)


def test_starts_from_the_still_interval(flight, result):
    seq, _, _ = flight
    run, _ = result
    start_s = (run["start_t"] - seq.imu_t[0]) * 1e-9
    assert 0.9 <= start_s <= 3.0


def test_accuracy_on_simulated_flight(result):
    _, m = result
    assert m["ate_rmse_m"] < 0.10
    assert m["tilt_rmse_deg"] < 0.5
    assert m["attitude_rmse_deg"] < 1.0


def test_filter_is_consistent():
    # Expected NEES is 3 per 3-D block. A single run is noisy (yaw error is a
    # slow random walk), so average a few. Over 12 seeds of 40 s it is 2.9
    # for attitude and 3.2 for position.
    nees = []
    for seed in (10, 12, 13):
        seq, frames, gt = sim.as_sequence(sim.simulate(20.0, seed=seed, still_s=3.0))
        m = metrics.evaluate_run(pipeline.run(seq, frames, SIM), gt)
        nees.append([m["nees_attitude"], m["nees_position"]])
    att, pos = np.mean(nees, axis=0)
    assert 1.0 < att < 6.0
    assert 1.0 < pos < 6.0


def test_covariance_stays_symmetric_positive(result):
    run, _ = result
    for P in run["P"][::20]:
        np.testing.assert_allclose(P, P.T, atol=1e-12)
        assert np.linalg.eigvalsh(P).min() > -1e-12


class PerfectGyro:
    """Gyro model that knows the true bias: the attitude should improve."""

    def __init__(self, data):
        self.data = data

    def correction(self, t, gyro, accel, bias):
        i = np.searchsorted(self.data.imu_t, t[0])
        true_bias = self.data.bg[i]
        return bias - true_bias, np.full(3, 1e-10)


def test_gyro_model_hook_is_used():
    data = sim.simulate(15.0, seed=3, still_s=3.0)
    seq, frames, gt = sim.as_sequence(data)
    plain = metrics.evaluate_run(pipeline.run(seq, frames, SIM), gt)
    perfect = metrics.evaluate_run(pipeline.run(seq, frames, SIM, gyro_model=PerfectGyro(data)), gt)
    assert perfect["rpe_1s_deg"] < plain["rpe_1s_deg"]


def test_body_velocity_jacobian(flight):
    seq, _, _ = flight
    from qvio import so3
    from qvio.msckf import MSCKF

    f = MSCKF(seq.camera, seq.noise)
    rng = np.random.default_rng(0)
    f.initialize(0, so3.exp(rng.normal(size=3)), np.zeros(3))
    f.v = rng.normal(size=3)
    vb = f.R.T @ f.v
    d_theta, d_v = 1e-6 * rng.normal(size=3), 1e-6 * rng.normal(size=3)
    moved = (f.R @ so3.exp(d_theta)).T @ (f.v + d_v)
    predicted = vb + so3.skew(vb) @ d_theta + f.R.T @ d_v
    np.testing.assert_allclose(moved, predicted, atol=1e-11)
    # A near-perfect measurement moves the estimate onto it.
    f.body_velocity_update(vb + 0.05, np.full(3, 1e-10))
    np.testing.assert_allclose(f.R.T @ f.v, vb + 0.05, atol=5e-3)


def test_reset_velocity_overrides_a_confident_estimate(flight):
    seq, _, _ = flight
    from qvio.msckf import MSCKF, VEL

    f = MSCKF(seq.camera, seq.noise)
    f.initialize(0, np.eye(3), np.zeros(3))
    f.v = np.array([1.0, 0.0, 0.0])          # 1 m/s off, with a 2 cm/s sigma
    f.reset_velocity(np.zeros(3), 0.2)
    assert np.linalg.norm(f.v) < 0.05
    sigma = np.sqrt(np.diag(f.P[VEL, VEL]))
    assert np.all(sigma < 0.2) and np.all(sigma > 0.01)
    assert np.linalg.eigvalsh(f.P).min() > -1e-12
