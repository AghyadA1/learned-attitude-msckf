import numpy as np
import pytest

import simulation as sim
from qvio import metrics, pipeline
from qvio.msckf import FilterConfig


@pytest.fixture(scope="module")
def flight():
    data = sim.simulate(25.0, seed=2, still_s=3.0)
    return sim.as_sequence(data)


@pytest.fixture(scope="module")
def result(flight):
    seq, frames, gt = flight
    run = pipeline.run(seq, frames, FilterConfig())
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
        m = metrics.evaluate_run(pipeline.run(seq, frames), gt)
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
    plain = metrics.evaluate_run(pipeline.run(seq, frames), gt)
    perfect = metrics.evaluate_run(pipeline.run(seq, frames, gyro_model=PerfectGyro(data)), gt)
    assert perfect["rpe_1s_deg"] < plain["rpe_1s_deg"]
