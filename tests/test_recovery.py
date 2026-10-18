import numpy as np
import pytest

import simulation as sim
from qvio import pipeline, recovery


@pytest.fixture(scope="module")
def data():
    return sim.simulate(15.0, seed=4)


def window_from_truth(d, k0, frames=20):
    """A recovery window filled with the true attitude and biases."""
    seq, tracks, _ = sim.as_sequence(d)
    w = recovery.RecoveryWindow(int(d.cam_t[k0]))
    for k in range(k0, k0 + frames):
        if k > k0:
            t, gyro, accel = pipeline._imu_samples(seq, int(d.cam_t[k - 1]), int(d.cam_t[k]))
            i = np.searchsorted(d.imu_t, t[0])
            w.add_imu(t, gyro, accel, d.gt_R[i], d.bg[i], d.ba[i])
        i = np.searchsorted(d.imu_t, d.cam_t[k])
        w.add_frame(int(d.cam_t[k]), d.gt_R[i], seq.camera, tracks[k].ids, tracks[k].xy)
    return w, d.gt_v[i]


def test_velocity_from_one_second_of_images(data):
    errors = []
    for k0 in (40, 80, 120, 160):
        w, v_true = window_from_truth(data, k0)
        v = w.velocity()
        assert v is not None
        errors.append(np.linalg.norm(v - v_true))
    assert max(errors) < 0.05


def test_refinement_removes_the_shrinkage_of_the_linear_solution(data):
    w, v_true = window_from_truth(data, 80)
    obs = w._observations(3)
    linear = recovery._linear_velocity(*obs) + w.dv
    refined = w.velocity()
    assert np.linalg.norm(refined - v_true) < np.linalg.norm(linear - v_true)


def test_too_few_features_gives_none(data):
    w, _ = window_from_truth(data, 80, frames=2)
    assert w.velocity() is None
