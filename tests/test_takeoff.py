import numpy as np
import pandas as pd

import simulation as sim
from qvio import takeoff


def test_still_start_and_takeoff_are_found():
    data = sim.simulate(12.0, seed=4, still_s=4.0, n_landmarks=10)
    seq, _, _ = sim.as_sequence(data)
    cfg = takeoff.DetectorConfig()
    imu = takeoff.imu_features(seq, cfg)
    t = (seq.cam_t - seq.cam_t[0]) * 1e-9
    # The camera part needs images; give it a clean "still, then moving" table.
    vis = pd.DataFrame({
        "timestamp_ns": seq.cam_t, "visual_quality": 0.8, "visual_quality_good": True,
        "visual_motion": np.where(t < 4.0, 0.0, 0.6), "reliable_visual_motion": t >= 4.2,
    })
    table = takeoff.combine(imu, vis, cfg)
    events = takeoff.state_machine(table, cfg)
    cal = takeoff.select_interval(seq, table, events, cfg)
    assert events["detected"]
    assert 3.8 <= cal.takeoff_time_s <= 4.6
    assert cal.interval_type == "PREFLIGHT_STATIC" and cal.causal
    assert cal.end_time_s <= 4.1
    np.testing.assert_allclose(np.degrees(cal.gyro_bias), np.degrees(data.bg[0]), atol=0.05)
