import numpy as np

from qvio import metrics, so3


def test_alignment_recovers_yaw_and_translation():
    rng = np.random.default_rng(0)
    p_gt = rng.normal(size=(200, 3)) * [3, 3, 1]
    Rz = so3.exp(np.array([0, 0, 0.7]))
    t = np.array([1.0, -2.0, 0.5])
    p_est = (p_gt - t) @ Rz        # p_gt = Rz p_est + t
    R_hat, t_hat = metrics.align_yaw_translation(p_est, p_gt)
    np.testing.assert_allclose(R_hat, Rz, atol=1e-10)
    np.testing.assert_allclose(t_hat, t, atol=1e-10)


def test_tilt_ignores_yaw():
    R = so3.exp(np.array([0.1, -0.2, 0.3]))
    yawed = so3.exp(np.array([0, 0, 1.2])) @ R
    assert metrics.tilt_error_deg(R[None], yawed[None])[0] < 1e-9
    tilted = so3.exp(np.radians([2.0, 0, 0])) @ R
    np.testing.assert_allclose(metrics.tilt_error_deg(R[None], tilted[None])[0], 2.0, atol=1e-9)


def test_first_pose_alignment():
    R_est0 = so3.exp(np.array([0.05, 0.02, 0.4]))
    Rz = so3.exp(np.array([0, 0, -1.1]))
    R_gt0 = Rz @ R_est0
    Rf, tf = metrics.align_first_pose(R_est0, np.zeros(3), R_gt0, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(Rf, Rz, atol=1e-10)
    np.testing.assert_allclose(tf, [1, 2, 3], atol=1e-12)


def test_outages_outside_groundtruth_are_skipped():
    from qvio.euroc import GroundTruth

    n = 200
    t = np.arange(n, dtype=np.int64) * 50_000_000
    p = np.column_stack([np.linspace(0, 10, n), np.zeros(n), np.zeros(n)])
    q = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    gt = GroundTruth(t=t[20:], p=p[20:], q=q[20:], v=np.zeros((n - 20, 3)),
                     bg=np.zeros((n - 20, 3)), ba=np.zeros((n - 20, 3)))
    hidden = np.zeros(n, bool)
    hidden[5:15] = True      # before ground truth starts
    hidden[100:120] = True
    drifted = p.copy()
    drifted[100:120, 1] += 0.3
    run = {"t": t, "p": drifted, "q": q, "hidden": hidden}
    rows = metrics.evaluate_outages(run, gt)
    assert len(rows) == 1
    np.testing.assert_allclose(rows[0]["translation_drift_m"], 0.3, atol=1e-9)
    np.testing.assert_allclose(rows[0]["start_s"], 99 * 0.05)
