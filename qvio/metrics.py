"""Trajectory metrics against EuRoC ground truth.

Monocular VIO cannot observe the global yaw or position, so the estimate is
aligned to ground truth with a yaw rotation and a translation (4-DoF) before
computing ATE. Tilt is reported separately: it needs no alignment, and it
avoids ZYX Euler angles, which sit next to gimbal lock in EuRoC's IMU frame
(x points up).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from . import so3
from .euroc import GroundTruth

UP = np.array([0.0, 0.0, 1.0])


def interpolate_groundtruth(gt: GroundTruth, t: np.ndarray):
    """Ground-truth position, rotation matrices and velocity at times ``t``."""
    t = np.asarray(t, np.int64)
    keep = (t >= gt.t[0]) & (t <= gt.t[-1])
    tq = t[keep].astype(float)
    tg = gt.t.astype(float)
    p = np.column_stack([np.interp(tq, tg, gt.p[:, i]) for i in range(3)])
    v = np.column_stack([np.interp(tq, tg, gt.v[:, i]) for i in range(3)])
    R = Slerp(tg, Rotation.from_quat(gt.q))(tq).as_matrix()
    return keep, p, R, v


def align_yaw_translation(p_est: np.ndarray, p_gt: np.ndarray):
    """Yaw rotation and translation that best map p_est onto p_gt."""
    me, mg = p_est.mean(axis=0), p_gt.mean(axis=0)
    a, b = p_est - me, p_gt - mg
    yaw = np.arctan2(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]),
                     np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
    Rz = so3.exp(np.array([0.0, 0.0, yaw]))
    return Rz, mg - Rz @ me


def align_first_pose(R_est0: np.ndarray, p_est0: np.ndarray, R_gt0: np.ndarray, p_gt0: np.ndarray):
    """Yaw and translation that make the first estimated pose match ground truth.

    The filter starts at p = 0 with an arbitrary yaw and a tiny covariance on
    both, so this is the alignment under which its covariance is meaningful.
    """
    D = R_gt0 @ R_est0.T
    yaw = np.arctan2(D[1, 0] - D[0, 1], D[0, 0] + D[1, 1])
    Rz = so3.exp(np.array([0.0, 0.0, yaw]))
    return Rz, p_gt0 - Rz @ p_est0


def tilt_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> np.ndarray:
    """Angle between estimated and true gravity direction in the body frame."""
    up_est = np.einsum("nji,j->ni", R_est, UP)
    up_gt = np.einsum("nji,j->ni", R_gt, UP)
    sin = np.linalg.norm(np.cross(up_est, up_gt), axis=1)
    return np.degrees(np.arctan2(sin, np.sum(up_est * up_gt, axis=1)))


def evaluate_run(run: dict, gt: GroundTruth, rpe_deltas_s=(1.0, 5.0, 10.0)) -> dict:
    t = run["t"]
    keep, p_gt, R_gt, v_gt = interpolate_groundtruth(gt, t)
    p_est = run["p"][keep]
    R_est = np.asarray([so3.quat_to_rot(q) for q in run["q"][keep]])
    P = run["P"][keep] if "P" in run else None
    t = t[keep]

    Rz, tz = align_yaw_translation(p_est, p_gt)
    p_al = p_est @ Rz.T + tz
    R_al = np.einsum("ij,njk->nik", Rz, R_est)
    pos_err = np.linalg.norm(p_al - p_gt, axis=1)
    att_err = np.array([so3.angle_between_deg(a, b) for a, b in zip(R_al, R_gt)])
    tilt = tilt_error_deg(R_est, R_gt)

    # NEES in the filter's own frame: move ground truth into it, matching the
    # first pose (where the filter's covariance is near zero). Runs without a
    # covariance (OpenVINS exports) get NaN.
    nees_att = nees_pos = np.nan
    if P is not None:
        Rf, tf = align_first_pose(R_est[0], p_est[0], R_gt[0], p_gt[0])
        R_gt_f = np.einsum("ji,njk->nik", Rf, R_gt)
        p_gt_f = (p_gt - tf) @ Rf
        e_th = np.array([so3.log(a.T @ b) for a, b in zip(R_est, R_gt_f)])
        e_p = p_gt_f - p_est
        nees_att = float(np.mean([e @ np.linalg.solve(Pk[0:3, 0:3], e) for e, Pk in zip(e_th, P)]))
        nees_pos = float(np.mean([e @ np.linalg.solve(Pk[3:6, 3:6], e) for e, Pk in zip(e_p, P)]))

    out = {
        "duration_s": (t[-1] - t[0]) * 1e-9,
        "ate_rmse_m": float(np.sqrt(np.mean(pos_err**2))),
        "attitude_rmse_deg": float(np.sqrt(np.mean(att_err**2))),
        "tilt_rmse_deg": float(np.sqrt(np.mean(tilt**2))),
        "velocity_rmse_m_s": float(np.sqrt(np.mean(np.sum((run["v"][keep] @ Rz.T - v_gt) ** 2, axis=1)))),
        "final_position_error_m": float(pos_err[-1]),
        "nees_attitude": nees_att,
        "nees_position": nees_pos,
    }
    for delta in rpe_deltas_s:
        trans, rot = relative_errors(t, p_est, R_est, p_gt, R_gt, delta)
        out[f"rpe_{delta:g}s_m"] = float(np.sqrt(np.mean(trans**2))) if len(trans) else np.nan
        out[f"rpe_{delta:g}s_deg"] = float(np.sqrt(np.mean(rot**2))) if len(rot) else np.nan
    return out


def relative_errors(t, p_est, R_est, p_gt, R_gt, delta_s: float, pairs=None):
    """Translation and rotation error of the motion between pairs of times.

    Both motions are expressed in the body frame at the first time, so no
    global alignment is involved.
    """
    if pairs is None:
        j = np.searchsorted(t, t + int(delta_s * 1e9))
        i = np.flatnonzero(j < len(t))
        pairs = zip(i, j[i])
    trans, rot = [], []
    for i, j in pairs:
        d_est = R_est[i].T @ (p_est[j] - p_est[i])
        d_gt = R_gt[i].T @ (p_gt[j] - p_gt[i])
        trans.append(np.linalg.norm(d_est - d_gt))
        rot.append(so3.angle_between_deg(R_est[i].T @ R_est[j], R_gt[i].T @ R_gt[j]))
    return np.asarray(trans), np.asarray(rot)


def evaluate_outages(run: dict, gt: GroundTruth) -> list[dict]:
    """Drift accumulated over each camera outage in a run."""
    hidden = run["hidden"]
    t = run["t"]
    keep, p_gt, R_gt, _ = interpolate_groundtruth(gt, t)
    if not keep.all():
        raise ValueError("outage evaluation needs ground truth over the whole run")
    p_est = run["p"]
    R_est = np.asarray([so3.quat_to_rot(q) for q in run["q"]])
    rows = []
    edges = np.diff(hidden.astype(int))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1)
    for a, b in zip(starts, ends):
        i, j = a - 1, b          # last state with vision, last state without
        trans, rot = relative_errors(t, p_est, R_est, p_gt, R_gt, 0.0, pairs=[(i, j)])
        tilt = tilt_error_deg(R_est[[i, j]], R_gt[[i, j]])
        rows.append({
            "start_s": (t[i] - t[0]) * 1e-9,
            "duration_s": (t[j] - t[i]) * 1e-9,
            "translation_drift_m": float(trans[0]),
            "rotation_drift_deg": float(rot[0]),
            "tilt_before_deg": float(tilt[0]),
            "tilt_after_deg": float(tilt[1]),
        })
    return rows
