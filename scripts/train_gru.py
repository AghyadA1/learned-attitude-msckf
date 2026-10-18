"""Calibrate or retrain the 50 ms attitude GRU for use inside the filter.

Three jobs, all reading EuRoC directly:

  calibrate   measure the per-window rotation error of a checkpoint on the
              validation sequence and store it as the gyro noise the filter
              should use with it.
  train       fine-tune a checkpoint with ground-truth bias as the bias input
              and a second head that predicts the error variance.
  compare     per-window error of plain gyro integration (rectangle and
              midpoint rule) and of the checkpoints, on validation and test.

The bias input is the point. The original network always saw the preflight
bias estimate and learned to cancel its constant error (about 0.1 deg/s);
inside the filter that job belongs to the bias state. Training against the
ground-truth bias leaves the network only what the bias cannot explain.

Examples
  python scripts/train_gru.py calibrate --euroc DATA --checkpoint models/gru_preflight.pt --out models/gru_preflight_cal.pt
  python scripts/train_gru.py train --euroc DATA --init models/gru_preflight.pt --out models/gru_filter.pt
  python scripts/train_gru.py compare --euroc DATA original=models/gru_preflight.pt fine-tuned=models/gru_filter.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qvio import so3  # noqa: E402
from qvio.euroc import TEST, TRAIN, VALIDATION, load_groundtruth, load_sequence  # noqa: E402
from qvio.gru import WINDOW, GRUGyroModel, build_model, integrate_rectangle  # noqa: E402


def build_windows(root: str, name: str) -> dict:
    """One sample per pair of consecutive camera frames, as in the original dataset."""
    seq = load_sequence(root, name)
    gt = load_groundtruth(root, name)
    slerp = Slerp(gt.t.astype(float), Rotation.from_quat(gt.q))
    X, T, Q, B, D, MID = [], [], [], [], [], []
    for t0, t1 in zip(seq.cam_t[:-1], seq.cam_t[1:]):
        if t0 < gt.t[0] or t1 > gt.t[-1]:
            continue
        i0, i1 = np.searchsorted(seq.imu_t, [t0, t1])
        if i1 - i0 != WINDOW or i1 >= len(seq.imu_t) or seq.imu_t[i1] != t1:
            continue
        R0, R1 = slerp([float(t0), float(t1)]).as_matrix()
        bias = np.array([np.interp(seq.imu_t[i0:i1], gt.t, gt.bg[:, k]).mean() for k in range(3)])
        X.append(np.hstack([seq.gyro[i0:i1], seq.accel[i0:i1]]))
        Q.append(so3.rot_to_quat(R0.T @ R1))
        B.append(bias)
        D.append((t1 - t0) * 1e-9)
        T.append(t0)
        # Midpoint integration with the true bias: the plain-gyro reference.
        R = np.eye(3)
        for i in range(i0, i1):
            dt = (seq.imu_t[i + 1] - seq.imu_t[i]) * 1e-9
            R = R @ so3.exp((0.5 * (seq.gyro[i] + seq.gyro[i + 1]) - bias) * dt)
        MID.append(so3.rot_to_quat(R))
    return {"X": np.asarray(X), "q_gt": np.asarray(Q), "bias": np.asarray(B),
            "duration": np.asarray(D), "t0": np.asarray(T), "q_mid": np.asarray(MID)}


def load_split(root: str, names, cache: Path) -> dict:
    parts = []
    for name in names:
        path = cache / f"{name}.npz"
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **build_windows(root, name))
        parts.append(dict(np.load(path)))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def rotation_errors(q_est: np.ndarray, q_gt: np.ndarray) -> np.ndarray:
    """Per-axis error e with R_gt = R_est Exp(e), in radians."""
    return so3.quat_to_rotvec(so3.quat_mul(so3.quat_conj(q_est), q_gt))


def summarize(name: str, e: np.ndarray, duration: np.ndarray) -> dict:
    angle = np.degrees(np.linalg.norm(e, axis=1))
    rate = np.degrees(e / duration[:, None])
    row = {
        "model": name,
        "windows": len(e),
        "mean_error_deg": float(angle.mean()),
        "p95_error_deg": float(np.percentile(angle, 95)),
        "rate_rmse_deg_s": float(np.sqrt(np.mean(np.sum(rate**2, axis=1)))),
        "mean_rate_offset_deg_s": [float(x) for x in rate.mean(axis=0)],
        "window_variance_rad2": [float(x) for x in np.mean(e**2, axis=0)],
    }
    print(f"{name:>14s}: mean {row['mean_error_deg']:.4f} deg, p95 {row['p95_error_deg']:.4f} deg, "
          f"rate RMSE {row['rate_rmse_deg_s']:.3f} deg/s, offset {np.round(row['mean_rate_offset_deg_s'], 3)}")
    return row


def calibrate(args) -> None:
    data = load_split(args.euroc, VALIDATION, Path(args.cache))
    rows = [summarize("plain gyro", rotation_errors(data["q_mid"], data["q_gt"]), data["duration"])]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import torch

    model = GRUGyroModel(args.checkpoint, window_variance=np.ones(3))
    model.variance_scale = np.ones(3)
    q, var = model.predict(data["X"][:, :, :3], data["X"][:, :, 3:], data["bias"], data["duration"])
    e = rotation_errors(q, data["q_gt"])
    rows.append(summarize("GRU", e, data["duration"]))
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ck["window_variance"] = np.asarray(rows[-1]["window_variance_rad2"])
    if var is not None:
        # The head was fitted on the training sequences; rescale it so the
        # normalised squared error on the validation sequence averages 1.
        ck["variance_scale"] = np.mean(e**2 / var, axis=0)
        rows[-1]["variance_scale"] = [float(x) for x in ck["variance_scale"]]
        print(f"variance head scale on {VALIDATION[0]}: {np.round(ck['variance_scale'], 3)}")
    ck["calibration"] = rows[-1]
    torch.save(ck, out)
    print(f"wrote {out}")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(rows, indent=2) + "\n")


def compare(args) -> None:
    """Per-window error with the reference bias removed, for every method."""
    rows = []
    for split, names in (("val", VALIDATION), ("test", TEST)):
        d = load_split(args.euroc, names, Path(args.cache))
        g = d["X"][:, :, :3] - d["bias"][:, None, :]
        methods = {"gyro (rectangle rule)": integrate_rectangle(g, d["duration"]),
                   "gyro (midpoint rule)": d["q_mid"]}
        for spec in args.checkpoints:
            name, _, path = spec.partition("=")
            methods[f"GRU {name}"], _ = GRUGyroModel(path, window_variance=np.ones(3)).predict(
                d["X"][:, :, :3], d["X"][:, :, 3:], d["bias"], d["duration"])
        for method, q in methods.items():
            angle = np.degrees(np.linalg.norm(rotation_errors(q, d["q_gt"]), axis=1))
            rows.append(f"{split},{method},{len(angle)},{angle.mean():.5f},{np.percentile(angle, 95):.5f}")
            print(f"{split:5s} {method:22s} mean {angle.mean():.4f} deg, p95 {np.percentile(angle, 95):.4f} deg")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("split,method,windows,mean_error_deg,p95_error_deg\n" + "\n".join(rows) + "\n")
    print(f"wrote {out}")


def train(args) -> None:
    import torch

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cache = Path(args.cache)
    tr = load_split(args.euroc, args.sequences or TRAIN, cache)
    va = load_split(args.euroc, VALIDATION, cache)

    init = torch.load(args.init, map_location="cpu", weights_only=False)
    mean = np.asarray(init["input_mean"], np.float32)
    std = np.asarray(init["input_std"], np.float32)
    model = build_model(with_variance=True)
    state = model.state_dict()
    state.update({k: v for k, v in init["model_state_dict"].items() if k in state})
    model.load_state_dict(state)
    start_var = np.mean(rotation_errors(tr["q_mid"], tr["q_gt"]) ** 2, axis=0)
    with torch.no_grad():
        model.fc_logvar.weight.zero_()
        model.fc_logvar.bias.copy_(torch.as_tensor(np.log(start_var), dtype=torch.float32))

    def tensors(d, jitter):
        bias = d["bias"] + rng.normal(0, np.radians(args.bias_jitter_deg_s), d["bias"].shape) * jitter
        g = d["X"][:, :, :3] - bias[:, None, :]
        prior = integrate_rectangle(g, d["duration"])
        x = (np.concatenate([g, d["X"][:, :, 3:]], -1) - mean) / std
        target = so3.quat_mul(so3.quat_conj(prior), d["q_gt"])      # R_prior^-1 R_gt
        return [torch.as_tensor(a, dtype=torch.float32) for a in (x, prior, target)]

    def loss_fn(x, prior, target):
        dtheta, logvar = model(x, prior)
        # Remaining error e with R_gt = R_prior Exp(dtheta) Exp(e).
        q = _quat_mul_torch(_rotvec_to_quat_torch(-dtheta), target)
        err = 2.0 * q[:, :3] * torch.sign(q[:, 3:4])
        nll = 0.5 * (err**2 / torch.exp(logvar) + logvar)
        return nll.sum(-1).mean(), (err**2).sum(-1).mean()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    xv, pv, tv = tensors(va, jitter=0.0)
    best, best_state, patience = np.inf, None, 0
    for epoch in range(args.epochs):
        x, prior, target = tensors(tr, jitter=1.0)
        order = torch.randperm(len(x))
        model.train()
        for i in range(0, len(x), args.batch):
            idx = order[i:i + args.batch]
            loss, _ = loss_fn(x[idx], prior[idx], target[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_nll, val_mse = loss_fn(xv, pv, tv)
        print(f"epoch {epoch:3d}  val NLL {val_nll.item():9.4f}  val RMS error {np.degrees(np.sqrt(val_mse.item())):.5f} deg")
        if val_nll.item() < best - 1e-4:
            best, patience = val_nll.item(), 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "input_mean": mean,
        "input_std": std,
        "architecture_config": {"hidden_size": 96, "max_residual_rad": 0.15},
        "trained_from": str(args.init),
        "bias_input": "ground-truth gyro bias with %.3f deg/s jitter" % args.bias_jitter_deg_s,
        "best_validation_nll": best,
    }, out)
    print(f"saved {out}")


def _quat_mul_torch(a, b):
    import torch

    x1, y1, z1, w1 = a.unbind(-1)
    x2, y2, z2, w2 = b.unbind(-1)
    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], -1)


def _rotvec_to_quat_torch(v):
    import torch

    angle = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.cat([v * torch.sin(0.5 * angle) / angle, torch.cos(0.5 * angle)], -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="job", required=True)
    for name in ("calibrate", "train", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--euroc", required=True, help="folder with one sub-folder per EuRoC sequence")
        p.add_argument("--cache", default="cache/gru_windows")
    c = sub.choices["calibrate"]
    c.add_argument("--checkpoint", required=True, help="network to calibrate")
    c.add_argument("--out", required=True, help="calibrated checkpoint (.pt)")
    c.add_argument("--report", help="optional JSON with the error statistics of the gyro and the network")
    m = sub.choices["compare"]
    m.add_argument("checkpoints", nargs="*", help="NAME=PATH for each network to include")
    m.add_argument("--out", default="results/gru/window_errors.csv")
    t = sub.choices["train"]
    t.add_argument("--init", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--patience", type=int, default=10)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--bias-jitter-deg-s", type=float, default=0.02)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--sequences", nargs="*", help="training sequences (default: the standard split)")
    args = parser.parse_args()
    {"calibrate": calibrate, "train": train, "compare": compare}[args.job](args)


if __name__ == "__main__":
    main()
