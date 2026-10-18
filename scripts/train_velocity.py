"""Train and evaluate the velocity GRU (see qvio/velocity.py).

Two jobs, both reading EuRoC directly:

  train     fit the network on the training sequences, keep the epoch with
            the lowest velocity error on the validation sequence, and scale
            its variance head so the normalised squared error there is 1.
            With --members N it trains N networks from different seeds and
            stores them as an ensemble.
  evaluate  per-axis velocity error on the validation and test sequences,
            next to the error of always predicting zero velocity.

Inputs are bias-corrected with the ground-truth biases plus noise, and the
direction of gravity gets a random tilt error, because inside the filter
both come from the filter's own estimates.

Examples
  python scripts/train_velocity.py train --euroc DATA --members 3 --out models/velocity_gru.pt
  python scripts/train_velocity.py train --euroc DATA --members 3 --out models/velocity_gru_mh.pt \
      --sequences MH_01_easy MH_02_easy MH_03_medium MH_04_difficult
  python scripts/train_velocity.py evaluate --euroc DATA --checkpoint models/velocity_gru.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qvio.euroc import TEST, TRAIN, VALIDATION, load_groundtruth, load_sequence  # noqa: E402
from qvio.velocity import UP, VelocityModel, build_model  # noqa: E402


def load_samples(root: str, name: str, cache: Path) -> dict:
    """IMU samples with the ground truth interpolated to their times."""
    path = cache / f"{name}.npz"
    if path.is_file():
        return dict(np.load(path))
    seq = load_sequence(root, name)
    gt = load_groundtruth(root, name)
    keep = (seq.imu_t >= gt.t[0]) & (seq.imu_t <= gt.t[-1])
    t = seq.imu_t[keep]
    interp = lambda x: np.column_stack([np.interp(t, gt.t, x[:, k]) for k in range(3)])  # noqa: E731
    data = {
        "gyro": seq.gyro[keep], "accel": seq.accel[keep],
        "R": Slerp(gt.t.astype(float), Rotation.from_quat(gt.q))(t.astype(float)).as_matrix(),
        "v": interp(gt.v), "bg": interp(gt.bg), "ba": interp(gt.ba),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    return data


def windows(d: dict, window: int, stride: int, gravity_per_sample: bool, rng=None, jitter=None):
    """Network inputs and body-frame velocity targets for windows ending every ``stride`` samples."""
    ends = np.arange(window, len(d["gyro"]), stride)
    idx = ends[:, None] + np.arange(-window, 0)[None, :]
    bg, ba = d["bg"][ends], d["ba"][ends]
    tilt = np.tile(np.eye(3), (len(ends), 1, 1))
    if rng is not None:
        bg = bg + rng.normal(0, jitter["gyro_bias"], bg.shape)
        ba = ba + rng.normal(0, jitter["accel_bias"], ba.shape)
        tilt = Rotation.from_rotvec(rng.normal(0, np.radians(jitter["tilt_deg"]), (len(ends), 3))).as_matrix()
    R_end = d["R"][ends - 1]
    # Up as the filter would see it, with a small attitude error in the
    # body frame of the last sample.
    up = np.einsum("nij,nkj,k->ni", tilt, R_end, UP)
    x = [d["gyro"][idx] - bg[:, None], d["accel"][idx] - ba[:, None]]
    if gravity_per_sample:
        # The same error, seen from every sample's body frame.
        up_world = np.einsum("nij,nj->ni", R_end, up)
        x.append(np.einsum("nlji,nj->nli", d["R"][idx], up_world))
    y = np.einsum("nji,nj->ni", R_end, d["v"][ends - 1])
    return np.concatenate(x, axis=-1).astype(np.float32), up.astype(np.float32), y.astype(np.float32)


def train(args) -> None:
    import torch

    torch.set_num_threads(args.threads)
    cache = Path(args.cache)
    sequences = list(args.sequences or TRAIN)
    tr = [load_samples(args.euroc, n, cache) for n in sequences]
    va = load_samples(args.euroc, VALIDATION[0], cache)
    gps = args.gravity_per_sample

    plain = [windows(d, args.window, args.stride, gps) for d in tr]
    x0 = np.concatenate([p[0] for p in plain])
    mean, std = x0.reshape(-1, x0.shape[-1]).mean(0), x0.reshape(-1, x0.shape[-1]).std(0)
    # Spread of the body-frame velocity in training, for the speed-prior control.
    velocity_rms = np.sqrt(np.mean(np.concatenate([p[2] for p in plain]) ** 2, axis=0))
    del plain, x0
    xv, uv, yv = windows(va, args.window, 10, gps)
    val = (torch.as_tensor((xv - mean) / std), torch.as_tensor(uv), torch.as_tensor(yv))

    members, v_all, var_all, report = [], [], [], []
    for seed in range(args.seed, args.seed + args.members):
        state, best, epoch, v, var = _fit(args, seed, tr, val, mean, std)
        members.append(state)
        v_all.append(v)
        var_all.append(var)
        report.append({"seed": seed, "best_epoch": epoch, "validation_rmse_m_s": best})
        print(f"seed {seed}: best epoch {epoch}, validation RMSE {best:.3f} m/s", flush=True)
    v_all, var_all = np.asarray(v_all), np.asarray(var_all)
    v, var = v_all.mean(axis=0), var_all.mean(axis=0) + v_all.var(axis=0)
    rmse = float(np.sqrt(np.mean((v - yv) ** 2)))
    # Rescale the variance on the validation sequence so the normalised
    # squared error there averages 1 on each axis.
    variance_scale = np.mean((v - yv) ** 2 / var, axis=0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "members": members,
        "input_mean": mean, "input_std": std,
        "window": args.window, "hidden": args.hidden, "pool": args.pool,
        "gravity_per_sample": gps,
        "variance_scale": variance_scale,
        "training_velocity_rms": velocity_rms,
        "validation_rmse_m_s": rmse,
        "members_report": report,
        "training_sequences": sequences,
        "jitter": {"gyro_bias": args.gyro_bias_jitter, "accel_bias": args.accel_bias_jitter,
                   "tilt_deg": args.tilt_jitter_deg},
    }, out)
    print(f"{len(members)} member(s), validation RMSE {rmse:.3f} m/s, variance scale {np.round(variance_scale, 2)}")
    print(f"saved {out}")


def _fit(args, seed, tr, val, mean, std):
    """Train one network; returns its best state and its validation output."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    jitter = {"gyro_bias": args.gyro_bias_jitter, "accel_bias": args.accel_bias_jitter, "tilt_deg": args.tilt_jitter_deg}
    as_t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    xv, uv, yv = val
    model = build_model(args.hidden, args.pool, args.gravity_per_sample)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, best_state, best_epoch = np.inf, None, -1
    for epoch in range(args.epochs):
        parts = [windows(d, args.window, args.stride, args.gravity_per_sample, rng, jitter) for d in tr]
        x = as_t((np.concatenate([p[0] for p in parts]) - mean) / std)
        up = as_t(np.concatenate([p[1] for p in parts]))
        y = as_t(np.concatenate([p[2] for p in parts]))
        order = torch.randperm(len(x))
        model.train()
        for i in range(0, len(x), args.batch):
            b = order[i:i + args.batch]
            v, logvar = model(x[b], up[b])
            err2 = (v - y[b]) ** 2
            # Gaussian NLL for the variance head, plus plain MSE so the
            # velocity is not traded away for a large variance.
            loss = (0.5 * (err2 * torch.exp(-logvar) + logvar)).mean() + err2.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v, _ = model(xv, uv)
        rmse = torch.sqrt(((v - yv) ** 2).mean(0)).numpy()
        score = float(np.sqrt(np.mean(rmse**2)))
        print(f"seed {seed} epoch {epoch:3d}  validation RMSE {np.round(rmse, 3)} m/s  ({score:.3f})", flush=True)
        if score < best:
            best, best_epoch = score, epoch
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        v, logvar = model(xv, uv)
    return best_state, best, best_epoch, v.numpy(), np.exp(logvar.numpy())


def evaluate(args) -> None:
    import pandas as pd

    model = VelocityModel(args.checkpoint)
    cache = Path(args.cache)
    rows = []
    for split, names in (("val", VALIDATION), ("test", TEST)):
        for name in names:
            d = load_samples(args.euroc, name, cache)
            ends = np.arange(model.window, len(d["gyro"]), 20)
            err, zero, z2 = [], [], []
            for e in ends:
                s = slice(e - model.window, e)
                t = np.arange(model.window, dtype=np.int64) * 5_000_000   # 200 Hz
                v, var = model.predict(t, d["gyro"][s], d["accel"][s], d["R"][e - 1], d["bg"][e - 1], d["ba"][e - 1])
                truth = d["R"][e - 1].T @ d["v"][e - 1]
                err.append(v - truth)
                zero.append(truth)
                z2.append((v - truth) ** 2 / var)
            err, zero, z2 = map(np.asarray, (err, zero, z2))
            for method, e2, nz in (("velocity GRU", err, z2), ("zero velocity", zero, None)):
                rmse = np.sqrt(np.mean(e2**2, axis=0))
                rows.append({"split": split, "sequence": name, "method": method, "windows": len(e2),
                             "rmse_x_m_s": rmse[0], "rmse_y_m_s": rmse[1], "rmse_z_m_s": rmse[2],
                             "rmse_m_s": float(np.sqrt(np.mean(np.sum(e2**2, axis=1)))),
                             "mean_nees_per_axis": float(nz.mean()) if nz is not None else np.nan})
    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.4f")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="job", required=True)
    for name in ("train", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--euroc", required=True, help="folder with one sub-folder per EuRoC sequence")
        p.add_argument("--cache", default="cache/velocity_samples")
    t = sub.choices["train"]
    t.add_argument("--out", required=True)
    t.add_argument("--window", type=int, default=200, help="IMU samples per window (200 = 1 s)")
    t.add_argument("--pool", type=int, default=4, help="average this many samples before the GRU")
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--gravity-per-sample", action="store_true")
    t.add_argument("--stride", type=int, default=5, help="samples between training windows")
    t.add_argument("--epochs", type=int, default=25)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--gyro-bias-jitter", type=float, default=0.003, help="rad/s")
    t.add_argument("--accel-bias-jitter", type=float, default=0.08, help="m/s^2")
    t.add_argument("--tilt-jitter-deg", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=0, help="seed of the first member")
    t.add_argument("--members", type=int, default=1, help="networks to train from consecutive seeds and average")
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--sequences", nargs="*", help="training sequences (default: the standard split)")
    e = sub.choices["evaluate"]
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--out", default="results/velocity/velocity_errors.csv")
    args = parser.parse_args()
    {"train": train, "evaluate": evaluate}[args.job](args)


if __name__ == "__main__":
    main()
