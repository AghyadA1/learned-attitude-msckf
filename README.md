# learned-attitude-msckf

Monocular visual-inertial odometry on the EuRoC MAV dataset, written in
Python. It grew out of my fifth-year project and has three parts:

- an **MSCKF** (an error-state EKF over a sliding window of camera poses) that
  starts from a still period it finds on its own, with no ground truth;
- a small **GRU that corrects 50 ms of gyro integration**, used inside the
  filter as a learned gyro model;
- a **takeoff detector** that finds when the motors start and the drone
  leaves the ground, and calibrates the gyro bias in between.

OpenVINS v2.7, run as a monocular MSCKF with the same window size, is the
reference.

<p align="center">
  <img src="docs/media/takeoff_v2_01.gif" width="49%" alt="Takeoff detection on V2_01">
  <img src="docs/media/openvins_tracks_v2_01.gif" width="49%" alt="OpenVINS tracks and landmarks on V2_01">
</p>

*Left: V2_01. The gyro bias is measured while the motors spin up and the
drone is still on the ground, and the detector marks the moment it lifts
off. Right: the OpenVINS baseline on the same flight, with KLT tracks and the
landmarks it triangulates.*

## Results

Training uses MH_01 to MH_04 and V1_01 to V1_03, MH_05 is for validation, and
everything below is on the three V2 sequences unless it says otherwise.

### Gyro bias from the pre-flight still period

| sequence | motors on [s] | takeoff [s] | calibration interval [s] | bias error [deg/s] |
|---|---|---|---|---|
| V2_01_easy | 0.25 | 4.25 | 0.0 – 4.2 | 0.160 |
| V2_02_medium | 0.25 | 3.35 | 0.0 – 2.7 | 0.147 |
| V2_03_difficult | 0.35 | 5.15 | 0.3 – 5.1 | 0.131 |

The detector combines IMU stillness, the vibration signature of the motors
and image motion, which can veto a "still" verdict when the camera is
clearly moving. It found the takeoff in 9 of the 11 sequences. MH_01 and
MH_02 begin with the drone already being carried, so it falls back to a
later still period and marks that calibration as non-causal.

The raw gyro bias is about 5 deg/s; after calibration the remaining error
is 0.13 to 0.16 deg/s on V2. That error is measured against EuRoC's
reference bias, which is itself a batch estimate rather than a direct
measurement. All eleven sequences are in
[`results/takeoff/reference_original_detector.csv`](results/takeoff/reference_original_detector.csv).

### The GRU on its own, without the camera

Attitude error in degrees when integrating the IMU alone:

| | one 50 ms window | 1 s (20 windows chained) | whole flight, mean | whole flight, at the end |
|---|---|---|---|---|
| gyro minus preflight bias | 0.033 | 0.227 | 3.16 | 6.60 |
| GRU, first version | 0.030 | 0.323 | 13.57 | 45.83 |
| GRU, bias-corrected prior | **0.025** | 0.231 | **2.57** | **5.53** |
| gyro minus EuRoC reference bias | 0.032 | 0.182 | 0.92 | 1.24 |

The network is one GRU layer with 96 units. It reads the ten IMU samples of
a camera interval (gyro with the bias removed, plus accelerometer) and
predicts a small rotation, at most 0.15 rad per axis, that is applied after
plain integration. The first version used the raw gyro as its prior and
drifted badly on V2_03; subtracting the preflight bias first fixed that.

Per-split numbers are in
[`results/gru/attitude_only_errors.csv`](results/gru/attitude_only_errors.csv).
In that evaluation the gyro was integrated with the rectangle rule, holding
each 5 ms sample constant until the next one, and that is what the network
beat by 23%. The midpoint rule, which the filter uses, does better than the
network without learning anything:

| one 50 ms window, reference bias removed [deg] | MH_05 (validation) | V2 (test) |
|---|---|---|
| gyro, rectangle rule | 0.0169 | 0.0321 |
| GRU, original | 0.0101 | 0.0247 |
| GRU, fine-tuned for the filter | 0.0075 | 0.0231 |
| gyro, midpoint rule | **0.0009** | **0.0184** |

The rectangle-rule row matches the original evaluation to four decimals, so
the two tables measure the same thing
([`results/gru/window_errors.csv`](results/gru/window_errors.csv)). Most of
what the network learned is a better integration rule. The midpoint rule's
very close match on MH_05 is probably helped by EuRoC's reference attitude
being fitted to this same IMU (see the caveats), but it also wins on V2.

Over longer horizons the bias dominates, and a 50 ms window cannot see it.
That is the reason to try the network inside a filter that estimates the
bias, rather than on its own.

### Full visual-inertial odometry

All trajectories are aligned to ground truth with a yaw rotation and a
translation (4-DoF). The OpenVINS baseline is described in
[`baselines/openvins`](baselines/openvins), with all eleven sequences in
[`results/openvins/euroc_openvins_v2.7.csv`](results/openvins/euroc_openvins_v2.7.csv).
The Python filter runs in four variants, all from the same cold start:

| variant | gyro model | gyro noise |
|---|---|---|
| `msckf` | plain gyro | sensor datasheet |
| `msckf-gru-noise` | plain gyro | the fine-tuned GRU's error, measured on MH_05 |
| `gru-preflight` | the GRU above, unchanged | its error, measured on MH_05 |
| `gru-filter` | the GRU fine-tuned for the filter | predicted by the network, scaled on MH_05 |

`msckf-gru-noise` is the control. It tells the filter the gyro is as noisy
as the fine-tuned GRU, but integrates the plain gyro. Mean over the three V2
sequences:

| | ATE RMSE [m] | attitude RMSE [deg] | velocity RMSE [m/s] | attitude NEES |
|---|---|---|---|---|
| OpenVINS v2.7 | 0.144 | **1.33** | 0.057 | |
| `msckf` | 0.138 | 1.60 | 0.058 | 513 |
| `msckf-gru-noise` | **0.128** | 1.40 | **0.051** | 234 |
| `gru-preflight` | 0.144 | 1.46 | 0.052 | 220 |
| `gru-filter` | 0.131 | 1.37 | 0.052 | 231 |

<p align="center">
  <img src="results/vio/trajectories_v2.png" width="100%" alt="Estimated and true trajectories on V2">
</p>

The Python filter is as accurate as OpenVINS on V2. The fine-tuned GRU
improves on datasheet noise (0.131 m against 0.138 m), but the control does
just as well, so the gain comes from the larger gyro noise and not from the
network's rotation. The training sequences say the same: mean ATE 0.30 m for
`msckf`, 0.19 m for the control and 0.20 m and 0.21 m for the two GRU
variants (OpenVINS 0.50 m, most of it on MH_03). Per-sequence numbers are in
[`results/vio/ate_per_sequence.md`](results/vio/ate_per_sequence.md) and
[`results/vio/full_sequences.csv`](results/vio/full_sequences.csv).

The larger noise helps because the filter is overconfident on real data. Its
attitude NEES is in the hundreds where a consistent filter gives 3, mostly
from tilt: it settles about half a degree away from EuRoC's reference tilt
while its own standard deviation is a few hundredths of a degree. The same
filter is consistent in simulation (below), so the extra error probably
comes from something the model leaves out, such as calibration or timing
errors.

Two problems turned up on the way to these numbers. The filter used to drop
features with less than 1° of parallax, so during a ten-second hover on
MH_04 it received no updates and drifted 7 m; without that gate MH_04 is at
0.36 m, level with OpenVINS. And measuring the plain gyro's noise against
EuRoC's reference gives a variance 4 to 50 times below the datasheet's,
probably because the reference attitude is fitted to the same IMU. A filter
given that noise diverged on V2_03, which is why the control borrows the
GRU's noise instead.

Each variant also runs with the camera switched off for 1, 2 or 5 s every
10 s, which is where a better gyro should matter most;
`scripts/reproduce.sh` writes those results to `results/vio/outages_summary.md`.

### Tilt from vanishing lines

Vertical edges in the scene give roll and pitch without drift. Added to the
GRU-only estimate as at most one bounded correction per second, they cut the
V2 attitude error from 2.01 to 1.22 deg and the tilt error from 1.87 to
0.96 deg. This part is not in the filter yet; it is described with the other
side experiments in [`docs/side_experiments.md`](docs/side_experiments.md).

## How the GRU is used inside the filter

For each camera interval of length T (50 ms, ten IMU samples) the filter
would normally integrate the gyro minus its current bias estimate b̂ and get
ΔR_gyro. The network sees the same samples, also with b̂ removed, and
returns ΔR_net. The filter then propagates with an extra constant rate

    r = Log(ΔR_gyro⁻¹ · ΔR_net) / T

so the rotation it integrates over the interval is exactly the network's,
and it uses the network's error variance as the gyro noise for that
interval.

Two details matter:

- **The network reads the filter's bias, not a fixed one.** The first
  attempt fed it the preflight bias and solved for the rate afterwards,
  which made the attitude independent of the bias state, so vision could
  no longer correct the bias through attitude. Here a bias correction
  changes what the network sees. The propagation Jacobian treats r as a
  known input and ignores its small dependence on b̂.
- **The fine-tuned version (`gru-filter`) is trained on the reference bias.**
  The original network saw the preflight bias, which is off by roughly
  0.1 to 0.16 deg/s, and learned to compensate for it. Inside the filter
  that is the bias state's job. Fine-tuning with EuRoC's reference bias as
  input (plus 0.02 deg/s of jitter) leaves the network with only what the
  bias cannot explain, and a second head learns the per-axis error variance
  with a Gaussian likelihood loss.

Two earlier ways of combining them, as an attitude measurement and as a
replacement process model, are described in
[`docs/side_experiments.md`](docs/side_experiments.md) along with why they
were dropped.

## The filter

- **State.** Attitude, position, velocity, gyro and accelerometer bias, plus
  up to 11 IMU-pose clones. Attitude errors are right-multiplicative and the
  world frame is gravity-aligned.
- **Propagation.** Midpoint integration at 200 Hz, with first-estimate
  Jacobians in both propagation and the feature updates.
- **Front end.** KLT on CLAHE-equalised images, a 1 px forward-backward
  check, fundamental-matrix RANSAC, and grid-based top-up to 200 features.
- **Update.** A feature is used when its track ends or reaches the oldest
  clone. It is triangulated (DLT, then Gauss-Newton), checked for depth and
  reprojection error, gated on its own with a 95% χ² test, and
  projected onto the left nullspace of its position Jacobian. The surviving
  rows are stacked, compressed with QR and applied in Joseph form, with at
  most 50 features per update.
- **Start-up.** The filter waits for one second of stillness (low-pass
  filtered IMU plus a median optical flow under 0.5 px), takes gravity and
  the gyro bias from it, and applies zero-velocity updates until the first
  motion. No ground truth is used.

The tests include a simulated flight with known landmarks. Over 12 random
seeds the average attitude and position NEES are 2.9 and 3.2, close to the
3 of a consistent filter. On EuRoC it is not consistent; see the results.

## Running it

Python 3.10 or newer. A CPU is enough, including for the GRU fine-tuning.

```bash
git clone https://github.com/AghyadA1/learned-attitude-msckf.git
cd learned-attitude-msckf
pip install -e ".[gru,dev]"
pytest -q
```

**Data.** EuRoC is hosted on the
[ETH Research Collection](https://doi.org/10.3929/ethz-b-000690084). Download
the machine hall and Vicon room archives into one folder, then

```bash
python scripts/prepare_euroc.py unpack --zips ~/Downloads/euroc --out data/euroc
```

Any copy in the ASL layout (`<sequence>/mav0/...`) works; pass its folder as
`--euroc`.

**Everything at once.**

```bash
bash scripts/reproduce.sh data/euroc
```

This checks the data, runs the tests, detects takeoff, calibrates and
fine-tunes the GRU, runs every filter variant on all sequences plus the V2
outage runs, and scores them. Finished runs are skipped, so it can be
restarted. `JOBS=4` sets the number of parallel filter runs. If you have
OpenVINS exports, pass their folder as a second argument to score them with
the same code.

**One step at a time.**

```bash
python scripts/detect_takeoff.py --euroc data/euroc --video
python scripts/train_gru.py calibrate --euroc data/euroc --checkpoint models/gru_preflight.pt --out models/gru_preflight_cal.pt
python scripts/train_gru.py train --euroc data/euroc --init models/gru_preflight.pt --out models/gru_filter.pt
python scripts/train_gru.py calibrate --euroc data/euroc --checkpoint models/gru_filter.pt --out models/gru_filter_cal.pt
python scripts/train_gru.py compare --euroc data/euroc original=models/gru_preflight.pt fine-tuned=models/gru_filter.pt
python scripts/run_vio.py --euroc data/euroc --variant msckf gru-filter --sequences V2_01_easy
python scripts/evaluate.py --euroc data/euroc
```

`models/gru_preflight.pt` is the trained network from the original project,
unchanged (SHA-256 `bfd22f8e…4bc3`).

## Layout

```
qvio/
  so3.py        rotations and quaternions (xyzw, Hamilton)
  euroc.py      EuRoC reader; ground truth is loaded separately
  frontend.py   KLT tracker
  msckf.py      the filter
  pipeline.py   cold start and the per-frame loop
  gru.py        the attitude GRU as a gyro model
  takeoff.py    takeoff detection and preflight calibration
  metrics.py    ATE, RPE, tilt, NEES, outage drift
scripts/        data preparation, training, runs, scoring, reproduce.sh
baselines/openvins/   offline OpenVINS runner, patch and configuration
results/        tables from the original project and from the scripts
docs/           side experiments and media
tests/          unit tests and a simulated flight
```

## Caveats

- EuRoC's ground-truth attitude and bias come from a batch fit of the IMU
  and motion capture. The 50 ms rotation targets for the GRU are therefore
  close to integrated gyro, which limits what the network can learn and
  partly explains why its gain over integration is small.
- V2 was never used for training or tuning in this repository, but it was
  looked at during the wider project, so it is a held-out test set rather
  than an untouched one.
- The filter is monocular and written for clarity, not speed. It is slower
  than OpenVINS and has no loop closure or persistent map.

## References

- M. Burri et al., *The EuRoC micro aerial vehicle datasets*, IJRR 2016.
- A. Mourikis and S. Roumeliotis, *A multi-state constraint Kalman filter for
  vision-aided inertial navigation*, ICRA 2007.
- G. Huang, A. Mourikis and S. Roumeliotis, *Analysis and improvement of the
  consistency of extended Kalman filter based SLAM*, ICRA 2008.
- P. Geneva et al., *OpenVINS: A research platform for visual-inertial
  estimation*, ICRA 2020.

## Licence

MIT, except `baselines/openvins/`, which builds on OpenVINS and is GPL-3.0.
