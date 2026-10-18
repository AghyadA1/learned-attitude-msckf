# Side experiments

These were part of the project but are not in the main pipeline. The numbers
come from the original result tables. The code stayed in the original
(private) project because none of it ended up in the final estimator.

## Earlier ways of putting the GRU into a filter

The network predicts the rotation over one 50 ms camera interval. Before the
gyro-model design used in this repository, two more direct options were
tried in an earlier Python filter.

**As an attitude measurement.** Every 50 ms the filter received the network's
relative rotation as a measurement. The trouble is that the network reads the
same gyro samples the filter has just integrated, so the measurement is not
independent of the prediction and the filter becomes overconfident. With the
original noise setting, attitude NEES was about 2300 where a consistent filter
gives 3, and still about 600 with the noise measured from the network's own
errors. Inflating that measured noise 50 times brought it down to about 9 on
the development sequences, at which point the update barely moves the state.

**As the process model.** Here the GRU rotation replaced gyro integration in
the prediction step, with its error covariance added as process noise. On
three 2 s segments of V2, each started from the ground-truth state:

| propagation | attitude [deg] | position [m] | velocity [m/s] | attitude NEES / 3 |
|---|---|---|---|---|
| gyro | 0.347 | 0.0238 | 0.0517 | 4.06 |
| GRU | 0.263 | 0.0289 | 0.0612 | 2.10 |

Attitude and its covariance improved, but position and velocity got worse.
The filter was also a limit in its own right: on the longest gap-free stretch
of V2_03 (19 s) it reached 42 m of position error with the plain gyro.

Both results fed into this repository: a new filter that runs whole sequences
from a sensor-detected still start, and the network used as a gyro model that
reads the filter's own bias estimate (see the README).

## Learned translation (CNN + BiGRU)

A 1-D CNN followed by a two-layer bidirectional GRU predicted velocity change
from IMU windows. It was trained with leave-one-sequence-out validation and a
loss over several displacement horizons. Given ground-truth attitude, it beat
plain strapdown integration on V2 at every horizon:

| V2 mean error | strapdown | CNN-BiGRU |
|---|---|---|
| displacement after 1 s | 0.076 m | 0.059 m |
| displacement after 5 s | 1.72 m | 1.30 m |
| position at the end of the flight | 691 m | 539 m |

It stayed a side experiment for two reasons. The input used ground-truth
attitude the whole time, which no real system has. And dead reckoning still
ends hundreds of metres off; even at one second the network's velocity error
(0.11 m/s) is about twice the full-flight velocity RMSE of OpenVINS with the
camera running (0.05 to 0.07 m/s on V2).

## IMU-guided feature search after an outage

The idea: when the camera comes back after a gap, predict where remembered
landmarks should appear from the IMU alone, search for them with XFeat
descriptors only inside an uncertainty ellipse, and re-anchor with PnP.

Measured on the development sequences, the 95th-percentile IMU-only error
grows to 0.43 deg and 0.36 m after 1 s, and to 3.3 deg and 27 m after 10 s.
Translation dominates: ellipses built from attitude uncertainty alone held
the true landmark 47% of the time, translation-aware ones 95%.

Searching inside the ellipses kept precision at 96% and raised recall from
2.4% (global search) to 6.7%. That was not enough: no frame reached the six
correct matches PnP needs, so the re-anchoring step never ran. Beyond about
0.25 s the ellipses were too large to help. It was only run on development
sequences and never on V2.

## Vanishing-line tilt correction

Indoor scenes have plenty of vertical edges. Their vanishing point gives
roll and pitch directly, with no drift. The detector fits the vertical
vanishing direction with RANSAC on calibrated line normals, guided by the
current attitude, and applies a bounded roll/pitch correction while leaving
yaw alone. It was tested on top of the GRU alone, without the MSCKF.

| V2, mean of three sequences | attitude [deg] | tilt [deg] | corrections per second |
|---|---|---|---|
| GRU only | 2.01 | 1.87 | 0 |
| GRU + lines, every frame (first version) | 2.42 | 0.90 | 15.9 |
| GRU + lines, at most one per 1 s window | 1.22 | 0.96 | 0.52 |

The first version corrected on almost every frame. Tilt dropped, but yaw
error grew and the total error went up. Limiting it to one
correction per second, scored by a confidence measure fitted on the
development sequences, kept the tilt gain and cut the total error by 39%.
Full numbers per sequence are in `results/vanishing_lines/v2_attitude.csv`.
This is the most promising piece to add to the filter as a roll/pitch
measurement.
