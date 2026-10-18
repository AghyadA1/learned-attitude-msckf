# OpenVINS baseline

A small offline runner that feeds EuRoC's `mav0` folders straight into the
OpenVINS v2.7 `VioManager` (no ROS) and writes the estimated state after every
image. It is the reference the Python filter is compared against.

## Configuration

`config/estimator_config.yaml` is the OpenVINS EuRoC default with three
changes, all chosen on the training sequences:

- monocular (`cam0` only) and no SLAM features, so it is a pure MSCKF like the
  Python filter (11 clones, 40 features per update, 1 px noise, FEJ, RK4);
- static initialisation only, waiting for a still period found from the
  sensors (`zupt_only_at_beginning`, `init_max_disparity: 2`), since the
  dynamic initialiser was unreliable on the Machine Hall sequences;
- serial OpenCV, so runs are repeatable.

No ground truth is used to initialise it, and nothing was tuned on V2.

## Build

```bash
git clone https://github.com/rpng/open_vins.git && cd open_vins
git checkout 93adc241390d13e99232652cf05cbe18a93c7bea      # v2.7
git apply /path/to/baselines/openvins/openvins_export_only.patch
# build ov_msckf without ROS (Eigen 3.4, OpenCV 4, Boost, Ceres 2.1)

cmake -S baselines/openvins -B build/openvins \
      -DOPENVINS_ROOT=/path/to/open_vins -DOPENVINS_BUILD=/path/to/open_vins/build
cmake --build build/openvins -j
```

The patch only adds read-only access to the tracker database, the features
used in the last MSCKF update and per-feature update diagnostics, for the
exporters. The estimator equations are unchanged.

## Run and score

```bash
for s in MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult \
         V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult; do
  build/openvins/euroc_file_runner --mav0 DATA/$s/mav0 \
      --config baselines/openvins/config/estimator_config.yaml --output runs/openvins_raw/$s
done
python scripts/evaluate.py --euroc DATA --openvins runs/openvins_raw
```

`evaluate.py` scores these exports with the same alignment and metrics as the
Python filter. The table in `results/openvins/euroc_openvins_v2.7.csv` comes
from the original evaluation of these runs.

## Licence

OpenVINS is GPL-3.0, and the runner and patch here are derived from it, so
the files in this folder are distributed under the GNU General Public License
v3.0 (https://www.gnu.org/licenses/gpl-3.0.html). The rest of the repository
is MIT.
