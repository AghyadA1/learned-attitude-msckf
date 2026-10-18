#!/usr/bin/env bash
# Everything behind the README tables, in order:
#
#   bash scripts/reproduce.sh /path/to/euroc [/path/to/openvins_exports]
#
# The second argument is optional: a folder with one OpenVINS export per
# sequence (see baselines/openvins) to score next to the Python filter.
# Finished filter runs are skipped, so the script can be restarted.
# JOBS=N sets the number of parallel filter runs (default: half the cores).
set -euo pipefail

EUROC=${1:?usage: bash scripts/reproduce.sh EUROC_DIR [OPENVINS_EXPORTS]}
OPENVINS=${2:-}
JOBS=${JOBS:-$(( $(nproc) > 1 ? $(nproc) / 2 : 1 ))}
V1="V1_01_easy V1_02_medium V1_03_difficult"
V2="V2_01_easy V2_02_medium V2_03_difficult"

cd "$(dirname "$0")/.."
mkdir -p logs
exec > >(tee -a logs/reproduce.log) 2>&1
step() { printf '\n== %s  (%s)\n' "$1" "$(date +%H:%M:%S)"; }

step "checking data and code"
python scripts/prepare_euroc.py check "$EUROC"
python -m pytest -q || echo "some tests failed; continuing, but check the log"

step "takeoff detection and preflight gyro bias"
# Nothing later depends on these, so a failure here should not stop the run.
python scripts/detect_takeoff.py --euroc "$EUROC" || echo "takeoff detection failed; continuing"
python scripts/detect_takeoff.py --euroc "$EUROC" --sequences V2_01_easy --video || true

step "GRU noise, measured on MH_05"
python scripts/train_gru.py calibrate --euroc "$EUROC" --checkpoint models/gru_preflight.pt \
    --out models/gru_preflight_cal.pt --report results/gru/calibration_gru_preflight.json

step "fine-tuning the GRU for use inside the filter"
python scripts/train_gru.py train --euroc "$EUROC" --init models/gru_preflight.pt --out models/gru_filter.pt
python scripts/train_gru.py calibrate --euroc "$EUROC" --checkpoint models/gru_filter.pt \
    --out models/gru_filter_cal.pt --report results/gru/calibration_gru_filter.json
python scripts/train_gru.py compare --euroc "$EUROC" \
    original=models/gru_preflight.pt fine-tuned=models/gru_filter.pt

step "velocity GRU"
# The trained ensembles are in the repository; delete them to train them again.
if [ ! -f models/velocity_gru.pt ]; then
    python scripts/train_velocity.py train --euroc "$EUROC" --members 3 --out models/velocity_gru.pt
fi
if [ ! -f models/velocity_gru_mh.pt ]; then
    python scripts/train_velocity.py train --euroc "$EUROC" --members 3 --out models/velocity_gru_mh.pt \
        --sequences MH_01_easy MH_02_easy MH_03_medium MH_04_difficult
fi
python scripts/train_velocity.py evaluate --euroc "$EUROC" --checkpoint models/velocity_gru.pt

step "MSCKF on all sequences ($JOBS parallel runs)"
python scripts/run_vio.py --euroc "$EUROC" --variant msckf msckf-gru-noise gru-preflight gru-filter --jobs "$JOBS"

step "camera outages on V2"
# The first pass over each outage length re-runs the tracker on the images.
# shellcheck disable=SC2086
python scripts/run_vio.py --euroc "$EUROC" \
    --variant msckf msckf-gru-noise gru-preflight gru-filter msckf-speed-prior gru-velocity gru-both \
    --sequences $V2 --outage 1 2 5 10 --jobs "$JOBS"

step "camera outages on V2 without the velocity reset after each gap"
# shellcheck disable=SC2086
python scripts/run_vio.py --euroc "$EUROC" --variant msckf msckf-speed-prior gru-velocity \
    --sequences $V2 --outage 5 10 --no-recovery --out runs/no_recovery --jobs "$JOBS"

step "velocity GRU in an unseen room"
# Trained on the machine hall only, flown in the Vicon room.
# shellcheck disable=SC2086
python scripts/run_vio.py --euroc "$EUROC" --variant msckf msckf-speed-prior gru-velocity --sequences $V1 \
    --outage 5 10 --velocity-model models/velocity_gru_mh.pt --out runs/cross_room --jobs "$JOBS"

step "scoring"
python scripts/evaluate.py --euroc "$EUROC" ${OPENVINS:+--openvins "$OPENVINS"}
python scripts/evaluate.py --euroc "$EUROC" --runs runs/no_recovery --out results/no_recovery
python scripts/evaluate.py --euroc "$EUROC" --runs runs/cross_room --out results/cross_room

step "done; tables are in results/, the full log in logs/reproduce.log"
