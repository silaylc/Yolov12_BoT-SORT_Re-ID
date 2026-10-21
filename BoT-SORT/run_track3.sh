#!/bin/bash
# Track 3 inference over all sequences in DATA_DIR.
# DATA_DIR must contain Test_imgs/<sequence>/ and TestLabels_FirstFrameOnly/<sequence>.txt.
# Override any path from the shell, e.g.:
#   DET_WEIGHTS=runs/uav/train/weights/best.pt REID_WEIGHTS=logs/my_reid/model_final.pth ./run_track3.sh

DATA_DIR="${DATA_DIR:-../data/MOT/MultiUAV_Test/}"
DET_WEIGHTS="${DET_WEIGHTS:-./yolov12/runs/uav/train/weights/best.pt}"
REID_CONFIG="${REID_CONFIG:-logs/sbs_S50/config.yaml}"
REID_WEIGHTS="${REID_WEIGHTS:-logs/sbs_S50/model_final.pth}"
OUT_DIR="${OUT_DIR:-./submit/track3/test/}"

for f in "$DET_WEIGHTS" "$REID_WEIGHTS"; do
    [ -f "$f" ] || { echo "Missing weights: $f (train your own model first)"; exit 1; }
done

python3 tools/predict_track3.py \
    --weights "$DET_WEIGHTS" \
    --source "${DATA_DIR%/}/" \
    --img-size 1600 \
    --track_buffer 60 \
    --device "0" \
    --agnostic-nms \
    --save_path_answer "$OUT_DIR" \
    --with-reid \
    --fast-reid-config "$REID_CONFIG" \
    --fast-reid-weights "$REID_WEIGHTS" \
    --hide-labels-name
