# Strong Baseline (YOLOv12n + BoT-SORT-ReID) — untrained

Multi-UAV tracking in thermal infrared video with a **YOLOv12n** detector and the
**BoT-SORT** tracker with an optional **FastReID** appearance model.

This repository holds the code of *Strong Baseline* **without any trained weights**, so the
detector and the ReID model can be trained from scratch on a chosen data split and compared
fairly against other trackers.

> **Credit.** Method and code by Yu-Hsi Chen —
> [*Strong Baseline: Multi-UAV Tracking via YOLOv12 with BoT-SORT-ReID*](https://arxiv.org/abs/2503.17237)
> (CVPR 2025 Workshops), source repository
> [wish44165/YOLOv12-BoT-SORT-ReID](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID).
> The original README is kept in [`docs/UPSTREAM_README.md`](docs/UPSTREAM_README.md).

## What differs from the source

| | |
|---|---|
| Source commit | `071fe7611eef373d503e87236c6a42ea613e6d0f` |
| Code | **Unchanged** — every source file is identical to the commit above |
| Removed | Trained weights `MOT_yolov12n.pt`, `SOT_yolov12l.pt`, `ViA_yolov12n.pt` |
| Moved | Original `README.md` → `docs/UPSTREAM_README.md` |

## Pipeline

```
thermal IR frame
  → YOLOv12n detector (img size 1600 at inference)
  → BoT-SORT: Kalman prediction + camera motion compensation (sparse optical flow)
  → association: IoU, high/low score stages, optional ReID (FastReID SBS-S50)
  → tracks in MOT format (frame, id, x, y, w, h, score)
```

## Repository structure

```
BoT-SORT/
├── yolov12/              YOLOv12 (modified Ultralytics), train.py, uav.yaml
├── tracker/              BoT-SORT (mc_bot_sort.py is used for Track 3)
├── fast_reid/            FastReID for the appearance model
├── tools/predict_track3.py   multi-UAV tracking entry point
└── run_track3.sh         batch inference over test sequences
TrackEval/                HOTA / MOTA / IDF1 evaluation
data/demo/                small demo sequences
docs/UPSTREAM_README.md   original README
```

## Setup

Matches the source environment. FlashAttention 2 needs an Ampere or newer GPU (e.g. A100);
on older GPUs the code falls back to standard attention. `torch==2.2.2` does not support
Blackwell GPUs.

```bash
conda create -n yolov12_botsort python=3.11 -y
conda activate yolov12_botsort

pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
cd BoT-SORT/yolov12
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install -r requirements.txt
cd ..
pip install -r requirements.txt ultralytics cython faiss-cpu seaborn
pip install --no-build-isolation cython_bbox "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
```

## Training from scratch

> ⚠️ The source scripts download released weights from Hugging Face by default.
> Override them as shown below, or the model will start from weights trained on the
> evaluation data.

**Detector** — `train.py` defaults to fine-tuning `MOT_yolov12n.pt`; pass the architecture
file instead. Set the train/val image folders in `uav.yaml` (YOLO format, one class).

```bash
cd BoT-SORT/yolov12
python train.py --model_name ultralytics/cfg/models/v12/yolov12n.yaml --yaml_path uav.yaml
```

**ReID** — train FastReID SBS-S50 on UAV crops. The config `logs/sbs_S50/config.yaml` comes
from the source authors' Hugging Face repository (`wish44165/YOLOv12-BoT-SORT-ReID`); record
whether `MODEL.BACKBONE.PRETRAIN` starts the backbone from ImageNet.

```bash
cd BoT-SORT
python fast_reid/tools/train_net.py --config-file ./logs/sbs_S50/config.yaml MODEL.DEVICE "cuda:0"
```

## Tracking

`--source` must contain `Test_imgs/<sequence>/` frames and
`TestLabels_FirstFrameOnly/<sequence>.txt`.

```bash
cd BoT-SORT
python tools/predict_track3.py \
    --weights <your_detector>.pt --source <data_dir>/ \
    --img-size 1600 --device "0" --track_buffer 60 \
    --with-reid --fast-reid-config logs/sbs_S50/config.yaml \
    --fast-reid-weights <your_reid>.pth \
    --save_path_answer ./submit/track3/ --hide-labels-name
```

On start-up, `predict_track3.py` always downloads the released ReID config and checkpoint into
`logs/sbs_S50/`. Point `--fast-reid-weights` to your own checkpoint so the released one is not used.

## Known issues in the source

- `BoT-SORT/run_track3.sh` points to `./yolov12/weights/MOT_yolov12n.pt`; pass your own
  weights path.
- The source README totals the dataset as 151,384 frames, while its train (77,293) and test
  (74,538) splits sum to 151,831.

## License

GNU AGPL-3.0, inherited from the source repository — see [`LICENSE`](LICENSE).
