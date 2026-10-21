# Provenance: untrained copy of Strong Baseline

This repository is a snapshot of
[wish44165/YOLOv12-BoT-SORT-ReID](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID)
("Strong Baseline: Multi-UAV Tracking via YOLOv12 with BoT-SORT-ReID", CVPRW 2025),
kept so the method can be retrained from scratch on our own data split.

| | |
|---|---|
| Upstream commit | `071fe7611eef373d503e87236c6a42ea613e6d0f` |
| Upstream license | GNU AGPL-3.0 (see [`LICENSE`](LICENSE)); this copy is distributed under the same license |
| Changes to code | **None.** Every source file is byte-identical to the upstream commit. |
| Removed files | All trained weights: `BoT-SORT/yolov12/weights/v1/MOT_yolov12n.pt`, `BoT-SORT/yolov12/weights/v1/SOT_yolov12l.pt`, `BoT-SORT/yolov12/weights/ViA_yolov12n.pt` |

The upstream [`README.md`](README.md) is kept unchanged; all credit for the method and code
belongs to the original authors.

## Training from scratch

The upstream scripts download released weights automatically. To keep every model free of
any exposure to the evaluation data, override those defaults explicitly.

**Detector (YOLOv12)** — `train.py` downloads `MOT_yolov12n.pt` from Hugging Face and, by
default, fine-tunes from it. Always pass the architecture file instead:

```bash
cd BoT-SORT/yolov12
python train.py --model_name ultralytics/cfg/models/v12/yolov12n.yaml --yaml_path uav.yaml
```

**ReID (FastReID SBS-S50)** — `tools/predict_track3.py` downloads `logs/sbs_S50/config.yaml`
and `model_0016.pth` from Hugging Face on import. Train the ReID model yourself and pass your
own checkpoint with `--fast-reid-weights`; check `MODEL.BACKBONE.PRETRAIN` in the config to
record whether the backbone starts from ImageNet weights.

```bash
cd BoT-SORT
python fast_reid/tools/train_net.py --config-file ./logs/sbs_S50/config.yaml MODEL.DEVICE "cuda:0"
```

## Environment

As in the upstream README: Python 3.11, `torch==2.2.2` / `torchvision==0.17.2` (CUDA 11.8),
`flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl`, plus
`BoT-SORT/yolov12/requirements.txt` and `BoT-SORT/requirements.txt`.
FlashAttention 2 requires an Ampere or newer GPU (e.g. A100); `torch==2.2.2` does not support
Blackwell GPUs.

## Known upstream issues

- `BoT-SORT/run_track3.sh` points to `./yolov12/weights/MOT_yolov12n.pt`; the upstream file
  lives under `weights/v1/`.
- The upstream README totals the BSB dataset as 151,384 frames, while its train (77,293) and
  test (74,538) splits sum to 151,831.
