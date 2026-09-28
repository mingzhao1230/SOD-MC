# DRMR: A Degradation-Resistant and Missing-Resilient Network for Dual-Modal Salient Object Detection

Official PyTorch implementation of DRMR for robust RGB-T and RGB-D salient object detection under full-modality and missing-modality conditions. The released datasets and checkpoints cover both RGB-T and RGB-D settings.

## Highlights

- Grid-wise Language-driven Adaptive Quality Modulation (G-LAQ) estimates local RGB quality and adaptively reweights dual-modal features.
- Reliability-Aware Modality Fusion (RAMF) uses local structural uncertainty to guide feature fusion and context modeling.
- Spatial-Frequency Self-Distillation (SFD) transfers full-modality priors under missing-modality conditions.
- A two-stage dual-branch training strategy supports full, RGB-only, and auxiliary-only (thermal or depth) inputs.

## Environment

The code was tested with Python 3.8, PyTorch 1.13.1, torchvision 0.14.1, and CUDA 11.7.

```bash
pip install -r requirements.txt
cd selective_scan
pip install .
cd ..
```

## Downloads

| Resource | Download | Access code |
| --- | --- | --- |
| Checkpoints | [Baidu Netdisk](https://pan.baidu.com/s/16fl5jbeJKeESESse6UEhnA?pwd=DRMR) | `DRMR` |
| RGB-T and RGB-D datasets | [Baidu Netdisk](https://pan.baidu.com/s/1RnPUn0a0xCMSmUHh9O2kKQ?pwd=DRMR) | `DRMR` |

## Pretrained Backbone

Download `vssmsmall_dp03_ckpt_epoch_238.pth` and place it at:

```text
models/pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth
```

The CLIP ViT-B/32 checkpoint is downloaded automatically on first use.

## Dataset Preparation

### RGB-T

Set the dataset root:

```bash
export RGBT_DATA_ROOT=/path/to/rgbt_dataset
```

The expected directory structure is:

```text
rgbt_dataset/
├── VT_train/
│   ├── RGB/
│   ├── T/
│   ├── GT/
│   └── train.txt
├── VT821/
│   ├── RGB/
│   ├── T/
│   ├── GT/
│   └── test.txt
├── VT1000/
│   ├── RGB/
│   ├── T/
│   ├── GT/
│   └── test.txt
└── VT5000/
    ├── RGB/
    ├── T/
    ├── GT/
    └── test.txt
```

Each line in a split file follows this format:

```text
/RGB/image.jpg /GT/mask.png /T/thermal.jpg
```

### RGB-D

For RGB-D training, we combine the training sets of NJUD, NLPR, and DUTLF-D. Evaluation is performed on the test sets of NJUD, NLPR, and DUTLF-D, together with the full SIP dataset. The split files use the same three-column format, with the third path referring to the depth map:

```text
/RGB/image.jpg /GT/mask.png /depth/depth.png
```

## Training

Run the training script from the repository root:

```bash
python train_rgbt.py
```

The default setting uses an input size of 448, a batch size of 2, and 50 epochs. Epochs 1-30 train the main branch with full RGB-T inputs. Epochs 31-50 freeze the main branch and train the control branch and zero convolutions. During Stage II, full, RGB-only, and thermal-only inputs are sampled with equal probability.

RGB-D training follows the same two-stage schedule, replacing the thermal modality with depth and using the combined NJUD, NLPR, and DUTLF-D training sets.

Use `--save_fold` to specify the checkpoint root:

```bash
python train_rgbt.py --save_fold /path/to/checkpoint_root
```

Resume training with:

```bash
python train_rgbt.py \
  --resume /path/to/epoch_30_checkpoint.pth \
  --start_epoch 30
```

## Evaluation

We use [SOD Evaluation Metrics](https://github.com/zyjwuyan/SOD_Evaluation_Metrics) to evaluate prediction maps on the RGB-T benchmarks (VT821, VT1000, and VT5000) and RGB-D benchmarks (NJUD, NLPR, DUTLF-D, and SIP) under three conditions:

1. Full dual-modal input (RGB-T or RGB-D)
2. Missing RGB input (thermal or depth only)
3. Missing auxiliary input (RGB only)

The reported metrics are S-measure, maximum F-measure, maximum E-measure, and MAE.

## Repository Structure

```text
DRMR/
├── clip/                    # CLIP implementation
├── models/                  # DRMR network modules
├── selective_scan/          # CUDA selective-scan extension
├── utils/                   # Utility functions
├── train_rgbt.py            # Two-stage RGB-T training
├── rgbt_dataset.py          # Dataset loader
├── transform_rgbd.py        # Data augmentation
├── IOU.py                   # IoU loss
├── Smeasure.py              # S-measure implementation
└── requirements.txt
```

## Acknowledgements

This code is built upon Samba, VMamba, Mamba selective scan, and OpenAI CLIP. We thank the authors of these projects and other dual-modal salient object detection baselines for their open-source contributions.

## Citation

Citation information will be added upon publication.
