# PathBalancer (NuScenes BEV Segmentation)

## Overview

`PathBalancer` is a PyTorch project for Bird's Eye View (BEV) semantic segmentation trained on the NuScenes dataset (`v1.0-mini`). It uses a custom UNet (`UNetForward.py`) and supports preprocessing, training, and evaluation.

- `preprocess_dataset.py` : generate BEV inputs and ground truth from NuScenes, output `.npy` to `processed_data`.
- `train.py` : train the UNet model, save best checkpoint as `unet_v4_weights.pth`.
- `testModel.py` : evaluate the trained model (loss, per-class IoU, mIoU, FPS).
- `NuScenesBevDataset.py` : dataset wrapper with train-time augmentation.
- `sampleToBEVgrid.py`, `groundTruthtoBEVgrid.py` : transformation logic (NuScenes sample -> BEV arrays).

## Dataset

Use NuScenes `v1.0-mini` dataset:

- download from official source and put under `trainingData/v1.0-mini`
- `preprocess_dataset.py` expects:
  - `trainingData/v1.0-mini` as `DATAROOT`
  - `VERSION='v1.0-mini'`

### Output (after preprocessing)

- `processed_data/input_bev`: input files named `{sample_token}_x.npy`
- `processed_data/ground_truth`: label files named `{sample_token}_y.npy`

## Setup

1. Create virtual environment (e.g. `venv` or `conda`).
2. Install dependencies:

```bash
pip install torch torchvision tqdm pandas nuscenes
```

3. (Windows) Ensure `python` command points to your environment.

## Preprocessing

Run the preprocess script to generate BEV data from NuScenes:

```bash
python preprocess_dataset.py
```

- Adjust `SAMPLE_RANGE` in `preprocess_dataset.py` if you want fewer or more samples.
- For small test runs, use `SAMPLE_RANGE = (0, 100)`.

## Training

The training pipeline in `train.py`:

```bash
python train.py
```

- Model: `UNet(n_channels=4, n_classes=4)`
- Loss: `CrossEntropyLoss` + `DiceLoss` (class weights `[1,5,15,10]`)
- Split: 80% train / 20% val (by index, hardcoded 404 samples)
- Saved checkpoint: `unet_v4_weights.pth`
- Training log saved: `training_log_4.csv`

Parameters in `train.py` to tune:
- `epochs` (default 15)
- `batch_size` in DataLoader (default 4)
- `TOTAL_SAMPLES` and split fraction

## Testing

Run evaluation using the trained weights:

```bash
python testModel.py
```

Metrics reported:
- average test loss for 20% split (`TRAIN_END` to `TOTAL_SAMPLES-1`)
- recorded best validation loss from `best_loss.txt` (optional)
- class IoU for `[Background, Vehicle, Pedestrian, Obstacle]`
- final mIoU
- inference speed (avg ms + FPS)

### Checkpoints

- `unet_v4_weights.pth` is loaded in `testModel.py`.
- `best_loss.txt` is read for reference but not required.

## Code Structure

- `UNetForward.py`: UNet model definition.
- `NuScenesBevDataset.py`: dataset loader.
- `sampleToBEVgrid.py`: input BEV conversion helper.
- `groundTruthtoBEVgrid.py`: ground truth BEV conversion.
- `check_my_tensors.py`: visualization helper.

## Common issues / troubleshooting

- `FileNotFoundError` for dataset path: ensure `trainingData/v1.0-mini` exists.
- `unet_v4_weights.pth` missing: run `train.py` first.
- `processed_data` folders do not exist: run `preprocess_dataset.py`.
- `mIoU` NaN values: class absent in test set.

## Optional

- Use `train_mobile.py` and `testModel_mobile.py` for lightweight/mobile variants (if included).
- Update `TOTAL_SAMPLES` as needed for data size changes.

## Example outputs / results

From `testModel.py` for 81 test samples (last 20% by default):

- Average Cumulative Loss: e.g. `0.1234`
- Background IoU: e.g. `0.97`
- Vehicle IoU: e.g. `0.74`
- Pedestrian IoU: e.g. `0.66`
- Obstacle IoU: e.g. `0.70`
- Final mIoU: e.g. `0.77`
- Avg inference time: e.g. `28.5 ms`
- Throughput: e.g. `35.1 FPS`

> Values above are illustrative. Actual values depend on training and dataset split.

## Notes

- This project is tuned for `v1.0-mini` scale of the NuScenes dataset.
- For full-size NuScenes, adjust sample counts, memory settings, and training runtime.

