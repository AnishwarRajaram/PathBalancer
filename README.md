# PathBalancer - Drivable Area Segmentation

A real-time semantic segmentation pipeline for autonomous driving that classifies Bird's Eye View (BEV) representations into drivable and non-drivable areas using the NuScenes dataset.

## Overview

PathBalancer processes BEV (Bird's Eye View) grid representations of traffic scenes and uses a UNet neural network to perform semantic segmentation. The project outputs a real-time inference pipeline that produces a binary or multi-class mask representing "Drivable" vs. "Non-Drivable" regions, enabling autonomous vehicles to safely navigate their environment.

### Dataset

This project uses the **NuScenes v1.0-mini dataset** (`v1.0-mini.tgz`). The mini dataset is a condensed version of the full NuScenes dataset, containing approximately 404 samples suitable for development and testing.

**Dataset Source:** [NuScenes Dataset](https://www.nuscenes.org/)

- **Size:** Approximately 350 samples for training, 54 samples for validation
- **Format:** BEV grid representations (200x200 pixels)
- **Classes:** Semantic segmentation for drivability assessment
  - **Drivable:** Safe areas for vehicle navigation
  - **Non-Drivable:** Obstacles, hazards, pedestrians, vehicles
  - **Additional Classes:** Fine-grained segmentation for specific hazard types

## Project Structure

### Core Components

- **`UNetForward.py`** - UNet neural network architecture
  - Encoder-decoder structure with skip connections
  - 3 input channels (RGB BEV representations)
  - 4 output classes for semantic segmentation
  - Batch normalization and ReLU activations

- **`NuScenesBevDataset.py`** - PyTorch Dataset class
  - Loads BEV input grids and ground truth labels
  - Handles tensor conversion and channel ordering
  - Automatically matches input and ground truth files

- **`train.py`** - Training script
  - Trains the UNet model on processed BEV data
  - Implements weighted cross-entropy loss to handle class imbalance
  - Includes validation loop and checkpoint saving
  - 350 training samples, 54 validation samples

### Data Processing

- **`preprocess_dataset.py`** - Main preprocessing pipeline
  - Converts raw NuScenes data to BEV grid format
  - Organizes data into input/ground truth directories

- **`groundTruthtoBEVgrid.py`** - Ground truth generation
  - Creates semantic segmentation masks from NuScenes annotations
  - Maps objects to BEV grid coordinates

- **`sampleToBEVgrid.py`** - Sample conversion
  - Converts sample data to BEV grid representations
  - Handles coordinate transformations

### Evaluation & Testing

- **`testModel.py`** - Full model evaluation
  - Tests trained model on validation set
  - Generates performance metrics

- **`testModelSanity.py`** - Sanity checks
  - Validates model output shapes and types
  - Ensures preprocessing pipeline correctness

- **`check_my_tensors.py`** - Tensor debugging utility
  - Inspects tensor dimensions and values
  - Validates data preprocessing

- **`file_count.py`** - Dataset statistics
  - Counts processed samples
  - Verifies dataset completeness

## Setup & Installation

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- NuScenes devkit
- TQDM (for progress bars)

### Installation Steps

1. **Extract the NuScenes dataset:**
   ```bash
   tar -xzf v1.0-mini.tgz
   ```

2. **Run preprocessing:**
   ```bash
   python preprocess_dataset.py
   ```
   This will create processed data directories:
   - `processed_data/input_bev/` - Input BEV representations
   - `processed_data/ground_truth/` - Ground truth labels

3. **Train the model:**
   ```bash
   python train.py
   ```

4. **Evaluate results:**
   ```bash
   python testModel.py
   ```

## Key Features

### Real-Time Inference Pipeline
Optimized for autonomous driving applications with emphasis on inference speed and accuracy balance.

### Drivability Classification
- **Drivable Areas:** Safe regions where vehicles can navigate
- **Non-Drivable Areas:** Obstacles, curbs, pedestrians, and other hazards

### Weighted Loss Function
The model uses weighted cross-entropy loss to handle class imbalance and emphasize critical drivability decisions:
- Non-drivable (obstacles/hazards): 10x penalty - critical for safety
- Drivable (safe road): 5x penalty
- Other regions: Weighted appropriately to ensure robust segmentation

This weighting prevents the model from converging to trivial "all drivable" predictions and ensures safe autonomous navigation.

### Model Architecture
- **Input:** 3-channel BEV representation (200×200)
- **Layers:** 3 encoder blocks, 2 decoder blocks with skip connections
- **Output:** Binary or multi-class drivability mask (200×200)
- **Optimization:** Designed for real-time inference with minimal latency

### Data Pipeline
- Files are stored as NumPy arrays (`_x.npy` for input, `_y.npy` for labels)
- Automatic train/validation split (350/54 samples)
- Batch size: 4 for training, 8 for validation
- No shuffling on validation set for reproducibility

## Usage

### Training from Scratch
```bash
python train.py
```
Saves best model checkpoint to `best_loss.txt`

### Running Sanity Checks
```bash
python testModelSanity.py
python check_my_tensors.py
```

### Counting Dataset Samples
```bash
python file_count.py
```

## Expected Outcomes & Metrics

### Primary Objective
Develop a real-time inference pipeline that outputs a binary or multi-class mask representing **"Drivable"** vs. **"Non-Drivable"** regions in autonomous driving scenarios.

### Evaluation Metrics

**1. Semantic Segmentation Accuracy**
- **mIoU (mean Intersection over Union):** Primary accuracy metric for evaluating segmentation performance
- Per-class IoU for both drivable and non-drivable regions
- Pixel-level accuracy on validation set

**2. Real-Time Performance (FPS)**
- Inference speed measured in frames per second (FPS)
- Critical requirement for autonomous driving applications
- Target: Real-time processing capability for live sensor streams

**3. Model Architecture**
- Efficient UNet architecture with skip connections
- Balance between model complexity and inference speed
- Parameter count and memory footprint optimization

**4. Training Efficiency**
- Training epochs and convergence speed
- Validation loss trajectory
- Model checkpoint management and resumption from best weights

### Success Criteria
- Achieve high mIoU on validation set
- Maintain real-time inference speed (prioritize FPS)
- Robust handling of diverse traffic scenarios
- Consistent performance across different lighting and weather conditions

## Performance Metrics

The model is evaluated on:
- **mIoU (Mean Intersection over Union):** Primary metric for drivability segmentation accuracy
- **Per-Class IoU:** Separate evaluation of drivable vs. non-drivable regions
- **Real-Time Inference:** Frames per second (FPS) during inference
- **Inference Latency:** Time per frame for autonomous driving constraints
- **Architecture Efficiency:** Model size and computational complexity
- **Validation Loss:** Cross-entropy loss trajectory during training

## Data Details

### Input Format
- **Shape:** (200, 200, 3) → PyTorch: (3, 200, 200)
- **Type:** Float32
- **Range:** [0, 1] (normalized RGB values)

### Output Format
- **Shape:** (200, 200)
- **Type:** Long (integer class indices)
- **Range:** 0-3 (class labels)

## Notes

- Model training uses MPS acceleration on Mac if available, otherwise CPU
- Total processed samples: 404
- Training/validation split is sequential (not randomized for reproducibility)
- Checkpoints resume from previous best validation loss if available

## License

See LICENSE file for details.

## Citation

If you use the NuScenes dataset, please cite:

```
Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, 
Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. 
"nuScenes: A multimodal dataset for autonomous driving." CVPR 2020.
```
