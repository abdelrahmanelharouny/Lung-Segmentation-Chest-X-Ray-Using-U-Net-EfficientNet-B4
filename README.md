# Lung Segmentation from Chest X-Ray Images

## Overview

This repository contains a production-grade deep learning pipeline for binary segmentation of lung regions in chest X-ray (CXR) images.

The architecture combines:

- **U-Net++ decoder**
- **EfficientNet-B4 encoder**
- **scSE attention mechanisms**
- **Deep supervision**
- **Advanced post-processing techniques**

to achieve robust performance on limited medical imaging datasets.

The implementation is structured as a single, fully reproducible Jupyter notebook that covers:

- Configuration
- Data augmentation
- Model definition
- Training
- Evaluation
- Inference

---

## Key Features

- **Architecture:** U-Net++ with EfficientNet-B4 (ImageNet pretrained) encoder
- **Attention Mechanism:** scSE (Spatial and Channel Squeeze & Excitation) blocks integrated into every decoder stage
- **Deep Supervision:** Three auxiliary segmentation heads with weighted loss aggregation `[0.4, 0.3, 0.3]`
- **Loss Function:** Hybrid Tversky + Focal Loss
- **Optimization:** SAM optimizer with AdamW, fp16 mixed precision, Warmup + CosineAnnealingLR scheduling
- **Data Augmentation:** Radiology-specific transformations
- **Inference Pipeline:** Test-Time Augmentation (TTA), Dense CRF refinement, connected-component filtering, and morphological closing
- **Configuration:** Centralized `Config` dataclass for deterministic hyperparameter management

---

## Dataset

- **Source:** [Chest X-Ray Masks and Labels (Kaggle)](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)
- **Samples:** 800 chest X-ray images with corresponding binary lung masks
- **Image Resolution:** Resized to `320 × 320`
- **Normalization:** ImageNet statistics
- **Train/Validation Split:** 80/20 stratified split (`640 / 160`)
- **Segmentation Type:** Binary segmentation (`lung` vs `background`)

---

## Methodology

### Data Augmentation Strategy

Transforms are specifically designed for monochromatic medical imaging and exclude unrealistic color-space distortions.

### Spatial Transformations
- Horizontal Flip (`p = 0.5`)
- Shift / Scale / Rotate (`p = 0.6`)

### Contrast & Illumination
- CLAHE (`p = 0.5`)
- RandomBrightnessContrast (`p = 0.5`)

### Anatomical Variation
- GridDistortion (`p = 0.4`)

### Artifact Simulation
- CoarseDropout (`p = 0.3`)
- Gaussian Noise (`p = 0.3`)

---

## Model Architecture

### Encoder

EfficientNet-B4 extracts multi-scale feature representations across five resolution levels.

### Decoder

The U-Net++ decoder utilizes:

- Nested dense skip connections
- Multi-scale feature fusion
- Progressive spatial refinement

to recover fine-grained lung boundaries.

### Attention Mechanism

scSE attention blocks recalibrate:

- Spatial activations
- Channel-wise feature importance

after each decoder stage.

### Deep Supervision

Three auxiliary segmentation heads are attached at intermediate decoder outputs.

Weighted supervision:

```python
[0.4, 0.3, 0.3]
```

helps:
- Stabilize gradient flow
- Accelerate convergence
- Improve feature consistency across scales

---

## Loss Function & Optimization

### Hybrid Loss

#### Tversky Loss
Parameters:

```python
alpha = 0.7
beta = 0.3
```

Designed to:
- Penalize false negatives
- Improve recall
- Reduce under-segmentation at lung boundaries

---

#### Focal Loss

Parameter:

```python
gamma = 2.0
```

Focuses optimization on:
- Hard pixels
- Boundary regions
- Minority segmentation areas

---

### SAM Optimizer

Sharpness-Aware Minimization (SAM):
- Performs two-step gradient updates
- Encourages flatter minima
- Improves generalization on small datasets

---

### Mixed Precision Training

fp16 automatic mixed precision:
- Reduces GPU memory consumption
- Speeds up training
- Maintains numerical stability

---

### Learning Rate Scheduling

Training strategy:
1. 5-epoch linear warmup
2. Cosine annealing decay

This prevents:
- Early encoder destabilization
- Aggressive weight updates during warmup

---

## Inference & Post-Processing

### 1. Test-Time Augmentation (TTA)

Predictions are averaged across:
- Original image
- Horizontal flip
- Vertical flip
- Combined flips

---

### 2. Dense CRF Refinement

Dense Conditional Random Fields sharpen segmentation boundaries using:
- Bilateral potentials
- Pairwise Gaussian smoothing
- Intensity-conditioned refinement

---

### 3. Connected Component Filtering

Retains only:
- Left lung
- Right lung

by preserving the two largest connected components.

---

### 4. Morphological Closing

Disk-shaped structuring elements are used to:
- Fill internal holes
- Remove vessel-induced gaps
- Improve mask smoothness

---

## Performance Metrics

Evaluation on the held-out validation set demonstrates state-of-the-art segmentation quality.

| Metric | Value | Standard Deviation |
|------|------|------|
| Dice Coefficient | 0.9668 | ± 0.0121 |
| Intersection over Union (IoU) | 0.9361 | ± 0.0178 |
| Pixel Accuracy | 0.9841 | - |
| Validation Loss | 0.0724 | - |

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- pip or conda package manager

---

### Setup

```bash
git clone <repository-url>
cd lung-segmentation-unetpp
pip install -r requirements.txt
```

---

## Dependencies

```text
torch>=1.10.0
segmentation-models-pytorch>=0.3.0
albumentations>=1.4.0
pydensecrf>=1.0
sam-pytorch>=0.1.0
opencv-python>=4.8.0
scikit-image>=0.21.0
matplotlib>=3.7.0
numpy>=1.24.0
```

---

## Usage

### Training

Execute the Jupyter notebook to run the complete training pipeline.

Example configuration:

```python
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Config:
    img_size: Tuple[int, int] = (320, 320)
    encoder: str = "efficientnet-b4"
    decoder_attention: str = "scse"
    deep_sup_weights: List[float] = field(
        default_factory=lambda: [0.4, 0.3, 0.3]
    )

    tversky_alpha: float = 0.7
    focal_gamma: float = 2.0

    use_sam: bool = True
    use_amp: bool = True
```

The training loop includes:
- Early stopping (`patience = 9`)
- Gradient clipping (`max_norm = 1.0`)
- Automatic checkpoint saving

Best model weights are stored as:

```text
lung_seg_best.pth
```

---

### Inference

Load the trained checkpoint and execute the complete inference pipeline.

```python
import torch
from notebook_module import (
    postprocess_mask,
    apply_dense_crf,
    predict_with_tta
)

model = torch.load(
    "lung_seg_best.pth",
    map_location="cpu"
)

model.eval()

prob_map, clean_mask = predict_single(
    model,
    img_tensor,
    img_np_uint8,
    cfg
)
```

---

## Project Structure

```text
├── lung_segmentation.ipynb
├── requirements.txt
├── README.md
└── results/
    ├── lung_seg_best.pth
    └── config.json
```

### Structure Description

- `lung_segmentation.ipynb` → Complete reproducible training & inference notebook
- `requirements.txt` → Python dependency list
- `README.md` → Project documentation
- `results/` → Saved checkpoints, metrics, plots, and configurations

---

## Future Work

### Multi-Class Segmentation
Extend the pipeline to segment:
- Lung lobes
- Cardiac silhouette
- Pathological regions

---

### 3D Medical Imaging
Incorporate:
- CT volumes
- 3D U-Net
- Swin-UNETR

for volumetric segmentation.

---

### Self-Supervised Learning
Pretrain encoders on large-scale unlabeled chest X-ray corpora.

---

### Deployment Optimization
Improve inference latency using:
- TensorRT
- ONNX Runtime
- Quantization-aware optimization

---

### Uncertainty Quantification
Integrate:
- Monte Carlo Dropout
- Deep Ensembles
- Bayesian segmentation

for clinically reliable confidence estimation.

---

## References

1. Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2018).
   *UNet++: A Nested U-Net Architecture for Medical Image Segmentation.*

2. Tan, M., & Le, Q. V. (2019).
   *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.*

3. Roy, A. G., et al. (2018).
   *Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks.*

4. Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021).
   *Sharpness-Aware Minimization for Efficiently Improving Generalization.*

5. Krähenbühl, P., & Koltun, V. (2011).
   *Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials.*

6. Milletari, F., Navab, N., & Ahmadi, S. A. (2016).
   *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.*
