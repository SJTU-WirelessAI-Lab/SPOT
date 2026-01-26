# SPOT-Rainbow: Single-Shot Near-Field Positioning via Trainable Rainbow Beamforming

## Related Publication

📄 **Paper**: *SPOT: Single-Shot Near-Field Positioning via Trainable Rainbow Beamforming*  

This repository provides the implementation for the SPOT paper, which presents a trainable rainbow beamforming for near-field positioning with low overhead.

---

## Introduction

Wideband PTAs can synthesize a controllable *“rainbow beam”* by steering different OFDM subcarriers to different directions, enabling low-latency positioning compared with sequential beam sweeping. The SPOT scheme jointly learns:
1) **Module 1 (Beamformer)**: PS/TTD parameters as learnable network weights, projected to feasible hardware ranges using modulo.
2) **Module 2 (Estimator)**: a small DNN that reconstructs user position from the feedback message (max power + subcarrier index).

The whole network is trained end-to-end to minimize 2D localization RMSE.

All code comments, documentation, and printed messages are in English for international collaboration.

---

## File Structure and Function Description

| Category / Method | Filename | Main Function Description |
|---|---|---|
| Data generation | `channel_generation.py` | Generate **train/val/test** datasets. |
| Proposed SPOT | `train.py` | End-to-end SPOT training: **Module 1** learnable PS/TTD beamformer + **Module 2** estimator. |
|  | `functions.py` | Shared core utilities: path management (`BASE_DIR`), system parameter loading, `ISACDataset`, received-signal simulation, differentiable peak picking, and localization losses/metrics. |
| CBS baseline | `CBS.py` | Conventional near-field positioning baseline. |
|  | `functions_CBS.py` | CBS-specific utilities. |
| k-NN baseline | `KNN.py` | Fixed CBS rainbow beam and kNN regressor map per-subcarrier power features to `(phi, r)`. |
| RaiNet baseline | `rainet.py` | Fixed CBS rainbow beam and **RaiNet** (1D CNN). |
| Docs | `README.md` | Project documentation: setup, data generation, training, baselines, and expected directory layout. |


---

## Quick Start

### 1. Environment Setup

Recommended:
- Python 3.9+
- numpy, torch


### 2. Data Generation
Generate train/val/test wideband LoS channels, user geometry labels, and system parameters:

(You may reduce the number of samples according to your storage constraints)
```bash

python channel_generation.py

```
**Main Parameter Descriptions:**
- `out_dir`: Output directory for the generated dataset (train/val/test will be created under this folder)
- `arch`: Array type (ULA or UPA, depending on your implementation)
- `dis_min`: Minimum user distance (meters)
- `dis_max`: Maximum user distance (meters)
- `train_samples`: Number of training samples
- `val_samples`: Number of validation samples
- `test_samples`: Number of test samples
- `chunk_size`: Samples per channel chunk file (recommended to avoid huge single files)
- Other parameters see script comments and command line help

### 3. Train SPOT

Train SPOT:
- Module 1 (Beamformer): learnable PS/TTD (hardware-projected via modulo)

- Module 2 (Estimator): small MLP mapping feedback (max_power, subcarrier_index) to (phi, r) (or scaled outputs used in your code)

```bash
python train.py
```

**Main Parameter Descriptions:**
- `data_dir`: Directory containing generated data
- `out_dir`: Output directory for checkpoints, exported PS/TTD, and figures
- `batch_size`: Training batch size
- `epochs`: Number of training epochs
- `device`: Training device (cpu, cuda:0, etc.)
- Other parameters see script comments and command line help


### 4. Baselines
#### (1) CBS Baseline
CBS provides a classical non-learning benchmark.
```bash

python CBS.py

```

#### (2) k-NN Baseline
kNN baseline uses fixed CBS rainbow beam for user positioning, then runs kNN regression to predict (phi, r).
```bash

python KNN.py

```

#### (3) RaiNet Baseline
RaiNet baseline trains a CNN estimator to predict (x, y) from received power vector.
```bash

python rainet.py

```
---

## Dependencies

- Python 3.10+
- numpy, torch, matplotlib, tqdm, scipy, torchvision, pandas, etc.
- Detailed dependencies see `environment.yml`

---

---

## Contribution and License

We welcome anyone to submit PRs or issues to improve this project.

This project is licensed under the MIT License.

---
