# Manifold Constrained Editing

This repository contains the public code and model checkpoints for the manuscript **"Manifold-Guided Deterministic Editing in Visual Representation Spaces"**

DOI: [10.5072/zenodo.469062](https://doi.org/10.5072/zenodo.469062)

## Overview

Manifold Constrained Editing (MCE) performs deterministic editing in visual representation spaces under a manifold-guided formulation. This repository provides the public implementation for model definition, dataset loading, training, evaluation, and released checkpoints used in our experiments.

## Repository Structure

- `main.py`: main evaluation script
- `Train_Stage1.py`: Stage-1 training script
- `Train_Stage2.py`: Stage-2 training script
- `evaluation/`: evaluation metrics
- `models/`: core model definitions
- `datasets/`: dataset loading utilities
- `requirements.txt`: Python dependencies for reproducing the released environment

## Environment Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

This repository does **not** redistribute dataset images.

For the experiments based on **CelebA-HQ**, please prepare the dataset locally. Ensure that the dataset is obtained from the original source or an authorized distribution channel, and that you comply with the dataset's license and terms of use.

Once you have the dataset, organize the image files and attribute annotation file locally, and provide their paths to the training or evaluation scripts using the following parameters:

- `--img_root` or `--data-root`: root directory of the prepared image files
- `--attr_path` or `--attr-path`: path to the corresponding attribute annotation file

## Checkpoints

Model checkpoints used for evaluation are provided in the **Releases** section of this repository.

The released evaluation code expects the following checkpoint inputs:

- `--stage2_ckpt`: Stage-2 editing model checkpoint
- `--probe_ckpt`: attribute probe checkpoint
- `--decoder_ckpt`: image decoder checkpoint

For training, Stage-2 additionally requires a Stage-1 checkpoint as initialization.

## Training

### Stage 1

Run Stage-1 training with:

```bash
python Train_Stage1.py \
  --data-root path/to/images \
  --attr-path path/to/attributes.txt \
  --ckpt checkpoints/stage1.pth \
  --log-dir runs/stage1
```

### Stage 2

Run Stage-2 training with:

```bash
python Train_Stage2.py \
  --img-root path/to/images \
  --attr-path path/to/attributes.txt \
  --attr-names Smiling,Young,Male \
  --stage1-ckpt checkpoints/stage1.pth \
  --out-ckpt checkpoints/stage2.pth \
  --log-dir runs/stage2
```

## Evaluation

Run evaluation with:

```bash
python main.py \
  --img_root path/to/images \
  --attr_path path/to/attributes.txt \
  --stage2_ckpt path/to/stage2_checkpoint.pth \
  --probe_ckpt path/to/probe_checkpoint.pth \
  --decoder_ckpt path/to/decoder_checkpoint.pth
```

## Reproducibility Note

This public release focuses on the code and checkpoints required for training and evaluation of the method described in the manuscript. Dataset images are not included because they are governed by separate licensing and distribution restrictions.

## Citation

If you find the released code useful in your research, please consider citing the associated manuscript.

