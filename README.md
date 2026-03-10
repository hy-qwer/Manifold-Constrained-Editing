# Manifold Constrained Editing

This repository contains the public code and model checkpoints for the manuscript **"Manifold-Guided Deterministic Editing in Visual Representation Spaces"**, which is currently under consideration at *The Visual Computer*.

This code release is directly related to the manuscript currently submitted to *The Visual Computer*. If you use this repository in your research, please cite the associated manuscript.

## Overview

Manifold Constrained Editing (MCE) performs deterministic editing in visual representation spaces under a manifold-guided formulation. This repository provides the public implementation for model definition, dataset loading, evaluation, and released checkpoints used in our experiments.

## Repository Structure

- `main.py`: main evaluation script
- `evaluation/`: evaluation metrics
- `models/`: core model definitions
- `datasets/`: dataset loading utilities
- `requirements.txt`: Python dependencies for reproducing the released evaluation environment

## Environment Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

This repository does **not** redistribute dataset images.

For the experiments based on **CelebA-HQ**, please prepare the dataset locally. Ensure that the dataset is obtained from the original source or an authorized distribution channel, and that you comply with the dataset's license and terms of use.

Once you have the dataset, organize the image files and attribute annotation file locally, and provide their paths to the evaluation script using the following parameters:

- `--img_root`: root directory of the prepared image files
- `--attr_path`: path to the corresponding attribute annotation file

## Checkpoints

Model checkpoints used for evaluation are provided in the **Releases** section of this repository.

The released evaluation code expects the following checkpoint inputs:

- `--stage2_ckpt`: Stage-2 editing model checkpoint
- `--probe_ckpt`: attribute probe checkpoint
- `--decoder_ckpt`: image decoder checkpoint

## Usage

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

This public release focuses on the code and checkpoints required for evaluation of the method described in the manuscript. Dataset images are not included because they are governed by separate licensing and distribution restrictions.

## Citation

If you find the released code useful in your research, please consider citing the associated manuscript.

