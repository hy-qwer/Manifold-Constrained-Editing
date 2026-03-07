# Manifold Constrained Editing

Public code release for the main evaluation pipeline of **Manifold Constrained Editing**.

## Contents

This repository currently includes:

- `main.py`: main evaluation script
- `evaluation/`: metric computation code
- `models/`: core model definitions
- `datasets/`: dataset loading code

## Dataset

This repository does not include dataset images.
Please prepare the dataset locally before running the code.

## Usage

```bash
python main.py \
  --img_root path/to/images \
  --attr_path path/to/attributes.txt \
  --stage2_ckpt path/to/stage2_checkpoint.pth \
  --probe_ckpt path/to/probe_checkpoint.pth \
  --decoder_ckpt path/to/decoder_checkpoint.pth
