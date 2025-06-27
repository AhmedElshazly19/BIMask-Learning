# BIMask Learning (Temporary Repo)

BIMask Learning is a training optimization technique that improves deep learning model accuracy using saliency-based masking, without increasing model size or parameters. This repository currently contains pretrained enhanced models and an evaluation script.

## Contents

- `models/`: Folder containing enhanced pretrained models.
- `evaluate.py`: Script to evaluate the accuracy of the models on CIFAR-10 and CIFAR-100 datasets.

## Usage

```bash
python evaluate.py --model MODEL_NAME --dataset {cifar10,cifar100}
