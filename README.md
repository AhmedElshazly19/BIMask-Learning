# BIMask Learning (Temporary Repo)

BIMask Learning is a training optimization technique that improves deep learning model accuracy using saliency-based masking, without increasing model size or parameters. This repository currently contains pretrained enhanced models and an evaluation script.

## Contents

- `models/`: Folder containing the architectures of enhanced pretrained models.
- `evaluate.py`: Script to evaluate the accuracy of the models on CIFAR-10 and CIFAR-100 datasets.


## Enhanced Pretrained Models

The enhanced pretrained models can be downloaded from the following Google Drive link:  
**[Download Enhanced Models](https://drive.google.com/drive/folders/1JDzqvaQqwqf_OmJ4-gqVQJdn8DbILeeZ?usp=sharing)**  
(Replace with your actual link)

After downloading, place the models in a `checkpoints/` folder or your desired directory and modify `evaluate.py` accordingly if needed.

## Usage

```bash
python evaluate.py --model 'MODEL_NAME' --model_path 'Path to the model checkpoint' --dataset {cifar10,cifar100}
