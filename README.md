# clip-asos
## Overview

This repository contains code to fine-tune CLIP with LoRA on ASOS product data from the [Kaggle dataset](https://www.kaggle.com/datasets/trainingdatapro/asos-e-commerce-dataset-30845-products).

## Project Structure

- **`pre_processing/`** - Scripts to preprocess raw ASOS data into text and image datasets ready for training
- **`clip_asos/`** - Package for fine-tuning CLIP with LoRA on the processed datasets
- **`post_training_analysis.ipynb`** - Notebook for analyzing embeddings after training using t-SNE and retrieval evaluation

## Usage

1. Download the dataset from Kaggle
2. Run preprocessing scripts to format the data
3. Fine-tune CLIP using LoRA with the processed datasets
4. Analyze the learned embeddings with the post-training analysis notebook