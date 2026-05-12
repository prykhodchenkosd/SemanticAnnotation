# Description and Installation

This repository contains a software implementation of automated complexity assessment of Ukrainian-language texts. The project implements a comparative analysis of four architectural approaches:

1. **Classic ML:** Classic machine learning (XGBoost) based on TF-IDF and engineering linguistic statistical features.

2. **Hybrid:** Hybrid approach combining semantic embeddings (RoBERTa), statistical features and XGBoost.

3. **Deep Learning:** Strategy for full training (Fine-Tuning) of the BERT neural network.

4. **CORAL** 

## Requirements

* Python 3.8 or later.

* Operating system: Windows, macOS or Linux.

## Installation

To configure the environment, follow these steps.

### Creating a virtual environment

It is recommended to use an isolated virtual environment (venv).

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Linux / macOS:**
```bash
python -m venv venv
venv\Scripts\activate
```

## Installing dependencies

```bash
pip install -r requirements.txt
```

**Note:** The nltk library requires additional language packs to be loaded. The feature_extractor.py script will attempt to load them automatically on first run.

# Procedure

The scripts must be run sequentially, as the results of the previous stages are used as input for the following ones.

## Step 1. Data preparation and feature generation

The script reads text files from the `data/` folder, cleans, tokenizes, and calculates statistical metrics (number of syllables, readability indices, etc.).

1. Create a `data` folder in the root of the project (if it does not exist).
2. Place your text files (e.g. `f1.txt`, `f2.txt`...) in the `data` folder.
3. Run the extractor:

```bash
python feature_extractor.py
```

* Input: Text files in the data/ folder.
* Output: The dataset_features.csv file (stored in the `feature_extractor_results/run_<date>/` folder).

**Important:** After the script is finished, move the resulting `dataset_features.csv` file to the root of the project for the following steps to work correctly.

## Step 2. Generating embeddings (For hybrid approach)

This step is only required for training the hybrid model. The script uses the `youscan/ukr-roberta-base` model to create vector representations of texts.

```bash
python generate_embeddings.py
```
* Input: `dataset_features.csv` (must be in the root).
* Output: `embeddings_roberta.csv`.

## Step 3. Training the models

You can run training for each of the studied approaches separately.

1. Classic approach (XGBoost + TF-IDF + Statistics)
```bash
python train_xgboost.py
```
2. Hybrid approach (XGBoost + BERT Embeddings + Statistics)
```bash
python train_hybrid.py
```
3. Deep Learning approach (End-to-End Fine-Tuning)
```bash
python train_bert_finetuning.py
```

# Results and notes

## Results

After training is complete, the results are automatically saved in the appropriate directories:

* `experiment_results_xgboost/` — classic model reports.
* `experiment_results_hybrid/` — hybrid model reports.
* `bert_finetune_results_<date>/` — neural network reports.

The generated files include:
* Text report with metrics (Accuracy, F1-Score, Precision, Recall,QWK).
* Confusion Matrix.
* Feature Importance / SHAP graphs for ML models.
* Saved model (for Fine-Tuning).

## Hardware Notes

* **macOS (Apple Silicon):** The code is optimized to use `mps` (Metal Performance Shaders).
* **Windows (NVIDIA GPU):** To speed up neural network training, it is recommended to install PyTorch with CUDA support. If `pip install` installed the CPU version, reinstall PyTorch according to the instructions on the official website.
