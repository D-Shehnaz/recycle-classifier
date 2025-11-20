# Recyclable vs Non-Recyclable Classifier

A **deep learning project** to classify images of objects as **recyclable** or **non-recyclable** using TensorFlow and Keras.  

This repository includes scripts for generating synthetic datasets, downloading datasets from Kaggle, training models, and predicting new images.

---

## Features

- Binary image classification: recyclable vs non-recyclable  
- Train using **simple CNN** or **MobileNetV2 transfer learning**  
- Generate synthetic datasets for testing  
- Download real datasets from Kaggle automatically  
- Predict on new images with confidence score  

---

## Repository Files

| File | Description |
|------|-------------|
| `project.py` | Simple CNN training script |
| `generatesynthetic.py` | Synthetic dataset generator |
| `utils.py` | Helper functions for loading images and ensuring directories |
| `train.py` | MobileNetV2 training script |
| `predict.py` | Predicts recyclable/non-recyclable images |
| `data_download_kaggle.py` | Download datasets from Kaggle |
| `README.md` | Project description and instructions |

> **Note:** Trained models are **not included**. You must train them locally before using `predict.py`.

---

## Requirements

- Python 3.8+  
- TensorFlow 2.x  
- NumPy  
- Matplotlib  
- Pillow  

Install dependencies:

```bash
pip install -r requirements.txt
