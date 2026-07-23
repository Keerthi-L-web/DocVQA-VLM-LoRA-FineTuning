# Document Visual Question Answering using BLIP + LoRA

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)

## 📌 Project Overview
DocVQA-LoRA-Pro is a production-ready Document Visual Question Answering system built using Salesforce's BLIP (`Salesforce/blip-vqa-base`) and fine-tuned using Low-Rank Adaptation (LoRA) via HuggingFace's PEFT library. The pipeline is designed for high performance, utilizing mixed-precision training, gradient accumulation, and a customizable Streamlit interface for inference.

## 🏗️ Architecture
- **Base Model**: `Salesforce/blip-vqa-base` (Vision-Language Transformers)
- **Adapter Strategy**: LoRA targeting `query` and `value` attention layers, dramatically reducing trainable parameters.
- **Dataset**: Custom synthetic invoice/memo dataset generator for reproducibility and robust offline training.
- **Training Pipeline**: PyTorch-based, supports CUDA amp, TensorBoard logging, and Early Stopping.
- **Evaluation**: Computes Exact Match and Average Normalized Levenshtein Similarity (ANLS).
- **UI**: Streamlit-based web application.

## 📂 Folder Structure
```
DocVQA-LoRA-Pro/
├── dataset/            # JSON splits and synthetic images
├── checkpoints/        # Saved LoRA adapter weights
├── outputs/            # Evaluation metrics (best_metrics.json)
├── screenshots/        # Project screenshots
├── logs/               # TensorBoard logs
├── src/                # Core Modules
│   ├── config.py       # Hyperparameters & Paths
│   ├── dataset.py      # PyTorch Dataset
│   ├── evaluate.py     # Evaluation Loop
│   ├── generate_data.py# Synthetic Data Generator
│   ├── inference.py    # Inference API
│   ├── logger.py       # TensorBoard Wrapper
│   ├── metrics.py      # EM and ANLS calculators
│   ├── model.py        # BLIP + PEFT Model builder
│   ├── preprocessing.py# AutoProcessor wrapper
│   └── train.py        # Main Training Loop
├── app.py              # Streamlit Web App
├── requirements.txt    # Dependencies
├── run.sh              # Quickstart Pipeline Script
└── README.md
```

## 🚀 Installation
Ensure you have Python 3.11+ installed.
```bash
git clone <your-repo-url>
cd DocVQA-LoRA-Pro
pip install -r requirements.txt
```

## 📊 Dataset Generation (Optional)
If you don't have a pre-existing DocVQA dataset formatted properly, generate the synthetic training data:
```bash
python src/generate_data.py
```
*This generates 1000 invoice-like document images with Q&A pairs split into train/val/test.*

## 🧠 Training
Initiate the LoRA fine-tuning process. This automatically handles GPU/CPU fallback, mixed precision, and early stopping.
```bash
python src/train.py
```
To monitor training via TensorBoard:
```bash
tensorboard --logdir logs/
```

## 🧪 Evaluation
The training loop automatically evaluates the model on the validation set. Evaluation metrics include:
- **Exact Match (EM)**: % of predictions that exactly match the ground truth.
- **ANLS**: Average Normalized Levenshtein Similarity (Handles minor OCR/tokenization variations).

## 🖥️ Streamlit Deployment
Run the user-friendly web application:
```bash
streamlit run app.py
```
Features:
- Upload any document image.
- Ask questions interactively.
- Receive AI-predicted answers and confidence scoring.
- Download prediction history as a CSV file.

## 🔮 Future Improvements
- Integrate LayoutLMv3 or Donut for native bounding box spatial awareness.
- Implement INT8 quantization for even faster inference.
- Connect to an OCR engine for purely text-based pre-processing.

---
**Author**: Senior AI/ML Engineer 
**License**: MIT
