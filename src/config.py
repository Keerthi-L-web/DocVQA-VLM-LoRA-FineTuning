import os
from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "dataset"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
OUTPUT_DIR = ROOT_DIR / "outputs"
LOG_DIR = ROOT_DIR / "logs"
MODEL_DIR = ROOT_DIR / "models"

# Ensure directories exist
for directory in [DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Training Hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.01
MAX_LENGTH = 128

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Model paths
BASE_MODEL_NAME = "Salesforce/blip-vqa-base"
