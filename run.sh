#!/bin/bash
# DocVQA-LoRA-Pro Pipeline Execution Script

set -e

echo "======================================"
echo " Starting DocVQA-LoRA-Pro Pipeline"
echo "======================================"

# Step 1: Install dependencies
echo "[1/4] Installing dependencies..."
pip install -r requirements.txt

# Step 2: Generate Synthetic Dataset
echo "[2/4] Generating Synthetic Dataset..."
python src/generate_data.py

# Step 3: Train the Model
echo "[3/4] Starting Model Training..."
python src/train.py

# Step 4: Run Streamlit Application
echo "[4/4] Launching Streamlit Application..."
streamlit run app.py
