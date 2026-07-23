import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import json
import random
import numpy as np
import src.config as config
from src.dataset import DocVQADataset
from src.preprocessing import get_processor
from src.model import get_model
from src.logger import TensorBoardLogger
from src.evaluate import evaluate

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train():
    set_seed()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Init Logger
    logger = TensorBoardLogger(run_name="docvqa_lora_run")
    
    # Load processor and dataset
    processor = get_processor()
    train_dataset = DocVQADataset(split="train", processor=processor)
    val_dataset = DocVQADataset(split="val", processor=processor)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Load Model with LoRA
    model = get_model(device)
    
    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    total_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    # Mixed Precision and Gradient Accumulation parameters
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    accumulation_steps = 4
    
    best_anls = 0.0
    patience = 3
    patience_counter = 0
    
    print("Starting training...")
    global_step = 0
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        for step, batch in enumerate(epoch_iterator):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                # Gradient Clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
                logger.log_scalar("Train/Loss", loss.item() * accumulation_steps, global_step)
                logger.log_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)
                global_step += 1
                
            train_loss += (loss.item() * accumulation_steps)
            epoch_iterator.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        print("Evaluating...")
        metrics, _, _ = evaluate(model, processor, val_loader, device)
        print(f"Validation Metrics: {metrics}")
        
        logger.log_scalar("Val/Exact_Match", metrics["Exact Match"], epoch)
        logger.log_scalar("Val/ANLS", metrics["ANLS"], epoch)
        
        # Checkpointing and Early Stopping
        current_anls = metrics["ANLS"]
        if current_anls > best_anls:
            best_anls = current_anls
            patience_counter = 0
            
            # Save Best Model
            best_model_path = config.CHECKPOINT_DIR / "best_model"
            print(f"New best model found! Saving to {best_model_path}")
            model.save_pretrained(best_model_path)
            
            # Save metrics
            with open(config.OUTPUT_DIR / "best_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
            
    logger.close()
    print("Training Complete!")

if __name__ == "__main__":
    train()
