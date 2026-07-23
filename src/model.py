import torch
from transformers import BlipForQuestionAnswering
from peft import LoraConfig, get_peft_model
import src.config as config

def get_model(device=None):
    """
    Loads the BLIP VQA base model and wraps it with LoRA adapters.
    """
    print(f"Loading base model: {config.BASE_MODEL_NAME}")
    
    # We load the base model
    model = BlipForQuestionAnswering.from_pretrained(config.BASE_MODEL_NAME)
    
    # Define LoRA Configuration
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        # Targeting attention modules in the text and vision encoders
        # Depending on the BLIP architecture, we target query and value projections
        target_modules=["query", "value"], 
    )
    
    print("Applying LoRA adapters...")
    # Wrap model with PEFT
    peft_model = get_peft_model(model, lora_config)
    
    if device:
        peft_model.to(device)
        
    peft_model.print_trainable_parameters()
    
    return peft_model

def load_trained_model(checkpoint_path, device=None):
    """
    Loads a trained LoRA model for inference or evaluation.
    """
    print(f"Loading trained model from {checkpoint_path}")
    from peft import PeftModel
    
    base_model = BlipForQuestionAnswering.from_pretrained(config.BASE_MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    
    if device:
        model.to(device)
        
    return model
