import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import src.metrics as metrics
import src.config as config
from src.dataset import DocVQADataset, get_processor
from src.model import load_trained_model

def evaluate(model, processor, dataloader, device):
    """
    Evaluates the model on the provided dataloader.
    Returns Exact Match, ANLS, and Accuracy (which is same as EM for VQA).
    """
    model.eval()
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Prepare inputs for generation
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Generate answers
            outputs = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50
            )
            
            # Decode predictions and references
            batch_preds = processor.batch_decode(outputs, skip_special_tokens=True)
            batch_refs = batch["raw_answer"]
            
            predictions.extend(batch_preds)
            references.extend(batch_refs)
            
    # Calculate metrics
    em = metrics.compute_exact_match(predictions, references)
    anls = metrics.compute_anls(predictions, references)
    accuracy = em # In this context, accuracy is typically equivalent to EM
    
    return {
        "Exact Match": em,
        "ANLS": anls,
        "Accuracy": accuracy
    }, predictions, references

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    processor = get_processor()
    val_dataset = DocVQADataset(split="val", processor=processor)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Load the best model (assumes it's saved in checkpoints/best_model)
    model_path = config.CHECKPOINT_DIR / "best_model"
    if model_path.exists():
        model = load_trained_model(model_path, device)
        results, preds, refs = evaluate(model, processor, val_dataloader, device)
        print("Evaluation Results:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
    else:
        print(f"No trained model found at {model_path}. Train the model first.")
