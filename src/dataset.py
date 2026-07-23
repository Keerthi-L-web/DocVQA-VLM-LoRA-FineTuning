import json
from PIL import Image
from torch.utils.data import Dataset
import src.config as config

class DocVQADataset(Dataset):
    def __init__(self, split="train", processor=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'
            processor (AutoProcessor): The huggingface processor for BLIP
        """
        self.data_dir = config.DATA_DIR
        self.split_file = self.data_dir / f"{split}.json"
        
        if not self.split_file.exists():
            raise FileNotFoundError(f"Dataset split file not found: {self.split_file}. Run generate_data.py first.")
            
        with open(self.split_file, "r") as f:
            self.data = json.load(f)
            
        self.processor = processor
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = self.data_dir / item["image_path"]
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")
            
        question = item["question"]
        answer = item["answers"][0]
        
        if self.processor:
            # Process image and question for BLIP
            # For BLIP, input format is image + text(question). Label is text(answer)
            encoding = self.processor(
                images=image, 
                text=question, 
                padding="max_length", 
                truncation=True,
                max_length=config.MAX_LENGTH,
                return_tensors="pt"
            )
            # Remove batch dimension added by processor
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}
            
            # Encode answer as labels for text generation loss
            labels = self.processor.tokenizer(
                answer, 
                padding="max_length", 
                truncation=True,
                max_length=config.MAX_LENGTH,
                return_tensors="pt"
            ).input_ids
            
            # Replace pad token id with -100 to ignore in loss calculation
            labels = labels.squeeze(0)
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
            encoding["labels"] = labels
            
            # Also keep raw text for evaluation purposes
            encoding["raw_question"] = question
            encoding["raw_answer"] = answer
            
            return encoding
            
        return {
            "image": image,
            "question": question,
            "answer": answer
        }
