import torch
from PIL import Image
from src.preprocessing import get_processor
from src.model import load_trained_model
import src.config as config

class DocVQAInferencer:
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = get_processor()
        
        path = model_path if model_path else config.CHECKPOINT_DIR / "best_model"
        
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {path}")
            
        self.model = load_trained_model(path, self.device)
        self.model.eval()
        
    def predict(self, image_path: str, question: str):
        """
        Runs inference on a single image and question.
        """
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs
        inputs = self.processor(
            images=image, 
            text=question, 
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=50
            )
            
        answer = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # For BLIP confidence is not directly accessible without custom generation, 
        # so we return a placeholder or dummy score (1.0)
        confidence = 1.0 
        
        return answer, confidence

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python inference.py <image_path> <question>")
    else:
        inferencer = DocVQAInferencer()
        ans, conf = inferencer.predict(sys.argv[1], sys.argv[2])
        print(f"Answer: {ans}")
