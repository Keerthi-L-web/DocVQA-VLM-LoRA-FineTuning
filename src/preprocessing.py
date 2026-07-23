from transformers import AutoProcessor
import src.config as config

def get_processor(model_name=None):
    """
    Initializes and returns the processor for the vision-language model.
    """
    if model_name is None:
        model_name = config.BASE_MODEL_NAME
        
    print(f"Loading processor: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    return processor
