import os
from torch.utils.tensorboard import SummaryWriter
import src.config as config

class TensorBoardLogger:
    def __init__(self, run_name="docvqa_lora"):
        log_dir = config.LOG_DIR / run_name
        self.writer = SummaryWriter(log_dir=str(log_dir))
        
    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        
    def close(self):
        self.writer.close()
