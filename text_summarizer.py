from transformers import pipeline
import torch

class TextSummarization:
    
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.model="transformersbook/pegasus-samsum"

        try:
            self.pipe = pipeline(
                "text2text-generation",
                model=self.model,
                device=self.device
                )
            print("Model loaded successfully on", "GPU" if self.device == 0 else "CPU")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


        