import torch
from transformers import pipeline

class LLMService:
    def __init__(self, model_path="Qwen/Qwen2.5-1.5B-Instruct"):
        print(f"Đang khởi tạo model: {model_path}...")
        
        self.pipe = pipeline(
            "text-generation",
            model=model_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    def generate(self, prompt):
        outputs = self.pipe(
            prompt,
            max_new_tokens=300,
            temperature=0.0,
            do_sample=False, 
            return_full_text=False
        )
        
        return outputs[0]["generated_text"].strip()