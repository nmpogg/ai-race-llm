import torch
from transformers import pipeline, BitsAndBytesConfig


class LLMService:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-7B-Instruct"):
        print(f"🔄 Đang khởi tạo model: {model_path} (4-bit quantized)...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.pipe = pipeline(
            "text-generation",
            model=model_path,
            device_map="auto",
            model_kwargs={
                "quantization_config": bnb_config,
                "low_cpu_mem_usage": True,
            },
        )
        self.pipe.tokenizer.padding_side = "left"

        # FIX: tránh conflict max_new_tokens vs max_length khi tokenizer.model_max_length quá nhỏ
        if hasattr(self.pipe.tokenizer, 'model_max_length'):
            if self.pipe.tokenizer.model_max_length < 4096:
                self.pipe.tokenizer.model_max_length = 4096

        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            print(f"   VRAM sau load: {free/1024**3:.1f}GB free")
        print("✅ Model đã sẵn sàng.")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            outputs = self.pipe(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0.01,
                do_sample=False,
                return_full_text=False,
                pad_token_id=self.pipe.tokenizer.eos_token_id,
                truncation=True,  # FIX: tránh warning khi input quá dài
            )
            return outputs[0]["generated_text"].strip()
        except Exception as e:
            print(f"⚠️ LLM lỗi: {e}")
            return ""