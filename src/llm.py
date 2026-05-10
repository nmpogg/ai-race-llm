import torch
from transformers import pipeline, BitsAndBytesConfig


class LLMService:
    """
    Local LLM wrapper dùng Qwen2.5-7B-Instruct với 4-bit quantization.
    Chạy được trên Colab T4 (16GB VRAM), dùng ~5-6GB VRAM.

    Cài thêm trước khi chạy:
        pip install bitsandbytes>=0.43.0 accelerate>=0.27.0
    """

    def __init__(self, model_path: str = "Qwen/Qwen2.5-7B-Instruct"):
        print(f"🔄 Đang khởi tạo model: {model_path} (4-bit quantized)...")

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
            model_kwargs={"quantization_config": bnb_config},
        )
        self.pipe.tokenizer.padding_side = "left"
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
            )
            return outputs[0]["generated_text"].strip()
        except Exception as e:
            print(f"⚠️ LLM lỗi: {e}")
            return ""