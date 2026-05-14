import json
import numpy as np

class APIAgent:
    def __init__(self, llm_service, retriever):
        self.llm = llm_service
        self.retriever = retriever

    def _clean_json(self, raw_str):
        text = raw_str.strip()
        if text.startswith("```json"): 
            text = text[7:]
        elif text.startswith("```"): 
            text = text[3:]
        if text.endswith("```"): 
            text = text[:-3]
        return text.strip()

    def process(self, question, top_k=5):
        # top k API configs
        api_configs = self.retriever.get_top_apis_config(question, top_k=top_k)
        
        prompt = f"""<|im_start|>system
Bạn là hệ thống trích xuất thông tin API tự động. 
Dựa vào các [CẤU HÌNH API ĐƯỢC CẤP] dưới đây, hãy tìm API phù hợp nhất với câu hỏi và trích xuất thành một JSON duy nhất gồm 'path' và 'body'. 
TUYỆT ĐỐI CHỈ TRẢ VỀ CHUỖI JSON, KHÔNG GIẢI THÍCH, KHÔNG THÊM BẤT KỲ KÝ TỰ NÀO KHÁC.

[CẤU HÌNH API ĐƯỢC CẤP]:
{api_configs}
<|im_end|>
<|im_start|>user
Câu hỏi: {question}
<|im_end|>
<|im_start|>assistant
"""
        # call LLM
        raw_output = self.llm.generate(prompt)
        clean_output = self._clean_json(raw_output)
        
        # parse json
        try:
            api_data = json.loads(clean_output)
            
            return json.dumps(api_data, ensure_ascii=False)
            
        except Exception as e:
            # Nếu LLM sinh ra rác không phải JSON, trả về object rỗng để hệ thống không sập
            print(f"Lỗi JSON Parse câu: '{question[:30]}...' - Details: {e}")
            return "{}"