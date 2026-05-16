import json
import numpy as np

from src.prompt.prompt import SYSTEM_PROMPT, build_user_prompt

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

    def process(self, question):
 
        api_configs = self.retriever.get_top_apis_config(question, k=5)
 
        # Dùng chat format thay vì raw prompt
        user_msg = build_user_prompt(question, api_configs)
 
        # Gọi LLM với system + user tách biệt
        raw_output = self.llm.generate_chat(SYSTEM_PROMPT, user_msg)
        clean_output = self._clean_json(raw_output)
 
        try:
            api_data = json.loads(clean_output)
            return json.dumps(api_data, ensure_ascii=False)
        except Exception as e:
            print(f"Lỗi JSON Parse: '{question[:40]}' — {e}")
            print(f"Raw output: {raw_output[:200]}")
            return "{}"
        
if __name__ == "__main__":
    # Test APIAgent
    from src.llm import LLMService
    from src.retrieval.apiretriever import APIRetriever

    llm_service = LLMService()
    retriever = APIRetriever(api_config_path="data/api_configs.json")
    
    api_agent = APIAgent(llm_service, retriever)
    
    test_question = "Làm thế nào để tạo một user mới?"
    result = api_agent.process(test_question, top_k=3)
    
    print("Kết quả trích xuất API:")
    print(result)