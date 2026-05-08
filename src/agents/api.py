import json, re

class APIAgent:
    def __init__(self, llm_service, retriever):
        self.llm = llm_service
        self.retriever = retriever

    def _extract_json(self, text):
        text = re.sub(r'^```(?:json)?\s*', '', text.strip())
        text = re.sub(r'\s*```$', '', text).strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        # Tìm JSON object có depth
        try:
            start = text.index('{')
            depth, end = 0, -1
            for i, ch in enumerate(text[start:], start):
                if ch == '{': depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end != -1:
                return json.loads(text[start:end + 1])
        except Exception:
            pass

        # Regex trích path + body riêng
        try:
            path_m = re.search(r'"path"\s*:\s*"([^"]+)"', text)
            body_m = re.search(r'"body"\s*:\s*(\{[^}]*\})', text, re.DOTALL)
            if path_m:
                result = {"path": path_m.group(1)}
                if body_m:
                    try: result["body"] = json.loads(body_m.group(1))
                    except: result["body"] = {}
                return result
        except Exception:
            pass

        return None

    def process(self, question):
        api_configs = self.retriever.get_top_apis_config(question, k=3)

        prompt = f"""<|im_start|>system
Bạn là hệ thống chọn và điền cấu hình API.
Dựa vào [CẤU HÌNH API], tìm API phù hợp và trả về JSON gồm "path" và "body".
CHỈ TRẢ VỀ JSON THUẦN TÚY, KHÔNG GIẢI THÍCH, KHÔNG CODE FENCE.
Ví dụ: {{"path": "/api/v1/example", "body": {{"param1": "value1"}}}}

[CẤU HÌNH API]:
{api_configs}
<|im_end|>
<|im_start|>user
Câu hỏi: {question}
<|im_end|>
<|im_start|>assistant
"""
        result = self._extract_json(self.llm.generate(prompt))
        if result and "path" in result:
            return json.dumps(result, ensure_ascii=False)

        # Retry mồi sẵn dấu {
        prompt2 = f"""<|im_start|>system
Trả về JSON với "path" và "body". Chỉ JSON, không giải thích.
{api_configs}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
{{"""
        result2 = self._extract_json("{" + self.llm.generate(prompt2))
        if result2 and "path" in result2:
            return json.dumps(result2, ensure_ascii=False)

        print(f"[APIAgent] Parse thất bại: '{question[:50]}'")
        return "{}"