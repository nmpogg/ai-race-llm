import json, os, pickle, faiss, numpy as np, re
from sentence_transformers import SentenceTransformer, CrossEncoder

class DocAgent:
    def __init__(self, llm_service, index_dir="./index_data"):
        self.llm = llm_service
        print("Đang load Hybrid Retrieval...")
        with open(os.path.join(index_dir, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)
        self.faiss_index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        self.embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")
        self.reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        print("Load thành công.")

    def _tokenize(self, text):
        return re.findall(r'\b\w+\b', str(text).lower())

    def retrieve_and_rerank(self, question, top_k_retrieve=10, top_k_rerank=4):
        q_emb = self.embed_model.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        _, faiss_indices = self.faiss_index.search(q_emb, top_k_retrieve)
        faiss_results = [self.chunks[i] for i in faiss_indices[0] if i < len(self.chunks)]

        tokenized_q = self._tokenize(question)
        bm25_scores = self.bm25.get_scores(tokenized_q)
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k_retrieve]
        bm25_results = [self.chunks[i] for i in bm25_indices if bm25_scores[i] > 0]

        seen, combined = set(), []
        for c in faiss_results + bm25_results:
            key = c[:80]
            if key not in seen:
                seen.add(key)
                combined.append(c)

        if not combined:
            return ""

        scores = self.reranker.predict([[question, c] for c in combined])
        best = [c for _, c in sorted(zip(scores, combined), reverse=True)[:top_k_rerank]]
        return "\n\n---\n\n".join(best)

    def _extract_answers(self, text):
        seen, result = set(), []
        for m in re.findall(r'[ABCD]', text.upper()):
            if m not in seen:
                seen.add(m)
                result.append(m)
        return result

    def _parse_output(self, raw):
        # Cách 1: JSON chuẩn
        try:
            m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                answers = self._extract_answers(str(parsed.get("result", "")))
                if answers:
                    return len(answers), ",".join(answers)
        except Exception:
            pass

        # Cách 2: regex tìm đáp án trực tiếp
        for pat in [
            r'(?:result|đáp án|answer)["\s:]+([ABCD](?:[,\s]+[ABCD])*)',
            r'([ABCD](?:,\s*[ABCD])+)',
            r'"([ABCD])"',
        ]:
            m = re.search(pat, raw.upper())
            if m:
                answers = self._extract_answers(m.group(1))
                if answers:
                    return len(answers), ",".join(answers)

        return None, None

    def process(self, question, note=""):
        context = self.retrieve_and_rerank(question)
        note_text = f"Gợi ý: {note}\n\n" if str(note).strip() else ""

        prompt = f"""<|im_start|>system
Bạn là hệ thống trả lời câu hỏi trắc nghiệm dựa trên tài liệu.
Đọc [TÀI LIỆU] và chọn đáp án đúng (A/B/C/D).
Trả về JSON: "numbers" (số đáp án đúng) và "result" (chữ cái cách nhau dấu phẩy).
CHỈ TRẢ VỀ JSON, KHÔNG GIẢI THÍCH.
Ví dụ 1 đáp án: {{"numbers": 1, "result": "B"}}
Ví dụ 2 đáp án: {{"numbers": 2, "result": "A,C"}}

[TÀI LIỆU]:
{note_text}{context}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""
        numbers, result = self._parse_output(self.llm.generate(prompt))
        if result:
            return json.dumps({"numbers": numbers, "result": result}, ensure_ascii=False)

        # Retry prompt ngắn hơn
        prompt2 = f"""<|im_start|>system
Chọn đáp án đúng (A/B/C/D). Chỉ trả về JSON: {{"numbers": 1, "result": "A"}}
<|im_end|>
<|im_start|>user
Tài liệu: {context[:800]}
Câu hỏi: {question}
<|im_end|>
<|im_start|>assistant
"""
        numbers2, result2 = self._parse_output(self.llm.generate(prompt2))
        if result2:
            return json.dumps({"numbers": numbers2, "result": result2}, ensure_ascii=False)

        return json.dumps({"numbers": 1, "result": "A"}, ensure_ascii=False)