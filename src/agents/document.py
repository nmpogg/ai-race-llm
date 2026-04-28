import json
import os
import pickle
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer, CrossEncoder

class DocAgent:
    def __init__(self, llm_service, index_dir="./index_data"):
        self.llm = llm_service
        
        print("Đang load hệ thống Hybrid Retrieval...")
        # load chunks
        with open(os.path.join(index_dir, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
            
        # load BM25 index
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)
            
        # load faiss, embedding model
        self.faiss_index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        self.embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")
        
        # load cross-encoder reranker
        self.reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        print("Đã load thành công.")

    def _tokenize(self, text):
        return re.findall(r'\b\w+\b', str(text).lower())

    def retrieve_and_rerank(self, question, top_k_retrieve=5, top_k_rerank=2):
        """Hàm truy xuất kết hợp và xếp hạng lại"""
        # faiss
        q_emb = self.embed_model.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        _, faiss_indices = self.faiss_index.search(q_emb, top_k_retrieve)
        faiss_results = [self.chunks[i] for i in faiss_indices[0]]

        # bm25
        tokenized_q = self._tokenize(question)
        bm25_scores = self.bm25.get_scores(tokenized_q)
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k_retrieve]
        bm25_results = [self.chunks[i] for i in bm25_indices if bm25_scores[i] > 0]

        # merge
        combined_chunks = list(set(faiss_results + bm25_results))
        
        if not combined_chunks:
            return ""

        # rerank
        cross_inp = [[question, chunk] for chunk in combined_chunks]
        cross_scores = self.reranker.predict(cross_inp)
        
        scored_chunks = list(zip(cross_scores, combined_chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        best_chunks = [chunk for score, chunk in scored_chunks[:top_k_rerank]]
        
        return "\n\n---\n\n".join(best_chunks)

    def process(self, question, note=""):
        # retrieve & rerank
        context = self.retrieve_and_rerank(question, top_k_retrieve=5, top_k_rerank=2)
        
        note_text = f"Gợi ý từ hệ thống: {note}\n\n" if str(note).strip() else ""

        # prompt
        prompt = f"""<|im_start|>system
Bạn là hệ thống trích xuất thông tin tự động dựa trên tài liệu.
Nhiệm vụ của bạn là đọc [TÀI LIỆU CUNG CẤP] và trả lời câu hỏi dưới ĐỊNH DẠNG JSON.
TUYỆT ĐỐI CHỈ TRẢ VỀ ĐÚNG 1 ĐỐI TƯỢNG JSON VỚI 2 TRƯỜNG SAU:
- "numbers": Một số nguyên (Ví dụ: 1, 2, 3...) là số lượng đáp án đúng.
- "result": Một hoặc nhiều chữ cái cách nhau bởi dấu ',' đại diện cho đáp án trắc nghiệm (A, B, C, hoặc D).
Không giải thích gì thêm.

Ví dụ output chuẩn:
{{"numbers": 2, "result": "A,C"}}

[TÀI LIỆU CUNG CẤP]:
{note_text}{context}
<|im_end|>
<|im_start|>user
Câu hỏi: {question}
<|im_end|>
<|im_start|>assistant
"""
        # call LLM
        raw_output = self.llm.generate(prompt)
        
        # parse output
        try:
            json_match = re.search(r'\{.*?\}', raw_output, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group(0))
                
                numbers = int(parsed_data.get("numbers", 1))
                result_raw = str(parsed_data.get("result", "")).upper()
                
                # Lấy tất cả đáp án A/B/C/D
                matches = re.findall(r'[ABCD]', result_raw)
                
                # Loại trùng + giữ thứ tự
                seen = set()
                final_list = []
                for m in matches:
                    if m not in seen:
                        seen.add(m)
                        final_list.append(m)
                
                # Nếu không có đáp án hợp lệ thì fallback A
                if not final_list:
                    final_list = ["A"]
                
                final_result = ",".join(final_list)
                
                # Optional: đồng bộ numbers với số lượng đáp án
                numbers = len(final_list)
                
                return json.dumps({
                    "numbers": numbers,
                    "result": final_result
                }, ensure_ascii=False)

        except Exception:
            pass
            
        # fallback
        return json.dumps({
            "numbers": 1,
            "result": "A"
        }, ensure_ascii=False)