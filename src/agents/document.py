import json
import os
import pickle
import faiss
import numpy as np
import re
import sqlite3
from sentence_transformers import SentenceTransformer, CrossEncoder

class DocAgent:
    def __init__(self, llm_service, index_dir="./data/knowledge"):
        self.llm = llm_service
        self.index_dir = index_dir
        
        print("Đang load hệ thống Hybrid Retrieval...")
        
        # load chunks
        with open(os.path.join(index_dir, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
            
        # SQLite Database
        self.db_path = os.path.join(index_dir, "bm25_index.db")
        if not os.path.exists(self.db_path):
            print(f"Cảnh báo: Không tìm thấy file {self.db_path}. Hãy build index trước!")
            
        # load faiss & embedding model
        self.faiss_index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        self.embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")
        
        # load cross-encoder reranker
        self.reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        print("Đã load thành công DocAgent.")

    def retrieve_bm25_sqlite(self, question, top_k=10):
        clean_query = re.sub(r'[^\w\s]', ' ', question).strip()
        if not clean_query:
            return []
            
        try:
            # Mở kết nối tạm thời và truy vấn siêu tốc
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT chunk_id 
                FROM bm25_chunks 
                WHERE bm25_chunks MATCH ? 
                ORDER BY rank 
                LIMIT ?
            ''', (clean_query, top_k))
            
            result_ids = [row[0] for row in cursor.fetchall()]
            conn.close()
            return result_ids
            
        except sqlite3.OperationalError as e:
            # Bắt lỗi an toàn nếu có từ khóa quá lạ
            print(f"Lỗi truy vấn BM25: {e}")
            return []

    def retrieve_and_rerank(self, question, top_k_retrieve=10, top_k_rerank=7):
        """Hàm truy xuất kết hợp và xếp hạng lại (Hybrid Search)"""
        
        # truy xuất bằng FAISS
        q_emb = self.embed_model.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        _, faiss_indices = self.faiss_index.search(q_emb, top_k_retrieve)
        # chỉ lấy các ID hợp lệ (khác -1)
        faiss_ids = [int(i) for i in faiss_indices[0] if i != -1]

        # tuy xuất bằng BM25
        bm25_ids = self.retrieve_bm25_sqlite(question, top_k_retrieve)

        # merged
        combined_ids = list(set(faiss_ids + bm25_ids))
        
        if not combined_ids:
            return ""

        candidate_contexts = []
        for cid in combined_ids:
            if 0 <= cid < len(self.chunks):
                chunk_obj = self.chunks[cid]
                source = chunk_obj.get("metadata", {}).get("source", "Unknown")
                text = chunk_obj.get("text", "")
                
                candidate_contexts.append(f"[Trích từ: {source}]\n{text}")

        if not candidate_contexts:
            return ""

        # rerank
        cross_inp = [[question, ctx] for ctx in candidate_contexts]
        cross_scores = self.reranker.predict(cross_inp)
        
        scored_chunks = list(zip(cross_scores, candidate_contexts))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # top_k_rerank kết quả tốt nhất
        best_chunks = [chunk for score, chunk in scored_chunks[:top_k_rerank]]
        
        return "\n\n---\n\n".join(best_chunks)

    def process(self, question, note="", top_k_retrieve=10, top_k_rerank=7):
        # Retrieve & rerank
        context = self.retrieve_and_rerank(question, top_k_retrieve=top_k_retrieve, top_k_rerank=top_k_rerank)
        
        note_text = f"Gợi ý từ hệ thống: {note}\n\n" if str(note).strip() else ""

        # Prompt
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
        # Call LLM
        raw_output = self.llm.generate(prompt)
        
        # Parse output
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
                
                # Đồng bộ numbers với số lượng đáp án thực tế tìm được
                numbers = len(final_list)
                
                return json.dumps({
                    "numbers": numbers,
                    "result": final_result
                }, ensure_ascii=False)

        except Exception:
            pass
            
        # Fallback cứng khi lỗi nặng
        return json.dumps({
            "numbers": 1,
            "result": "A"
        }, ensure_ascii=False)