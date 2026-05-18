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
        
        # Load chunk_store.pkl - dict {"id_map": [...], "store": {...}}
        with open(os.path.join(index_dir, "chunk_store.pkl"), "rb") as f:
            payload = pickle.load(f)
        self.id_map = payload["id_map"]   # list: faiss_position → chunk_id
        self.store  = payload["store"]    # dict: chunk_id → full chunk
            
        # SQLite
        self.db_path = os.path.join(index_dir, "bm25_index.db")
        if not os.path.exists(self.db_path):
            print(f"Cảnh báo: Không tìm thấy {self.db_path}. Hãy build index trước!")
            
        # FAISS + embedding model
        self.faiss_index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        self.embed_model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2")
        
        # Reranker
        self.reranker = CrossEncoder("AITeamVN/Vietnamese_Reranker")
        print("Đã load thành công DocAgent.")


    def retrieve_bm25_sqlite(self, question: str, top_k: int = 10) -> list[str]:
        """Trả về list chunk_id (str), không phải int index."""
        clean_query = re.sub(r'[^\w\s]', ' ', question).strip()
        if not clean_query:
            return []
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """SELECT chunk_id
                FROM bm25_chunks
                WHERE bm25_chunks MATCH ?
                ORDER BY rank
                LIMIT ?""",
                (clean_query, top_k)
            )
            result_ids = [row[0] for row in cursor.fetchall()]
            conn.close()
            return result_ids
        except sqlite3.OperationalError as e:
            print(f"Lỗi truy vấn BM25: {e}")
            return []


    def _chunk_to_context(self, chunk: dict) -> str:
        """Format 1 chunk thành string context để đưa vào reranker + LLM."""
        meta    = chunk["metadata"]
        content = chunk["content"]
        source  = meta.get("source_file", "Unknown")
        section = meta.get("section", "")

        header_parts = [f"[Trích từ: {source}]"]
        if section:
            header_parts.append(f"[Mục: {section}]")

        # Với table chunk: strip HTML để reranker đọc được
        if chunk["chunk_type"] == "table":
            table_name = meta.get("table_name", "")
            if table_name:
                header_parts.append(f"[Bảng: {table_name}]")
            plain = re.sub(r'<[^>]+>', ' ', content)
            plain = re.sub(r'\s+', ' ', plain).strip()
            body = plain
        else:
            body = content

        return "\n".join(header_parts) + "\n" + body


    def retrieve_and_rerank(
        self,
        question: str,
        top_k_retrieve: int = 10,
        top_k_rerank: int = 7,
    ) -> str:
        q_emb = self.embed_model.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        _, faiss_indices = self.faiss_index.search(q_emb, top_k_retrieve)

        faiss_chunk_ids = [
            self.id_map[int(i)]
            for i in faiss_indices[0]
            if i != -1 and int(i) < len(self.id_map)
        ]

        #  BM25 retrieve → chunk_id 
        bm25_chunk_ids = self.retrieve_bm25_sqlite(question, top_k_retrieve)

        # Merge, dedup, giữ thứ tự (FAISS trước, BM25 sau)
        seen = set()
        combined_chunk_ids = []
        for cid in faiss_chunk_ids + bm25_chunk_ids:
            if cid not in seen and cid in self.store:
                seen.add(cid)
                combined_chunk_ids.append(cid)

        if not combined_chunk_ids:
            return ""
        
        candidate_contexts = [
            self._chunk_to_context(self.store[cid])
            for cid in combined_chunk_ids
        ]

        # Rerank
        cross_scores = self.reranker.predict(
            [[question, ctx] for ctx in candidate_contexts]
        )

        scored = sorted(
            zip(cross_scores, candidate_contexts),
            key=lambda x: x[0],
            reverse=True,
        )

        best = [ctx for _, ctx in scored[:top_k_rerank]]
        return "\n\n---\n\n".join(best)

    def process(self, question, note="", top_k_retrieve=10, top_k_rerank=7):
        # Retrieve & rerank
        context = self.retrieve_and_rerank(question, top_k_retrieve=top_k_retrieve, top_k_rerank=top_k_rerank)
        
        note_text = f"Gợi ý từ hệ thống: {note}\n\n" if str(note).strip() else ""

        # Prompt
        prompt = f"""<|im_start|>system
Bạn là hệ thống trích xuất thông tin tự động dựa trên tài liệu.
Nhiệm vụ của bạn là đọc [TÀI LIỆU CUNG CẤP] và trả lời câu hỏi dưới ĐỊNH DẠNG JSON.
Nếu là câu hỏi vận dụng để tính toán, nếu [TÀI LIỆU CUNG CẤP] không có thông tin hữu ích, hãy tự tính toán dựa trên kiến thức chung của bạn.
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