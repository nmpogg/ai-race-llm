import json
import os
import re
import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
import sqlite3


class DocAgent:

    def __init__(
        self,
        llm_service,
        index_dir: str = "./index_data",
        fewshot_loader=None,
        use_ensemble: bool = False,
    ):
        self.llm          = llm_service
        self.fewshot      = fewshot_loader
        self.use_ensemble = use_ensemble

        print("DocAgent: Đang load Hybrid Retrieval...")

        import pickle
        pkl_path = os.path.join(index_dir, "chunk_store.pkl")
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)
        self.id_map = payload["id_map"]
        self.store  = payload["store"]
        self.chunks = [self.store[cid]["content"] for cid in self.id_map]

        self.bm25_db     = os.path.join(index_dir, "bm25_index.db")
        self.faiss_index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        self.embed_model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2")
        self.reranker    = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        print("DocAgent: Sẵn sàng.")

    @staticmethod
    def _tok(text: str) -> list:
        return re.findall(r"\w+", str(text).lower())

    def _build_search_query(self, question: str, note: str) -> str:
        if not note:
            return question
        option_texts = re.findall(r'\b[ABCD][,.\)]\s*(.+?)(?=\n\s*[ABCD][,.\)]|\Z)', note, re.DOTALL)
        option_content = " ".join(t.strip() for t in option_texts if t.strip())
        if option_content:
            return f"{question} {option_content}"
        return f"{question} {note}"

    def _bm25_search(self, query: str, top_k: int = 15) -> list:
        try:
            conn   = sqlite3.connect(self.bm25_db)
            cur    = conn.cursor()
            safe_q = query.replace('"', '""')
            cur.execute(
                'SELECT chunk_id FROM bm25_chunks WHERE chunk_text MATCH ? ORDER BY rank LIMIT ?',
                (f'"{safe_q}"', top_k),
            )
            rows = cur.fetchall()
            conn.close()
            results = []
            for (chunk_id,) in rows:
                if chunk_id in self.store:
                    results.append(self.store[chunk_id]["content"])
            return results
        except Exception:
            try:
                conn    = sqlite3.connect(self.bm25_db)
                cur     = conn.cursor()
                words   = re.findall(r'\w+', query)[:5]
                match_q = " OR ".join(words)
                cur.execute(
                    'SELECT chunk_id FROM bm25_chunks WHERE chunk_text MATCH ? ORDER BY rank LIMIT ?',
                    (match_q, top_k),
                )
                rows = cur.fetchall()
                conn.close()
                results = []
                for (chunk_id,) in rows:
                    if chunk_id in self.store:
                        results.append(self.store[chunk_id]["content"])
                return results
            except Exception:
                return []

    def retrieve_and_rerank(
        self,
        question: str,
        note: str = "",
        top_k_retrieve: int = 15,
        top_k_rerank: int = 5,
    ) -> str:
        note_str   = str(note).strip()
        main_query = self._build_search_query(question, note_str)

        q_emb = self.embed_model.encode([main_query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        _, faiss_idx  = self.faiss_index.search(q_emb, top_k_retrieve)
        faiss_results = [self.chunks[i] for i in faiss_idx[0] if 0 <= i < len(self.chunks)]

        bm25_results = self._bm25_search(main_query, top_k_retrieve)

        note_results = []
        if note_str and len(note_str) > 20:
            n_emb = self.embed_model.encode([question], convert_to_numpy=True)
            faiss.normalize_L2(n_emb)
            _, n_idx     = self.faiss_index.search(n_emb, 5)
            note_results = [self.chunks[i] for i in n_idx[0] if 0 <= i < len(self.chunks)]

        seen: set      = set()
        combined: list = []
        for chunk in note_results + faiss_results + bm25_results:
            key = chunk[:80]
            if key not in seen:
                seen.add(key)
                combined.append(chunk)

        if not combined:
            return ""

        # 25 → 8 pairs 
        combined   = combined[:8]
        scores     = self.reranker.predict([[question, c] for c in combined])
        top_chunks = [
            c for _, c in sorted(zip(scores, combined), reverse=True)[:top_k_rerank]
        ]
        return "\n\n---\n\n".join(top_chunks)

    @staticmethod
    def _extract_letters(text: str) -> list:
        seen: set    = set()
        result: list = []
        for m in re.findall(r"[ABCD]", text.upper()):
            if m not in seen:
                seen.add(m)
                result.append(m)
        return result

    def _parse_output(self, raw: str):
        try:
            m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            if m:
                obj     = json.loads(m.group(0))
                letters = self._extract_letters(str(obj.get("result", "")))
                if letters:
                    return len(letters), ",".join(letters)
        except Exception:
            pass

        last_lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()][-3:]
        for line in reversed(last_lines):
            letters = self._extract_letters(line)
            if letters and len(letters) <= 4:
                return len(letters), ",".join(letters)

        for pat in [
            r"(?:đáp án|answer|kết quả|result|chọn)[^\w]*([ABCD](?:[,\s]+[ABCD])*)",
            r"(?:chính xác|đúng)[^\w]*([ABCD](?:[,\s]+[ABCD])*)",
            r"([ABCD](?:[,\s]+[ABCD])+)",
            r'"([ABCD])"',
        ]:
            m = re.search(pat, raw.upper())
            if m:
                letters = self._extract_letters(m.group(1))
                if letters:
                    return len(letters), ",".join(letters)

        return None, None

    # Think then Answer 
    _ANSWER_PROMPT = """{fewshot}Dựa vào tài liệu, chọn đáp án đúng cho câu hỏi trắc nghiệm.
Có thể có 1 hoặc nhiều đáp án đúng.

[TÀI LIỆU]:
{context}

[CÂU HỎI]:
{question}

Chỉ trả về các chữ cái đáp án đúng, cách nhau bằng dấu phẩy nếu có nhiều đáp án.
Ví dụ 1 đáp án: B
Ví dụ 2 đáp án: A,C
Đáp án:"""

    _JSON_FALLBACK_PROMPT = """Câu hỏi: {question}

Tài liệu: {context}

Trả về JSON: {{"numbers": <số đáp án>, "result": "<chữ cái>"}}
JSON:"""

    def process(self, question: str, note: str = "") -> str:
        note_str      = str(note).strip()
        full_question = f"{question}\n{note_str}" if note_str else question
        context       = self.retrieve_and_rerank(question, note=note_str)
        ctx_short     = context[:3000]

        fewshot_block = ""
        if self.fewshot is not None:
            fewshot_block = self.fewshot.get_doc_fewshot(question)

        # Call 1: chỉ trả chữ cái → max_tokens=10 
        prompt1 = self._ANSWER_PROMPT.format(
            fewshot=fewshot_block,
            context=ctx_short,
            question=full_question,
        )
        raw1    = self.llm.generate(prompt1, max_tokens=10)
        letters = self._extract_letters(raw1.strip().split("\n")[0])

        if letters:
            return json.dumps(
                {"numbers": len(letters), "result": ",".join(letters)},
                ensure_ascii=False,
            )

        # Call 2 (fallback): JSON ngắn → max_tokens=20
        prompt2 = self._JSON_FALLBACK_PROMPT.format(
            question=full_question,
            context=context[:800],
        )
        raw2   = self.llm.generate(prompt2, max_tokens=20)
        n2, r2 = self._parse_output(raw2)
        if r2:
            return json.dumps({"numbers": n2, "result": r2}, ensure_ascii=False)

        # Fallback cuối: parse từ raw1
        n1, r1 = self._parse_output(raw1)
        if r1:
            return json.dumps({"numbers": n1, "result": r1}, ensure_ascii=False)

        return json.dumps({"numbers": 1, "result": "A"}, ensure_ascii=False)