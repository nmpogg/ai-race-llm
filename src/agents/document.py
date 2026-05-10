import json
import os
import pickle
import re

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer


class DocAgent:
    def __init__(
        self,
        llm_service,
        index_dir: str = "./index_data",
        fewshot_loader=None,
        use_ensemble: bool = True,
    ):
        self.llm = llm_service
        self.fewshot = fewshot_loader   # FewShotLoader instance (optional)
        self.use_ensemble = use_ensemble

        print("DocAgent: Đang load Hybrid Retrieval...")
        with open(os.path.join(index_dir, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)
        self.faiss_index = faiss.read_index(
            os.path.join(index_dir, "faiss.index")
        )
        self.embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")
        self.reranker = CrossEncoder(
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        )
        print("DocAgent: Sẵn sàng.")

    # TOKENIZE 
    @staticmethod
    def _tok(text: str) -> list[str]:
        return re.findall(r"\w+", str(text).lower())

    # QUERY EXPANSION
    _EXPAND_PROMPT = (
        "Trích xuất 5-8 từ khóa/cụm từ quan trọng nhất từ câu hỏi dưới đây "
        "để tìm kiếm trong tài liệu kỹ thuật.\n"
        "Chỉ trả về các từ khóa cách nhau bằng dấu phẩy, không giải thích.\n\n"
        "Câu hỏi: {question}\n\n"
        "Từ khóa:"
    )

    def _expand_query(self, question: str) -> str:
        """Dùng LLM extract từ khóa, ghép với câu hỏi gốc để retrieval tốt hơn."""
        try:
            prompt = self._EXPAND_PROMPT.format(question=question[:500])
            keywords = self.llm.generate(prompt, max_tokens=60).strip()
            # Chỉ dùng nếu output hợp lý (không quá dài, không rỗng)
            if 5 < len(keywords) < 200:
                return f"{question} {keywords}"
        except Exception as e:
            print(f"⚠️ Query expansion lỗi: {e}")
        return question

    # RETRIEVAL
    def retrieve_and_rerank(
        self,
        question: str,
        note: str = "",
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
    ) -> str:
        # 1. Expand query
        expanded = self._expand_query(question)

        # 2. FAISS dense search trên expanded query
        q_emb = self.embed_model.encode([expanded], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        _, faiss_idx = self.faiss_index.search(q_emb, top_k_retrieve)
        faiss_results = [
            self.chunks[i] for i in faiss_idx[0] if i < len(self.chunks)
        ]

        # 3. BM25 sparse search
        bm25_scores = self.bm25.get_scores(self._tok(expanded))
        bm25_idx = np.argsort(bm25_scores)[::-1][:top_k_retrieve]
        bm25_results = [
            self.chunks[i] for i in bm25_idx if bm25_scores[i] > 0
        ]

        # 4. Note-aware search: nếu có note thì tìm thêm chunk từ note
        note_results = []
        note_str = str(note).strip()
        if note_str:
            n_emb = self.embed_model.encode([note_str], convert_to_numpy=True)
            faiss.normalize_L2(n_emb)
            _, n_idx = self.faiss_index.search(n_emb, 5)
            note_results = [
                self.chunks[i] for i in n_idx[0] if i < len(self.chunks)
            ]

        # 5. Deduplicate (note chunks ưu tiên đầu tiên)
        seen: set[str] = set()
        combined: list[str] = []
        for chunk in note_results + faiss_results + bm25_results:
            key = chunk[:80]
            if key not in seen:
                seen.add(key)
                combined.append(chunk)

        if not combined:
            return ""

        # 6. Rerank bằng CrossEncoder
        scores = self.reranker.predict([[question, c] for c in combined])
        top_chunks = [
            c
            for _, c in sorted(zip(scores, combined), reverse=True)[
                :top_k_rerank
            ]
        ]
        return "\n\n---\n\n".join(top_chunks)

    # OUTPUT PARSING 
    @staticmethod
    def _extract_letters(text: str) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for m in re.findall(r"[ABCD]", text.upper()):
            if m not in seen:
                seen.add(m)
                result.append(m)
        return result

    def _parse_output(self, raw: str):
        # 1. JSON chuẩn {"result": "A"} hoặc {"numbers":1,"result":"A,B"}
        try:
            m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            if m:
                obj = json.loads(m.group(0))
                letters = self._extract_letters(str(obj.get("result", "")))
                if letters:
                    return len(letters), ",".join(letters)
        except Exception:
            pass

        # 2. Dòng cuối cùng của output (sau CoT thường là kết luận)
        last_lines = [
            ln.strip() for ln in raw.strip().splitlines() if ln.strip()
        ][-3:]
        for line in reversed(last_lines):
            letters = self._extract_letters(line)
            if letters and len(letters) <= 4:
                return len(letters), ",".join(letters)

        # 3. Pattern "Đáp án: A" / "Answer: B,C"
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

    # PROMPTS 
    _COT_PROMPT = """{fewshot}Bạn là chuyên gia trả lời câu hỏi trắc nghiệm dựa trên tài liệu.

[TÀI LIỆU]:
{context}

[HƯỚNG DẪN]:
1. Đọc kỹ câu hỏi và các đáp án.
2. Tìm thông tin liên quan trong tài liệu cho từng đáp án.
3. Loại trừ các đáp án sai.
4. Kết luận bằng JSON: {{"numbers": <số đáp án đúng>, "result": "<chữ cái>"}}
   Ví dụ 1 đáp án: {{"numbers": 1, "result": "B"}}
   Ví dụ 2 đáp án: {{"numbers": 2, "result": "A,C"}}
   QUAN TRỌNG: JSON phải là dòng CUỐI CÙNG, không thêm gì sau đó.

[CÂU HỎI]:
{question}

Suy luận:"""

    _DIRECT_PROMPT = """{fewshot}Đọc tài liệu và chọn đáp án đúng (A/B/C/D).
Chỉ trả về JSON: {{"numbers": 1, "result": "A"}}

[TÀI LIỆU]:
{context}

[CÂU HỎI]:
{question}

JSON:"""

    # ENSEMBLE VOTING
    def _vote(self, answers: list[str | None]) -> tuple[int | None, str | None]:
        """Lấy đáp án xuất hiện nhiều nhất trong danh sách."""
        valid = [a for a in answers if a is not None]
        if not valid:
            return None, None
        counts: dict[str, int] = {}
        for ans in valid:
            counts[ans] = counts.get(ans, 0) + 1
        best = max(counts, key=lambda x: counts[x])
        letters = self._extract_letters(best)
        return len(letters), best

    # MAIN PROCESS
    def process(self, question: str, note: str = "") -> str:
        note_str = str(note).strip()
        context = self.retrieve_and_rerank(question, note=note_str)

        # Few-shot từ example_data
        fewshot_block = ""
        if self.fewshot is not None:
            fewshot_block = self.fewshot.get_doc_fewshot(question)

        # Chuẩn bị context (giới hạn độ dài để tránh OOM)
        ctx_short = context[:2500]

        # Lần 1: Chain-of-thought prompt 
        prompt1 = self._COT_PROMPT.format(
            fewshot=fewshot_block,
            context=ctx_short,
            question=question,
        )
        raw1 = self.llm.generate(prompt1, max_tokens=350)
        n1, r1 = self._parse_output(raw1)

        # Lần 2: Direct prompt (chỉ nếu use_ensemble=True) 
        if self.use_ensemble:
            prompt2 = self._DIRECT_PROMPT.format(
                fewshot=fewshot_block,
                context=ctx_short,
                question=question,
            )
            raw2 = self.llm.generate(prompt2, max_tokens=80)
            n2, r2 = self._parse_output(raw2)

            # Vote: nếu đồng thuận → dùng, nếu khác nhau → dùng CoT (lần 1)
            if r1 and r2:
                if r1 == r2:
                    return json.dumps(
                        {"numbers": n1, "result": r1}, ensure_ascii=False
                    )
                # Không đồng thuận → tin CoT hơn
                return json.dumps(
                    {"numbers": n1, "result": r1}, ensure_ascii=False
                )
            # Một trong hai thành công
            result = r1 or r2
            numbers = n1 or n2
            if result:
                return json.dumps(
                    {"numbers": numbers, "result": result}, ensure_ascii=False
                )
        else:
            # Không ensemble, chỉ dùng kết quả lần 1
            if r1:
                return json.dumps(
                    {"numbers": n1, "result": r1}, ensure_ascii=False
                )

        # Retry cuối
        prompt_retry = (
            f"Câu hỏi: {question}\n"
            f"Tài liệu: {context[:800]}\n"
            f"Chỉ trả về đáp án (A/B/C/D): "
        )
        raw_retry = self.llm.generate(prompt_retry, max_tokens=20)
        letters = self._extract_letters(raw_retry)
        if letters:
            return json.dumps(
                {"numbers": len(letters), "result": ",".join(letters)},
                ensure_ascii=False,
            )

        # Fallback cuối cùng
        return json.dumps({"numbers": 1, "result": "A"}, ensure_ascii=False)