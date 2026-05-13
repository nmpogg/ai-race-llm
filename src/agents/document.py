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
        use_ensemble: bool = False,
    ):
        self.llm          = llm_service
        self.fewshot      = fewshot_loader
        self.use_ensemble = use_ensemble

        print("DocAgent: Đang load Hybrid Retrieval...")
        with open(os.path.join(index_dir, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)
        self.faiss_index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        self.embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")
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

    def retrieve_and_rerank(
        self,
        question: str,
        note: str = "",
        top_k_retrieve: int = 12,   # giảm 15→12
        top_k_rerank: int = 4,      # giảm 5→4
    ) -> str:
        note_str   = str(note).strip()
        main_query = self._build_search_query(question, note_str)

        q_emb = self.embed_model.encode([main_query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        _, faiss_idx  = self.faiss_index.search(q_emb, top_k_retrieve)
        faiss_results = [self.chunks[i] for i in faiss_idx[0] if i < len(self.chunks)]

        bm25_scores  = self.bm25.get_scores(self._tok(main_query))
        bm25_idx     = np.argsort(bm25_scores)[::-1][:top_k_retrieve]
        bm25_results = [self.chunks[i] for i in bm25_idx if bm25_scores[i] > 0]

        note_results = []
        if note_str and len(note_str) > 20:
            n_emb = self.embed_model.encode([question], convert_to_numpy=True)
            faiss.normalize_L2(n_emb)
            _, n_idx     = self.faiss_index.search(n_emb, 4)  # giảm 5→4
            note_results = [self.chunks[i] for i in n_idx[0] if i < len(self.chunks)]

        seen: set      = set()
        combined: list = []
        for chunk in note_results + faiss_results + bm25_results:
            key = chunk[:80]
            if key not in seen:
                seen.add(key)
                combined.append(chunk)

        if not combined:
            return ""

        combined = combined[:20]  # giảm 25→20
        scores   = self.reranker.predict([[question, c] for c in combined])
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
        # FIX: lấy JSON cuối cùng có "result" — LLM thường suy luận trước, kết luận sau
        all_matches = list(re.finditer(r"\{[^{}]*\}", raw, re.DOTALL))
        for m in reversed(all_matches):
            try:
                obj     = json.loads(m.group(0))
                letters = self._extract_letters(str(obj.get("result", "")))
                if letters:
                    return len(letters), ",".join(letters)
            except Exception:
                continue

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

    # PROMPTS

    _COT_PROMPT = """{fewshot}Bạn là chuyên gia trả lời câu hỏi trắc nghiệm dựa trên tài liệu.

[TÀI LIỆU]:
{context}

[HƯỚNG DẪN]:
1. Đọc kỹ câu hỏi và TẤT CẢ các đáp án A, B, C, D.
2. Tìm thông tin chính xác trong tài liệu cho từng đáp án — ưu tiên số liệu/từ ngữ khớp chính xác.
3. Loại trừ đáp án sai dựa trên tài liệu.
4. Có thể có 1 hoặc nhiều đáp án đúng.
5. Kết luận bằng JSON: {{"numbers": <số đáp án đúng>, "result": "<chữ cái>"}}
   Ví dụ 1 đáp án: {{"numbers": 1, "result": "B"}}
   Ví dụ 2 đáp án: {{"numbers": 2, "result": "A,C"}}
   QUAN TRỌNG: JSON phải là dòng CUỐI CÙNG, không thêm gì sau đó.

[CÂU HỎI]:
{question}

Suy luận:"""

    _DIRECT_PROMPT = """{fewshot}Đọc tài liệu và chọn đáp án đúng.
Chỉ trả về JSON: {{"numbers": 1, "result": "A"}}

[TÀI LIỆU]:
{context}

[CÂU HỎI]:
{question}

JSON:"""

    _FORCE_PROMPT = """Câu hỏi trắc nghiệm:
{question}

Tài liệu tham khảo:
{context}

Chỉ trả về đúng 1 ký tự là đáp án đúng nhất (A, B, C hoặc D):"""

    def process(self, question: str, note: str = "") -> str:
        note_str      = str(note).strip()
        full_question = f"{question}\n{note_str}" if note_str else question
        context       = self.retrieve_and_rerank(question, note=note_str)

        fewshot_block = ""
        if self.fewshot is not None:
            fewshot_block = self.fewshot.get_doc_fewshot(question)

        ctx_short = context[:2800]  # giảm 3000→2800

        if self.use_ensemble:
            prompts = [
                self._COT_PROMPT.format(fewshot=fewshot_block, context=ctx_short, question=full_question),
                self._DIRECT_PROMPT.format(fewshot=fewshot_block, context=ctx_short, question=full_question),
                self._COT_PROMPT.format(fewshot="", context=ctx_short, question=full_question),
            ]
            max_tokens_list = [220, 50, 220]
            answers = []
            for prompt, max_tok in zip(prompts, max_tokens_list):
                raw = self.llm.generate(prompt, max_tokens=max_tok)
                _, r = self._parse_output(raw)
                answers.append(r)
            valid = [a for a in answers if a is not None]
            if valid:
                counts = {}
                for a in valid:
                    counts[a] = counts.get(a, 0) + 1
                best    = max(counts, key=lambda x: counts[x])
                letters = self._extract_letters(best)
                return json.dumps({"numbers": len(letters), "result": best}, ensure_ascii=False)

        # FAST PATH: COT → Direct → Force
        prompt = self._COT_PROMPT.format(
            fewshot=fewshot_block, context=ctx_short, question=full_question
        )
        raw = self.llm.generate(prompt, max_tokens=220)  # giảm 300→220
        n, r = self._parse_output(raw)
        if r:
            return json.dumps({"numbers": n, "result": r}, ensure_ascii=False)

        prompt_direct = self._DIRECT_PROMPT.format(
            fewshot="", context=ctx_short, question=full_question
        )
        raw2 = self.llm.generate(prompt_direct, max_tokens=40)  # giảm 60→40
        n2, r2 = self._parse_output(raw2)
        if r2:
            return json.dumps({"numbers": n2, "result": r2}, ensure_ascii=False)

        prompt_force = self._FORCE_PROMPT.format(
            question=full_question, context=context[:600]
        )
        raw_force = self.llm.generate(prompt_force, max_tokens=5)
        letters   = self._extract_letters(raw_force)
        if letters:
            return json.dumps({"numbers": 1, "result": letters[0]}, ensure_ascii=False)

        return json.dumps({"numbers": 1, "result": "A"}, ensure_ascii=False)