import os
import re
import json
import glob
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class FewShotLoader:
    def __init__(self, example_dir: str, embed_model: SentenceTransformer, top_k: int = 3):
        self.top_k    = top_k
        self.embedder = embed_model

        self.doc_examples: list[dict] = []
        self.api_examples: list[dict] = []  # {"question": ..., "answer_json": ..., "func_code": ...}

        self.doc_embs: np.ndarray | None = None
        self.api_embs: np.ndarray | None = None

        self._load(example_dir)

    # LOAD 

    def _load(self, example_dir: str):
        if not os.path.isdir(example_dir):
            print(f"⚠️ FewShotLoader: Không tìm thấy '{example_dir}'.")
            return

        # Load CSV (format cũ)
        for fpath in glob.glob(os.path.join(example_dir, "**", "*.csv"), recursive=True):
            try:
                df = pd.read_csv(fpath)
                df.columns = [c.strip().lower() for c in df.columns]
                self._parse_df(df)
            except Exception as e:
                print(f"⚠️ FewShotLoader CSV lỗi '{fpath}': {e}")

        # Load Excel (format mới: 2 sheet question + result)
        for fpath in glob.glob(os.path.join(example_dir, "**", "*.xlsx"), recursive=True):
            try:
                self._load_excel(fpath)
            except Exception as e:
                print(f"⚠️ FewShotLoader Excel lỗi '{fpath}': {e}")

        self._build_embeddings()

    def _load_excel(self, fpath: str):
        xls    = pd.ExcelFile(fpath)
        sheets = xls.sheet_names
        q_sheet = next((s for s in sheets if "question" in s.lower()), None)
        r_sheet = next((s for s in sheets if "result"   in s.lower()
                                          or "answer"   in s.lower()), None)
        if q_sheet is None or r_sheet is None:
            return

        df_q = pd.read_excel(fpath, sheet_name=q_sheet)
        df_r = pd.read_excel(fpath, sheet_name=r_sheet)
        df_q.columns = [c.strip().lower() for c in df_q.columns]
        df_r.columns = [c.strip().lower() for c in df_r.columns]

        merged = pd.merge(df_q, df_r, on="id", how="inner")

        # Detect question column
        q_col = next((c for c in ["fun_question", "question"] if c in merged.columns), None)
        if q_col is None:
            return

        for _, row in merged.iterrows():
            question  = str(row.get(q_col, "")).strip()
            func_code = str(row.get("func_code", "")).strip()
            func_param = str(row.get("func_param", "")).strip()

            if not question:
                continue

            if func_code == "call_api":
                try:
                    json.loads(func_param)  # validate
                    self.api_examples.append({
                        "question":    question,
                        "answer_json": func_param,
                        "func_code":   func_code,
                    })
                except Exception:
                    pass
            elif func_code == "call_document":
                answer_str = str(row.get("func_param", "")).strip()
                letters    = re.findall(r"[ABCD]", answer_str.upper())
                if letters:
                    self.doc_examples.append({
                        "question":       question,
                        "options":        None,
                        "answer_letters": letters,
                        "answer_json":    json.dumps(
                            {"numbers": len(letters), "result": ",".join(letters)},
                            ensure_ascii=False,
                        ),
                    })

    def _parse_df(self, df: pd.DataFrame):
        cols = set(df.columns)
        if {"question", "a", "b", "c", "d"} <= cols:
            answer_col = next(
                (c for c in ["answer", "correct", "correct_answer", "label", "func_param"] if c in cols), None)
            for _, row in df.iterrows():
                q   = str(row.get("question", "")).strip()
                ans = str(row.get(answer_col, "")).strip() if answer_col else ""
                if not q:
                    continue
                if ans and not ans.startswith("{"):
                    letters = re.findall(r"[ABCD]", ans.upper())
                    if letters:
                        self.doc_examples.append({
                            "question": q,
                            "options": {k: str(row.get(k.lower(), "")) for k in "ABCD"},
                            "answer_letters": letters,
                            "answer_json": json.dumps(
                                {"numbers": len(letters), "result": ",".join(letters)},
                                ensure_ascii=False),
                        })
            return

        answer_col = next(
            (c for c in ["func_param", "answer", "correct", "func_answer"] if c in cols), None)
        if "question" in cols and answer_col:
            for _, row in df.iterrows():
                q   = str(row.get("question", "")).strip()
                ans = str(row.get(answer_col, "")).strip()
                if not q or not ans:
                    continue
                try:
                    parsed = json.loads(ans)
                    if "path" in parsed or "func_code" in str(parsed):
                        self.api_examples.append({"question": q, "answer_json": ans, "func_code": "call_api"})
                        continue
                except Exception:
                    pass
                letters = re.findall(r"[ABCD]", ans.upper())
                if letters:
                    self.doc_examples.append({
                        "question": q, "options": None,
                        "answer_letters": letters,
                        "answer_json": json.dumps(
                            {"numbers": len(letters), "result": ",".join(letters)},
                            ensure_ascii=False),
                    })

    def _build_embeddings(self):
        if self.doc_examples:
            self.doc_embs = self.embedder.encode(
                [ex["question"] for ex in self.doc_examples],
                convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            print(f"✅ FewShotLoader: {len(self.doc_examples)} MCQ examples.")

        if self.api_examples:
            self.api_embs = self.embedder.encode(
                [ex["question"] for ex in self.api_examples],
                convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            print(f"✅ FewShotLoader: {len(self.api_examples)} API examples.")

    # RETRIEVAL 
    def _find_similar(self, query: str, examples: list[dict], embs: np.ndarray) -> list[dict]:
        if not examples or embs is None:
            return []
        q_emb  = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores = (embs @ q_emb.T).squeeze()
        # Nếu chỉ có 1 example thì scores là scalar
        if scores.ndim == 0:
            return [examples[0]]
        top_idx = np.argsort(scores)[::-1][: self.top_k]
        return [examples[i] for i in top_idx]

    def get_doc_fewshot(self, query: str) -> str:
        similar = self._find_similar(query, self.doc_examples, self.doc_embs)
        if not similar:
            return ""
        lines = ["=== VÍ DỤ THAM KHẢO ==="]
        for i, ex in enumerate(similar, 1):
            lines.append(f"\nVí dụ {i}:")
            lines.append(f"Câu hỏi: {ex['question']}")
            if ex.get("options"):
                for k, v in ex["options"].items():
                    lines.append(f"  {k}. {v}")
            lines.append(f"Đáp án: {ex['answer_json']}")
        lines.append("=== HẾT VÍ DỤ ===\n")
        return "\n".join(lines)

    def get_api_fewshot(self, query: str, k: int | None = None) -> str:
        """
        Hướng 2: Trả về top-k examples gần nhất, format đầy đủ để LLM học pattern.
        k=None → dùng self.top_k.
        """
        top_k_orig = self.top_k
        if k is not None:
            self.top_k = k
        similar = self._find_similar(query, self.api_examples, self.api_embs)
        self.top_k = top_k_orig

        if not similar:
            return ""
        lines = ["=== VÍ DỤ THAM KHẢO (học theo pattern này) ==="]
        for i, ex in enumerate(similar, 1):
            lines.append(f"\nVí dụ {i}:")
            lines.append(f"Câu hỏi: {ex['question']}")
            lines.append(f"Kết quả JSON:\n{ex['answer_json']}")
        lines.append("\n=== HẾT VÍ DỤ ===\n")
        return "\n".join(lines)

    def get_api_fewshot_by_path(self, path_fragment: str, k: int = 2) -> str:
        """
        Lấy examples có path chứa path_fragment — dùng khi đã biết API group.
        Ví dụ: path_fragment="leakage-rate" → lấy examples của leakage-rate APIs.
        """
        filtered = []
        for ex in self.api_examples:
            try:
                ans = json.loads(ex["answer_json"])
                if path_fragment.lower() in ans.get("path", "").lower():
                    filtered.append(ex)
            except Exception:
                pass
        if not filtered:
            return ""
        # Lấy tối đa k
        selected = filtered[:k]
        lines = [f"=== VÍ DỤ CỦA API NHÓM '{path_fragment}' ==="]
        for i, ex in enumerate(selected, 1):
            lines.append(f"\nVí dụ {i}:")
            lines.append(f"Câu hỏi: {ex['question']}")
            lines.append(f"Kết quả JSON:\n{ex['answer_json']}")
        lines.append("\n=== HẾT VÍ DỤ ===\n")
        return "\n".join(lines)