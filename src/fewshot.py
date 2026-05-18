import os
import re
import json
import glob
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class FewShotLoader:
    def __init__(
        self,
        example_dir: str,
        embed_model: SentenceTransformer,
        top_k: int = 2,
    ):
        self.top_k = top_k
        self.embedder = embed_model

        self.doc_examples: list[dict] = []   # MCQ examples
        self.api_examples: list[dict] = []   # API examples

        self.doc_embs: np.ndarray | None = None
        self.api_embs: np.ndarray | None = None

        self._load(example_dir)

    # LOAD 

    def _load(self, example_dir: str):
        if not os.path.isdir(example_dir):
            print(f"⚠️ FewShotLoader: Không tìm thấy thư mục '{example_dir}'. Bỏ qua few-shot.")
            return

        csv_files = glob.glob(os.path.join(example_dir, "**", "*.csv"), recursive=True)
        csv_files += glob.glob(os.path.join(example_dir, "*.csv"))
        csv_files = list(set(csv_files))

        for fpath in csv_files:
            try:
                df = pd.read_csv(fpath)
                df.columns = [c.strip().lower() for c in df.columns]
                self._parse_file(df, fpath)
            except Exception as e:
                print(f"⚠️ FewShotLoader: Lỗi đọc '{fpath}': {e}")

        # Build embeddings
        if self.doc_examples:
            questions = [ex["question"] for ex in self.doc_examples]
            self.doc_embs = self.embedder.encode(
                questions, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            )
            print(f"✅ FewShotLoader: {len(self.doc_examples)} MCQ examples đã load.")

        if self.api_examples:
            questions = [ex["question"] for ex in self.api_examples]
            self.api_embs = self.embedder.encode(
                questions, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            )
            print(f"✅ FewShotLoader: {len(self.api_examples)} API examples đã load.")

    def _parse_file(self, df: pd.DataFrame, fpath: str):
        cols = set(df.columns)

        # MCQ format: có cột A, B, C, D (các option) 
        if {"question", "a", "b", "c", "d"} <= cols:
            answer_col = next(
                (c for c in ["answer", "correct", "correct_answer", "label", "func_param"]
                 if c in cols),
                None,
            )
            for _, row in df.iterrows():
                q = str(row.get("question", "")).strip()
                if not q:
                    continue
                ans = str(row.get(answer_col, "")).strip() if answer_col else ""
                # ans có thể là "A", "B,C", hoặc JSON string
                if ans and not ans.startswith("{"):
                    letters = re.findall(r"[ABCD]", ans.upper())
                    if letters:
                        self.doc_examples.append({
                            "question": q,
                            "options": {
                                "A": str(row.get("a", "")),
                                "B": str(row.get("b", "")),
                                "C": str(row.get("c", "")),
                                "D": str(row.get("d", "")),
                            },
                            "answer_letters": letters,
                            "answer_json": json.dumps(
                                {"numbers": len(letters), "result": ",".join(letters)},
                                ensure_ascii=False,
                            ),
                        })
            return

        # API format: có cột func_param hoặc answer dạng JSON 
        answer_col = next(
            (c for c in ["func_param", "answer", "correct", "func_answer"]
             if c in cols),
            None,
        )
        if "question" in cols and answer_col:
            for _, row in df.iterrows():
                q = str(row.get("question", "")).strip()
                ans = str(row.get(answer_col, "")).strip()
                if not q or not ans:
                    continue
                # Thử parse JSON → API example
                try:
                    parsed = json.loads(ans)
                    if "path" in parsed or "func_code" in str(parsed):
                        self.api_examples.append({
                            "question": q,
                            "answer_json": ans,
                        })
                        continue
                except Exception:
                    pass
                # Nếu ans là đáp án chữ cái → MCQ example không có options
                letters = re.findall(r"[ABCD]", ans.upper())
                if letters:
                    self.doc_examples.append({
                        "question": q,
                        "options": None,
                        "answer_letters": letters,
                        "answer_json": json.dumps(
                            {"numbers": len(letters), "result": ",".join(letters)},
                            ensure_ascii=False,
                        ),
                    })

    # RETRIEVAL 

    def _find_similar(
        self,
        query: str,
        examples: list[dict],
        embs: np.ndarray,
    ) -> list[dict]:
        if not examples or embs is None:
            return []
        q_emb = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        scores = (embs @ q_emb.T).squeeze()
        top_idx = np.argsort(scores)[::-1][: self.top_k]
        return [examples[i] for i in top_idx]

    def get_doc_fewshot(self, query: str) -> str:
        """Trả về chuỗi few-shot MCQ để inject vào prompt."""
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

    def get_api_fewshot(self, query: str) -> str:
        """Trả về chuỗi few-shot API để inject vào prompt."""
        similar = self._find_similar(query, self.api_examples, self.api_embs)
        if not similar:
            return ""

        lines = ["=== VÍ DỤ THAM KHẢO ==="]
        for i, ex in enumerate(similar, 1):
            lines.append(f"\nVí dụ {i}:")
            lines.append(f"Câu hỏi: {ex['question']}")
            lines.append(f"Kết quả: {ex['answer_json']}")
        lines.append("=== HẾT VÍ DỤ ===\n")
        return "\n".join(lines)