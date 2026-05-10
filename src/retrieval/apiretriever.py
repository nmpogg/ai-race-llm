import re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class APIRetriever:
    RRF_K = 60

    def __init__(
        self,
        api_csv_path: str,
        embed_model: SentenceTransformer | None = None,
        embed_model_name: str = "keepitreal/vietnamese-sbert",
    ):
        self.df_api = pd.read_csv(api_csv_path)

        # Tạo search text tổng hợp từ các cột thông tin
        self.df_api["_search_text"] = (
            self.df_api["name"].fillna("") + " "
            + self.df_api["description"].fillna("") + " "
            + self.df_api["Example question"].fillna("")
        )

        # BM25 
        self._corpus_tokens = [
            self._tok(t) for t in self.df_api["_search_text"]
        ]
        self.bm25 = BM25Okapi(self._corpus_tokens)

        # SBERT semantic 
        # Nhận embed_model từ bên ngoài (dùng chung với DocAgent) nếu có,
        # nếu không thì tự load để không tốn thêm VRAM
        if embed_model is not None:
            self.embedder = embed_model
        else:
            print("APIRetriever: Đang load embedding model...")
            self.embedder = SentenceTransformer(embed_model_name)

        self._corpus_embs = self.embedder.encode(
            self.df_api["_search_text"].tolist(),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        print(f"APIRetriever: Sẵn sàng ({len(self.df_api)} APIs).")

    @staticmethod
    def _tok(text: str) -> list[str]:
        return re.findall(r"\w+", str(text).lower())

    def get_top_apis_df(self, question: str, k: int = 5) -> pd.DataFrame:
        n = len(self.df_api)

        # BM25 rank
        bm25_scores = self.bm25.get_scores(self._tok(question))
        bm25_rank = np.argsort(bm25_scores)[::-1]

        # SBERT semantic rank
        q_emb = self.embedder.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True
        )
        sem_scores = (self._corpus_embs @ q_emb.T).squeeze()
        sem_rank = np.argsort(sem_scores)[::-1]

        # Reciprocal Rank Fusion
        rrf = np.zeros(n)
        for pos, idx in enumerate(bm25_rank):
            rrf[idx] += 1.0 / (self.RRF_K + pos + 1)
        for pos, idx in enumerate(sem_rank):
            rrf[idx] += 1.0 / (self.RRF_K + pos + 1)

        self.df_api["score"] = rrf
        return (
            self.df_api.sort_values("score", ascending=False)
            .head(k)
            .reset_index(drop=True)
        )