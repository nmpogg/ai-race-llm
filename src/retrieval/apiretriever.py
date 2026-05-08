import pandas as pd
import re
from rank_bm25 import BM25Okapi

class APIRetriever:
    def __init__(self, api_csv_path):
        self.df_api = pd.read_csv(api_csv_path)
        self.df_api['search_text'] = (
            self.df_api['name'].fillna('') + " " + 
            self.df_api['description'].fillna('') + " " + 
            self.df_api['Example question'].fillna('')
        )
        self.tokenized_corpus = [self._tokenize(t) for t in self.df_api['search_text']]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text):
        return re.findall(r'\b\w+\b', str(text).lower())

    def get_top_apis_df(self, question, k=3):
        """Trả về DataFrame thay vì string để APIAgent tự xử lý"""
        tokenized_query = self._tokenize(question)
        self.df_api['score'] = self.bm25.get_scores(tokenized_query)
        return self.df_api.sort_values(by='score', ascending=False).head(k).reset_index(drop=True)

    def get_top_apis_config(self, question, k=3):
        """Giữ lại để tương thích"""
        top_k_df = self.get_top_apis_df(question, k)
        configs = []
        for _, row in top_k_df.iterrows():
            if row['score'] > 0:
                configs.append(f"Tên API: {row['name']}\nCấu hình:\n{row['Endpoint config']}")
        return "\n---\n".join(configs)