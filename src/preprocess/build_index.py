import os
import pickle
import faiss
import gc
import torch

import sqlite3
from sentence_transformers import SentenceTransformer


def build_faiss_index(chunks, output_dir):
    print("Đang tải mô hình Embedding...")
    embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")
    
    texts = [chunk["text"] for chunk in chunks]
    
    print("Đang tạo FAISS Index...")
    embeddings = embed_model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)
    
    faiss.write_index(faiss_index, os.path.join(output_dir, "faiss.index"))
    print("Đã lưu FAISS Index thành công tại: " + os.path.join(output_dir, "faiss.index"))

    del embed_model
    del embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def build_bm25_index(chunks, output_dir):
    print("Đang tạo BM25 Index (SQLite FTS5)...")

    db_path = os.path.join(output_dir, "bm25_index.db")
    
    # reset db
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE VIRTUAL TABLE bm25_chunks 
        USING fts5(chunk_id UNINDEXED, chunk_text, tokenize='unicode61');
    ''')
    
    print(f"Đang ghi {len(chunks)} chunks xuống db và xây dựng Vocab...")
    
    cursor.executemany(
        "INSERT INTO bm25_chunks (chunk_id, chunk_text) VALUES (?, ?)",
        [(chunk["id"], chunk["text"]) for chunk in chunks]
    )
    
    conn.commit()
    conn.close()
    
    print(f"Đã lưu BM25 Index thành công tại {db_path}.")
    
    gc.collect()

def build_index(chunks, output_dir="./data/knowledge"):
    print("Bắt đầu xây dựng knowledge...")
    os.makedirs(output_dir, exist_ok=True)

    # Truyền mảng chunks xuống 2 hàm con
    build_bm25_index(chunks, output_dir)
    build_faiss_index(chunks, output_dir)

    print(f"Success! Knowledge đã được lưu hoàn tất tại: {output_dir}")

# if __name__ == "__main__":
    # build_index("corpus.md")