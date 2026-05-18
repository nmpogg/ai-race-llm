import os
import gc
import json
import pickle
import sqlite3
import re

import torch
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


def load_chunks_from_json(json_path: str) -> list[dict]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data["chunks"]
    return data


def _chunk_to_embed_text(chunk: dict) -> str:
    meta = chunk["metadata"]
    content = chunk["content"]

    if chunk["chunk_type"] == "table":
        parts = []
        if meta.get("section"):
            parts.append(f"Mục: {meta['section']}")
        if meta.get("table_name"):
            parts.append(f"Bảng: {meta['table_name']}")
        plain = re.sub(r'<[^>]+>', ' ', content)
        plain = re.sub(r'\s+', ' ', plain).strip()
        parts.append(plain)
        return "\n".join(parts)

    return content


def _chunk_to_bm25_text(chunk: dict) -> str:
    meta = chunk["metadata"]
    parts = []
    if meta.get("section"):
        parts.append(meta["section"])
    if meta.get("table_name"):
        parts.append(meta["table_name"])
    plain = re.sub(r'<[^>]+>', ' ', chunk["content"])
    plain = re.sub(r'\s+', ' ', plain).strip()
    parts.append(plain)
    return " ".join(parts)


def build_chunk_store(chunks: list[dict], output_dir: str) -> dict:
    """
    Lưu 2 thứ vào 1 file .pkl:
      - id_map  : list[str] — mapping [faiss_position] → chunk_id
      - store   : dict      — {chunk_id: full_chunk}

    Gộp chung 1 file để đảm bảo thứ tự id_map
    luôn khớp với thứ tự add vào FAISS index.
    """
    id_map = [c["chunk_id"] for c in chunks]          # giữ đúng thứ tự
    store  = {c["chunk_id"]: c for c in chunks}

    payload = {
        "id_map": id_map,   # list — index by faiss position
        "store": store,     # dict — lookup by chunk_id
    }

    pkl_path = os.path.join(output_dir, "chunk_store.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    store_path = os.path.join(output_dir, "chunk_store.json")
    Path(store_path).write_text(
        json.dumps(store, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Đã lưu Chunk Store: {store_path} ({len(store)} chunks)")

    return payload


def load_chunk_store(output_dir: str) -> tuple[list[str], dict]:
    """
    Load chunk_store.pkl, trả về (id_map, store).
    Dùng khi retrieve:
        id_map, store = load_chunk_store("./data/knowledge")
        chunk = store[id_map[faiss_position]]
    """
    pkl_path = os.path.join(output_dir, "chunk_store.pkl")
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    return payload["id_map"], payload["store"]



def build_faiss_index(chunks: list[dict], output_dir: str):
    print("Đang tải mô hình Embedding...")
    embed_model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2")

    texts = [_chunk_to_embed_text(c) for c in chunks]

    print(f"Đang embed {len(texts)} chunks...")
    embeddings = embed_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)

    index_path = os.path.join(output_dir, "faiss.index")
    faiss.write_index(faiss_index, index_path)
    print(f"Đã lưu FAISS Index: {index_path} "
          f"({faiss_index.ntotal} vectors, dim={dimension})")

    del embed_model, embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_bm25_index(chunks: list[dict], output_dir: str):
    print("Đang tạo BM25 Index (SQLite FTS5)...")

    db_path = os.path.join(output_dir, "bm25_index.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE VIRTUAL TABLE bm25_chunks
        USING fts5(
            chunk_id    UNINDEXED,
            chunk_type  UNINDEXED,
            source_file UNINDEXED,
            section     UNINDEXED,
            table_name  UNINDEXED,
            chunk_text,
            tokenize='unicode61'
        );
    """)

    rows = [
        (
            c["chunk_id"],
            c["chunk_type"],
            c["metadata"]["source_file"],
            c["metadata"].get("section", ""),
            c["metadata"].get("table_name") or "",
            _chunk_to_bm25_text(c),
        )
        for c in chunks
    ]

    cursor.executemany(
        """INSERT INTO bm25_chunks
           (chunk_id, chunk_type, source_file, section, table_name, chunk_text)
           VALUES (?, ?, ?, ?, ?, ?)""",
        rows,
    )

    conn.commit()
    conn.close()
    print(f"Đã lưu BM25 Index: {db_path} ({len(chunks)} chunks)")
    gc.collect()


def build_index(
    chunks_or_path,
    output_dir: str = "./data/knowledge",
):
    print("Bắt đầu xây dựng knowledge base...")
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(chunks_or_path, (str, Path)):
        print(f"Loading chunks từ {chunks_or_path}...")
        chunks = load_chunks_from_json(str(chunks_or_path))
    else:
        chunks = chunks_or_path

    print(f"Tổng chunks: {len(chunks)} "
          f"({sum(1 for c in chunks if c['chunk_type']=='table')} tables, "
          f"{sum(1 for c in chunks if c['chunk_type']=='text')} text)")

    # Thứ tự quan trọng: build_chunk_store trước FAISS
    # để đảm bảo id_map khớp với thứ tự embed
    build_chunk_store(chunks, output_dir)
    build_bm25_index(chunks, output_dir)
    build_faiss_index(chunks, output_dir)

    print(f"\n✓ Knowledge base sẵn sàng tại: {output_dir}")
    print(f"  - chunk_store.pkl  (id_map + store)")
    print(f"  - faiss.index")
    print(f"  - bm25_index.db")


def retrieve_example(query: str, output_dir: str, top_k: int = 5):
    """Ví dụ retrieve flow dùng chunk_store.pkl."""
    from sentence_transformers import SentenceTransformer

    # Load store
    id_map, store = load_chunk_store(output_dir)

    # Load FAISS
    faiss_index = faiss.read_index(os.path.join(output_dir, "faiss.index"))

    # Embed query
    model = SentenceTransformer("keepitreal/vietnamese-sbert")
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    # Search
    scores, indices = faiss_index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk_id = id_map[idx]          # faiss position → chunk_id
        chunk    = store[chunk_id]      # chunk_id → full chunk
        results.append({"score": float(score), "chunk": chunk})

    return results


if __name__ == "__main__":
    build_index("all_chunks.json", output_dir="./data/knowledge")