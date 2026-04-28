import os
import pickle
import faiss
import re
import gc
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

def custom_text_splitter(text, chunk_size=800, chunk_overlap=150):
    paragraphs = re.split(r'\n\n+', text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            overlap_start = max(0, len(current_chunk) - chunk_overlap)
            overlap_text = current_chunk[overlap_start:]
            space_idx = overlap_text.find(' ')
            if space_idx != -1:
                overlap_text = overlap_text[space_idx:].strip()
            current_chunk = overlap_text + "\n\n" + para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

        while len(current_chunk) > chunk_size:
            cut_point = current_chunk.rfind('. ', 0, chunk_size)
            if cut_point == -1:
                cut_point = current_chunk.rfind(' ', 0, chunk_size)
            if cut_point == -1:
                cut_point = chunk_size 

            chunks.append(current_chunk[:cut_point].strip())
            overlap_start = max(0, cut_point - chunk_overlap)
            current_chunk = current_chunk[overlap_start:].strip()

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def build_faiss_index(chunks, output_dir):
    print("Đang tải mô hình Embedding...")
    embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")
    
    print("Đang tạo FAISS Index...")
    embeddings = embed_model.encode(chunks, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)
    
    faiss.write_index(faiss_index, os.path.join(output_dir, "faiss.index"))
    print("Đã lưu FAISS Index.")

    del embed_model
    del embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def build_bm25_index(chunks, output_dir):
    print("Đang tạo BM25 Index...")
    def tokenize(t): return re.findall(r'\b\w+\b', str(t).lower())
    
    bm25_index = BM25Okapi((tokenize(chunk) for chunk in chunks))
    
    with open(os.path.join(output_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25_index, f)
        
    print("Đã lưu BM25 Index.")
    
    del bm25_index
    gc.collect()


def build_index(md_file_path, output_dir="./data/knowledge"):
    print("Bắt đầu xây dựng knowledge...")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Đang đọc file và chunking...")
    with open(md_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = custom_text_splitter(text, chunk_size=800, chunk_overlap=150)
    
    with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    print(f"Đã chia thành {len(chunks)} chunks và lưu thành công.")

    del text
    gc.collect()

    build_bm25_index(chunks, output_dir)
    build_faiss_index(chunks, output_dir)

    print(f"Success! Knowledge đã được lưu hoàn tất tại: {output_dir}")

# if __name__ == "__main__":
    # build_index("corpus.md")