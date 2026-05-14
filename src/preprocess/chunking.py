import re
import os
import json

class SmartDocumentChunker:
    def __init__(self, max_chunk_size=1200):
        self.max_chunk_size = max_chunk_size

    def _clean_text(self, text):
        # Xóa thẻ picture omitted
        text = re.sub(r'\*\*==> picture \[.*?\] intentionally omitted <==\*\*', '', text)
        # Xóa dấu in đậm **, in nghiêng _
        text = re.sub(r'[\*_]', '', text)
        # Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def stream_and_chunk(self, file_path):
        current_paragraph_lines = []
        
        current_source = "Unknown_Document"
        current_header = "" 
        chunk_id = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()

                # metadata: source
                if stripped_line.startswith("# Tài liệu:"):
                    current_source = stripped_line.replace("# Tài liệu:", "").strip()
                    current_header = "" # Reset header khi sang tài liệu mới
                    continue

                # header
                if stripped_line.startswith("##"):
                    current_header = self._clean_text(stripped_line.replace("##", ""))
                    continue

                if "intentionally omitted" in stripped_line:
                    continue

                # 4. Xử lý băm đoạn văn khi gặp dòng trống
                if not stripped_line:
                    if current_paragraph_lines:
                        raw_para = " ".join(current_paragraph_lines)
                        clean_para = self._clean_text(raw_para)

                        if len(clean_para) > 50: 
                            # Gắn Header vào đầu đoạn văn
                            if current_header:
                                chunk_text = f"[{current_header}] {clean_para}"
                            else:
                                chunk_text = clean_para

                            # Kiểm tra giới hạn an toàn
                            if len(chunk_text) > self.max_chunk_size:
                                cut_point = chunk_text.rfind('. ', 0, self.max_chunk_size)
                                if cut_point == -1: cut_point = self.max_chunk_size
                                
                                yield {
                                    "id": chunk_id,
                                    "text": chunk_text[:cut_point + 1].strip(),
                                    "metadata": {"source": current_source}
                                }
                                chunk_id += 1
                                chunk_text = chunk_text[cut_point + 1:].strip()

                            if chunk_text:
                                yield {
                                    "id": chunk_id,
                                    "text": chunk_text,
                                    "metadata": {"source": current_source}
                                }
                                chunk_id += 1

                        current_paragraph_lines = [] 
                else:
                    current_paragraph_lines.append(stripped_line)

        # Xử lý đoạn cuối cùng khi hết file
        if current_paragraph_lines:
            raw_para = " ".join(current_paragraph_lines)
            clean_para = self._clean_text(raw_para)
            if len(clean_para) > 50:
                chunk_text = f"[{current_header}] {clean_para}" if current_header else clean_para
                yield {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {"source": current_source}
                }

if __name__ == "__main__":
    md_file_path = "../../data/markdown/corpus.md"
    output_dir = "../../data/knowledge"

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Bắt đầu đọc và chunk file: {md_file_path}")
    
    chunker = SmartDocumentChunker(max_chunk_size=3000)
    
    chunks = list(chunker.stream_and_chunk(md_file_path))
    print(f"Đã tạo thành công {len(chunks)} chunks!")
    
    # save chunk to file .json
    chunks_json_path = os.path.join(output_dir, "chunks.json")

    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Đã lưu toàn bộ dữ liệu ra file: {chunks_json_path}")
