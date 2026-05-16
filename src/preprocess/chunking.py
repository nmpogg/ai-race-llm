import re
import os
import json

class MarkdownChunker:
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
    
    chunker = MarkdownChunker(max_chunk_size=3000)
    
    chunks = list(chunker.stream_and_chunk(md_file_path))
    print(f"Đã tạo thành công {len(chunks)} chunks!")
    
    # save chunk to file .json
    chunks_json_path = os.path.join(output_dir, "chunks.json")

    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Đã lưu toàn bộ dữ liệu ra file: {chunks_json_path}")




import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import hashlib
import json


@dataclass
class Chunk:
    chunk_id: str
    content: str
    chunk_type: str          # "text" | "table"
    source_file: str
    page_hint: Optional[int] # trang ước tính (nếu có)
    section: str             # heading gần nhất phía trên
    table_name: Optional[str]# tên bảng (nếu là table chunk)
    chunk_index: int         # thứ tự trong file
    total_chunks: int        # tổng số chunk (fill sau)
    char_count: int
    token_estimate: int      # ước tính token (~4 chars/token)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chunk_type": self.chunk_type,
            "metadata": {
                "source_file": self.source_file,
                "section": self.section,
                "table_name": self.table_name,
                "chunk_index": self.chunk_index,
                "total_chunks": self.total_chunks,
                "char_count": self.char_count,
                "token_estimate": self.token_estimate,
            }
        }


# ── Patterns ──────────────────────────────────────────────────────────────────

# Heading: dòng bắt đầu bằng số + dấu chấm, hoặc chữ hoa toàn phần ngắn
HEADING_PATTERN = re.compile(
    r'^(\d+[\.\d]*[\.\s]|[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ]{4,})',
    re.MULTILINE | re.UNICODE
)

# Tên bảng: "Bảng X." hoặc "Bảng X:" hoặc "Table X"
TABLE_NAME_PATTERN = re.compile(
    r'^(Bảng\s+[\d\.]+[\.:]?\s*.+|Table\s+[\d\.]+[\.:]?\s*.+|BẢNG\s+[\d\.]+[\.:]?\s*.+)',
    re.IGNORECASE
)

# HTML table block
HTML_TABLE_PATTERN = re.compile(
    r'<table>.*?</table>',
    re.DOTALL
)


def _generate_id(content: str, index: int, source: str) -> str:
    h = hashlib.md5(f"{source}_{index}_{content[:50]}".encode()).hexdigest()[:8]
    return f"chunk_{index:04d}_{h}"


def _estimate_tokens(text: str) -> int:
    # Tiếng Việt có nhiều dấu, ~3.5 chars/token là hợp lý hơn 4
    return max(1, int(len(text) / 3.5))


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    # "1. Xxx", "1.1 Xxx", "1.1.1 Xxx"
    if re.match(r'^\d+[\.\d]*[\.\s]\s*\S', line):
        return True
    # "PHỤ LỤC", "KẾT LUẬN", "GIỚI THIỆU" — toàn hoa, không quá dài
    if line.isupper() and 3 <= len(line.split()) <= 8:
        return True
    return False


def _extract_table_name(text_before_table: str) -> Optional[str]:
    """
    Tìm tên bảng trong vài dòng cuối trước thẻ <table>.
    Ưu tiên pattern 'Bảng X.' gần nhất.
    """
    lines = [l.strip() for l in text_before_table.strip().split('\n') if l.strip()]
    # Quét từ dưới lên, tối đa 3 dòng
    for line in reversed(lines[-3:]):
        if TABLE_NAME_PATTERN.match(line):
            return line
    return None


def _split_text_into_paragraphs(text: str, max_chars: int = 1500) -> list[str]:
    """
    Tách text thành các paragraph chunks.
    - Ưu tiên tách tại paragraph break (dòng trống)
    - Nếu paragraph quá dài → tách tiếp tại câu
    - Nếu paragraph quá ngắn → merge với paragraph kế
    """
    # Tách theo dòng trống
    raw_paras = re.split(r'\n{2,}', text.strip())
    raw_paras = [p.strip() for p in raw_paras if p.strip()]

    chunks = []
    buffer = ""

    for para in raw_paras:
        # Paragraph đơn đã vượt max → tách theo câu
        if len(para) > max_chars:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            sentence_chunks = _split_by_sentence(para, max_chars)
            chunks.extend(sentence_chunks)
            continue

        # Thêm vào buffer
        if buffer:
            candidate = buffer + "\n\n" + para
        else:
            candidate = para

        if len(candidate) > max_chars:
            # Buffer đã đủ lớn → flush, bắt đầu buffer mới
            chunks.append(buffer.strip())
            buffer = para
        else:
            buffer = candidate

    if buffer.strip():
        chunks.append(buffer.strip())

    return [c for c in chunks if c]


def _split_by_sentence(text: str, max_chars: int) -> list[str]:
    """Tách text dài thành chunks theo câu."""
    # Tách tại dấu câu kết thúc
    sentences = re.split(r'(?<=[.!?:])\s+', text)
    chunks = []
    buffer = ""

    for sent in sentences:
        candidate = (buffer + " " + sent).strip() if buffer else sent
        if len(candidate) > max_chars and buffer:
            chunks.append(buffer.strip())
            buffer = sent
        else:
            buffer = candidate

    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks


# ── Main chunker ──────────────────────────────────────────────────────────────

def chunk_markdown(
    md_content: str,
    source_file: str,
    max_text_chars: int = 1500,
    min_text_chars: int = 100,
) -> list[dict]:
    """Trả về list[dict] thay vì list[Chunk]"""
    chunks: list[dict] = []
    chunk_index = 0
    current_section = ""
    source_name = Path(source_file).name

    table_matches = list(HTML_TABLE_PATTERN.finditer(md_content))

    segments = []
    prev_end = 0
    for m in table_matches:
        if m.start() > prev_end:
            segments.append(("text", prev_end, m.start(), md_content[prev_end:m.start()]))
        segments.append(("table", m.start(), m.end(), md_content[m.start():m.end()]))
        prev_end = m.end()
    if prev_end < len(md_content):
        segments.append(("text", prev_end, len(md_content), md_content[prev_end:]))

    for seg_type, seg_start, seg_end, seg_content in segments:
        if seg_type == "text":
            for line in seg_content.split('\n'):
                if _is_heading(line.strip()):
                    current_section = line.strip()

            para_chunks = _split_text_into_paragraphs(seg_content, max_text_chars)

            for para in para_chunks:
                if len(para) < min_text_chars:
                    if _is_heading(para):
                        current_section = para
                    continue

                first_line = para.split('\n')[0].strip()
                if _is_heading(first_line):
                    current_section = first_line

                chunks.append({
                    "chunk_id": _generate_id(para, chunk_index, source_name),
                    "content": para,
                    "chunk_type": "text",
                    "metadata": {
                        "source_file": source_name,
                        "section": current_section,
                        "table_name": None,
                        "chunk_index": chunk_index,
                        "total_chunks": 0,
                        "char_count": len(para),
                        "token_estimate": _estimate_tokens(para),
                    }
                })
                chunk_index += 1

        elif seg_type == "table":
            text_before = md_content[max(0, seg_start - 300):seg_start]
            table_name = _extract_table_name(text_before)
            full_content = f"{table_name}\n\n{seg_content}" if table_name else seg_content

            chunks.append({
                "chunk_id": _generate_id(full_content, chunk_index, source_name),
                "content": full_content,
                "chunk_type": "table",
                "metadata": {
                    "source_file": source_name,
                    "section": current_section,
                    "table_name": table_name,
                    "chunk_index": chunk_index,
                    "total_chunks": 0,
                    "char_count": len(full_content),
                    "token_estimate": _estimate_tokens(full_content),
                }
            })
            chunk_index += 1

    # Fill total_chunks
    total = len(chunks)
    for c in chunks:
        c["metadata"]["total_chunks"] = total

    return chunks


def chunk_directory(
    md_dir: str,
    output_json: str,
    max_text_chars: int = 1500,
    min_text_chars: int = 100,
):
    md_path = Path(md_dir)
    files = list(md_path.glob("*.md"))
    print(f"Tìm thấy {len(files)} file .md")

    all_chunks = []

    for md_file in files:
        content = md_file.read_text(encoding="utf-8")
        chunks = chunk_markdown(
            content,
            source_file=str(md_file),
            max_text_chars=max_text_chars,
            min_text_chars=min_text_chars,
        )
        all_chunks.extend(chunks)
        print(f"  ✓ {md_file.name}: {len(chunks)} chunks "
              f"({sum(1 for c in chunks if c['chunk_type']=='table')} tables, "
              f"{sum(1 for c in chunks if c['chunk_type']=='text')} text)")

    output = {
        "total_chunks": len(all_chunks),
        "total_files": len(files),
        "chunks": all_chunks,
    }

    Path(output_json).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\nTổng: {len(all_chunks)} chunks → {output_json}")
    return output


if __name__ == "__main__":
    # Test 1 file
    md_content = Path("Public_571.md").read_text(encoding="utf-8")
    chunks = chunk_markdown(md_content, source_file="Public_571.md")

    output = {
        "total_chunks": len(chunks),
        "total_files": 1,
        "chunks": chunks,
    }

    Path("Public_571_chunks.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))

    # Batch toàn bộ thư mục
    chunk_directory("./output_md", "chunks.json")