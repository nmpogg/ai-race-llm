import pdfplumber
from pathlib import Path
import re

HEADER_HEIGHT_RATIO = 0.12
FOOTER_HEIGHT_RATIO = 0.88

TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 5,
    "join_tolerance": 5,
    "edge_min_length": 3,
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
    "intersection_tolerance": 3,
}


def is_header_content(row: list) -> bool:
    row_text = " ".join((cell or "") for cell in row)
    return "VIETTEL AI RACE" in row_text or "Lần ban hành" in row_text


def table_to_html(data: list[list]) -> str:
    if not data or len(data) < 2:
        return ""
    if any(is_header_content(row) for row in data):
        return ""
    rows = ["<table>", "  <thead><tr>"]
    for cell in data[0]:
        rows.append(f"    <th>{(cell or '').strip().replace(chr(10), ' ')}</th>")
    rows.append("  </tr></thead>\n  <tbody>")
    for row in data[1:]:
        if all((cell or "").strip() == "" for cell in row):
            continue
        rows.append("  <tr>")
        for cell in row:
            rows.append(f"    <td>{(cell or '').strip().replace(chr(10), ' ')}</td>")
        rows.append("  </tr>")
    rows.append("  </tbody>\n</table>")
    return "\n".join(rows)


def words_to_lines(words: list[dict]) -> list[dict]:
    """
    Gom words thành lines, mỗi line là dict:
    {"y": float, "text": str, "type": "text"}
    """
    if not words:
        return []

    heights = [w["bottom"] - w["top"] for w in words]
    avg_h = sum(heights) / len(heights) if heights else 12
    same_line_tol = avg_h * 0.6

    lines = []
    cur_words = [words[0]]
    cur_y = words[0]["top"]

    for w in words[1:]:
        if abs(w["top"] - cur_y) <= same_line_tol:
            cur_words.append(w)
        else:
            lines.append({
                "y": cur_y,
                "text": " ".join(x["text"] for x in cur_words),
                "type": "text",
                "bottom": cur_words[-1]["bottom"],
            })
            cur_words = [w]
            cur_y = w["top"]

    lines.append({
        "y": cur_y,
        "text": " ".join(x["text"] for x in cur_words),
        "type": "text",
        "bottom": cur_words[-1]["bottom"],
    })
    return lines


def merge_lines_to_text(lines: list[dict]) -> str:
    """
    Gom lines thành paragraphs dựa vào khoảng cách giữa các dòng.
    """
    if not lines:
        return ""

    gaps = []
    for i in range(1, len(lines)):
        gaps.append(lines[i]["y"] - lines[i-1]["bottom"])
    
    # Ngưỡng paragraph: median gap * 1.5
    if gaps:
        sorted_gaps = sorted(g for g in gaps if g > 0)
        median_gap = sorted_gaps[len(sorted_gaps)//2] if sorted_gaps else 5
        para_threshold = median_gap * 1.8
    else:
        para_threshold = 10

    paragraphs = []
    cur_para = [lines[0]["text"]]

    for i in range(1, len(lines)):
        gap = lines[i]["y"] - lines[i-1]["bottom"]
        if gap > para_threshold:
            paragraphs.append("\n".join(cur_para))
            cur_para = [lines[i]["text"]]
        else:
            cur_para.append(lines[i]["text"])

    paragraphs.append("\n".join(cur_para))
    return "\n\n".join(p for p in paragraphs if p.strip())


def extract_page_elements(page) -> list[dict]:
    """
    Trả về list các elements (text lines + tables) đã được sắp xếp theo y.
    Mỗi element: {"y": float, "type": "text"|"table", "content": str}
    """
    header_h = page.height * HEADER_HEIGHT_RATIO
    footer_h = page.height * FOOTER_HEIGHT_RATIO

    # Detect bảng content
    all_tables = page.find_tables(table_settings=TABLE_SETTINGS)
    content_tables = []
    for t in all_tables:
        _, top, _, bottom = t.bbox
        # Bỏ bảng header (theo vị trí)
        if top < header_h:
            continue
        data = t.extract()
        if not data:
            continue
        # Bỏ bảng header (theo nội dung)
        if any(is_header_content(row) for row in data):
            continue
        content_tables.append(t)

    table_bboxes = [t.bbox for t in content_tables]

    # Extract words, loại header/footer/vùng bảng
    words = page.extract_words(
        x_tolerance=3,
        y_tolerance=3,
        keep_blank_chars=False,
        use_text_flow=True,
    )

    filtered_words = []
    for w in words:
        y = w["top"]
        if y < header_h or y > footer_h:
            continue
        in_table = any(
            (top - 3) <= y <= (bottom + 3)
            for (_, top, _, bottom) in table_bboxes
        )
        if not in_table:
            filtered_words.append(w)

    # Gom words → lines
    text_lines = words_to_lines(filtered_words)

    # Tạo unified element list
    elements = []

    # Thêm text lines
    for line in text_lines:
        # Bỏ dòng chỉ có số (page number sót lại)
        if re.fullmatch(r"\s*\d{1,3}\s*", line["text"]):
            continue
        elements.append({
            "y": line["y"],
            "bottom": line["bottom"],
            "type": "text",
            "content": line["text"],
        })

    # Thêm bảng
    for t, bbox in zip(content_tables, table_bboxes):
        _, top, _, bottom = bbox
        data = t.extract()
        html = table_to_html(data)
        if html:
            elements.append({
                "y": top,
                "bottom": bottom,
                "type": "table",
                "content": html,
            })

    # Sort theo y
    elements.sort(key=lambda x: x["y"])
    return elements

def merge_cross_page_elements(all_page_elements: list[list[dict]]) -> list[dict]:
    """
    Merge elements từ nhiều trang, xử lý text bị cắt giữa trang.
    all_page_elements: list of list, mỗi list là elements của 1 trang (đã sort theo y)
    """
    merged = []

    for page_idx, page_elems in enumerate(all_page_elements):
        if not page_elems:
            continue

        if not merged:
            merged.extend(page_elems)
            continue

        # Kiểm tra element cuối của merged và element đầu của trang mới
        last = merged[-1]
        first = page_elems[0]

        # Nếu cả hai đều là text và last_text không kết thúc bằng
        # dấu câu hoàn chỉnh → khả năng cao bị cắt giữa câu
        if (last["type"] == "text" and first["type"] == "text"
                and not _ends_sentence(last["content"])):
            # Merge 2 text lines thành 1
            merged[-1] = {
                **last,
                "content": last["content"].rstrip() + " " + first["content"].lstrip(),
                "bottom": first["bottom"],
            }
            merged.extend(page_elems[1:])
        else:
            merged.extend(page_elems)

    return merged


def _ends_sentence(text: str) -> bool:
    """Text kết thúc bằng dấu câu hoàn chỉnh."""
    stripped = text.rstrip()
    # Kết thúc câu: . ? ! : hoặc bullet point cuối (•)
    # KHÔNG kết thúc câu: , ; - ( ... hoặc chữ thường/số
    return bool(re.search(r'[.?!:»\]]$', stripped))

def elements_to_markdown(elements: list[dict]) -> str:
    """
    Chuyển list elements thành markdown, gom text lines thành paragraphs,
    bảng giữ nguyên HTML, xen kẽ đúng thứ tự.
    """
    if not elements:
        return ""

    # Tính ngưỡng paragraph từ các text elements
    text_elems = [e for e in elements if e["type"] == "text"]
    if len(text_elems) > 1:
        gaps = [
            text_elems[i]["y"] - text_elems[i-1]["bottom"]
            for i in range(1, len(text_elems))
        ]
        pos_gaps = sorted(g for g in gaps if g > 0)
        median_gap = pos_gaps[len(pos_gaps)//2] if pos_gaps else 5
        para_threshold = median_gap * 1.8
    else:
        para_threshold = 10

    output = []
    cur_text_lines = []

    def flush_text():
        if not cur_text_lines:
            return
        # Merge thành paragraphs
        paras = []
        cur_para = [cur_text_lines[0]["content"]]
        for i in range(1, len(cur_text_lines)):
            gap = cur_text_lines[i]["y"] - cur_text_lines[i-1]["bottom"]
            if gap > para_threshold:
                paras.append("\n".join(cur_para))
                cur_para = [cur_text_lines[i]["content"]]
            else:
                cur_para.append(cur_text_lines[i]["content"])
        paras.append("\n".join(cur_para))
        merged = "\n\n".join(p for p in paras if p.strip())
        if merged.strip():
            output.append(merged)
        cur_text_lines.clear()

    for elem in elements:
        if elem["type"] == "text":
            cur_text_lines.append(elem)
        else:  # table
            flush_text()
            output.append(elem["content"])

    flush_text()
    return "\n\n".join(output)


def pdf_to_markdown(pdf_path: str, debug: bool = False) -> str:
    all_page_elements = []  # list of list

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            elems = extract_page_elements(page)
            if debug:
                tables = [e for e in elems if e["type"] == "table"]
                texts = [e for e in elems if e["type"] == "text"]
                print(f"  Page {page_num+1}: {len(texts)} text lines, {len(tables)} tables")
            all_page_elements.append(elems)

    # Merge cross-page, xử lý text bị cắt
    merged_elements = merge_cross_page_elements(all_page_elements)

    result = elements_to_markdown(merged_elements)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def batch_convert(input_dir: str, output_dir: str, debug: bool = False):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = list(input_path.glob("*.pdf"))
    print(f"Tổng: {len(files)} file")

    for i, f in enumerate(files):
        try:
            result = pdf_to_markdown(str(f), debug=debug)
            out = output_path / (f.stem + ".md")
            out.write_text(result, encoding="utf-8")
            print(f"[{i+1}/{len(files)}] ✓ {f.name}")
        except Exception as e:
            print(f"[{i+1}/{len(files)}] ✗ {f.name}: {e}")


if __name__ == "__main__":
    # result = pdf_to_markdown("/content/Public_003.pdf", debug=True)
    # Path("/content/Public_003.md").write_text(result, encoding="utf-8")
    # print(result)

    PDF_DIR = "./data/Document_config_data"         # Thư mục chứa 600 file PDF gốc
    MD_DIR = "./data/markdown"           # Thư mục sẽ chứa 600 file MD rời
    
    batch_convert(PDF_DIR, MD_DIR, debug=False)