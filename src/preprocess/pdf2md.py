import os
import re
import fitz
import pymupdf4llm
import glob

def clean_markdown(md_text, base_name):
    clean_md = re.sub(r'!\[.*?\]\(.*?\)', '', md_text)
    clean_md = re.sub(r'\n{3,}', '\n\n', clean_md)

    def is_fake_heading(content):
        return content.strip().endswith(('.', ':', ',')) or len(content.split()) > 15

    def format_hierarchical_headings(match):
        num_part = match.group(1).strip('.')
        content = match.group(2).strip()
        
        if is_fake_heading(content):
            return match.group(0)
            
        level = len(num_part.split('.')) + 1 
        hash_prefix = '#' * min(level, 6)
        return f'{hash_prefix} {num_part}. {content}'

    hierarchical_regex = r'^(\d+(?:\.\d+)*)\.?\s+(.*)$'
    clean_md = re.sub(hierarchical_regex, format_hierarchical_headings, clean_md, flags=re.MULTILINE)

    lines = clean_md.split('\n')
    for i, line in enumerate(lines):
        match = re.match(r'^(#{1,6})\s+(.*)', line)
        if match and is_fake_heading(match.group(2)):
            lines[i] = match.group(2)
            
    final_text = f"# Tài liệu: {base_name}\n\n{clean_md.strip()}\n\n"
    return final_text

def pdf2md(input_pdf_folder, output_md_folder, crop_margin=0.1):
    os.makedirs(output_md_folder, exist_ok=True)
    pdf_files = glob.glob(os.path.join(input_pdf_folder, "*.pdf"))
    total_files = len(pdf_files)
    
    if total_files == 0:
        print(f"Không tìm thấy PDF trong: {input_pdf_folder}")
        return

    print(f"Bắt đầu chuyển đổi {total_files} pdf sang md...")
    success = 0
    
    for idx, pdf_path in enumerate(pdf_files):
        file_name = os.path.basename(pdf_path)
        base_name = os.path.splitext(file_name)[0]
        output_md_path = os.path.join(output_md_folder, f"{base_name}.md")
        
        # Bỏ qua nếu file MD đã tồn tại
        if os.path.exists(output_md_path):
            success += 1
            continue

        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                rect = page.rect
                page.set_cropbox(fitz.Rect(
                    rect.x0, rect.y0 + rect.height * crop_margin,
                    rect.x1, rect.y1 - rect.height * crop_margin
                ))

            md_text = pymupdf4llm.to_markdown(doc, write_images=False, page_chunks=False)
            final_md = clean_markdown(md_text, base_name)
            
            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write(final_md)
                
            success += 1
            print(f"Đã chuyển đổi: {idx + 1}/{total_files} files...")
                
        except Exception as e:
            print(f"Lỗi ở file {file_name}: {e}")

    print(f"Hoàn tất chuyển đổi: {success}/{total_files} files nằm tại thư mục [{output_md_folder}]")


def merge(input_md_folder, master_output_file):
    # Sắp xếp file theo tên (Alphabet) để file tổng có thứ tự rõ ràng
    md_files = sorted(glob.glob(os.path.join(input_md_folder, "*.md")))
    total_files = len(md_files)
    
    if total_files == 0:
        print(f"Không tìm thấy file MD nào trong: {input_md_folder}")
        return
        
    print(f"Bắt đầu gộp {total_files} file md thành [{master_output_file}]...")
    
    with open(master_output_file, "w", encoding="utf-8") as outfile:
        for idx, md_path in enumerate(md_files):
            with open(md_path, "r", encoding="utf-8") as infile:
                content = infile.read()
                outfile.write(content)
                outfile.write("\n\n")
                
    print(f"Success! Đã gộp toàn bộ vào file: {os.path.abspath(master_output_file)}")

def process_pdf_data(pdf_folder, md_folder, master_md_file):
    pdf2md(pdf_folder, md_folder)
    merge(md_folder, master_md_file)

if __name__ == "__main__":

    PDF_DIR = "./data/Document_config_data"         # Thư mục chứa 600 file PDF gốc
    MD_DIR = "./data/markdown/mds"           # Thư mục sẽ chứa 600 file MD rời
    MASTER_FILE = "./data/markdown/corpus.md" # File tổng cuối cùng
    
    os.makedirs(PDF_DIR, exist_ok=True)

    process_pdf_data(PDF_DIR, MD_DIR, MASTER_FILE)