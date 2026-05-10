import re


class RouterAgent:

    def __init__(self, llm_service=None):
        self.llm = llm_service

    # MCQ SIGNALS 
    # Hard: có options A/B/C/D dạng option list
    _MCQ_HARD = [
        r"(?<!\w)[aAbBcCdD][.)]\s{0,3}\S",   # a. / b) / C. dạng option
        r"\b[ABCD]\s*[.)]\s*\w",             # A. hoặc B) viết hoa
    ]
    # Soft: hỏi về đáp án/phương án nhưng không có options rõ ràng
    _MCQ_SOFT = [
        r"đáp án (nào|đúng|sau|sau đây|dưới đây)",
        r"phương án (nào|đúng|sau|sau đây)",
        r"(câu|ý|phát biểu|nhận định|khẳng định)\s+nào\s+(sau đây|dưới đây|đúng|sai|chính xác)",
        r"(đúng|sai|chính xác|không đúng|không chính xác)\s*\??\s*$",
        r"(chọn|lựa chọn)\s+(đáp án|phương án|câu trả lời)",
        r"trong (các|những) (đáp án|phương án|lựa chọn)",
        r"(tất cả|bao nhiêu).*(đúng|sai|chính xác)",
    ]

    # API SIGNALS
    _API_HARD = [
        # Domain acronyms
        r"\bttpm\w*\b", r"\bcbnv\b", r"\bnslđ\b", r"\bosdc\b",
        r"\bslnt\b", r"\bslsx\b", r"\bcpnc\b", r"\blcnt\b",
        r"\bkpi\b", r"\bsla\b", r"\botd\b", r"\bfpy\b", r"\bdpmo\b",
        r"\bocs\b", r"\boee\b", r"\btakt\b",
        r"leakage rate", r"defect rate", r"yield rate",
        r"tr\.?\s*đồng", r"trđ\b", r"mm/người",
        r"\bpackage\b", r"presale[s]?", r"\bodc\b",
        # Nghiệp vụ
        r"năng suất lao động",
        r"hiệu suất (tổng thể|thiết bị|lao động|sản xuất)",
        r"sản lượng (sản xuất|thực tế|kế hoạch|đầu ra|hoàn thành)",
        r"doanh thu (thực tế|kế hoạch|thuần|gộp|ròng)",
        r"chi phí (nhân công|sản xuất|vận hành|trực tiếp|gián tiếp)",
        r"lợi nhuận (gộp|ròng|thuần|trước thuế|sau thuế)",
        r"tỷ lệ (lỗi|hỏng|đạt|hoàn thành|nghỉ|vắng|thất thoát)",
        r"số (ca làm việc|dự án|nhân viên|công nhân|lao động)\b",
        r"(gọi|call|lấy|tra cứu|lọc)\s+(api|dữ liệu|số liệu|báo cáo)",
        r"api\s+(nào|cần|để|phù hợp|cho)",
        r"cấu hình\s+api",
        r"endpoint\b",
    ]
    _API_TIME = [
        r"trong năm 20\d{2}",
        r"trong t(?:háng)?\s*\d{1,2}[/\-]20\d{2}",
        r"tháng \d{1,2}[\s/]*(?:năm\s*)?20\d{2}",
        r"trong quý [1-4][\s/]*20\d{2}",
        r"q[1-4][/\-]20\d{2}",
        r"\d{1,2}/20\d{2}\b",
        r"\bt\d{1,2}/20\d{2}\b",
        r"quý [1-4]\s+(?:năm\s*)?20\d{2}",
        r"(?:tháng|t\.)\s*\d{1,2}\s*(?:đến|~|\-)\s*(?:tháng|t\.)?\s*\d{1,2}",
    ]

    # LLM FALLBACK PROMPT 
    _LLM_PROMPT = (
        "Phân loại câu hỏi sau vào ĐÚNG MỘT nhãn:\n"
        '- "call_document": câu hỏi trắc nghiệm hoặc hỏi về quy định/khái niệm/lý thuyết\n'
        '- "call_api": câu hỏi cần lấy số liệu thực tế (KPI, doanh thu, sản lượng, cấu hình API)\n\n'
        "Chỉ trả về đúng 1 chuỗi: call_document hoặc call_api\n\n"
        "Câu hỏi: {question}\n\n"
        "Nhãn:"
    )

    # CLASSIFY

    def classify(self, question: str) -> str:
        text = str(question)
        text_lower = text.lower()

        # Tầng 1: MCQ hard — có ít nhất 2 options A/B/C/D → chắc chắn document
        mcq_hard = sum(1 for p in self._MCQ_HARD if re.search(p, text))
        if mcq_hard >= 2:
            return "call_document"

        # Tầng 2: API hard signal
        api_score = sum(1 for p in self._API_HARD if re.search(p, text_lower))
        time_score = sum(1 for p in self._API_TIME if re.search(p, text_lower))
        api_total = api_score + time_score * 2

        if api_total >= 2:
            return "call_api"

        # API score = 1 nhưng không có MCQ soft → call_api
        mcq_soft = sum(1 for p in self._MCQ_SOFT if re.search(p, text_lower))
        if api_total == 1 and mcq_soft == 0:
            return "call_api"

        # Tầng 3: MCQ soft signal rõ ràng
        if mcq_soft >= 2:
            return "call_document"

        # Tầng 4: LLM fallback khi tín hiệu mơ hồ
        if self.llm is not None:
            try:
                prompt = self._LLM_PROMPT.format(question=text[:600])
                raw = self.llm.generate(prompt, max_tokens=8).lower().strip()
                if "call_api" in raw:
                    return "call_api"
                if "call_document" in raw:
                    return "call_document"
            except Exception as e:
                print(f"⚠️ Router LLM lỗi: {e}")

        # Default: nếu có bất kỳ soft MCQ signal → document, còn lại → api
        return "call_document" if mcq_soft >= 1 else "call_api"