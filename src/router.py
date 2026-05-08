import re

class RouterAgent:
    def classify(self, question):
        text = str(question).lower()

        # 1. Dấu hiệu MCQ → call_document
        mcq_signals = [
            r'\ba[\.\)]\s+\w', r'\bb[\.\)]\s+\w',
            r'\bc[\.\)]\s+\w', r'\bd[\.\)]\s+\w',
            r'đáp án (nào|đúng|sau)',
            r'phương án (nào|đúng|sau)',
            r'(câu|ý|điều|phát biểu|nhận định|khẳng định)\s+nào\s+(sau đây|dưới đây|đúng|sai)',
            r'(đúng|sai|chính xác|không chính xác|không đúng)\??\s*$',
            r'(chọn|lựa chọn)\s+(đáp án|phương án)',
        ]
        if sum(1 for p in mcq_signals if re.search(p, text)) >= 2:
            return "call_document"

        # 2. Keyword nghiệp vụ → call_api
        api_keywords = [
            r'\bttpm\w*\b', r'\bcbnv\b', r'\bnslđ\b', r'\bosdc\b',
            r'\bslnt\b', r'\bslsx\b', r'\bcpnc\b', r'\blcnt\b',
            r'\bkpi\b', r'\bsla\b', r'\botd\b', r'\bfpy\b', r'\bdpmo\b',
            r'\bocs\b', r'\boee\b', r'\btakt\b',
            r'leakage rate', r'defect rate',
            r'tr đồng', r'trđ', r'mm/người', r'\bpackage\b',
            r'năng suất lao động', r'hiệu suất',
            r'sản lượng (sản xuất|thực tế|kế hoạch)',
            r'doanh thu (thực tế|kế hoạch|thuần)',
            r'chi phí (nhân công|sản xuất|vận hành)',
            r'lợi nhuận (gộp|ròng|thuần)',
            r'tỷ lệ (lỗi|hỏng|đạt|hoàn thành)',
        ]
        for pattern in api_keywords:
            if re.search(pattern, text):
                return "call_api"

        # 3. Time pattern → call_api
        time_patterns = [
            r'trong năm 202\d', r'trong t\d{1,2}/202\d',
            r'trong tháng \d{1,2}\s*/202\d', r'tháng \d{1,2} năm 202\d',
            r'trong quý \d[\s/]202\d', r'q\d[/-]202\d', r'\d{1,2}/202\d',
            r'(?:tháng|t)\s*\d{1,2}/\d{4}\s*(?:-|->|đến|~)\s*(?:tháng|t)\s*\d{1,2}/\d{4}',
        ]
        for pattern in time_patterns:
            if re.search(pattern, text):
                return "call_api"

        # 4. Tra cứu dữ liệu động → call_api
        data_query_patterns = [
            r'(cho biết|tính|tìm|xác định|lấy|tra cứu)\s+.{0,30}(số liệu|dữ liệu|giá trị|chỉ số|kết quả)',
            r'(báo cáo|thống kê|tổng hợp)\s+(số liệu|kết quả|dữ liệu)',
            r'api\s+(nào|cần|để)', r'(gọi|call)\s+api',
        ]
        for pattern in data_query_patterns:
            if re.search(pattern, text):
                return "call_api"

        return "call_document"