import re

class RouterAgent:
    STRICT_API_KEYWORDS = [
        r'\bttpm\w*\b',
        r'\btt\w{2,}\b',
        r'\bcbnv\b',
        r'\bnslđ\b',
        r'\bosdc\b',
        r'\bslnt\b',
        r'\bslsx\b',
        r'\bcpnc\b',
        r'\blcnt\b',
        r'\bpmo[-\s]?\d+\b',
        r'\bra\b(?=.*mm)',
        r'leakage\s*rate',
        r'defect\s*rate',
        r'error\s*rate',
        r'bug\s*rate',
        r'tr\s*đồng',
        r'trđ\b',
        r'mm/người',
        r'\bmm\b(?=.*(thực hiện|kế hoạch|ra\b))',
        r'\bpackage\b',
        r'\bdashboard\b',
        r'\bkpi\b',
        r'báo\s*cáo\b(?=.*(q\d|quý|tháng|năm|202\d))',
        r'chỉ\s*số',
        r'xếp\s*hạng',
        r'nhân\s*sự(?=.*(trung\s*tâm|202\d|\bttpm|\btt\b))',

        # Nhóm 1: Chỉ số tài chính
        r'\bebitda\b',
        r'\bròng\b',
        r'chỉ\s*số\s*tài\s*chính',
        r'\bkh\b(?=.*(th\b|thực hiện))',  # KH / TH (kế hoạch / thực hiện)

        # Nhóm 2: Tài sản
        r'tài\s*sản(?=.*(hiện có|mua mới|tồn kho|cấp phát|nhóm))',
        r'tồn\s*kho\s*sx',
        r'cấp\s*phát\s*đặc\s*thù',
        r'nhóm\s*tài\s*sản',

        # Nhóm 3: Dự án / ticket tracking
        r'\b[A-Z]{2}\d{2}\.[A-Z]+\.[A-Za-z0-9]+\b',  # BU01.BCN.SAP, BU05.VCS.ThreaIntel
        r'\bepic\b',
        r'\buser\s*stor\w*\b',
        r'\b(us|cr|task)\b(?=.*(trạng thái|status|open|done|resolved|reopened|in.progress))',
        r'trạng\s*thái(?=.*(open|done|resolved|reopened|in.progress|closed))',
        r'số\s*lượng\s*(epic|task|us\b|cr\b)',
    ]

    TIME_PATTERNS = [
        r'trong\s+năm\s+202\d',
        r'trong\s+t\d{1,2}/202\d',
        r'trong\s+tháng\s+\d{1,2}\s*/\s*202\d',
        r'trong\s+quý\s+\d/202\d',
        r'q\d/202\d',
        r'(?:tháng|t)\s*\d{1,2}/\d{4}\s*(?:-|->|đến)\s*(?:tháng|t)\s*\d{1,2}/\d{4}',
        r't\d{1,2}/202\d\s*[-–]\s*t\d{1,2}/202\d',
        r'\bnăm\s+202\d\b',

        # Nhóm 1: Range quý
        r'quý\s*\d/\d{4}\s*[-–]\s*quý\s*\d/\d{4}',       # Quý 4/2024 - Quý 2/2025
        r'quý\s*\d/\d{4}\s*(đến|->)\s*quý\s*\d/\d{4}',

        # Nhóm 3: "ở thời điểm tháng X/YYYY"
        r'thời\s*điểm\s*tháng\s*\d{1,2}/202\d',
        r'vào\s*tháng\s*\d{1,2}/202\d',
    ]

    API_INTENT_PATTERNS = [
        r'(xem|tra\s*cứu|lấy|get|fetch)\s+.{0,30}(số\s*liệu|dữ\s*liệu|thông\s*tin|chỉ\s*số)',
        r'(tôi\s+muốn\s+xem|cho\s+tôi\s+xem)',
        r'(thực\s*hiện|kế\s*hoạch)\s+(?:là|bao\s*nhiêu)',
        r'lọc\s*ra.{0,30}(status|trạng\s*thái)',           # "Lọc ra những status..."
    ]

    def classify(self, question: str) -> str:
        text = str(question).lower().strip()

        for pattern in self.STRICT_API_KEYWORDS:
            if re.search(pattern, text, re.IGNORECASE):
                return "call_api"

        for pattern in self.TIME_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return "call_api"

        for pattern in self.API_INTENT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                number_cue = re.search(
                    r'\d{4}|quý|tháng|năm|bao\s*nhiêu|tổng|tỉ\s*lệ|rate|ratio',
                    text, re.IGNORECASE
                )
                if number_cue:
                    return "call_api"

        return "call_document"

    def evaluate(self, example_file: str) -> dict:
        import pandas as pd

        df_q  = pd.read_excel(example_file, sheet_name="example_question")
        df_rs = pd.read_excel(example_file, sheet_name="example_result")
        df    = pd.merge(df_q, df_rs, on="id", how="inner")

        total, correct = len(df), 0
        api_total, api_correct = 0, 0
        doc_total, doc_correct = 0, 0
        errors = []

        for _, row in df.iterrows():
            question   = str(row["fun_question"])
            truth_code = str(row["func_code"]).strip()
            pred_code  = self.classify(question)
            is_ok      = (pred_code == truth_code)

            if is_ok:
                correct += 1
            if truth_code == "call_api":
                api_total += 1
                if is_ok: api_correct += 1
            else:
                doc_total += 1
                if is_ok: doc_correct += 1
            if not is_ok:
                errors.append({
                    "id":       row["id"],
                    "question": question,
                    "truth":    truth_code,
                    "pred":     pred_code,
                })

        result = {
            "total":        total,
            "correct":      correct,
            "accuracy":     round(correct / total * 100, 2) if total else 0,
            "call_api_acc": round(api_correct / api_total * 100, 2) if api_total else 0,
            "call_doc_acc": round(doc_correct / doc_total * 100, 2) if doc_total else 0,
            "errors":       errors,
        }

        print("\n====== ROUTER EVALUATION REPORT ======")
        print(f"Tổng câu           : {total}")
        print(f"Đúng               : {correct} ({result['accuracy']}%)")
        print(f"call_api accuracy  : {api_correct}/{api_total} ({result['call_api_acc']}%)")
        print(f"call_document acc  : {doc_correct}/{doc_total} ({result['call_doc_acc']}%)")
        print(f"Số câu sai         : {len(errors)}")
        if errors:
            print("\nChi tiết câu sai:")
            for e in errors:
                print(f"  ID {e['id']}: truth={e['truth']}, pred={e['pred']}")
                print(f"    Q: {e['question'][:80]}")
        print("======================================\n")

        return result