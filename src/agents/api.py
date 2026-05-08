import json, re
from datetime import date
import calendar

class APIAgent:
    def __init__(self, llm_service, retriever):
        self.llm = llm_service
        self.retriever = retriever

    # ─── DATE EXTRACTION ──────────────────────────────────────────────────────

    def _extract_dates(self, question):
        """Trích fromDate, toDate từ câu hỏi"""
        q = question.lower()

        # Khoảng tháng: tháng 6/2025 - tháng 11/2025
        m = re.search(
            r'(?:tháng|t)\s*(\d{1,2})[/\-](\d{4})\s*(?:-|->|đến|~)\s*(?:tháng|t)?\s*(\d{1,2})[/\-](\d{4})',
            q
        )
        if m:
            m1, y1, m2, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            return self._month_start(y1, m1), self._month_end(y2, m2)

        # Tháng đơn: tháng 8/2025 hoặc t8/2025
        m = re.search(r'(?:tháng|t)\s*(\d{1,2})[/\-](\d{4})', q)
        if m:
            mo, yr = int(m.group(1)), int(m.group(2))
            return self._month_start(yr, mo), self._month_end(yr, mo)

        # Quý: quý 3/2025 hoặc q3/2025
        m = re.search(r'(?:quý|q)\s*(\d)[/\-\s](\d{4})', q)
        if m:
            qtr, yr = int(m.group(1)), int(m.group(2))
            start_m = (qtr - 1) * 3 + 1
            end_m = qtr * 3
            return self._month_start(yr, start_m), self._month_end(yr, end_m)

        # Năm: năm 2025
        m = re.search(r'(?:năm|year)\s*(20\d{2})', q)
        if m:
            yr = int(m.group(1))
            return f"{yr}-01-01", f"{yr}-12-31"

        return "", ""

    def _month_start(self, yr, mo):
        return f"{yr}-{mo:02d}-01"

    def _month_end(self, yr, mo):
        last_day = calendar.monthrange(yr, mo)[1]
        return f"{yr}-{mo:02d}-{last_day}"

    # ─── ORGANIZATION EXTRACTION ──────────────────────────────────────────────

    ORG_ALIASES = {
        "ttpmqt": "TTPMQT", "ttpmtcs": "TTPMTCS", "ttpmvt": "TTPMVT",
        "ttpmcnm": "TTPMCNM", "ttpmcds": "TTPMCDS", "ttcndt": "TTCNDT",
        "ttcnđt": "TTCNDT", "tt cnđt": "TTCNDT",
        "pm qt": "TTPMQT", "pm tcs": "TTPMTCS", "pm vt": "TTPMVT",
        "pm cnm": "TTPMCNM", "pm cds": "TTPMCDS",
    }

    def _extract_orgs(self, question):
        q = question.lower()
        found = []
        for alias, canonical in self.ORG_ALIASES.items():
            if alias in q and canonical not in found:
                found.append(canonical)
        # "cả công ty" → không trả org list
        if re.search(r'cả\s+công\s+ty|toàn\s+công\s+ty', q):
            return []
        return found

    # ─── ENUM EXTRACTION ─────────────────────────────────────────────────────

    PROJECT_TYPE_MAP = {
        "package": "Package", "gói": "Package",
        "osdc": "osdc", "odc": "odc", "odc/osdc": "odc/osdc",
        "t&m": "T&M", "tm": "T&M", "time": "T&M",
        "presale": "presales", "presales": "presales",
    }
    PROJECT_STATUS_MAP = {
        "in-progress": "in-progress", "đang thực hiện": "in-progress", "đang triển khai": "in-progress",
        "hold": "hold", "tạm dừng": "hold",
        "closed": "closed", "đóng": "closed", "kết thúc": "closed",
        "open": "open", "mở": "open",
    }

    def _extract_enum(self, question, mapping):
        q = question.lower()
        found = []
        for kw, val in mapping.items():
            if kw in q and val not in found:
                found.append(val)
        return found

    # ─── BODY BUILDER ─────────────────────────────────────────────────────────

    def _build_body(self, question, endpoint_config_str):
        """Điền body dựa trên required/optional params + rule extraction"""
        try:
            cfg = json.loads(endpoint_config_str)
        except Exception:
            return {}

        all_params = cfg.get("required_params", []) + cfg.get("optional_params", [])
        from_date, to_date = self._extract_dates(question)
        orgs = self._extract_orgs(question)
        proj_types = self._extract_enum(question, self.PROJECT_TYPE_MAP)
        proj_status = self._extract_enum(question, self.PROJECT_STATUS_MAP)

        body = {}
        for p in all_params:
            name = p["name"]
            ptype = p.get("type", "")
            desc = p.get("description", "").lower()

            if name == "fromDate":
                body[name] = from_date
            elif name == "toDate":
                body[name] = to_date
            elif name in ("organization", "orgAlias"):
                body[name] = orgs
            elif name == "projectType":
                body[name] = proj_types
            elif name == "projectStatus":
                body[name] = proj_status
            elif name == "isCompany":
                body[name] = not bool(orgs)
            elif name in ("page",):
                body[name] = 0
            elif name in ("size",):
                body[name] = 20
            elif "List" in ptype:
                body[name] = []
            # Bỏ qua các param optional không xác định được

        return body

    # ─── MAIN PROCESS ─────────────────────────────────────────────────────────

    def _select_api(self, question, api_configs_raw, configs_df):
        """LLM chỉ chọn API phù hợp, trả về func_code"""
        prompt = f"""<|im_start|>system
Chọn API phù hợp nhất với câu hỏi. Chỉ trả về func_code (một chuỗi, không giải thích).

[DANH SÁCH API]:
{api_configs_raw}
<|im_end|>
<|im_start|>user
Câu hỏi: {question}
<|im_end|>
<|im_start|>assistant
"""
        raw = self.llm.generate(prompt).strip()
        # Tìm func_code khớp trong danh sách
        for fc in configs_df["func_code"].tolist():
            if fc.lower() in raw.lower():
                return fc
        return None

    def process(self, question):
        # Lấy top 3 API configs
        top_df = self.retriever.get_top_apis_df(question, k=3)
        if top_df.empty:
            return "{}"

        api_list = "\n---\n".join(
            f"func_code: {row['func_code']}\nMô tả: {row['description']}\nVí dụ: {row['Example question']}"
            for _, row in top_df.iterrows()
        )

        # LLM chọn func_code
        selected_fc = self._select_api(question, api_list, top_df)

        # Nếu LLM không chọn được → lấy top 1
        if selected_fc is None:
            selected_row = top_df.iloc[0]
        else:
            rows = top_df[top_df["func_code"] == selected_fc]
            selected_row = rows.iloc[0] if not rows.empty else top_df.iloc[0]

        # Lấy path và build body bằng rules
        try:
            cfg = json.loads(selected_row["Endpoint config"])
            path = cfg["request"]["path"]
        except Exception:
            return "{}"

        body = self._build_body(question, selected_row["Endpoint config"])

        return json.dumps({"path": path, "body": body}, ensure_ascii=False)