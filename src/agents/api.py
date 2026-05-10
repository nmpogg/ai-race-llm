"""
api.py — APIAgent

Cải tiến so với bản cũ:
  1. Retrieve top-5 APIs thay vì top-3
  2. LLM chọn API dựa trên mô tả + ví dụ câu hỏi mẫu
  3. Few-shot động từ example_data inject vào prompt chọn API
  4. Body building giữ nguyên rule-based (ổn định hơn LLM cho structured data)
"""

import json
import re
import calendar


class APIAgent:
    def __init__(self, llm_service, retriever, fewshot_loader=None):
        self.llm = llm_service
        self.retriever = retriever
        self.fewshot = fewshot_loader

    # DATE EXTRACTION 
    def _extract_dates(self, question: str) -> tuple[str, str]:
        q = question.lower()

        # Range: tháng X/YYYY đến tháng Y/YYYY
        m = re.search(
            r"(?:tháng|t)\s*(\d{1,2})[/\-](\d{4})\s*(?:-|->|đến|~)\s*"
            r"(?:tháng|t)?\s*(\d{1,2})[/\-](\d{4})",
            q,
        )
        if m:
            m1, y1, m2, y2 = (
                int(m.group(1)), int(m.group(2)),
                int(m.group(3)), int(m.group(4)),
            )
            return self._ms(y1, m1), self._me(y2, m2)

        # Single month: tháng X/YYYY hoặc tX/YYYY
        m = re.search(r"(?:tháng|t)\s*(\d{1,2})[/\-](\d{4})", q)
        if m:
            mo, yr = int(m.group(1)), int(m.group(2))
            return self._ms(yr, mo), self._me(yr, mo)

        # Quarter: quý X/YYYY hoặc qX-YYYY
        m = re.search(r"(?:quý|q)\s*([1-4])[/\-\s](\d{4})", q)
        if m:
            qtr, yr = int(m.group(1)), int(m.group(2))
            sm = (qtr - 1) * 3 + 1
            return self._ms(yr, sm), self._me(yr, sm + 2)

        # Full year: năm YYYY
        m = re.search(r"(?:năm|year)\s*(20\d{2})", q)
        if m:
            yr = int(m.group(1))
            return f"{yr}-01-01", f"{yr}-12-31"

        return "", ""

    def _ms(self, yr: int, mo: int) -> str:
        return f"{yr}-{mo:02d}-01"

    def _me(self, yr: int, mo: int) -> str:
        return f"{yr}-{mo:02d}-{calendar.monthrange(yr, mo)[1]}"

    # ENUM EXTRACTION 
    ORG_ALIASES = {
        "ttpmqt": "TTPMQT", "ttpmtcs": "TTPMTCS", "ttpmvt": "TTPMVT",
        "ttpmcnm": "TTPMCNM", "ttpmcds": "TTPMCDS", "ttcndt": "TTCNDT",
        "ttcnđt": "TTCNDT", "tt cnđt": "TTCNDT",
        "pm qt": "TTPMQT", "pm tcs": "TTPMTCS", "pm vt": "TTPMVT",
        "pm cnm": "TTPMCNM", "pm cds": "TTPMCDS",
    }
    PROJECT_TYPE_MAP = {
        "package": "Package", "gói": "Package",
        "osdc": "osdc", "odc/osdc": "odc/osdc", "odc": "odc",
        "t&m": "T&M", "time and material": "T&M",
        "presale": "presales", "presales": "presales",
    }
    PROJECT_STATUS_MAP = {
        "in-progress": "in-progress", "đang thực hiện": "in-progress",
        "hold": "hold", "tạm dừng": "hold",
        "closed": "closed", "đóng": "closed", "kết thúc": "closed",
        "open": "open", "mở": "open",
    }

    def _extract_orgs(self, question: str) -> list[str]:
        q = question.lower()
        if re.search(r"cả\s+công\s+ty|toàn\s+công\s+ty|tất cả", q):
            return []
        found, seen = [], set()
        for alias, canon in self.ORG_ALIASES.items():
            if alias in q and canon not in seen:
                found.append(canon)
                seen.add(canon)
        return found

    def _extract_enum(self, question: str, mapping: dict) -> list[str]:
        q = question.lower()
        found, seen = [], set()
        for kw, val in mapping.items():
            if kw in q and val not in seen:
                found.append(val)
                seen.add(val)
        return found

    # API SELECTION VIA LLM 
    _SELECT_PROMPT = (
        "{fewshot}"
        "Chọn API phù hợp nhất với câu hỏi bên dưới.\n\n"
        "[DANH SÁCH API]:\n{api_list}\n\n"
        "[CÂU HỎI]: {question}\n\n"
        "Chỉ trả về func_code của API phù hợp nhất, không giải thích:\n"
        "func_code:"
    )

    def _select_api(self, question: str, top_df) -> str | None:
        api_list_str = "\n---\n".join(
            f"func_code: {row['func_code']}\n"
            f"Mô tả: {row.get('description', '')}\n"
            f"Ví dụ câu hỏi: {row.get('Example question', '')}"
            for _, row in top_df.iterrows()
        )

        fewshot_block = ""
        if self.fewshot is not None:
            fewshot_block = self.fewshot.get_api_fewshot(question)

        prompt = self._SELECT_PROMPT.format(
            fewshot=fewshot_block,
            api_list=api_list_str,
            question=question,
        )
        raw = self.llm.generate(prompt, max_tokens=30).strip()

        # Khớp chính xác
        for fc in top_df["func_code"].tolist():
            if fc.lower() in raw.lower():
                return fc
        # Khớp gần đúng (bỏ khoảng trắng)
        raw_clean = re.sub(r"\s+", "", raw).lower()
        for fc in top_df["func_code"].tolist():
            if re.sub(r"\s+", "", fc).lower() in raw_clean:
                return fc
        return None

    # BODY BUILDER
    def _build_body(self, question: str, endpoint_config_str: str) -> dict:
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
            name = p.get("name", "")
            ptype = p.get("type", "")
            alias = p.get("alias", name)

            if name in ("fromDate", "from_date", "startDate"):
                body[name] = from_date
            elif name in ("toDate", "to_date", "endDate"):
                body[name] = to_date
            elif name in ("organization", "orgAlias", "org"):
                body[name] = orgs
            elif name == "projectType":
                body[name] = proj_types
            elif name == "projectStatus":
                body[name] = proj_status
            elif name == "isCompany":
                body[name] = not bool(orgs)
            elif name == "page":
                body[name] = 0
            elif name == "size":
                body[name] = 20
            elif "List" in ptype or "list" in ptype.lower():
                body[name] = []

        return body

    # MAIN PROCESS
    def process(self, question: str) -> str:
        # Retrieve top-5 APIs
        top_df = self.retriever.get_top_apis_df(question, k=5)
        if top_df.empty:
            return "{}"

        # LLM chọn API tốt nhất
        selected_fc = self._select_api(question, top_df)

        if selected_fc is None:
            selected_row = top_df.iloc[0]
        else:
            rows = top_df[top_df["func_code"] == selected_fc]
            selected_row = rows.iloc[0] if not rows.empty else top_df.iloc[0]

        # Lấy path từ endpoint config
        try:
            cfg = json.loads(selected_row["Endpoint config"])
            path = cfg["request"]["path"]
        except Exception:
            return "{}"

        body = self._build_body(question, selected_row["Endpoint config"])
        return json.dumps({"path": path, "body": body}, ensure_ascii=False)