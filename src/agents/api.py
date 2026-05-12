import json
import re
import calendar


class APIAgent:
    def __init__(self, llm_service, retriever, fewshot_loader=None):
        self.llm = llm_service
        self.retriever = retriever
        self.fewshot = fewshot_loader

    # ── DATE EXTRACTION ───────────────────────────────────────────────────────
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

        # Single month đầy đủ: tháng X/YYYY
        m = re.search(r"tháng\s*(\d{1,2})[/\-](20\d{2})", q)
        if m:
            mo, yr = int(m.group(1)), int(m.group(2))
            return self._ms(yr, mo), self._me(yr, mo)

        # Viết tắt: T8/2025 hoặc t8/2025 (có khoảng trắng hoặc không)
        m = re.search(r"\bt(\d{1,2})[/\-](20\d{2})\b", q)
        if m:
            mo, yr = int(m.group(1)), int(m.group(2))
            return self._ms(yr, mo), self._me(yr, mo)

        # Dạng: trong T9/2025
        m = re.search(r"trong\s+t(\d{1,2})[/\-](20\d{2})", q)
        if m:
            mo, yr = int(m.group(1)), int(m.group(2))
            return self._ms(yr, mo), self._me(yr, mo)

        # Quarter: quý X/YYYY hoặc qX-YYYY
        m = re.search(r"(?:quý|q)\s*([1-4])[/\-\s](20\d{2})", q)
        if m:
            qtr, yr = int(m.group(1)), int(m.group(2))
            sm = (qtr - 1) * 3 + 1
            return self._ms(yr, sm), self._me(yr, sm + 2)

        # Full year: năm YYYY
        m = re.search(r"(?:năm|year)\s*(20\d{2})", q)
        if m:
            yr = int(m.group(1))
            return f"{yr}-01-01", f"{yr}-12-31"

        # Fallback: chỉ có YYYY (ví dụ "trong 2025")
        m = re.search(r"\b(20\d{2})\b", q)
        if m:
            yr = int(m.group(1))
            return f"{yr}-01-01", f"{yr}-12-31"

        return "", ""

    def _ms(self, yr: int, mo: int) -> str:
        mo = max(1, min(12, mo))
        return f"{yr}-{mo:02d}-01"

    def _me(self, yr: int, mo: int) -> str:
        mo = max(1, min(12, mo))
        return f"{yr}-{mo:02d}-{calendar.monthrange(yr, mo)[1]}"

    # ── ENUM EXTRACTION ───────────────────────────────────────────────────────
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
    ASSET_GROUP_MAP = {
        "dịch vụ cntt": "Dịch vụ CNTT",
        "công cụ dụng cụ": "Công cụ dụng cụ",
        "máy móc thiết bị": "Máy móc thiết bị",
        "phần mềm": "Phần mềm",
        "tài sản cố định": "Tài sản cố định",
    }
    LCNT_OPTION_MAP = {
        "đtrr": "ĐTRR", "mstt": "MSTT", "chlt": "CHLT",
        "chlc": "CHLC", "đthc": "ĐTHC",
    }
    BID_PLAN_TYPE_MAP = {
        "đấu thầu không qua mạng": "Đấu thầu không qua mạng",
        "đấu thầu qua mạng": "Đấu thầu qua mạng",
        "chỉ định thầu": "Chỉ định thầu",
        "mua sắm trực tiếp": "Mua sắm trực tiếp",
        "tự thực hiện": "Tự thực hiện",
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

    # ── API SELECTION ─────────────────────────────────────────────────────────
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
        # Khớp gần đúng
        raw_clean = re.sub(r"\s+", "", raw).lower()
        for fc in top_df["func_code"].tolist():
            if re.sub(r"\s+", "", fc).lower() in raw_clean:
                return fc
        return None

    # ── BODY BUILDER ──────────────────────────────────────────────────────────
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
        asset_groups = self._extract_enum(question, self.ASSET_GROUP_MAP)
        lcnt_options = self._extract_enum(question, self.LCNT_OPTION_MAP)
        bid_plan_types = self._extract_enum(question, self.BID_PLAN_TYPE_MAP)

        body = {}
        for p in all_params:
            name = p.get("name", "")
            ptype = p.get("type", "")

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
            elif name == "assetGroup":
                body[name] = asset_groups
            elif name == "lcntOption":
                body[name] = lcnt_options
            elif name == "lcntOptionDoing":
                body[name] = lcnt_options
            elif name == "bidPlanType":
                body[name] = bid_plan_types
            elif name == "isCompany":
                body[name] = not bool(orgs)
            elif name == "page":
                body[name] = 0
            elif name == "size":
                body[name] = 20
            elif "List" in ptype or "list" in ptype.lower():
                body[name] = []

        return body

    # ── MAIN PROCESS ──────────────────────────────────────────────────────────
    def process(self, question: str) -> str:
        top_df = self.retriever.get_top_apis_df(question, k=5)
        if top_df.empty:
            return "{}"

        selected_fc = self._select_api(question, top_df)

        if selected_fc is None:
            selected_row = top_df.iloc[0]
        else:
            rows = top_df[top_df["func_code"] == selected_fc]
            selected_row = rows.iloc[0] if not rows.empty else top_df.iloc[0]

        try:
            cfg = json.loads(selected_row["Endpoint config"])
            path = cfg["request"]["path"]
        except Exception:
            return "{}"

        body = self._build_body(question, selected_row["Endpoint config"])
        return json.dumps({"path": path, "body": body}, ensure_ascii=False)