import json
import re
import calendar


class APIAgent:

    def __init__(self, llm_service, retriever, fewshot_loader=None):
        self.llm       = llm_service
        self.retriever = retriever
        self.fewshot   = fewshot_loader

    # DATE EXTRACTION

    def _extract_dates(self, question: str) -> tuple[str, str]:
        q = question.lower()
        m = re.search(
            r"(?:tháng|t)\s*(\d{1,2})[/\-](\d{4})\s*(?:-|->|đến|~)\s*"
            r"(?:tháng|t)?\s*(\d{1,2})[/\-](\d{4})", q,
        )
        if m:
            m1, y1, m2, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            return self._ms(y1, m1), self._me(y2, m2)

        m = re.search(r"tháng\s*(\d{1,2})[/\-](20\d{2})", q)
        if m:
            mo, yr = int(m.group(1)), int(m.group(2))
            return self._ms(yr, mo), self._me(yr, mo)

        m = re.search(r"\bt(\d{1,2})[/\-](20\d{2})\b", q)
        if m:
            mo, yr = int(m.group(1)), int(m.group(2))
            return self._ms(yr, mo), self._me(yr, mo)

        m = re.search(r"(?:quý|q)\s*([1-4])[/\-\s](20\d{2})", q)
        if m:
            qtr, yr = int(m.group(1)), int(m.group(2))
            sm = (qtr - 1) * 3 + 1
            return self._ms(yr, sm), self._me(yr, sm + 2)

        m = re.search(r"(?:năm|year)\s*(20\d{2})", q)
        if m:
            yr = int(m.group(1))
            return f"{yr}-01-01", f"{yr}-12-31"

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

    # EXTRACT HELPERS

    def _extract_type(self, question: str) -> int:
        q = question.lower()
        if re.search(r"trong năm|cả năm|theo năm|\bnăm 20\d{2}\b", q):
            return 5
        if re.search(r"quý|quarter", q):
            return 4
        return 3

    def _extract_sort(self, question: str) -> int | None:
        q = question.lower()
        if re.search(r"tăng dần|ascending|asc", q):
            return 1
        if re.search(r"giảm dần|descending|desc", q):
            return 2
        return None

    def _extract_standard_comparison(self, question: str) -> int | None:
        q = question.lower()
        if re.search(r"(cao hơn|trên|vượt|đạt|bằng hoặc cao hơn)\s*(chuẩn|tiêu chuẩn|kpi|sla)", q):
            return 1
        if re.search(r"(thấp hơn|dưới|không đạt)\s*(chuẩn|tiêu chuẩn|kpi|sla)", q):
            return 2
        return None

    def _extract_summary_date(self, question: str) -> str:
        _, to_date = self._extract_dates(question)
        return to_date if to_date else ""

    def _extract_org_code(self, question: str) -> str:
        q = question.lower()
        for alias, canon in self.ORG_CODE_MAP.items():
            if alias in q:
                return canon
        return "VTIT"

    # ENUM MAPS

    # f1 base + f2 aliases mở rộng
    ORG_ALIASES = {
        "ttpmqt":   "TTPMQT",  "ttpmtcs":  "TTPMTCS", "ttpmvt":   "TTPMVT",
        "ttpmcnm":  "TTPMCNM", "ttpmcds":  "TTPMCDS", "ttcndt":   "TTCNDT",
        "ttcnđt":   "TTCNDT",  "tt cnđt":  "TTCNDT",
        "pm qt":    "TTPMQT",  "pm tcs":   "TTPMTCS", "pm vt":    "TTPMVT",
        "pm cnm":   "TTPMCNM", "pm cds":   "TTPMCDS",
        # Thêm từ f2
        "ttpmcđs":  "TTPMCDS", "tt pmcds": "TTPMCDS",
        "tt pmvt":  "TTPMVT",  "tt pmqt":  "TTPMQT",
        "tt pmtcs": "TTPMTCS", "tt pmcnm": "TTPMCNM",
        "trung tâm pm qt":  "TTPMQT",
        "trung tâm pm tcs": "TTPMTCS",
        "trung tâm pm vt":  "TTPMVT",
        "trung tâm pm cnm": "TTPMCNM",
        "trung tâm pm cds": "TTPMCDS",
        "trung tâm cnđt":   "TTCNDT",
    }

    ORG_CODE_MAP = {
        "ttpmqt":   "TTPMQT",  "ttpmtcs":  "TTPMTCS", "ttpmvt":   "TTPMVT",
        "ttpmcnm":  "TTPMCNM", "ttpmcds":  "TTPMCDS", "ttcndt":   "TTCNDT",
        "ttcnđt":   "TTCNDT",  "ttpmcđs":  "TTPMCDS", "vtit":     "VTIT",
    }

    PROJECT_TYPE_MAP = {
        "package":           "Package",  "gói":               "Package",
        "osdc":              "osdc",     "odc/osdc":          "odc/osdc",
        "odc":               "odc",      "t&m":               "T&M",
        "time and material": "T&M",      "presale":           "presales",
        "presales":          "presales",
    }

    PROJECT_STATUS_MAP = {
        "in-progress": "in-progress", "đang thực hiện": "in-progress",
        "hold":        "hold",        "tạm dừng":        "hold",
        "closed":      "closed",      "đóng":            "closed",
        "kết thúc":    "closed",      "open":            "open",
        "mở":          "open",
    }

    ASSET_GROUP_MAP = {
        "dịch vụ cntt":     "Dịch vụ CNTT",
        "công cụ dụng cụ":  "Công cụ dụng cụ",
        "máy móc thiết bị": "Máy móc thiết bị",
        "phần mềm":         "Phần mềm",
        "tài sản cố định":  "Tài sản cố định",
    }

    LCNT_OPTION_MAP = {
        "đtrr": "ĐTRR", "mstt": "MSTT", "chlt": "CHLT",
        "chlc": "CHLC", "đthc": "ĐTHC",
    }

    LCNT_TYPE_MAP = {
        "đtrr": "ĐTRR", "mstt": "MSTT", "chlt": "CHLT",
        "chlc": "CHLC", "đthc": "ĐTHC",
    }

    BID_PLAN_TYPE_MAP = {
        "đấu thầu không qua mạng": "Đấu thầu không qua mạng",
        "đấu thầu qua mạng":       "Đấu thầu qua mạng",
        "chỉ định thầu":           "Chỉ định thầu",
        "mua sắm trực tiếp":       "Mua sắm trực tiếp",
        "tự thực hiện":            "Tự thực hiện",
    }

    # TARGET CODE MAP: dành cho GET APIs KD/TC
    TARGET_CODE_MAP = {
        "doanh thu":            "DT",
        "doanh thu dịch vụ":    "DT",
        "doanh thu thuần":      "DT",
        "slnt":                 "SLNT",
        "sản lượng nghiệm thu": "SLNT",
        "giá trị hợp đồng":     "GTHĐKM",
        "hợp đồng ký mới":      "GTHĐKM",
        "lợi nhuận gộp":        "LNG",
        "lợi nhuận":            "LNG",
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

    def _extract_target_code(self, question: str, default: str = "DT") -> str:
        q = question.lower()
        for kw, code in self.TARGET_CODE_MAP.items():
            if kw in q:
                return code
        return default

    # LLM: CHỌN API + EXTRACT PARAMS

    _SELECT_AND_BODY_PROMPT = (
        "{fewshot}"
        "Dựa vào câu hỏi, hãy:\n"
        "1. Chọn API phù hợp nhất từ danh sách\n"
        "2. Điền các tham số body dựa trên thông tin trong câu hỏi\n"
        "   Lưu ý: Nếu câu hỏi không đề cập đến tham số nào thì để []/null theo mô tả\n\n"
        "[DANH SÁCH API]:\n{api_list}\n\n"
        "[CÂU HỎI]: {question}\n\n"
        "Chỉ trả về JSON theo đúng format sau, không giải thích:\n"
        "{{\n"
        "  \"func_code\": \"<func_code của API phù hợp nhất>\",\n"
        "  \"body_params\": {{\n"
        "    \"<tên param>\": <giá trị>\n"
        "  }}\n"
        "}}\n"
        "JSON:"
    )

    def _select_and_extract_params(self, question: str, top_df) -> tuple[str | None, dict]:
        # Giữ description[:200] và example[:150] từ f1 — context đầy đủ hơn
        api_list_str = "\n---\n".join(
            f"func_code: {row['func_code']}\n"
            f"Mô tả: {row.get('description', '')[:200]}\n"
            f"Ví dụ: {str(row.get('Example question', ''))[:150]}"
            for _, row in top_df.iterrows()
        )

        fewshot_block = ""
        if self.fewshot is not None:
            fewshot_block = self.fewshot.get_api_fewshot(question)

        prompt = self._SELECT_AND_BODY_PROMPT.format(
            fewshot=fewshot_block,
            api_list=api_list_str,
            question=question,
        )
        # Giữ max_tokens=250 từ f1 — đủ chỗ cho JSON phức tạp
        raw = self.llm.generate(prompt, max_tokens=250).strip()

        selected_fc = None
        llm_params  = {}
        try:
            # f2 fix: dùng raw_decode thay vì regex để parse JSON chính xác hơn
            decoder = json.JSONDecoder()
            start   = raw.find('{')
            if start != -1:
                obj, _  = decoder.raw_decode(raw, start)
                fc_raw  = obj.get("func_code", "")
                for fc in top_df["func_code"].tolist():
                    if fc.lower() in fc_raw.lower():
                        selected_fc = fc
                        break
                if selected_fc is None:
                    fc_clean = re.sub(r"\s+", "", fc_raw).lower()
                    for fc in top_df["func_code"].tolist():
                        if re.sub(r"\s+", "", fc).lower() in fc_clean:
                            selected_fc = fc
                            break
                # f2 fix: or {} để tránh crash khi LLM trả None
                llm_params = obj.get("body_params", {}) or {}
        except Exception as e:
            print(f"⚠️ LLM select+extract lỗi: {e}")
            for fc in top_df["func_code"].tolist():
                if fc.lower() in raw.lower():
                    selected_fc = fc
                    break

        return selected_fc, llm_params

    # BODY BUILDER

    def _build_body(
        self,
        question: str,
        endpoint_config_str: str,
        llm_params: dict = None,
    ) -> dict:
        try:
            cfg = json.loads(endpoint_config_str)
        except Exception:
            return {}

        all_params     = cfg.get("required_params", []) + cfg.get("optional_params", [])
        from_date, to_date = self._extract_dates(question)
        orgs           = self._extract_orgs(question)
        proj_types     = self._extract_enum(question, self.PROJECT_TYPE_MAP)
        proj_status    = self._extract_enum(question, self.PROJECT_STATUS_MAP)
        asset_groups   = self._extract_enum(question, self.ASSET_GROUP_MAP)
        lcnt_options   = self._extract_enum(question, self.LCNT_OPTION_MAP)
        lcnt_types     = self._extract_enum(question, self.LCNT_TYPE_MAP)
        bid_plan_types = self._extract_enum(question, self.BID_PLAN_TYPE_MAP)
        type_val       = self._extract_type(question)
        sort_val       = self._extract_sort(question)
        std_comp       = self._extract_standard_comparison(question)
        org_code       = self._extract_org_code(question)
        summary_date   = self._extract_summary_date(question)

        body = {}
        for p in all_params:
            name  = p.get("name", "")
            ptype = p.get("type", "")

            if name in ("fromDate", "from_date", "startDate"):
                body[name] = from_date
            elif name in ("toDate", "to_date", "endDate"):
                body[name] = to_date
            elif name in ("organization", "org"):
                body[name] = orgs
            elif name == "orgAlias":
                body[name] = orgs
            elif name == "organizationCode":
                body[name] = org_code
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
            elif name == "lcntType":
                body[name] = lcnt_types
            elif name == "bidPlanType":
                body[name] = bid_plan_types
            elif name == "isCompany":
                body[name] = not bool(orgs)
            elif name == "isAllProject":
                body[name] = True
            elif name == "IsAllCustomer":
                body[name] = True
            elif name == "type":
                body[name] = type_val
            elif name == "sort":
                body[name] = sort_val if sort_val is not None else 2
            elif name == "standardComparison":
                body[name] = std_comp
            elif name == "summaryDate":
                body[name] = summary_date
            elif name == "page":
                body[name] = 0
            elif name == "size":
                body[name] = 20
            elif name == "isProbation":
                # isProbation: thực tập sinh = 1, chính thức = 0, mặc định None (lấy hết)
                q = question.lower()
                if re.search(r"thực tập|tts\b|probation", q):
                    body[name] = 1
                elif re.search(r"chính thức|biên chế", q):
                    body[name] = 0
                else:
                    body[name] = None
            elif name == "getIsProbation":
                q = question.lower()
                body[name] = bool(re.search(r"thực tập|tts\b|probation", q))
            elif name == "targetCode":
                # Lấy default từ path template nếu có
                path = cfg.get("request", {}).get("path", "")
                tc_match = re.search(r"targetCode=([A-Z]+)", path)
                default_tc = tc_match.group(1) if tc_match else "DT"
                body[name] = self._extract_target_code(question, default=default_tc)
            elif name == "cycleType":
                # cycleType: month/quarter/year
                q = question.lower()
                if re.search(r"quý|quarter", q):
                    body[name] = "quarter"
                elif re.search(r"năm|year", q):
                    body[name] = "year"
                else:
                    body[name] = "month"
            elif "List" in ptype or "list" in ptype.lower():
                body[name] = []

        # Merge LLM params
        if llm_params:
            for k, v in llm_params.items():
                if k in body and body[k] in ([], "", None):
                    body[k] = v
                elif k not in body:
                    body[k] = v

        return body

    # BUILD PATH: Handle GET APIs với query params trong URL

    def _build_path_with_params(self, cfg: dict, body: dict, question: str) -> str:
        """
        GET APIs dùng query params trong URL path.
        Inject summary_date, org_code, target_code, cycleType vào path.
        """
        method = cfg.get("request", {}).get("method", "POST")
        path   = cfg.get("request", {}).get("path", "")

        if method != "GET":
            return path

        summary_date = body.get("summaryDate", "")
        if summary_date and re.search(r"summaryDate=[\d\-]+", path):
            path = re.sub(r"summaryDate=[\d\-]+", f"summaryDate={summary_date}", path)

        org_code = body.get("organizationCode", "VTIT")
        if org_code and re.search(r"organizationCode=\w+", path):
            path = re.sub(r"organizationCode=\w+", f"organizationCode={org_code}", path)

        target_code = body.get("targetCode", "DT")
        if target_code and re.search(r"targetCode=\w+", path):
            path = re.sub(r"targetCode=\w+", f"targetCode={target_code}", path)

        cycle_type = body.get("cycleType", "month")
        if cycle_type and re.search(r"cycleType=\w+", path):
            path = re.sub(r"cycleType=\w+", f"cycleType={cycle_type}", path)

        return path

    # MAIN PROCESS

    def process(self, question: str) -> str:
        top_df = self.retriever.get_top_apis_df(question, k=5)
        if top_df.empty:
            return "{}"

        selected_fc, llm_params = self._select_and_extract_params(question, top_df)
        if selected_fc is None:
            selected_row = top_df.iloc[0]
        else:
            rows = top_df[top_df["func_code"] == selected_fc]
            selected_row = rows.iloc[0] if not rows.empty else top_df.iloc[0]

        try:
            cfg = json.loads(selected_row["Endpoint config"])
        except Exception:
            return "{}"

        body = self._build_body(question, selected_row["Endpoint config"], llm_params=llm_params)

        # GET APIs: inject params vào URL path thay vì body
        path   = self._build_path_with_params(cfg, body, question)
        method = cfg.get("request", {}).get("method", "POST")
        if method == "GET":
            return json.dumps({"path": path, "body": {}}, ensure_ascii=False)

        return json.dumps({"path": path, "body": body}, ensure_ascii=False)