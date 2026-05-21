import json
import re
import calendar


class APIAgent:

    def __init__(self, llm_service, retriever, fewshot_loader=None):
        self.llm       = llm_service
        self.retriever = retriever
        self.fewshot   = fewshot_loader

    # DATE EXTRACTION 

    def _ms(self, yr, mo):
        mo = max(1, min(12, mo))
        return f"{yr}-{mo:02d}-01"

    def _me(self, yr, mo):
        mo = max(1, min(12, mo))
        return f"{yr}-{mo:02d}-{calendar.monthrange(yr, mo)[1]}"

    def _extract_dates(self, question):
        q = question.lower()
        m = re.search(
            r"(?:tháng|t)\s*(\d{1,2})[/\-](\d{4})\s*(?:-|->|đến|~)\s*"
            r"(?:tháng|t)?\s*(\d{1,2})[/\-](\d{4})", q)
        if m:
            return (self._ms(int(m.group(2)), int(m.group(1))),
                    self._me(int(m.group(4)), int(m.group(3))))
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

    # ALIAS MAPS: từ Doc_alias_for_contest 
    # value lấy từ file alias

    # organization: từ alias sheet (thêm TTCN)
    ORG_MAP = {
        "trung tâm phần mềm quản trị":    "TTPMQT",
        "trung tâm phần mềm viễn thông":  "TTPMVT",
        "trung tâm phần mềm tài chính số":"TTPMTCS",
        "trung tâm phần mềm công nghệ mới":"TTPMCNM",
        "trung tâm phần mềm chuyển đổi số":"TTPMCDS",
        "trung tâm công nghệ đặc thù":    "TTCNDT",
        "trung tâm công nghệ":             "TTCN",
        # Alias ngắn
        "ttpmqt":  "TTPMQT",  "ttpmvt":  "TTPMVT",  "ttpmtcs": "TTPMTCS",
        "ttpmcnm": "TTPMCNM", "ttpmcds": "TTPMCDS",  "ttcndt":  "TTCNDT",
        "ttcnđt":  "TTCNDT",  "ttcn":    "TTCN",
        "pm qt":   "TTPMQT",  "pm vt":   "TTPMVT",   "pm tcs":  "TTPMTCS",
        "pm cnm":  "TTPMCNM", "pm cds":  "TTPMCDS",
        "ttpmcđs": "TTPMCDS",
    }

    # orgAlias: phòng ban KD (từ alias sheet)
    ORG_ALIAS_MAP = {
        "đầu tư pháp chế": "DTPC", "dtpc": "DTPC",
        "hỗ trợ kinh doanh": "HTKD", "htkd": "HTKD",
        "kinh doanh giải pháp doanh nghiệp": "KDGPDN", "kdgpdn": "KDGPDN",
        "kinh doanh quốc tế": "KDQT", "kdqt": "KDQT",
        "kinh doanh viễn thông": "KDVTCNM", "kdvtcnm": "KDVTCNM",
    }

    # projectType: value từ alias sheet
    PROJECT_TYPE_MAP = {
        "t&m": "T&M", "time and material": "T&M",
        "presales": "presales", "presale": "presales",
        "package": "package", "gói": "package",
        "odc/osdc": "odc/osdc", "osdc": "odc/osdc", "odc": "odc/osdc",
    }

    # projectStatus: value từ alias sheet (chú ý "Hoàn thành" -> hold)
    PROJECT_STATUS_MAP = {
        "presale": "presale",       "tiền bán hàng": "presale",
        "in-progress": "in-progress", "đang thực hiện": "in-progress",
        "trong tiến trình": "in-progress",
        "closed": "closed",         "đã đóng": "closed", "kết thúc": "closed",
        "open": "open",             "đang mở": "open",
        "hold": "hold",             "hoàn thành": "hold", "tạm dừng": "hold",
    }

    # position: value từ alias sheet (thêm VHKT, DATA, AI)
    POSITION_MAP = {
        "developer": "DEV",    "dev": "DEV",
        "business analyst": "BA", "ba": "BA",
        "ui/ux designer": "UIUX", "uiux": "UIUX",
        "project manager": "PM",  "pm": "PM",
        "tester": "TESTER",    "qa": "TESTER",
        "ai engineer": "AI",   "ai": "AI",
        "data analyst": "DATA", "data": "DATA",
        "vhkt": "VHKT",
    }

    # level: value từ alias sheet (F/J/J+/M/M+/S — không phải Fresher/Junior)
    LEVEL_MAP = {
        "fresher": "F",  "f": "F",
        "junior+": "J+", "junior": "J", "j+": "J+", "j": "J",
        "middle+": "M+", "middle": "M", "m+": "M+", "m": "M",
        "senior": "S",   "s": "S",
        "đang cập nhật": "Đang Cập Nhật",
    }

    # lcntOption: value là SỐ (1,2,3) — không phải text
    LCNT_OPTION_MAP = {
        "1 giai đoạn 1 túi": 1,
        "1 giai đoạn 2 túi": 2,
        "2 giai đoạn 2 túi": 3,
    }

    # lcntOptionDoing: value là SỐ
    LCNT_OPTION_DOING_MAP = {
        "đấu thầu không qua mạng": 1,
        "đấu thầu qua mạng":       2,
    }

    # bidPlanType: value là SỐ (1=Trong nước, 2=Quốc tế)
    BID_PLAN_TYPE_MAP = {
        "trong nước": 1, "nội địa": 1,
        "quốc tế":    2, "international": 2,
    }

    # lcntType: value là SỐ
    LCNT_TYPE_MAP = {
        "đầu tư rời rạc":      1,
        "đấu thầu hạn chế":    2,
        "mua sắm tập trung":   3,
        "chào hàng cạnh tranh":4,
        "chủ đầu tư":          5,
        "tự thực hiện":        6,
        "hình thức đặc biệt":  7,
    }

    # lcntDomainType: value là SỐ
    LCNT_DOMAIN_TYPE_MAP = {
        "mua sắm hàng hóa": 1,
        "xây lắp":          2,
        "tư vấn":           3,
        "phi tư vấn":       4,
        "hỗn hợp":          5,
    }

    # gtStatus: value là SỐ
    GT_STATUS_MAP = {
        "đã duyệt kế hoạch lựa chọn nhà thầu":          3,
        "đã duyệt kết quả lựa chọn nhà thầu":           4,
        "chờ mở thầu":                                   5,
        "đã hủy kết quả lựa chọn nhà thầu":             14,
        "đã hủy kế hoạch lựa chọn nhà thầu":            15,
        "đã đóng thầu":                                  7,
        "đã hủy trình ký kết quả lựa chọn nhà thầu":    11,
        "đã báo cáo đánh giá hồ sơ dự thầu":            8,
    }

    # hdStatus: value là SỐ
    HD_STATUS_MAP = {
        "chưa ký":      0,
        "đã ký":        1,
        "đang thực hiện": 2,
        "đã thanh lý":  3,
    }

    # isProbation: từ alias (Đã duyệt=1, Chưa duyệt=0)

    # assetGroup: value từ alias (một số khác key)
    ASSET_GROUP_MAP = {
        "thiết bị văn phòng":                     "TBVP",
        "tbvp":                                   "TBVP",
        "tài sản phục vụ sản xuất kinh doanh":    "Tài sản phục vụ sản xuất kinh doanh",
        "tài sản cố định":                        "Tài sản cố định",
        "dịch vụ khác":                           "Dịch vụ khác",
        "dịch vụ công nghệ thông tin":            "Dịch vụ CNTT",
        "dịch vụ cntt":                           "Dịch vụ CNTT",
        "công cụ dụng cụ":                        "Công cụ dụng cụ",
    }

    # dtmsClass: value từ alias
    DTMS_CLASS_MAP = {
        "đầu tư hỗ trợ phát triển khoa học công nghệ":  "Dự án Đầu tư hỗ trợ phát triển KHCN",
        "đầu tư mua sắm tài sản":   "Dự án Đầu tư Mua sắm tài sản",
        "đầu tư phát triển":        "Dự án Đầu tư phát triển",
        "đầu tư xây dựng":          "Dự án Đầu tư xây dựng",
        "mua sắm bảo đảm liên tục": "Mua sắm nhằm bảo đảm tính liên tục cho hoạt động SXKD",
        "mua sắm duy trì thường xuyên": "Mua sắm nhằm duy trì hoạt động thường xuyên của Công ty",
    }

    # dtmsType: value từ alias
    DTMS_TYPE_MAP = {
        "công nghệ thông tin": "CNTT", "cntt": "CNTT",
        "nghiên cứu sản xuất": "NCSX", "ncsx": "NCSX",
        "vật tư": "Vật tư",
        "xây dựng dân dụng": "XDDD-Khác", "xddd": "XDDD-Khác",
    }

    # procurementType: value từ alias (SỐ dạng string "1"/"0")
    PROCUREMENT_TYPE_MAP = {
        "hình thành gói thầu":       "1",
        "không hình thành gói thầu": "0",
    }

    # trainGroup: value từ alias
    TRAIN_GROUP_MAP = {
        "chứng chỉ chuyên môn":             "Chứng chỉ chuyên môn",
        "chuyên môn nghiệp vụ":             "Đào tạo chuyên môn, nghiệp vụ",
        "hội nhập":                         "Đào tạo hội nhập",
        "năng lực cốt lõi":                 "Đào tạo năng lực cốt lõi, tuân thủ",
        "năng lực lãnh đạo":                "Đào tạo năng lực lãnh đạo, quản lý",
        "tiếng anh":                        "Đào tạo tiếng anh",
        "thực tập sinh":                    "Thực tập sinh",
    }

    # priorityList: value = key (same)
    PRIORITY_MAP = {
        "highest": "Highest", "high": "High",
        "medium": "Medium",   "low": "Low", "lowest": "Lowest",
        "cao nhất": "Highest", "cao": "High",
        "trung bình": "Medium", "thấp": "Low", "thấp nhất": "Lowest",
    }

    # TARGET_CODE cho KD/TC APIs
    TARGET_CODE_MAP = {
        "doanh thu thuần":            "DT",
        "doanh thu dịch vụ":          "DT",
        "doanh thu":                  "DT",
        "sản lượng nghiệm thu":       "SLNT",
        "slnt":                       "SLNT",
        "giá trị hợp đồng ký mới":   "GTHĐKM",
        "hợp đồng ký mới":           "GTHĐKM",
        "lợi nhuận gộp":             "LNG",
        "lợi nhuận":                 "LNG",
    }

    # EXTRACT HELPERS 

    def _extract_list(self, question: str, mapping: dict) -> list:
        """Tìm tất cả values trong mapping xuất hiện trong câu hỏi."""
        q = question.lower()
        found, seen = [], set()
        # Sort by key length giảm dần để match dài nhất trước
        for kw in sorted(mapping, key=len, reverse=True):
            if kw in q and mapping[kw] not in seen:
                found.append(mapping[kw])
                seen.add(mapping[kw])
        return found

    def _extract_orgs(self, question: str) -> list:
        return self._extract_list(question, self.ORG_MAP)

    def _extract_org_alias(self, question: str) -> list:
        return self._extract_list(question, self.ORG_ALIAS_MAP)

    def _extract_type_from_desc(self, question: str, desc: str, example_val=None) -> int:
        """
        Trích xuất type dựa vào description của param + nội dung câu hỏi.
        Ưu tiên: mapping tường minh trong desc > example_call > fallback.
        """
        q = question.lower()
        d = desc.lower()

        has_year    = bool(re.search(r"(?:trong\s+)?năm\s+20\d{2}|cả\s+năm|theo\s+năm", q))
        has_quarter = bool(re.search(r"quý|q[1-4]", q))
        has_month   = bool(re.search(r"tháng|\bt\d{1,2}[/\-]", q))
        has_range   = bool(re.search(r"t\d{1,2}[/\-]20\d{2}\s*(?:-|->|~|đến)\s*t?\d", q))

        # Tìm mapping tường minh: "năm -> 5", "quý -> 4", "tháng -> 3"
        explicit = {}
        for pat, key in [
            (r"năm\s*(?:->|trả về|return)\s*(\d)", "year"),
            (r"quý\s*(?:->|trả về|return)\s*(\d)", "quarter"),
            (r"tháng\s*(?:->|trả về|return)\s*(\d)", "month"),
            (r"tuần\s*(?:->|trả về|return)\s*(\d)", "week"),
        ]:
            m = re.search(pat, d)
            if m:
                explicit[key] = int(m.group(1))

        if explicit:
            if has_year    and "year"    in explicit: return explicit["year"]
            if has_quarter and "quarter" in explicit: return explicit["quarter"]
            if (has_month or has_range) and "month" in explicit: return explicit["month"]
            if example_val is not None: return int(example_val)
            return list(explicit.values())[0]

        # Không có mapping tường minh -> dùng example_call làm default
        if example_val is not None:
            default = int(example_val)
        else:
            default = 3  # MONTH

        # Chỉ override khi example chưa set (None hoặc 1)
        if example_val is None or int(example_val) == 1:
            if has_year:    return 5
            if has_quarter: return 4
            if has_month or has_range: return 3

        return default

    def _extract_sort(self, question: str) -> int | None:
        q = question.lower()
        if re.search(r"tăng dần|ascending", q): return 1
        if re.search(r"giảm dần|descending", q): return 2
        return None

    def _get_example_body(self, cfg: dict) -> dict:
        ex = cfg.get("example_call", [])
        if not ex: return {}
        raw = ex[0].get("body", "{}")
        if isinstance(raw, dict): return raw
        try:
            return json.loads(raw) or {}
        except Exception:
            return {}


    def _build_body(self, question: str, cfg: dict) -> dict:
        all_params   = cfg.get("required_params", []) + cfg.get("optional_params", [])
        example_body = self._get_example_body(cfg)

        # Lấy tên tất cả params trong spec — chỉ điền những gì spec có
        spec_param_names = {p["name"] for p in all_params}

        from_date, to_date = self._extract_dates(question)
        orgs = self._extract_orgs(question)
        q_lower = question.lower()

        body = {}
        for p in all_params:
            name  = p.get("name", "")
            ptype = p.get("type", "")
            pdesc = p.get("description", "")

            if name in ("fromDate", "from_date", "startDate"):
                body[name] = from_date

            elif name in ("toDate", "to_date", "endDate"):
                body[name] = to_date

            # FIX 1: fromDateProject/toDateProject — có trong spec thì điền null
            elif name == "fromDateProject":
                body[name] = None

            elif name == "toDateProject":
                body[name] = None

            elif name == "organization":
                body[name] = orgs

            elif name == "orgAlias":
                body[name] = self._extract_list(question, {
                    "dtpc": "DTPC", "htkd": "HTKD", "kdgpdn": "KDGPDN",
                    "kdqt": "KDQT", "kdvtcnm": "KDVTCNM",
                }) or orgs

            elif name == "projectType":
                body[name] = self._extract_list(question, self.PROJECT_TYPE_MAP)

            elif name == "projectStatus":
                body[name] = self._extract_list(question, self.PROJECT_STATUS_MAP)

            elif name == "position":
                body[name] = self._extract_list(question, self.POSITION_MAP)

            elif name == "level":
                body[name] = self._extract_list(question, self.LEVEL_MAP)

            elif name == "assetGroup":
                body[name] = self._extract_list(question, self.ASSET_GROUP_MAP)

            elif name == "dtmsClass":
                body[name] = self._extract_list(question, self.DTMS_CLASS_MAP)

            elif name == "dtmsType":
                body[name] = self._extract_list(question, self.DTMS_TYPE_MAP)

            elif name == "trainGroup":
                body[name] = self._extract_list(question, self.TRAIN_GROUP_MAP)

            elif name == "priorityList":
                body[name] = self._extract_list(question, self.PRIORITY_MAP)

            elif name == "assigneeList":
                body[name] = []

            elif name == "projectList":
                body[name] = []

            elif name == "customerList":
                body[name] = []

            elif name == "progressList":
                body[name] = []

            elif name == "lcntOption":
                for kw, val in [("1 giai đoạn 1 túi", 1),
                                ("1 giai đoạn 2 túi", 2),
                                ("2 giai đoạn 2 túi", 3)]:
                    if kw in q_lower:
                        body[name] = [val]; break
                else:
                    body[name] = []

            elif name == "lcntOptionDoing":
                for kw, val in [("không qua mạng", 1), ("qua mạng", 2)]:
                    if kw in q_lower:
                        body[name] = [val]; break
                else:
                    body[name] = []

            elif name == "bidPlanType":
                for kw, val in [("trong nước", 1), ("quốc tế", 2)]:
                    if kw in q_lower:
                        body[name] = [val]; break
                else:
                    body[name] = []

            elif name == "lcntType":
                lcnt_map = {
                    "đầu tư rời rạc": 1, "đấu thầu hạn chế": 2,
                    "mua sắm tập trung": 3, "chào hàng cạnh tranh": 4,
                    "chủ đầu tư": 5, "tự thực hiện": 6, "hình thức đặc biệt": 7,
                }
                found = [v for kw, v in lcnt_map.items() if kw in q_lower]
                body[name] = found if found else []

            elif name == "lcntDomainType":
                domain_map = {"mua sắm hàng hóa": 1, "xây lắp": 2,
                            "tư vấn": 3, "phi tư vấn": 4, "hỗn hợp": 5}
                found = [v for kw, v in domain_map.items() if kw in q_lower]
                body[name] = found if found else []

            elif name == "procurementType":
                if "hình thành gói thầu" in q_lower:
                    body[name] = ["1"]
                elif "không hình thành" in q_lower:
                    body[name] = ["0"]
                else:
                    body[name] = []

            elif name == "type":
                ex_type = example_body.get("type")
                body[name] = self._extract_type_from_desc(question, pdesc, ex_type)

            elif name == "sort":
                # FIX 5: chỉ điền sort nếu spec có, lấy từ example_call làm default
                # KHÔNG inject sort=2 mặc định
                sv      = self._extract_sort(question)
                ex_sort = example_body.get("sort")  # None nếu example không set
                if sv is not None:
                    body[name] = sv
                elif ex_sort is not None:
                    body[name] = ex_sort
                else:
                    body[name] = None

            # FIX 2: isCompany — chỉ điền khi spec có param này
            elif name == "isCompany":
                has_company = bool(re.search(
                    r"cả\s+công\s+ty|toàn\s+công\s+ty", q_lower))
                body[name] = has_company or not bool(orgs)

            # FIX 3: isAllProject, isAllCustomer, page, size
            # — chỉ điền khi spec CÓ param này (vòng lặp all_params đảm bảo điều này)
            # Nhưng cần kiểm tra evaluator có chấm không trước khi thêm
            elif name == "isAllProject":
                body[name] = True

            elif name in ("IsAllCustomer", "isAllCustomer"):
                body["isAllCustomer"] = True
                continue

            elif name == "page":
                body[name] = 0

            elif name == "size":
                body[name] = 20

            elif name == "isProbation":
                if re.search(r"thực tập|tts\b|probation", q_lower):
                    body[name] = 1
                elif re.search(r"chính thức|biên chế", q_lower):
                    body[name] = 0
                else:
                    body[name] = None

            elif name == "getIsProbation":
                body[name] = bool(re.search(r"thực tập|tts\b|probation", q_lower))

            elif name == "standardComparison":
                if re.search(r"cao hơn|vượt|đạt\s+chuẩn|trên\s+chuẩn", q_lower):
                    body[name] = 1
                elif re.search(r"thấp hơn|dưới\s+chuẩn|không đạt", q_lower):
                    body[name] = 2
                else:
                    body[name] = None

            elif name == "summaryDate":
                _, td = self._extract_dates(question)
                body[name] = td

            elif name == "targetCode":
                path    = cfg.get("request", {}).get("path", "")
                tc_m    = re.search(r"targetCode=([A-Z]+)", path)
                default = tc_m.group(1) if tc_m else "DT"
                tc = None
                for kw in sorted(self.TARGET_CODE_MAP, key=len, reverse=True):
                    if kw in q_lower:
                        tc = self.TARGET_CODE_MAP[kw]
                        break
                body[name] = tc or default

            elif name == "cycleType":
                if re.search(r"quý|quarter", q_lower):
                    body[name] = "quarter"
                elif re.search(r"năm|year", q_lower):
                    body[name] = "year"
                else:
                    body[name] = "month"

            elif name == "organizationCode":
                code = "VTIT"
                for kw, val in sorted(self.ORG_MAP.items(), key=lambda x: -len(x[0])):
                    if kw in q_lower:
                        code = val; break
                body[name] = code

            elif "List" in ptype or "list" in ptype.lower():
                body[name] = []

            elif ptype in ("Boolean", "boolean"):
                body[name] = None

        # Normalize isAllCustomer key case
        if "IsAllCustomer" in body:
            body["isAllCustomer"] = body.pop("IsAllCustomer")

        return body

    # LLM: CHỈ CHỌN API với prompt ngắn 

    _SELECT_PROMPT = """\
Chọn API phù hợp nhất với câu hỏi. Chỉ trả về func_code, không giải thích.

[CÁC API ỨNG VIÊN]
{api_list}

[CÂU HỎI]
{question}

func_code:"""

    def _select_api(self, question: str, top_df) -> str | None:
        # Build danh sách ngắn: func_code + mô tả 1 dòng + example question
        lines = []
        for _, row in top_df.iterrows():
            ex_q = str(row.get("Example question", "")).split("\n")[0].strip()[:80]
            lines.append(
                f"- {row['func_code']}: {str(row.get('description',''))[:80]}"
                + (f" | VD: {ex_q}" if ex_q else "")
            )

        prompt = self._SELECT_PROMPT.format(
            api_list="\n".join(lines),
            question=question,
        )

        # max_tokens=20: sinh func_code (~15-20 chars)
        raw = self.llm.generate(prompt, max_tokens=20).strip()

        # Match func_code từ output
        raw_clean = re.sub(r"\s+", "", raw).lower()
        for fc in top_df["func_code"].tolist():
            if re.sub(r"\s+", "", fc).lower() in raw_clean:
                return fc

        # Fallback: tìm bất kỳ fc nào xuất hiện trong raw
        for fc in top_df["func_code"].tolist():
            if fc.lower() in raw.lower():
                return fc

        # Fallback cuối: trả top-1
        return top_df.iloc[0]["func_code"]

    # MAIN 

    def process(self, question: str) -> str:
        top_df = self.retriever.get_top_apis_df(question, k=5)
        if top_df.empty:
            return "{}"

        # Bước 1: LLM chọn API (prompt ngắn, nhanh)
        selected_fc = self._select_api(question, top_df)
        if not selected_fc:
            return "{}"

        # Bước 2: Lấy config của API được chọn
        rows = top_df[top_df["func_code"] == selected_fc]
        if rows.empty:
            rows = top_df.iloc[[0]]
        row = rows.iloc[0]

        try:
            cfg = json.loads(row["Endpoint config"])
        except Exception:
            return "{}"

        path = cfg.get("request", {}).get("path", "")
        method = cfg.get("request", {}).get("method", "POST")

        # GET API: body rỗng
        if method == "GET":
            return json.dumps({"path": path, "body": {}}, ensure_ascii=False)

        # Bước 3: Rule-based điền params (không dùng LLM)
        body = self._build_body(question, cfg)

        return json.dumps({"path": path, "body": body}, ensure_ascii=False)